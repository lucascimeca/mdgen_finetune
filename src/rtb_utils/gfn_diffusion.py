from collections import deque
from datetime import datetime
from functools import partial
import random

from accelerate import Accelerator
from matplotlib import pyplot as plt

from rtb_utils.diffusers.pipelines.ddpm_gfn.pipeline_ddpm import DDPMGFNPipeline
from rtb_utils.diffusers.schedulers.scheduling_ddpm_gfn import DDPMGFNScheduler
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, load_peft_weights, set_peft_model_state_dict
from huggingface_hub import hf_hub_download

from models.samplers import PosteriorPriorDGFN, PosteriorPriorDGFN, PosteriorPriorBaselineSampler
from rtb_utils.plot_utils import compare_distributions, plot_relative_distance_distributions
from rtb_utils.pytorch_utils import NoContext, freeze_model, unfreeze_model, safe_reinit, Logger
from torchvision import transforms

from rtb_utils.replay_buffer import ReplayBuffer
from rtb_utils.simple_io import *

import torch
import torch.nn as nn
import copy
import os
import wandb


def get_DDPM_diffuser_pipeline(args, prior_model):
    """posterior policy diffusion gfn from hf diffusers """

    # Prior flow model pipeline
    prior_model.model.model = freeze_model(prior_model.model.model)
    prior_model.model.model = prior_model.model.model.eval()

    # Prior & Trainable posterior
    outsourced_posterior = unfreeze_model(copy.deepcopy(prior_model.model.model)).train().to(args.device)

    if args.load_outsourced_ckpt:
        print("Loading pretrained outsourced model...")
        checkpoint = torch.load(args.load_outsourced_path)
        outsourced_posterior.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained outsourced model loaded.")
    else:
        safe_reinit(outsourced_posterior)

    outsourced_prior = copy.deepcopy(prior_model.model.model).to(args.device)

    if args.lora:
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=[
                # IPA layers – attention projections and feed-forward blocks
                "linear_q",
                "linear_kv",
                "linear_q_points",
                "linear_kv_points",
                "linear_out",
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
                # Projection from embedding back to latent space
                "emb_to_latent.linear",
                # Time embedder MLP layers (typically the 0th and 2nd layers are linear)
                "t_embedder.mlp.0",
                "t_embedder.mlp.2",
            ],
        )
        outsourced_posterior = get_peft_model(outsourced_posterior, unet_lora_config).to(args.device)

    noise_scheduler = DDPMGFNScheduler(
        num_train_timesteps=args.traj_length,
        beta_end=0.02,
        beta_schedule="linear",
        beta_start=0.0001,
        clip_sample=True,
        variance_type='fixed_large'
    )

    oursourced_posterior_pipeline = DDPMGFNPipeline(unet=outsourced_posterior, scheduler=noise_scheduler)
    oursourced_prior_pipeline = DDPMGFNPipeline(unet=outsourced_prior, scheduler=noise_scheduler)

    diff_gfn = PosteriorPriorDGFN(dim=prior_model.dims[1:],
                                  outsourced_prior_policy=oursourced_prior_pipeline,
                                  outsourced_posterior_policy=oursourced_posterior_pipeline,
                                  prior_model=prior_model,
                                  mixed_precision=False,
                                  config=args)

    if args.mixed_precision and 'cuda' in args.device:
        diff_gfn.half()

    return diff_gfn



class Trainer:

    """general class to achieve finetuning - connects to wandb (save run logs) and hugging face (push repo for easy loading later)"""
    def __init__(
            self,
            sampler,
            reward_function,
            config,
            peptide,
            save_folder,
            optimizer,
            train_loader=None,
            test_loader=None,
            scorer=True,
            *args,
            **kwargs
    ):

        wandb_key = os.getenv('WANDB_API_KEY', None)
        if wandb_key is None:
            print("NOTE: WANDB_API_KEY has not been set in the environment. Wandb tracking is off.")
            self.push_to_wandb = False
        else:
            self.push_to_wandb = True
            wandb.login(key=wandb_key)

        self.sampler = sampler
        self.opt = optimizer
        self.config = config

        self.replay_buffer = ReplayBuffer(
            rb_size=self.config.rb_size,
            rb_sample_strategy=self.config.rb_sample_strategy,
            rb_beta=self.config.rb_beta,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.reward_function = reward_function

        self.save_folder = save_folder

        # ------------------------------------------------------------------------------------------------
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if config.mixed_precision else 'no',
            log_with="wandb" if self.push_to_wandb else "tensorboard",
            gradient_accumulation_steps=config.accumulate_gradient_every,
            cpu=not config.use_cuda,
            project_dir=config.save_folder
        )

        # WANDB & HF

        if self.accelerator.is_main_process:
            exp_name = copy.deepcopy(self.config.exp_name)
            if self.config.exp_name.endswith('_ldm') and self.config.ldm:
                # remove '_ldm' from name first
                exp_name = "_".join(self.config.exp_name.split("_")[:-1])

            exp_name = "_".join(exp_name.split("_")[:-1]).split('-')[0] if '_' in exp_name else exp_name

            wandb.init(
                project="".join(exp_name.split('_')[:2]),
                dir=self.config.save_folder,
                resume=True,
                mode='online' if self.config.push_to_wandb else "offline",
                config={k: str(val) if isinstance(val, (list, tuple)) else val for k, val in self.config.__dict__.items()},
                notes=self.config.notes,
                name=f"{self.config.exp_name.split('_')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            self.checkpoint_dir = f"{self.config.save_folder}checkpoints/"
            self.checkpoint_file = self.checkpoint_dir + "checkpoint.tar"
            folder_create(self.checkpoint_dir, exist_ok=True)

            # -------------------------------------------------------------------------------------------------
            # custom logger for easy tracking and plots later
            self.logger = Logger(config)
            self.logger.save_args()

        self.x_dim = sampler.dim
        self.sampler, self.opt = self.accelerator.prepare(self.sampler, self.opt)

    def run(self, peptide, epochs=5000, **sampler_kwargs):
        it = 0

        self.resume()

        assert self.config.accumulate_gradient_every > 0, "must set 'accumulate_gradient_every' > 0"

        # ------------  Train posterior  ---------
        while it < epochs:

            self.cond_args = self.sampler.prior_model.get_cond_args(device=self.config.device)

            self.sampler.train()
            for si in range(self.config.accumulate_gradient_every):
                print(f"it {it} [{si + 1}/{self.config.accumulate_gradient_every}] : ".endswith(""))

                loss, results_dict = self.sampler_step(peptide, cond=self.cond_args, it=it)

                if loss is not None:
                    self.accelerator.backward(loss)

            self.opt.step()
            self.opt.zero_grad()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:

                self.logger.log(results_dict)  # log results locally

                if it % 10 == 0:

                    plot_logs = self.generate_plots(results_dict, **sampler_kwargs)

                    results_dict.update(plot_logs)

                    self.logger.print(it)  # print progress to terminal
                    self.logger.save()  # save logs file locally

                    self.sampler.save(
                        folder=self.checkpoint_dir,
                        opt=self.opt,
                        push_to_hf=self.config.lora and it % 1000 == 0,
                        it=it
                    )

                    wandb.log(data=results_dict, step=it)  # log results in wandb

                torch.cuda.empty_cache()

            it += 1

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.sampler.save(
                folder=self.checkpoint_dir,
                opt=self.opt,
                push_to_hf=True,
                it=it
            )
        self.accelerator.end_training()
        print("training ended")

    def resume(self):
        raise NotImplementedError()

    def sampler_step(self, *args, **kwargs):
        raise NotImplementedError()

    def generate_plots(self, *args, **kwargs):
        raise NotImplementedError()

class FinetunePlotter:
    def generate_plots(self, batch_logs, *args, **kwargs):
        """generate plots for current prior/posterior modeled distribution"""
        logs = {}
        context = NoContext() if self.sampler.config.method in ['dps', 'fps'] else torch.no_grad()
        with context:
            # compute distribution change
            if self.sampler.prior_model.target_dist is None:
                print("data energy distribution has yet to be computed. Computing...")
                # save all the frames from the actual data
                self.sampler.prior_model.fix_and_save_pdbs(torch.FloatTensor(self.sampler.prior_model.batch_arr))
                # save all the frames from the actual data
                self.sampler.prior_model.target_dist = self.reward_function(self.sampler.prior_model.peptide,
                                                                            data_path=self.sampler.config.data_path,
                                                                            tmp_dir=self.sampler.prior_model.out_dir)
                print("Done!")

            print("Generating energy distribution for current model iteration")
            results_dict = self.sampler(batch_size=self.config.test_sample_size, condition=self.cond_args)
            self.sampler.prior_model.sample(zs0=results_dict['x'])  # sample in place, forms pdbs on disk
            rwd_logs = self.reward_function(
                self.sampler.prior_model.peptide,
                data_path=self.sampler.config.data_path,
                tmp_dir=self.sampler.prior_model.out_dir
            )  # compute reward of whatever is in data_path, then cleans it up
            running_dist = rwd_logs['log_r'].to(self.sampler.device)

            print("Distribution generated!")
            logs.update(compare_distributions(
                self.sampler.prior_model.target_dist['log_r'].detach().cpu(),
                running_dist.detach().cpu())
            )
            logs.update(plot_relative_distance_distributions(
                xyz=rwd_logs['x'],
                n_plots=4,  # Show 4 comparison columns
                target_dist=self.sampler.prior_model.target_dist['x']
            ))

        return logs


class RTBTrainer(Trainer):

    def __init__(
            self,
            sampler,
            reward_function,
            config,
            peptide,
            save_folder,
            optimizer,
            *args,
            **kwargs
    ):
        super().__init__(sampler, reward_function, config, peptide, save_folder, optimizer, *args, **kwargs)

    def resume(self):
        """handles resuming of training from experiment folder"""
        if wandb.run.resumed and file_exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.sampler.load(self.checkpoint_dir)
            params = [param for param in self.sampler.posterior_node.get_unet_parameters() if param.requires_grad]
            self.sampler.logZ = torch.nn.Parameter(torch.tensor(checkpoint["logZ"]).to(self.sampler.device))
            self.opt = type(self.opt.optimizer)([{'params': params,
                                                  'lr': self.config.lr},
                                                 {'params': [self.sampler.logZ],
                                                  'lr': self.config.lr_logZ,
                                                  'weight_decay': self.config.z_weight_decay}])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            it = checkpoint["it"]
            print(f"***** RESUMING PREVIOUS RUN AT IT={it}")

    def sampler_step(self, peptide, it, cond=None, back_and_forth=False, *args, **kwargs):
        """handles one training step"""
        if not back_and_forth:
            # ------ do regular finetuning ------

            if cond['x_cond'].shape[0] > 1:
                batch_size = cond['x_cond'].shape[0]
            else:
                batch_size = self.config.batch_size

            logr_x_prime = None
            x_0 = None

            # todo base rb on ratio from args
            if self.config.replay_buffer and it > 0 and it % 2 == 0:
                x_0, logr_x_prime = self.replay_buffer.sample(batch_size)

            # sample x
            results_dict = self.sampler(
                xs=x_0,
                batch_size=batch_size,
                sample_from_prior=self.config.prior_sampling and it % self.config.prior_sampling_every == 0,
                detach_freq=self.config.detach_freq,
                condition=cond
            )

            # get reward
            if logr_x_prime is None:
                self.sampler.prior_model.sample(zs0=results_dict['x'].to(self.config.device).detach())  # sample in place, forms pdbs on disk
                rwd_logs = self.reward_function(
                    self.sampler.prior_model.peptide,
                    data_path=self.sampler.config.data_path,
                    tmp_dir=self.sampler.prior_model.out_dir
                )  # compute reward of whatever is in data_path, then cleans it up
                logr_x_prime = rwd_logs['log_r'].to(self.sampler.device)
                x_0 = rwd_logs['x']

            self.running_dist = logr_x_prime

            if self.accelerator.is_main_process:

                if self.config.method == 'tb':
                    log_pf_prior_or_pb = results_dict['logpb']
                else:
                    log_pf_prior_or_pb = results_dict['logpf_prior']

                if self.config.vargrad:
                    if cond.shape[0] > 1:
                        i = 0
                        estimates = []
                        while i < results_dict['x'].shape[0]:
                            estimates += [(- results_dict['logpf_posterior'][i:i+self.config.vargrad_sample_n0]
                                           + log_pf_prior_or_pb[i:i+self.config.vargrad_sample_n0]
                                           + logr_x_prime[i:i+self.config.vargrad_sample_n0]).mean()] * self.config.vargrad_sample_n0
                            i += self.config.vargrad_sample_n0
                        self.sampler.logZ.data = torch.FloatTensor(estimates).to(self.config.device)
                    else:
                        vargrad_logz = (- results_dict['logpf_posterior'] + log_pf_prior_or_pb + logr_x_prime).mean()
                        with torch.no_grad():
                            self.sampler.logZ.data = vargrad_logz

                # compute loss rtb for posterior
                loss = 0.5 * (((results_dict['logpf_posterior'] + self.sampler.logZ - log_pf_prior_or_pb - logr_x_prime) ** 2)
                              - self.config.learning_cutoff).relu()

                if x_0 is None:
                    self.replay_buffer.add(x_0.detach(), logr_x_prime.detach(), loss.clone().detach())

                results_dict['PF_divergence'] = (results_dict['logpf_posterior'] - log_pf_prior_or_pb).mean().item()

        else:
            raise NotImplementedError("Back and forth not yet implemented")

        # log additional stuff & save
        results_dict['loss'] = loss.mean()
        results_dict['logr'] = logr_x_prime.mean()
        results_dict['logZ'] = self.sampler.logZ.item()

        if 'x' in results_dict.keys(): del results_dict['x']
        if 'x_prime' in results_dict.keys(): del results_dict['x_prime']
        if 'x_prime_orig' in results_dict.keys(): del results_dict['x_prime_orig']

        return loss.mean(), results_dict


class FinetuneRTBTrainer(RTBTrainer, FinetunePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_plots(self, *args, **kwargs):
        # Call the generate_plots method from InversePlotter
        return FinetunePlotter.generate_plots(self, *args, **kwargs)


class RTBBatchedTrainer(RTBTrainer):

    def sampler_step(self, peptide, cond, back_and_forth=False, *args, **kwargs):
        # ------ do batched finetuning ----
        # get some samples from data

        #  first pas through, get trajectory & loss for correction
        with torch.no_grad():
            # run whole trajectory, and get PFs
            results_dict = self.sampler(
                batch_size=self.config.batch_size,
                save_traj=True  # save trajectory path
            )

            self.sampler.prior_model.sample(zs0=results_dict['x'])  # sample in place, forms pdbs on disk
            rwd_logs = self.reward_function(
                self.sampler.prior_model.peptide,
                data_path=self.sampler.config.data_path,
                tmp_dir=self.sampler.prior_model.out_dir
            )  # compute reward of whatever is in data_path, then cleans it up
            logr_x_prime = rwd_logs['log_r'].to(self.device)

            # this is just for logging & monitoring
            rtb_loss = 0.5 * (((results_dict['logpf_posterior'] + self.sampler.logZ - results_dict['logpf_prior'] - logr_x_prime) ** 2) - self.config.learning_cutoff).relu().mean()

            # compute correction
            clip_idx = ((results_dict['logpf_posterior'] + self.sampler.logZ - results_dict['logpf_prior'] - logr_x_prime) ** 2) < self.config.learning_cutoff
            correction = (results_dict['logpf_posterior'] + self.sampler.logZ - results_dict['logpf_prior'] - logr_x_prime)
            correction[clip_idx] = 0.

        self.sampler.batched_train(
            results_dict['traj'],
            correction,
            batch_size=self.config.batched_rtb_size,
            detach_freq=self.config.detach_freq,
            detach_cut_off=self.config.detach_cut_off
        )

        if self.accelerator.is_main_process:
            results_dict['PF_divergence'] = (results_dict['logpf_posterior'] - results_dict['logpf_prior']).mean().item()

        results_dict['loss'] = rtb_loss
        results_dict['logr'] = logr_x_prime.mean()
        results_dict['logZ'] = self.sampler.logZ

        if 'x' in results_dict.keys(): del results_dict['x']
        if 'x_prime' in results_dict.keys(): del results_dict['x_prime']
        if 'x_prime_orig' in results_dict.keys(): del results_dict['x_prime_orig']
        if 'traj' in results_dict.keys(): del results_dict['traj']

        return None, results_dict


class FinetuneRTBBatchedTrainer(RTBBatchedTrainer, FinetunePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_plots(self, *args, **kwargs):
        # Call the generate_plots method from InversePlotter
        return FinetunePlotter.generate_plots(self, *args, **kwargs)

