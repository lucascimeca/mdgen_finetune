import diffusers

from diffusers import DDPMPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from rtb_utils.pytorch_utils import NoContext, freeze_model, unfreeze_model, safe_reinit, Logger
from rtb_utils.plot_utils import compare_distributions, plot_relative_distance_distributions, plot_TICA, plot_TICA_PCA
from models.samplers import PosteriorPriorDGFN, PosteriorPriorDGFN, PosteriorPriorBaselineSampler

from rtb_utils.replay_buffer import ReplayBuffer
from datetime import datetime
from diffusers.training_utils import compute_snr
from rtb_utils.diffusers.schedulers.scheduling_ddpm_gfn import DDPMGFNScheduler
from random import random
from rtb_utils.simple_io import *

from diffusers.utils import make_image_grid
from huggingface_hub import create_repo, upload_folder
from pathlib import Path
from tqdm.auto import tqdm
from accelerate import Accelerator

from rtb_utils.diffusers.pipelines.ddpm_gfn.pipeline_ddpm import DDPMGFNPipeline
from rtb_utils.simple_io import DictObj, folder_create

import random
import copy
import wandb

import os
import torch
import torch.nn.functional as F
import huggingface_hub as hb


def get_DDPM_diffuser_pipeline(args, prior_model, outsourced_sampler=None):
    """posterior policy diffusion gfn from hf diffusers """

    # Prior flow model pipeline
    prior_model.model.model = freeze_model(prior_model.model.model)
    prior_model.model.model = prior_model.model.model.eval()

    if outsourced_sampler is None:
        # Prior & Trainable posterior
        outsourced_posterior = unfreeze_model(copy.deepcopy(prior_model.model.model)).train().to(args.device)

        if args.load_outsourced_ckpt:
            print("Loading pretrained outsourced model...")
            checkpoint = torch.load(args.load_outsourced_path)
            outsourced_posterior.load_state_dict(checkpoint)
            print("Pretrained outsourced model loaded.")
        else:
            safe_reinit(outsourced_posterior)
    else:
        outsourced_posterior = outsourced_sampler

    outsourced_prior = freeze_model(copy.deepcopy(outsourced_posterior)).eval().to(args.device)

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
        num_inference_steps=args.sampling_length,
        beta_end=0.02,
        beta_start=0.0001,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=True,
        clip_sample_range=3,
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
            save_folder,
            optimizer,
            train_loader=None,
            test_loader=None,
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
                project="_".join(exp_name.split('_')[:2]),
                dir=self.config.save_folder,
                resume=True,
                mode='online' if self.config.push_to_wandb else "offline",
                config={k: str(val) if isinstance(val, (list, tuple)) else val for k, val in self.config.__dict__.items()},
                notes=self.config.notes,
                name=self.config.exp_name
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

    def run(self, peptide=None, epochs=5000, **sampler_kwargs):
        it = 0

        self.resume()

        assert self.config.accumulate_gradient_every > 0, "must set 'accumulate_gradient_every' > 0"

        # ------------  Train posterior  ---------
        while it < epochs:

            self.cond_args = self.sampler.prior_model.get_cond_args(device=self.config.device)

            self.sampler.train()
            for si in range(self.config.accumulate_gradient_every):
                print(f"it {it} [{si + 1}/{self.config.accumulate_gradient_every}] : ".endswith(""))

                loss, results_dict = self.sampler_step(it, peptide=peptide, cond=self.cond_args)

                if loss is not None:
                    self.accelerator.backward(loss)

            # self.accelerator.clip_grad_norm_(self.sampler.posterior_node.policy.unet.parameters(), 10.0)
            self.opt.step()
            self.opt.zero_grad()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:

                self.logger.log(results_dict)  # log results locally

                if it % 50 == 0:
                    plot_logs = self.generate_plots(
                        prior_model=self.sampler.prior_model,
                        reward_function=self.reward_function,
                        sampler=self.sampler,
                        config=self.config,
                        cond_args=self.sampler.prior_model.get_cond_args(device=self.config.device,
                                                                         multi_peptide=it % 250 == 0,
                                                                         size=self.config.test_sample_size),
                        all_peptides=it % 250 == 0,
                    )

                    results_dict.update(plot_logs)

                    self.logger.print(it)  # print progress to terminal
                    self.logger.save()  # save logs file locally

                    self.sampler.save(
                        logZ=self.sampler.logZ.item(),
                        folder=self.checkpoint_dir,
                        opt=self.opt,
                        push_to_hf=self.config.lora and it % 1000 == 0,
                        it=it
                    )

                if it % 5 == 0:

                    for k, v in results_dict.items():
                        if isinstance(v, torch.Tensor):
                            results_dict[k] = v.mean().item()

                    wandb.log(data=results_dict, step=it)  # log results in wandb

                torch.cuda.empty_cache()

            it += 1

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.sampler.save(
                logZ=self.sampler.logZ.item(),
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


class FinetunePlotter: # self.sampler.prior_model, self.reward_function, self.sampler, self.config, self.sampler.config.data_path, self.cond_args
    @staticmethod
    def generate_plots(prior_model, reward_function, sampler, config, cond_args, all_peptides=False):
        """generate plots for current prior/posterior modeled distribution"""

        batch = cond_args[1]
        cond_args = cond_args[0]

        logs = {}
        context = NoContext() if config.method in ['dps', 'fps'] else torch.no_grad()
        with context:
            peptides = np.unique(cond_args['peptide'])
            for peptide in peptides:
                if peptide not in prior_model.target_dist:
                    # compute distribution change
                    print("data energy distribution has yet to be computed. Computing...")
                    # save all the frames from the actual data
                    file_path = os.path.join(prior_model.data_dir, f"{peptide}{prior_model.suffix}.npy")
                    arr = np.lib.format.open_memmap(file_path, 'r')
                    idxes = np.random.randint(0, len(arr), size=config.test_sample_size)
                    prior_model.fix_and_save_pdbs(torch.FloatTensor(arr[idxes]), peptide)

            # save all the frames from the actual data
            prior_model.target_dist = reward_function(data_path=config.data_path, tmp_dir=prior_model.out_dir)[0]
            print("Done!")

            print("Generating energy distribution for current model iteration")

            results_dict = sampler(
                batch_size=config.test_sample_size,
                condition=cond_args
            )
            _, _, _, paths = prior_model.sample(batch=batch, zs0=results_dict['x'].to(config.device).detach())  # sample in place, forms pdbs on disk
            rwd_logs, logrs = reward_function(
                paths=paths,
                data_path=sampler.config.data_path,
                tmp_dir=sampler.prior_model.out_dir
            )  # compute reward of whatever is in data_path, then cleans it up

            for peptide in peptides:
                logs[peptide] = {}

                print("Distribution generated!")
                logs[peptide].update(compare_distributions(
                    sampler.prior_model.target_dist[peptide]['log_r'].detach().cpu(),
                    rwd_logs[peptide]['log_r'].to(sampler.device).detach().cpu())
                )
                logs[peptide].update(plot_relative_distance_distributions(
                    xyz=rwd_logs[peptide]['x'],
                    n_plots=4,  # Show 4 comparison columns
                    target_dist=sampler.prior_model.target_dist[peptide]['x']
                ))
                logs[peptide].update(plot_TICA_PCA(
                    samples_torsions=rwd_logs[peptide]['torsions'],
                    target_torsion=sampler.prior_model.target_dist[peptide]['torsions']
                ))
                logs[peptide].update(plot_TICA_PCA(
                    samples_torsions=rwd_logs[peptide]['torsions'],
                    target_torsion=sampler.prior_model.target_dist[peptide]['torsions'],
                    scale=True
                ))

                if not all_peptides:
                    break

        return logs


class RTBTrainer(Trainer):

    def __init__(
            self,
            sampler,
            reward_function,
            config,
            save_folder,
            optimizer,
            peptide=None,
            *args,
            **kwargs
    ):
        super().__init__(sampler, reward_function, config, save_folder, optimizer, peptide, *args, **kwargs)

    def resume(self):
        """handles resuming of training from experiment folder"""
        if file_exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.sampler.load(self.checkpoint_dir)
            self.sampler.logZ = torch.nn.Parameter(torch.tensor(checkpoint["logZ"]).to(self.sampler.device))
            params = [param for param in self.sampler.posterior_node.get_unet_parameters() if param.requires_grad]
            self.opt = type(self.opt.optimizer)([{'params': params,
                                                  'lr': self.config.learning_rate},
                                                 {'params': [self.sampler.logZ],
                                                  'lr': self.config.lr_logZ}])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            it = checkpoint["it"]
            print(f"***** RESUMING PREVIOUS RUN AT IT={it}")

    def sampler_step(self, it, peptide=None, cond=None, back_and_forth=False, *args, **kwargs):

        cond_args, batch = cond
        """handles one training step"""
        if not back_and_forth:
            # ------ do regular finetuning ------

            if cond_args['x_cond'].shape[0] > 1:
                batch_size = cond_args['x_cond'].shape[0]
            else:
                batch_size = self.config.batch_size

            logr_x_prime = None
            x_0 = None

            if self.config.replay_buffer and it > self.config.batch_size and random.random() < self.config.rb_ratio:
                print("REPLAY BUFFER")
                x_0, logr_x_prime = self.replay_buffer.sample(batch_size)

            # sample x
            results_dict = self.sampler(
                xs=x_0,
                batch_size=batch_size,
                sample_from_prior=self.config.prior_sampling and random.random() < self.config.prior_sampling_ratio,
                detach_freq=self.config.detach_freq,
                condition=cond_args,
            )

            # get reward
            if logr_x_prime is None:
                _, _, _, paths = self.sampler.prior_model.sample(batch=batch, zs0=results_dict['x'].to(self.config.device).detach())  # sample in place, forms pdbs on disk
                rwd_logs, logr_x_prime = self.reward_function(
                    paths=paths,
                    data_path=self.sampler.config.data_path,
                    tmp_dir=self.sampler.prior_model.out_dir
                )  # compute reward of whatever is in data_path, then cleans it up

            self.running_dist = logr_x_prime

            if self.accelerator.is_main_process:

                if self.config.method == 'tb':
                    log_pf_prior_or_pb = results_dict['logpb']
                else:
                    log_pf_prior_or_pb = results_dict['logpf_prior']

                if self.config.vargrad:

                    if cond_args['x_cond'].shape[0] > 1:
                        results_dict['logZ'] = {}
                        peptides = np.unique(cond_args['peptide'])
                        vargrad_logzs = torch.zeros(len(cond_args['peptide'])).float()
                        for peptide in peptides:
                            idx = [i for i in range(len(cond_args['peptide'])) if peptide == cond_args['peptide'][i]]
                            vargrad = (- results_dict['logpf_posterior'][idx].to(self.config.device)
                                       + log_pf_prior_or_pb[idx].to(self.config.device)
                                       + logr_x_prime[idx].to(self.config.device)).mean().detach()

                            results_dict['logZ'][peptide] = vargrad.item()
                            vargrad_logzs[idx] = vargrad
                    else:
                        vargrad_logzs = (- results_dict['logpf_posterior'] + log_pf_prior_or_pb + logr_x_prime).detach()
                        with torch.no_grad():
                            self.sampler.logZ.data = vargrad_logzs.mean()
                        results_dict['logZ'] = self.sampler.logZ.item()

                # compute loss rtb for posterior
                loss = 0.5 * (((results_dict['logpf_posterior'] + self.sampler.logZ - log_pf_prior_or_pb - logr_x_prime) ** 2)
                              - self.config.learning_cutoff).relu()

                if x_0 is None:
                    self.replay_buffer.add(results_dict['x'].detach(), logr_x_prime.detach(), loss.clone().detach())

                if self.config.vargrad:
                    results_dict['vargrad_var'] = vargrad_logzs.var()

                results_dict['PF_divergence'] = (results_dict['logpf_posterior'] - results_dict['logpf_prior']).mean().item()

        else:
            raise NotImplementedError("Back and forth not yet implemented")

        # log additional stuff & save
        results_dict['loss'] = loss.mean()
        results_dict['logr'] = logr_x_prime.mean()
        results_dict['peptide_logrs'] = {peptide: rwd_logs[peptide]['log_r'].mean().item() for peptide in rwd_logs.keys()}

        if 'x' in results_dict.keys(): del results_dict['x']
        if 'x_prime' in results_dict.keys(): del results_dict['x_prime']
        if 'x_prime_orig' in results_dict.keys(): del results_dict['x_prime_orig']

        return loss.mean(), results_dict


class FinetuneRTBTrainer(RTBTrainer, FinetunePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_plots(self, *args, **kwargs):
        # Call the generate_plots method from InversePlotter
        return FinetunePlotter.generate_plots(*args, **kwargs)


class RTBBatchedTrainer(RTBTrainer):

    def sampler_step(self, peptide, cond_args, back_and_forth=False, *args, **kwargs):
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


# trainer class
class TrainingConfig:
    def __init__(self, args):
        self.train_batch_size = args.batch_size
        self.eval_batch_size = args.plot_batch_size  # how many images to sample during evaluation
        self.num_epochs = args.epochs
        self.gradient_accumulation_steps = args.accumulate_gradient_every
        self.learning_rate = args.learning_rate
        self.test_sample_size = args.test_sample_size
        self.lr_warmup_steps = 500
        self.save_image_epochs = 300
        self.save_model_epochs = 200
        self.method = args.method
        self.data_path = args.data_path
        self.mixed_precision = "fp16" if args.mixed_precision else 'no'  # `no` for float32, `fp16` for automatic mixed precision
        self.output_dir = args.save_folder  # the model name locally and on the HF Hub
        self.inference_steps = args.sampling_length
        self.variance_type = 'fixed_large'  # 'fixed_large'
        self.push_to_wandb = args.push_to_wandb
        self.notes = args.notes
        self.snr_training = args.snr_training
        self.snr_gamma = args.snr_gamma

        self.push_to_hf = args.push_to_hf  # whether to upload the saved model to the HF Hub
        self.exp_name = args.exp_name  # the name of the repository to create on the HF Hub
        self.hub_private_repo = True
        self.overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        self.seed = args.seed
        self.device = args.device


def evaluate(batch_size, epoch, pipeline, folder, inference_steps=50, seed=123):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=batch_size,
        generator=torch.manual_seed(seed),
        num_inference_steps=inference_steps,
    ).images

    image_grid = make_image_grid(images, rows=5, cols=6)

    # Save the images
    test_dir = os.path.join(folder, "samples")
    os.makedirs(test_dir, exist_ok=True)
    filename = f"{test_dir}/{epoch:04d}.png"
    image_grid.save(filename)
    return filename


class DiffuserTrainer():

    def __init__(
            self,
            config,
            model,
            sampler,
            source_sampler,
            scheduler,
            optimizer,
            lr_scheduler,
            reward_function,
            xT_type = "gaussian",
    ):
        self.config = config
        self.model = model
        self.sampler = sampler
        self.noise_scheduler = scheduler
        self.source_sampler = source_sampler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reward_function = reward_function
        self.xT_type = xT_type

        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.config.output_dir, "logs"),
            cpu=self.config.device == 'cpu'
        )
        if self.accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
            if self.config.push_to_hf:
                hf_token = os.getenv('HF_TOKEN', None)
                if hf_token is None:
                    print("No HuggingFace token was set in 'HF_TOKEN' env. variable. Setting push_to_hf to false.")
                    self.config.push_to_hf = False
                else:
                    print("HF login succesfull!")
                    wandb.login(token=hf_token)

                    self.hub_model_id = f"{hb.whoami()['name']}/{self.config.exp_name}"
                    self.repo_id = create_repo(
                        repo_id=self.hub_model_id or Path(self.config.output_dir).name, exist_ok=True
                    ).repo_id

            exp_name = "_".join(self.config.exp_name.split("_")[:-1]).split('-')[0] if '_' in self.config.exp_name else self.config.exp_name

            wandb.init(
                project="_".join(exp_name.split('_')[:2]),
                dir=self.config.output_dir,
                resume=True,
                mode='online' if self.config.push_to_wandb else "offline",
                config={k: str(val) if isinstance(val, (list, tuple)) else val for k, val in self.config.__dict__.items()},
                name=self.config.exp_name
            )
            self.checkpoint_dir = f"{self.config.output_dir}checkpoints/"
            self.checkpoint_file = self.checkpoint_dir + "checkpoint.tar"
            folder_create(self.checkpoint_dir, exist_ok=True)

    def train(self):
        # Initialize accelerator and tensorboard logging

        it = 0
        if wandb.run.resumed and file_exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            self.sampler.load(self.checkpoint_dir)
            pipeline = self.sampler.posterior_node.policy

            self.model = pipeline.unet
            self.model.train()
            self.noise_scheduler = pipeline.scheduler

            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.last_epoch = checkpoint["scheduler_last_epoch"]
            it = checkpoint["it"]
            print(f"***** RESUMING PREVIOUS RUN AT IT={it}")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        accel_set = self.model, self.optimizer
        if self.lr_scheduler is not None:
            accel_set = accel_set + (self.lr_scheduler,)
        optim_set = self.accelerator.prepare(accel_set)
        if len(optim_set) == 2:
            model, optimizer = optim_set
            lr_scheduler = None
        else:
            model, optimizer, lr_scheduler = optim_set

        # train the model
        progress_bar = tqdm(total=self.config.num_epochs, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Iteration: {it}")

        cond_args = self.sampler.prior_model.get_cond_args(device=self.config.device)

        while it < self.config.num_epochs:

            clean_images = self.source_sampler.sample()

            # Sample noise to add to the images
            if self.xT_type == 'uniform':
                noise = torch.rand(clean_images.shape, device=clean_images.device) * 6 - 3
            else:
                noise = torch.randn(clean_images.shape, device=clean_images.device)

            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

            with self.accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, **cond_args[0])[0]

                if self.config.snr_training:
                    snr = compute_snr(self.noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(noise_pred, noise)

                self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(),
                    "step": it}
            if lr_scheduler is not None:
                logs["lr"] = lr_scheduler.get_last_lr()[0]

            progress_bar.set_postfix(**logs)

            if self.accelerator.is_main_process:

                with torch.no_grad():

                    # After each epoch you optionally sample some demo images with evaluate() and save the model
                    if (it % self.config.save_image_epochs == 0 or it == self.config.num_epochs - 1):

                        try:
                            logs.update(FinetunePlotter.generate_plots(
                                prior_model=self.sampler.prior_model,
                                reward_function=self.reward_function,
                                sampler=self.sampler,
                                config=self.config,
                                cond_args=cond_args[0]
                            ))
                        except Exception as e:
                            print("no plots could be generated, the prior likely needs to train for longer")

                    wandb.log(logs, step=it)  # log results in wandb

                    # After each epoch you optionally sample some demo images with evaluate() and save the model
                    if (it % self.config.save_model_epochs == 0 or it == self.config.num_epochs - 1):
                        self.sampler.save(
                            self.checkpoint_dir,
                            self.config.push_to_hf,
                            self.optimizer,
                            it=it,
                            logZ=0.,
                            scheduler_last_epoch=lr_scheduler.last_epoch if lr_scheduler is not None else 0
                        )

            it += 1

        if self.config.push_to_hf:
            upload_folder(
                repo_id=self.repo_id,
                folder_path=self.checkpoint_dir,
                commit_message=f"Iteration {it}",
                ignore_patterns=["step_*", "epoch_*", "wandb*"],
            )

