import copy
import os
import shutil

import matplotlib.pyplot as plt

import mdgen.rigid_utils
from mdgen.model.latent_model import LatentMDGenModel
from rtb_utils.plot_utils import compare_distributions, plot_relative_distance_distributions
from rtb_utils.pytorch_utils import safe_reinit, get_batch, create_batches

os.environ['PYMOL_QUIET'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
import torch
import torch.nn as nn
from torchcfm.models.unet.unet import UNetModelWrapper
import wandb
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, load_peft_weights, set_peft_model_state_dict

import random

from rtb_utils.sde import VPSDE, DDPM
import rtb_utils as utils

from datetime import datetime


class ProteinRTBModel(nn.Module):

    cond_args = None

    def __init__(self,
                 device,
                 reward_model,
                 prior_model,
                 in_shape,
                 reward_args,
                 id,
                 model_save_path,
                 langevin=False,
                 inference_type='vpsde',
                 tb=False,
                 load_ckpt=False,
                 load_ckpt_path=None,
                 entity='swish',
                 diffusion_steps=100,
                 beta_start=1.0,
                 beta_end=10.0,
                 loss_batch_size=64,
                 replay_buffer=None,
                 orig_scale=1.,
                 config={}
                 ):
        super().__init__()

        self.device = device
        self.config = config

        if inference_type == 'vpsde':
            self.sde = VPSDE(
                device=self.device,
                beta_schedule='cosine',
                orig_scale=orig_scale
            )
        else:
            self.sde = DDPM(
                device=self.device,
                beta_schedule='cosine',
                orig_scale=orig_scale
            )
        self.sde_type = self.sde.sde_type

        self.steps = diffusion_steps
        self.reward_args = reward_args
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start
        self.in_shape = in_shape
        self.loss_batch_size = loss_batch_size
        self.replay_buffer = replay_buffer
        self.use_rb = False if replay_buffer is None else True

        self.wandb_track = config.wandb_track

        # for run name
        self.id = id
        self.entity = entity

        self.latent_prior = torch.distributions.Normal(torch.tensor(0.).to(self.device),
                                                       torch.tensor(1.).to(self.device))
        self.tb = tb

        # Posterior noise model
        self.logZ = torch.nn.Parameter(torch.tensor(0.).to(self.device))

        # shape as (C, H, W), then reshape to in-shape (64, 7) when passing to prior model
        self.gfn_shape = (21, 2, 2)  # 10, 10)

        self.mlp_dim = self.gfn_shape[0] * self.gfn_shape[1] * self.gfn_shape[2]

        self.model = copy.deepcopy(prior_model.model.model).to(self.device)
        safe_reinit(self.model)

        if not self.tb:
            self.ref_model = copy.deepcopy(prior_model.model.model).to(self.device)
            self._freeze_model(self.ref_model)

        # Prior flow model pipeline
        self.prior_model = prior_model

        self.reward_model = reward_model

        self.langevin = langevin

        self.model_save_path = os.path.expanduser(model_save_path)

        self.load_ckpt = load_ckpt
        if load_ckpt_path is not None:
            self.load_ckpt_path = os.path.expanduser(load_ckpt_path)
        else:
            self.load_ckpt_path = load_ckpt_path

    def _freeze_model(self, model):
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def get_cond_args(self):
        if self.cond_args is None:
            item = get_batch(self.prior_model.peptide,
                             self.prior_model.peptide,
                             tps=self.prior_model.tps,
                             no_frames=self.prior_model.model.args.no_frames,
                             data_dir=self.prior_model.data_dir,
                             suffix=self.prior_model.suffix)
            batch = next(iter(torch.utils.data.DataLoader([item])))
            prep = self.prior_model.model.prep_batch(batch)
            self.cond_args = prep['model_kwargs']
            for k, v in self.cond_args.items():
                self.cond_args[k] = v.to(self.device)
        return self.cond_args

    def save_checkpoint(self, model, optimizer, epoch, run_name):
        if self.model_save_path is None:
            print("Model save path not provided. Checkpoint not saved.")
            return

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        savedir = self.model_save_path + run_name + '/'
        os.makedirs(savedir, exist_ok=True)

        filepath = savedir + 'checkpoint_' + str(epoch) + '.pth'
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def load_checkpoint(self, model, optimizer):
        if self.load_ckpt_path is None:
            print("Checkpoint path not provided. Checkpoint not loaded.")
            return model.to(self.device), optimizer

        checkpoint = torch.load(self.load_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {self.load_ckpt_path}")

        it = 0

        return model.to(self.device), optimizer, it

    def classifier_reward(self, x):
        if self.num_classes == 1:
            log_r_target = self.log_reward(x)['log_r'].to(self.device)
            log_r_pred = self.trainable_reward(x)
            loss = ((log_r_pred - log_r_target) ** 2).mean()
        else:
            im = self.prior_model(x)
            target_log_probs = self.reward_model.get_class_logits(im, *self.reward_args)
            pred_log_probs = torch.nn.functional.log_softmax(self.trainable_reward(x))
            loss = torch.nn.CrossEntropyLoss()(pred_log_probs, target_log_probs.argmax(dim=-1)).mean()
        return loss

    def update_trainable_reward(self, x):
        if not self.langevin:
            print("Trainable reward not initialized.")
            return

        self.cls_optimizer.zero_grad()
        loss = self.classifier_reward(x)
        loss.backward()
        self.cls_optimizer.step()
        return

    def pretrain_trainable_reward(self, batch_size, n_iters=100, learning_rate=5e-5):
        if not self.langevin:
            print("Trainable reward not initialized.")
            return

        B = batch_size
        D = self.in_shape

        run_name = self.id + '_pretrain_reward_lr_' + str(learning_rate)

        if self.wandb_track:
            wandb.init(
                project='mdgen_cfm_posterior',
                entity=self.entity,
                save_code=True,
                name=run_name,
                config=self.config
            )
            hyperparams = {
                "learning_rate": learning_rate,
                "reward_args": self.reward_args,
                "training reward": self.langevin
            }
            wandb.config.update(hyperparams)

        for i in range(n_iters):
            self.cls_optimizer.zero_grad()

            x = torch.randn(B, *D, device=self.device)

            if self.num_classes == 1:
                log_r_target = self.log_reward(x)['log_r'].to(self.device)
                log_r_pred = self.trainable_reward(x)
                loss = ((log_r_pred - log_r_target) ** 2).mean()

            else:
                im = self.prior_model(x)
                target_log_probs = self.reward_model.get_class_logits(im, *self.reward_args)
                pred_log_probs = torch.nn.functional.log_softmax(self.trainable_reward(x))
                loss = torch.nn.CrossEntropyLoss()(pred_log_probs, target_log_probs.argmax(dim=-1)).mean()
            loss.backward()
            self.cls_optimizer.step()

            if self.wandb_track:
                wandb.log({"loss": loss.item(), "iter": i})
                if i % 100 == 0:

                    with torch.no_grad():
                        x = torch.randn(20, *self.in_shape, device=self.device)
                        img = self.prior_model(x)
                        prior_reward = self.reward_model(img, *self.reward_args)
                        trained_reward = self.trainable_reward(x).log_softmax(dim=-1)

                        wandb.log({"prior_samples": [wandb.Image(img[k],
                                                                 caption="logR(x1) = {}, TrainlogR(z) = {}".format(
                                                                     prior_reward[k], trained_reward[k])) for k in
                                                     range(len(img))]})

            if i % 10 == 0:
                print("Iter: ", i, "Loss: ", loss.item())

        return

        # linearly anneal between beta_start and beta_end

    def get_beta(self, it, anneal, anneal_steps):
        if anneal and it < anneal_steps:
            beta = ((anneal_steps - it) / anneal_steps) * self.beta_start + (it / anneal_steps) * self.beta_end
        else:
            beta = self.beta_start

        return beta

        # return shape is: (B, *D)

    def prior_log_prob(self, x):
        return self.latent_prior.log_prob(x).sum(dim=tuple(range(1, len(x.shape))))

    def log_reward(self, x, from_prior=False):
        with torch.no_grad():
            if not from_prior:
                self.prior_model.sample(zs0=x)
            else:
                self.prior_model.sample()
            rwd_logs = self.reward_model(
                self.prior_model.peptide,
                data_path=self.config.data_path,
                tmp_dir=self.prior_model.out_dir
            )
        return rwd_logs

    def batched_rtb(self, shape, learning_cutoff=.1, prior_sample=False, rb_sample=False):
        # first pas through, get trajectory & loss for correction
        B, *D = shape
        x_1 = None
        if rb_sample:
            x_1, logr_x_prime = self.replay_buffer.sample(shape[0])

        with torch.no_grad():
            # run whole trajectory, and get PFs
            if self.sde_type == 'vpsde' and not self.tb:
                print("In batched RTB doing REF PROC")
                fwd_logs = self.forward_ref_proc(
                    shape=shape,
                    steps=self.steps,
                    save_traj=True,  # save trajectory fwd
                    prior_sample=prior_sample,
                    x_1=x_1,
                    backward=rb_sample
                )
            elif self.sde_type == 'vpsde':
                fwd_logs = self.forward(
                    shape=shape,
                    steps=self.steps,
                    save_traj=True,  # save trajectory fwd
                    prior_sample=prior_sample,
                    x_1=x_1,
                    backward=rb_sample
                )

            elif self.sde_type == 'ddpm':
                fwd_logs = self.forward_ddpm(
                    shape=shape,
                    steps=self.steps,
                    save_traj=True,  # save trajectory fwd
                    prior_sample=prior_sample,
                    x_1=x_1,
                    backward=rb_sample
                )
            x_mean_posterior, logpf_prior, logpf_posterior = fwd_logs['x_mean_posterior'], fwd_logs['logpf_prior'], fwd_logs['logpf_posterior']

            if rb_sample:
                scale_factor = 1.0  # 0.5
            else:
                scale_factor = 1.0

            if not rb_sample:
                rwd_logs = self.log_reward(x_mean_posterior)
                logr_x_prime = rwd_logs['log_r'].to(self.device)
                x_1 = rwd_logs['x']

            # for off policy stability 
            logpf_posterior = logpf_posterior * scale_factor
            logpf_prior = logpf_prior * scale_factor

            self.logZ.data = (-logpf_posterior + logpf_prior + self.beta * logr_x_prime).mean().detach()

            print("logpf_posterior: ", logpf_posterior.mean().item())
            print("logpf_prior: ", logpf_prior.mean().item())
            print("logr_x_prime: ", logr_x_prime.mean().item())
            print("logZ: ", self.logZ.item())

            rtb_loss = 0.5 * (((logpf_posterior + self.logZ - logpf_prior - self.beta * logr_x_prime) ** 2) - learning_cutoff).relu()

            # Add to replay_buffer
            if not rb_sample and self.replay_buffer is not None:
                self.replay_buffer.add(x_mean_posterior.detach(), logr_x_prime.detach(), rtb_loss.detach())

            # compute correction
            clip_idx = ((logpf_posterior + self.logZ - logpf_prior - self.beta * logr_x_prime) ** 2) < learning_cutoff
            correction = (logpf_posterior + self.logZ - logpf_prior - self.beta * logr_x_prime)
            correction[clip_idx] = 0.

        if self.sde_type == 'vpsde':
            self.batched_forward(
                shape=shape,
                traj=fwd_logs['traj'],
                correction=correction,
                batch_size=B,
                rb=rb_sample
            )
        elif self.sde_type == 'ddpm':

            self.batched_forward_ddpm(
                shape=shape,
                traj=fwd_logs['traj'],
                correction=correction,
                batch_size=B
            )

        return {
            'loss': rtb_loss,
            'logr': logr_x_prime,
            'logpf_posterior': logpf_posterior,
            'logpf_prior': logpf_prior,
            'x': x_1
        }

    def dsm_loss(self, x0, t):
        B = x0.shape[0]
        D = x0.shape[1:]

        m, std = self.sde.marginal_prob_scalars(t)
        noise = torch.randn_like(x0)
        m = m.reshape((-1, *[1] * len(D)))
        std = std.reshape((-1, *[1] * len(D)))

        # print("x0 shape: ",x0.shape)
        # print("m shape: ", m.shape)
        # print("std shape: ", std.shape)

        xt = m * x0 + std * noise

        # get model prediction 
        dt = -1 / self.steps

        m_t1, std_t1 = self.sde.marginal_prob_scalars(t + dt)
        m_t1 = m_t1.reshape((-1, *[1] * len(D)))

        xt1_mean = m_t1 * x0

        g = self.sde.diffusion(t, xt)

        model_out = self.model(xt, t, **self.get_cond_args())
        # xt_r = xt.permute(0, )
        posterior_drift = -self.sde.drift(t, xt) - (g ** 2) * model_out / self.sde.sigma(t).view(-1,*[1] * len(D))

        f_posterior = posterior_drift
        # compute parameters for denoising step (wrt posterior)
        x_mean_posterior = xt + f_posterior * dt  # * (-1.0 if backward else 1.0)
        std = g * (np.abs(dt)) ** (1 / 2)

        loss = ((xt1_mean - x_mean_posterior) ** 2 / std).mean()  # .mean(dim=[1,2])

        return loss

        # do denoising score matching to pretrain gfn

    def denoising_score_matching_unif(self, n_iters=1000, learning_rate=5e-5, clip=0.1):
        param_list = [{'params': self.model.parameters()}]
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
        run_name = '../pretrained/DSM_unif_' + "unet_type" + self.id  # + '_sde_' + self.sde_type +'_steps_' + str(self.steps) + '_lr_' + str(learning_rate)

        if self.wandb_track:
            wandb.init(
                project='mdgen_cfm_prior',
                entity=self.entity,
                save_code=True,
                name=run_name,
                config=self.config
            )
        B = 64  # 32

        if self.load_ckpt:
            self.model, optimizer, load_it = self.load_checkpoint(self.model, optimizer)
        else:
            load_it = 0

        for it in range(load_it, load_it + n_iters):
            optimizer.zero_grad()

            # sample uniform so3
            # x0 = self.prior_model.sample_prior(batch_size=B, sample_length=self.in_shape[0])
            # x0 = x0["rigids_t"].to(self.device)
            T, L, _ = self.in_shape
            x0 = self.prior_model.model.sample_prior_latent(B, T, L, self.device)

            # sample random timestep 
            t = torch.rand(B).to(self.device) * self.sde.T

            # compute drift
            x0.requires_grad = True

            # denoising score matching loss
            loss = self.dsm_loss(x0, t)

            loss.backward()

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)

            optimizer.step()

            print("Iteration {}, Loss: {:.4f}".format(it, loss.item()))

            if self.wandb_track and not it % 100 == 0:
                wandb.log({"loss": loss.item(), "epoch": it})

            if self.wandb_track and it % 100 == 0:
                wandb.log({"loss": loss.item(), "epoch": it})
                self.save_checkpoint(self.model, optimizer, it, run_name)

        return

    def finetune(self, shape, n_iters=100000, learning_rate=5e-5, clip=0.1, prior_sample_prob=0.0,
                 replay_buffer_prob=0.0, anneal=False, anneal_steps=15000):

        param_list = [{'params': self.model.parameters()}]
        optimizer = torch.optim.Adam(param_list, lr=learning_rate)
        run_name = self.id + '_sde_' + self.sde_type + '_steps_' + str(self.steps) + '_lr_' + str(
            learning_rate) + '_beta_start_' + str(self.beta_start) + '_beta_end_' + str(
            self.beta_end) + '_anneal_' + str(anneal) + '_prior_prob_' + str(prior_sample_prob) + '_rb_prob_' + str(
            replay_buffer_prob) + '_clip_' + str(clip) + '_tb_' + str(self.tb) + f'_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

        self.tmp_dir = os.path.expanduser("~/scratch/CNF_tmp/" + run_name + '/')
        self.running_dist = torch.FloatTensor([])
        print("TMP DIR for pdb files: ", self.tmp_dir)

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.prior_model.out_dir = self.tmp_dir + "samples/"

        if not os.path.exists(self.prior_model.out_dir):
            os.makedirs(self.prior_model.out_dir)

        if self.load_ckpt:
            self.model, optimizer, load_it = self.load_checkpoint(self.model, optimizer)
            if not self.tb:
                self.ref_model = copy.deepcopy(self.model).to(self.device)
                self.ref_model = self.ref_model.eval()
                if self.config.lora:
                    unet_lora_config = LoraConfig(
                        r=self.config.rank,
                        lora_alpha=self.config.rank,
                        init_lora_weights="gaussian",
                        target_modules=[
                            "conv",
                            "in_layers.2",
                            "in_layers.3",
                            "emb_layers.1",
                            "out_layers.3",
                            "out.2",
                            "op",
                        ],
                    )
                    self.model = get_peft_model(self.model, unet_lora_config).to(self.device)
                    prior_params = sum(p.numel() for p in self.ref_model.parameters())
                    posterior_params = sum(p.numel() for p in self.model.parameters())
                    trainable_posterior_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    print(f"\nTotal params: "
                          f"\nPRIOR model: {prior_params / 1e6:.2f}M "
                          f"\nPOSTERIOR model: {posterior_params / 1e6:.2f}M"
                          f"\nTrainable posterior parameters: {trainable_posterior_params / 1e6:.2f}M/{posterior_params / 1e6:.2f}M  ({trainable_posterior_params * 100 / posterior_params:.2f}%)\n")

        else:
            load_it = 0

        if self.wandb_track:
            wandb.init(
                project='mdgen_cfm_posterior',
                entity=self.entity,
                save_code=True,
                name=run_name,
                config=self.config
            )
            hyperparams = {
                "learning_rate": learning_rate,
                "n_iters": n_iters,
                "reward_args": self.reward_args,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "anneal": anneal,
                "anneal_steps": anneal_steps,
                "clip": clip,
                "tmp_dir": self.tmp_dir,
                "load ckpt path": self.load_ckpt_path
            }
            wandb.config.update(hyperparams)

        for it in range(load_it, n_iters):
            prior_traj = False
            rb_traj = False
            rand_n = np.random.uniform()
            # No replay buffer for first 10 iters

            if rand_n < prior_sample_prob:
                prior_traj = True
            elif (it - load_it) > 5 and rand_n < prior_sample_prob + replay_buffer_prob:
                rb_traj = True

            self.beta = self.get_beta(it, anneal, anneal_steps)

            if rb_traj:
                num_it = 1  # 20
            else:
                num_it = 1

            for j in range(num_it):
                print("Repeat vals: ", j)
                optimizer.zero_grad()
                batch_logs = self.batched_rtb(
                    shape=shape,
                    prior_sample=prior_traj,
                    rb_sample=rb_traj
                )
                loss, logr = batch_logs['loss'], batch_logs['logr']

                # self.running_dist = torch.FloatTensor(self.running_dist[-(len(self.prior_model.batch_arr)-len(logr)):].tolist() + logr.detach().cpu().tolist())
                self.running_dist = logr.detach().cpu()

                if clip > 0:
                    v_g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                    gnorm = v_g.item()
                else:
                    gnorm = -1

                optimizer.step()

                # trainable reward classifier
                if self.langevin:
                    x_1, logr_x_prime = self.replay_buffer.sample(shape[0])
                    self.update_trainable_reward(x_1)

                if self.wandb_track:

                    # compute distribution change
                    if self.prior_model.target_dist is None:
                        print("data energy distribution has yet to be computed. Computing...")
                        # save all the frames from the actual data
                        self.prior_model.fix_and_save_pdbs(torch.FloatTensor(self.prior_model.batch_arr))
                        # save all the frames from the actual data
                        self.prior_model.target_dist = self.reward_model(self.prior_model.peptide,
                                                                         data_path=self.config.data_path,
                                                                         tmp_dir=self.prior_model.out_dir)
                        print("Done!")

                    logs = {"loss": loss.mean().item(),
                            "logZ": self.logZ.detach().cpu().numpy(),
                            "log_r": logr.mean().item(),
                            "pf_divergence": (batch_logs['logpf_posterior'] - batch_logs['logpf_prior']).sum().detach().cpu().numpy(),
                            "gnorm": gnorm,
                            "epoch": it}

                    if not it % 20 == 0:
                        logs.update(compare_distributions(
                            self.prior_model.target_dist['log_r'].detach().cpu(),
                            self.running_dist.detach().cpu())
                        )
                        logs.update(plot_relative_distance_distributions(
                            xyz=batch_logs['x'],
                            n_plots=4,  # Show 4 comparison columns
                            target_dist=self.prior_model.target_dist['x']
                        ))

                    wandb.log(logs)

                    plt.close('all')
                    # save model and optimizer state
                    self.save_checkpoint(self.model, optimizer, it, run_name)

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def get_langevin_correction(self, x):
        # add gradient wrt x of trainable reward to model
        if self.langevin:
            with torch.set_grad_enabled(True):
                x.requires_grad = True
                log_rx = self.trainable_reward(x)
                grad_log_rx = torch.autograd.grad(log_rx.sum(), x, create_graph=True)[0]

            lp_correction = grad_log_rx  # .detach()
        else:
            lp_correction = torch.zeros_like(x)
        return lp_correction.detach()

    def forward(
            self,
            shape,
            steps,
            condition: list = [],
            likelihood_score_fn=None,
            guidance_factor=0.,
            detach_freq=0.0,
            backward=False,
            x_1=None,
            save_traj=False,
            prior_sample=False,
            time_discretisation='uniform'  # uniform/random,
    ):
        """
        An Euler-Maruyama integration of the model SDE with GFN for RTB

        shape: Shape of the tensor to sample (including batch size)
        steps: Number of Euler-Maruyam steps to perform
        likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
        guidance_factor: Multiplicative factor for the likelihood drift
        detach_freq: Fraction of steps on which not to train
        """
        # if not isinstance(condition, (list, tuple)):
        #     raise ValueError(f"condition must be a list or tuple or torch.Tensor, received {type(condition)}")
        B, *D = shape
        sampling_from = "prior" if prior_sample  is None else "posterior"

        if backward:
            x = x_1.to(self.device)
            t = torch.zeros(B).to(self.device) + self.sde.epsilon
        else:
            x = self.sde.prior(D).sample([B]).to(self.device)
            t = torch.ones(B).to(self.device) * self.sde.T

        # assume x is gaussian noise
        normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
                                                 torch.ones((B,) + tuple(D), device=self.device))

        logpf_posterior = 0 * normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        logpb = 0 * normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(
            self.device)  # torch.zeros_like(logpf_posterior)
        dt = -1 / (steps + 1)

        #####
        if save_traj:
            traj = [x.clone()]

        for step, _ in enumerate((pbar := tqdm(range(steps)))):
            pbar.set_description(
                f"Sampling from the {sampling_from} | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                f"| scale ~ {x.std().item():.1e}")
            if backward:
                g = self.sde.diffusion(t, x)
                std = g * (np.abs(dt)) ** (1 / 2)
                x_prev = x.detach()
                x = (x - self.sde.drift(t, x) * dt) + (std * torch.randn_like(x))
            else:
                x_prev = x.detach()

            t += dt * (-1.0 if backward else 1.0)
            if t[0] < self.sde.epsilon:  # Accounts for numerical error in the way we discretize t.
                continue  # continue instead of break because it works for forward and backward

            g = self.sde.diffusion(t, x)

            lp_correction = self.get_langevin_correction(x)

            model_out = self.model(x, t, **self.get_cond_args())
            posterior_drift = -self.sde.drift(t, x) - (g ** 2) * (
                        model_out + lp_correction
            ) / self.sde.sigma(t).view(-1, *[1] * len(D))

            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = x + posterior_drift * dt  # * (-1.0 if backward else 1.0)
            std = g * (np.abs(dt)) ** (1 / 2)

            # compute step
            if prior_sample and not backward:
                x = x - self.sde.drift(t, x) * dt + std * torch.randn_like(x)
            elif not backward:
                x = x_mean_posterior + std * torch.randn_like(x)
            x = x.detach()

            # compute parameters for pb
            # t_next = t + dt
            # pb_drift = self.sde.drift(t_next, x)
            # x_mean_pb = x + pb_drift * (-dt)
            if backward:
                pb_drift = -self.sde.drift(t, x)
                x_mean_pb = x + pb_drift * (dt)
            else:
                pb_drift = -self.sde.drift(t, x_prev)
                x_mean_pb = x_prev + pb_drift * (dt)
            # x_mean_pb = x_prev + pb_drift * (dt)
            pb_std = g * (np.abs(dt)) ** (1 / 2)

            if save_traj:
                traj.append(x.clone())

            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
            pb_dist = torch.distributions.Normal(x_mean_pb, pb_std)

            # compute log-likelihoods of reached pos wrt to prior & posterior models
            # logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            if backward:
                logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                logpf_posterior += pf_post_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            else:
                logpb += pb_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))
                logpf_posterior += pf_post_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break
        if backward:
            traj = list(reversed(traj))
        logs = {
            'x_mean_posterior': x,  # ,x_mean_posterior,
            'logpf_prior': logpb,
            'logpf_posterior': logpf_posterior,
            'traj': traj if save_traj else None
        }

        return logs

    def forward_ref_proc(
            self,
            shape,
            steps,
            condition: list = [],
            likelihood_score_fn=None,
            guidance_factor=0.,
            detach_freq=0.0,
            backward=False,
            x_1=None,
            save_traj=False,
            prior_sample=False,
            time_discretisation='uniform'  # uniform/random
    ):
        """
        An Euler-Maruyama integration of the model SDE with GFN for RTB, with the prior process being given by "ref_model" (assumed identical to GFN model paramterization)

        shape: Shape of the tensor to sample (including batch size)
        steps: Number of Euler-Maruyam steps to perform
        likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
        guidance_factor: Multiplicative factor for the likelihood drift
        detach_freq: Fraction of steps on which not to train
        """
        print("USING REF_PROC")

        # if not isinstance(condition, (list, tuple)):
        #     raise ValueError(f"condition must be a list or tuple or torch.Tensor, received {type(condition)}")
        B, *D = shape
        sampling_from = "prior" if prior_sample is None else "posterior"

        if backward:
            x = x_1
            t = torch.zeros(B).to(self.device) + self.sde.epsilon
        else:
            x = self.sde.prior(D).sample([B]).to(self.device)
            t = torch.ones(B).to(self.device) * self.sde.T

        # assume x is gaussian noise
        normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
                                                 torch.ones((B,) + tuple(D), device=self.device))

        logpf_posterior = 0 * normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        logpb = 0 * normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(
            self.device)  # torch.zeros_like(logpf_posterior)
        dt = -1 / (steps + 1)

        #####
        if save_traj:
            traj = [x.clone()]

        for step, _ in enumerate((pbar := tqdm(range(steps)))):
            pbar.set_description(
                f"Sampling from the {sampling_from} | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                f"| scale ~ {x.std().item():.1e}")
            if backward:
                g = self.sde.diffusion(t, x)
                std = g * (np.abs(dt)) ** (1 / 2)
                x_prev = x.detach()
                x = (x - self.sde.drift(t, x) * dt) + (std * torch.randn_like(x))
            else:
                x_prev = x.detach()

            t += dt * (-1.0 if backward else 1.0)
            if t[0] < self.sde.epsilon:  # Accounts for numerical error in the way we discretize t.
                continue  # continue instead of break because it works for forward and backward

            g = self.sde.diffusion(t, x)

            # lp_correction = self.get_langevin_correction(x)
            model_out = self.model(x, t, **self.get_cond_args())
            posterior_drift = -self.sde.drift(t, x) - (g ** 2) * model_out / self.sde.sigma(t).view(-1, *[1] * len(D))

            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = x + posterior_drift * dt  # * (-1.0 if backward else 1.0)
            std = g * (np.abs(dt)) ** (1 / 2)

            # compute step
            if prior_sample and not backward:
                x = x - self.sde.drift(t, x) * dt + std * torch.randn_like(x)
            elif not backward:
                x = x_mean_posterior + std * torch.randn_like(x)
            x = x.detach()

            # compute parameters for pb
            # t_next = t + dt
            # pb_drift = model drift(t_next, x)
            # x_mean_pb = x + pb_drift * (-dt)

            if backward:
                ref_drift = -self.sde.drift(t, x) - (g ** 2) * self.ref_model(x, t, **self.get_cond_args()) / self.sde.sigma(
                    t).view(-1, *[1] * len(D))
                pb_drift = ref_drift  # -self.sde.drift(t, x)
                x_mean_pb = x + pb_drift * (dt)
            else:
                ref_drift = -self.sde.drift(t, x_prev) - (g ** 2) * self.ref_model(x, t, **self.get_cond_args()) / self.sde.sigma(t).view(-1, *[1] * len(D))
                pb_drift = ref_drift  # -self.sde.drift(t, x_prev)
                x_mean_pb = x_prev + pb_drift * (dt)
            # x_mean_pb = x_prev + pb_drift * (dt)
            pb_std = g * (np.abs(dt)) ** (1 / 2)

            if save_traj:
                traj.append(x.clone())

            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
            pb_dist = torch.distributions.Normal(x_mean_pb, pb_std)

            # compute log-likelihoods of reached pos wrt to prior & posterior models
            # logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            if backward:
                logpb += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                logpf_posterior += pf_post_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            else:
                logpb += pb_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))
                logpf_posterior += pf_post_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break
        if backward:
            traj = list(reversed(traj))
        logs = {
            'x_mean_posterior': x,  # ,x_mean_posterior,
            'logpf_prior': logpb,
            'logpf_posterior': logpf_posterior,
            'traj': traj if save_traj else None
        }

        return logs

    def forward_ddpm(
            self,
            shape,
            steps,
            condition: list = [],
            guidance_factor=0.,
            detach_freq=0.0,
            backward=False,
            x_1=None,
            save_traj=False,
            prior_sample=False,
            time_discretisation='uniform'  # uniform/random
    ):
        """
        DDPM Update for SDE

        shape: Shape of the tensor to sample (including batch size)
        steps: Number of Euler-Maruyam steps to perform
        likelihood_score_fn: Add an additional drift to the sampling for posterior sampling. Must have the signature f(t, x)
        guidance_factor: Multiplicative factor for the likelihood drift
        detach_freq: Fraction of steps on which not to train
        """
        # if not isinstance(condition, (list, tuple)):
        #     raise ValueError(f"condition must be a list or tuple or torch.Tensor, received {type(condition)}")
        B, *D = shape
        sampling_from = "prior" if prior_sample else "posterior"

        if backward:
            x = x_1
            # timesteps = np.flip(timesteps)
            t = torch.zeros(B).to(self.device) + self.sde.epsilon
        else:
            x = self.sde.prior(D).sample([B]).to(self.device)
            t = torch.ones(B).to(self.device) * self.sde.T

        # assume x is gaussian noise
        normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
                                                 torch.ones((B,) + tuple(D), device=self.device))

        logpf_posterior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        logpf_prior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        dt = -1 / (steps + 1)

        #####
        if save_traj:
            traj = [x.clone()]

        for step, _ in enumerate((pbar := tqdm(range(steps)))):
            pbar.set_description(
                f"Sampling from the {sampling_from} | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                f"| scale ~ {x.std().item():.1e}")

            if backward:
                g = self.sde.diffusion(t, x, np.abs(dt))
                std = g * (np.abs(dt)) ** (1 / 2)
                x_prev = x.detach()
                x = x - self.sde.drift(t, x, np.abs(dt)) + std * torch.randn_like(x)
            else:
                x_prev = x.detach()

            t += dt * (-1.0 if backward else 1.0)
            if t[0] < self.sde.epsilon:  # Accounts for numerical error in the way we discretize t.
                continue  # continue instead of break because it works for forward and backward

            std = self.sde.diffusion(t, x, np.abs(dt))

            lp_correction = self.get_langevin_correction(x)
            posterior_drift = self.sde.drift(t, x, np.abs(dt)) + self.model(x, t, **self.get_cond_args()) + lp_correction  # / self.sde.sigma(t).view(-1, *[1]*len(D))
            f_posterior = posterior_drift
            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = x + f_posterior
            # std = g * (np.abs(dt)) ** (1 / 2)

            # compute step
            if prior_sample and not backward:
                x = x + self.sde.drift(t, x, np.abs(dt)) + std * torch.randn_like(x)
            elif not backward:
                x = x_mean_posterior + std * torch.randn_like(x)
            x = x.detach()

            # compute parameters for pb
            # t_next = t + dt
            # pb_drift = self.sde.drift(t_next, x)
            # x_mean_prior = x + pb_drift * (-dt)
            if backward:
                pb_drift = self.sde.drift(t, x, np.abs(dt))
            else:
                pb_drift = self.sde.drift(t, x_prev, np.abs(dt))
            x_mean_prior = x_prev + pb_drift
            pb_std = self.sde.diffusion(t, x_prev, np.abs(dt))  # g * (np.abs(dt)) ** (1 / 2)

            if save_traj:
                traj.append(x.clone())

            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)
            pf_prior_dist = torch.distributions.Normal(x_mean_prior, pb_std)

            # compute log-likelihoods of reached pos wrt to prior & posterior models
            # logpf_prior += pb_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            if backward:
                logpf_prior += pf_prior_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                logpf_posterior += pf_post_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
            else:
                logpf_posterior += pf_post_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))
                if self.tb:
                    logpf_prior += pf_prior_dist.log_prob(x_prev).sum(tuple(range(1, len(x.shape))))
                else:
                    logpf_prior += pf_prior_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break

        if backward:
            traj = list(reversed(traj))
        logs = {
            'x_mean_posterior': x,  # ,x_mean_posterior,
            'logpf_prior': logpf_prior,
            'logpf_posterior': logpf_posterior,
            'traj': traj if save_traj else None
        }

        return logs

    def batched_forward(
            self,
            shape,
            traj,
            correction,
            batch_size=64,
            condition: list = [],
            likelihood_score_fn=None,
            guidance_factor=0.,
            detach_freq=0.0,
            backward=False,
            rb=False
    ):
        """
        Batched implementation of self.forward. See self.forward for details.
        """

        B, *D = shape
        # compute batch size wrt traj (since each node in the traj is already a batch)
        traj_batch = batch_size // traj[0].shape[0]

        steps = len(traj) - 1
        timesteps = np.linspace(1, 0, steps + 1)

        if likelihood_score_fn is None:
            likelihood_score_fn = lambda t, x: 0.

        if backward:
            x = traj[-1]
            timesteps = np.flip(timesteps)
        else:
            x = traj[0].to(self.device)

        no_grad_steps = random.sample(range(steps),
                                      int(steps * 0.0))  # Sample detach_freq fraction of timesteps for no grad

        # # assume x is gaussian noise
        # normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
        #                                          torch.ones((B,) + tuple(D), device=self.device))
        #
        # logpf_posterior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        # logpf_prior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        # we iterate through the traj
        steps = list(range(len(traj)))
        steps = [step for step in steps[:-1] if step not in no_grad_steps]

        for i, batch_steps in enumerate(create_batches(steps, traj_batch)):

            # pbar.set_description(f"Sampling from the posterior | batch = {i}/{int(len(steps)//batch_size)} - {i*100/len(steps)//batch_size:.2f}%")

            dt = timesteps[np.array(batch_steps) + 1] - timesteps[batch_steps]

            t_ = []
            xs = []
            xs_next = []
            dts = []
            bs = 0  # this might be different than traj_batch
            for step in batch_steps:
                if timesteps[step + 1] < self.sde.epsilon:
                    continue
                t_.append(torch.full((x.shape[0],), timesteps[step + 1]).float())
                dts.append(torch.full((x.shape[0],), timesteps[step + 1] - timesteps[step]).float())
                xs.append(traj[step])
                xs_next.append(traj[step + 1])
                bs += 1

            if len(t_) == 0:
                continue

            t_ = torch.cat(t_, dim=0).to(self.device).view(-1, 1, 1, 1)
            dts = torch.cat(dts, dim=0).to(self.device).view(-1, 1, 1, 1)
            xs = torch.cat(xs, dim=0).to(self.device)
            xs_next = torch.cat(xs_next, dim=0).to(self.device)

            g = self.sde.diffusion(t_, xs).to(self.device)

            lp_correction = self.get_langevin_correction(xs)
            f_posterior = -self.sde.drift(t_, xs) - g ** 2 * (
                        self.model(xs, t_[:, 0, 0, 0], **self.get_cond_args()) + lp_correction) / \
                          self.sde.sigma(t_).view(-1,
                                                  *[1] * len(D))

            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = xs + f_posterior * dts
            std = g * (-dts) ** (1 / 2)

            # compute step
            if backward:
                # retrieve original variance noise & compute step back
                variance_noise = (self.sde.drift(t_, xs) * dt) / std
                xs = (xs + self.sde.drift(t_, xs) * dt) + (std * variance_noise)
            else:
                # retrieve original variance noise & compute step fwd
                variance_noise = (xs_next - x_mean_posterior) / std
                xs = x_mean_posterior + std * variance_noise

            xs = xs.detach()

            # define distributions wrt posterior score model
            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)

            # compute log-likelihoods of reached pos wrt to posterior model
            logpf_posterior = pf_post_dist.log_prob(xs).sum(tuple(range(1, len(xs.shape))))

            # compute loss for posterior & accumulate gradients.
            partial_rtb = ((logpf_posterior + self.logZ) * correction.repeat(bs)).mean()
            partial_rtb.backward()

            # print("In rtb backward logpf poserior + logZ * loss: ", (logpf_posterior + self.logZ) * correction.repeat(bs))

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break

        return True

    def batched_forward_ddpm(
            self,
            shape,
            traj,
            correction,
            batch_size=64,
            condition: list = [],
            likelihood_score_fn=None,
            guidance_factor=0.,
            detach_freq=0.0,
            backward=False,
    ):
        """
        Batched implementation of self.forward. See self.forward for details.
        """

        B, *D = shape
        # compute batch size wrt traj (since each node in the traj is already a batch)
        traj_batch = batch_size // traj[0].shape[0]

        steps = len(traj) - 1
        timesteps = np.linspace(1, 0, steps + 1)

        if likelihood_score_fn is None:
            likelihood_score_fn = lambda t, x: 0.

        x = traj[0].to(self.device)

        no_grad_steps = random.sample(range(steps),
                                      int(steps * 0.0))  # Sample detach_freq fraction of timesteps for no grad

        # # assume x is gaussian noise
        # normal_dist = torch.distributions.Normal(torch.zeros((B,) + tuple(D), device=self.device),
        #                                          torch.ones((B,) + tuple(D), device=self.device))
        #
        # logpf_posterior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        # logpf_prior = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        # we iterate through the traj
        steps = list(range(len(traj)))
        steps = [step for step in steps[:-1] if step not in no_grad_steps]

        for i, batch_steps in enumerate(create_batches(steps, traj_batch)):

            # pbar.set_description(f"Sampling from the posterior | batch = {i}/{int(len(steps)//batch_size)} - {i*100/len(steps)//batch_size:.2f}%")

            dt = timesteps[np.array(batch_steps) + 1] - timesteps[batch_steps]

            t_ = []
            xs = []
            xs_next = []
            dts = []
            bs = 0  # this might be different than traj_batch
            for step in batch_steps:
                if timesteps[step + 1] < self.sde.epsilon:
                    continue
                t_.append(torch.full((x.shape[0],), timesteps[step + 1]).float())
                dts.append(torch.full((x.shape[0],), timesteps[step + 1] - timesteps[step]).float())
                xs.append(traj[step])
                xs_next.append(traj[step + 1])
                bs += 1

            if len(t_) == 0:
                continue

            t_ = torch.cat(t_, dim=0).to(self.device).view(-1, 1, 1, 1)
            dts = torch.cat(dts, dim=0).to(self.device).view(-1, 1, 1, 1)
            xs = torch.cat(xs, dim=0).to(self.device)
            xs_next = torch.cat(xs_next, dim=0).to(self.device)

            std = self.sde.diffusion(t_, xs, -dts).to(self.device)

            lp_correction = self.get_langevin_correction(xs)
            f_posterior = self.sde.drift(t_, xs, -dts) + self.model(xs, t_[:, 0, 0, 0], **self.get_cond_args()) + lp_correction  # / self.sde.sigma(t_).view(-1, *[1]*len(D))

            # compute parameters for denoising step (wrt posterior)
            x_mean_posterior = xs + f_posterior
            std = std  # g * (-dts) ** (1 / 2)

            # retrieve original variance noise & compute step fwd
            variance_noise = (xs_next - x_mean_posterior) / std
            xs = x_mean_posterior + std * variance_noise

            xs = xs.detach()

            # define distributions wrt posterior score model
            pf_post_dist = torch.distributions.Normal(x_mean_posterior, std)

            # compute log-likelihoods of reached pos wrt to posterior model
            logpf_posterior = pf_post_dist.log_prob(xs).sum(tuple(range(1, len(xs.shape))))

            # compute loss for posterior & accumulate gradients.
            partial_rtb = ((logpf_posterior + self.logZ) * correction.repeat(bs)).mean()
            partial_rtb.backward()

            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break

        return True
