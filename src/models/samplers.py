import copy
import os
import lpips
from typing import Optional, Union

import diffusers
from diffusers import DDIMPipeline, LDMPipeline, ScoreSdeVeScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from peft import PeftConfig, PeftModel, load_peft_weights, set_peft_model_state_dict
from tqdm import tqdm
from diffusers.models.unets.unet_2d import UNet2DOutput
from huggingface_hub import create_repo, upload_folder, login

from functools import partial
from rtb_utils.pytorch_utils import NoContext, print_gpu_memory, create_batches, check_gradients, cycle
from torch.cuda.amp import autocast

import torch
import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import huggingface_hub as hb
import math
import wandb
import random

from rtb_utils.pytorch_utils import maybe_detach

logtwopi = np.log(2 * 3.14159265358979)


def identity(t, *args, **kwargs):
    return t


class HGFNode(nn.Module):
    """
    Class handling gfn node specific operations
    """

    langevin = False
    dps_drift = None

    def __init__(
            self,
            policy_model,
            x_dim,
            config,
            drift_model=None,
            ddim=False,
            sampling_step=1.,
            variance_type='fixed_small',
            train=False,
            clip=True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.config = config

        self.device = self.config.device
        self.x_dim = x_dim
        self.logvarrange = 0
        self.maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip else identity
        self.ddim = ddim
        self.sampling_step = sampling_step
        self.training = train
        self.variance_type = variance_type

        self.policy = policy_model
        if not self.config.checkpointing and not isinstance(self.policy.unet, nn.DataParallel):
            self.policy.unet = nn.DataParallel(self.policy.unet).to(self.device)
        else:
            self.policy.unet = self.policy.unet.to(self.device)
        self.drift_model = nn.DataParallel(drift_model).to(self.device) if drift_model is not None and not isinstance(drift_model, nn.DataParallel) else None

        # freeze
        if not train or drift_model is not None:
            for p in self.policy.unet.parameters():
                p.requires_grad = False
            self.policy.unet.eval()
        if not train and self.drift_model is not None:
            for p in self.drift_model.parameters():
                p.requires_grad = False
            self.drift_model.eval()

        self.checkpointing = self.config.checkpointing

    def get_unet_parameters(self, named=False):
        if named:
            return self.policy.unet.named_parameters()
        else:
            return self.policy.unet.parameters()

    def add_langevin(
            self,
            lgv_model=None,
            log_reward=None,
            lgv_clip=1e2,
            lgv_clipping=True,
            peptide='',
    ):

        self.langevin = True
        self.lgv_model = nn.DataParallel(lgv_model).to(self.device) if not isinstance(lgv_model, nn.DataParallel) else lgv_model
        self.log_reward = log_reward
        self.lgv_clip = lgv_clip
        self.lgv_clipping = lgv_clipping
        self.peptide = peptide

        if not self.training:
            for l in self.lgv_model.parameters():
                l.requires_grad = False
            self.lgv_model.eval()

    def run_policy(self, x, t, detach=False, condition=None):
        """
        in place policy run, modifies self.pfs and self.pflogvar to be used in later operations
        @param x:
        @param t:
        @return:
        """
        # m0 = T.cuda.memory_allocated()

        context = torch.no_grad() if not self.training or self.drift_model is not None or detach else NoContext()
        with context:
            if len(t.shape) == 0:
                t_ = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            else:
                t_ = t.to(torch.long).to(x.device)

            if self.checkpointing:
                res = torch.utils.checkpoint.checkpoint(self.policy.unet, x, t_, **condition, use_reentrant=False)
            else:
                res = self.policy.unet(x, t_, **condition)

            if isinstance(res, torch.Tensor):
                model_output = res
            elif isinstance(res, UNet2DOutput):
                model_output = res.sample
            else:
                raise TypeError(f"The type '{type(res)}' for diffusion model output is not supported")

        context = torch.no_grad() if not self.training or detach else NoContext()
        with context:
            if self.drift_model is not None:
                res = self.drift_model(x, t_)
                if isinstance(res, tuple):
                    mean_drift, _ = res
                else:
                    mean_drift = res
                self.pf_mean += mean_drift

            langevin_correction = 0
            if self.langevin:
                scale = self.lgv_model(x, t)
                x.requires_grad_(True)
                with torch.enable_grad():
                    grad_log_r = torch.autograd.grad(self.log_reward(x).log_softmax(1)[:, self.peptide].sum(), x)[
                        0].detach()
                    grad_log_r = torch.nan_to_num(grad_log_r)
                    if self.lgv_clipping:
                        grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

                langevin_correction = scale * grad_log_r

        return model_output, langevin_correction

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(
            self,
            x,
            t,
            t_next=None,
            ddim_eta=None,
            noise=None,
            target=None,
            backward=False,
            x_0=None,
            clip_output=False,
            condition=None,
            detach=False,
            *args,
            **kwargs,
    ):

        # --- FORWARD MOTION: compute policy then move to x_{t+1} ----

        # get prob. distribution for next step -- in place, modifies self.pf_mean and self.pflogvar
        model_output, langevin_correction = self.run_policy(x, t, detach, condition)

        results = self.policy.scheduler.step(
            model_output, t, x,
            eta=ddim_eta,
            use_clipped_model_output=clip_output,
            langevin_correction=langevin_correction,
            noise=self.noise if backward else noise,
            target=target,
        )
        self.posterior_mean = results.posterior_mean.to(self.device)
        self.posterior_std = results.posterior_std.to(self.device)
        self.noise = results.noise.to(self.device) if results.noise is not None else results.noise

        return results.prev_sample

    def get_logpf(self, x, mean=None, std=None):
        """
        @param x: prev x state
        @param new_x: new x state
        @param pf_mean: if pfs is specified, then it overrides the saved self.pfs
        @param pflogvars: if pflogvars is specified, then it overrides the saved self.pflogvars
        @return: logpf of last call to "forward method"
        """

        mean = mean if mean is not None else self.posterior_mean
        std = std if std is not None else self.posterior_std

        pf_dist = torch.distributions.Normal(mean, std)
        return pf_dist.log_prob(x).sum(tuple(range(1, len(x.shape))))

    def get_logpb(self, t, denoised_x=None, delta_x=None, denoised=False):
        # todo
        pass


class PosteriorPriorDGFN(nn.Module):
    """ Version of posterior-prior dgfn to work with hugging face native library"""
    def __init__(
            self,
            dim,
            outsourced_prior_policy,
            outsourced_posterior_policy,
            prior_model,
            config,
            ddim_sampling_eta=0.,
            mixed_precision=False,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.config = config

        self.use_cuda = self.config.use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.mixed_precision = mixed_precision

        if self.mixed_precision:
            self.context = autocast()
        else:
            self.context = NoContext()

        self.detach_cut_off = config.detach_cut_off
        self.dim = dim
        self.traj_length = self.config.traj_length
        self.sampling_length = self.config.sampling_length
        self.sampling_step = self.config.traj_length / self.config.sampling_length
        self.ddim = self.sampling_step > 1.
        self.ddim_sampling_eta = ddim_sampling_eta  # 0-> DDIM, 1-> DDPM

        # ----------------------------------------------
        prior_node = HGFNode(config=config,
                             policy_model=outsourced_prior_policy,
                             x_dim=dim,
                             sampling_step=self.sampling_step,
                             ddim=self.ddim,
                             clip=outsourced_prior_policy.scheduler.config.clip_sample,
                             train=False)
        self.register_module('prior_node', prior_node)

        posterior_node = HGFNode(config=config,
                                 policy_model=outsourced_posterior_policy,
                                 x_dim=dim,
                                 clip=outsourced_posterior_policy.scheduler.config.clip_sample,
                                 sampling_step=self.sampling_step,
                                 train=True)

        self.prior_model = prior_model

        self.register_module('posterior_node', posterior_node)

        self.print_parameters()

        self.vargrad = self.config.vargrad
        self.logZ = T.nn.Parameter(T.tensor(0.).to(self.device))
        if not self.vargrad:
            self.logZ.requires_grad = True

        self.lora = self.config.lora
        self.push_to_hf = self.config.push_to_hf
        self.exp_name = self.config.exp_name

        if self.push_to_hf:
            hf_token = os.getenv('HF_TOKEN', None)
            if hf_token is None:
                print("No HuggingFace token was set in 'HF_TOKEN' env. variable. "
                      "Setting push_to_hf to false.")
                self.push_to_hf = False
            else:
                print("HF login succesfull!")
                login(token=hf_token)

                self.hub_model_id = f"{hb.whoami()['name']}/{self.exp_name}"  # xkronosx
                self.repo_id = create_repo(
                    repo_id=self.hub_model_id, exist_ok=True
                ).repo_id

    def print_parameters(self):

        prior_params = sum(p.numel() for p in self.prior_node.get_unet_parameters())
        posterior_params = sum(p.numel() for p in self.posterior_node.get_unet_parameters())
        trainable_posterior_params = sum(p.numel() for p in self.posterior_node.get_unet_parameters() if p.requires_grad)
        print(f"\nTotal params: "
              f"\nPRIOR model: {prior_params / 1e6:.2f}M "
              f"\nPOSTERIOR model: {posterior_params / 1e6:.2f}M")
        if self.posterior_node.drift_model is None:
              print(f"Trainable posterior parameters: {trainable_posterior_params / 1e6:.2f}M/{posterior_params / 1e6:.2f}M  ({trainable_posterior_params*100/posterior_params:.2f}%)\n")
        else:
            drift_params = sum(p.numel() for p in self.posterior_node.drift_model.parameters())
            trainable_drift_params = sum(p.numel() for p in self.posterior_node.drift_model.parameters() if p.requires_grad)
            print(f"Trainable drift parameters: {(trainable_drift_params + trainable_drift_params)/ 1e6:.2f}M/{(drift_params + posterior_params) / 1e6:.2f}M\n")

    def to(self, device):
        self.device = device
        self.posterior_node.to(device)
        self.prior_node.to(device)
        return super().to(device)

    def train(self: T, mode: bool = True) -> T:
        super().train()
        self.posterior_node.policy.unet.train()
        self.prior_node.policy.unet.eval()

    def eval(self: T, mode: bool = True) -> T:
        super().eval()
        self.posterior_node.policy.unet.eval()
        self.prior_node.policy.unet.eval()

    def set_loader(self, dataloader):
        self.dataloader = cycle(dataloader)

    def add_classifier(self, reward_function):
        self.reward_function = reward_function

    def get_scheduler(self):
        return self.posterior_node.policy.scheduler

    def get_schedule_args(self):
        return {
            'ddim_eta': self.ddim_sampling_eta,
            'clip_output': self.prior_node.policy.scheduler.config.clip_sample if 'clip_sample' in self.prior_node.policy.scheduler.config else False
        }

    def add_langevin(self, *args, **kwargs):
        self.prior_node.add_langevin(*args, **kwargs)
        self.posterior_node.add_langevin(*args, **kwargs)

    def forward(self, xs=None, back_and_forth=False, *args, **kwargs):
        with self.context:
            if back_and_forth:
                return self.sample_back_and_forth(*args, **kwargs)
            elif xs is not None:
                # if a denoised image is given then assume we want backward trajectories
                return self.sample_bkw(xs, *args, **kwargs)
            else:
                # otherwise do forward trajectories
                return self.sample_fwd(*args, **kwargs)

    def sample_fwd(
            self,
            batch_size=None,
            x_start=None,
            save_traj=False,
            sample_from_prior_only=False,
            sample_from_prior=False,
            detach_freq=0.,
            detach_cut_off=None,
            sampling_length=None,
            condition=None,
            *args,
            **kwargs
    ):

        assert batch_size is not None, "provide batch_size for sample_fwd"
        if sample_from_prior_only:
            sample_from_prior = True

        sampling_length = sampling_length if sampling_length is not None else self.sampling_length
        detach_cut_off = detach_cut_off if detach_cut_off is not None else self.detach_cut_off

        return_dict = {}

        normal_dist = torch.distributions.Normal(torch.zeros((batch_size,) + tuple(self.dim), device=self.device),
                                                 torch.ones((batch_size,) + tuple(self.dim), device=self.device))

        x = normal_dist.sample() if x_start is None else x_start
        if self.mixed_precision and 'cuda' in self.device:
            x = x.half()

        x_start = x.clone()
        return_dict['logpf_posterior'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        return_dict['logpf_prior'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        return_dict['logpb'] = 0 * normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        self.posterior_node.policy.scheduler.set_timesteps(sampling_length)
        self.prior_node.policy.scheduler.set_timesteps(sampling_length)
        scheduler = copy.deepcopy(self.posterior_node.policy.scheduler)

        sampling_times = scheduler.timesteps

        times_to_detach = np.random.choice([t.item() for t in sampling_times], int(sampling_length * detach_freq), replace=False).tolist()
        times_to_detach += sampling_times[sampling_times > detach_cut_off*scheduler.config.num_train_timesteps].tolist()
        times_to_detach = set(times_to_detach)

        if save_traj:
            traj = [x.clone()]

        for i, t in tqdm(enumerate(sampling_times), total=len(sampling_times)):

            t_specific_args = {
                'noise': None if t > 0. else 0.,
                'condition_noise': None if t > 0. else 0.,
                'detach': t.item() in times_to_detach,
            }

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            # extra params
            step_args['condition'] = condition

            # -- make step in x by prior model -- (updates internal values of mean and std for prior node)
            new_x = self.prior_node(x, t, **step_args).detach()

            if not sample_from_prior_only:

                # ------ compute prior pf for posterior step --------
                step_args['noise'] = self.prior_node.noise if t > 0. else 0.  # adjust noise to match prior
                step_args['condition'] = condition
                step_args['prior_mean'] = self.prior_node.posterior_mean

                # # -- make a step in x by posterior model -- (updates internal values of mean and std for posterior node)
                posterior_new_x = self.posterior_node(x, t, **step_args)

                new_x = new_x if sample_from_prior else posterior_new_x

                # get prior pf
                return_dict['logpf_prior'] += self.prior_node.get_logpf(x=new_x)

                # get posterior pf
                return_dict['logpf_posterior'] += self.posterior_node.get_logpf(x=new_x)

            pb_mean, _, _ = scheduler.step_noise(new_x, x_start, t=t)
            return_dict['logpb'] += self.posterior_node.get_logpf(x=x.detach(), mean=pb_mean.detach())

            if save_traj:
                traj.append(new_x.clone())

            x = new_x.detach().clone()

        return_dict['x'] = x

        if save_traj:
            return_dict['traj'] = traj

        return return_dict

    def sample_bkw(
            self,
            x,
            steps=50,
            detach_freq=0.,
            sampling_length=None,
            condition=None,
            *args,
            **kwargs
    ):

        assert x is not None, "provide starting samples for backward steps"
        sampling_length = sampling_length if sampling_length is not None else self.sampling_length

        sampling_length = sampling_length if sampling_length is not None else self.sampling_length

        self.posterior_node.policy.scheduler.set_timesteps(sampling_length)
        self.prior_node.policy.scheduler.set_timesteps(sampling_length)
        scheduler = copy.deepcopy(self.posterior_node.policy.scheduler)
        sampling_times = scheduler.timesteps

        return_dict = {}
        # ---------------------------------------------------
        # --------------- Move Backward ---------------------
        # ---------------------------------------------------
        if self.mixed_precision and 'cuda' in self.device:
            x = x.half()

        return_dict['logpf_posterior'] = 0.
        return_dict['logpf_prior'] = 0.
        return_dict['logpb'] = 0.

        backward_sampling_times = list(sampling_times)[::-1][:steps]

        x_start = x[:]
        return_dict['x'] = x_start
        times_to_detach = np.random.choice([t for t in sampling_times], int(sampling_length * detach_freq), replace=False)

        for i, t in tqdm(enumerate(backward_sampling_times), total=len(backward_sampling_times)):

            t_next = t + self.traj_length // self.sampling_length
            if t_next == self.traj_length:
                t_next -= 1

            b_noise = torch.randn_like(x)
            new_x, _, _ = scheduler.step_noise(x, b_noise, t=t)

            return_dict['logpb'] += self.posterior_node.get_logpf(x=new_x.detach(), mean=new_x)
            # x = self.policy.scheduler.add_noise(x_0, self.noise, t_next)

            t_specific_args = {
                'noise': None if t > 0 else 0.,  # fix noise for backward process
                'condition_noise': None if t > 0. else 0.,
                'condition': condition,
                't_next': t_next if self.traj_length > t_next else torch.LongTensor([self.traj_length - 1])[0],
                'detach': t.item() in times_to_detach,
                'x_0': x_start,
                'backward': True,  # flag backward for backward step
            }

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            # -- make a backward step in x, compute mean and var in place by posterior model --
            self.posterior_node(new_x, t, **step_args).detach()

            # get posterior pf
            return_dict['logpf_posterior'] += maybe_detach(self.posterior_node.get_logpf(x=x.detach()), t, times_to_detach)

            # ------ compute prior pf for posterior step --------
            # update internal values of pfs and logvar for prior -- inplace
            self.prior_node(new_x, t, **step_args)

            # get prior pf, given posterior move
            return_dict['logpf_prior'] += self.prior_node.get_logpf(x=x.detach()).detach()

            # ---------------------------------------------------

            x = new_x

        normal_dist = torch.distributions.Normal(torch.zeros((len(x),) + tuple(self.dim), device=self.device),
                                                 torch.ones((len(x),) + tuple(self.dim), device=self.device))

        return_dict['logpf_posterior'] += normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        return_dict['logpf_prior'] += normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        return_dict['logpb'] += normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        return_dict['x'] = x

        return return_dict

    def sample_back_and_forth(self, batch_size=None, steps=50, detach_freq=0., sampling_length=None):

        assert self.dataloader is not None, 'please provide a batch of starting samples x'

        x = next(self.dataloader)['images'][:batch_size].to(self.device)

        sampling_length = sampling_length if sampling_length is not None else self.sampling_length

        normal_dist = torch.distributions.Normal(torch.zeros((len(x),) + tuple(self.dim), device=self.device),
                                                 torch.ones((len(x),) + tuple(self.dim), device=self.device))

        self.posterior_node.policy.scheduler.set_timesteps(sampling_length)
        self.prior_node.policy.scheduler.set_timesteps(sampling_length)
        scheduler = copy.deepcopy(self.posterior_node.policy.scheduler)
        sampling_times = scheduler.timesteps

        return_dict = {}
        # ---------------------------------------------------
        # --------------- Move Backward ---------------------
        # ---------------------------------------------------
        if self.mixed_precision and 'cuda' in self.device:
            x = x.half()

        return_dict['logpf_posterior_b'] = 0.
        return_dict['logpf_prior_b'] = 0.

        backward_sampling_times = list(sampling_times)[::-1][:steps]

        x_start = x[:]
        return_dict['x'] = x_start
        times_to_detach = np.random.choice([t for t in sampling_times], int(sampling_length * detach_freq), replace=False)
        backward_noise = torch.randn_like(x)
        for i, t in tqdm(enumerate(backward_sampling_times), total=len(backward_sampling_times)):

            t_specific_args = {
                'noise': backward_noise,  # fix noise for backward process
                't_next': t + self.traj_length // self.sampling_length,
                'x_0': x_start,
                'backward': True,  # flag backward for backward step
            }

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            # -- make a backward step in x by posterior model --
            new_x = self.posterior_node(x, t, **step_args).detach()

            # get posterior pf
            return_dict['logpf_posterior_b'] += maybe_detach(self.posterior_node.get_logpf(x=x.detach()), t, times_to_detach)

            # ------ compute prior pf for posterior step --------
            # update internal values of pfs and logvar for prior -- inplace
            self.prior_node(x, t, **step_args)

            # get prior pf, given posterior move
            return_dict['logpf_prior_b'] += maybe_detach(self.prior_node.get_logpf(x=x.detach()), t, times_to_detach)
            # ---------------------------------------------------

            x = new_x

        # ---------------------------------------------------
        # ------------------ Move Forward  ------------------
        # ---------------------------------------------------

        return_dict['logpf_posterior_f'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)
        return_dict['logpf_prior_f'] = normal_dist.log_prob(x).sum(tuple(range(1, len(x.shape)))).to(self.device)

        forward_sampling_times = backward_sampling_times[::-1]

        for i, t in tqdm(enumerate(forward_sampling_times), total=len(forward_sampling_times)):

            t_specific_args = {'noise': None if t > 0 else 0.}

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            # -- make a step in x by posterior model --
            new_x = self.posterior_node(x, t, **step_args).detach()

            # get posterior pf
            return_dict['logpf_posterior_f'] += maybe_detach(self.posterior_node.get_logpf(x=new_x), t, times_to_detach)

            # ------ compute prior pf for posterior step --------
            # update internal values of pfs and logvar for prior -- inplace
            step_args['noise'] = self.posterior_node.noise  # adjust noice to match posterior
            self.prior_node(x, t, **step_args)

            # get prior pf, given posterior move
            return_dict['logpf_prior_f'] += maybe_detach(self.prior_node.get_logpf(x=new_x), t, times_to_detach)
            # ---------------------------------------------------

            x = new_x

        return_dict['x_prime'] = x.detach()

        return return_dict

    def batched_train(self, traj, correction, batch_size=64, detach_freq=0., detach_cut_off=1.):
        detach_cut_off = detach_cut_off if detach_cut_off is not None else self.detach_cut_off

        # compute batch size wrt traj (since each node in the traj is already a batch)
        traj_batch = batch_size//traj[0].shape[0]
        sampling_length = len(traj)-1
        x1 = traj[0]  # starting noise

        self.posterior_node.policy.scheduler.set_timesteps(sampling_length)
        self.prior_node.policy.scheduler.set_timesteps(sampling_length)
        scheduler = copy.deepcopy(self.posterior_node.policy.scheduler)
        sampling_times = scheduler.timesteps

        times_to_detach = np.random.choice([t.item() for t in sampling_times], int(sampling_length * detach_freq), replace=False).tolist()
        times_to_detach += sampling_times[sampling_times < (1-detach_cut_off)*scheduler.config.num_train_timesteps].tolist()
        times_to_detach = set(times_to_detach)

        # print(f"batch learning on trajectory. effective bs is {traj_batch * traj[0].shape[0]}")

        # we iterate through the traj
        ids = list(range(len(traj)))
        ids = [id for id in ids[:-1] if sampling_times[id].item() not in times_to_detach]
        # random.shuffle(ids)   # should be able to randomize safely

        for batch_ids in tqdm(create_batches(ids, traj_batch), total=int(len(ids)//batch_size)):

            bs = len(batch_ids)  # this might be different than traj_batch on the last iteration
            xs = torch.cat([traj[id] for id in batch_ids], dim=0)

            # get next targets (this will force the variance at each step to match the previous trajectories')
            step_targets = torch.cat([traj[id+1] if id+1 < len(traj) else traj[-1] for id in batch_ids], dim=0)
            t_specific_args = {
                'target': step_targets,
                # 'detach': t.item() in times_to_detach
            }

            step_args = self.get_schedule_args()
            step_args.update(t_specific_args)

            # -- make a step in x by posterior model --
            t_ = []
            for id in batch_ids:
                t_ += [torch.full((x1.shape[0],), sampling_times[id], dtype=torch.long)]
            t_ = torch.cat(t_)

            new_x = self.posterior_node(xs, t_, **step_args).detach()

            # get posterior pf
            logpf_posterior = self.posterior_node.get_logpf(x=new_x)

            # ---------------------------------------------------

            # compute loss for posterior & accumulate gradients.
            partial_rtb = ((logpf_posterior + self.logZ) * correction.repeat(bs)).mean()
            partial_rtb.backward()
            
        return True

    def compute_prior_reward(self, samples=None, no_of_samples=100, batch_size=32, peptide=None, rewards=None, n_traj_per_sample=10, sampling_length=None, reference_img=None, metric_only=False, *args, **kwargs):

        return_dict = {}
        with torch.no_grad():
            if samples is None or rewards is None:

                its = (no_of_samples // batch_size) + 1
                samples = []
                rewards = []
                for _ in range(its):
                    # sample like usual
                    x = self.forward(batch_size=batch_size, sampling_length=self.sampling_length)['x']

                    # compute log reward
                    if isinstance(self.reward_function, (nn.Module, nn.DataParallel)):
                        logr = self.reward_function(x.float()).log_softmax(1)[:, peptide].max(-1)[0].detach().cpu().numpy().tolist()
                    else:
                        res = self.reward_function(x.float(), xs_reference=reference_img)
                        logr = res['logr'].cpu().detach().numpy().tolist()

                    samples.append(x)
                    rewards.append(logr)

                samples = torch.cat(samples)
                rewards = torch.cat([torch.tensor(rev) for rev in rewards])

            # Convert reference image to match dimensions
            if reference_img is not None:
                # If the reference image is not batched, add a batch dimension
                reference_img = reference_img.unsqueeze(0).to(self.device)

                # If reference_img has 1 channel and samples have 3 channels, repeat across channels
                if reference_img.shape[1] == 1 and samples.shape[1] == 3:
                    reference_img = reference_img.repeat(1, 3, 1, 1)
                elif reference_img.shape[1] == 3 and samples.shape[1] == 1:
                    samples = samples.repeat(1, 3, 1, 1)

                # Compute Pixel-wise Distance (Mean Squared Error)
                pixelwise_distance = F.mse_loss(samples[:no_of_samples], reference_img.repeat(min(len(samples), no_of_samples), 1, 1, 1))

                # Compute Perceptual Distance using LPIPS
                lpips_fn = lpips.LPIPS(net='vgg').to(self.device)  # You can also use 'alex' or 'squeeze' for faster comparisons
                perceptual_distance = lpips_fn(samples[:no_of_samples], reference_img.repeat(min(len(samples), no_of_samples), 1, 1, 1)).mean()

                return_dict['MSE'] = pixelwise_distance.item()
                return_dict['LPIPS'] = perceptual_distance.item()

                if metric_only:
                    return return_dict

            x = samples[:no_of_samples].repeat_interleave(n_traj_per_sample, 0).to(self.device)

            sampling_length = sampling_length if sampling_length is not None else self.sampling_length

            scheduler = copy.deepcopy(self.posterior_node.policy.scheduler)
            scheduler.set_timesteps(sampling_length)
            sampling_times = scheduler.timesteps

            res = {}
            # ---------------------------------------------------
            # --------------- Move Backward ---------------------
            # ---------------------------------------------------
            if self.mixed_precision and 'cuda' in self.device:
                x = x.half()

            res['logpb'] = torch.zeros(x.shape[0]).to(self.device)
            res['logpf_prior'] = torch.zeros(x.shape[0]).to(self.device)

            backward_sampling_times = list(sampling_times)[::-1]

            x_start = x[:]
            res['x'] = x_start
            backward_noise = torch.randn_like(x)
            for i, t in tqdm(enumerate(backward_sampling_times), total=len(backward_sampling_times)):
                t_specific_args = {
                    'noise': backward_noise,  # fix noise for backward process
                    't_next': t + self.traj_length // self.sampling_length,
                    'x_0': x_start,
                    'backward': True,  # flag backward for backward step
                }

                if t_specific_args['t_next'] >= self.traj_length: t_specific_args['t_next'] = torch.tensor([self.traj_length - 1])[0]
                step_args = self.get_schedule_args()
                step_args.update(t_specific_args)

                # ------ compute prior pf for posterior step --------
                # update internal values of pfs and logvar for prior -- inplace
                x_next = self.prior_node(x, t, **step_args)

                # get pb and prior pf, given backward step
                std = self.prior_node.posterior_std
                pb_mean = x_next - self.prior_node.posterior_std * backward_noise
                res['logpb'] += self.prior_node.get_logpf(x=x.detach(), mean=pb_mean, std=std)

                res['logpf_prior'] += self.prior_node.get_logpf(x=x.detach())

                # ---------------------------------------------------

                x = x_next

        normal_dist = torch.distributions.Normal(torch.zeros_like(x, device=self.device),
                                                 torch.ones_like(x, device=self.device))
        res['logpf_prior'] += normal_dist.log_prob(x_next).sum(tuple(range(1, len(x.shape))))

        log_prior_prob = (res['logpf_prior'] - res['logpb']).view(-1, n_traj_per_sample).logsumexp(dim=1) - np.log(n_traj_per_sample)

        return_dict['ode_logpf_prior'] = res['logpf_prior'].mean()
        return_dict['ode_logpb'] = res['logpb'].mean()
        return_dict['entropy'] = -log_prior_prob.mean()
        return_dict['log_prior_reward'] = torch.tensor(rewards[:no_of_samples]).mean() + return_dict['entropy']

        return return_dict

    def compute_metrics(self, no_of_samples=1000, batch_size=32, peptide=None, reference_img=None, reference_measurement=None,
                        loader=None, conditional=False, *args, **kwargs):

        print("Testing Metrics...")
        return_dict = {}
        context = NoContext() if isinstance(self, PosteriorPriorBaselineSampler) else torch.no_grad()
        with context:

            its = (no_of_samples // batch_size) + 1
            reference_imgs = []
            samples = []
            rewards = []
            logpf_priors = []
            logpb_posteriors = []
            for _ in tqdm(range(its), total=its):
                if loader is not None:
                    ref_imgs = []
                    ds = 0
                    while ds < batch_size:
                        ref_imgs.append(next(loader))
                        ds += ref_imgs[-1].shape[0]
                    reference_img = torch.cat(ref_imgs)[:batch_size].to(self.device)
                    # get measurement for real images
                    reference_measurement = self.reward_function(reference_img, forward_only=True)
                    reference_imgs.append(reference_img)

                # forward given condition of reference measurement
                res = self.forward(
                    measurement=reference_measurement,
                    batch_size=batch_size,
                    sampling_length=self.sampling_length,
                    peptide=peptide,
                    condition=reference_measurement if conditional else None
                )
                x = res['x'].detach()

                # compute log reward
                if isinstance(self.reward_function, (nn.Module, nn.DataParallel)):
                    logr = self.reward_function(x.float()).log_softmax(1)[:, peptide].max(-1)[0].detach().cpu().numpy().tolist()
                else:
                    rw = self.reward_function(x.float(), xs_reference_measurement=reference_measurement)
                    logr = (-rw['measurement_norm'].cpu().detach().numpy()).tolist()

                samples.append(x)
                rewards.append(logr)
                logpf_priors.append(res['logpf_prior'])
                logpb_posteriors.append(res['logpf_posterior'])

            samples = torch.cat(samples)
            rewards = torch.cat([torch.tensor(rev) for rev in rewards])

            if len(reference_imgs) > 0:
                reference_img = torch.cat(reference_imgs)

            logpf_priors = torch.cat(logpf_priors)
            logpb_posteriors = torch.cat(logpb_posteriors)

            print(f"Generated {len(samples)} samples!")

            # Convert reference image to match dimensions
            if reference_img is not None:
                ref_img = copy.deepcopy(reference_img)
                if len(ref_img.shape) == 3:
                    # If the reference image is not batched, add a batch dimension
                    ref_img = ref_img.unsqueeze(0).to(self.device)
                    ref_img = ref_img.repeat(min(len(samples), no_of_samples), 1, 1, 1)

                # If reference_img has 1 channel and samples have 3 channels, repeat across channels
                if ref_img.shape[1] == 1 and samples.shape[1] == 3:
                    ref_img = ref_img.repeat(1, 3, 1, 1)
                elif ref_img.shape[1] == 3 and samples.shape[1] == 1:
                    samples = samples.repeat(1, 3, 1, 1)

                # Compute Pixel-wise Distance (Mean Squared Error)
                pixelwise_distance = F.mse_loss(samples[:no_of_samples], ref_img[:no_of_samples])

                # Compute Perceptual Distance using LPIPS
                lpips_fn = lpips.LPIPS(net='vgg').to(self.device)  # You can also use 'alex' or 'squeeze' for faster comparisons
                perceptual_distance = lpips_fn(samples[:no_of_samples], ref_img[:no_of_samples]).mean()

                return_dict['MSE'] = pixelwise_distance.item()
                return_dict['LPIPS'] = perceptual_distance.item()

            return_dict['logr_IW'] = rewards.cpu().detach().numpy().mean()
            return_dict['measurement_norm_IW'] = - return_dict['logr_IW']
            return_dict['logZ_IW'] = torch.logsumexp(-logpb_posteriors + logpf_priors + rewards.to(logpf_priors.device), dim=0) - np.log(logpf_priors.shape[0])
            return_dict['logpf_prior_IW'] = logpf_priors.mean()
            return_dict['logpf_posterior_IW'] = logpb_posteriors.mean()
            return_dict['x'] = samples
        print("Done!")
        return return_dict

    def sample(self, batch_size=16, sample_from_prior=False):
        return self.sample_fwd(batch_size=batch_size, sample_from_prior=sample_from_prior, x_start=None)['x'].clamp(-1, 1)

    def save(self, folder, push_to_hf, opt, it=0, logZ=0., **kwargs):

        torch.save({
            "it": it,
            "optimizer_state_dict": opt.state_dict(),
            "logZ": logZ,
            **kwargs,
        }, folder + "checkpoint.tar")

        if isinstance(self.posterior_node.policy.unet, nn.DataParallel):
            model = self.posterior_node.policy.unet.module
        else:
            model = self.posterior_node.policy.unet

        if self.lora:
            model.save_pretrained(folder)

            if self.push_to_hf and push_to_hf:
                # self.posterior_node.policy.unet.module.push_to_hf(self.hub_model_id)
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=folder,
                    commit_message=f"Iteration {it}",
                    ignore_patterns=["step_*", "epoch_*", "wandb*"],
                )
        else:
            if isinstance(model, diffusers.ModelMixin):
                pipeline = DDIMPipeline(unet=model, scheduler=self.posterior_node.policy.scheduler)
                pipeline.save_pretrained(folder)
            else:
                torch.save(model.state_dict(), folder + "mdgen_source_sampler.bin")

    def load(self, folder):

        if isinstance(self.posterior_node.policy.unet, nn.DataParallel):
            model = self.posterior_node.policy.unet.module
        else:
            model = self.posterior_node.policy.unet

        if self.lora:

            # attach lora posterior
            lora_weights = load_peft_weights(folder)
            set_peft_model_state_dict(model, lora_weights)

        else:
            if isinstance(model, diffusers.models.ModelMixin):  # this is a diffusers model
                pipeline = DDIMPipeline.from_pretrained(folder)
                self.posterior_node.policy = pipeline
            else:
                # note this assumes the "scheduler" in the pipeline is compatible with the saved model
                model.load_state_dict(torch.load(folder + "mdgen_source_sampler.bin"))
                self.posterior_node.policy.unet = model


class PosteriorPriorBaselineSampler(PosteriorPriorDGFN):

    """ Version of posterior-prior dgfn to work with hugging face native library"""
    def __init__(
            self,
            scale=1.,
            mc=False,
            particles=10,
            cla=True,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.scale = scale  # scale of guidance

        self.mc = mc
        self.cla = cla
        self.particles = particles

    def forward(self, condition=None, batch_size=None, sampling_length=None, sample_from_prior=False,
                sample_from_prior_only=False, peptide=None, mc=None, particles=None, *args, **kwargs):

        if sample_from_prior_only:
            sample_from_prior = True

        self.mc = mc if mc is not None else self.mc
        self.particles = particles if particles is not None else self.particles

        config = self.prior_node.policy.unet.module.config if isinstance(self.prior_node.policy.unet, nn.DataParallel) else self.prior_node.policy.unet.config

        assert condition is not None or sample_from_prior, "a condition is required for classifier  baselines"
        assert sample_from_prior or self.reward_function is not None, "a classifier/forward function must be added before sampling from posterior"
        assert batch_size is not None, "provide batch_size for sampling"
        sampling_length = sampling_length if sampling_length is not None else self.sampling_length

        return_dict = {}

        if isinstance(config.sample_size, int):
            image_shape = (
                batch_size,
                config.in_channels,
                config.sample_size,
                config.sample_size,
            )
        else:
            image_shape = (batch_size, config.in_channels, *config.sample_size)

        image = randn_tensor(image_shape).to(self.device)  # noise
        if condition is not None:
            condition_noise = randn_tensor(condition.shape).to(self.device)

        if self.mixed_precision and 'cuda' in self.device:
            image = image.half()

        normal_dist = torch.distributions.Normal(torch.zeros((batch_size,) + tuple(self.dim), device=self.device),
                                                 torch.ones((batch_size,) + tuple(self.dim), device=self.device))

        return_dict['logpf_posterior'] = normal_dist.log_prob(image).sum(tuple(range(1, len(image.shape)))).to(self.device)
        return_dict['logpf_prior'] = normal_dist.log_prob(image).sum(tuple(range(1, len(image.shape)))).to(self.device)

        # set step values
        self.prior_node.policy.scheduler.set_timesteps(sampling_length)

        for t in self.prior_node.policy.progress_bar(self.prior_node.policy.scheduler.timesteps):
            # 1. predict noise model_output
            image = image.requires_grad_(True)
            model_output = self.prior_node.policy.unet(image, t.item()).sample

            # 2. compute previous image: x_t -> x_t-1
            res = self.prior_node.policy.scheduler.step(model_output, t, image)
            image_t_minus_1 = res.prev_sample  # x_t-1 according to prior

            if not sample_from_prior:
                x_0_hat = res.pred_original_sample

                if self.decoder is not None:
                    x_0_hat = self.decoder.decode(x_0_hat, clip=True)

                noisy_condition = self.prior_node.policy.scheduler.add_noise(condition, condition_noise, t)

                if self.mc:
                    # shapes = [self.particles] + list(x_0_hat.shape)
                    # batch_size = x_0_hat.shape[0]  # ?
                    sigma_t = res.posterior_std
                    r_t = sigma_t / torch.sqrt(1 + sigma_t ** 2)

                    differences = [
                        noisy_condition -
                        self.prior_node.policy.scheduler.add_noise(
                            self.reward_function(x_0_hat + torch.randn_like(x_0_hat) * r_t, forward_only=True, temperature=self.config.energy_temperature),
                            condition_noise,
                            t
                        )
                        for _ in range(self.particles)
                    ]
                    norms = torch.stack([torch.linalg.norm(difference) for difference in differences], dim=0)

                    mc_norm_estimates = torch.logsumexp(norms, dim=0) - math.log(float(self.particles))

                    guidance = -torch.autograd.grad(outputs=mc_norm_estimates, inputs=image)[0]
                else:
                    prediction = self.reward_function(x_0_hat, forward_only=True, temperature=self.config.energy_temperature)

                    # compute guidance diff. and corresponding norm
                    if self.cla:
                        noisy_prediction = self.prior_node.policy.scheduler.add_noise(prediction, condition_noise, t)
                        noisy_condition = self.prior_node.policy.scheduler.add_noise(condition, condition_noise, t)
                        norm = torch.linalg.norm(noisy_condition - noisy_prediction)
                    else:
                        norm = torch.linalg.norm(condition - prediction)

                    norm_grad = torch.autograd.grad(outputs=norm, inputs=image)[0]
                    guidance = -norm_grad

                # apply guidance
                new_image = (image_t_minus_1 + guidance * self.scale).detach()

                return_dict['logpf_posterior'] += self.prior_node.get_logpf(x=new_image, mean=res.posterior_mean + guidance * self.scale, std=res.posterior_std)
                return_dict['logpf_prior'] += self.prior_node.get_logpf(x=new_image, mean=res.posterior_mean, std=res.posterior_std)

            else:
                new_image = image_t_minus_1.detach()

            image = new_image

        if self.decoder is not None:
            image = self.decoder.decode(image, clip=True)

        return_dict['x'] = image

        return return_dict

