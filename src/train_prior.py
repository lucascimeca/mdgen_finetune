
from diffusers.optimization import get_cosine_schedule_with_warmup

from rtb_utils.gfn_diffusion import DiffuserTrainer, TrainingConfig, get_DDPM_diffuser_pipeline

from rtb_utils.args import fetch_args
from rtb_utils.priors import MDGenSimulator
from rtb_utils.pytorch_utils import seed_experiment, unfreeze_model, safe_reinit
from rtb_utils.rewards import Amber14Reward

import torch
import numpy as np


# get arguments for the run
args, state = fetch_args(exp_prepend='train_prior')
print(f"Running experiment on '{args.device}'")

# -------- OVERRIDE ARGUMENTS (if you have to) ------

logtwopi = np.log(2 * 3.14159265358979)

seed_experiment(args.seed)

reward_model = Amber14Reward(device=args.device, energy_temperature=args.energy_temperature, implicit=True)
r_str = "ss_div_seed_" + str(args.seed)
reward_args = []
prior_model = MDGenSimulator(
    peptide=args.peptide,
    sim_ckpt=f'{args.load_path}forward_sim.ckpt',
    data_dir=f'{args.data_path}4AA_data',
    split=f'{args.splits_path}4AA_test.csv',
    num_rollouts=1,
    num_frames=args.num_frames,
    retain=args.test_sample_size,
    xtc=True,
    out_dir=f"{args.save_folder}/samples/",
    suffix='_i100'
)

ddpmgfn_pipeline = get_DDPM_diffuser_pipeline(args, prior_model)

params = [param for param in ddpmgfn_pipeline.posterior_node.get_unet_parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=args.epochs,
)


class SourceSampler:
    def __init__(self):
        _, self.T, self.L, _ = prior_model.dims

    def sample(self, T=None, L=None):
        if T is None: T = self.T
        if L is None: L = self.L
        return prior_model.model.sample_prior_latent(args.batch_size, T, L, device=args.device, uniform=True)


trainer = DiffuserTrainer(
    config=TrainingConfig(args),
    model=ddpmgfn_pipeline.posterior_node.policy.unet,
    sampler=ddpmgfn_pipeline,
    source_sampler=SourceSampler(),
    scheduler=ddpmgfn_pipeline.posterior_node.policy.scheduler,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    reward_function=reward_model
)

trainer.train()

