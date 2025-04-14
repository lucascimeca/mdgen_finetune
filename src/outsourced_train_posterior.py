import sys
import os
import shutil
import atexit
import signal

from rtb_utils.args import fetch_args
from rtb_utils.gfn_diffusion import get_DDPM_diffuser_pipeline, FinetuneRTBTrainer, FinetuneRTBBatchedTrainer
from rtb_utils.priors import MDGenSimulator
from rtb_utils.pytorch_utils import seed_experiment
from rtb_utils.rewards import Amber14Reward

import torch
import numpy as np


# get arguments for the run
args, state = fetch_args(exp_prepend='train_posterior')
print(f"Running experiment on '{args.device}'")

print(args)
# -------- OVERRIDE ARGUMENTS (if you have to) ------

logtwopi = np.log(2 * 3.14159265358979)

seed_experiment(args.seed)

reward_model = Amber14Reward(
    device=args.device,
    energy_temperature=args.energy_temperature,
    implicit=False
)
r_str = "ss_div_seed_" + str(args.seed)
reward_args = []
prior_model = MDGenSimulator(
    peptide=args.peptide,
    sim_ckpt=f'{args.load_path}forward_sim.ckpt',
    data_dir=f'{args.data_path}4AA_data',
    split=f'{args.splits_path}4AA_test.csv',
    num_rollouts=1,
    num_frames=1,
    retain=args.test_sample_size,
    xtc=True,
    out_dir=f"{args.save_folder}/samples/",
    suffix='_i100'
)

ddpm_pipeline = get_DDPM_diffuser_pipeline(args, prior_model)
params = [param for param in ddpm_pipeline.posterior_node.get_unet_parameters() if param.requires_grad]
opt = torch.optim.Adam([{'params': params,
                         'lr': args.learning_rate},
                        {'params': [ddpm_pipeline.logZ],
                         'lr': args.lr_logZ,
                         'weight_decay':args.z_weight_decay}])

Trainer = FinetuneRTBTrainer if not args.rtb_batched_train else FinetuneRTBBatchedTrainer

trainer = Trainer(
    sampler=ddpm_pipeline,
    reward_function=reward_model,
    optimizer=opt,
    peptide=args.peptide,
    save_folder=args.save_folder,
    config=args
)

# Main execution (training, inference, etc.)
trainer.run(
    peptide=args.peptide,
    epochs=args.epochs,
    back_and_forth=args.back_and_forth,
)

