import sys
import os
import shutil
import atexit
import signal

from rtb_utils.priors import MDGenSimulator
from rtb_utils.rewards import Amber14Reward

import torch
import numpy as np
import random
import argparse
from distutils.util import strtobool

from rtb_utils import protein_rtb
from rtb_utils.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running experiments on '{device}'")
parser = argparse.ArgumentParser()

# Argument parsing remains unchanged
parser.add_argument('--exp_name', default="test", type=str, help='Experiment name')
parser.add_argument('--tb', default=False, type=strtobool, help='Whether to use tb (vs rtb)')
parser.add_argument('--n_iters', default=50000, type=int, metavar='N', help='Number of training iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Training Batch Size.")
parser.add_argument('--num_test_samples', type=int, default=64, help="Test Batch Size.")
parser.add_argument('--num_target_confs', type=int, default=300, help="Test Batch Size.")
parser.add_argument('--loss_batch_size', type=int, default=-1, help="Batched RTB loss batch size")
parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float, help='Initial learning rate.')
parser.add_argument('--peptide', type=str, default='FLRH', help='Peptide sequence.')
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--wandb_track', default=False, type=strtobool, help='Whether to track with wandb.')
parser.add_argument('--entity', default=None, type=str, help='Wandb entity')
parser.add_argument('--prior_sample', default=False, type=strtobool, help="Whether to use off policy samples from prior")
parser.add_argument('--replay_buffer', default='none', type=str, help='Type of replay buffer to use', choices=['none','uniform','reward'])
parser.add_argument('--prior_sample_prob', default=0.0, type=float, help='Probability of using prior samples')
parser.add_argument('--replay_buffer_prob', default=0.0, type=float, help='Probability of using replay buffer samples')
parser.add_argument('--beta_start', default=1.0, type=float, help='Initial Inverse temperature for reward (Also used if anneal=False)')
parser.add_argument('--beta_end', default=10.0, type=float, help='Final Inverse temperature for reward')
parser.add_argument('--anneal', default=False, type=strtobool, help='Whether to anneal beta (From beta_start to beta_end)')
parser.add_argument('--anneal_steps', default=15000, type=int, help="Number of steps for temperature annealing")
parser.add_argument('--orig_scale', default=1.0, type=float, help='STD of the gaussian at x_0.')

parser.add_argument('--save_path', default='~/scratch/CNF_RTB_ckpts/', type=str, help='Path to save model checkpoints')
parser.add_argument('--load_ckpt', default=True, type=strtobool, help='Whether to load checkpoint')
parser.add_argument('--load_path', default='../pretrained/', type=str, help='Path to load model checkpoint')

parser.add_argument('--load_outsourced_ckpt', default=True, type=strtobool, help='Whether to load checkpoint')
parser.add_argument('--load_outsourced_path', default='../pretrained/', type=str, help='Path to load model checkpoint')

parser.add_argument('--lora', default=True, type=strtobool, help='low rank approximation training.')
parser.add_argument('--rank', default=32, type=int, help='lora rank.')

parser.add_argument('--langevin', default=False, type=strtobool, help="Whether to use Langevin dynamics for sampling")
parser.add_argument('--inference', default='vpsde', type=str, help='Inference method for prior', choices=['vpsde', 'ddpm'])
parser.add_argument('--seed', default=0, type=int, help='Random seed for training')
parser.add_argument('--clip', default=0.1, type=float, help='Gradient clipping value')
parser.add_argument('--data_path', default='~/scratch/mdgen/data/', type=str, help='Path to save model checkpoints')
parser.add_argument('--splits_path', default='~/scratch/mdgen/splits/', type=str, help='Path to save model checkpoints')

args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

reward_model = Amber14Reward(device=device)
r_str = "ss_div_seed_" + str(args.seed)
reward_args = []
prior_model = MDGenSimulator(
    peptide=args.peptide,
    sim_ckpt=f'{args.load_path}forward_sim.ckpt',
    data_dir=f'{args.data_path}4AA_data',
    split=f'{args.splits_path}4AA_test.csv',
    num_rollouts=1,
    num_frames=1,
    retain=args.num_target_confs,
    xtc=True,
    out_dir=args.save_path,
    suffix='_i100'
)

in_shape = prior_model.dims[1:]
seq_len = in_shape[2]
id = "protein_mdgen_"+ r_str +"_len_" + str(seq_len)

replay_buffer = None
if args.replay_buffer != 'none':
    replay_buffer = ReplayBuffer(rb_size=10000, rb_sample_strategy=args.replay_buffer)

rtb_model = protein_rtb.ProteinRTBModel(
    device=device,
    reward_model=reward_model,
    prior_model=prior_model,
    in_shape=in_shape,
    reward_args=reward_args,
    id=id,
    model_save_path=args.save_path,
    langevin=args.langevin,
    inference_type=args.inference,
    tb=args.tb,
    load_ckpt=args.load_outsourced_ckpt,
    load_ckpt_path=args.load_outsourced_path,
    entity=args.entity,
    diffusion_steps=args.diffusion_steps,
    beta_start=args.beta_start,
    beta_end=args.beta_end,
    loss_batch_size=args.loss_batch_size,
    replay_buffer=replay_buffer,
    orig_scale=args.orig_scale,
    config=args
)


def cleanup():
    if hasattr(rtb_model, 'tmp_dir') and os.path.exists(rtb_model.tmp_dir):
        print(f"Cleaning up temporary directory: {rtb_model.tmp_dir}")
        shutil.rmtree(rtb_model.tmp_dir)

# Register cleanup function for normal exit
atexit.register(cleanup)

# Register cleanup function for termination signals
def signal_handler(sig, frame):
    cleanup()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle termination



try:
    # Main execution (training, inference, etc.)
    # rtb_model.finetune(
    #     shape=(args.batch_size, *in_shape),
    #     n_iters=args.n_iters,
    #     learning_rate=args.lr,
    #     clip=args.clip,
    #     prior_sample_prob=args.prior_sample_prob,
    #     replay_buffer_prob=args.replay_buffer_prob,
    #     anneal=args.anneal,
    #     anneal_steps=args.anneal_steps
    # )

    rtb_model.denoising_score_matching_unif(
        shape=(args.batch_size, *in_shape),
        n_iters=10000,
        learning_rate=5e-5,
        clip=0.1
    )
finally:
    cleanup()

