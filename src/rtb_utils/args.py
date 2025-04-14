import warnings

from rtb_utils.simple_io import folder_exists, folder_create

warnings.filterwarnings("ignore")

import argparse
import platform
import os
import torch
from distutils.util import strtobool

if platform.system() == "Windows":
    home_folder = ""  # Specify the home folder path for Windows, if different from the default
    system = 'win'
elif platform.system() == "Darwin":  # Darwin is the system name for macOS
    home_folder = os.path.expanduser("~")  # Home folder path for macOS
    system = 'mac'
else:
    # This will cover Linux and other Unix-like systems
    home_folder = os.path.expanduser("~")  # Home folder path for Linux/Unix
    system = 'linux'


def fetch_args(experiment_run=True, exp_prepend='exp', ldm=None):
    parser = argparse.ArgumentParser(description='PyTorch Feature Combination Mode')

    parser.add_argument('-md', '--model', default='UNet', type=str, help="'UNet', 'ScoreNet' supported at the moment.")
    parser.add_argument('-sf', '--save_folder', type=str, default="./../results", help='Path to save results to.')
    parser.add_argument('-pf', '--load_path', type=str, default="../pretrained", help="Folder to keep pretrained 'best' weights.")
    parser.add_argument('-dp', '--data_path', type=str, default=f"{home_folder}/data", help="Folder containing datasets.")
    parser.add_argument('-rp', '--resume', type=strtobool, default=True, help='Replace run logs.')  # todo: replace true

    parser.add_argument('-fs', '--show_figures', type=strtobool, default=False, help='show plots.')
    parser.add_argument('-fo', '--save_figures', type=strtobool, default=True, help='save plots.')
    parser.add_argument('-sw', '--save_model_weights', type=strtobool, default=False, help='save model weights.')
    parser.add_argument('--plot_batch_size', default=30, type=int)

    parser.add_argument('--batched_rtb_size', default=64, type=int)
    parser.add_argument('--rtb_batched_train', default=False, type=strtobool)

    parser.add_argument('-en', '--exp_name', type=str, default='', help='Experiment name.')

    parser.add_argument('--method', default='rtb', type=str, help='Method to use for training (reinforce, rtb, tb).')
    parser.add_argument('--cla', default=False, type=strtobool, help='Whether to do CLA trick.')

    # Optimization options
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='Number of data loading workers (default: 4).')

    # GFN/Training Parameters
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='Number of epochs to run')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help="Training Batch Size.")
    parser.add_argument('-tbs', '--test_sample_size', default=300, type=int, help='Test batchsize.')

    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_logZ', default=1e-1, type=float, help='Learning rate for logZ.')
    parser.add_argument('--z_weight_decay', default=0, type=float, help='Weight decay for logZ.')
    parser.add_argument('--vargrad', default=False, type=strtobool, help='Whether to use vargrad.')
    parser.add_argument('--vargrad_sample_n0', default=4, type=int, help='How many samples to use for vargrad.')
    parser.add_argument('--energy_temperature', default=1., type=float, help='temperature of energy function.')
    parser.add_argument('--conditional', default=False, type=strtobool, help='This will run the conditional version of the models. Use only in conjuction with inverse_conditional_finetune.py at the moment.')

    parser.add_argument('--snr_training', default=True, type=strtobool, help='Whether to use snr scaling loss for prior training.')
    parser.add_argument('--snr_gamma', default=5., type=float, help='Clipping value for snr loss.')

    parser.add_argument('--peptide', type=str, default='FLRH', help='Peptide to use.')

    parser.add_argument('--traj_length', default=100, type=int, help='Legth of trajectory.')
    parser.add_argument('--sampling_length', default=100, type=int, help='Legth of sampling traj. If sampling_length < traj_length then DDPM automatically kicks in.')
    parser.add_argument('--learning_cutoff', default=1e-1, type=float, help='Cut-off for allowed closeness of prior/posterior given inpferfect TB assumption.')
    parser.add_argument('--learn_var', default=False, type=strtobool, help='Whether to learn the variance. Mind the models will give a tuple with two outputs if flagged to True.')
    parser.add_argument('--loss_type', type=str, default='l2', help='Loss type to use for regular diffusion training.')
    parser.add_argument('-age', '--accumulate_gradient_every', default=1, type=int, help='Number of iterations to accumulate gradient for.')
    parser.add_argument('--detach_freq', default=0., type=float, help='Fraction of steps on which not to train')
    parser.add_argument('--detach_cut_off', default=1., type=float, help='Fraction of steps to keep from t=1 (full noise).')

    parser.add_argument('--back_and_forth', default=False, type=strtobool, help='Whether to train based on back and forth trajectories.')
    parser.add_argument('--bf_length', default=50, type=int, help='backward steps in the back and forth learning algoritm')
    parser.add_argument('--mixed_precision', default=False, type=strtobool, help='Whether to train with mixed precision.')
    parser.add_argument('--checkpointing', default=True, type=strtobool, help='Uses checkpointing to save memory in exchange for compute.')

    parser.add_argument('--use_prior_drift', type=strtobool, default=False)

    parser.add_argument('--prior_sampling', type=strtobool, default=False, help='Whether to use prior sampling for stability.')
    parser.add_argument('--prior_sampling_ratio', type=float, default=.1, help='Ratio to sample from prior.')

    parser.add_argument('-rb', '--replay_buffer', type=strtobool, default=False, help='Whether to use replay buffer.')
    # parser.add_argument('--rb_every', type=int, default=8, help='after how many epochs to do rb sampling.')
    parser.add_argument('--rb_ratio', type=float, default=.2, help='after how many epochs to do rb sampling.')
    parser.add_argument('--rb_size', default=1000, type=int, help='Max size of replay buffer.')
    parser.add_argument('--rb_sample_strategy', default='uniform', type=str, help='Sampling strategy from replay buffer (uniform/reward)')
    parser.add_argument('--rb_beta', default=1.0, type=float, help='Inverse temperature of sampling disctribution (unused for uniform)')

    # lora
    parser.add_argument('--lora', default=True, type=strtobool, help='low rank approximation training.')
    parser.add_argument('--rank', default=32, type=int, help='lora rank.')

    # hugging face
    parser.add_argument('--push_to_hf', default=False, type=strtobool)
    parser.add_argument('--push_to_wandb', default=True, type=strtobool)

    # wandb
    parser.add_argument('--notes', default="", type=str)

    # Miscs
    parser.add_argument('--seed', default=912, type=int, help='Manual seed.')

    # Paths
    parser.add_argument('--splits_path', default='../splits/', type=str, help='Path to save model checkpoints')
    parser.add_argument('--load_ckpt', default=True, type=strtobool, help='Whether to load checkpoint')

    parser.add_argument('--load_outsourced_ckpt', default=True, type=strtobool, help='Whether to load checkpoint')
    parser.add_argument('--load_outsourced_path', default='../pretrained/', type=str, help='Path to load model checkpoint')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # --------- multi-gpu special ---------------
    if ldm is not None:
        args.ldm = True  # can force ldm if training prior

    args.use_cuda = torch.cuda.device_count() != 0
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.system = system

    # --------- FINETUNING SPECIAL ARGS---------------
    # ---------  gradient accumulation ---------
    # if args.batch_size * args.accumulate_gradient_every < 32:
    #     args.accumulate_gradient_every = int(32/args.batch_size)
    #     print(f"*forcing '--accumulate_gradient_every' to {args.accumulate_gradient_every}"
    #           f"\n*effective batch size: {args.batch_size * args.accumulate_gradient_every}")

    # if we're doing batch train logic let's turn off checkpointing
    if args.rtb_batched_train:
        args.checkpointing = False

        if args.batched_rtb_size < args.batch_size:
            print(f"Argument '--batched_rtb_size' must be at least as large as '--batch_size' when the '--rtb_batched_train' modality is active. \n"
                  f"Forcing '--batched_rtb_size=batch_size'")
            args.batched_rtb_size = args.batch_size

    # ------ FOLDER CREATION AND SETTINGS -----------------
    # create exp_name if it wasn't given, then choose and create a folder to save exps based on parameters

    exp_critical_args = [
        # 'traj_length',
        # 'sampling_length',
        # 'batch_size',
        'method',
        'energy_temperature',
        'vargrad',
        # 'replay_buffer',
        # 'load_outsourced_ckpt'
        'prior_sampling'
    ]
    args.exp_name = f"{'_'.join([f'{k}_{args.__dict__[k]}' for k in exp_critical_args])}" + "normal" # todo remove new
    if len(args.exp_name) == 0:
        args.exp_name = f"mdgen_finetune_{args.model}"
    args.exp_name = f"{exp_prepend}_{args.exp_name}"

    if experiment_run:
        num = 0
        save_folder_name = f"{args.save_folder}/{args.exp_name}_{num}/"
        if not args.resume:
            num = 0
            while folder_exists(save_folder_name):
                num += 1
                save_folder_name = f"{args.save_folder}/{args.exp_name}_{num}/"
        args.save_folder = save_folder_name
        # args.load_path += f"/{args.exp_name}_{num}"

    folder_create(args.save_folder, exist_ok=True)
    folder_create(args.load_path, exist_ok=True)
    folder_create(f"{args.save_folder}/samples/", exist_ok=True)

    os.environ["DEEPLAKE_DOWNLOAD_PATH"] = args.data_path + '/'

    return args, state
