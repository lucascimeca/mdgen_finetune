#!/bin/bash
#SBATCH --constraint="40gb|48gb|80gb"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=96:00:00
#SBATCH --nodes=1

# or > salloc --gres=gpu:1 --constraint="40gb|48gb|80gb" --cpus-per-task=8 --mem=32G  --time=12:00:00 --nodes=1 --partition=main

source ~/.bashrc

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load anaconda/3
module load cuda/11.7

# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge rich tqdm
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich tqdm

# Activate pre-existing environment.
conda activate /home/mila/l/luca.scimeca/scratch/envs/mdgen

git pull


# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES

python ../prot_train.py --diffusion_steps 20 --save_path ~/scratch/mdgen/samples/ \
                        --data_path ~/scratch/mdgen/data/ --splits_path ../../splits/ \
                        --load_path ../../pretrained/ --tb False --learning_rate 1e-4  --inference ddpm \
                        --beta_start $1 --replay_buffer_prob .1 --replay_buffer uniform --wandb_track True
