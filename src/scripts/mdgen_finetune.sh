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


# run the following line to run prior code -- note, the path assume you run the files from the "scripts" folder
#python ../finetune_posterior.py --algo mle --batch_size 128 --lr $1 --lr_logZ $2 --traj_length 100 \
#        --epochs 1000 --learning_cutoff $3 --save_folder $SCRATCH/gfn_generated_results \
#        --load_path ./../models/pretrained --data_path $SCRATCH/data --exp_name $4
#        > ~/script_outputs/"${SLURM_JOB_NAME}.txt"

python ../prot_train.py --diffusion_steps 20 --save_path ~/scratch/mdgen/samples/ \
                        --data_path ~/scratch/mdgen/data/ --splits_path ../../splits/ \
                        --load_path ../../pretrained/ --tb True --clip 0 --learning_rate 1e-5 \
                        --wandb_track True
