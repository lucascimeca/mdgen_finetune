#!/bin/bash
#SBATCH --constraint="40gb|48gb|80gb"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out

# or > salloc --gres=gpu:1 --constraint="40gb|48gb|80gb" --cpus-per-task=8 --mem=32G  --time=12:00:00 --nodes=1 --partition=main

# Compute the output directory after the SBATCH directives
export OUTPUT_DIR=$HOME/script_outputs
# Update the output path of the SLURM output file
export SLURM_JOB_OUTPUT=${OUTPUT_DIR}/$(basename ${SLURM_JOB_OUTPUT})

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

python ../mdgen_finetune.py --data_path $SCRATCH/data/ --save_folder $SCRATCH/rtb_inverse_results \
       --load_path ./models/pretrained --use_rtb_drift True --learn_dps_drift False \
       --use_prior_drift False --replay_buffer False --conditional True --compute_fid True \
       --dataset $1 --inv_task $2 --lr $3 --sampling_length $4 --batch_size $5 \
       --accumulate_gradient_every $6 --epochs $7 --energy_temperature $8 \
       --rtb_batched_train $9 --batched_rtb_size ${10} --vargrad ${11} --ldm ${12} --particles ${13}\
       --checkpointing ${14} --push_to_hf ${15} --method ${16} --cla ${17}  --use_dps_drift ${18} --exp_name ${19}