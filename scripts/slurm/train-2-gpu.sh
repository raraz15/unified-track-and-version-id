#!/bin/bash
#SBATCH -J train-unif
#SBATCH --partition=impa
#SBATCH --qos=impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=24g
#SBATCH --chdir=/home/raraz/unified-track-and-version-id
#SBATCH -o logs/slurm_logs/%A.%N.out
#SBATCH -e logs/slurm_logs/%A.%N.err

###########################################################################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

# Activate the project conda environment
conda activate unified

###########################################################################################################################

# Set CUDA memory allocation to be more flexible
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For debugging
# export CUDA_LAUNCH_BLOCKING=1

# Set CUDA devices
export MASTER_PORT=$((12000 + RANDOM % 20000))
export NCCL_DEBUG=WARN

# Kill job faster if OOM/hang (3 min timeout)
export NCCL_TIMEOUT=180
export TORCH_NCCL_BLOCKING_WAIT=1

# Get number of GPUs
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPUs"

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS train.py "$@" \
    --num-workers $(( (SLURM_CPUS_PER_TASK / NUM_GPUS) - 1 ))
