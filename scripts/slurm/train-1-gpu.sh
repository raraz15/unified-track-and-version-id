#!/bin/bash
#SBATCH -J train-unif
#SBATCH -p impa
#SBATCH --qos=impa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
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

python train.py "$@" --num-workers $(( SLURM_CPUS_PER_TASK - 1 ))
