#!/bin/bash
#SBATCH -J inf-unif
#SBATCH --qos=impa
#SBATCH --partition=impa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8g
#SBATCH --chdir=/home/raraz/unified-track-and-version-id
#SBATCH -o logs/slurm_logs/%A.%N.out
#SBATCH -e logs/slurm_logs/%A.%N.err

###########################################################################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

conda activate unified

###########################################################################################################################

# Set CUDA devices
export MASTER_PORT=$((12000 + RANDOM % 20000))
export NCCL_DEBUG=WARN

# Kill job faster if OOM/hang (3 min timeout)
export NCCL_TIMEOUT=180
export TORCH_NCCL_BLOCKING_WAIT=1

# Get number of GPUs
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPUs"

AUDIO=$1
CKPT=$2
OUTPUT_ROOT=$3
shift 3

# Any extra arguments are forwarded, e.g. --segment-duration and --overlap-ratio
torchrun --standalone --nproc_per_node=$NUM_GPUS inference.py \
    $AUDIO $CKPT $OUTPUT_ROOT \
    --num-workers $((SLURM_CPUS_PER_TASK / NUM_GPUS)) "$@"