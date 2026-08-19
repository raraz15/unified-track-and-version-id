#!/bin/bash
#SBATCH -J inf-unif
#SBATCH --qos=impa
#SBATCH --partition=impa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
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

AUDIO=$1
CKPT=$2
OUTPUT_ROOT=$3
shift 3

# Any extra arguments are forwarded, e.g. --segment-duration and --overlap-ratio
python inference.py $AUDIO $CKPT $OUTPUT_ROOT --num-workers $SLURM_CPUS_PER_TASK "$@"