#!/bin/bash
#SBATCH -J eval-TI
#SBATCH --nodelist=node028
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
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

QUERIES=$1
GT=$2
OUTPUT=$3
DB_FLAG=$4
DB_PATH=$5

N=10

###########################################################################################################################

python -u evaluate.py "${QUERIES}" "${GT}" \
    --output-dir "${OUTPUT}" \
    "${DB_FLAG}" "${DB_PATH}" \
    --id-level "track" \
    --top-N $N \
    --num-workers $SLURM_CPUS_PER_TASK
