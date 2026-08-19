#!/bin/bash
#SBATCH -J unified-prepare
#SBATCH -p medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=12g
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

INPUT_DIR="/projects/mtg/projects/unified-similarity/datasets/neural-music-fp-dataset-test-val-16kHz/test/database/"

###########################################################################################################################

python -u manipulate-and-degrade.py "$INPUT_DIR" \
    --write-tracks \
    --sample-chunks \
    --batch-size $SLURM_CPUS_PER_TASK \
    --workers $SLURM_CPUS_PER_TASK