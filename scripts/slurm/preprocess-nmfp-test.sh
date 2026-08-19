#!/bin/bash
#SBATCH -J unified-prepare
#SBATCH -p high-cpu
#SBATCH --time=1-00:00:00        # Sets a time limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-47
#SBATCH --cpus-per-task=1       # Using 40 processes per job
#SBATCH --mem=4g
#SBATCH --chdir=/home/raraz/unified-track-and-version-id
#SBATCH -o logs/slurm_logs/%A_%a.%N.out
#SBATCH -e logs/slurm_logs/%A_%a.%N.err

###########################################################################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

conda activate unified

###########################################################################################################################

# TODO: write next to the original directory to have another copy
# then cp to unified-similarity
INPUTS="/projects/mtg/projects/unified-similarity/datasets/neural-music-fp-dataset-16kHz-16bit/music/test/nmfp_test_paths_in_fma.txt"
OUTPUT_DIR="/projects/mtg/projects/unified-similarity/datasets/neural-music-fp-dataset-16kHz-16bit/music/test/database/"

###########################################################################################################################

python -u scripts/preprocess-audio/preprocess-nmfp-test.py "$INPUTS" "$OUTPUT_DIR" \
    --num-partitions 48 --partition-idx $SLURM_ARRAY_TASK_ID
