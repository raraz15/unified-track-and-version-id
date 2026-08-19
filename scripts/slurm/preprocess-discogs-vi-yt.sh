#!/bin/bash
#SBATCH -J unified-prepare
#SBATCH -p high-cpu
#SBATCH --time=2-00:00:00        # Sets a time limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-99
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

INPUT_DIR="/projects/mtg/projects/version-identification/datasets/Discogs-VI/all_music/music/"
OUTPUT_DIR="/projects/mtg/projects/version-identification/datasets/Discogs-VI/all_music/music-wav-16bit16kHz-python/"

###########################################################################################################################

python -u scripts/preprocess-audio/preprocess-discogs-vi.py "$INPUT_DIR" "$OUTPUT_DIR" \
    --num-partitions 100 --partition-idx $SLURM_ARRAY_TASK_ID
