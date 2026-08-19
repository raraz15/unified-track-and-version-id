#!/bin/bash
#SBATCH -J unified-prepare
#SBATCH -p high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192g
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

INPUTS="/projects/mtg/projects/unified-similarity/datasets/discogs-vi-yt-16kHz/audio"
OUTPUT_DIR="/projects/mtg/projects/unified-similarity/datasets/discogs-vi-yt-16kHz/rms-silent-track-paths"

###########################################################################################################################

python -u scripts/preprocess-audio/find-silent-tracks.py "$INPUTS" "$OUTPUT_DIR" --workers 48