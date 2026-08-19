#!/bin/bash
#SBATCH -J silence-analysis
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=460g
#SBATCH --chdir=/home/raraz/unified-track-and-version-id/
#SBATCH -o logs/slurm_logs/%A.%N.out
#SBATCH -e logs/slurm_logs/%A.%N.err

######################################################################## Anaconda ###########################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

conda activate unified

######################################################################### Variables #########################################################################

INPUT_DIR="/localdisk/raraz/unified-similarity/datasets/discogs-vi-yt-train-16kHz"
OUTPUT_DIR="/localdisk/raraz/unified-similarity/logs/silence-analysis/discogs-vi-yt-16kHz/silence-detection"

######################################################################## Main ###############################################################################

export TORCH_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -u scripts/preprocess-audio/analyze-rms-dbfs.py $INPUT_DIR $OUTPUT_DIR --workers 24
