#!/bin/bash
#SBATCH -J silence-analysis
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180g
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

INPUT_DIR="/home/raraz/dvi-segment-tags/tags"
OUTPUT_DIR_JSON="/home/raraz/dvi-segment-tags/nonmusic-segments"
OUTPUT_DIR_TXT="/home/raraz/dvi-segment-tags/nonmusic-tracks"

######################################################################## Main ###############################################################################

python -u scripts/preprocess-audio/non-music-finder.py $INPUT_DIR $OUTPUT_DIR_JSON $OUTPUT_DIR_TXT --workers 36
