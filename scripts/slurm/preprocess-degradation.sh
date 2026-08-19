#!/bin/bash
#SBATCH -J unified-prepare-tut
#SBATCH -p high-cpu
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
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

# INPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/TUT-acoustic-scenes-2016/"
# OUTPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/TUT-acoustic-scenes-2016-16khz-16bit"

# INPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/microphone/Microphone_Impulse_Responses/"
# OUTPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/microphone/Microphone_Impulse_Responses-16khz-16bit"

# INPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/room/MIT_Survey/"
# OUTPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/room/MIT_Survey-16khz-16bit/"

# INPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/room/AIR_1_4-Binaural-from_mat-no_dummy_head/"
# OUTPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/room/AIR_1_4-Binaural-from_mat-no_dummy_head-16khz-16bit/"

INPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/room/OPENAIR-mono_and_stereo/"
OUTPUT_DIR="/projects/mtg/projects/track-identification/datasets/degradations/IR/room/OPENAIR-mono_and_stereo-16khz-16bit/"

###########################################################################################################################

python -u scripts/preprocess-audio/preprocess-degradation.py \
    ${INPUT_DIR} ${OUTPUT_DIR} \
    --num-partitions 4 --partition-idx $SLURM_ARRAY_TASK_ID
