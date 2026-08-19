#!/bin/bash
#SBATCH -J clews-emb
#SBATCH --nodelist=node028
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8g
#SBATCH --chdir=/home/raraz/unified-track-and-version-id/
#SBATCH -o logs/slurm_logs/%A.%N.out
#SBATCH -e logs/slurm_logs/%A.%N.err

source /etc/profile

######################################################################## Anaconda ###########################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

PYTHON_EXEC="/home/raraz/.conda/envs/clews/bin/python"

######################################################################### Variables #########################################################################

CLEWS_CKPT_PATH="/home/raraz/version_identification/clews/clews_zenodo/dvi-clews/checkpoint_best.ckpt"

MUSIC_DIR=$1
OUTPUT_DIR=$2
HOP_DUR=$3

######################################################################## Main ###############################################################################

srun $PYTHON_EXEC scripts/similar-region/clews-embedding-extraction.py \
  $MUSIC_DIR \
  $CLEWS_CKPT_PATH \
  $OUTPUT_DIR \
  --shingle-hop-dur $HOP_DUR \
  --num-workers $SLURM_CPUS_PER_TASK