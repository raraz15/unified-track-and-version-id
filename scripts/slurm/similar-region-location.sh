#!/bin/bash
#SBATCH -J region-loc
#SBATCH -p medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=4G
#SBATCH --chdir=/home/raraz/unified-track-and-version-id/
#SBATCH -o logs/slurm_logs/%A.%N.out
#SBATCH -e logs/slurm_logs/%A.%N.err

source /etc/profile

######################################################################## Anaconda ###########################################################################

# Load the anaconda module
module load Anaconda3/2023.09-0

# Enable the bash shell
eval "$(conda shell.bash hook)"

conda activate unified

######################################################################### Variables #########################################################################

DISCOGS_VI_DIR="/projects/mtg/projects/version-identification/datasets/Discogs-VI/dataset_release/discogs_20240701/main/"
MUSIC_DIR="/projects/mtg/projects/unified-similarity/datasets/discogs-vi-yt-16kHz/audio"
EMBEDDINGS_ROOT_DIR="/projects/mtg/projects/unified-similarity/dvi-clews-embeddings/20s"
NON_MUSIC_DIR="/projects/mtg/projects/unified-similarity/dvi-segment-tags/nonmusic-segments"

OUTPUT_ROOT_DIR=$1

case "$2" in
  train)
    DISCOGS_VI_YT_PATH="$DISCOGS_VI_DIR/Discogs-VI-YT-20240701-light.json.train"
    EMBEDDINGS_DIR="$EMBEDDINGS_ROOT_DIR/train"
    OUTPUT_DIR="$OUTPUT_ROOT_DIR/train"
    ;;
  val)
    DISCOGS_VI_YT_PATH="$DISCOGS_VI_DIR/Discogs-VI-YT-20240701-light.json.val"
    EMBEDDINGS_DIR="$EMBEDDINGS_ROOT_DIR/val/database"
    OUTPUT_DIR="$OUTPUT_ROOT_DIR/val"
    ;;
  *)
    echo "Invalid split '$2'. Use 'train' or 'val'."
    exit 1
    ;;
esac

######################################################################## Threading #########################################################################

# Prevent BLAS/PyTorch thread oversubscription (1 thread per worker process).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

######################################################################## Main ###############################################################################

python -u scripts/similar-region/similar-region-location.py \
  $DISCOGS_VI_YT_PATH \
  $MUSIC_DIR \
  $EMBEDDINGS_DIR \
  $NON_MUSIC_DIR \
  $OUTPUT_DIR \
  --num-workers $SLURM_CPUS_PER_TASK
