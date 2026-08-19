#!/bin/bash
#SBATCH -J segment-clique
#SBATCH -p high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4g
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

INPUT_CSV=$1
OUTPUT_DIR=$2

######################################################################## Main ###############################################################################

python -u scripts/similar-region/segment-clique-location.py $INPUT_CSV \
    --output-dir $OUTPUT_DIR \
    --workers $SLURM_CPUS_PER_TASK
