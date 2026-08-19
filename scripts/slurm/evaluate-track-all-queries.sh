#!/bin/bash
#SBATCH -J eval-TI
#SBATCH --qos=impa
#SBATCH --partition=impa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8g
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

EMB_ROOT=$1
AUDIO_ROOT=$2 # for gt
QUERY_TYPE_IDX=${3:-""} # optional: 0=clean, 1=clean-manipulated, 2=clean-manipulated-degraded

N=10
QUERY_TYPES=("clean" "clean-manipulated" "clean-manipulated-degraded")

if [[ -n "$QUERY_TYPE_IDX" ]]; then
    QUERY_TYPES=("${QUERY_TYPES[$QUERY_TYPE_IDX]}")
fi


###########################################################################################################################

DATASET=$(basename "${EMB_ROOT}")
OUTPUT_ROOT="$(dirname "$(dirname "${EMB_ROOT}")")/eval/track-id/${DATASET}"

QUERY_EMB_ROOT="${EMB_ROOT}/queries/chunks"
DB_EMB_ROOT="${EMB_ROOT}/database"
QUERY_AUDIO_ROOT="${AUDIO_ROOT}/queries/chunks"

for QUERY_TYPE in "${QUERY_TYPES[@]}"; do
    echo "Processing query type: ${QUERY_TYPE}"

    QUERIES="${QUERY_EMB_ROOT}/${QUERY_TYPE}"
    echo "Queries path: ${QUERIES}"
    GT="${QUERY_AUDIO_ROOT}/${QUERY_TYPE}/ground-truth.csv"
    echo "Ground truth path: ${GT}"

    # Check if database.mm exists
    DB_PATH="${DB_EMB_ROOT}/database.mm"
    if [[ -f $DB_PATH ]]; then
        DB_FLAG="--database-memmap"
    else
        DB_FLAG="--database-embeddings"
        DB_PATH="${DB_EMB_ROOT}"
    fi
    echo "Database flag: ${DB_FLAG}"
    echo "Database path: ${DB_PATH}"

    OUTPUT="${OUTPUT_ROOT}/chunks/${QUERY_TYPE}"
    echo "Output path: ${OUTPUT}"
    OUTPUT_ARG=(--output-dir "${OUTPUT}")

    python -u evaluate.py "${QUERIES}" "${GT}" \
        "${OUTPUT_ARG[@]}" \
        "${DB_FLAG}" "${DB_PATH}" \
        --id-level "track" \
        --top-N $N \
        --num-workers $SLURM_CPUS_PER_TASK

done
