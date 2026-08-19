#!/bin/bash
#SBATCH -J eval-VI
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
QUERY_TYPE_IDX=${2:-""} # optional: 0=clean, 1=clean-manipulated, 2=clean-manipulated-degraded
if [[ "${EMB_ROOT,,}" == *"shs"* ]]; then
    GT=/projects/mtg/projects/version-identification/datasets/SHS100K2/SHS100K2-TEST-cliques.json.filled
elif [[ "${EMB_ROOT,,}" == *"dvi"* ]]; then
    GT=/projects/mtg/projects/version-identification/datasets/Discogs-VI/dataset_release/discogs_20240701/main/Discogs-VI-YT-20240701-light.json.test
else
    echo "Error: EMB_ROOT must contain 'shs' or 'discogs'" >&2
    exit 1
fi

N=10000
QUERY_TYPES=("clean" "clean-manipulated" "clean-manipulated-degraded")

if [[ -n "$QUERY_TYPE_IDX" ]]; then
    QUERY_TYPES=("${QUERY_TYPES[$QUERY_TYPE_IDX]}")
fi

###########################################################################################################################

DATASET=$(basename "${EMB_ROOT}")
OUTPUT_ROOT="$(dirname "$(dirname "${EMB_ROOT}")")/eval/version-id/${DATASET}"

QUERY_EMB_ROOT="${EMB_ROOT}/queries/tracks"
DB_EMB_ROOT="${EMB_ROOT}/database"

for QUERY_TYPE in "${QUERY_TYPES[@]}"; do

    # The clean query tracks are the database tracks themselves
    if [[ "${QUERY_TYPE}" == "clean" ]]; then
        QUERIES="${DB_EMB_ROOT}"
    else
        QUERIES="${QUERY_EMB_ROOT}/${QUERY_TYPE}"
    fi
    echo "Queries path: ${QUERIES}"
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

    OUTPUT="${OUTPUT_ROOT}/${QUERY_TYPE}"
    echo "Output path: ${OUTPUT}"
    OUTPUT_ARG=(--output-dir "${OUTPUT}")

    python -u evaluate.py "${QUERIES}" "${GT}" \
        "${OUTPUT_ARG[@]}" \
        "${DB_FLAG}" "${DB_PATH}" \
        --id-level "version" \
        --top-N $N \
        --num-workers $SLURM_CPUS_PER_TASK \

done