#!/bin/bash
#SBATCH --job-name=agg_test
#SBATCH --output=logs/agg_test_%A_%a.out
#SBATCH --error=logs/agg_test_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1-19%2

# Aggregation justification experiment: vary DinoV2 token aggregation,
# measure II against DeepSeek (layer 55 fixed).
#
# Tasks 1-9:  preprocess DinoV2 with different aggregation strategies
# Tasks 10-19: compute II for each variant against DeepSeek

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
DINOV2_RAW="${IMAGE_BASE}/facebook-dinov2-large"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
DEEPSEEK_MEAN="/home/laios/activations/flickr/deepseek_windsorised_mean"
DINOV2_MEAN_ALL="${MEAN_BASE}/facebook-dinov2-large"  # already exists

AGG_BASE="${MEAN_BASE}/aggregation_test"
OUTPUT_BASE="${IMAGE_BASE}/results-mean/aggregation_test"

# =========================================================================
# Phase 1: Preprocess DinoV2 variants (tasks 1-9)
# =========================================================================
case ${SLURM_ARRAY_TASK_ID} in
    # --- Average variants ---
    1)
        NAME="dinov2_avg_last1"
        echo "Preprocess: average last 1 token"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --last_k 1 --verify
        ;;
    2)
        NAME="dinov2_avg_cls"
        echo "Preprocess: average first 1 token (CLS)"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --first_k 1 --verify
        ;;
    3)
        NAME="dinov2_avg_last10"
        echo "Preprocess: average last 10 tokens"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --last_k 10 --verify
        ;;
    4)
        NAME="dinov2_avg_last50"
        echo "Preprocess: average last 50 tokens"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --last_k 50 --verify
        ;;
    5)
        NAME="dinov2_avg_last100"
        echo "Preprocess: average last 100 tokens"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --last_k 100 --verify
        ;;
    # --- Concatenation variants ---
    6)
        NAME="dinov2_concat_last10"
        echo "Preprocess: concatenate last 10 tokens"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --last_k 10 --concat --verify
        ;;
    7)
        NAME="dinov2_concat_last50"
        echo "Preprocess: concatenate last 50 tokens"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --last_k 50 --concat --verify
        ;;
    8)
        NAME="dinov2_concat_last100"
        echo "Preprocess: concatenate last 100 tokens"
        python preprocess_mean.py \
            --input_path "${DINOV2_RAW}" \
            --output_path "${AGG_BASE}/${NAME}" \
            --last_k 100 --concat --verify
        ;;
    9)
        # Placeholder: all-tokens average already exists, just verify
        echo "Verifying existing all-tokens mean: ${DINOV2_MEAN_ALL}"
        python -c "
from safetensors.numpy import load_file
from pathlib import Path
files = sorted(Path('${DINOV2_MEAN_ALL}').glob('mean_chunk_*.safetensors'))
d = load_file(str(files[0]))
print(f'Existing mean-all: {len(files)} chunks, {len(d)} layers, shape={d[\"layer_0\"].shape}')
"
        ;;

    # =====================================================================
    # Phase 2: Compute II for each variant against DeepSeek (tasks 10-19)
    # =====================================================================
    # Average variants
    10)
        NAME="deepseek_dino_avg_last1"
        INPUT_B="${AGG_BASE}/dinov2_avg_last1"
        ;;
    11)
        NAME="deepseek_dino_avg_cls"
        INPUT_B="${AGG_BASE}/dinov2_avg_cls"
        ;;
    12)
        NAME="deepseek_dino_avg_last10"
        INPUT_B="${AGG_BASE}/dinov2_avg_last10"
        ;;
    13)
        NAME="deepseek_dino_avg_last50"
        INPUT_B="${AGG_BASE}/dinov2_avg_last50"
        ;;
    14)
        NAME="deepseek_dino_avg_last100"
        INPUT_B="${AGG_BASE}/dinov2_avg_last100"
        ;;
    15)
        NAME="deepseek_dino_avg_all"
        INPUT_B="${DINOV2_MEAN_ALL}"
        ;;
    # Concatenation variants
    16)
        NAME="deepseek_dino_concat_last10"
        INPUT_B="${AGG_BASE}/dinov2_concat_last10"
        ;;
    17)
        NAME="deepseek_dino_concat_last50"
        INPUT_B="${AGG_BASE}/dinov2_concat_last50"
        ;;
    18)
        NAME="deepseek_dino_concat_last100"
        INPUT_B="${AGG_BASE}/dinov2_concat_last100"
        ;;
    19)
        # Duplicate of task 10 (last-1 avg = last-1 concat), used as concat baseline
        NAME="deepseek_dino_concat_last1"
        INPUT_B="${AGG_BASE}/dinov2_avg_last1"
        ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

# Phase 2 tasks (10-19): compute II
if [ ${SLURM_ARRAY_TASK_ID} -ge 10 ]; then
    echo "========================================="
    echo "II aggregation test: ${NAME}"
    echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
    echo "Node: $(hostname)"
    echo "Started at: $(date)"
    echo "A: ${DEEPSEEK_MEAN} (layer 55)"
    echo "B: ${INPUT_B}"
    echo "========================================="

    python compute_ii_mean.py \
        --input_path_A "${DEEPSEEK_MEAN}" \
        --input_path_B "${INPUT_B}" \
        --output_folder "${OUTPUT_BASE}/${NAME}" \
        --layer_A 55 \
        --mean_center \
        --l2_normalize \
        --metrics l2

    STATUS=$?

    echo ""
    echo "Finished at: $(date)"
    echo "Exit code: ${STATUS}"
    if [ ${STATUS} -eq 0 ]; then
        DIR="${OUTPUT_BASE}/${NAME}/l2_centered_normalized"
        [ -d "${DIR}" ] && echo "Output: $(ls ${DIR} | wc -l) layer files"
    fi
    exit ${STATUS}
fi
