#!/bin/bash
#SBATCH --job-name=ii_mean
#SBATCH --output=logs/ii_mean_%A_%a.out
#SBATCH --error=logs/ii_mean_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1-30%1

# Compute information imbalance from mean-pooled activations.
# One comparison per array task, %1 = sequential (shared GPU).

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
DEEPSEEK_MEAN="/home/laios/activations/flickr/deepseek_windsorised_mean"
LLAMA_MEAN="${MEAN_BASE}/llama"
DINOV2_MEAN="${MEAN_BASE}/facebook-dinov2-large"
DINOV2_BASE_MEAN="${MEAN_BASE}/facebook-dinov2-base"
DINOV2_SMALL_MEAN="${MEAN_BASE}/facebook-dinov2-small"
IGPT_MEAN="${MEAN_BASE}/openai-imagegpt-large"
IGPT_MED_MEAN="${MEAN_BASE}/openai-imagegpt-medium"
IGPT_SMALL_MEAN="${MEAN_BASE}/openai-imagegpt-small"
CLIP_IMAGE_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-image"
CLIP_TEXT_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-text"

OUTPUT_BASE="${IMAGE_BASE}/results-mean"

case ${SLURM_ARRAY_TASK_ID} in
    # --- igpt <-> deepseek ---
    1)
        NAME="deepseek_igpt"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        LAYER_A=55
        ;;
    2)
        NAME="igpt_deepseek"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=20
        ;;
    # --- dino (large) <-> deepseek ---
    3)
        NAME="deepseek_dino"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=55
        ;;
    4)
        NAME="dino_deepseek"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=24
        ;;
    # --- dino (base) <-> deepseek ---
    5)
        NAME="deepseek_dino_base"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINOV2_BASE_MEAN}"
        LAYER_A=55
        ;;
    6)
        NAME="dino_base_deepseek"
        INPUT_A="${DINOV2_BASE_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=12
        ;;
    # --- dino (small) <-> deepseek ---
    7)
        NAME="deepseek_dino_small"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINOV2_SMALL_MEAN}"
        LAYER_A=55
        ;;
    8)
        NAME="dino_small_deepseek"
        INPUT_A="${DINOV2_SMALL_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=12
        ;;
    # --- igpt <-> dino (large) ---
    9)
        NAME="igpt_dino"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=20
        ;;
    10)
        NAME="dino_igpt"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        LAYER_A=24
        ;;
    # --- igpt <-> llama ---
    11)
        NAME="igpt_llama"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=20
        ;;
    12)
        NAME="llama_igpt"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        LAYER_A=20
        ;;
    # --- dino (large) <-> llama ---
    13)
        NAME="dino_llama"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=24
        ;;
    14)
        NAME="llama_dino"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=20
        ;;
    # --- dino (large) <-> clip image ---
    15)
        NAME="dino_clip_image"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        LAYER_A=24
        ;;
    16)
        NAME="clip_image_dino"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=24
        ;;
    # --- clip text <-> clip image ---
    17)
        NAME="clip_text_clip_image"
        INPUT_A="${CLIP_TEXT_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        LAYER_A=12
        ;;
    18)
        NAME="clip_image_clip_text"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${CLIP_TEXT_MEAN}"
        LAYER_A=24
        ;;
    # --- igpt-medium <-> deepseek ---
    19)
        NAME="deepseek_igpt_medium"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${IGPT_MED_MEAN}"
        LAYER_A=55
        ;;
    20)
        NAME="igpt_medium_deepseek"
        INPUT_A="${IGPT_MED_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=15
        ;;
    # --- igpt-medium <-> llama ---
    21)
        NAME="igpt_medium_llama"
        INPUT_A="${IGPT_MED_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=15
        ;;
    22)
        NAME="llama_igpt_medium"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${IGPT_MED_MEAN}"
        LAYER_A=20
        ;;
    # --- igpt-medium <-> dino (large) ---
    23)
        NAME="igpt_medium_dino"
        INPUT_A="${IGPT_MED_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=15
        ;;
    24)
        NAME="dino_igpt_medium"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${IGPT_MED_MEAN}"
        LAYER_A=24
        ;;
    # --- igpt-small <-> deepseek ---
    25)
        NAME="deepseek_igpt_small"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${IGPT_SMALL_MEAN}"
        LAYER_A=55
        ;;
    26)
        NAME="igpt_small_deepseek"
        INPUT_A="${IGPT_SMALL_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=10
        ;;
    # --- igpt-small <-> llama ---
    27)
        NAME="igpt_small_llama"
        INPUT_A="${IGPT_SMALL_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=10
        ;;
    28)
        NAME="llama_igpt_small"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${IGPT_SMALL_MEAN}"
        LAYER_A=20
        ;;
    # --- igpt-small <-> dino (large) ---
    29)
        NAME="igpt_small_dino"
        INPUT_A="${IGPT_SMALL_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=10
        ;;
    30)
        NAME="dino_igpt_small"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${IGPT_SMALL_MEAN}"
        LAYER_A=24
        ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "========================================="
echo "II (mean-pooled): ${NAME}"
echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "A: ${INPUT_A} (layer ${LAYER_A})"
echo "B: ${INPUT_B}"
echo "========================================="

METRICS="${METRICS:-hamming l2}"

python compute_ii_mean.py \
    --input_path_A "${INPUT_A}" \
    --input_path_B "${INPUT_B}" \
    --output_folder "${OUTPUT_BASE}/${NAME}" \
    --layer_A ${LAYER_A} \
    --mean_center \
    --l2_normalize \
    --metrics ${METRICS}

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
if [ ${STATUS} -eq 0 ]; then
    for m in ${METRICS}; do
        DIR="${OUTPUT_BASE}/${NAME}/${m}"
        [ "${m}" = "l2" ] && DIR="${OUTPUT_BASE}/${NAME}/l2_centered_normalized"
        [ -d "${DIR}" ] && echo "Output (${m}): $(ls ${DIR} | wc -l) layer files"
    done
fi

exit ${STATUS}
