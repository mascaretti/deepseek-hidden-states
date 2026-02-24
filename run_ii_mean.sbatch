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
#SBATCH --array=1-26

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
CLIP_IMAGE_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-image"
CLIP_TEXT_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-text"

OUTPUT_BASE="${IMAGE_BASE}/results-mean"

case ${SLURM_ARRAY_TASK_ID} in
    1)
        NAME="dino_igpt"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        LAYER_A=24
        ;;
    2)
        NAME="deepseek_clip_text"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${CLIP_TEXT_MEAN}"
        LAYER_A=52
        ;;
    3)
        NAME="deepseek_dino"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=52
        ;;
    4)
        NAME="deepseek_clip_image"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        LAYER_A=52
        ;;
    5)
        NAME="clip_text_clip_image"
        INPUT_A="${CLIP_TEXT_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        LAYER_A=12
        ;;
    6)
        NAME="dino_clip_image"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        LAYER_A=24
        ;;
    7)
        NAME="deepseek_llama"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=52
        ;;
    8)
        NAME="llama_deepseek"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=20  # TODO: pick after exploratory runs
        ;;
    9)
        NAME="llama_clip_text"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${CLIP_TEXT_MEAN}"
        LAYER_A=20  # TODO: pick after exploratory runs
        ;;
    10)
        NAME="llama_clip_image"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        LAYER_A=20  # TODO: pick after exploratory runs
        ;;
    # --- Reverse directions (11-18) ---
    11)
        NAME="igpt_dino"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=20
        ;;
    12)
        NAME="clip_text_deepseek"
        INPUT_A="${CLIP_TEXT_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=12
        ;;
    13)
        NAME="dino_deepseek"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=24
        ;;
    14)
        NAME="clip_image_deepseek"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=24
        ;;
    15)
        NAME="clip_image_clip_text"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${CLIP_TEXT_MEAN}"
        LAYER_A=24
        ;;
    16)
        NAME="clip_image_dino"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=24
        ;;
    17)
        NAME="clip_text_llama"
        INPUT_A="${CLIP_TEXT_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=12
        ;;
    18)
        NAME="clip_image_llama"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=24
        ;;
    19)
        NAME="deepseek_igpt"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        LAYER_A=52
        ;;
    20)
        NAME="igpt_deepseek"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        LAYER_A=20
        ;;
    21)
        NAME="llama_igpt"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        LAYER_A=20  # TODO: pick after exploratory runs
        ;;
    22)
        NAME="igpt_llama"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=20
        ;;
    23)
        NAME="llama_dino"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=20  # TODO: pick after exploratory runs
        ;;
    24)
        NAME="dino_llama"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        LAYER_A=24
        ;;
    # --- Self-comparisons (sanity checks) ---
    25)
        NAME="dino_dino"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=12
        ;;
    26)
        NAME="igpt_igpt"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        LAYER_A=20
        ;;
    27)
        NAME="deepseek_dino_base"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINO_MEAN_BASE}"
        LAYER_A=52
        ;;
    28)
        NAME="deepseek_dino_small"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINO_MEAN_SMALL}"
        LAYER_A=52
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

METRICS="${METRICS:-hamming l2 cka}"

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
