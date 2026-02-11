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
#SBATCH --array=1-7%1

# Compute information imbalance from mean-pooled activations.
# One comparison per array task, %1 = sequential (shared GPU).

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/mean"
DEEPSEEK_MEAN="/home/laios/activations/flickr/deepseek_mean"
LLAMA_MEAN="${MEAN_BASE}/llama"
DINOV2_MEAN="${MEAN_BASE}/facebook-dinov2-large"
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

python compute_ii_mean.py \
    --input_path_A "${INPUT_A}" \
    --input_path_B "${INPUT_B}" \
    --output_folder "${OUTPUT_BASE}/${NAME}" \
    --layer_A ${LAYER_A}

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
if [ ${STATUS} -eq 0 ]; then
    echo "Output: $(ls ${OUTPUT_BASE}/${NAME} | wc -l) layer files"
fi

exit ${STATUS}
