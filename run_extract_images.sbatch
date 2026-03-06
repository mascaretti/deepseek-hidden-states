#!/bin/bash
#SBATCH --job-name=extract_img
#SBATCH --output=logs/extract_img_%A_%a.out
#SBATCH --error=logs/extract_img_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=0-1%1

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"

MODELS=(
    "openai/imagegpt-small"
    "openai/imagegpt-medium"
    "openai/imagegpt-large"
    "facebook/dinov2-large"
    "facebook/dinov2-base"
    "facebook/dinov2-small"
    "openai/clip-vit-large-patch14"
)

MODEL_NAME="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_DIR="${IMAGE_BASE}/$(echo ${MODEL_NAME} | tr '/' '-')"

echo "========================================="
echo "Extract image hidden states: ${MODEL_NAME}"
echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="

if [ -d "${OUTPUT_DIR}" ] && [ "$(ls ${OUTPUT_DIR}/chunk_*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Activations already exist at ${OUTPUT_DIR} ($(ls ${OUTPUT_DIR}/chunk_*.pkl | wc -l) chunks). Skipping."
    exit 0
fi

python extract_hidden_images_states.py \
    --model "${MODEL_NAME}" \
    --output "${OUTPUT_DIR}"

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
if [ ${STATUS} -eq 0 ]; then
    echo "Output: $(ls ${OUTPUT_DIR} | wc -l) files, $(du -sh ${OUTPUT_DIR} | cut -f1)"
fi

exit ${STATUS}
