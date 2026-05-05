#!/bin/bash
#SBATCH --job-name=ii_clip_max30
#SBATCH --output=logs/ii_clip_max30_%A_%a.out
#SBATCH --error=logs/ii_clip_max30_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1-2%1

# II for the published max=30 CLIP-text ↔ CLIP-image. Mirrors run_ii_mean.sbatch
# tasks 17/18 with optional KEEP_INDICES filter and SUFFIX to disambiguate the
# output folder. Used for two of the four rebuttal cells:
#   - Cell A (recheck published): SUFFIX=_recheck, no KEEP_INDICES.
#       sbatch --export=ALL,SUFFIX=_recheck run_ii_clip_max30.sbatch
#   - Cell C (filter to ≤77 BPE): KEEP_INDICES + SUFFIX=_filtered.
#       sbatch --export=ALL,KEEP_INDICES=data/clip_max77_keep.npy,SUFFIX=_filtered \
#              run_ii_clip_max30.sbatch

source /home/$USER/myjax/bin/activate
cd /home/$USER/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
CLIP_IMAGE_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-image"
CLIP_TEXT_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-text"
OUTPUT_BASE="${IMAGE_BASE}/results-mean"

case ${SLURM_ARRAY_TASK_ID} in
    1)
        NAME="clip_text_clip_image"
        INPUT_A="${CLIP_TEXT_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        LAYER_A=11
        ;;
    2)
        NAME="clip_image_clip_text"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${CLIP_TEXT_MEAN}"
        LAYER_A=20
        ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

SUFFIX="${SUFFIX:-}"
KEEP_FLAG=""
if [ -n "${KEEP_INDICES}" ]; then
    KEEP_FLAG="--keep_indices ${KEEP_INDICES}"
fi

echo "========================================="
echo "II (max30 filtered): ${NAME}${SUFFIX}"
echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "A: ${INPUT_A} (layer ${LAYER_A})"
echo "B: ${INPUT_B}"
echo "Keep indices: ${KEEP_INDICES:-<none>}"
echo "Started at: $(date)"
echo "========================================="

METRICS="${METRICS:-l2}"

python compute_ii_mean.py \
    --input_path_A "${INPUT_A}" \
    --input_path_B "${INPUT_B}" \
    --output_folder "${OUTPUT_BASE}/${NAME}${SUFFIX}" \
    --layer_A ${LAYER_A} \
    --mean_center \
    --l2_normalize \
    --metrics ${METRICS} \
    ${KEEP_FLAG}

STATUS=$?
echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
exit ${STATUS}
