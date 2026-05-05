#!/bin/bash
#SBATCH --job-name=ii_deepseek_dino_filtered
#SBATCH --output=logs/ii_deepseek_dino_filtered_%A_%a.out
#SBATCH --error=logs/ii_deepseek_dino_filtered_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1%1

# DeepSeek↔DinoV2 II on the ≤77-BPE filtered subset, mirroring published
# run_ii_mean.sbatch task 3 (deepseek_dino, LAYER_A=36 = 60% relative depth,
# matching paper main.tex:405). Adds --keep_indices to restrict to the 3,330
# images whose 5-caption concat fits in CLIP's 77-token cap.
#
# Submit:
#   sbatch --export=ALL,KEEP_INDICES=data/clip_max77_keep.npy,SUFFIX=_filtered \
#          run_ii_deepseek_dino_filtered.sbatch

source /home/$USER/myjax/bin/activate
cd /home/$USER/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
DEEPSEEK_MEAN="/home/laios/activations/flickr/deepseek_windsorised_mean"
DINOV2_MEAN="${MEAN_BASE}/facebook-dinov2-large"
OUTPUT_BASE="${IMAGE_BASE}/results-mean"

case ${SLURM_ARRAY_TASK_ID} in
    1)
        NAME="deepseek_dino"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        LAYER_A=36
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
echo "II (deepseek_dino filtered): ${NAME}${SUFFIX}"
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
