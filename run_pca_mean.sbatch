#!/bin/bash
#SBATCH --job-name=pca_mean
#SBATCH --output=logs/pca_mean_%A_%a.out
#SBATCH --error=logs/pca_mean_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1-10%2

# Per-layer PCA of the mean-pooled activations, one model per array task.
# No --gres=gpu here (unlike run_ii_mean.sbatch): compute_pca_mean.py is pure
# LAPACK on CPU, so a GPU would sit idle and only lengthen the queue wait.
# Cost is dominated by DeepSeek (task 1): 62 layers x eigh of a 5000x5000 Gram.

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
PROJECT_K="${PROJECT_K:-100}"
NPZ_BASE="${IMAGE_BASE}/pca-mean"
PROJ_BASE="${IMAGE_BASE}/pca_mean_k${PROJECT_K}"

case ${SLURM_ARRAY_TASK_ID} in
    1)  NAME="deepseek";          INPUT="/home/laios/activations/flickr/deepseek_windsorised_mean" ;;
    2)  NAME="llama";             INPUT="${MEAN_BASE}/llama" ;;
    3)  NAME="dinov2-large";      INPUT="${MEAN_BASE}/facebook-dinov2-large" ;;
    4)  NAME="dinov2-base";       INPUT="${MEAN_BASE}/facebook-dinov2-base" ;;
    5)  NAME="dinov2-small";      INPUT="${MEAN_BASE}/facebook-dinov2-small" ;;
    6)  NAME="imagegpt-large";    INPUT="${MEAN_BASE}/openai-imagegpt-large" ;;
    7)  NAME="imagegpt-medium";   INPUT="${MEAN_BASE}/openai-imagegpt-medium" ;;
    8)  NAME="clip-image";        INPUT="${MEAN_BASE}/openai-clip-vit-large-patch14-image" ;;
    9)  NAME="clip-text";         INPUT="${MEAN_BASE}/openai-clip-vit-large-patch14-text" ;;
    10) NAME="clip-text-max77";   INPUT="${MEAN_BASE}/openai-clip-vit-large-patch14-text-max77" ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

# KEEP_INDICES / SUFFIX follow the convention of run_ii_clip_max*.sbatch.
# NOTE: projections written under a KEEP_INDICES run are already filtered —
# do not pass --keep_indices again to compute_ii_mean.py on that folder.
if [ -n "${KEEP_INDICES}" ]; then
    KEEP_FLAG="--keep_indices ${KEEP_INDICES}"
fi

OUTPUT_NPZ="${NPZ_BASE}/${NAME}${SUFFIX}.npz"
PROJECT_OUT="${PROJ_BASE}/${NAME}${SUFFIX}"

echo "========================================="
echo "PCA: ${NAME}"
echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Input:   ${INPUT}"
echo "NPZ:     ${OUTPUT_NPZ}"
echo "Project: ${PROJECT_OUT} (k=${PROJECT_K})"
echo "Keep:    ${KEEP_INDICES:-<none>}"
echo "========================================="

python compute_pca_mean.py \
    --input_path "${INPUT}" \
    --output_npz "${OUTPUT_NPZ}" \
    --project_k "${PROJECT_K}" \
    --project_out "${PROJECT_OUT}" \
    ${KEEP_FLAG}

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
if [ ${STATUS} -eq 0 ]; then
    echo "Output: $(du -sh ${PROJECT_OUT} | cut -f1) projections, $(du -h ${OUTPUT_NPZ} | cut -f1) npz"
fi

exit ${STATUS}
