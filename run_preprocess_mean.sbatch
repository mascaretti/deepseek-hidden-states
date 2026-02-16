#!/bin/bash
#SBATCH --job-name=mean_pool
#SBATCH --output=logs/mean_pool_%A_%a.out
#SBATCH --error=logs/mean_pool_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1-6%1

# Mean-pool activations over the token dimension.
# One model per array task, %1 = one job at a time (disk-bound).
# No GPU needed. Estimated total wall time: ~6-8 hours.

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
DEEPSEEK_TOKENS=20

case ${SLURM_ARRAY_TASK_ID} in
    1)
        NAME="deepseek"
        INPUT="/home/laios/activations/flickr/deepseek_all"
        OUTPUT="/home/laios/activations/flickr/deepseek_windsorised_mean"
        PREFIX="chunk_"
        ;;
    2)
        NAME="llama"
        INPUT="/home/laios/llama_flickr_cpt_all"
        OUTPUT="${MEAN_BASE}/llama"
        PREFIX="chunk_"
	FLAGS="--skip-first --last_k=20"
        ;;
    3)
        NAME="dinov2"
        INPUT="${IMAGE_BASE}/facebook-dinov2-large"
        OUTPUT="${MEAN_BASE}/facebook-dinov2-large"
        PREFIX="chunk_"
        ;;
    4)
        NAME="imagegpt"
        INPUT="${IMAGE_BASE}/openai-imagegpt-large"
        OUTPUT="${MEAN_BASE}/openai-imagegpt-large"
        PREFIX="chunk_"
        ;;
    5)
        NAME="clip_image"
        INPUT="${IMAGE_BASE}/openai-clip-vit-large-patch14"
        OUTPUT="${MEAN_BASE}/openai-clip-vit-large-patch14-image"
        PREFIX="image-chunk_"
        ;;
    6)
        NAME="clip_text"
        INPUT="${IMAGE_BASE}/openai-clip-vit-large-patch14"
        OUTPUT="${MEAN_BASE}/openai-clip-vit-large-patch14-text"
        PREFIX="text-chunk_"
        ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "========================================="
echo "Mean-pooling: ${NAME}"
echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Input:  ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Prefix: ${PREFIX}"
echo "DeepSeek Tokens: ${DEEPSEEK_TOKENS}"
echo "========================================="

python preprocess_mean.py \
    --input_path "${INPUT}" \
    --output_path "${OUTPUT}" \
    --prefix "${PREFIX}" \
    --last_k "${DEEPSEEK_TOKENS}"\
    --verify \
    ${FLAGS}

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
if [ ${STATUS} -eq 0 ]; then
    echo "Output: $(ls ${OUTPUT} | wc -l) files, $(du -sh ${OUTPUT} | cut -f1)"
fi

exit ${STATUS}
