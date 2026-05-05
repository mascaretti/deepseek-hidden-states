#!/bin/bash
#SBATCH --job-name=mean_clip_max77
#SBATCH --output=logs/mean_clip_max77_%j.out
#SBATCH --error=logs/mean_clip_max77_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it

# Mean-pool the max_length=77 CLIP-text chunks using the same window as the
# published CLIP-text mean (--last_k 20, no --skip-first, matching the legacy
# DEEPSEEK_TOKENS=20 regime documented in run_preprocess_mean.sbatch task 6).
# Only the underlying extraction seq length differs (30 vs 77).

source /home/$USER/myjax/bin/activate
cd /home/$USER/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"

INPUT="${IMAGE_BASE}/openai-clip-vit-large-patch14_max77"
OUTPUT="${MEAN_BASE}/openai-clip-vit-large-patch14-text-max77"
PREFIX="text-chunk_"

echo "========================================="
echo "Mean-pool CLIP-text (max_length=77)"
echo "Input:  ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "========================================="

python preprocess_mean.py \
    --input_path "${INPUT}" \
    --output_path "${OUTPUT}" \
    --prefix "${PREFIX}" \
    --last_k 20 \
    --verify

STATUS=$?
echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
if [ ${STATUS} -eq 0 ]; then
    echo "Output: $(ls ${OUTPUT} | wc -l) files, $(du -sh ${OUTPUT} | cut -f1)"
fi
exit ${STATUS}
