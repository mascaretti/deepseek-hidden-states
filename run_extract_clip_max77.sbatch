#!/bin/bash
#SBATCH --job-name=extract_clip_max77
#SBATCH --output=logs/extract_clip_max77_%j.out
#SBATCH --error=logs/extract_clip_max77_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it

source /home/$USER/mysglang/bin/activate
cd /home/$USER/deepseek-hidden-states
mkdir -p logs

# Defaults are smoke-test (20 images). Full run:
#   sbatch --export=ALL,SUBSET_SIZE=5000 run_extract_clip_max77.sbatch
SUBSET_SIZE="${SUBSET_SIZE:-20}"
MAX_LENGTH="${MAX_LENGTH:-77}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_max77}"

echo "========================================="
echo "CLIP-text extraction (max_length=${MAX_LENGTH})"
echo "Subset size: ${SUBSET_SIZE}"
echo "Output suffix: ${OUTPUT_SUFFIX}"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "========================================="

python extract_hidden_clip.py \
    --subset_size "${SUBSET_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --output_suffix "${OUTPUT_SUFFIX}"

STATUS=$?
echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
exit ${STATUS}
