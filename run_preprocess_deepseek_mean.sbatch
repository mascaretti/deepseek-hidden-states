#!/bin/bash
#SBATCH --job-name=ds_mean
#SBATCH --output=logs/ds_mean_%j.out
#SBATCH --error=logs/ds_mean_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

# Mean-pool DeepSeek activations over the token dimension.
# No GPU needed — this is pure CPU work.

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

INPUT_PATH="/home/laios/activations/flickr/deepseek_all"
OUTPUT_PATH="/home/laios/activations/flickr/deepseek_mean"

echo "========================================="
echo "DeepSeek mean-pooling preprocessing"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Input:  ${INPUT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "========================================="

python preprocess_deepseek_mean.py \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --verify

echo ""
echo "Finished at: $(date)"
echo "Output files:"
ls -lh "${OUTPUT_PATH}" | head -5
echo "..."
ls -lh "${OUTPUT_PATH}" | wc -l
echo "total files"
du -sh "${OUTPUT_PATH}"
