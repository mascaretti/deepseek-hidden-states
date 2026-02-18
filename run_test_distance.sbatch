#!/bin/bash
#SBATCH --job-name=dist_test
#SBATCH --output=logs/dist_test_%j.out
#SBATCH --error=logs/dist_test_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it

# Test 4 distance strategies for II concentration of measure.
# Single job: runs Hamming, Centered+Norm L2, Centered L1, PCA+L2
# on DeepSeek→DinoV2 and DeepSeek→ImageGPT.

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
DEEPSEEK_MEAN="/home/laios/activations/flickr/deepseek_windsorised_mean"
DINOV2_MEAN="${MEAN_BASE}/facebook-dinov2-large"
IGPT_MEAN="${MEAN_BASE}/openai-imagegpt-large"

OUTPUT_PREFIX="results/distance_test"

echo "========================================="
echo "Distance strategy test"
echo "Job ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "A: ${DEEPSEEK_MEAN} (layer 52)"
echo "B_dino: ${DINOV2_MEAN}"
echo "B_igpt: ${IGPT_MEAN}"
echo "========================================="

python test_distance_strategies.py \
    --input_path_A "${DEEPSEEK_MEAN}" \
    --input_path_B_dino "${DINOV2_MEAN}" \
    --input_path_B_igpt "${IGPT_MEAN}" \
    --layer_A 52 \
    --subset_size 5000 \
    --pca_components 100 \
    --output_prefix "${OUTPUT_PREFIX}"

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"

exit ${STATUS}
