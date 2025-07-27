#!/bin/bash
# Sequential script to run all comparisons (for testing or if job array fails)

# Activate environment
source /home/mascaretti/myjax/bin/activate

# Base paths
DEEPSEEK_PATH="/home/laios/activations/flickr/deepseek_all"
IMAGE_BASE="/home/laios/image-information-imbalance"
OUTPUT_BASE="/home/laios/image-information-imbalance/results"
REPO_DIR="/home/mascaretti/deepseek-hidden-states"

# Create directories
mkdir -p ${OUTPUT_BASE}/bootstrap
mkdir -p ${REPO_DIR}/plots

cd ${REPO_DIR}

echo "Starting Information Imbalance comparisons..."
echo "============================================"

# Priority 1: DeepSeek vs CLIP text
echo "[1/3] DeepSeek vs CLIP text"
python compute_asymmetry_unified.py \
    --comparison deepseek_clip_text \
    --model_a deepseek \
    --model_b clip_text \
    --input_path_A "${DEEPSEEK_PATH}" \
    --input_path_B "${IMAGE_BASE}/openai-clip-vit-large-patch14" \
    --output_folder "${OUTPUT_BASE}/deepseek-clip-text-ranks"

# Priority 2: CLIP text vs CLIP image
echo "[2/3] CLIP text vs CLIP image"
python compute_asymmetry_unified.py \
    --comparison clip_text_image \
    --model_a clip_text \
    --model_b clip_image \
    --input_path_A "${IMAGE_BASE}/openai-clip-vit-large-patch14" \
    --input_path_B "${IMAGE_BASE}/openai-clip-vit-large-patch14" \
    --output_folder "${OUTPUT_BASE}/clip-text-image-ranks"

# Priority 3: DinoV2 vs CLIP image
echo "[3/3] DinoV2 vs CLIP image"
python compute_asymmetry_unified.py \
    --comparison dino_clip_image \
    --model_a dino \
    --model_b clip_image \
    --input_path_A "${IMAGE_BASE}/facebook-dinov2-large" \
    --input_path_B "${IMAGE_BASE}/openai-clip-vit-large-patch14" \
    --output_folder "${OUTPUT_BASE}/dino-clip-image-ranks"

echo "============================================"
echo "All comparisons completed!"
echo "Results saved to: ${OUTPUT_BASE}"

# Optional: Run bootstrap and plotting
echo ""
echo "To run bootstrap analysis:"
echo "python bootstrap_analysis.py --data_path ${OUTPUT_BASE}/[comparison]-ranks --output_path ${OUTPUT_BASE}/bootstrap ..."
echo ""
echo "To generate plots:"
echo "python plot_results.py --data_path ${OUTPUT_BASE}/bootstrap --output_path ${REPO_DIR}/plots ..."
