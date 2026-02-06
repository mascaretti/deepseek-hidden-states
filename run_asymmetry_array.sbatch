#!/bin/bash
#SBATCH --job-name=inf_imb_array
#SBATCH --output=logs/inf_imb_%A_%a.out
#SBATCH --error=logs/inf_imb_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=1-5%1  # Three priority comparisons

# SLURM job array for running all information imbalance comparisons
# Array jobs will run in parallel on different GPUs

# Load modules and activate environment
source /home/mascaretti/myjax/bin/activate

# Base paths
DEEPSEEK_PATH="/home/laios/activations/flickr/deepseek_all"
IMAGE_BASE="/home/laios/image-information-imbalance"
OUTPUT_BASE="/home/laios/image-information-imbalance/results"

# Create output directories
mkdir -p ${OUTPUT_BASE}
mkdir -p logs

# Define comparisons based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        # Priority 1: DeepSeek vs CLIP text
        echo "Running DeepSeek vs CLIP text comparison"
        COMPARISON="deepseek_clip_text"
        MODEL_A="deepseek"
        MODEL_B="clip_text"
        INPUT_A="${DEEPSEEK_PATH}"
        INPUT_B="${IMAGE_BASE}/openai-clip-vit-large-patch14"
        OUTPUT="${OUTPUT_BASE}/deepseek-clip-text-ranks"
        ;;
    2)
        # Priority 2: CLIP text vs CLIP image
        echo "Running CLIP text vs CLIP image comparison"
        COMPARISON="clip_text_image"
        MODEL_A="clip_text"
        MODEL_B="clip_image"
        INPUT_A="${IMAGE_BASE}/openai-clip-vit-large-patch14"
        INPUT_B="${IMAGE_BASE}/openai-clip-vit-large-patch14"
        OUTPUT="${OUTPUT_BASE}/clip-text-image-ranks"
        ;;
    3)
        # Priority 3: DinoV2 vs CLIP image
        echo "Running DinoV2 vs CLIP image comparison"
        COMPARISON="dino_clip_image"
        MODEL_A="dino"
        MODEL_B="clip_image"
        INPUT_A="${IMAGE_BASE}/facebook-dinov2-large"
        INPUT_B="${IMAGE_BASE}/openai-clip-vit-large-patch14"
        OUTPUT="${OUTPUT_BASE}/dino-clip-image-ranks"
        ;;
    4)
	# Priority 4: DeepSeek vs CLIP image
	echo "Running DeepSeek vs CLIP image comparison"
	COMPARISON="deepseek_clip_image"
	MODEL_A="deepseek"
	MODEL_B="clip_image"
	INPUT_A="${DEEPSEEK_PATH}"
	INPUT_B="${IMAGE_BASE}/openai-clip-vit-large-patch14"
	OUTPUT="${OUTPUT_BASE}/deepseek-clip-image-ranks"
	;;
    5)
	# Priority 5: image-gpt vs dinov2
	echo "Running image-gpt vs dinov2 comparison"
	COMPARISON="igpt_dino"
	MODEL_A="imagegpt"
	MODEL_B="dino"
	INPUT_A="${IMAGE_BASE}/openai-imagegpt-large"
	INPUT_B="${IMAGE_BASE}/facebook-dinov2-large"
	OUTPUT="${OUTPUT_BASE}/imagegpt-dinov2-ranks"
	;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

# Create output directory
mkdir -p ${OUTPUT}

# Log job information
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Comparison: ${COMPARISON}"
echo "Model A: ${MODEL_A} (${INPUT_A})"
echo "Model B: ${MODEL_B} (${INPUT_B})"
echo "Output: ${OUTPUT}"
echo "========================================="

# Change to repository directory
cd /home/mascaretti/deepseek-hidden-states

# Run the unified script
python compute_asymmetry_unified.py \
    --comparison ${COMPARISON} \
    --model_a ${MODEL_A} \
    --model_b ${MODEL_B} \
    --input_path_A "${INPUT_A}" \
    --input_path_B "${INPUT_B}" \
    --output_folder "${OUTPUT}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
    
    # Run bootstrap analysis immediately after
    echo "Starting bootstrap analysis..."
   4
    # Determine number of layers for model B
    case ${MODEL_B} in
        "clip_text")
            N_LAYERS=12 # CLIP text has layers 0-12 (13 total)
            ;;
        "clip_image")
            N_LAYERS=24  # CLIP image has layers 0-24 (25 total)
            ;;
        "dino")
            N_LAYERS=24  # DinoV2 has layers 0-24 (25 total)
            ;;
        *)
            N_LAYERS=24  # Default
            ;;
    esac
    
    python bootstrap_analysis.py \
        --data_path "${OUTPUT}" \
        --output_path "${OUTPUT_BASE}/bootstrap" \
        --n_layers ${N_LAYERS} \
        --model_a ${MODEL_A} \
        --model_b ${MODEL_B} \
        --n_replicas 500 \
        --bootstrap_size 500 \
        --sample_size 5000 \
        --layer 52
    
    echo "Bootstrap analysis completed"
else
    echo "Job failed with exit code: $?"
    exit 1
fi

echo "========================================="
echo "All processing completed at: $(date)"
