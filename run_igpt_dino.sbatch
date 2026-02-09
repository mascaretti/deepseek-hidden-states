#!/bin/bash
#SBATCH --job-name=inf_imb_array
#SBATCH --output=logs/inf_imb.out
#SBATCH --error=logs/inf_imb.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

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
# Priority 5: image-gpt vs dinov2
echo "Running image-gpt vs dinov2 comparison"
COMPARISON="dino_igpt"
MODEL_A="dino"
MODEL_B="igpt"
INPUT_A="${IMAGE_BASE}/facebook-dinov2-large"
INPUT_B="${IMAGE_BASE}/openai-imagegpt-large"
OUTPUT="${OUTPUT_BASE}/imagegpt-dinov2-ranks"

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
