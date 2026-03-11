#!/bin/bash
#SBATCH --job-name=ii_sweep
#SBATCH --output=logs/ii_sweep_%A_%a.out
#SBATCH --error=logs/ii_sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1-10%1

# Sweep ALL target layers for DeepSeek <-> ImageGPT, DeepSeek <-> DinoV2,
# Llama <-> ImageGPT, Llama <-> DinoV2, and CLIP-Text <-> CLIP-Image.
# 10 jobs total: one per direction. Data is loaded once per job, then all target
# layers are computed in a loop (no redundant I/O).

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
DEEPSEEK_MEAN="/home/laios/activations/flickr/deepseek_windsorised_mean"
LLAMA_MEAN="${MEAN_BASE}/llama"
DINOV2_MEAN="${MEAN_BASE}/facebook-dinov2-large"
IGPT_MEAN="${MEAN_BASE}/openai-imagegpt-large"
CLIP_IMAGE_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-image"
CLIP_TEXT_MEAN="${MEAN_BASE}/openai-clip-vit-large-patch14-text"

OUTPUT_BASE="${IMAGE_BASE}/results-mean/sweep"

case ${SLURM_ARRAY_TASK_ID} in
    # DeepSeek[varied] -> sweep all ImageGPT layers
    1)
        NAME="deepseek_igpt"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        # All DeepSeek layers (1-60)
        LAYERS_A="$(seq 1 60)"
        ;;
    # ImageGPT[varied] -> sweep all DeepSeek layers
    2)
        NAME="igpt_deepseek"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        # All ImageGPT layers (1-48)
        LAYERS_A="$(seq 1 48)"
        ;;
    # DeepSeek[varied] -> sweep all DinoV2 layers
    3)
        NAME="deepseek_dino"
        INPUT_A="${DEEPSEEK_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        # All DeepSeek layers (1-60)
        LAYERS_A="$(seq 1 60)"
        ;;
    # DinoV2[varied] -> sweep all DeepSeek layers
    4)
        NAME="dino_deepseek"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${DEEPSEEK_MEAN}"
        # All DinoV2 layers (1-24)
        LAYERS_A="$(seq 1 24)"
        ;;
    # Llama[varied] -> sweep all ImageGPT layers
    5)
        NAME="llama_igpt"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${IGPT_MEAN}"
        # All Llama layers (1-32)
        LAYERS_A="$(seq 1 32)"
        ;;
    # ImageGPT[varied] -> sweep all Llama layers
    6)
        NAME="igpt_llama"
        INPUT_A="${IGPT_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        # All ImageGPT layers (1-48)
        LAYERS_A="$(seq 1 48)"
        ;;
    # Llama[varied] -> sweep all DinoV2 layers
    7)
        NAME="llama_dino"
        INPUT_A="${LLAMA_MEAN}"
        INPUT_B="${DINOV2_MEAN}"
        # All Llama layers (1-32)
        LAYERS_A="$(seq 1 32)"
        ;;
    # DinoV2[varied] -> sweep all Llama layers
    8)
        NAME="dino_llama"
        INPUT_A="${DINOV2_MEAN}"
        INPUT_B="${LLAMA_MEAN}"
        # All DinoV2 layers (1-24)
        LAYERS_A="$(seq 1 24)"
        ;;
    # CLIP-Text[varied] -> sweep all CLIP-Image layers
    9)
        NAME="clip_text_clip_image"
        INPUT_A="${CLIP_TEXT_MEAN}"
        INPUT_B="${CLIP_IMAGE_MEAN}"
        # All CLIP-Text layers (1-12)
        LAYERS_A="$(seq 1 12)"
        ;;
    # CLIP-Image[varied] -> sweep all CLIP-Text layers
    10)
        NAME="clip_image_clip_text"
        INPUT_A="${CLIP_IMAGE_MEAN}"
        INPUT_B="${CLIP_TEXT_MEAN}"
        # All CLIP-Image layers (1-24)
        LAYERS_A="$(seq 1 24)"
        ;;
    *)
        echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "========================================="
echo "II sweep: ${NAME}"
echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "A: ${INPUT_A}"
echo "B: ${INPUT_B}"
echo "Target layers: ${LAYERS_A}"
echo "========================================="

METRICS="${METRICS:-l2}"

python compute_ii_mean.py \
    --input_path_A "${INPUT_A}" \
    --input_path_B "${INPUT_B}" \
    --output_folder "${OUTPUT_BASE}/${NAME}" \
    --layer_A ${LAYERS_A} \
    --mean_center \
    --l2_normalize \
    --metrics ${METRICS}

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"

exit ${STATUS}
