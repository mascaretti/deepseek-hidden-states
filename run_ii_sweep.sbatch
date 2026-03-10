#!/bin/bash
#SBATCH --job-name=ii_sweep
#SBATCH --output=logs/ii_sweep_%A_%a.out
#SBATCH --error=logs/ii_sweep_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --array=1-46%1

# Sweep multiple target layers for DeepSeek <-> ImageGPT and DeepSeek <-> DinoV2.
# Builds (partial) layer x layer II matrices.
# To run all layers instead of every k-th, regenerate the LAYERS arrays with stride 1.

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

IMAGE_BASE="/home/laios/image-information-imbalance"
MEAN_BASE="${IMAGE_BASE}/windsorised_mean"
DEEPSEEK_MEAN="/home/laios/activations/flickr/deepseek_windsorised_mean"
DINOV2_MEAN="${MEAN_BASE}/facebook-dinov2-large"
IGPT_MEAN="${MEAN_BASE}/openai-imagegpt-large"

OUTPUT_BASE="${IMAGE_BASE}/results-mean/sweep"

# --- Target layers for each direction ---
# DeepSeek (60 layers): every 5th
DS_LAYERS=(5 10 15 20 25 30 35 40 45 50 55)
# ImageGPT (48 layers): every 4th
IGPT_LAYERS=(4 8 12 16 20 24 28 32 36 40 44 48)
# DinoV2 (24 layers): every 2nd
DINO_LAYERS=(2 4 6 8 10 12 14 16 18 20 22 24)

# --- Map array task ID to (direction, layer) ---
# Tasks  1-11: deepseek[varied] -> sweep igpt     (11 tasks)
# Tasks 12-23: igpt[varied]     -> sweep deepseek (12 tasks)
# Tasks 24-34: deepseek[varied] -> sweep dino     (11 tasks)
# Tasks 35-46: dino[varied]     -> sweep deepseek (12 tasks)

N_DS=${#DS_LAYERS[@]}      # 11
N_IGPT=${#IGPT_LAYERS[@]}  # 12
N_DINO=${#DINO_LAYERS[@]}  # 12

ID=${SLURM_ARRAY_TASK_ID}

if [ ${ID} -le ${N_DS} ]; then
    # Direction 0: deepseek[layer] -> sweep igpt
    IDX=$((ID - 1))
    LAYER_A=${DS_LAYERS[$IDX]}
    NAME="sweep/deepseek_igpt/ds_layer_${LAYER_A}"
    INPUT_A="${DEEPSEEK_MEAN}"
    INPUT_B="${IGPT_MEAN}"

elif [ ${ID} -le $((N_DS + N_IGPT)) ]; then
    # Direction 1: igpt[layer] -> sweep deepseek
    IDX=$((ID - N_DS - 1))
    LAYER_A=${IGPT_LAYERS[$IDX]}
    NAME="sweep/igpt_deepseek/igpt_layer_${LAYER_A}"
    INPUT_A="${IGPT_MEAN}"
    INPUT_B="${DEEPSEEK_MEAN}"

elif [ ${ID} -le $((N_DS + N_IGPT + N_DS)) ]; then
    # Direction 2: deepseek[layer] -> sweep dino
    IDX=$((ID - N_DS - N_IGPT - 1))
    LAYER_A=${DS_LAYERS[$IDX]}
    NAME="sweep/deepseek_dino/ds_layer_${LAYER_A}"
    INPUT_A="${DEEPSEEK_MEAN}"
    INPUT_B="${DINOV2_MEAN}"

elif [ ${ID} -le $((N_DS + N_IGPT + N_DS + N_DINO)) ]; then
    # Direction 3: dino[layer] -> sweep deepseek
    IDX=$((ID - N_DS - N_IGPT - N_DS - 1))
    LAYER_A=${DINO_LAYERS[$IDX]}
    NAME="sweep/dino_deepseek/dino_layer_${LAYER_A}"
    INPUT_A="${DINOV2_MEAN}"
    INPUT_B="${DEEPSEEK_MEAN}"

else
    echo "Invalid array task ID: ${ID}"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"

# Skip if already computed
if [ -d "${OUTPUT_DIR}/l2_centered_normalized" ] && \
   [ "$(ls ${OUTPUT_DIR}/l2_centered_normalized/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Results already exist at ${OUTPUT_DIR}. Skipping."
    exit 0
fi

echo "========================================="
echo "II sweep: ${NAME}"
echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "A: ${INPUT_A} (layer ${LAYER_A})"
echo "B: ${INPUT_B}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="

METRICS="${METRICS:-hamming l2 cka}"

python compute_ii_mean.py \
    --input_path_A "${INPUT_A}" \
    --input_path_B "${INPUT_B}" \
    --output_folder "${OUTPUT_DIR}" \
    --layer_A ${LAYER_A} \
    --mean_center \
    --l2_normalize \
    --metrics ${METRICS}

STATUS=$?

echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
if [ ${STATUS} -eq 0 ]; then
    for m in ${METRICS}; do
        DIR="${OUTPUT_DIR}/${m}"
        [ "${m}" = "l2" ] && DIR="${OUTPUT_DIR}/l2_centered_normalized"
        [ -d "${DIR}" ] && echo "Output (${m}): $(ls ${DIR} | wc -l) layer files"
    done
fi

exit ${STATUS}
