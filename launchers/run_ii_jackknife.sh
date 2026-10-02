#!/bin/bash
#SBATCH --job-name=ii_jackknife
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mmahaut/projects/exps/ii_jackknife/%j.out
#SBATCH --error=/home/mmahaut/projects/exps/ii_jackknife/%j.err

source ~/.bashrc
conda activate emecom

cd /home/mmahaut/projects/deepseek-hidden-states

# Create output directories
mkdir -p /home/mmahaut/projects/exps/ii_jackknife
mkdir -p imagenet_same_class_dino2
mkdir -p imagenet_same_class_igpt

echo "=== Step 1: Generate same-class image pairs (2500 for 5x500 jackknife) ==="
python II_mateo.py data-selection --n-files=2500

echo "=== Step 2: Extract DinoV2 features ==="
python II_mateo.py data-pickle \
    ./imagenet_same_class.txt \
    ./imagenet_same_class_dino2 \
    /projects/colt/imagenet21k_resized/imagenet21k_train/ \
    dino2 \
    --batch-size=16 \
    --rank=0 \
    --n-ranks=1

echo "=== Step 3: Extract iGPT features ==="
python II_mateo.py data-pickle \
    ./imagenet_same_class.txt \
    ./imagenet_same_class_igpt \
    /projects/colt/imagenet21k_resized/imagenet21k_train/ \
    igpt \
    --batch-size=8 \
    --rank=0 \
    --n-ranks=1

echo "=== Step 4: Compute II for DinoV2 (5 jackknife replicas of 500 samples) ==="
for replica in 0 1 2 3 4; do
    echo "DinoV2 replica $replica"
    python II_mateo.py main \
        ./imagenet_same_class_dino2 \
        --output-folder=./imagenet_same_class_dino2/replica_${replica} \
        --n-files=500 \
        --rank=$replica \
        --average-tokens \
        --diagonal-only
done

echo "=== Step 5: Compute II for iGPT (5 jackknife replicas of 500 samples) ==="
for replica in 0 1 2 3 4; do
    echo "iGPT replica $replica"
    python II_mateo.py main \
        ./imagenet_same_class_igpt \
        --output-folder=./imagenet_same_class_igpt/replica_${replica} \
        --n-files=500 \
        --rank=$replica \
        --average-tokens \
        --diagonal-only
done

echo "=== Done ==="
