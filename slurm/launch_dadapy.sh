#!/bin/bash

#SBATCH --job-name=twoNN
#SBATCH --output=logs/twoNN.out
#SBATCH --error=logs/twoNN.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G

module load conda
/cm/shared/apps/miniconda/bin/activate twonn 

# Use all allocated CPUs
python3 test_twoNN.py \
    --input_path /home/laios/image-information-imbalance/windsorised_mean/openai-imagegpt-large-full \
    --output_prefix results/twoNN_igpt \
    --subset_size 5000 \
    --seed 42
