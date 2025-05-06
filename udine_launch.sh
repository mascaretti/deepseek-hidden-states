#!/bin/bash

# Parameters
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --nodes=1
#SBATCH --job-name=captions
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

# Config script
script_name="$1"

# setup and run
module load cuda11.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
eval "$(/home/mascaretti/miniforge3/bin/conda shell.bash hook)"
srun conda run -n clipenv python -m "$script_name"
