#!/bin/bash

#SBATCH --job-name=twoNN
#SBATCH --output=logs/twoNN.out
#SBATCH --error=logs/twoNN.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G

module load conda
/cm/shared/apps/miniconda/bin/activate activate mydadapy

# Use all allocated CPUs
python3 test_twoNN.py
