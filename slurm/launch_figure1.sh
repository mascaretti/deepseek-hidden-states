#!/bin/bash

#SBATCH --job-name=cka
#SBATCH --output=logs/cka.out
#SBATCH --error=logs/cka.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G

source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states
mkdir -p logs

python notebooks/figure1_compute.py \
       	--p_values 1000 100000 \
	--n_reps 10 \
        --n_samples 2500 \
        --output notebooks/figure1_data_p1e3_p1e5

