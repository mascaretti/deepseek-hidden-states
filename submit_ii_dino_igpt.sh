#!/bin/bash
#SBATCH --job-name=ii_dino_igpt
#SBATCH --mem=200G
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --output=/home/mmahaut/projects/exps/ii_jackknife/%j.out
#SBATCH --error=/home/mmahaut/projects/exps/ii_jackknife/%j.err
#SBATCH --nice=10
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate emecom

cd /home/mmahaut/projects/deepseek-hidden-states

echo "Starting II computation for DinoV2 vs iGPT"
echo "Using normalized L2 distance with token averaging"
echo "Jackknife: 5 replicas x 500 samples"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""

python run_ii_dino_igpt.py

echo "=== Done ==="
