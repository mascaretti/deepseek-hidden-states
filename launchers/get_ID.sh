#!/bin/bash

# Parameters
#SBATCH --mem=500G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --error=/home/mmahaut/projects/exps/bin_reps/%j.err
#SBATCH --output=/home/mmahaut/projects/exps/bin_reps/%j.out
#SBATCH --exclude=node044
#SBATCH --nice=10
#SBATCH --job-name=bin_reps

source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate emecom
which python
cd ~/projects/emecom/

# poetry run python /home/mmahaut/projects/deepseek-hidden-states/II_mateo.py main "./imagenet_same_class" --output-folder="./imagenet_same_class" --n-tokens-list=1 --n-tokens-list=2 --n-tokens-list=4 --n-tokens-list=8 --n-tokens-list=16 --n-tokens-list=32
poetry run python /home/mmahaut/projects/deepseek-hidden-states/II_mateo.py main "./imagenet_same_class_dino2" --output-folder="./imagenet_same_class_dino2" --do-tokens --n-files=35 --diagonal-only --rank=0 --n-tokens-list=$ntokens

for 