#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error


source /home/rende/myjax/bin/activate

cd /home/rende/deepseek-hidden-states

row_number=${1}
python3 compute_information_imbalance.py $row_number