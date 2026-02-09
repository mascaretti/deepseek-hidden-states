#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --qos=mira
#SBATCH --reservation=deepseek
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1      
#SBATCH --gres=gpu:8
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

ARGS=("$@")
echo "Arguments passed: ${ARGS[@]}"
srun --ntasks=2 --nodes=2 /home/rende/mysglang/bin/python3 extract_hidden_states.py "${ARGS[@]}"