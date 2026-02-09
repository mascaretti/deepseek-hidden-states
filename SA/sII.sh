#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1
#SBATCH --qos=mira
#SBATCH --mem=1000G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error


ARGS=("$@") # list of commandline inputs
echo "${ARGS[@]}"
source /home/acevedo/venv/bin/activate
python3 -u II_santiago.py "${ARGS[@]}"