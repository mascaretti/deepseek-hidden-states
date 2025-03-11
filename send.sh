#!/bin/bash
#SBATCH --nodes=2
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error


source /home/rende/mysglang/bin/activate

cd /home/rende/deepseek

srun --ntasks=2 --nodes=2 python3 extract_hidden_states.py


# srun --ntasks=2 --nodes=2 python3 -m sglang.launch_server --model /home/rende/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d --trust-remote-code --tp 16 --dist-init-addr 10.149.146.53:8000 --nnodes 2 --grammar-backend xgrammar --node-rank $SLURM_NODEID