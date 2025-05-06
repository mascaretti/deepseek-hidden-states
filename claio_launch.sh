#!/usr/bin/env bash
#SBATCH --job-name=binary
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#[optional] #SBATCH --nodes=1
#[optional] #SBATCH --ntasks-per-node=1
#[optional] #SBATCH --ntasks-per-socket=1
#[optional] #SBATCH --ntasks-per-core=1
#[optional] #SBATCH --distribution=cyclic:cyclic
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#[optional] #SBATCH --mem-per-cpu=4000mb
#[unconfig] #SBATCH --array=01-10
#SBATCH --partition=regular
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x.o%j
#SBATCH --error=logs/%x.e%j
module load cuda/12.4
ENV_DIR=/home/amascare/virtualenvs/clipvenv
python3.12 -m venv ${ENV_DIR}
source ${ENV_DIR}/bin/activate
## python src/20241118_masca_debug.py
## python src/20241118_masca_debugvit.py
## python src/20241118_masca_debug_clip.py
## python src/20241120_masca_clip_ktokens.py
## python src/20241120_masca_vit_ktokens.py
## python src/20241125_masca_clip_alltoone.py
## python src/20241128_masca_attentionmask.py
## python src/20241202_masca_pythiaktoall.py
python src/20250109_masca_binaryactivations.py
## python check_gpu_jax.py 
