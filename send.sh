#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amascare@sissa.it
#SBATCH --nodes=1
#SBATCH --job-name=dino
#SBATCH --mem=500GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out

source /home/$USER/mysglang/bin/activate
## source /home/$USER/myjax/bin/activate
cd /home/$USER/deepseek-hidden-states

python3 extract_hidden_images_states.py
## srun python3 extract_hidden_states.py
## srun python3 compute_information_imbalance_images.py --input_path_A '/home/laios/activations/flickr/deepseek_captions5/' --input_path_B '/home/laios/image-information-imbalance/facebook-dinov2-large/' --output_folder '/home/laios/image-information-imbalance/dino-cpt-5/'
