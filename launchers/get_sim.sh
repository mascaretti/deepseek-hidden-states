#!/bin/bash

# Parameters
#SBATCH --mem=100G
##SBATCH --cpus-per-task=48
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --error=/home/mmahaut/projects/exps/bin_reps/sim_%j.err
#SBATCH --output=/home/mmahaut/projects/exps/bin_reps/sim_%j.out

n_rank=1000

for rank in $(seq 0 $n_rank); do
    JOB_NAME="sim_bin_reps_${rank}"
    save_path="/home/mmahaut/projects/exps/bin_reps/sim/sim_${rank}"
    mkdir -p $(dirname $save_path)
    ERROR_LOG="$save_path.err"
    OUTPUT_LOG="$save_path.out"
    script=$(cat <<EOF
echo "$SBATCH_PARAMS" > sbatch_params.txt
source ~/.bashrc
echo \$SLURMD_NODENAME
conda activate emecom
which python
cd ~/projects/emecom/
poetry run python /home/mmahaut/projects/deepseek-hidden-states/II_mateo.py data-selection \
    --rank=${rank} \
    --n-rank=${n_rank} 
EOF
)
    sbatch --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=100G --partition=alien --qos=alien --exclude=node044 --nice=10 --gres=gpu:1 --wrap="$script"
done