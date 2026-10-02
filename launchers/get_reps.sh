#!/bin/bash

n_ranks=10

for rank in $(seq 1 $((n_ranks - 9))); do
    JOB_NAME="bin_reps_${rank}"
    save_path="/home/mmahaut/projects/exps/bin_reps/reps/${rank}"
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
poetry run python /home/mmahaut/projects/deepseek-hidden-states/II_mateo.py data-pickle \
    "./imagenet_same_class.txt" \
    "./imagenet_same_class_igpt" \
    "/projects/colt/imagenet21k_resized/imagenet21k_train/" \
    "igpt" \
    --rank=$rank \
    --n-ranks=$n_ranks \
    --batch-size=8
EOF
)
    sbatch --job-name=$JOB_NAME --output=$OUTPUT_LOG --error=$ERROR_LOG --mem=200G --partition=alien --qos=alien --exclude=node044 --nice=1 --gres=gpu:1 --wrap="$script"
done