#!/bin/bash
# Submit spast-future.sh with an afterany dependency on all your current jobs.
# All arguments are hard-coded here.

dbg=0
min_token_length=40
modelA="deepseek"
language="dutch"
batch_shuffle=0
subtract_mean=0
# -----------------------------------------------

deps=$(squeue -u "$USER" -h -o "%i" | paste -sd: -)

args=( "$dbg" "$min_token_length" "$modelA" "$language" "$batch_shuffle" "$subtract_mean" )
echo "${args[@]}"

if [ "$dbg" -eq 1 ]; then
  gpu_arg="--gres=gpu:0"
else
  gpu_arg="--gres=gpu:1"
fi

if [ -n "$deps" ]; then
  sbatch $gpu_arg --dependency=afterany:$deps spast-future.sh "${args[@]}"
else
  sbatch $gpu_arg spast-future.sh "${args[@]}"
fi
