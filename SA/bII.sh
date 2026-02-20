#!/bin/bash
# Submit sII.sh with an afterany dependency on all your current jobs.
# All arguments are hard-coded here.

dbg=0
min_token_length=40
modelA="deepseek"
modelB="deepseek"
aux_A="source"
aux_B="target"
languageA="english"
languageB="french"
diagonal_constraint=1
# -----------------------------------------------

deps=$(squeue -u "$USER" -h -o "%i" | paste -sd: -)

args=( "$dbg" "$min_token_length" "$modelA" "$modelB" "$aux_A" "$aux_B" "$languageA" "$languageB" "$diagonal_constraint" )
echo "${args[@]}"

if [ "$dbg" -eq 1 ]; then
  gpu_arg="--gres=gpu:0"
else
  gpu_arg="--gres=gpu:1"
fi

if [ -n "$deps" ]; then
  sbatch $gpu_arg --dependency=afterany:$deps sII.sh "${args[@]}"
else
  sbatch $gpu_arg sII.sh "${args[@]}"
fi
