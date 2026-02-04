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
languageB="spanish"
# -----------------------------------------------

deps=$(squeue -u "$USER" -h -o "%i" | paste -sd: -)

args=( "$dbg" "$min_token_length" "$modelA" "$modelB" "$aux_A" "$aux_B" "$languageA" "$languageB" )
echo "${args[@]}"

if [ -n "$deps" ]; then
  sbatch --dependency=afterany:$deps sII.sh "${args[@]}"
else
  sbatch sII.sh "${args[@]}"
fi
