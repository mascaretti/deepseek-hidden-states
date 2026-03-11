#!/bin/bash

deps=$(squeue -u "$USER" -h -t R -o "%i" | paste -sd: -)

if [ -n "$deps" ]; then
    sbatch --dependency=afterany:"$deps" send_llama.sh "$@"
else
    sbatch send_llama.sh "$@"
fi
