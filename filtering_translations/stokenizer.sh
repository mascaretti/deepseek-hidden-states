#!/bin/bash
# SBATCH -A Sis25_laio
#SBATCH --job-name=tokenization
# SBATCH -p boost_usr_prod
# SBATCH --qos=boost_qos_dbg
#SBATCH --time 1:00:00
# SBATCH -p regular # for claio
# SBATCH --time 3:00:00
#SBATCH --mem=100G
#SBATCH --output=./log_tokenizer/%x.o%j              # Standard output
#SBATCH --error=./log_tokenizer/%x.o%j               # Standard error

ARGS=("$@") # list of commandline inputs
python3 -u tokenizer.py "${ARGS[@]}"