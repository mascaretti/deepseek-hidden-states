#!/bin/bash
#SBATCH --job-name=multilayer_all
#SBATCH --output=logs/multilayer_%j.out
#SBATCH --error=logs/multilayer_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# SLURM job script for running all multi-layer analyses overnight

echo "========================================"
echo "OVERNIGHT MULTI-LAYER ANALYSIS (SLURM)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================"

# Load modules
module load python/3.9
module load cuda/11.8

# Activate environment
source /home/mascaretti/myjax/bin/activate

# Set working directory
cd /home/mascaretti/deepseek-hidden-states

# Create necessary directories
mkdir -p logs
mkdir -p /home/mascaretti/deepseek-hidden-states/plots/multilayer

# Export environment variables for better performance
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the overnight analysis script
chmod +x run_overnight_analysis.sh
./run_overnight_analysis.sh

echo ""
echo "Job completed at: $(date)"
echo "Exit status: $?"
