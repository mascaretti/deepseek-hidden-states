#!/bin/bash
#SBATCH --job-name=ds_llama_compare
#SBATCH --output=logs/ds_llama_%j.out
#SBATCH --error=logs/ds_llama_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# SLURM job for DeepSeek-V3 vs LLama3.1-8B comparison

echo "========================================="
echo "DeepSeek-V3 vs LLama3.1-8B Comparison"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================="

# Load modules
module load cuda/11.8
module load python/3.9

# Activate environment
source /home/mascaretti/myjax/bin/activate

# Set working directory
cd /home/mascaretti/deepseek-hidden-states

# Create necessary directories
mkdir -p logs

# Export environment variables for better performance
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the comparison
echo ""
echo "Running comparison..."
echo ""

# Check if test mode requested
if [ "$1" = "test" ]; then
    echo "RUNNING IN TEST MODE"
    python compute_deepseek_llama_comparison.py --test --include_validation
elif [ "$1" = "validation_only" ]; then
    echo "RUNNING VALIDATION ONLY"
    python compute_deepseek_llama_comparison.py --skip_main --include_validation
else
    echo "RUNNING FULL ANALYSIS WITH VALIDATION"
    python compute_deepseek_llama_comparison.py --include_validation
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Job completed successfully at: $(date)"
    echo "========================================="
    
    # List output files
    echo ""
    echo "Output files:"
    ls -la /home/laios/image-information-imbalance/results/deepseek-llama-comparison*/
else
    echo ""
    echo "========================================="
    echo "Job FAILED at: $(date)"
    echo "Exit code: $?"
    echo "========================================="
fi
