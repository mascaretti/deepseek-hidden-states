#!/bin/bash
# Quick test script for DeepSeek vs LLama comparison
# Run this first to verify everything works

echo "DeepSeek-V3 vs LLama3.1-8B Quick Test"
echo "====================================="
echo ""

# Activate environment
source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states

# First, verify the script exists and has no syntax errors
echo "1. Checking script syntax..."
python -m py_compile compute_deepseek_llama_comparison.py
if [ $? -eq 0 ]; then
    echo "✓ Script syntax OK"
else
    echo "✗ Script has syntax errors!"
    exit 1
fi

# Run in test mode (should complete in <10 minutes)
echo ""
echo "2. Running test mode (500 samples, 50 bootstrap iterations)..."
echo "   This should take 5-10 minutes depending on data loading speed"
echo ""

time python compute_deepseek_llama_comparison.py \
    --test \
    --test_size 500 \
    --include_validation

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test completed successfully!"
    echo ""
    echo "Test results saved to:"
    ls -la /home/laios/image-information-imbalance/results/deepseek-llama-comparison_test/
    
    echo ""
    echo "To view the test plots:"
    echo "  Main: display /home/laios/image-information-imbalance/results/deepseek-llama-comparison_test/deepseek_llama_comparison.png"
    echo "  Validation: display /home/laios/image-information-imbalance/results/deepseek-llama-comparison_test/validation_deepseek_self.png"
    
    echo ""
    echo "Quick validation check (should run in ~2 minutes):"
    echo "  python compute_deepseek_llama_comparison.py --test --test_size 200 --skip_main --include_validation"
    
    echo ""
    echo "To run the full analysis (5000 samples, 500 bootstrap):"
    echo "  sbatch submit_deepseek_llama.sbatch"
    echo ""
    echo "Or to run another test with SLURM:"
    echo "  sbatch submit_deepseek_llama.sbatch test"
else
    echo ""
    echo "✗ Test failed! Check the error messages above."
    exit 1
fi
