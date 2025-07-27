#!/bin/bash
# Simple plot generation without scienceplots dependency

echo "Information Imbalance Plot Generation (Simple Version)"
echo "===================================================="

# Activate environment
source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states

# Run the simple plotting script
echo "Generating plots..."
python plot_results_simple.py \
    --results_dir "/home/laios/image-information-imbalance/results" \
    --output_dir "/home/mascaretti/deepseek-hidden-states/plots"

echo ""
echo "Plot generation complete!"
echo ""
echo "Generated plots:"
ls -la /home/mascaretti/deepseek-hidden-states/plots/*.png
echo ""
ls -la /home/mascaretti/deepseek-hidden-states/plots/*.pdf

echo ""
echo "To copy plots to your local machine:"
echo "scp mascaretti@dgx02:/home/mascaretti/deepseek-hidden-states/plots/*.pdf ."
