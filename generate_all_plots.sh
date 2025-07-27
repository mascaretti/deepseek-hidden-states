#!/bin/bash
# Generate all plots for Information Imbalance results

echo "Information Imbalance Plot Generation"
echo "===================================="

# Activate environment
source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states

# Paths
RESULTS_DIR="/home/laios/image-information-imbalance/results"
PLOTS_DIR="/home/mascaretti/deepseek-hidden-states/plots"

# Create plots directory
mkdir -p ${PLOTS_DIR}
mkdir -p ${PLOTS_DIR}/individual

echo ""
echo "Step 1: Check if multilayer bootstrap is needed..."
if [ ! -d "${RESULTS_DIR}/bootstrap/multilayer" ]; then
    echo "Running bootstrap for DinoV2 multi-layer results..."
    python bootstrap_multilayer.py \
        --input_dir "${RESULTS_DIR}/dino-clip-image-multilayer-5000" \
        --output_dir "${RESULTS_DIR}/bootstrap/multilayer"
else
    echo "Multi-layer bootstrap already exists, skipping..."
fi

echo ""
echo "Step 2: Generate all plots..."
python plot_all_results.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${PLOTS_DIR}"

echo ""
echo "Step 3: Generate enhanced plots with multi-layer analysis..."
python plot_enhanced.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${PLOTS_DIR}"

echo ""
echo "===================================="
echo "Plot generation complete!"
echo ""
echo "Generated plots in ${PLOTS_DIR}:"
ls -la ${PLOTS_DIR}/*.pdf

echo ""
echo "Individual plots in ${PLOTS_DIR}/individual/:"
ls -la ${PLOTS_DIR}/individual/*.pdf 2>/dev/null || echo "No individual plots found"

echo ""
echo "To view plots:"
echo "  PDF viewer: evince ${PLOTS_DIR}/*.pdf"
echo "  Copy to local: scp mascaretti@dgx02:${PLOTS_DIR}/*.pdf ."
