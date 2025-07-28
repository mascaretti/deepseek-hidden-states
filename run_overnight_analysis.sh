#!/bin/bash
# Overnight runner for all multi-layer analyses
# Uses existing scripts to run all comparisons

set -e  # Exit on error

echo "========================================"
echo "OVERNIGHT MULTI-LAYER ANALYSIS"
echo "Started at: $(date)"
echo "========================================"

# Configuration
REPO_DIR="/home/mascaretti/deepseek-hidden-states"
RESULTS_BASE="/home/laios/image-information-imbalance/results"
PLOTS_DIR="${REPO_DIR}/plots/multilayer"

# Activate environment
source /home/mascaretti/myjax/bin/activate
cd ${REPO_DIR}

# Create directories
mkdir -p ${PLOTS_DIR}
mkdir -p logs

# Function to run analysis for a model pair
run_analysis() {
    local NAME=$1
    local COMPARISON=$2
    local MODEL_A=$3
    local MODEL_B=$4
    local PATH_A=$5
    local PATH_B=$6
    local LAYERS=$7
    
    echo ""
    echo "----------------------------------------"
    echo "Processing: ${NAME}"
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    # Step 1: Run multi-layer analysis using the modified script
    if [ ! -d "${RESULTS_BASE}/${NAME}-multilayer-5000" ]; then
        echo "Running information imbalance computation..."
        
        # Create a temporary modified version of compute_asymmetry_multi_layer.py
        cp compute_asymmetry_multi_layer.py temp_${NAME}.py
        
        # Modify the main function for this specific comparison
        cat >> temp_${NAME}.py << EOF

# Override main for ${NAME}
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--layers", type=str, default="${LAYERS}")
    args = parser.parse_args()
    
    subset_size = args.subset_size
    n_files = int(subset_size / 10)
    
    print("Running ${MODEL_A} vs ${MODEL_B} comparison")
    print(f"Layers: {args.layers}")
    
    # Load data based on model types
EOF

        # Add appropriate data loading based on model type
        case ${MODEL_A} in
            "deepseek")
                cat >> temp_${NAME}.py << 'EOF'
    from compute_asymmetry_unified import collect_deepseek_data
    all_activations_A = collect_deepseek_data("${PATH_A}", n_loaded_tokens=9, subset_size=subset_size)
EOF
                ;;
            "clip_text")
                cat >> temp_${NAME}.py << 'EOF'
    from compute_asymmetry_unified import collect_clip_text_data
    all_activations_A = collect_clip_text_data("${PATH_A}", n_files=n_files, last_tokens=30)
EOF
                ;;
            "dino")
                cat >> temp_${NAME}.py << 'EOF'
    from compute_asymmetry_unified import collect_image_model_data
    all_activations_A = collect_image_model_data("${PATH_A}", n_files=n_files, last_tokens=200)
EOF
                ;;
        esac
        
        # Add model B loading
        case ${MODEL_B} in
            "clip_text")
                cat >> temp_${NAME}.py << 'EOF'
    from compute_asymmetry_unified import collect_clip_text_data
    all_activations_B = collect_clip_text_data("${PATH_B}", n_files=n_files, last_tokens=30)
EOF
                ;;
            "clip_image")
                cat >> temp_${NAME}.py << 'EOF'
    from compute_asymmetry_unified import collect_clip_image_data
    all_activations_B = collect_clip_image_data("${PATH_B}", n_files=n_files, last_tokens=200)
EOF
                ;;
            "dino")
                cat >> temp_${NAME}.py << 'EOF'
    from compute_asymmetry_unified import collect_image_model_data
    all_activations_B = collect_image_model_data("${PATH_B}", n_files=n_files, last_tokens=200)
EOF
                ;;
        esac
        
        # Complete the script
        cat >> temp_${NAME}.py << EOF
    
    # Define layers based on model
    if "${MODEL_A}" == "deepseek":
        layers_A = {"early": 10, "middle": 30, "semantic": 52}
    elif "${MODEL_A}" == "clip_text":
        layers_A = {"early": 1, "middle": 6, "last": 12}
    else:  # dino
        layers_A = {"early": 1, "middle": 12, "last": 24}
    
    # Filter requested layers
    if args.layers != "all":
        layer_names = args.layers.split(",")
        layers_A = {name: layers_A[name] for name in layer_names if name in layers_A}
    
    output_base = "${RESULTS_BASE}/${NAME}-multilayer-5000"
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    # Determine comparison type
    if "${MODEL_A}" in ["deepseek", "clip_text"] and "${MODEL_B}" in ["clip_image", "dino"]:
        comparison_type = "text_image"
    elif "${MODEL_A}" == "deepseek" and "${MODEL_B}" == "clip_text":
        comparison_type = "text_text"  
    else:
        comparison_type = "image_image"
    
    compute_inf_imb_matrix_multi_layer(
        all_activations_A=all_activations_A,
        all_activations_B=all_activations_B,
        output_folder=output_base,
        layers_A=layers_A,
        model_a_name="${MODEL_A}",
        model_b_name="${MODEL_B}",
        comparison_type=comparison_type
    )
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
EOF
        
        # Run the analysis
        python temp_${NAME}.py --layers "${LAYERS}" 2>&1 | tee logs/${NAME}_analysis.log
        rm -f temp_${NAME}.py
    else
        echo "Analysis results already exist, skipping..."
    fi
    
    # Step 2: Run bootstrap if needed
    BOOTSTRAP_DIR="${RESULTS_BASE}/bootstrap/${NAME}-multilayer"
    if [ ! -d "${BOOTSTRAP_DIR}" ]; then
        echo "Running bootstrap analysis..."
        
        # Determine n_layers_B based on model
        case ${MODEL_B} in
            "clip_text")
                N_LAYERS_B=13
                ;;
            "clip_image"|"dino")
                N_LAYERS_B=25
                ;;
        esac
        
        python bootstrap_multilayer.py \
            --input_dir "${RESULTS_BASE}/${NAME}-multilayer-5000" \
            --output_dir "${BOOTSTRAP_DIR}" \
            --sample_size 5000 \
            --n_layers ${N_LAYERS_B} 2>&1 | tee logs/${NAME}_bootstrap.log
    else
        echo "Bootstrap results already exist, skipping..."
    fi
    
    echo "✓ Completed ${NAME}"
}

# Run all analyses
echo ""
echo "Running analyses for all model pairs..."

# 1. DeepSeek vs CLIP-Text
run_analysis "deepseek_clip_text" "deepseek_clip_text" "deepseek" "clip_text" \
    "/home/laios/activations/flickr/deepseek_all" \
    "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14" \
    "early,middle,semantic"

# 2. DeepSeek vs CLIP-Image  
run_analysis "deepseek_clip_image" "deepseek_clip_image" "deepseek" "clip_image" \
    "/home/laios/activations/flickr/deepseek_all" \
    "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14" \
    "early,middle,semantic"

# 3. DeepSeek vs DinoV2
run_analysis "deepseek_dino" "deepseek_dino" "deepseek" "dino" \
    "/home/laios/activations/flickr/deepseek_all" \
    "/home/laios/image-information-imbalance/facebook-dinov2-large" \
    "early,middle,semantic"

# 4. CLIP-Text vs CLIP-Image
run_analysis "clip_text_clip_image" "clip_text_image" "clip_text" "clip_image" \
    "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14" \
    "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14" \
    "early,middle,last"

# 5. DinoV2 vs CLIP-Image (already done but let's check)
run_analysis "dino_clip_image" "dino_clip_image" "dino" "clip_image" \
    "/home/laios/image-information-imbalance/facebook-dinov2-large" \
    "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14" \
    "early,middle,last"

echo ""
echo "========================================"
echo "Step 3: Generating all plots..."
echo "========================================"

# Generate plots using the enhanced plotting script
python plot_results_simple.py \
    --results_dir "${RESULTS_BASE}" \
    --output_dir "${PLOTS_DIR}" 2>&1 | tee logs/plotting.log

# Also generate individual multi-layer plots
python << 'EOF'
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

results_base = "/home/laios/image-information-imbalance/results"
plots_dir = "/home/mascaretti/deepseek-hidden-states/plots/multilayer"
Path(plots_dir).mkdir(parents=True, exist_ok=True)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

model_pairs = [
    ("deepseek_clip_text", "DeepSeek", "CLIP-Text"),
    ("deepseek_clip_image", "DeepSeek", "CLIP-Image"),
    ("deepseek_dino", "DeepSeek", "DinoV2"),
    ("clip_text_clip_image", "CLIP-Text", "CLIP-Image"),
    ("dino_clip_image", "DinoV2", "CLIP-Image")
]

for pair_name, model_a_display, model_b_display in model_pairs:
    bootstrap_dir = Path(results_base) / "bootstrap" / f"{pair_name}-multilayer"
    summary_file = bootstrap_dir / "multilayer_summary.json"
    
    if not summary_file.exists():
        print(f"Skipping {pair_name} - no bootstrap results")
        continue
    
    with open(summary_file, 'r') as f:
        results = json.load(f)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define colors for layers
    layer_colors = {
        'early': '#87CEEB',
        'middle': '#4682B4', 
        'last': '#000080',
        'semantic': '#2E8B57'
    }
    
    # Plot each layer
    for key, data in results.items():
        # Extract layer info from key
        parts = key.split('_')
        layer_name = parts[1]  # early, middle, last, semantic
        layer_id = int(parts[-1].replace('layer', ''))
        
        if layer_name not in layer_colors:
            continue
            
        forward_mean = np.array(data['forward_mean'])
        backward_mean = np.array(data['backward_mean'])
        forward_std = np.array(data['forward_std'])
        backward_std = np.array(data['backward_std'])
        
        n_layers = len(forward_mean)
        x = np.linspace(0., 1., n_layers)
        
        color = layer_colors[layer_name]
        label = f'{layer_name.title()} (L{layer_id})'
        
        # Forward direction
        ax1.plot(x, forward_mean, '-o', color=color, label=label,
                markersize=4, markevery=5, linewidth=2)
        ax1.fill_between(x, forward_mean - forward_std, forward_mean + forward_std,
                        alpha=0.2, color=color)
        
        # Backward direction
        ax2.plot(x, backward_mean, '-s', color=color, label=label,
                markersize=4, markevery=5, linewidth=2)
        ax2.fill_between(x, backward_mean - backward_std, backward_mean + backward_std,
                        alpha=0.2, color=color)
    
    # Configure axes
    ax1.set_title(f'{model_a_display} → {model_b_display}', fontsize=14)
    ax1.set_xlabel('Target Model Layer (normalized)', fontsize=12)
    ax1.set_ylabel('Information Imbalance', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2.set_title(f'{model_b_display} → {model_a_display}', fontsize=14)
    ax2.set_xlabel('Target Model Layer (normalized)', fontsize=12)
    ax2.set_ylabel('Information Imbalance', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.suptitle(f'{model_a_display} vs {model_b_display} Multi-Layer Analysis\n5000 samples, 500 bootstrap iterations',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = Path(plots_dir) / f"{pair_name}_multilayer_analysis"
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

print("\nAll multi-layer plots generated!")
EOF

echo ""
echo "========================================"
echo "ANALYSIS COMPLETE!"
echo "Finished at: $(date)"
echo "========================================"
echo ""
echo "Results locations:"
echo "- Analysis results: ${RESULTS_BASE}/*/multilayer-5000/"
echo "- Bootstrap results: ${RESULTS_BASE}/bootstrap/*-multilayer/"
echo "- Plots: ${PLOTS_DIR}/"
echo ""
echo "Log files: ${REPO_DIR}/logs/"

# List all generated plots
echo ""
echo "Generated plots:"
ls -la ${PLOTS_DIR}/*.png 2>/dev/null || echo "No plots found"