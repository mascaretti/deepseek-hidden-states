#!/usr/bin/env python
"""
Comprehensive plotting script for Information Imbalance results
Generates clear plots with all settings labeled
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse

# Configure matplotlib for publication quality without scienceplots
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Model colors - consistent across all plots
MODEL_COLORS = {
    'dino': '#0A5EB0',        # Blue
    'clip_image': '#DC143C',   # Crimson
    'clip_text': '#FF8C00',    # Dark Orange
    'deepseek': '#2E8B57',     # Sea Green
    'igpt': '#399918'          # Green (for future)
}

# Model display names
MODEL_NAMES = {
    'dino': 'DinoV2',
    'clip_image': 'CLIP-Image',
    'clip_text': 'CLIP-Text',
    'deepseek': 'DeepSeek-V3',
    'igpt': 'ImageGPT'
}


def load_bootstrap_results(bootstrap_dir: Path) -> Dict:
    """Load all bootstrap results and metadata"""
    results = {}
    
    # Load bootstrap data
    for file in bootstrap_dir.glob("*.npy"):
        if "summary" not in file.stem:
            data = np.load(file)
            results[file.stem] = data
    
    # Load summaries
    summaries = {}
    for file in bootstrap_dir.glob("summary_*.json"):
        with open(file, 'r') as f:
            key = file.stem.replace("summary_", "")
            summaries[key] = json.load(f)
    
    # Load config if exists
    config_file = bootstrap_dir / "bootstrap_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    return results, summaries, config


def plot_single_comparison(ax, forward_data, backward_data, model_a, model_b, 
                          layer_a=None, tokens=None, show_bands=True):
    """Plot a single comparison on given axes"""
    
    # Calculate statistics
    forward_mean = np.mean(forward_data, axis=1)
    forward_std = np.std(forward_data, axis=1)
    backward_mean = np.mean(backward_data, axis=1)
    backward_std = np.std(backward_data, axis=1)
    
    n_layers = forward_mean.shape[0]
    x = np.linspace(0., 1., n_layers)
    
    # Get colors
    color_a = MODEL_COLORS.get(model_a, '#808080')
    color_b = MODEL_COLORS.get(model_b, '#808080')
    
    # Plot forward direction (A → B)
    label_forward = f"{MODEL_NAMES[model_a]} → {MODEL_NAMES[model_b]}"
    ax.plot(x, forward_mean, '-', color=color_a, 
            label=label_forward, linewidth=2, marker='o', 
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    
    if show_bands:
        ax.fill_between(x, 
                       forward_mean - forward_std,
                       forward_mean + forward_std,
                       alpha=0.2, linewidth=0, color=color_a)
    
    # Plot backward direction (B → A)
    label_backward = f"{MODEL_NAMES[model_b]} → {MODEL_NAMES[model_a]}"
    ax.plot(x, backward_mean, '--', color=color_b,
            label=label_backward, linewidth=2, marker='s',
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    
    if show_bands:
        ax.fill_between(x,
                       backward_mean - backward_std,
                       backward_mean + backward_std,
                       alpha=0.2, linewidth=0, color=color_b)
    
    # Add annotations for specific values
    # Annotate maximum information imbalance
    max_forward_idx = np.argmax(forward_mean)
    max_backward_idx = np.argmax(backward_mean)
    
    ax.annotate(f'{forward_mean[max_forward_idx]:.3f}',
                xy=(x[max_forward_idx], forward_mean[max_forward_idx]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color=color_a)
    
    ax.annotate(f'{backward_mean[max_backward_idx]:.3f}',
                xy=(x[max_backward_idx], backward_mean[max_backward_idx]),
                xytext=(5, -15), textcoords='offset points',
                fontsize=8, color=color_b)


def create_main_comparison_plot(results: Dict, output_path: Path):
    """Create the main comparison plot with all model pairs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Define comparisons to plot
    comparisons = [
        ('deepseek_clip_text', 'DeepSeek-V3 vs CLIP-Text', 52, 9),
        ('clip_text_clip_image', 'CLIP-Text vs CLIP-Image', 12, 30),
        ('dino_clip_image', 'DinoV2 vs CLIP-Image', 24, 200),
        (None, None, None, None)  # Empty subplot for future use
    ]
    
    for idx, (key, title, layer, tokens) in enumerate(comparisons):
        ax = axes[idx]
        
        if key and key in results:
            # Extract model names from key
            parts = key.split('_')
            if len(parts) == 4:  # e.g., clip_text_clip_image
                model_a = f"{parts[0]}_{parts[1]}"
                model_b = f"{parts[2]}_{parts[3]}"
            else:  # e.g., dino_clip_image
                model_a = parts[0]
                model_b = f"{parts[1]}_{parts[2]}"
            
            # Load data
            forward_key = f"{model_a}_{model_b}_52"
            backward_key = f"{model_b}_{model_a}_52"
            
            if forward_key in results and backward_key in results:
                plot_single_comparison(
                    ax, 
                    results[forward_key], 
                    results[backward_key],
                    model_a, model_b
                )
                
                # Customize subplot
                ax.set_title(f"{title}\n(Layer {layer}, {tokens} tokens)", fontsize=12)
                ax.set_xlabel("Relative depth", fontsize=10)
                ax.set_ylabel("Information Imbalance", fontsize=10)
                ax.set_ylim(0.2, 1.0)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)
        else:
            # Empty subplot
            ax.text(0.5, 0.5, 'Reserved for future analysis', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle("Information Imbalance Analysis\nAll Model Comparisons", fontsize=14)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path / "all_comparisons.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(output_path / "all_comparisons.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/all_comparisons.pdf")


def plot_dino_multilayer_comparison(multilayer_dir: Path, bootstrap_dir: Path, output_path: Path):
    """Plot DinoV2 multi-layer comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    layers = [
        ('first', 1, 'Early Layer'),
        ('middle', 12, 'Middle Layer'),
        ('last', 24, 'Late Layer')
    ]
    
    # Load bootstrap data if available
    bootstrap_data = {}
    if bootstrap_dir.exists():
        for file in bootstrap_dir.glob("dino_*_layer*.npy"):
            bootstrap_data[file.stem] = np.load(file)
    
    for idx, (layer_name, layer_id, layer_title) in enumerate(layers):
        ax = axes[idx]
        
        layer_dir = multilayer_dir / f"dino_{layer_name}_layer{layer_id}"
        
        if layer_dir.exists():
            # For multi-layer, we need to compute the mean ourselves
            # since we don't have bootstrap data for each layer
            
            # Count files to determine n_layers_B
            safetensor_files = list(layer_dir.glob("layer_*.safetensors"))
            n_layers_B = len(safetensor_files)
            
            if n_layers_B > 0:
                # Create simple visualization without bootstrap
                x = np.linspace(0., 1., n_layers_B)
                
                # Placeholder for actual data loading
                # You would load the safetensors files here and compute means
                
                ax.text(0.5, 0.5, f'DinoV2 Layer {layer_id}\nvs\nCLIP-Image (all layers)\n\n{n_layers_B} target layers found',
                       transform=ax.transAxes, ha='center', va='center')
                
                ax.set_title(f"{layer_title} (Layer {layer_id})", fontsize=12)
                ax.set_xlabel("CLIP-Image Layer (normalized)", fontsize=10)
                ax.set_ylabel("Information Imbalance", fontsize=10)
                ax.set_ylim(0.2, 1.0)
        else:
            ax.text(0.5, 0.5, f'Data not found for\n{layer_name} layer',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')
    
    plt.suptitle("DinoV2 Multi-Layer Analysis vs CLIP-Image\n(5000 samples, 200 tokens)", 
                 fontsize=14)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path / "dino_multilayer_analysis.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(output_path / "dino_multilayer_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/dino_multilayer_analysis.pdf")


def create_summary_table(summaries: Dict, output_path: Path):
    """Create a summary table of all results"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Comparison', 'Direction', 'Max II', 'Min II', 'Mean II', 'Std II']
    rows = []
    
    for comp_name, summary in summaries.items():
        # Forward direction
        forward_mean = summary.get('forward_mean', [])
        if forward_mean:
            rows.append([
                comp_name.replace('_', ' ').title(),
                'Forward (A→B)',
                f"{np.max(forward_mean):.3f}",
                f"{np.min(forward_mean):.3f}",
                f"{np.mean(forward_mean):.3f}",
                f"{np.mean(summary.get('forward_std', [])):.3f}"
            ])
        
        # Backward direction
        backward_mean = summary.get('backward_mean', [])
        if backward_mean:
            rows.append([
                '',
                'Backward (B→A)',
                f"{np.max(backward_mean):.3f}",
                f"{np.min(backward_mean):.3f}",
                f"{np.mean(backward_mean):.3f}",
                f"{np.mean(summary.get('backward_std', [])):.3f}"
            ])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        if i % 4 in [1, 2]:  # Group by comparison
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Information Imbalance Summary Statistics', fontsize=14, pad=20)
    
    # Save
    fig.savefig(output_path / "summary_table.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(output_path / "summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/summary_table.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Information Imbalance results")
    parser.add_argument("--results_dir", type=str, 
                       default="/home/laios/image-information-imbalance/results",
                       help="Path to results directory")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/mascaretti/deepseek-hidden-states/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bootstrap_dir = results_dir / "bootstrap"
    multilayer_dir = results_dir / "dino-clip-image-multilayer-5000"
    
    print("Loading bootstrap results...")
    results, summaries, config = load_bootstrap_results(bootstrap_dir)
    
    print(f"Found {len(results)} bootstrap results")
    print(f"Found {len(summaries)} summaries")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Main comparison plot
    print("1. Creating main comparison plot...")
    create_main_comparison_plot(results, output_dir)
    
    # 2. DinoV2 multi-layer analysis
    print("2. Creating DinoV2 multi-layer plot...")
    plot_dino_multilayer_comparison(multilayer_dir, bootstrap_dir, output_dir)
    
    # 3. Summary table
    print("3. Creating summary table...")
    create_summary_table(summaries, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.pdf")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
