#!/usr/bin/env python
"""
Standalone plotting script for Information Imbalance results
No external dependencies except matplotlib and numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

# Configure matplotlib for clean plots
def setup_matplotlib():
    """Configure matplotlib for publication-quality plots without scienceplots"""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2,
        'lines.markersize': 5,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

# Model colors
MODEL_COLORS = {
    'dino': '#1f77b4',        # Blue
    'clip_image': '#ff7f0e',   # Orange
    'clip_text': '#2ca02c',    # Green
    'deepseek': '#d62728',     # Red
}

MODEL_NAMES = {
    'dino': 'DinoV2',
    'clip_image': 'CLIP-Image',
    'clip_text': 'CLIP-Text',
    'deepseek': 'DeepSeek-V3',
}


def plot_comparison(forward_data, backward_data, model_a, model_b, 
                    title, layer_info, output_path):
    """Create a single comparison plot"""
    
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate statistics
    forward_mean = np.mean(forward_data, axis=1)
    forward_std = np.std(forward_data, axis=1)
    backward_mean = np.mean(backward_data, axis=1)
    backward_std = np.std(backward_data, axis=1)
    
    n_layers = forward_mean.shape[0]
    x = np.linspace(0., 1., n_layers)
    
    # Colors
    color_a = MODEL_COLORS.get(model_a, '#666666')
    color_b = MODEL_COLORS.get(model_b, '#999999')
    
    # Plot forward direction
    ax.plot(x, forward_mean, '-o', color=color_a, 
            label=f"{MODEL_NAMES[model_a]} → {MODEL_NAMES[model_b]}",
            markersize=4, markevery=3)
    ax.fill_between(x, 
                   forward_mean - forward_std,
                   forward_mean + forward_std,
                   alpha=0.2, color=color_a)
    
    # Plot backward direction
    ax.plot(x, backward_mean, '--s', color=color_b,
            label=f"{MODEL_NAMES[model_b]} → {MODEL_NAMES[model_a]}",
            markersize=4, markevery=3)
    ax.fill_between(x,
                   backward_mean - backward_std,
                   backward_mean + backward_std,
                   alpha=0.2, color=color_b)
    
    # Labels and title
    ax.set_xlabel('Target Model Relative Depth', fontsize=12)
    ax.set_ylabel('Information Imbalance', fontsize=12)
    ax.set_title(f'{title}\n{layer_info}', fontsize=14, pad=10)
    
    # Dynamic y-axis limits with padding
    all_data = np.concatenate([forward_mean, backward_mean])
    y_min = max(0.0, np.min(all_data) - 0.1)
    y_max = min(1.0, np.max(all_data) + 0.1)
    ax.set_ylim(y_min, y_max)
    
    # Legend
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Statistics box
    stats_text = f'Forward: max={np.max(forward_mean):.3f}, mean={np.mean(forward_mean):.3f}\n'
    stats_text += f'Backward: max={np.max(backward_mean):.3f}, mean={np.mean(backward_mean):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_all_comparisons(results_dir, output_dir):
    """Create all comparison plots"""
    
    bootstrap_dir = results_dir / "bootstrap"
    plots_dir = output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Define comparisons
    comparisons = [
        {
            'name': 'deepseek_clip_text',
            'model_a': 'deepseek',
            'model_b': 'clip_text',
            'title': 'DeepSeek-V3 vs CLIP-Text',
            'layer_info': 'Source Layer: 52 (DeepSeek), Tokens: 9',
            'forward_file': 'deepseek_clip_text_52.npy',
            'backward_file': 'clip_text_deepseek_52.npy'
        },
        {
            'name': 'clip_text_clip_image',
            'model_a': 'clip_text',
            'model_b': 'clip_image',
            'title': 'CLIP-Text vs CLIP-Image',
            'layer_info': 'Source Layer: 12 (CLIP-Text), Tokens: 30',
            'forward_file': 'clip_text_clip_image_52.npy',
            'backward_file': 'clip_image_clip_text_52.npy'
        },
        {
            'name': 'dino_clip_image',
            'model_a': 'dino',
            'model_b': 'clip_image', 
            'title': 'DinoV2 vs CLIP-Image',
            'layer_info': 'Source Layer: 24 (DinoV2), Tokens: 200',
            'forward_file': 'dino_clip_image_52.npy',
            'backward_file': 'clip_image_dino_52.npy'
        },
        {
            'name': 'deepseek_dino',
            'model_a': 'deepseek',
            'model_b': 'dino',
            'title': 'DeepSeek-V3 vs DinoV2',
            'layer_info': 'Source Layer: 52 (DeepSeek), Tokens: 200',
            'forward_file': 'deepseek_dino_52.npy',
            'backward_file': 'dino_deepseek_52.npy'
        }
    ]
    
    # Create individual plots
    for comp in comparisons:
        forward_path = bootstrap_dir / comp['forward_file']
        backward_path = bootstrap_dir / comp['backward_file']
        
        if forward_path.exists() and backward_path.exists():
            print(f"\nProcessing {comp['name']}...")
            
            forward_data = np.load(forward_path)
            backward_data = np.load(backward_path)
            
            output_path = plots_dir / f"{comp['name']}_plot"
            
            plot_comparison(
                forward_data, backward_data,
                comp['model_a'], comp['model_b'],
                comp['title'], comp['layer_info'],
                output_path
            )
        else:
            print(f"Skipping {comp['name']} - data files not found")


def plot_summary_grid(results_dir, output_dir):
    """Create a 2x2 grid with all comparisons including bootstrap bands"""
    
    setup_matplotlib()
    bootstrap_dir = results_dir / "bootstrap"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    comparisons = [
        ('deepseek_clip_text', 'deepseek', 'clip_text', 'DeepSeek vs CLIP-Text\n(Layer 52, 9 tokens)'),
        ('clip_text_clip_image', 'clip_text', 'clip_image', 'CLIP-Text vs CLIP-Image\n(Layer 12, 30 tokens)'),
        ('dino_clip_image', 'dino', 'clip_image', 'DinoV2 vs CLIP-Image\n(Layer 24, 200 tokens)'),
        ('deepseek_dino', 'deepseek', 'dino', 'DeepSeek vs DinoV2\n(Layer 52, 200 tokens)')
    ]
    
    # First pass: find global y-axis limits
    global_y_min = 1.0
    global_y_max = 0.0
    
    all_data = []
    for name, model_a, model_b, _ in comparisons:
        # Check if this is the DeepSeek-Dino comparison (might need different path)
        if name == 'deepseek_dino':
            # Check in the regular results directory first
            forward_file = results_dir / f"deepseek-dino-ranks/bootstrap/{model_a}_{model_b}_52.npy"
            backward_file = results_dir / f"deepseek-dino-ranks/bootstrap/{model_b}_{model_a}_52.npy"
            if not forward_file.exists():
                forward_file = bootstrap_dir / f"{model_a}_{model_b}_52.npy"
                backward_file = bootstrap_dir / f"{model_b}_{model_a}_52.npy"
        else:
            forward_file = bootstrap_dir / f"{model_a}_{model_b}_52.npy"
            backward_file = bootstrap_dir / f"{model_b}_{model_a}_52.npy"
        
        if forward_file.exists() and backward_file.exists():
            forward_data = np.load(forward_file)
            backward_data = np.load(backward_file)
            
            forward_mean = np.mean(forward_data, axis=1)
            backward_mean = np.mean(backward_data, axis=1)
            forward_std = np.std(forward_data, axis=1)
            backward_std = np.std(backward_data, axis=1)
            
            all_values = np.concatenate([
                forward_mean - forward_std,
                forward_mean + forward_std,
                backward_mean - backward_std,
                backward_mean + backward_std
            ])
            global_y_min = min(global_y_min, np.min(all_values))
            global_y_max = max(global_y_max, np.max(all_values))
            
            all_data.append((forward_data, backward_data))
        else:
            all_data.append((None, None))
    
    # Add padding to y-axis
    y_padding = (global_y_max - global_y_min) * 0.1
    global_y_min = max(0.0, global_y_min - y_padding)
    global_y_max = min(1.0, global_y_max + y_padding)
    
    # Second pass: create plots with shared y-axis and bootstrap bands
    for idx, ((name, model_a, model_b, title), (forward_data, backward_data)) in enumerate(zip(comparisons, all_data)):
        ax = axes[idx]
        
        if forward_data is not None and backward_data is not None:
            # Calculate statistics
            forward_mean = np.mean(forward_data, axis=1)
            forward_std = np.std(forward_data, axis=1)
            backward_mean = np.mean(backward_data, axis=1)
            backward_std = np.std(backward_data, axis=1)
            
            n_layers = len(forward_mean)
            x = np.linspace(0., 1., n_layers)
            
            # Colors
            color_a = MODEL_COLORS.get(model_a, '#666666')
            color_b = MODEL_COLORS.get(model_b, '#999999')
            
            # Plot with confidence bands
            ax.plot(x, forward_mean, '-', color=color_a, linewidth=2,
                   label=f"{MODEL_NAMES.get(model_a, model_a)} → {MODEL_NAMES.get(model_b, model_b)}")
            ax.fill_between(x, 
                           forward_mean - forward_std,
                           forward_mean + forward_std,
                           alpha=0.2, color=color_a)
            
            ax.plot(x, backward_mean, '--', color=color_b, linewidth=2,
                   label=f"{MODEL_NAMES.get(model_b, model_b)} → {MODEL_NAMES.get(model_a, model_a)}")
            ax.fill_between(x,
                           backward_mean - backward_std,
                           backward_mean + backward_std,
                           alpha=0.2, color=color_b)
            
            ax.set_ylim(global_y_min, global_y_max)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'Max: {np.max(forward_mean):.3f}, {np.max(backward_mean):.3f}'
            ax.text(0.98, 0.02, stats_text,
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.7))
        else:
            # For missing DeepSeek-Dino, add a note
            if name == 'deepseek_dino':
                ax.text(0.5, 0.5, 'DeepSeek-DinoV2 comparison\nData not found\n(Run analysis if needed)', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, color='red')
            else:
                ax.text(0.5, 0.5, 'Data not found', 
                       transform=ax.transAxes, ha='center', va='center')
            ax.set_ylim(global_y_min, global_y_max)
            
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Relative Depth', fontsize=10)
        
        # Label y-axis on left plots
        if idx % 2 == 0:
            ax.set_ylabel('Information Imbalance', fontsize=11)
    
    plt.suptitle('Information Imbalance Analysis with Bootstrap Confidence\n5000 samples, 500 bootstrap iterations', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_path = output_dir / 'summary_grid_2x2_bootstrap'
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_dino_multilayer(results_dir, output_dir):
    """Create plots for DinoV2 multi-layer analysis"""
    
    setup_matplotlib()
    
    # Check for multi-layer directories
    multilayer_base = results_dir / "dino-clip-image-multilayer-5000"
    if not multilayer_base.exists():
        print("Multi-layer DinoV2 results not found")
        return
    
    # Check if bootstrap was run for multilayer
    bootstrap_multilayer = results_dir / "bootstrap" / "multilayer"
    use_bootstrap = bootstrap_multilayer.exists()
    
    if use_bootstrap:
        # Load bootstrap results
        summary_file = bootstrap_multilayer / "multilayer_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                multilayer_data = json.load(f)
        else:
            multilayer_data = {}
    
    # Create figure with subplots for each layer
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    layer_configs = [
        ('first', 1, 'DinoV2 Layer 1 (Early)'),
        ('middle', 12, 'DinoV2 Layer 12 (Middle)'),
        ('last', 24, 'DinoV2 Layer 24 (Late)')
    ]
    
    # Colors for DinoV2 layers
    layer_colors = {
        'first': '#87CEEB',   # Light blue
        'middle': '#4682B4',  # Steel blue
        'last': '#000080'     # Navy blue
    }
    
    # Find global y-limits
    global_y_min = 1.0
    global_y_max = 0.0
    
    # First pass to find limits
    for layer_name, layer_id, _ in layer_configs:
        if use_bootstrap:
            key = f"dino_{layer_name}_layer{layer_id}"
            if key in multilayer_data:
                data = multilayer_data[key]
                forward_mean = np.array(data['forward_mean'])
                backward_mean = np.array(data['backward_mean'])
                forward_std = np.array(data['forward_std'])
                backward_std = np.array(data['backward_std'])
                
                all_values = np.concatenate([
                    forward_mean - forward_std,
                    forward_mean + forward_std,
                    backward_mean - backward_std,
                    backward_mean + backward_std
                ])
                global_y_min = min(global_y_min, np.min(all_values))
                global_y_max = max(global_y_max, np.max(all_values))
    
    # Add padding
    if global_y_max > global_y_min:
        y_padding = (global_y_max - global_y_min) * 0.1
        global_y_min = max(0.0, global_y_min - y_padding)
        global_y_max = min(1.0, global_y_max + y_padding)
    else:
        global_y_min, global_y_max = 0.3, 0.9
    
    # Create plots
    for idx, (layer_name, layer_id, title) in enumerate(layer_configs):
        ax = axes[idx]
        
        if use_bootstrap:
            # Use bootstrap data
            key = f"dino_{layer_name}_layer{layer_id}"
            if key in multilayer_data:
                data = multilayer_data[key]
                forward_mean = np.array(data['forward_mean'])
                backward_mean = np.array(data['backward_mean'])
                forward_std = np.array(data['forward_std'])
                backward_std = np.array(data['backward_std'])
                
                n_layers = len(forward_mean)
                x = np.linspace(0., 1., n_layers)
                
                color = layer_colors[layer_name]
                
                # Plot forward (DinoV2 → CLIP)
                ax.plot(x, forward_mean, '-', color=color, linewidth=2,
                       label='DinoV2 → CLIP-Image')
                ax.fill_between(x, 
                               forward_mean - forward_std,
                               forward_mean + forward_std,
                               alpha=0.2, color=color)
                
                # Plot backward (CLIP → DinoV2)
                ax.plot(x, backward_mean, '--', color=MODEL_COLORS['clip_image'], linewidth=2,
                       label='CLIP-Image → DinoV2')
                ax.fill_between(x,
                               backward_mean - backward_std,
                               backward_mean + backward_std,
                               alpha=0.2, color=MODEL_COLORS['clip_image'])
                
                # Add statistics
                stats_text = f'Max: {np.max(forward_mean):.3f}, {np.max(backward_mean):.3f}'
                ax.text(0.98, 0.02, stats_text,
                       transform=ax.transAxes, ha='right', va='bottom',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', alpha=0.7))
            else:
                ax.text(0.5, 0.5, f'Bootstrap data not found\nfor {layer_name} layer', 
                       transform=ax.transAxes, ha='center', va='center')
        else:
            # Load raw safetensor files if no bootstrap
            layer_dir = multilayer_base / f"dino_{layer_name}_layer{layer_id}"
            if layer_dir.exists():
                ax.text(0.5, 0.5, f'Raw data exists\nRun bootstrap_multilayer.py\nfor confidence bands', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, color='orange')
            else:
                ax.text(0.5, 0.5, 'Data not found', 
                       transform=ax.transAxes, ha='center', va='center')
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('CLIP-Image Layer (normalized)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Information Imbalance', fontsize=11)
        ax.set_ylim(global_y_min, global_y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.suptitle('DinoV2 Multi-Layer Analysis vs CLIP-Image\n5000 samples, 200 tokens', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'dino_multilayer_analysis'
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # Also create a comparison plot showing all three layers on one axis
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for layer_name, layer_id, title in layer_configs:
        if use_bootstrap:
            key = f"dino_{layer_name}_layer{layer_id}"
            if key in multilayer_data:
                data = multilayer_data[key]
                forward_mean = np.array(data['forward_mean'])
                backward_mean = np.array(data['backward_mean'])
                
                n_layers = len(forward_mean)
                x = np.linspace(0., 1., n_layers)
                
                color = layer_colors[layer_name]
                
                # Forward plot
                ax1.plot(x, forward_mean, '-', color=color, linewidth=2,
                        label=f'Layer {layer_id}', marker='o', markersize=3, markevery=5)
                
                # Backward plot
                ax2.plot(x, backward_mean, '-', color=color, linewidth=2,
                        label=f'Layer {layer_id}', marker='s', markersize=3, markevery=5)
    
    ax1.set_title('DinoV2 → CLIP-Image', fontsize=14)
    ax1.set_xlabel('CLIP-Image Layer (normalized)', fontsize=11)
    ax1.set_ylabel('Information Imbalance', fontsize=11)
    ax1.set_ylim(global_y_min, global_y_max)
    ax1.legend(title='DinoV2 Layer', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('CLIP-Image → DinoV2', fontsize=14)
    ax2.set_xlabel('CLIP-Image Layer (normalized)', fontsize=11)
    ax2.set_ylabel('Information Imbalance', fontsize=11)
    ax2.set_ylim(global_y_min, global_y_max)
    ax2.legend(title='DinoV2 Layer', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('DinoV2 Layer Comparison\n5000 samples, 200 tokens', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path2 = output_dir / 'dino_layer_comparison'
    fig2.savefig(output_path2.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(output_path2.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path2}")


def main():
    parser = argparse.ArgumentParser(description="Plot Information Imbalance results")
    parser.add_argument("--results_dir", type=str,
                       default="/home/laios/image-information-imbalance/results",
                       help="Path to results directory")
    parser.add_argument("--output_dir", type=str,
                       default="/home/mascaretti/deepseek-hidden-states/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("Information Imbalance Plotting")
    print("=" * 50)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create all plots
    print("\nCreating individual comparison plots...")
    plot_all_comparisons(results_dir, output_dir)
    
    print("\nCreating 2x2 summary grid with bootstrap bands...")
    plot_summary_grid(results_dir, output_dir)
    
    print("\nCreating DinoV2 multi-layer analysis plots...")
    plot_dino_multilayer(results_dir, output_dir)
    
    print("\nDone! Check the output directory for plots.")
    print("\nNote: If DeepSeek-DinoV2 data is missing, run:")
    print("python compute_asymmetry_unified.py \\")
    print("    --comparison deepseek_dino \\")
    print("    --model_a deepseek --model_b dino \\")
    print("    --input_path_A /home/laios/activations/flickr/deepseek_all \\")
    print("    --input_path_B /home/laios/image-information-imbalance/facebook-dinov2-large \\")
    print("    --output_folder /home/laios/image-information-imbalance/results/deepseek-dino-ranks")
    
    print("\nIf DinoV2 multi-layer bootstrap is missing, run:")
    print("python bootstrap_multilayer.py")


if __name__ == "__main__":
    main()
