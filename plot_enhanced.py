#!/usr/bin/env python
"""
Enhanced plotting script with multi-layer DinoV2 analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple

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

# Enhanced color scheme
MODEL_COLORS = {
    'dino': '#0A5EB0',
    'dino_first': '#87CEEB',    # Light blue
    'dino_middle': '#4682B4',   # Steel blue  
    'dino_last': '#000080',     # Navy blue
    'clip_image': '#DC143C',
    'clip_text': '#FF8C00',
    'deepseek': '#2E8B57'
}


def plot_dino_multilayer_detailed(bootstrap_dir: Path, output_path: Path):
    """Create detailed DinoV2 multi-layer analysis plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Check for multi-layer bootstrap results
    multilayer_dir = bootstrap_dir / "multilayer"
    
    if not multilayer_dir.exists():
        print("Multi-layer bootstrap not found. Run bootstrap_multilayer.py first.")
        return
    
    # Load multi-layer summary
    summary_file = multilayer_dir / "multilayer_summary.json"
    if not summary_file.exists():
        print("Multi-layer summary not found.")
        return
        
    with open(summary_file, 'r') as f:
        multilayer_results = json.load(f)
    
    # Left plot: Forward direction (DinoV2 → CLIP)
    # Right plot: Backward direction (CLIP → DinoV2)
    
    layer_configs = [
        ('first', 1, 'Early Layer (1)', '-o'),
        ('middle', 12, 'Middle Layer (12)', '-s'),
        ('last', 24, 'Late Layer (24)', '-^')
    ]
    
    for layer_name, layer_id, label, style in layer_configs:
        key = f"dino_{layer_name}_layer{layer_id}"
        if key in multilayer_results:
            data = multilayer_results[key]
            
            forward_mean = np.array(data['forward_mean'])
            forward_std = np.array(data['forward_std'])
            backward_mean = np.array(data['backward_mean'])
            backward_std = np.array(data['backward_std'])
            
            n_layers = len(forward_mean)
            x = np.linspace(0., 1., n_layers)
            
            color = MODEL_COLORS[f'dino_{layer_name}']
            
            # Forward plot
            ax1.plot(x, forward_mean, style, color=color, label=label,
                    linewidth=2, markersize=4, markeredgecolor='black', 
                    markeredgewidth=0.5, markevery=3)
            ax1.fill_between(x, 
                           forward_mean - forward_std,
                           forward_mean + forward_std,
                           alpha=0.15, linewidth=0, color=color)
            
            # Backward plot
            ax2.plot(x, backward_mean, style, color=color, label=label,
                    linewidth=2, markersize=4, markeredgecolor='black',
                    markeredgewidth=0.5, markevery=3)
            ax2.fill_between(x,
                           backward_mean - backward_std,
                           backward_mean + backward_std,
                           alpha=0.15, linewidth=0, color=color)
    
    # Customize forward plot
    ax1.set_title('DinoV2 → CLIP-Image', fontsize=14)
    ax1.set_xlabel('CLIP-Image Layer (normalized)', fontsize=12)
    ax1.set_ylabel('Information Imbalance', fontsize=12)
    ax1.set_ylim(0.3, 0.9)
    ax1.legend(title='DinoV2 Layer', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Customize backward plot
    ax2.set_title('CLIP-Image → DinoV2', fontsize=14)
    ax2.set_xlabel('CLIP-Image Layer (normalized)', fontsize=12)
    ax2.set_ylabel('Information Imbalance', fontsize=12)
    ax2.set_ylim(0.3, 0.9)
    ax2.legend(title='DinoV2 Layer', fontsize=10, title_fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Layer-wise Information Imbalance: DinoV2 vs CLIP-Image\n(5000 samples, 200 tokens)',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path / "dino_multilayer_detailed.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(output_path / "dino_multilayer_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/dino_multilayer_detailed.pdf")


def create_individual_plots(results: Dict, summaries: Dict, output_path: Path):
    """Create individual plots for each comparison"""
    
    # Create subdirectory for individual plots
    individual_dir = output_path / "individual"
    individual_dir.mkdir(exist_ok=True)
    
    comparisons = [
        ('deepseek_clip_text', 'deepseek', 'clip_text', 52, 9, 'DeepSeek-V3 (Layer 52) vs CLIP-Text'),
        ('clip_text_clip_image', 'clip_text', 'clip_image', 12, 30, 'CLIP-Text (Layer 12) vs CLIP-Image'),
        ('dino_clip_image', 'dino', 'clip_image', 24, 200, 'DinoV2 (Layer 24) vs CLIP-Image')
    ]
    
    for comp_key, model_a, model_b, layer, tokens, title in comparisons:
        forward_key = f"{model_a}_{model_b}_52"
        backward_key = f"{model_b}_{model_a}_52"
        
        if forward_key not in results or backward_key not in results:
            print(f"Skipping {comp_key} - data not found")
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        plot_single_comparison(
            ax,
            results[forward_key],
            results[backward_key],
            model_a, model_b,
            layer, tokens,
            show_bands=True
        )
        
        # Add detailed title with all parameters
        full_title = f"{title}\nSamples: 5000, Tokens: {tokens}, Source Layer: {layer}"
        ax.set_title(full_title, fontsize=14, pad=15)
        ax.set_xlabel("Target Model Relative Depth", fontsize=12)
        ax.set_ylabel("Information Imbalance", fontsize=12)
        ax.set_ylim(0.2, 1.0)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        if comp_key in summaries:
            summary = summaries[comp_key]
            forward_mean_vals = summary.get('forward_mean', [])
            backward_mean_vals = summary.get('backward_mean', [])
            
            if forward_mean_vals and backward_mean_vals:
                stats_text = (
                    f"Forward: max={np.max(forward_mean_vals):.3f}, "
                    f"mean={np.mean(forward_mean_vals):.3f}\n"
                    f"Backward: max={np.max(backward_mean_vals):.3f}, "
                    f"mean={np.mean(backward_mean_vals):.3f}"
                )
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        filename = f"{comp_key}_detailed"
        fig.savefig(individual_dir / f"{filename}.pdf", dpi=300, bbox_inches='tight')
        fig.savefig(individual_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {individual_dir}/{filename}.pdf")


def create_layer_comparison_plot(bootstrap_dir: Path, output_path: Path):
    """Create a plot comparing different layer choices"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load regular bootstrap results
    regular_results = {}
    for file in bootstrap_dir.glob("*.npy"):
        if "multilayer" not in str(file) and "summary" not in file.stem:
            regular_results[file.stem] = np.load(file)
    
    # Load multilayer results if available
    multilayer_summary = bootstrap_dir / "multilayer" / "multilayer_summary.json"
    if multilayer_summary.exists():
        with open(multilayer_summary, 'r') as f:
            multilayer_data = json.load(f)
    else:
        multilayer_data = {}
    
    # Plot comparison of different layer choices for DinoV2
    comparisons = []
    
    # Add standard DinoV2 layer 24 result
    if 'dino_clip_image_52' in regular_results:
        data = regular_results['dino_clip_image_52']
        mean_val = np.mean(np.mean(data, axis=1))
        comparisons.append(('DinoV2 L24\n(Standard)', mean_val, MODEL_COLORS['dino']))
    
    # Add multi-layer results
    for layer_type in ['first', 'middle', 'last']:
        for layer_id in [1, 12, 24]:
            key = f"dino_{layer_type}_layer{layer_id}"
            if key in multilayer_data:
                mean_val = np.mean(multilayer_data[key]['forward_mean'])
                comparisons.append((f'DinoV2 L{layer_id}\n({layer_type.title()})', 
                                  mean_val, 
                                  MODEL_COLORS[f'dino_{layer_type}']))
    
    if comparisons:
        labels, values, colors = zip(*comparisons)
        x_pos = np.arange(len(labels))
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Layer Configuration', fontsize=12)
        ax.set_ylabel('Mean Information Imbalance', fontsize=12)
        ax.set_title('Impact of Layer Choice on Information Imbalance\nDinoV2 → CLIP-Image',
                    fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_path / "layer_comparison.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(output_path / "layer_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/layer_comparison.pdf")


def main():
    """Enhanced main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced plots")
    parser.add_argument("--results_dir", type=str,
                       default="/home/laios/image-information-imbalance/results",
                       help="Path to results directory")
    parser.add_argument("--output_dir", type=str,
                       default="/home/mascaretti/deepseek-hidden-states/plots",
                       help="Output directory for plots")
    parser.add_argument("--run_bootstrap_multilayer", action='store_true',
                       help="Run bootstrap for multilayer results first")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bootstrap_dir = results_dir / "bootstrap"
    
    # Run multilayer bootstrap if requested
    if args.run_bootstrap_multilayer:
        print("Running bootstrap for multi-layer results...")
        import subprocess
        subprocess.run([
            "python", "bootstrap_multilayer.py",
            "--input_dir", str(results_dir / "dino-clip-image-multilayer-5000"),
            "--output_dir", str(bootstrap_dir / "multilayer")
        ])
    
    # Load standard bootstrap results
    print("Loading bootstrap results...")
    results = {}
    summaries = {}
    
    for file in bootstrap_dir.glob("*.npy"):
        if "summary" not in file.stem:
            results[file.stem] = np.load(file)
    
    for file in bootstrap_dir.glob("summary_*.json"):
        with open(file, 'r') as f:
            key = file.stem.replace("summary_", "")
            summaries[key] = json.load(f)
    
    print(f"Found {len(results)} bootstrap results")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # 1. Main comparison plot (from previous script)
    print("1. Creating main comparison plot...")
    from plot_all_results import create_main_comparison_plot
    create_main_comparison_plot(results, output_dir)
    
    # 2. Individual detailed plots
    print("2. Creating individual plots...")
    create_individual_plots(results, summaries, output_dir)
    
    # 3. Enhanced multi-layer plot
    print("3. Creating enhanced multi-layer plot...")
    plot_dino_multilayer_detailed(bootstrap_dir, output_dir)
    
    # 4. Layer comparison plot
    print("4. Creating layer comparison plot...")
    create_layer_comparison_plot(bootstrap_dir, output_dir)
    
    # 5. Summary table (from previous script)
    print("5. Creating summary table...")
    from plot_all_results import create_summary_table
    create_summary_table(summaries, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.pdf")):
        print(f"  - {file.name}")
    
    individual_files = list((output_dir / "individual").glob("*.pdf"))
    if individual_files:
        print(f"\nIndividual plots: {len(individual_files)} files in {output_dir}/individual/")


if __name__ == "__main__":
    main()