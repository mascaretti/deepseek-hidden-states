#!/usr/bin/env python
"""
Unified plotting script for Information Imbalance results
Generates publication-ready plots from bootstrap analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple


def load_bootstrap_data(data_path: Path, models: List[Tuple[str, str]], layer: str = "52"):
    """Load bootstrap results for multiple model pairs."""
    results = {}
    
    for model_a, model_b in models:
        forward_path = Path(data_path, f"{model_a}_{model_b}_{layer}.npy")
        backward_path = Path(data_path, f"{model_b}_{model_a}_{layer}.npy")
        
        if forward_path.exists() and backward_path.exists():
            results[f"{model_a}_{model_b}"] = {
                'forward': np.load(forward_path),
                'backward': np.load(backward_path)
            }
            print(f"Loaded data for {model_a} ↔ {model_b}")
        else:
            print(f"Warning: Missing data for {model_a} ↔ {model_b}")
    
    return results


def load_token_sweep_data(data_paths: Dict[str, Path], model: str):
    """Load token sweep data for a single model."""
    results = {}
    
    for token_count, path in data_paths.items():
        file_path = Path(path, f"mins_{model}.npy")
        if file_path.exists():
            results[token_count] = np.load(file_path)
            print(f"Loaded {token_count} token data for {model}")
    
    return results


def plot_image_caption_comparison(results: Dict, output_path: Path, y_min: float = 0.2, y_max: float = 1.0):
    """Create the main comparison plot for image-caption pairs."""
    plt.style.use(['nature', 'science'])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 2.5))
    ax1, ax2 = axes
    
    # Define colors
    colors = {
        'dino': '#0A5EB0',
        'igpt': '#399918',
        'clip': '#FF6B6B'
    }
    
    # Plot settings
    fnt_labels = 12
    
    # Left plot: Image-Caption
    for model_pair, data in results.items():
        if 'ds' in model_pair:  # DeepSeek results
            model_name = model_pair.replace('ds_', '').replace('_ds', '')
            color = colors.get(model_name, '#808080')
            
            forward_mean = np.mean(data['forward'], axis=1)
            forward_std = np.std(data['forward'], axis=1)
            backward_mean = np.mean(data['backward'], axis=1)
            backward_std = np.std(data['backward'], axis=1)
            
            n_layers = forward_mean.shape[0]
            x = np.linspace(0., 1., n_layers)
            
            # Plot forward direction (model → DeepSeek)
            ax1.plot(x, backward_mean, '--', color=color, 
                    label=f"{model_name} → DeepSeek-V3", 
                    markersize=3, markeredgecolor='black', markeredgewidth=0.3)
            ax1.fill_between(x, 
                           backward_mean - backward_std,
                           backward_mean + backward_std,
                           alpha=0.2, linewidth=0, color=color)
            
            # Plot backward direction (DeepSeek → model)
            ax1.plot(x, forward_mean, '-o', color=color,
                    label=f"DeepSeek-V3 → {model_name}",
                    markersize=3, markeredgecolor='black', markeredgewidth=0.3)
            ax1.fill_between(x,
                           forward_mean - forward_std,
                           forward_mean + forward_std,
                           alpha=0.2, linewidth=0, color=color)
    
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel("Information Imbalance", fontsize=fnt_labels)
    ax1.set_xlabel("Relative depth", fontsize=fnt_labels)
    ax1.legend(fontsize=7.)
    ax1.set_title("Image-Caption", fontsize=fnt_labels)
    
    # Right plot: Image-Image (placeholder for now - you'll need to add this data)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel("Relative depth", fontsize=fnt_labels)
    ax2.set_title("Image-Image", fontsize=fnt_labels)
    ax2.text(0.5, 0.5, "Add Image-Image data", 
             transform=ax2.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save figures
    fig.savefig(Path(output_path, "image-caption-comparison.pdf"), dpi=300, bbox_inches='tight')
    fig.savefig(Path(output_path, "image-caption-comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_path}")


def plot_token_analysis(results: Dict[str, Dict], token_counts: List[str], 
                       output_path: Path, layer: int = 52):
    """Create multi-panel plot for different token counts."""
    plt.style.use(['nature', 'science'])
    
    n_panels = len(token_counts)
    fig, axes = plt.subplots(nrows=1, ncols=n_panels, figsize=(4*n_panels, 3.0))
    
    if n_panels == 1:
        axes = [axes]
    
    colors = {
        'dino': '#0A5EB0',
        'igpt': '#399918',
        'clip': '#FF6B6B'
    }
    
    y_min, y_max = 0.25, 1.0
    fnt_labels = 12
    
    for idx, (ax, token_count) in enumerate(zip(axes, token_counts)):
        for model_pair, data in results.items():
            if token_count in model_pair:
                # Extract model name and get color
                parts = model_pair.split('_')
                model_name = [p for p in parts if p not in ['ds', token_count]][0]
                color = colors.get(model_name, '#808080')
                
                forward_mean = np.mean(data['forward'], axis=1)
                forward_std = np.std(data['forward'], axis=1)
                backward_mean = np.mean(data['backward'], axis=1)
                backward_std = np.std(data['backward'], axis=1)
                
                n_layers = forward_mean.shape[0]
                x = np.linspace(0., 1., n_layers)
                
                # Plot both directions
                ax.plot(x, backward_mean, '--', color=color,
                       label=f"{model_name} → DeepSeek-V3",
                       markersize=3, markeredgecolor='black', markeredgewidth=0.3)
                ax.fill_between(x,
                              backward_mean - backward_std,
                              backward_mean + backward_std,
                              alpha=0.2, linewidth=0, color=color)
                
                ax.plot(x, forward_mean, '-o', color=color,
                       label=f"DeepSeek-V3 → {model_name}",
                       markersize=3, markeredgecolor='black', markeredgewidth=0.3)
                ax.fill_between(x,
                              forward_mean - forward_std,
                              forward_mean + forward_std,
                              alpha=0.2, linewidth=0, color=color)
        
        ax.set_ylim(y_min, y_max)
        if idx == 0:
            ax.set_ylabel("Information Imbalance", fontsize=fnt_labels)
        ax.set_xlabel("Relative depth", fontsize=fnt_labels)
        ax.legend(fontsize=7.5)
        
        # Set title based on token count
        if token_count == "mean":
            ax.set_title("Mean token", fontsize=fnt_labels)
        else:
            ax.set_title(f"{token_count} tokens", fontsize=fnt_labels)
    
    plt.tight_layout()
    
    # Save figures
    fig.savefig(Path(output_path, f"image-caption-tokens-layer{layer}.pdf"), 
                dpi=300, bbox_inches='tight')
    fig.savefig(Path(output_path, f"image-caption-tokens-layer{layer}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved token analysis plot to {output_path}")


def plot_last_tokens_analysis(token_data: Dict, output_path: Path):
    """Create plot for last token analysis."""
    plt.style.use(['nature', 'science'])
    fig, axes = plt.subplots(ncols=2, figsize=(6, 2.5))
    ax1, ax2 = axes
    
    fnt_labels = 8
    
    # You'll need to implement this based on your specific token analysis needs
    # This is a placeholder structure
    
    ax1.set_ylabel("Information Imbalance", fontsize=12)
    ax1.set_xlabel(r"\# Tokens", fontsize=fnt_labels)
    ax1.set_title(r"DinoV2", fontsize=12)
    ax1.legend(fontsize=fnt_labels)
    
    ax2.set_xlabel("N. Tokens", fontsize=fnt_labels)
    ax2.set_title("image-gpt", fontsize=12)
    ax2.legend(fontsize=fnt_labels)
    
    plt.tight_layout()
    
    fig.savefig(Path(output_path, "last-tokens.pdf"), dpi=300, bbox_inches='tight')
    fig.savefig(Path(output_path, "last-tokens.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Information Imbalance results")
    parser.add_argument("--data_path", type=str, required=True, help="Path to bootstrap results")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--models", nargs='+', default=['dino', 'igpt', 'clip'], 
                       help="Model names to plot")
    parser.add_argument("--layer", type=str, default="52", help="Layer number")
    parser.add_argument("--plot_type", choices=['comparison', 'tokens', 'all'], default='all',
                       help="Type of plots to generate")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    # Define model pairs
    model_pairs = [(model, 'ds') for model in args.models]
    
    # Load data
    print("Loading bootstrap results...")
    results = load_bootstrap_data(Path(args.data_path), model_pairs, args.layer)
    
    if not results:
        print("Error: No data loaded. Check your data path and file names.")
        return
    
    # Generate plots based on plot_type
    if args.plot_type in ['comparison', 'all']:
        print("Generating comparison plots...")
        plot_image_caption_comparison(results, Path(args.output_path))
    
    if args.plot_type in ['tokens', 'all']:
        # For token analysis, you might need to load different datasets
        # This is a placeholder - adjust based on your actual data structure
        print("Generating token analysis plots...")
        # plot_token_analysis(results, ['50', '200', 'mean'], Path(args.output_path))
    
    print(f"All plots saved to {args.output_path}")


if __name__ == "__main__":
    main()
