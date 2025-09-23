#!/usr/bin/env python
"""
Quick comparison script for DeepSeekV3 vs LLama3.1-8B
Reuses existing code from compute_asymmetry_unified.py
For NeurIPS rebuttal - minimal changes, maximum reuse
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add current directory to path to import local modules
sys.path.insert(0, '.')

# Import everything from existing scripts
from compute_asymmetry_unified import *
from bootstrap_analysis import *
import matplotlib.pyplot as plt
import numpy as np
import json

def quick_test_data_loading(deepseek_path, llama_path, test_size=100):
    """Quick test to verify data loading works"""
    print("\n" + "="*60)
    print("QUICK DATA LOADING TEST")
    print("="*60)
    
    # Test DeepSeek loading
    print("\nTesting DeepSeek data loading...")
    start = time.time()
    try:
        ds_test = collect_deepseek_data(deepseek_path, n_loaded_tokens=9, subset_size=test_size)
        print(f"✓ DeepSeek loaded in {time.time()-start:.1f}s")
        print(f"  Shape: {ds_test['layer_0'].shape}")
        print(f"  Layers: {len(ds_test)}")
        print(f"  Layer 52 exists: {'layer_52' in ds_test}")
    except Exception as e:
        print(f"✗ DeepSeek loading failed: {e}")
        return False
    
    # Test LLama loading (assuming similar structure to DeepSeek)
    print("\nTesting LLama data loading...")
    start = time.time()
    try:
        # Try to load LLama data - assuming same format as DeepSeek
        llama_test = collect_deepseek_data(llama_path, n_loaded_tokens=9, subset_size=test_size)
        print(f"✓ LLama loaded in {time.time()-start:.1f}s")
        print(f"  Shape: {llama_test['layer_0'].shape}")
        print(f"  Layers: {len(llama_test)}")
        
        # Find the last layer for LLama
        llama_layers = sorted([int(k.split('_')[1]) for k in llama_test.keys()])
        print(f"  Layer range: 0-{llama_layers[-1]}")
    except Exception as e:
        print(f"✗ LLama loading failed: {e}")
        return False
    
    print("\n✓ Data loading test passed!")
    return True


def run_deepseek_llama_comparison(args):
    """Main comparison function"""
    
    print("\n" + "="*60)
    print("DEEPSEEK V3 vs LLAMA 3.1-8B COMPARISON")
    print("="*60)
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"Subset size: {args.subset_size}")
    print(f"Output: {args.output_folder}")
    print(f"Include validation: {args.include_validation}")
    
    # Create output directory
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    # Start timing
    total_start = time.time()
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    load_start = time.time()
    
    print(f"Loading DeepSeek from: {args.deepseek_path}")
    all_activations_deepseek = collect_deepseek_data(
        args.deepseek_path, 
        n_loaded_tokens=9, 
        subset_size=args.subset_size
    )
    
    print(f"Loading LLama from: {args.llama_path}")
    all_activations_llama = collect_deepseek_data(
        args.llama_path, 
        n_loaded_tokens=9, 
        subset_size=args.subset_size
    )
    
    print(f"Data loading completed in {time.time()-load_start:.1f}s")
    
    # Verify shapes
    ds_shape = all_activations_deepseek["layer_0"].shape
    llama_shape = all_activations_llama["layer_0"].shape
    print(f"\nDeepSeek shape: {ds_shape}")
    print(f"LLama shape: {llama_shape}")
    
    if ds_shape[0] != llama_shape[0]:
        print(f"WARNING: Sample sizes don't match! DS: {ds_shape[0]}, LLama: {llama_shape[0]}")
        # Take minimum
        min_samples = min(ds_shape[0], llama_shape[0])
        print(f"Using {min_samples} samples")
        for key in all_activations_deepseek:
            all_activations_deepseek[key] = all_activations_deepseek[key][:min_samples]
        for key in all_activations_llama:
            all_activations_llama[key] = all_activations_llama[key][:min_samples]
    
    # Step 2: Compute information imbalance
    print("\nStep 2: Computing information imbalance...")
    compute_start = time.time()
    
    # Use the existing compute_inf_imb_matrix function
    compute_inf_imb_matrix(
        all_activations_A=all_activations_deepseek,
        all_activations_B=all_activations_llama,
        output_folder=args.output_folder,
        layer_A_id=52,  # DeepSeek layer 52
        comparison_type="text_text"  # Both are text models
    )
    
    print(f"Information imbalance computed in {time.time()-compute_start:.1f}s")
    
    # Step 3: Bootstrap analysis
    print("\nStep 3: Running bootstrap analysis...")
    bootstrap_start = time.time()
    
    # Determine number of LLama layers
    llama_layers = len(all_activations_llama)
    print(f"LLama has {llama_layers} layers")
    
    # Run bootstrap using existing script
    bootstrap_output = Path(args.output_folder) / "bootstrap"
    bootstrap_output.mkdir(exist_ok=True)
    
    # Load the safetensors files we just created
    inf_imb = load_inf_imb_data(args.output_folder, llama_layers - 1)  # -1 because we skip layer 0
    inf_matrix, mult_matrix, n_rows = extract_matrices(inf_imb, llama_layers - 1)
    
    # Compute mean information imbalance
    forward_means, backward_means = compute_mean_inf_imb(
        inf_matrix, mult_matrix, llama_layers - 1, n_rows, args.subset_size
    )
    
    print(f"Mean II - DeepSeek → LLama: {np.mean(forward_means):.3f}")
    print(f"Mean II - LLama → DeepSeek: {np.mean(backward_means):.3f}")
    
    # Perform bootstrap
    rng = np.random.default_rng(args.seed)
    forward_bootstrap, backward_bootstrap = bootstrap(
        rng, inf_matrix, mult_matrix,
        args.n_replicas, args.bootstrap_size,
        llama_layers - 1, n_rows, args.subset_size
    )
    
    # Save bootstrap results
    np.save(bootstrap_output / "deepseek_llama_52.npy", forward_bootstrap)
    np.save(bootstrap_output / "llama_deepseek_52.npy", backward_bootstrap)
    
    print(f"Bootstrap completed in {time.time()-bootstrap_start:.1f}s")
    
    # Step 4: Generate plot
    print("\nStep 4: Generating plot...")
    plot_start = time.time()
    
    generate_comparison_plot(
        forward_bootstrap, 
        backward_bootstrap,
        output_path=args.output_folder,
        llama_layers=llama_layers - 1
    )
    
    print(f"Plot generated in {time.time()-plot_start:.1f}s")
    
    # Total time
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")
    
    # Save summary
    summary = {
        'model_a': 'deepseek',
        'model_b': 'llama',
        'layer_a': 52,
        'n_layers_b': llama_layers - 1,
        'sample_size': args.subset_size,
        'n_replicas': args.n_replicas,
        'bootstrap_size': args.bootstrap_size,
        'forward_mean': forward_means.tolist(),
        'backward_mean': backward_means.tolist(),
        'forward_bootstrap_mean': np.mean(forward_bootstrap, axis=1).tolist(),
        'backward_bootstrap_mean': np.mean(backward_bootstrap, axis=1).tolist(),
        'total_time_seconds': total_time
    }
    
    with open(Path(args.output_folder) / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {args.output_folder}")
    
    # Step 5: Validation - DeepSeek vs DeepSeek (self-comparison)
    if args.include_validation:
        print("\n" + "="*60)
        print("VALIDATION: DeepSeek Layer 52 vs all DeepSeek layers")
        print("="*60)
        
        validation_start = time.time()
        validation_output = Path(args.output_folder) / "validation_deepseek_self"
        validation_output.mkdir(exist_ok=True)
        
        print("Computing self-comparison for validation...")
        
        # Compute information imbalance for DeepSeek vs itself
        compute_inf_imb_matrix(
            all_activations_A=all_activations_deepseek,
            all_activations_B=all_activations_deepseek,
            output_folder=validation_output,
            layer_A_id=52,
            comparison_type="text_text"
        )
        
        # Quick analysis of layer 52 vs layer 52
        layer_52_file = validation_output / "layer_52.safetensors"
        if layer_52_file.exists():
            from safetensors.numpy import load_file
            layer_52_data = load_file(layer_52_file)
            
            # Calculate information imbalance for layer 52
            ranks = layer_52_data['ranks']
            cards = layer_52_data['cards']
            r_ranks = layer_52_data['r_ranks']
            r_cards = layer_52_data['r_cards']
            
            n_cols = args.subset_size - len(ranks)
            forward_ii = 2. * (np.sum(ranks) / np.sum(cards)) / (n_cols - 2.)
            backward_ii = 2. * (np.sum(r_ranks) / np.sum(r_cards)) / (n_cols - 2.)
            
            print(f"\nVALIDATION RESULT - Layer 52 vs Layer 52:")
            print(f"  Forward II (should be ~0): {forward_ii:.6f}")
            print(f"  Backward II (should be ~0): {backward_ii:.6f}")
            
            if abs(forward_ii) > 0.01 or abs(backward_ii) > 0.01:
                print("  ⚠️ WARNING: Self-comparison values larger than expected!")
            else:
                print("  ✓ Validation passed: Self-comparison near zero as expected")
        
        # Generate validation plot
        print("\nGenerating validation plot...")
        deepseek_layers = len(all_activations_deepseek)
        
        # Load validation data
        val_inf_imb = load_inf_imb_data(validation_output, deepseek_layers - 1)
        val_inf_matrix, val_mult_matrix, val_n_rows = extract_matrices(val_inf_imb, deepseek_layers - 1)
        
        # Compute means
        val_forward_means, val_backward_means = compute_mean_inf_imb(
            val_inf_matrix, val_mult_matrix, deepseek_layers - 1, val_n_rows, args.subset_size
        )
        
        # Bootstrap for validation
        val_rng = np.random.default_rng(args.seed)
        val_forward_bootstrap, val_backward_bootstrap = bootstrap(
            val_rng, val_inf_matrix, val_mult_matrix,
            min(100, args.n_replicas), args.bootstrap_size,  # Fewer replicas for validation
            deepseek_layers - 1, val_n_rows, args.subset_size
        )
        
        # Generate validation plot
        generate_validation_plot(
            val_forward_bootstrap,
            val_backward_bootstrap,
            output_path=args.output_folder,
            deepseek_layers=deepseek_layers - 1,
            layer_52_ii=(forward_ii, backward_ii)
        )
        
        print(f"Validation completed in {time.time()-validation_start:.1f}s")
    
    print("\n✓ All analyses complete!")


def generate_comparison_plot(forward_data, backward_data, output_path, llama_layers):
    """Generate publication-ready plot"""
    
    # Calculate statistics
    forward_mean = np.mean(forward_data, axis=1)
    forward_std = np.std(forward_data, axis=1)
    backward_mean = np.mean(backward_data, axis=1)
    backward_std = np.std(backward_data, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X-axis: normalized layer depth
    x = np.linspace(0., 1., llama_layers)
    
    # Plot forward direction (DeepSeek → LLama)
    ax.plot(x, forward_mean, '-o', color='#2E8B57', 
            label='DeepSeek-V3 (L52) → LLama3.1-8B', 
            linewidth=2, markersize=4, markevery=max(1, llama_layers//10))
    ax.fill_between(x, 
                   forward_mean - forward_std,
                   forward_mean + forward_std,
                   alpha=0.2, linewidth=0, color='#2E8B57')
    
    # Plot backward direction (LLama → DeepSeek)
    ax.plot(x, backward_mean, '--s', color='#B22222',
            label='LLama3.1-8B → DeepSeek-V3 (L52)',
            linewidth=2, markersize=4, markevery=max(1, llama_layers//10))
    ax.fill_between(x,
                   backward_mean - backward_std,
                   backward_mean + backward_std,
                   alpha=0.2, linewidth=0, color='#B22222')
    
    # Annotations - highlight MINIMUM values
    min_forward_idx = np.argmin(forward_mean)
    min_backward_idx = np.argmin(backward_mean)
    
    # Annotate minimum forward
    ax.annotate(f'Min: {forward_mean[min_forward_idx]:.3f}',
                xy=(x[min_forward_idx], forward_mean[min_forward_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='#2E8B57',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2E8B57', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=1.5))
    
    # Annotate minimum backward
    ax.annotate(f'Min: {backward_mean[min_backward_idx]:.3f}',
                xy=(x[min_backward_idx], backward_mean[min_backward_idx]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, color='#B22222',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B22222', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.5))
    
    # Add markers at minimum points
    ax.plot(x[min_forward_idx], forward_mean[min_forward_idx], 'o', 
            color='#2E8B57', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.plot(x[min_backward_idx], backward_mean[min_backward_idx], 's', 
            color='#B22222', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    # Formatting
    ax.set_xlabel('LLama3.1-8B Relative Layer Depth', fontsize=14)
    ax.set_ylabel('Information Imbalance', fontsize=14)
    ax.set_title('Information Imbalance: DeepSeek-V3 vs LLama3.1-8B\n(5000 samples, 9 tokens, 500 bootstrap iterations)', 
                 fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add text box with key statistics
    stats_text = (
        f"Mean II (DS→LLama): {np.mean(forward_mean):.3f} ± {np.mean(forward_std):.3f}\n"
        f"Mean II (LLama→DS): {np.mean(backward_mean):.3f} ± {np.mean(backward_std):.3f}\n"
        f"Min II (DS→LLama): {forward_mean[min_forward_idx]:.3f} at layer {min_forward_idx}\n"
        f"Min II (LLama→DS): {backward_mean[min_backward_idx]:.3f} at layer {min_backward_idx}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    fig.savefig(output_path / "deepseek_llama_comparison.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(output_path / "deepseek_llama_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {output_path}/deepseek_llama_comparison.pdf")


def generate_validation_plot(forward_data, backward_data, output_path, deepseek_layers, layer_52_ii):
    """Generate validation plot for DeepSeek self-comparison"""
    
    # Calculate statistics
    forward_mean = np.mean(forward_data, axis=1)
    forward_std = np.std(forward_data, axis=1)
    backward_mean = np.mean(backward_data, axis=1)
    backward_std = np.std(backward_data, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X-axis: normalized layer depth
    x = np.linspace(0., 1., deepseek_layers)
    
    # Plot forward direction
    ax.plot(x, forward_mean, '-o', color='#4169E1', 
            label='DeepSeek L52 → DeepSeek (all layers)', 
            linewidth=2, markersize=4, markevery=max(1, deepseek_layers//10))
    ax.fill_between(x, 
                   forward_mean - forward_std,
                   forward_mean + forward_std,
                   alpha=0.2, linewidth=0, color='#4169E1')
    
    # Plot backward direction
    ax.plot(x, backward_mean, '--s', color='#DC143C',
            label='DeepSeek (all layers) → DeepSeek L52',
            linewidth=2, markersize=4, markevery=max(1, deepseek_layers//10))
    ax.fill_between(x,
                   backward_mean - backward_std,
                   backward_mean + backward_std,
                   alpha=0.2, linewidth=0, color='#DC143C')
    
    # Highlight minimums
    min_forward_idx = np.argmin(forward_mean)
    min_backward_idx = np.argmin(backward_mean)
    
    # Add markers at minimum points
    ax.plot(x[min_forward_idx], forward_mean[min_forward_idx], 'o', 
            color='#4169E1', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.plot(x[min_backward_idx], backward_mean[min_backward_idx], 's', 
            color='#DC143C', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    # Mark layer 52
    layer_52_idx = 51  # 0-indexed
    if layer_52_idx < len(x):
        ax.axvline(x[layer_52_idx], color='green', linestyle=':', alpha=0.7, label='Layer 52')
        
        # Annotate the actual II values at layer 52
        ax.annotate(f'L52→L52: {layer_52_ii[0]:.6f}',
                    xy=(x[layer_52_idx], 0.1),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=10, color='green',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('DeepSeek Target Layer (normalized)', fontsize=14)
    ax.set_ylabel('Information Imbalance', fontsize=14)
    ax.set_title('VALIDATION: DeepSeek Layer 52 Self-Comparison\n(Should be ~0 at Layer 52)', 
                 fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)
    
    # Add text box with validation info
    validation_text = (
        f"Layer 52 vs Layer 52:\n"
        f"Forward II: {layer_52_ii[0]:.6f}\n"
        f"Backward II: {layer_52_ii[1]:.6f}\n"
        f"{'✓ PASS' if abs(layer_52_ii[0]) < 0.01 else '⚠️ WARNING'}\n\n"
        f"Min Forward: {forward_mean[min_forward_idx]:.3f} (L{min_forward_idx+1})\n"
        f"Min Backward: {backward_mean[min_backward_idx]:.3f} (L{min_backward_idx+1})"
    )
    ax.text(0.02, 0.98, validation_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen' if abs(layer_52_ii[0]) < 0.01 else 'lightcoral', 
                     alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    fig.savefig(output_path / "validation_deepseek_self.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(output_path / "validation_deepseek_self.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plot saved: {output_path}/validation_deepseek_self.pdf")


def main():
    parser = argparse.ArgumentParser(description="Compare DeepSeek-V3 and LLama3.1-8B representations")
    
    # Paths
    parser.add_argument("--deepseek_path", type=str, 
                       default="/home/laios/activations/flickr/deepseek_all",
                       help="Path to DeepSeek activations")
    parser.add_argument("--llama_path", type=str,
                       default="/home/laios/llama_flickr_cpt_all",
                       help="Path to LLama activations")
    parser.add_argument("--output_folder", type=str,
                       default="/home/laios/image-information-imbalance/results/deepseek-llama-comparison",
                       help="Output folder for results")
    
    # Parameters
    parser.add_argument("--subset_size", type=int, default=5000,
                       help="Number of samples to use")
    parser.add_argument("--n_replicas", type=int, default=500,
                       help="Number of bootstrap replicas")
    parser.add_argument("--bootstrap_size", type=int, default=500,
                       help="Bootstrap sample size")
    parser.add_argument("--seed", type=int, default=129,
                       help="Random seed")
    
    # Test mode
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with reduced data")
    parser.add_argument("--test_size", type=int, default=500,
                       help="Sample size for test mode")
    
    # Validation
    parser.add_argument("--include_validation", action="store_true",
                       help="Include DeepSeek self-comparison validation")
    parser.add_argument("--skip_main", action="store_true",
                       help="Skip main comparison and only run validation")
    
    args = parser.parse_args()
    
    # Override parameters for test mode
    if args.test:
        print("Running in TEST MODE")
        args.subset_size = args.test_size
        args.n_replicas = 50
        args.bootstrap_size = 100
        args.output_folder = args.output_folder + "_test"
    
    # Quick verification
    if not Path(args.deepseek_path).exists():
        print(f"ERROR: DeepSeek path not found: {args.deepseek_path}")
        return 1
    
    if not Path(args.llama_path).exists():
        print(f"ERROR: LLama path not found: {args.llama_path}")
        return 1
    
    # Run quick test first
    if args.test:
        if not quick_test_data_loading(args.deepseek_path, args.llama_path, test_size=50):
            print("Data loading test failed!")
            return 1
    
    # Run main comparison
    try:
        if args.skip_main:
            print("Skipping main comparison, running validation only...")
            # Still need to load DeepSeek data for validation
            print(f"Loading DeepSeek from: {args.deepseek_path}")
            all_activations_deepseek = collect_deepseek_data(
                args.deepseek_path, 
                n_loaded_tokens=9, 
                subset_size=args.subset_size
            )
            
            # Run validation
            if args.include_validation or args.skip_main:
                validation_start = time.time()
                validation_output = Path(args.output_folder) / "validation_deepseek_self"
                validation_output.mkdir(parents=True, exist_ok=True)
                
                print("\nRunning DeepSeek self-comparison validation...")
                
                # Compute information imbalance for DeepSeek vs itself
                compute_inf_imb_matrix(
                    all_activations_A=all_activations_deepseek,
                    all_activations_B=all_activations_deepseek,
                    output_folder=validation_output,
                    layer_A_id=52,
                    comparison_type="text_text"
                )
                
                # Quick analysis of layer 52 vs layer 52
                layer_52_file = validation_output / "layer_52.safetensors"
                if layer_52_file.exists():
                    from safetensors.numpy import load_file
                    layer_52_data = load_file(layer_52_file)
                    
                    # Calculate information imbalance for layer 52
                    ranks = layer_52_data['ranks']
                    cards = layer_52_data['cards']
                    r_ranks = layer_52_data['r_ranks']
                    r_cards = layer_52_data['r_cards']
                    
                    n_cols = args.subset_size - len(ranks)
                    forward_ii = 2. * (np.sum(ranks) / np.sum(cards)) / (n_cols - 2.)
                    backward_ii = 2. * (np.sum(r_ranks) / np.sum(r_cards)) / (n_cols - 2.)
                    
                    print(f"\nVALIDATION RESULT - Layer 52 vs Layer 52:")
                    print(f"  Forward II (should be ~0): {forward_ii:.6f}")
                    print(f"  Backward II (should be ~0): {backward_ii:.6f}")
                    
                    if abs(forward_ii) > 0.01 or abs(backward_ii) > 0.01:
                        print("  ⚠️ WARNING: Self-comparison values larger than expected!")
                    else:
                        print("  ✓ Validation passed: Self-comparison near zero as expected")
                
                # Generate validation plot
                print("\nGenerating validation plot...")
                deepseek_layers = len(all_activations_deepseek)
                
                # Load validation data
                val_inf_imb = load_inf_imb_data(validation_output, deepseek_layers - 1)
                val_inf_matrix, val_mult_matrix, val_n_rows = extract_matrices(val_inf_imb, deepseek_layers - 1)
                
                # Compute means
                val_forward_means, val_backward_means = compute_mean_inf_imb(
                    val_inf_matrix, val_mult_matrix, deepseek_layers - 1, val_n_rows, args.subset_size
                )
                
                # Bootstrap for validation
                val_rng = np.random.default_rng(args.seed)
                val_forward_bootstrap, val_backward_bootstrap = bootstrap(
                    val_rng, val_inf_matrix, val_mult_matrix,
                    min(100, args.n_replicas), args.bootstrap_size,  # Fewer replicas for validation
                    deepseek_layers - 1, val_n_rows, args.subset_size
                )
                
                # Generate validation plot
                generate_validation_plot(
                    val_forward_bootstrap,
                    val_backward_bootstrap,
                    output_path=args.output_folder,
                    deepseek_layers=deepseek_layers - 1,
                    layer_52_ii=(forward_ii, backward_ii)
                )
                
                print(f"Validation completed in {time.time()-validation_start:.1f}s")
        else:
            run_deepseek_llama_comparison(args)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
