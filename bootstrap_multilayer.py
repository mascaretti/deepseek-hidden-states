#!/usr/bin/env python
"""
Bootstrap analysis for DinoV2 multi-layer results
Processes the safetensors files from multi-layer analysis
"""

import numpy as np
from pathlib import Path
from safetensors.numpy import load_file
from tqdm import tqdm
import argparse
import json


def load_multilayer_data(layer_dir: Path, n_layers_B: int = 24):
    """Load information imbalance data from a single layer directory"""
    inf_imb = {}
    
    # Load all layer files
    for i in range(1, n_layers_B + 1):
        file_path = layer_dir / f"layer_{i}.safetensors"
        if file_path.exists():
            inf_imb[f"layer_{i}"] = load_file(file_path)
        else:
            print(f"Warning: {file_path} not found")
    
    return inf_imb


def extract_matrices(inf_imb, n_layers):
    """Extract information and multiplicity matrices from loaded data"""
    if not inf_imb:
        return None, None, 0
        
    # Get number of rows from first layer
    first_key = list(inf_imb.keys())[0]
    n_rows = inf_imb[first_key].get('ranks').shape[0]
    
    inf_matrix = np.empty((n_layers, n_rows, 2))
    mult_matrix = np.empty((n_layers, n_rows, 2))
    
    for i in range(n_layers):
        layer_key = f"layer_{i+1}"
        if layer_key in inf_imb:
            inf_matrix[i, :, 0] = inf_imb[layer_key]["ranks"]
            inf_matrix[i, :, 1] = inf_imb[layer_key]["r_ranks"]
            mult_matrix[i, :, 0] = inf_imb[layer_key]["cards"]
            mult_matrix[i, :, 1] = inf_imb[layer_key]["r_cards"]
    
    return inf_matrix, mult_matrix, n_rows


def bootstrap_layer(inf_matrix, mult_matrix, n_layers, n_rows, sample_size, 
                   n_replicas=500, bootstrap_size=500, seed=129):
    """Perform bootstrap analysis for a single layer"""
    n_cols = sample_size - n_rows
    rng = np.random.default_rng(seed)
    
    inf_forward = np.empty((n_layers, n_replicas))
    inf_backward = np.empty((n_layers, n_replicas))
    
    for layer_id in tqdm(range(n_layers), desc="Bootstrap layers", leave=False):
        ranks_forward = inf_matrix[layer_id, :, 0]
        ranks_backward = inf_matrix[layer_id, :, 1]
        cards_forward = mult_matrix[layer_id, :, 0]
        cards_backward = mult_matrix[layer_id, :, 1]
        
        for i in range(n_replicas):
            indices = rng.choice(n_rows, size=bootstrap_size, replace=True)
            inf_forward[layer_id, i] = 2. * (np.sum(ranks_forward[indices]) / 
                                           np.sum(cards_forward[indices])) / (n_cols - 2.)
            inf_backward[layer_id, i] = 2. * (np.sum(ranks_backward[indices]) / 
                                            np.sum(cards_backward[indices])) / (n_cols - 2.)
    
    return inf_forward, inf_backward


def process_multilayer_results(multilayer_dir: Path, output_dir: Path, 
                             sample_size: int = 5000, n_layers_B: int = 24):
    """Process all layers in the multi-layer directory"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all layer directories
    layer_dirs = [d for d in multilayer_dir.iterdir() if d.is_dir() and "layer" in d.name]
    layer_dirs.sort()
    
    print(f"Found {len(layer_dirs)} layer directories")
    
    results = {}
    
    for layer_dir in layer_dirs:
        print(f"\nProcessing {layer_dir.name}...")
        
        # Extract layer info from directory name (e.g., "dino_first_layer1")
        parts = layer_dir.name.split('_')
        layer_name = parts[1]  # first, middle, last
        layer_id = int(parts[2].replace('layer', ''))
        
        # Load data
        inf_imb = load_multilayer_data(layer_dir, n_layers_B)
        inf_matrix, mult_matrix, n_rows = extract_matrices(inf_imb, n_layers_B)
        
        if inf_matrix is None:
            print(f"  Skipping {layer_dir.name} - no data found")
            continue
        
        print(f"  Loaded data: {n_rows} origins, {n_layers_B} layers")
        
        # Perform bootstrap
        print(f"  Running bootstrap analysis...")
        forward_data, backward_data = bootstrap_layer(
            inf_matrix, mult_matrix, n_layers_B, n_rows, sample_size
        )
        
        # Save results
        forward_file = output_dir / f"dino_{layer_name}_layer{layer_id}_forward.npy"
        backward_file = output_dir / f"clip_image_dino_{layer_name}_layer{layer_id}_backward.npy"
        
        np.save(forward_file, forward_data)
        np.save(backward_file, backward_data)
        
        # Store results for summary
        results[f"dino_{layer_name}_layer{layer_id}"] = {
            'forward_mean': np.mean(forward_data, axis=1).tolist(),
            'forward_std': np.std(forward_data, axis=1).tolist(),
            'backward_mean': np.mean(backward_data, axis=1).tolist(),
            'backward_std': np.std(backward_data, axis=1).tolist(),
            'layer_name': layer_name,
            'layer_id': layer_id,
            'n_layers': n_layers_B,
            'n_origins': n_rows,
            'sample_size': sample_size
        }
        
        print(f"  Saved bootstrap results")
        print(f"  Forward mean II: {np.mean(results[f'dino_{layer_name}_layer{layer_id}']['forward_mean']):.3f}")
        print(f"  Backward mean II: {np.mean(results[f'dino_{layer_name}_layer{layer_id}']['backward_mean']):.3f}")
    
    # Save summary
    summary_file = output_dir / "multilayer_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved summary to {summary_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap analysis for multi-layer results")
    parser.add_argument("--input_dir", type=str,
                       default="/home/laios/image-information-imbalance/results/dino-clip-image-multilayer-5000",
                       help="Path to multi-layer results")
    parser.add_argument("--output_dir", type=str,
                       default="/home/laios/image-information-imbalance/results/bootstrap/multilayer",
                       help="Output directory for bootstrap results")
    parser.add_argument("--sample_size", type=int, default=5000,
                       help="Original sample size")
    parser.add_argument("--n_layers", type=int, default=24,
                       help="Number of layers in model B (CLIP)")
    
    args = parser.parse_args()
    
    print("DinoV2 Multi-layer Bootstrap Analysis")
    print("=" * 50)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Sample size: {args.sample_size}")
    print(f"Target model layers: {args.n_layers}")
    
    results = process_multilayer_results(
        Path(args.input_dir),
        Path(args.output_dir),
        args.sample_size,
        args.n_layers
    )
    
    print("\nBootstrap analysis complete!")


if __name__ == "__main__":
    main()
