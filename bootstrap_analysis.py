#!/usr/bin/env python
"""
Unified bootstrap analysis script for Information Imbalance computation
Processes results from check_asymmetry*.py scripts and generates plots
"""

import numpy as np
from pathlib import Path
from safetensors.numpy import load_file, save_file
from tqdm import tqdm
import argparse
import json


def load_inf_imb_data(data_path, n_layers):
    """Load information imbalance data from safetensors files."""
    inf_imb = {}
    for i in range(1, n_layers + 1):
        inf_imb[f"layer_{i}"] = load_file(Path(data_path, f"layer_{i}.safetensors"))
    return inf_imb


def extract_matrices(inf_imb, n_layers):
    """Extract information and multiplicity matrices from loaded data."""
    n_rows = inf_imb.get("layer_1").get('ranks').shape[0]
    
    inf_matrix = np.empty((n_layers, n_rows, 2))
    mult_matrix = np.empty((n_layers, n_rows, 2))
    
    for i in range(n_layers):
        inf_matrix[i, :, 0] = inf_imb[f"layer_{i+1}"]["ranks"]
        inf_matrix[i, :, 1] = inf_imb[f"layer_{i+1}"]["r_ranks"]
        mult_matrix[i, :, 0] = inf_imb[f"layer_{i+1}"]["cards"]
        mult_matrix[i, :, 1] = inf_imb[f"layer_{i+1}"]["r_cards"]
    
    return inf_matrix, mult_matrix, n_rows


def compute_mean_inf_imb(inf_matrix, mult_matrix, n_layers, n_rows, sample_size):
    """Compute mean information imbalance for each layer."""
    n_cols = sample_size - n_rows
    
    # Forward direction
    forward_means = np.zeros(n_layers)
    for i in range(n_layers):
        vals = inf_matrix[i, :, 0]
        cards = mult_matrix[i, :, 0]
        forward_means[i] = 2. * (np.sum(vals) / np.sum(cards)) / (n_cols - 2.)
    
    # Backward direction
    backward_means = np.zeros(n_layers)
    for i in range(n_layers):
        vals = inf_matrix[i, :, 1]
        cards = mult_matrix[i, :, 1]
        backward_means[i] = 2. * (np.sum(vals) / np.sum(cards)) / (n_cols - 2.)
    
    return forward_means, backward_means


def bootstrap(rng, inf_matrix, mult_matrix, n_replicas, bootstrap_size, n_layers, n_rows, sample_size):
    """Perform bootstrap analysis on information imbalance data."""
    n_cols = sample_size - n_rows
    
    inf_forward = np.empty((n_layers, n_replicas))
    inf_backward = np.empty((n_layers, n_replicas))
    
    for layer_id in tqdm(np.arange(n_layers), desc="Bootstrap layers"):
        ranks_forward = inf_matrix[layer_id, :, 0]
        ranks_backward = inf_matrix[layer_id, :, 1]
        cards_forward = mult_matrix[layer_id, :, 0]
        cards_backward = mult_matrix[layer_id, :, 1]
        
        for i in range(n_replicas):
            indices = rng.choice(n_rows, size=bootstrap_size, replace=True)
            inf_forward[layer_id, i] = 2. * (np.sum(ranks_forward[indices]) / np.sum(cards_forward[indices])) / (n_cols - 2.)
            inf_backward[layer_id, i] = 2. * (np.sum(ranks_backward[indices]) / np.sum(cards_backward[indices])) / (n_cols - 2.)
    
    return inf_forward, inf_backward


def save_bootstrap_results(output_path, forward_data, backward_data, config):
    """Save bootstrap results and configuration."""
    # Save numpy arrays
    np.save(Path(output_path, f"{config['model_a']}_{config['model_b']}_{config['layer']}.npy"), forward_data)
    np.save(Path(output_path, f"{config['model_b']}_{config['model_a']}_{config['layer']}.npy"), backward_data)
    
    # Save configuration
    with open(Path(output_path, "bootstrap_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save summary statistics
    summary = {
        'forward_mean': np.mean(forward_data, axis=1).tolist(),
        'forward_std': np.std(forward_data, axis=1).tolist(),
        'backward_mean': np.mean(backward_data, axis=1).tolist(),
        'backward_std': np.std(backward_data, axis=1).tolist(),
        'n_layers': forward_data.shape[0],
        'n_replicas': forward_data.shape[1]
    }
    
    with open(Path(output_path, f"summary_{config['model_a']}_{config['model_b']}.json"), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap analysis for Information Imbalance")
    parser.add_argument("--data_path", type=str, required=True, help="Path to safetensors files")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for bootstrap results")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of layers in model B")
    parser.add_argument("--n_replicas", type=int, default=500, help="Number of bootstrap replicas")
    parser.add_argument("--bootstrap_size", type=int, default=500, help="Bootstrap sample size")
    parser.add_argument("--sample_size", type=int, default=5000, help="Original sample size")
    parser.add_argument("--model_a", type=str, required=True, help="Model A name (e.g., 'ds' for DeepSeek)")
    parser.add_argument("--model_b", type=str, required=True, help="Model B name (e.g., 'dino' for DinoV2)")
    parser.add_argument("--layer", type=str, default="52", help="Layer number for naming")
    parser.add_argument("--seed", type=int, default=129, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {args.data_path}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Bootstrap replicas: {args.n_replicas}")
    print(f"Bootstrap size: {args.bootstrap_size}")
    
    # Load data
    inf_imb = load_inf_imb_data(args.data_path, args.n_layers)
    inf_matrix, mult_matrix, n_rows = extract_matrices(inf_imb, args.n_layers)
    
    print(f"Number of rows (origins): {n_rows}")
    print(f"Sample size: {args.sample_size}")
    
    # Compute mean information imbalance
    forward_means, backward_means = compute_mean_inf_imb(
        inf_matrix, mult_matrix, args.n_layers, n_rows, args.sample_size
    )
    
    print("\nMean Information Imbalance:")
    print(f"Forward ({args.model_a} → {args.model_b}): {forward_means}")
    print(f"Backward ({args.model_b} → {args.model_a}): {backward_means}")
    
    # Initialize RNG
    rng = np.random.default_rng(args.seed)
    
    # Perform bootstrap
    print("\nPerforming bootstrap analysis...")
    forward_bootstrap, backward_bootstrap = bootstrap(
        rng, inf_matrix, mult_matrix, 
        args.n_replicas, args.bootstrap_size, 
        args.n_layers, n_rows, args.sample_size
    )
    
    # Save results
    config = {
        'model_a': args.model_a,
        'model_b': args.model_b,
        'layer': args.layer,
        'n_layers': args.n_layers,
        'n_replicas': args.n_replicas,
        'bootstrap_size': args.bootstrap_size,
        'sample_size': args.sample_size,
        'seed': args.seed,
        'data_path': args.data_path
    }
    
    save_bootstrap_results(args.output_path, forward_bootstrap, backward_bootstrap, config)
    
    print(f"\nResults saved to: {args.output_path}")
    print(f"Forward: {args.model_a}_{args.model_b}_{args.layer}.npy")
    print(f"Backward: {args.model_b}_{args.model_a}_{args.layer}.npy")


if __name__ == "__main__":
    main()
