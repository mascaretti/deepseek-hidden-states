#!/usr/bin/env python
"""
Compute information imbalance between two models using mean-pooled activations.

Input:  mean_chunk_N.safetensors from preprocess_mean.py
        Each file: {layer_N: (n_samples, embed)} float32

Pipeline: load → concatenate chunks → binarize → Hamming II
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file
from tqdm import tqdm

from information_imbalance_module import build_information_matrix, hamming_distance


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _extract_index(f):
    match = re.search(r"chunk_(\d+)", f.name)
    return int(match.group(1)) if match else float("inf")


def load_mean_activations(input_path, subset_size=5000, prefix="mean_"):
    """Load mean-pooled activations and concatenate across chunks.

    Returns:
        dict[str, np.ndarray]: {layer_N: (min(total, subset_size), embed)}
    """
    folder = Path(input_path)
    files = sorted(
        [f for f in folder.iterdir() if f.name.startswith(prefix) and f.suffix == ".safetensors"],
        key=_extract_index,
    )
    if not files:
        raise FileNotFoundError(f"No {prefix}*.safetensors files in {input_path}")

    layer_data = defaultdict(list)
    total = 0
    for f in tqdm(files, desc=f"Loading {folder.name}", leave=False):
        if total >= subset_size:
            break
        chunk = load_file(str(f))
        for layer_key, arr in chunk.items():
            layer_data[layer_key].append(arr)
        total += chunk["layer_0"].shape[0]

    result = {}
    for layer_key, arrays in layer_data.items():
        result[layer_key] = np.concatenate(arrays, axis=0)[:subset_size]

    sample = result["layer_0"]
    print(f"  {folder.name}: {sample.shape[0]} samples, "
          f"{len(result)} layers, embed={sample.shape[1]}")
    return result


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------

def binarize(a):
    a = jnp.sign(a).astype(jnp.int8)
    a = a.at[a == 0].set(-1)
    return a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute information imbalance from mean-pooled activations"
    )
    parser.add_argument("--input_path_A", type=str, required=True,
                        help="Directory with mean-pooled safetensors for model A")
    parser.add_argument("--input_path_B", type=str, required=True,
                        help="Directory with mean-pooled safetensors for model B")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--layer_A", type=int, required=True,
                        help="Fixed layer index for model A")
    parser.add_argument("--subset_size", type=int, default=5000)
    args = parser.parse_args()

    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading activations...")
    act_A = load_mean_activations(args.input_path_A, subset_size=args.subset_size)
    act_B = load_mean_activations(args.input_path_B, subset_size=args.subset_size)

    # Validate
    layer_A_key = f"layer_{args.layer_A}"
    if layer_A_key not in act_A:
        available = sorted(act_A.keys(), key=lambda k: int(k.split("_")[1]))
        raise ValueError(f"{layer_A_key} not in model A. Available: {available}")

    sample_size_A = act_A[layer_A_key].shape[0]
    sample_size_B = act_B["layer_0"].shape[0]
    if sample_size_A != sample_size_B:
        raise ValueError(f"Sample size mismatch: A={sample_size_A}, B={sample_size_B}")
    sample_size = sample_size_A

    n_layers_B = len(act_B)
    print(f"\nModel A layer: {layer_A_key} ({act_A[layer_A_key].shape[1]}d)")
    print(f"Model B layers: {n_layers_B} ({act_B['layer_0'].shape[1]}d)")
    print(f"Sample size: {sample_size}")

    # Setup II computation
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_matrix_fn = build_information_matrix(
        key=subkey, sample_size=sample_size, distance_fn=hamming_distance
    )

    # Binarize model A (fixed layer)
    a = binarize(jnp.array(act_A[layer_A_key]))

    # Sweep over model B layers
    cols = [f"layer_{i}" for i in range(1, n_layers_B)]
    for layer_B_key in tqdm(cols, desc="Computing II"):
        b = binarize(jnp.array(act_B[layer_B_key]))
        result = information_matrix_fn(a, b)
        # Convert JAX arrays to numpy for safetensors
        result_np = {k: np.asarray(v) for k, v in result.items()}
        save_file(result_np, str(Path(args.output_folder, f"{layer_B_key}.safetensors")))

    print(f"\nResults saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
