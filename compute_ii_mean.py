#!/usr/bin/env python
"""
Compute information imbalance between two models using mean-pooled activations.

Input:  mean_chunk_N.safetensors from preprocess_mean.py
        Each file: {layer_N: (n_samples, embed)} float32

Pipeline: load → concatenate chunks → binarize (Hamming) / raw or normalise (L2) → II
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

from information_imbalance_module import (
    L2_distance,
    build_information_imbalance,
    build_information_matrix,
    hamming_distance,
)
from utils import binarize


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
# L2 normalisation
# ---------------------------------------------------------------------------

def l2_normalize(a):
    norms = jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a / jnp.maximum(norms, 1e-8)


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
    parser.add_argument("--l2_normalize", action="store_true",
                        help="L2-normalise vectors before computing L2 distance (cosine mode)")
    parser.add_argument("--mean_center", action="store_true",
                        help="Subtract per-layer mean before computing L2 distance")
    args = parser.parse_args()

    Path(args.output_folder, "hamming").mkdir(parents=True, exist_ok=True)
    l2_parts = ["l2"]
    if args.mean_center:
        l2_parts.append("centered")
    if args.l2_normalize:
        l2_parts.append("normalized")
    l2_label = "_".join(l2_parts)
    Path(args.output_folder, l2_label).mkdir(parents=True, exist_ok=True)

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

    # Setup II computation (two distance functions)
    key = jax.random.key(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    hamming_fn = build_information_matrix(
        key=subkey1, sample_size=sample_size, distance_fn=hamming_distance
    )
    l2_fn = build_information_imbalance(
        key=subkey2, sample_size=sample_size, distance_fn=L2_distance
    )

    # Prepare model A (fixed layer)
    if args.l2_normalize:
        print("L2 Normalisation")
    if args.mean_center:
        print("Mean centering")
    raw_a = jnp.array(act_A[layer_A_key])
    a_hamming = binarize(raw_a)
    a_l2 = raw_a
    if args.mean_center:
        a_l2 = a_l2 - a_l2.mean(axis=0)
    if args.l2_normalize:
        a_l2 = l2_normalize(a_l2)

    # Sweep over model B layers
    cols = [f"layer_{i}" for i in range(1, n_layers_B)]
    for layer_B_key in tqdm(cols, desc="Computing II"):
        raw_b = jnp.array(act_B[layer_B_key])

        # Hamming
        b_hamming = binarize(raw_b)
        result = hamming_fn(a_hamming, b_hamming)
        result_np = {k: np.asarray(v) for k, v in result.items()}
        save_file(result_np, str(Path(args.output_folder, "hamming", f"{layer_B_key}.safetensors")))

        # L2 (scalar II, no cardinality normalisation — no ties with continuous distances)
        b_l2 = raw_b
        if args.mean_center:
            b_l2 = b_l2 - b_l2.mean(axis=0)
        if args.l2_normalize:
            b_l2 = l2_normalize(b_l2)
        inf_imb, reciprocal_inf_imb = l2_fn(a_l2, b_l2)
        l2_result = {
            "inf_imb": np.asarray(inf_imb),
            "reciprocal_inf_imb": np.asarray(reciprocal_inf_imb),
        }
        save_file(l2_result, str(Path(args.output_folder, l2_label, f"{layer_B_key}.safetensors")))

    print(f"\nResults saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
