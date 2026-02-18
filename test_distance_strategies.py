#!/usr/bin/env python
"""
Test distance strategies for II concentration of measure.

Compare 4 distance strategies on DeepSeek→DinoV2 and DeepSeek→ImageGPT
to identify which metrics survive concentration of measure.

Strategies:
  1. Hamming           – binarize → hamming_distance
  2. Centered+Norm L2  – mean_center → l2_normalize → L2_distance
  3. Centered L1       – mean_center → L1_distance
  4. PCA + L2          – PCA(n_components) → L2_distance
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from compute_ii_mean import load_mean_activations, l2_normalize
from information_imbalance_module import (
    L1_distance,
    L2_distance,
    build_information_imbalance,
    hamming_distance,
)
from utils import binarize


# ── Transforms ──────────────────────────────────────────────────────


def pca_project(X, n_components):
    """Mean-center and project onto top n PCA components."""
    Xc = X - X.mean(axis=0)
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, U.shape[1])
    return U[:, :k] * S[:k]


def transform_hamming(X, **_kw):
    return np.asarray(binarize(jnp.array(X)))


def transform_center_norm(X, **_kw):
    Xc = X - X.mean(axis=0)
    return np.asarray(l2_normalize(jnp.array(Xc)))


def transform_center(X, **_kw):
    return (X - X.mean(axis=0)).astype(np.float32)


def transform_pca(X, pca_components=100, **_kw):
    return pca_project(X, pca_components).astype(np.float32)


# ── Strategy definitions ────────────────────────────────────────────

STRATEGIES = [
    {"name": "Hamming",          "distance_fn": hamming_distance, "transform": transform_hamming},
    {"name": "Centered+Norm L2", "distance_fn": L2_distance,      "transform": transform_center_norm},
    {"name": "Centered L1",      "distance_fn": L1_distance,      "transform": transform_center},
    {"name": "PCA + L2",         "distance_fn": L2_distance,      "transform": transform_pca},
]


# ── Core computation ────────────────────────────────────────────────


def run_comparison(name, raw_a, act_B, shared_key, sample_size,
                   pca_components):
    """Run all strategies for one comparison (e.g. DeepSeek → DinoV2).

    Returns:
        results: dict[strategy_name] -> {"fwd": array, "bwd": array}
        layer_indices: list of layer indices swept
    """
    n_layers_B = len(act_B)
    layer_indices = list(range(1, n_layers_B))

    results = {}

    for strategy in STRATEGIES:
        sname = strategy["name"]
        print(f"  Strategy: {sname}")

        # Build II function with shared key → identical subsampling
        ii_fn = build_information_imbalance(
            key=shared_key, sample_size=sample_size,
            distance_fn=strategy["distance_fn"],
        )

        # Transform model A (once per strategy)
        a_transformed = jnp.array(
            strategy["transform"](raw_a, pca_components=pca_components)
        )

        fwd_vals = []
        bwd_vals = []

        for layer_idx in tqdm(layer_indices, desc=f"    {sname}", leave=False):
            layer_key = f"layer_{layer_idx}"
            raw_b = act_B[layer_key].astype(np.float32)
            b_transformed = jnp.array(
                strategy["transform"](raw_b, pca_components=pca_components)
            )

            fwd, bwd = ii_fn(a_transformed, b_transformed)
            fwd_vals.append(float(fwd))
            bwd_vals.append(float(bwd))

        results[sname] = {
            "fwd": np.array(fwd_vals),
            "bwd": np.array(bwd_vals),
        }

    return results, layer_indices


# ── Plotting ────────────────────────────────────────────────────────


def plot_results(results_dino, results_igpt, layers_dino, layers_igpt,
                 output_path):
    """2-panel plot: forward II vs model B relative depth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    panels = [
        (ax1, results_dino, layers_dino, r"DeepSeek $\to$ DinoV2"),
        (ax2, results_igpt, layers_igpt, r"DeepSeek $\to$ ImageGPT"),
    ]

    for ax, results, layers, title in panels:
        n_layers = max(layers) if layers else 1
        rel_depth = np.array(layers) / n_layers

        for sname, vals in results.items():
            ax.plot(rel_depth, vals["fwd"], marker="o", markersize=3,
                    label=sname)

        ax.set_xlabel("Model B relative depth")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Forward II (A → B)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Test distance strategies for II concentration of measure"
    )
    parser.add_argument("--input_path_A", type=str, required=True)
    parser.add_argument("--input_path_B_dino", type=str, required=True)
    parser.add_argument("--input_path_B_igpt", type=str, required=True)
    parser.add_argument("--layer_A", type=int, default=52)
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--pca_components", type=int, default=100)
    parser.add_argument("--output_prefix", type=str, required=True)
    args = parser.parse_args()

    # Load activations
    print("Loading activations...")
    act_A = load_mean_activations(args.input_path_A, subset_size=args.subset_size)
    act_B_dino = load_mean_activations(args.input_path_B_dino,
                                       subset_size=args.subset_size)
    act_B_igpt = load_mean_activations(args.input_path_B_igpt,
                                       subset_size=args.subset_size)

    # Validate
    layer_A_key = f"layer_{args.layer_A}"
    if layer_A_key not in act_A:
        available = sorted(act_A.keys(), key=lambda k: int(k.split("_")[1]))
        raise ValueError(f"{layer_A_key} not in model A. Available: {available}")

    sample_size = act_A[layer_A_key].shape[0]
    for tag, act_B in [("DinoV2", act_B_dino), ("ImageGPT", act_B_igpt)]:
        sb = act_B["layer_0"].shape[0]
        if sample_size != sb:
            raise ValueError(
                f"Sample size mismatch: A={sample_size}, {tag}={sb}"
            )

    print(f"\nModel A: layer {args.layer_A} ({act_A[layer_A_key].shape[1]}d)")
    print(f"DinoV2:  {len(act_B_dino)} layers ({act_B_dino['layer_0'].shape[1]}d)")
    print(f"ImageGPT: {len(act_B_igpt)} layers ({act_B_igpt['layer_0'].shape[1]}d)")
    print(f"Sample size: {sample_size}")
    print(f"PCA components: {args.pca_components}")

    # Shared key for identical subsampling across all strategies
    shared_key = jax.random.key(42)

    raw_a = act_A[layer_A_key].astype(np.float32)

    # Run comparisons
    print("\n=== DeepSeek → DinoV2 ===")
    results_dino, layers_dino = run_comparison(
        "DinoV2", raw_a, act_B_dino, shared_key, sample_size,
        args.pca_components,
    )

    print("\n=== DeepSeek → ImageGPT ===")
    results_igpt, layers_igpt = run_comparison(
        "ImageGPT", raw_a, act_B_igpt, shared_key, sample_size,
        args.pca_components,
    )

    # Save results as .npz
    npz_data = {}
    for comp_tag, results in [("dino", results_dino), ("igpt", results_igpt)]:
        for sname, vals in results.items():
            safe = sname.replace(" ", "_").replace("+", "_")
            npz_data[f"{safe}_{comp_tag}_fwd"] = vals["fwd"]
            npz_data[f"{safe}_{comp_tag}_bwd"] = vals["bwd"]
    npz_data["layers_dino"] = np.array(layers_dino)
    npz_data["layers_igpt"] = np.array(layers_igpt)

    npz_path = f"{args.output_prefix}.npz"
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **npz_data)
    print(f"\nSaved data: {npz_path}")

    # Plot
    plot_path = f"{args.output_prefix}.png"
    plot_results(results_dino, results_igpt, layers_dino, layers_igpt,
                 plot_path)


if __name__ == "__main__":
    main()
