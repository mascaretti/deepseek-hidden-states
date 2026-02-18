#!/usr/bin/env python
"""
Compare 5 distance strategies for Information Imbalance to diagnose
concentration-of-measure effects in high-dimensional representations.

Strategies:
  1. Hamming          — binarize -> hamming_distance
  2. Centered+Norm L2 — mean_center -> l2_normalize -> L2_distance
  3. Centered L1      — mean_center -> L1_distance
  4. PCA-100 + L2     — PCA(100 PCs, per-model) -> L2_distance
  5. Log-map + L2     — l2_normalize -> Log_mu (tangent space) -> L2_distance

All 5 use build_information_imbalance with the same key -> identical subsampling.

Includes spectral entropy diagnostic: H_spec before/after Log-map.

Usage:
  python test_distance_strategies.py \
      --input_path_A  data/<deepseek_windsorised>/ \
      --input_path_B_dino data/facebook-dinov2-large/ \
      --input_path_B_igpt data/openai-imagegpt-large/ \
      --layer_A 52 --subset_size 500 \
      --output_prefix results/distance_test
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).parent / ".cache" / "matplotlib"),
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from tqdm import tqdm

try:
    import scienceplots
    plt.style.use(["nature", "science"])
except Exception:
    pass

from compute_ii_mean import load_mean_activations
from information_imbalance_module import (
    L1_distance,
    L2_distance,
    build_information_imbalance,
    hamming_distance,
)
from utils import binarize


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def mean_center(a):
    return a - a.mean(axis=0)


def l2_normalize(a):
    norms = jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a / jnp.maximum(norms, 1e-8)


def pca_project(a, n_components=100):
    """Mean-center then project onto top-n_components PCA directions."""
    a_np = np.array(a, dtype=np.float64)
    a_np -= a_np.mean(axis=0)
    _, _, Vt = np.linalg.svd(a_np, full_matrices=False)
    projected = a_np @ Vt[:n_components].T
    return jnp.array(projected, dtype=jnp.float32)


def log_map(a):
    """
    Map L2-normalised points on S^(d-1) to tangent space at Frechet mean mu.

        v = (theta / sin(theta)) * (x - cos(theta) * mu)

    where theta = arccos(mu^T x).  Handles theta->0 (scale->1).
    """
    x_hat = l2_normalize(a)
    mu = jnp.mean(x_hat, axis=0)
    mu = mu / jnp.maximum(jnp.linalg.norm(mu), 1e-8)

    cos_theta = jnp.clip(x_hat @ mu, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_theta)
    scale = jnp.where(theta < 1e-7, 1.0, theta / jnp.sin(theta))
    return scale[:, None] * (x_hat - cos_theta[:, None] * mu[None, :])


# ---------------------------------------------------------------------------
# Spectral entropy diagnostic
# ---------------------------------------------------------------------------

def spectral_entropy(a):
    """Shannon entropy of normalised PCA eigenvalues: H = -sum(l_i * ln(l_i))."""
    a_np = np.array(a, dtype=np.float64)
    a_np -= a_np.mean(axis=0)
    _, S, _ = np.linalg.svd(a_np, full_matrices=False)
    lambdas = S ** 2
    lambdas_hat = lambdas / lambdas.sum()
    lambdas_hat = lambdas_hat[lambdas_hat > 0]
    return float(-np.sum(lambdas_hat * np.log(lambdas_hat)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_A", required=True)
    parser.add_argument("--input_path_B_dino", required=True)
    parser.add_argument("--input_path_B_igpt", required=True)
    parser.add_argument("--layer_A", type=int, default=52)
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--pca_components", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_prefix", type=str, default="results/distance_test")
    args = parser.parse_args()

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("Loading activations...")
    act_A    = load_mean_activations(args.input_path_A,      subset_size=args.subset_size)
    act_dino = load_mean_activations(args.input_path_B_dino,  subset_size=args.subset_size)
    act_igpt = load_mean_activations(args.input_path_B_igpt,  subset_size=args.subset_size)

    layer_A_key = f"layer_{args.layer_A}"
    if layer_A_key not in act_A:
        avail = sorted(act_A.keys(), key=lambda k: int(k.split("_")[1]))
        raise ValueError(f"{layer_A_key} not in model A. Available: {avail}")

    sample_size = act_A[layer_A_key].shape[0]
    for label, act in [("DinoV2", act_dino), ("ImageGPT", act_igpt)]:
        n = act["layer_0"].shape[0]
        if n != sample_size:
            raise ValueError(f"Sample size mismatch {label}: A={sample_size}, B={n}")

    raw_a = jnp.array(act_A[layer_A_key])
    print(f"\nModel A : {layer_A_key}, embed={raw_a.shape[1]}, N={sample_size}")
    print(f"DinoV2  : {len(act_dino)} layers, embed={act_dino['layer_0'].shape[1]}")
    print(f"ImageGPT: {len(act_igpt)} layers, embed={act_igpt['layer_0'].shape[1]}")

    # ------------------------------------------------------------------
    # Strategy definitions
    # ------------------------------------------------------------------
    pca_name = f"PCA-{args.pca_components}+L2"
    STRATEGIES = [
        ("Hamming",      hamming_distance, binarize),
        ("Cent+Norm L2", L2_distance,      lambda a: l2_normalize(mean_center(a))),
        ("Cent L1",      L1_distance,      mean_center),
        (pca_name,       L2_distance,      lambda a: pca_project(a, args.pca_components)),
        ("Log-map+L2",   L2_distance,      log_map),
    ]

    # ------------------------------------------------------------------
    # Build II functions — same key -> identical row/col subsampling
    # ------------------------------------------------------------------
    ii_key = random.PRNGKey(args.seed)
    ii_fns = {
        name: build_information_imbalance(
            key=ii_key, sample_size=sample_size, distance_fn=dist_fn
        )
        for name, dist_fn, _ in STRATEGIES
    }

    # Pre-compute model A transforms (once per strategy)
    print("\nTransforming model A layer...")
    a_transforms = {name: transform(raw_a) for name, _, transform in STRATEGIES}

    # ------------------------------------------------------------------
    # Spectral entropy diagnostic — one column per strategy transform
    # ------------------------------------------------------------------
    strategy_names = [name for name, _, _ in STRATEGIES]
    header = f"{'Model/layer':<22}" + "".join(f"  {n:>14}" for n in strategy_names)
    print("\n=== Spectral entropy H_spec per strategy transform ===")
    print(header)
    print("-" * (22 + 16 * len(STRATEGIES)))
    for label, act_B in [("DeepSeek", {layer_A_key: np.array(raw_a)}),
                          ("DinoV2",   act_dino),
                          ("ImageGPT", act_igpt)]:
        keys = sorted(act_B.keys(), key=lambda k: int(k.split("_")[1]))
        step = max(1, len(keys) // 6)
        for k in keys[::step]:
            b = jnp.array(act_B[k])
            entropies = [spectral_entropy(transform(b)) for _, _, transform in STRATEGIES]
            row = f"  {label+'/'+k:<20}" + "".join(f"  {h:>14.3f}" for h in entropies)
            print(row)

    # ------------------------------------------------------------------
    # II sweep
    # ------------------------------------------------------------------
    def sweep(act_B, label):
        n_layers = len(act_B)
        results = {name: {"fwd": [], "bwd": []} for name, _, _ in STRATEGIES}
        for i in tqdm(range(n_layers), desc=f"II vs {label}"):
            raw_b = jnp.array(act_B[f"layer_{i}"])
            for name, _, transform in STRATEGIES:
                b_t = transform(raw_b)
                fwd, bwd = ii_fns[name](a_transforms[name], b_t)
                results[name]["fwd"].append(float(fwd))
                results[name]["bwd"].append(float(bwd))
        return {n: {k: np.array(v) for k, v in d.items()} for n, d in results.items()}

    print("\nComputing II...")
    results_dino = sweep(act_dino, "DinoV2")
    results_igpt = sweep(act_igpt, "ImageGPT")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dict = {
        "layer_A": np.array(args.layer_A),
        "subset_size": np.array(sample_size),
        "pca_components": np.array(args.pca_components),
    }
    for name, _, _ in STRATEGIES:
        slug = name.replace(" ", "_").replace("+", "_").lower()
        save_dict[f"dino_{slug}_fwd"] = results_dino[name]["fwd"]
        save_dict[f"dino_{slug}_bwd"] = results_dino[name]["bwd"]
        save_dict[f"igpt_{slug}_fwd"] = results_igpt[name]["fwd"]
        save_dict[f"igpt_{slug}_bwd"] = results_igpt[name]["bwd"]
    np.savez(f"{args.output_prefix}.npz", **save_dict)
    print(f"Saved {args.output_prefix}.npz")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    COLORS = {
        "Hamming":      "#364F6B",
        "Cent+Norm L2": "#C44536",
        "Cent L1":      "#609966",
        pca_name:       "#FC5185",
        "Log-map+L2":   "#E8A87C",
    }
    ms, mew = 2, 0.3

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, (results, title) in zip(
        axes,
        [(results_dino, f"DeepSeek L{args.layer_A} → DinoV2"),
         (results_igpt, f"DeepSeek L{args.layer_A} → ImageGPT")],
    ):
        n = len(next(iter(results.values()))["fwd"])
        rel = np.arange(n) / n
        for name, _, _ in STRATEGIES:
            ax.plot(rel, results[name]["fwd"], "-o",
                    color=COLORS.get(name, "gray"),
                    markersize=ms, markeredgecolor="black", markeredgewidth=mew,
                    label=name)
        ax.set_xlabel("Model B relative depth", fontsize=10)
        ax.set_ylabel("II (forward)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.output_prefix}.{ext}", dpi=300)
    print(f"Saved {args.output_prefix}.pdf/.png")


if __name__ == "__main__":
    main()
