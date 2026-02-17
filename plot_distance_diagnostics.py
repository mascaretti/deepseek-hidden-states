#!/usr/bin/env python
"""
Distance diagnostic plots for hidden-state activations.

Produces a 3-row x N-column figure (one column per selected layer):
  Row 1: Pairwise raw L2 distance histogram
  Row 2: Norm distribution histogram
  Row 3: Cosine similarity histogram
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_ii_mean import load_mean_activations


def parse_args():
    parser = argparse.ArgumentParser(description="Distance diagnostic plots")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Directory with mean-pooled safetensors")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name for figure title")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layer indices to plot")
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--output", type=str, required=True,
                        help="Output PNG path")
    return parser.parse_args()


def upper_triangle(matrix):
    """Extract upper-triangle values (no diagonal) from a square matrix."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def main():
    args = parse_args()
    n_cols = len(args.layers)

    activations = load_mean_activations(args.input_path, subset_size=args.subset_size)

    fig, axes = plt.subplots(5, n_cols, figsize=(5 * n_cols, 16), squeeze=False,
                             sharey="row")
    # Share x-axis within each row for direct cross-layer comparison
    for row in range(5):
        for col in range(1, n_cols):
            axes[row, col].sharex(axes[row, 0])
    fig.suptitle(args.model_name, fontsize=16, fontweight="bold")

    for col, layer_idx in enumerate(args.layers):
        key = f"layer_{layer_idx}"
        X = activations[key].astype(np.float64)

        # --- Row 0: Pairwise L2 distances ---
        diff = X[:, None, :] - X[None, :, :]
        l2_matrix = np.linalg.norm(diff, axis=-1)
        l2_vals = upper_triangle(l2_matrix)
        l2_mean = l2_vals.mean()
        l2_cv = l2_vals.std() / l2_mean if l2_mean > 0 else 0.0

        ax = axes[0, col]
        ax.hist(l2_vals, bins=80, edgecolor="none", alpha=0.8)
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("L2 distance")
        ax.annotate(f"mean={l2_mean:.2f}\nCV={l2_cv:.4f}",
                     xy=(0.95, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # --- Row 1: Norm distribution ---
        norms = np.linalg.norm(X, axis=-1)
        norm_mean = norms.mean()
        norm_cv = norms.std() / norm_mean if norm_mean > 0 else 0.0

        ax = axes[1, col]
        ax.hist(norms, bins=80, edgecolor="none", alpha=0.8)
        ax.set_xlabel("||x||")
        ax.annotate(f"mean={norm_mean:.2f}\nCV={norm_cv:.4f}",
                     xy=(0.95, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # --- Row 2: Cosine similarity ---
        X_normed = X / np.linalg.norm(X, axis=-1, keepdims=True).clip(min=1e-8)
        cos_matrix = X_normed @ X_normed.T
        cos_vals = upper_triangle(cos_matrix)
        cos_mean = cos_vals.mean()
        cos_std = cos_vals.std()

        ax = axes[2, col]
        ax.hist(cos_vals, bins=80, edgecolor="none", alpha=0.8)
        ax.set_xlabel("Cosine similarity")
        ax.annotate(f"mean={cos_mean:.4f}\nstd={cos_std:.4f}",
                     xy=(0.95, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # --- Mean-centered versions ---
        Xc = X - X.mean(axis=0)

        # --- Row 3: Centered norm distribution ---
        c_norms = np.linalg.norm(Xc, axis=-1)
        c_norm_mean = c_norms.mean()
        c_norm_cv = c_norms.std() / c_norm_mean if c_norm_mean > 0 else 0.0

        ax = axes[3, col]
        ax.hist(c_norms, bins=80, edgecolor="none", alpha=0.8, color="C1")
        ax.set_xlabel("||x - mean||")
        ax.annotate(f"mean={c_norm_mean:.2f}\nCV={c_norm_cv:.4f}",
                     xy=(0.95, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # --- Row 4: Centered pairwise L2 ---
        c_diff = Xc[:, None, :] - Xc[None, :, :]
        c_l2_matrix = np.linalg.norm(c_diff, axis=-1)
        c_l2_vals = upper_triangle(c_l2_matrix)
        c_l2_mean = c_l2_vals.mean()
        c_l2_cv = c_l2_vals.std() / c_l2_mean if c_l2_mean > 0 else 0.0

        ax = axes[4, col]
        ax.hist(c_l2_vals, bins=80, edgecolor="none", alpha=0.8, color="C1")
        ax.set_xlabel("L2 distance (centered)")
        ax.annotate(f"mean={c_l2_mean:.2f}\nCV={c_l2_cv:.4f}",
                     xy=(0.95, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Row labels on the left
    axes[0, 0].set_ylabel("Pairwise L2")
    axes[1, 0].set_ylabel("Norms")
    axes[2, 0].set_ylabel("Cosine sim")
    axes[3, 0].set_ylabel("Centered norms")
    axes[4, 0].set_ylabel("Centered L2")

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
