#!/usr/bin/env python
"""
Analyze pairwise distance distributions under various transforms.

For each layer, compute pairwise distances and report mean + CV for:
  - Raw L2
  - Mean-centered L2
  - Mean-centered + L2-normalized L2
  - Cosine similarity (raw)
  - Cosine similarity (mean-centered)
  - PCA (whitened) L2
  - Lp norms: p = 0.5, 1, 2, 4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_ii_mean import load_mean_activations


def upper_triangle(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def pairwise_lp(X, p):
    """Compute pairwise Lp distances, extract upper triangle."""
    diff = X[:, None, :] - X[None, :, :]
    if p == np.inf:
        dists = np.abs(diff).max(axis=-1)
    else:
        dists = (np.abs(diff) ** p).sum(axis=-1) ** (1.0 / p)
    return upper_triangle(dists)


def cosine_sims(X):
    norms = np.linalg.norm(X, axis=-1, keepdims=True).clip(min=1e-8)
    X_n = X / norms
    return upper_triangle(X_n @ X_n.T)


def cv(vals):
    m = vals.mean()
    return vals.std() / m if m > 0 else 0.0


def pca_whiten(X, n_components=None):
    """Mean-center, PCA, whiten (unit variance per component)."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if n_components is not None:
        U, S, Vt = U[:, :n_components], S[:n_components], Vt[:n_components]
    # Project and whiten: each component gets unit variance
    return U * (np.sqrt(len(X) - 1))  # U columns are already normalized, scale to unit var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--subset_size", type=int, default=410)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    activations = load_mean_activations(args.input_path, subset_size=args.subset_size)

    methods = [
        "Raw L2",
        "Centered L2",
        "Cent+Norm L2",
        "Cosine (raw)",
        "Cosine (cent)",
        "PCA-whiten L2",
        "Lp p=0.5",
        "Lp p=1",
        "Lp p=4",
        "Lp p=inf",
    ]

    print(f"\n{'':=^80}")
    print(f"  {args.model_name} — Pairwise distance CV by method and layer")
    print(f"{'':=^80}")
    print(f"{'Method':<16}", end="")
    for layer in args.layers:
        print(f"  {'L'+str(layer):>10}", end="")
    print()
    print("-" * (16 + 12 * len(args.layers)))

    all_cvs = {m: [] for m in methods}

    for layer_idx in args.layers:
        key = f"layer_{layer_idx}"
        X = activations[key].astype(np.float64)
        Xc = X - X.mean(axis=0)
        Xc_norm = Xc / np.linalg.norm(Xc, axis=-1, keepdims=True).clip(min=1e-8)
        X_pca = pca_whiten(X)

        results = {}
        results["Raw L2"] = cv(pairwise_lp(X, 2))
        results["Centered L2"] = cv(pairwise_lp(Xc, 2))
        results["Cent+Norm L2"] = cv(pairwise_lp(Xc_norm, 2))
        results["Cosine (raw)"] = cv(cosine_sims(X))
        results["Cosine (cent)"] = cv(cosine_sims(Xc))
        results["PCA-whiten L2"] = cv(pairwise_lp(X_pca, 2))
        results["Lp p=0.5"] = cv(pairwise_lp(X, 0.5))
        results["Lp p=1"] = cv(pairwise_lp(X, 1))
        results["Lp p=4"] = cv(pairwise_lp(X, 4))
        results["Lp p=inf"] = cv(pairwise_lp(X, np.inf))

        for m in methods:
            all_cvs[m].append(results[m])

    # Print table
    for m in methods:
        print(f"{m:<16}", end="")
        for val in all_cvs[m]:
            print(f"  {val:>10.4f}", end="")
        print()

    # Also print mean + std for context
    print(f"\n{'':=^80}")
    print(f"  {args.model_name} — Pairwise distance MEAN by method and layer")
    print(f"{'':=^80}")
    print(f"{'Method':<16}", end="")
    for layer in args.layers:
        print(f"  {'L'+str(layer):>10}", end="")
    print()
    print("-" * (16 + 12 * len(args.layers)))

    for layer_idx in args.layers:
        key = f"layer_{layer_idx}"
        X = activations[key].astype(np.float64)
        Xc = X - X.mean(axis=0)
        Xc_norm = Xc / np.linalg.norm(Xc, axis=-1, keepdims=True).clip(min=1e-8)
        X_pca = pca_whiten(X)

        # Store for printing below
        if layer_idx == args.layers[0]:
            all_means = {m: [] for m in methods}

        all_means["Raw L2"].append(pairwise_lp(X, 2).mean())
        all_means["Centered L2"].append(pairwise_lp(Xc, 2).mean())
        all_means["Cent+Norm L2"].append(pairwise_lp(Xc_norm, 2).mean())
        all_means["Cosine (raw)"].append(cosine_sims(X).mean())
        all_means["Cosine (cent)"].append(cosine_sims(Xc).mean())
        all_means["PCA-whiten L2"].append(pairwise_lp(X_pca, 2).mean())
        all_means["Lp p=0.5"].append(pairwise_lp(X, 0.5).mean())
        all_means["Lp p=1"].append(pairwise_lp(X, 1).mean())
        all_means["Lp p=4"].append(pairwise_lp(X, 4).mean())
        all_means["Lp p=inf"].append(pairwise_lp(X, np.inf).mean())

    for m in methods:
        print(f"{m:<16}", end="")
        for val in all_means[m]:
            print(f"  {val:>10.4f}", end="")
        print()

    # Plot
    if args.output:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(args.layers))
        width = 0.08
        for i, m in enumerate(methods):
            offset = (i - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, all_cvs[m], width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {l}" for l in args.layers])
        ax.set_ylabel("CV of pairwise distances")
        ax.set_title(f"{args.model_name} — Distance CV by method")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
