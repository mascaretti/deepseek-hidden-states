#!/usr/bin/env python
"""
Figure 1: Information Imbalance vs CKA on synthetic Gaussian data.

Left:  II and CKA for Y = B_r X + eps, with B_r having monotonically increasing rank.
Right: II and CKA vs fraction of shared features, for p=1e3 and p=1e5.

Both panels show mean +/- standard error over jackknife repetitions.

Usage:
    python notebooks/figure1_ii_cka.py [--dim 10] [--n_samples 500] [--n_reps 10]
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure matplotlib cache stays inside the workspace
mpl_cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from cka import build_cka_distance
from information_imbalance_module import L2_distance, build_information_imbalance


# ---------------------------------------------------------------------------
# Panel A — Asymmetry of II (monotonically increasing rank)
# ---------------------------------------------------------------------------

def make_rank_matrix(key, rank, dim=10):
    key_u, key_v = random.split(key)
    U = random.normal(key_u, (dim, rank))
    V = random.normal(key_v, (rank, dim))
    return U @ V


def simulate_XY(key, L=10, n_samples=500, tau=1.0, dim=10, sigma=0.1):
    """Simulate Y_r = B_r X_r + eps, with rank(B_r) = r."""
    key_B, key_X, key_eps = random.split(key, 3)

    # Monotonically increasing rank: 1, 2, ..., dim (capped)
    ranks = jnp.arange(1, L + 1)
    ranks = jnp.minimum(ranks, dim)

    keys_B = random.split(key_B, L)
    Bs = jnp.stack([make_rank_matrix(keys_B[l], int(ranks[l]), dim=dim) for l in range(L)])

    X = tau * random.normal(key_X, (L, n_samples, dim))
    eps = sigma * random.normal(key_eps, (L, n_samples, dim))
    Y = jnp.einsum("lij,lnj->lni", Bs, X) + eps

    return X, Y, ranks


def compute_panel_a_single(key, L=10, n_samples=500, dim=10, sigma=0.1):
    """One rep of panel A."""
    key, sim_key = random.split(key)
    X, Y, ranks = simulate_XY(sim_key, L=L, n_samples=n_samples, dim=dim, sigma=sigma)

    imb_XY = np.zeros(L)
    imb_YX = np.zeros(L)
    cka = np.zeros(L)

    for l in range(L):
        key, subkey = random.split(key)
        ii_fn = build_information_imbalance(
            key=subkey, sample_size=n_samples, distance_fn=L2_distance,
        )
        fwd, bwd = ii_fn(X[l], Y[l])
        imb_XY[l] = float(fwd)
        imb_YX[l] = float(bwd)
        cka[l] = float(build_cka_distance(X[l], Y[l], distance_fn=L2_distance))

    return imb_XY, imb_YX, cka, np.array(ranks)


# ---------------------------------------------------------------------------
# Panel B — II and CKA vs shared features (Gaussian)
# ---------------------------------------------------------------------------

def compute_panel_b_single(key, p=1000, n_samples=500, n_data=100):
    """One rep of panel B for a given p."""
    key, data_key = random.split(key)
    Y = random.normal(data_key, (n_samples, p), dtype=jnp.float32)

    n_data_eff = min(n_data, p)
    feature_indices = jnp.arange(1, p, max(p // n_data_eff, 1))
    fracs = np.array(feature_indices) / p

    cka_vals = np.zeros(len(feature_indices))
    ii_fwd_vals = np.zeros(len(feature_indices))
    ii_bwd_vals = np.zeros(len(feature_indices))

    for i, jj in enumerate(feature_indices):
        jj = int(jj)
        Y_sub = Y[:, :jj]
        cka_vals[i] = float(build_cka_distance(Y, Y_sub, distance_fn=L2_distance))
        key, subkey = random.split(key)
        ii_fn = build_information_imbalance(
            key=subkey, sample_size=n_samples, distance_fn=L2_distance,
        )
        fwd, bwd = ii_fn(Y, Y_sub)
        ii_fwd_vals[i] = float(fwd)
        ii_bwd_vals[i] = float(bwd)

    return fracs, cka_vals, ii_fwd_vals, ii_bwd_vals


# ---------------------------------------------------------------------------
# Jackknife helpers
# ---------------------------------------------------------------------------

def jackknife_stats(runs):
    """Given list of arrays (one per rep), return mean and standard error."""
    arr = np.stack(runs)
    mean = arr.mean(axis=0)
    n = arr.shape[0]
    se = arr.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, se


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Figure 1: II vs CKA on synthetic data")
    parser.add_argument("--dim", type=int, default=10, help="Dimension D for panel A")
    parser.add_argument("--L", type=int, default=10, help="Number of layers/indices in panel A")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise level sigma for panel A")
    parser.add_argument("--p_values", type=int, nargs="+", default=[1000, 100000],
                        help="Dimension(s) p for panel B")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--n_data", type=int, default=100, help="Feature-fraction steps in panel B")
    parser.add_argument("--n_reps", type=int, default=10, help="Jackknife repetitions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="notebooks/figure1_ii_cka",
                        help="Output filename (no extension)")
    args = parser.parse_args()

    key = random.PRNGKey(args.seed)

    # --- Panel A: multiple reps ---
    print(f"Computing panel A ({args.n_reps} reps, dim={args.dim}, L={args.L})...")
    runs_XY, runs_YX, runs_cka_a = [], [], []
    for rep in tqdm(range(args.n_reps), desc="Panel A"):
        key, subkey = random.split(key)
        imb_XY, imb_YX, cka_a, ranks = compute_panel_a_single(
            subkey, L=args.L, n_samples=args.n_samples, dim=args.dim, sigma=args.sigma,
        )
        runs_XY.append(imb_XY)
        runs_YX.append(imb_YX)
        runs_cka_a.append(cka_a)

    mean_XY, se_XY = jackknife_stats(runs_XY)
    mean_YX, se_YX = jackknife_stats(runs_YX)
    mean_cka_a, se_cka_a = jackknife_stats(runs_cka_a)

    # --- Panel B: multiple reps per p ---
    panel_b_results = {}
    for p in args.p_values:
        print(f"Computing panel B ({args.n_reps} reps, p={p})...")
        runs_cka, runs_fwd, runs_bwd = [], [], []
        fracs = None
        for rep in tqdm(range(args.n_reps), desc=f"Panel B (p={p})"):
            key, subkey = random.split(key)
            f, c, fwd, bwd = compute_panel_b_single(
                subkey, p=p, n_samples=args.n_samples, n_data=args.n_data,
            )
            fracs = f
            runs_cka.append(c)
            runs_fwd.append(fwd)
            runs_bwd.append(bwd)

        panel_b_results[p] = {
            "fracs": fracs,
            "cka": jackknife_stats(runs_cka),
            "ii_fwd": jackknife_stats(runs_fwd),
            "ii_bwd": jackknife_stats(runs_bwd),
        }

    # --- Plot ---
    ms = 3
    mew = 0.3
    n_panels_b = len(args.p_values)
    fig, axes = plt.subplots(1, 1 + n_panels_b, figsize=(3.5 * (1 + n_panels_b), 2.5))
    if not hasattr(axes, '__len__'):
        axes = [axes]

    col_xy = "#364F6B"
    col_yx = "#609966"
    col_cka = "#C44536"

    # Panel A
    ax = axes[0]
    rel_depth = np.arange(args.L) / args.L

    ax.plot(rel_depth, mean_XY, "-o", color=col_xy, markersize=ms,
            markeredgecolor="black", markeredgewidth=mew,
            label=r"$\Delta(X \rightarrow Y)$")
    ax.fill_between(rel_depth, mean_XY - se_XY, mean_XY + se_XY, color=col_xy, alpha=0.2)

    ax.plot(rel_depth, mean_YX, "-o", color=col_yx, markersize=ms,
            markeredgecolor="black", markeredgewidth=mew,
            label=r"$\Delta(Y \rightarrow X)$")
    ax.fill_between(rel_depth, mean_YX - se_YX, mean_YX + se_YX, color=col_yx, alpha=0.2)

    ax.plot(rel_depth, mean_cka_a, "-o", color=col_cka, markersize=ms,
            markeredgecolor="black", markeredgewidth=mew,
            label="CKA")
    ax.fill_between(rel_depth, mean_cka_a - se_cka_a, mean_cka_a + se_cka_a, color=col_cka, alpha=0.2)

    ax.set_xlabel("Index $r$", fontsize=11)
    ax.set_ylabel("II / CKA", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title(f"$D = {args.dim}$", fontsize=11)

    # Panel B (one per p)
    for i, p in enumerate(args.p_values):
        ax = axes[1 + i]
        res = panel_b_results[p]
        fracs = res["fracs"]
        m_cka, se_cka = res["cka"]
        m_fwd, se_fwd = res["ii_fwd"]
        m_bwd, se_bwd = res["ii_bwd"]

        ax.plot(fracs, m_cka, "-o", color=col_cka, markersize=ms,
                markeredgecolor="black", markeredgewidth=mew, label="CKA")
        ax.fill_between(fracs, m_cka - se_cka, m_cka + se_cka, color=col_cka, alpha=0.2)

        ax.plot(fracs, m_fwd, "-o", color=col_xy, markersize=ms,
                markeredgecolor="black", markeredgewidth=mew,
                label=r"$\Delta(Y \rightarrow Y_{\mathrm{sub}})$")
        ax.fill_between(fracs, m_fwd - se_fwd, m_fwd + se_fwd, color=col_xy, alpha=0.2)

        ax.plot(fracs, m_bwd, "-o", color=col_yx, markersize=ms,
                markeredgecolor="black", markeredgewidth=mew,
                label=r"$\Delta(Y_{\mathrm{sub}} \rightarrow Y)$")
        ax.fill_between(fracs, m_bwd - se_bwd, m_bwd + se_bwd, color=col_yx, alpha=0.2)

        ax.set_xlabel("Fraction of shared features", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=6, loc="center right")
        ax.set_title(f"$p = {p}$", fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{args.output}.pdf", dpi=300)
    fig.savefig(f"{args.output}.png", dpi=300)
    print(f"Saved {args.output}.pdf and {args.output}.png")


if __name__ == "__main__":
    main()
