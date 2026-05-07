#!/usr/bin/env python
"""
Compute data for Figure 1: II, CKA, and Neighborhood Overlap on synthetic Gaussians.

Panel A: Y = B_r X + eps with rank(B_r) increasing. Shows II asymmetry vs CKA/NO symmetry.
Panel B: Y_sub is a fraction of Y's features. Two values of p (e.g. 1e3, 1e5).

Saves all results to .npz — no matplotlib or LaTeX needed.

Usage (cluster):
    python notebooks/figure1_compute.py --p_values 1000 100000 --n_reps 10 \
        --n_samples 500 --output notebooks/figure1_data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from tqdm import tqdm

from cka import cka_distance, pairwise_l2
from information_imbalance_module import L2_distance, build_information_imbalance
from rsa import rankdata_flat


# ---------------------------------------------------------------------------
# Neighborhood Overlap
# ---------------------------------------------------------------------------

def _knn_from_dist(D, k):
    """Indices of the k nearest neighbours per row of D, self excluded."""
    n = D.shape[0]
    mask = jnp.eye(n, dtype=bool)
    D_masked = jnp.where(mask, jnp.inf, D)
    _, idx = jax.lax.top_k(-D_masked, k)
    return np.asarray(idx)


def _overlap_fraction(knn_X, knn_Y, k):
    """Mean per-row Jaccard@k of two (n, k) integer kNN index arrays."""
    eq = knn_X[:, :, None] == knn_Y[:, None, :]
    return float(eq.any(axis=2).sum(axis=1).mean() / k)


def neighborhood_overlap(X, Y, k=10):
    """Mean fraction of k-nearest neighbors shared between X and Y."""
    return _overlap_fraction(
        _knn_from_dist(pairwise_l2(X), k),
        _knn_from_dist(pairwise_l2(Y), k),
        k,
    )


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

    ranks = jnp.arange(1, L + 1)
    ranks = jnp.minimum(ranks, dim)

    keys_B = random.split(key_B, L)
    Bs = jnp.stack([make_rank_matrix(keys_B[l], int(ranks[l]), dim=dim) for l in range(L)])

    X = tau * random.normal(key_X, (L, n_samples, dim))
    eps = sigma * random.normal(key_eps, (L, n_samples, dim))
    Y = jnp.einsum("lij,lnj->lni", Bs, X) + eps

    return X, Y, ranks


def compute_panel_a_single(key, L=10, n_samples=500, dim=10, sigma=0.1, k_nn=10):
    """One rep of panel A."""
    key, sim_key = random.split(key)
    X, Y, ranks = simulate_XY(sim_key, L=L, n_samples=n_samples, dim=dim, sigma=sigma)

    imb_XY = np.zeros(L)
    imb_YX = np.zeros(L)
    cka = np.zeros(L)
    spearman = np.zeros(L)
    no = np.zeros(L)

    key, ii_key = random.split(key)
    ii_fn = build_information_imbalance(
        key=ii_key, sample_size=n_samples, distance_fn=L2_distance,
    )

    for l in range(L):
        fwd, bwd = ii_fn(X[l], Y[l])
        imb_XY[l] = float(fwd)
        imb_YX[l] = float(bwd)
        dx = pairwise_l2(X[l])
        dy = pairwise_l2(Y[l])
        cka[l] = float(cka_distance(dx.reshape(-1), dy.reshape(-1)))
        spearman[l] = float(cka_distance(rankdata_flat(dx), rankdata_flat(dy)))
        no[l] = _overlap_fraction(_knn_from_dist(dx, k_nn),
                                   _knn_from_dist(dy, k_nn), k_nn)

    return imb_XY, imb_YX, cka, spearman, no, np.array(ranks)


# ---------------------------------------------------------------------------
# Panel B — II, CKA, NO vs shared features (Gaussian)
# ---------------------------------------------------------------------------

def compute_panel_b_single(key, p=1000, n_samples=500, n_data=100, k_nn=10):
    """One rep of panel B for a given p."""
    key, data_key = random.split(key)
    Y = random.normal(data_key, (n_samples, p), dtype=jnp.float32)

    n_data_eff = min(n_data, p)
    feature_indices = jnp.arange(1, p, max(p // n_data_eff, 1))
    fracs = np.array(feature_indices) / p

    cka_vals = np.zeros(len(feature_indices))
    spearman_vals = np.zeros(len(feature_indices))
    ii_fwd_vals = np.zeros(len(feature_indices))
    ii_bwd_vals = np.zeros(len(feature_indices))
    no_vals = np.zeros(len(feature_indices))

    dist_full_mat = pairwise_l2(Y)
    dist_full_flat = dist_full_mat.reshape(-1)
    rank_full_flat = rankdata_flat(dist_full_flat)
    knn_full = _knn_from_dist(dist_full_mat, k_nn)

    key, ii_key = random.split(key)
    ii_fn = build_information_imbalance(
        key=ii_key, sample_size=n_samples, distance_fn=L2_distance,
    )

    for i, jj in enumerate(feature_indices):
        jj = int(jj)
        Y_sub = Y[:, :jj]
        dist_sub_mat = pairwise_l2(Y_sub)
        dist_sub_flat = dist_sub_mat.reshape(-1)
        cka_vals[i] = float(cka_distance(dist_full_flat, dist_sub_flat))
        spearman_vals[i] = float(cka_distance(rank_full_flat, rankdata_flat(dist_sub_flat)))
        fwd, bwd = ii_fn(Y, Y_sub)
        ii_fwd_vals[i] = float(fwd)
        ii_bwd_vals[i] = float(bwd)
        no_vals[i] = _overlap_fraction(knn_full, _knn_from_dist(dist_sub_mat, k_nn), k_nn)

    return fracs, cka_vals, spearman_vals, ii_fwd_vals, ii_bwd_vals, no_vals


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
    parser = argparse.ArgumentParser(description="Compute Figure 1 data (no plotting)")
    parser.add_argument("--dim", type=int, default=10, help="Dimension D for panel A")
    parser.add_argument("--L", type=int, default=10, help="Number of indices in panel A")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise level for panel A")
    parser.add_argument("--p_values", type=int, nargs="+", default=[1000, 100000],
                        help="Dimension(s) p for panel B")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--n_data", type=int, default=100, help="Feature-fraction steps in panel B")
    parser.add_argument("--n_reps", type=int, default=10, help="Jackknife repetitions")
    parser.add_argument("--k_nn", type=int, default=10, help="k for neighborhood overlap")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="notebooks/figure1_data",
                        help="Output .npz path (no extension)")
    args = parser.parse_args()

    key = random.PRNGKey(args.seed)

    # --- Panel A ---
    print(f"Panel A: {args.n_reps} reps, dim={args.dim}, L={args.L}, k_nn={args.k_nn}")
    runs_XY, runs_YX, runs_cka_a, runs_sp_a, runs_no_a = [], [], [], [], []
    for rep in tqdm(range(args.n_reps), desc="Panel A"):
        key, subkey = random.split(key)
        imb_XY, imb_YX, cka_a, sp_a, no_a, ranks = compute_panel_a_single(
            subkey, L=args.L, n_samples=args.n_samples, dim=args.dim,
            sigma=args.sigma, k_nn=args.k_nn,
        )
        runs_XY.append(imb_XY)
        runs_YX.append(imb_YX)
        runs_cka_a.append(cka_a)
        runs_sp_a.append(sp_a)
        runs_no_a.append(no_a)

    mean_XY, se_XY = jackknife_stats(runs_XY)
    mean_YX, se_YX = jackknife_stats(runs_YX)
    mean_cka_a, se_cka_a = jackknife_stats(runs_cka_a)
    mean_sp_a, se_sp_a = jackknife_stats(runs_sp_a)
    mean_no_a, se_no_a = jackknife_stats(runs_no_a)

    save_dict = {
        "dim": np.array(args.dim),
        "L": np.array(args.L),
        "sigma": np.array(args.sigma),
        "n_samples": np.array(args.n_samples),
        "n_reps": np.array(args.n_reps),
        "k_nn": np.array(args.k_nn),
        "p_values": np.array(args.p_values),
        "ranks": np.array(ranks),
        # Panel A
        "a_mean_XY": mean_XY, "a_se_XY": se_XY,
        "a_mean_YX": mean_YX, "a_se_YX": se_YX,
        "a_mean_cka": mean_cka_a, "a_se_cka": se_cka_a,
        "a_mean_spearman": mean_sp_a, "a_se_spearman": se_sp_a,
        "a_mean_no": mean_no_a, "a_se_no": se_no_a,
    }

    # --- Panel B ---
    for p in args.p_values:
        print(f"Panel B: {args.n_reps} reps, p={p}")
        runs_cka, runs_sp, runs_fwd, runs_bwd, runs_no = [], [], [], [], []
        fracs = None
        for rep in tqdm(range(args.n_reps), desc=f"Panel B (p={p})"):
            key, subkey = random.split(key)
            f, c, sp, fwd, bwd, no = compute_panel_b_single(
                subkey, p=p, n_samples=args.n_samples,
                n_data=args.n_data, k_nn=args.k_nn,
            )
            fracs = f
            runs_cka.append(c)
            runs_sp.append(sp)
            runs_fwd.append(fwd)
            runs_bwd.append(bwd)
            runs_no.append(no)

        m_cka, se_cka = jackknife_stats(runs_cka)
        m_sp, se_sp = jackknife_stats(runs_sp)
        m_fwd, se_fwd = jackknife_stats(runs_fwd)
        m_bwd, se_bwd = jackknife_stats(runs_bwd)
        m_no, se_no = jackknife_stats(runs_no)

        save_dict[f"b_{p}_fracs"] = fracs
        save_dict[f"b_{p}_mean_cka"] = m_cka
        save_dict[f"b_{p}_se_cka"] = se_cka
        save_dict[f"b_{p}_mean_spearman"] = m_sp
        save_dict[f"b_{p}_se_spearman"] = se_sp
        save_dict[f"b_{p}_mean_fwd"] = m_fwd
        save_dict[f"b_{p}_se_fwd"] = se_fwd
        save_dict[f"b_{p}_mean_bwd"] = m_bwd
        save_dict[f"b_{p}_se_bwd"] = se_bwd
        save_dict[f"b_{p}_mean_no"] = m_no
        save_dict[f"b_{p}_se_no"] = se_no

    np.savez(f"{args.output}.npz", **save_dict)
    print(f"Saved {args.output}.npz")


if __name__ == "__main__":
    main()
