#!/usr/bin/env python
# coding: utf-8

"""
Combined Gaussian experiment:
- Information imbalance (like 20250505_gaussianII.py)
- CKA (like 20251201_mat_gaussianII.py)

Supports multiple p values in one run and plots all curves with the
notebook styling (scienceplots). Defaults are adjustable via env vars:
  P_LIST (comma-separated, default: 1000,10000,100000)
  SAMPLES (comma-separated, default: 2000)
  MAXK (int, default: 80)
  SUBSET_SIZE (int, optional)
  FEATURE_POINTS (int, number of feature grid points, default: 50)
  CKA_N_DATA (int, default: 200)
"""

import os
from pathlib import Path
from typing import List, Sequence, Tuple

# Keep matplotlib cache inside the repo to avoid permission issues
mpl_cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pickle

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from information_imbalance.imbalance import (
    build_information_imbalance,
    l2_distance,
)
from cka import build_cka_distance

try:
    plt.style.use(["nature", "science"])
except Exception:
    pass


def l2_distance_jax(x, y):
    return jnp.linalg.norm(x - y)


def lin_feature_grid(p: int, n_points: int) -> np.ndarray:
    # Evenly spaced feature counts from 1 to p (exclusive of p to avoid trivial 1.0 CKA)
    points = np.linspace(1, p - 1, num=n_points, dtype=int)
    points = np.unique(points)
    points = points[points > 0]
    return points


def compute_info_and_cka(
    key: jnp.ndarray,
    Y: jnp.ndarray,
    feature_counts: Sequence[int],
    maxk: int,
    subset_size: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute information imbalance (X->Y, Y->X) and CKA between full Y and feature subsets.
    Returns: imbal_xy, imbal_yx, cka_vals arrays shaped (len(feature_counts),)
    """
    n, _ = Y.shape
    rng_seed = int(random.randint(key, shape=(), minval=0, maxval=2**31 - 1))
    info_fn = build_information_imbalance(
        distance_fn=l2_distance,
        k=1,
        maxk=maxk,
        subset_size=subset_size,
        rng=rng_seed,
    )

    imbal_xy = []
    imbal_yx = []
    cka_vals = []
    Y_full_np = np.asarray(Y)
    Y_full_jnp = jnp.asarray(Y)

    for m in tqdm(feature_counts):
        Y_sub_np = Y_full_np[:, :m]
        d_xy, d_yx = info_fn(Y_sub_np, Y_full_np)
        imbal_xy.append(d_xy)
        imbal_yx.append(d_yx)

        Y_sub_jnp = Y_full_jnp[:, :m]
        cka_val = build_cka_distance(Y_full_jnp, Y_sub_jnp, distance_fn=l2_distance_jax)
        cka_vals.append(float(cka_val))

    return np.asarray(imbal_xy), np.asarray(imbal_yx), np.asarray(cka_vals)


def main():
    seed = 20250507
    base_key = random.key(seed)

    p_env = os.environ.get("P_LIST", "1000,10000,100000")
    p_values = [int(s.strip()) for s in p_env.split(",") if s.strip()]

    samples_env = os.environ.get("SAMPLES", "2000")
    samples_list = [int(s.strip()) for s in samples_env.split(",") if s.strip()]

    maxk = int(os.environ.get("MAXK", "80"))
    subset_env = os.environ.get("SUBSET_SIZE")
    subset_size = int(subset_env) if subset_env else None
    n_points = int(os.environ.get("FEATURE_POINTS", "50"))
    cka_n_data = int(os.environ.get("CKA_N_DATA", "200"))

    print("P values:", p_values)
    print("Samples per run:", samples_list)
    print(f"maxk={maxk}, subset_size={subset_size}, feature_points={n_points}, cka_n_data={cka_n_data}")

    out_dir = Path("gauss_combo")
    out_dir.mkdir(exist_ok=True)

    results = {}

    for p in p_values:
        for n_samples in samples_list:
            print(f"\n** Running p={p}, n_samples={n_samples} **")
            cache_file = out_dir / f"data_p{p}_n{n_samples}.npz"
            if cache_file.exists() and os.environ.get("FORCE_RECOMPUTE", "0") != "1":
                data = np.load(cache_file)
                results[(p, n_samples)] = {
                    "frac": data["frac"],
                    "counts": data["counts"],
                    "imbal_xy": data["imbal_xy"],
                    "imbal_yx": data["imbal_yx"],
                    "cka": data["cka"],
                }
                print(f"Loaded cached results from {cache_file}")
                continue

            base_key, subkey = random.split(base_key)
            Y = random.normal(key=subkey, shape=(n_samples, p), dtype=jnp.float32)

            feature_counts = lin_feature_grid(p, n_points=n_points)
            # Ensure CKA sampling roughly matches n_data spacing
            step = max(1, p // cka_n_data)
            cka_counts = np.arange(1, p, step, dtype=int)
            counts = np.unique(np.concatenate([feature_counts, cka_counts]))
            counts = counts[counts < p]

            imbal_xy, imbal_yx, cka_vals = compute_info_and_cka(
                key=subkey,
                Y=Y,
                feature_counts=counts,
                maxk=maxk,
                subset_size=subset_size,
            )

            frac = counts / p
            payload = {
                "frac": frac,
                "counts": counts,
                "imbal_xy": imbal_xy,
                "imbal_yx": imbal_yx,
                "cka": cka_vals,
            }
            results[(p, n_samples)] = payload
            np.savez(
                cache_file,
                frac=frac,
                counts=counts,
                imbal_xy=imbal_xy,
                imbal_yx=imbal_yx,
                cka=cka_vals,
            )

    # Build simplified legends: colors per p, styles per metric
    # Plot all p on one figure
    plt.figure(figsize=(3.5, 2.0))
    use_single_palette = len(results) == 1
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))
    if use_single_palette:
        metric_colors = {
            "imbal_xy": "#364F6B",
            "imbal_yx": "#609966",
            "cka": "#C44536",
        }
    for idx, ((p, n_samples), data) in enumerate(results.items()):
        base_color = colors[idx]
        color_xy = metric_colors["imbal_xy"] if use_single_palette else base_color
        color_yx = metric_colors["imbal_yx"] if use_single_palette else base_color
        color_cka = metric_colors["cka"] if use_single_palette else base_color
        label_base = f"p={p}, n={n_samples}"
        plt.plot(
            data["frac"],
            data["imbal_xy"],
            "-o",
            color=color_xy,
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=label_base if use_single_palette else f"{label_base} Δ(X→Y)",
        )
        plt.plot(
            data["frac"],
            data["imbal_yx"],
            "--o",
            color=color_yx,
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=None if use_single_palette else f"{label_base} Δ(Y→X)",
        )
        plt.plot(
            data["frac"],
            data["cka"],
            "-s",
            color=color_cka,
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=None if use_single_palette else f"{label_base} CKA",
        )

    plt.xlabel("% of shared features", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "gauss_infoimb_cka.pdf", dpi=300)
    plt.savefig(out_dir / "gauss_infoimb_cka.png", dpi=300)


if __name__ == "__main__":
    main()
