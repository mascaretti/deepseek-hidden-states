#!/usr/bin/env python
# coding: utf-8

"""
Compute information imbalance for Gaussian and Bernoulli features,
mirroring 20250505_gaussianII.py but using the installed
information_imbalance package, and plot with the notebook styling.
"""

import os
from pathlib import Path

# Keep matplotlib cache inside the repo to avoid permission issues
mpl_cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import random
import jax.numpy as jnp

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from information_imbalance.imbalance import (
    build_information_imbalance,
    hamming_distance,
    l2_distance,
)

try:
    plt.style.use(["nature", "science"])
except Exception:
    pass


def compute_information(key, Y, distance_fn, maxk=None, subset_size=None, n_points=50):
    """Compute information imbalance as a function of shared features."""
    n, p = Y.shape
    maxk = maxk if maxk is not None else max(1, min(80, n - 1))
    rng_seed = int(random.randint(key, shape=(), minval=0, maxval=2**31 - 1))
    info_fn = build_information_imbalance(
        distance_fn=distance_fn,
        k=1,
        maxk=maxk,
        subset_size=subset_size,
        rng=rng_seed,
    )

    step = max(1, p // n_points)
    imbalances = []
    feature_counts = list(range(1, p, step))
    for jj in tqdm(feature_counts):
        # Use the same samples with growing feature subsets
        imbalances.append(np.array(info_fn(Y[:, :jj], Y)))
    return np.asarray(imbalances), np.asarray(feature_counts)


def make_bernoulli(key, p, samples, maxk=None, subset_size=None, n_points=50):
    inf_imb = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = 2.0 * random.bernoulli(key=subkey_data, shape=(n_samples, p)) - 1.0
        Y = Y.astype(jnp.int8)
        vals, counts = compute_information(
            key=subkey_inf,
            Y=Y,
            distance_fn=hamming_distance,
            maxk=maxk,
            subset_size=subset_size,
            n_points=n_points,
        )
        inf_imb[str(n_samples)] = {"values": vals, "counts": counts}
    return inf_imb


def make_gaussian(key, p, samples, maxk=None, subset_size=None, n_points=50):
    inf_imb = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = random.normal(key=subkey_data, shape=(n_samples, p), dtype=jnp.float32)
        vals, counts = compute_information(
            key=subkey_inf,
            Y=Y,
            distance_fn=l2_distance,
            maxk=maxk,
            subset_size=subset_size,
            n_points=n_points,
        )
        inf_imb[str(n_samples)] = {"values": vals, "counts": counts}
    return inf_imb


def main():
    print("** SIMULATING RANDOM PROCESS (information imbalance)**")
    key = random.key(20250507)

    p = int(os.environ.get("INFIMB_P", "200"))
    print(f"p is {p}")

    samples_env = os.environ.get("INFIMB_SAMPLES")
    samples = [int(s.strip()) for s in samples_env.split(",") if s.strip()] if samples_env else [200]
    print(f"Samples: {samples}")

    maxk_env = os.environ.get("INFIMB_MAXK")
    maxk = int(maxk_env) if maxk_env else None
    subset_env = os.environ.get("INFIMB_SUBSET")
    subset_size = int(subset_env) if subset_env else None
    n_points = int(os.environ.get("INFIMB_POINTS", "50"))

    data_dir = Path("infimbgauss_plotstyle")
    data_dir.mkdir(exist_ok=True)
    bern_path = data_dir / f"bernoulli_p_{p}_data.pickle"
    gauss_path = data_dir / f"gaussian_p_{p}_data.pickle"

    print("Computing information imbalance...")
    key, bernoulli_subkey, gaussian_subkey = random.split(key, 3)
    bernoulli_inf_imb = make_bernoulli(
        key=bernoulli_subkey,
        p=p,
        samples=samples,
        maxk=maxk,
        subset_size=subset_size,
        n_points=n_points,
    )
    gaussian_inf_imb = make_gaussian(
        key=gaussian_subkey,
        p=p,
        samples=samples,
        maxk=maxk,
        subset_size=subset_size,
        n_points=n_points,
    )

    # Persist results
    import pickle

    with open(bern_path, "wb") as handle:
        pickle.dump(bernoulli_inf_imb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(gauss_path, "wb") as handle:
        pickle.dump(gaussian_inf_imb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot like the asymmetry notebook: relative feature fraction on x-axis
    plt.figure(figsize=(3.5, 2.0))
    colors = {"gaussian": "#364F6B", "bernoulli": "#609966"}

    for sample_size, payload in gaussian_inf_imb.items():
        frac = payload["counts"] / p
        vals = payload["values"]
        plt.plot(
            frac,
            vals[:, 0],  # Delta(X->Y)
            "-o",
            color=colors["gaussian"],
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=f"Gaussian Δ (n={sample_size})",
        )
        plt.plot(
            frac,
            vals[:, 1],  # Delta(Y->X)
            "--o",
            color=colors["gaussian"],
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=f"Gaussian Δ rev (n={sample_size})",
        )

    for sample_size, payload in bernoulli_inf_imb.items():
        frac = payload["counts"] / p
        vals = payload["values"]
        plt.plot(
            frac,
            vals[:, 0],
            "-o",
            color=colors["bernoulli"],
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=f"Bernoulli Δ (n={sample_size})",
        )
        plt.plot(
            frac,
            vals[:, 1],
            "--o",
            color=colors["bernoulli"],
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=f"Bernoulli Δ rev (n={sample_size})",
        )

    plt.xlabel("% of shared features", fontsize=11)
    plt.ylabel("Information imbalance", fontsize=11)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(data_dir / f"infimb_p_{p}.pdf", dpi=300)
    plt.savefig(data_dir / f"infimb_p_{p}.png", dpi=300)


if __name__ == "__main__":
    main()
