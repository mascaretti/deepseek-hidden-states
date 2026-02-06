#!/usr/bin/env python
# coding: utf-8

"""
Compute CKA for Gaussian and Bernoulli data and store pickles for later plotting.
Outputs are written to the same folder used by 20250505_gaussianII.py (infimbgauss/).

Env vars (defaults: p=100000, samples=2000, 10 replicas):
  P           (default: 100000)  feature dimension
  SAMPLES     (default: 2000)    comma-separated list of sample sizes
  N_DATA      (default: 200)     number of feature steps in CKA (sets spacing via p//N_DATA)
  N_REPLICAS  (default: 10)      number of replicas
  OUT_DIR     (default: infimbgauss) output directory
"""

import os
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from cka import compute_cka


def l2_distance_jax(x, y):
    return jnp.linalg.norm(x - y)


def hamming_distance_jax(x, y):
    return jnp.sum(x != y)


def make_bernoulli(key, p, samples, n_data):
    results = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = 2.0 * random.bernoulli(key=subkey_data, shape=(n_samples, p)) - 1.0
        Y = Y.astype(jnp.int8)
        results[str(n_samples)] = compute_cka(
            key=subkey_inf, Y=Y, distance_fn=hamming_distance_jax, n_data=n_data
        )
    return results


def make_gaussian(key, p, samples, n_data):
    results = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = random.normal(key=subkey_data, shape=(n_samples, p), dtype=jnp.float32)
        results[str(n_samples)] = compute_cka(
            key=subkey_inf, Y=Y, distance_fn=l2_distance_jax, n_data=n_data
        )
    return results


def parse_int_list(env_value, default):
    if not env_value:
        return default
    return [int(s.strip()) for s in env_value.split(",") if s.strip()]


def main():
    p = int(os.environ.get("P", "100000"))
    samples = parse_int_list(os.environ.get("SAMPLES"), [2000])
    n_data = int(os.environ.get("N_DATA", "200"))
    n_replicas = int(os.environ.get("N_REPLICAS", "10"))
    out_dir = Path(os.environ.get("OUT_DIR", "infimbgauss"))
    out_dir.mkdir(exist_ok=True)

    print(f"P={p}, samples={samples}, n_data={n_data}, replicas={n_replicas}")
    key = random.key(20250507)
    bernoulli_out = {}
    gaussian_out = {}

    for rep in range(n_replicas):
        print(f"Replica {rep+1}/{n_replicas}")
        key, bern_key, gauss_key = random.split(key, 3)
        bernoulli_out[str(rep)] = make_bernoulli(bern_key, p=p, samples=samples, n_data=n_data)
        gaussian_out[str(rep)] = make_gaussian(gauss_key, p=p, samples=samples, n_data=n_data)

    bern_path = out_dir / f"cka_bernoulli_p_{p}.pickle"
    gauss_path = out_dir / f"cka_gaussian_p_{p}.pickle"
    with open(bern_path, "wb") as handle:
        pickle.dump(bernoulli_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(gauss_path, "wb") as handle:
        pickle.dump(gaussian_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved Bernoulli CKA to {bern_path}")
    print(f"Saved Gaussian CKA to {gauss_path}")


if __name__ == "__main__":
    main()
