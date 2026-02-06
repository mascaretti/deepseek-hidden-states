#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

# Ensure matplotlib cache stays inside the workspace (avoids permission issues)
mpl_cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import jax
from jax import random
from jax import numpy as jnp
import matplotlib.pyplot as plt
import scienceplots  # Requires LaTeX
import numpy as np
import pickle

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

from cka import build_cka_distance, compute_cka

try:
    plt.style.use(['nature', 'science'])  # Match notebook styling
except Exception:
    pass


def l2_distance_jax(x, y):
    return jnp.linalg.norm(x - y)


def hamming_distance_jax(x, y):
    return jnp.sum(x != y)

def make_bernoulli(key, p, samples, n_data):
    inf_imb = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = 2. * random.bernoulli(key=subkey_data, shape=(n_samples, p)) - 1.
        Y = Y.astype(jnp.int8)
        inf_imb[str(n_samples)] = compute_cka(key=subkey_inf, Y=Y, distance_fn=hamming_distance_jax, n_data=n_data)
    return inf_imb

def make_gaussian(key, p, samples, n_data):
    cka = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = random.normal(key=subkey_data, shape=(n_samples, p), dtype=jnp.float32)
        cka[str(n_samples)] = compute_cka(key=subkey_inf, Y=Y, distance_fn=l2_distance_jax, n_data=n_data)
    return cka


print("** SIMULATING RANDOM PROCESS **")
key = random.key(20250507)

p = int(os.environ.get("CKA_P", "100000"))
print(f"p is {p}")

samples_env = os.environ.get("CKA_SAMPLES")
samples = [int(s.strip()) for s in samples_env.split(",") if s.strip()] if samples_env else [2000]
print(f"Samples: {samples}")
n_data = int(os.environ.get("CKA_N_DATA", "200"))
print(f"n_data: {n_data}")

data_dir = Path("cka_gauss")
data_dir.mkdir(exist_ok=True)
bernoulli_path = data_dir / f'bernoulli_p_{p}_small.pickle'
gaussian_path = data_dir / f'gaussian_p_{p}_small.pickle'

SKIP_COMPUTE = os.environ.get("CKA_SKIP_COMPUTE", "1") != "0"  # Set to False to recompute

if SKIP_COMPUTE:
    print("Skipping computation, loading existing data...")
    try:
        with open(bernoulli_path, 'rb') as handle:
            bernoulli_inf_imb = pickle.load(handle)
        with open(gaussian_path, 'rb') as handle:
            gaussian_cka = pickle.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Precomputed data missing for p={p}. "
            "Set CKA_SKIP_COMPUTE=0 to recompute."
        ) from exc
else:
    print("Computing from scratch...")
    key, bernoulli_subkey, gaussian_subkey = random.split(key, 3)
    bernoulli_inf_imb = {}
    gaussian_cka = {}
    n_replicas = 1
    for ii in tqdm(range(n_replicas)):
        key, bernoulli_subkey, gaussian_subkey = random.split(key, 3)
        bernoulli_inf_imb[str(ii)] = make_bernoulli(key=bernoulli_subkey, p=p, samples=samples, n_data=n_data)
        gaussian_cka[str(ii)] = make_gaussian(key=gaussian_subkey, p=p, samples=samples, n_data=n_data)

    with open(bernoulli_path, 'wb') as handle:
        pickle.dump(bernoulli_inf_imb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(gaussian_path, 'wb') as handle:
        pickle.dump(gaussian_cka, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plot in same style as asymmetry
plt.figure(figsize=(3.5, 2.))
for replica in gaussian_cka.keys():
    for sample_size in gaussian_cka[replica].keys():
        cka_values = gaussian_cka[replica][sample_size]
        x_values = np.linspace(0, 1, len(cka_values))
        plt.plot(x_values, cka_values, '-o', markersize=3, markeredgecolor='black', markeredgewidth=0.3, label=f'Replica {replica}, Samples {sample_size}')
plt.xlabel('% of shared features', fontsize=11)
plt.ylabel('CKA', fontsize=11)
plt.ylim(-0.1, 1.0)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=6, loc="upper right")
plt.tight_layout()
plt.savefig(data_dir / f'cka_p_{p}_small.pdf', dpi=300)
plt.savefig(data_dir / f'cka_p_{p}_small.png', dpi=300)
