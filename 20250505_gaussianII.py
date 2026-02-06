#!/usr/bin/env python
# coding: utf-8

import jax
from jax import random
from jax import numpy as jnp
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from scipy.linalg import toeplitz
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance
import pickle

def compute_information(key, Y, distance_fn):
    n, p = Y.shape
    key, subkey = random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=n, distance_fn=distance_fn)
    inf_imb = []
    for jj in tqdm(jnp.arange(1, p, 1)):
        inf_imb.append(np.array(information_imbalance_fn(Y[:, :jj], Y)))
    return np.asarray(inf_imb)

def make_bernoulli(key, p, samples):
    inf_imb = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = 2. * random.bernoulli(key=subkey_data, shape=(n_samples, p)) - 1.
        Y = Y.astype(jnp.int8)
        inf_imb[str(n_samples)] = compute_information(key=subkey_inf, Y=Y, distance_fn=hamming_distance)
    return inf_imb

def make_gaussian(key, p, samples):
    inf_imb = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = random.normal(key=subkey_data, shape=(n_samples, p), dtype=jnp.float32)
        inf_imb[str(n_samples)] = compute_information(key=subkey_inf, Y=Y, distance_fn=L2_distance)
    return inf_imb


print("** SIMULATING RANDOM PROCESS **")
key = random.key(20250507)

p = 100000 ##100000
print(f"p is {p}")

samples = [2000]
print(f"Samples: {samples}")
key, bernoulli_subkey, gaussian_subkey = random.split(key, 3)
bernoulli_inf_imb = {}
gaussian_inf_imb = {}
n_replicas = 10
for ii in tqdm(range(n_replicas)):
    key, bernoulli_subkey, gaussian_subkey = random.split(key, 3)
    bernoulli_inf_imb[str(ii)] = make_bernoulli(key=bernoulli_subkey, p=p, samples=samples)
    gaussian_inf_imb[str(ii)] = make_gaussian(key=gaussian_subkey, p=p, samples=samples)

with open(f'infimbgauss/bernoulli_p_{p}_small.pickle', 'wb') as handle:
    pickle.dump(bernoulli_inf_imb, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'infimbgauss/gaussian_p_{p}_small.pickle', 'wb') as handle:
    pickle.dump(gaussian_inf_imb, handle, protocol=pickle.HIGHEST_PROTOCOL)
