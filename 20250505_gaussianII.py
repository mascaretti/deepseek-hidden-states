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

def compute_information(key, Y):
    n, p = Y.shape
    print(f"Y: {Y.shape}")
    key, subkey = random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=n, distance_fn=hamming_distance)
    inf_imb = []
    for jj in tqdm(jnp.arange(1, p + 1, 100)):
        inf_imb.append(np.array(information_imbalance_fn(Y[:, :jj], Y)[0]))
        print(inf_imb)
        exit()
    return np.asarray(inf_imb)

def make_bernoulli(key, p, samples):
    inf_imb = {}
    for n_samples in tqdm(samples):
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = 2. * random.bernoulli(key=subkey_data, shape=(n_samples, p)) - 1.
        Y = Y.astype(jnp.int8)
        inf_imb[str(n_samples)] = compute_information(key=subkey_inf, Y=Y)
    return inf_imb
        
key = random.key(20250507)

p = 100_000

samples = [2000, 4000]
key, subkey = random.split(key)
inf_imb = make_bernoulli(key=subkey, p=p, samples=samples)
with open(f'infimbgauss/bernoulli_p_{p}.pickle', 'wb') as handle:
        pickle.dump(inf_imb, handle, protocol=pickle.HIGHEST_PROTOCOL)
