from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance
import jax
import jax.numpy as jnp
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, binarize, makefolder
import pickle
from tqdm import tqdm
from pathlib import Path
import torch
import io
import os
import argparse
import matplotlib.pyplot as plt
from jax import random

def main():
    print("Load tensors...")
    igpt = torch.load("/home/laios/image-information-imbalance/test/igpt_18.pt")
    print("Loaded igpt")
    ds = torch.load("/home/laios/image-information-imbalance/test/ds_2.pt")
    print("Tensors loaded")

    sample_size = ds.shape[0]
    print("Sample size DS: ", sample_size)
    sample_size_B = igpt.shape[0]
    print("Sample size IGPT: ", sample_size_B)
    assert sample_size == sample_size_B

    key = jax.random.key(42)
    print("Random key", key)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    results = {}

    act_ds = bf16_torch_to_jax(ds)
    act_ds = flatten_tokens_features(act_ds)
    act_ds = binarize(act_ds)

    act_igpt = jnp.array(igpt)
    act_igpt = flatten_tokens_features(act_igpt)
    act_igpt = binarize(act_igpt)

    inf_imb, r_inf_imb = information_imbalance_fn(act_ds,  act_igpt)
    results["DS_to_IGPT"] = inf_imb
    results["IGPT_to_DS"] = r_inf_imb
    print("Inf Imb. DS -> IGPT:  ", inf_imb)
    print("Inf Imb. IGPT -> DS: ", r_inf_imb)

    sample_sizes = [1500, 1750, 2000, 2250, 2500]
    inf_imb_samples = []
    r_inf_imb_samples = []
    for i, subset_size in enumerate(sample_sizes):
        print("+++  ", subset_size)
        print(ds[:subset_size, :, :].shape)
        act_ds = bf16_torch_to_jax(ds[:subset_size, :, :])
        act_ds = flatten_tokens_features(act_ds)
        act_ds = binarize(act_ds)

        print(igpt[:subset_size, :, :].shape)
        act_igpt = jnp.array(igpt[:subset_size, :, :])
        act_igpt = flatten_tokens_features(act_igpt)
        act_igpt = binarize(act_igpt)
       
        key, subkey = random.split(key)
        information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=subset_size, distance_fn=hamming_distance)
        
        inf_imb, r_inf_imb = information_imbalance_fn(act_ds[:subset_size], act_igpt[:subset_size])
        print(i, "Inf Imb. DS -> IGPT: ", inf_imb, " Sample size: ", subset_size)
        print("IGPT -> DS ", r_inf_imb)
        results[f"DS_to_IGPT_n_{subset_size}"] = inf_imb
        results[f"IGPT_to_DS_n_{subset_size}"] = r_inf_imb
        inf_imb_samples.append(inf_imb)
        r_inf_imb_samples.append(r_inf_imb)

    fig, ax = plt.subplots()
    ax.plot(sample_sizes, inf_imb_samples, label="DS to IGPT", marker='o')
    ax.axhline(y=0.559, linestyle='--', color="r", label="DS to IGPT - baseline")
    ax.plot(sample_sizes, r_inf_imb_samples, label="IGPT to DS", marker='8')
    ax.axhline(y=0.756, linestyle=":", color="g", label="IGPT to DS - baseline")
    ax.legend()
    ax.set_title("II - Sample Sizes")
    plt.savefig("sample-sizes-ii-debug.pdf", dpi=300)

    tokens = []
    r_tokens = []
    try_tokens = [1, 3, 6, 9]
    key, subkey = random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=2500, distance_fn=hamming_distance)
    for last_tokens in try_tokens: 
        act_ds = bf16_torch_to_jax(ds[:, -last_tokens:, :])
        act_ds = flatten_tokens_features(act_ds)
        act_ds = binarize(act_ds)

        act_igpt = jnp.array(igpt[:, :, :])
        act_igpt = flatten_tokens_features(act_igpt)
        act_igpt = binarize(act_igpt)
        inf_imb, r_inf_imb = information_imbalance_fn(act_ds, act_igpt)
        print(i, "Inf Imb DS -> IGPT: ", inf_imb, "N tokens: ", last_tokens)
        print("IGPT -> DS ", r_inf_imb)
        results[f"DS_to_IGPT_t_{last_tokens}"] = inf_imb
        results[f"IGPT_to_DS_t_{last_tokens}"] =  r_inf_imb
        tokens.append(inf_imb)
        r_tokens.append(r_inf_imb)
    
    
    act_ds = bf16_torch_to_jax(ds)
    act_ds = flatten_tokens_features(act_ds)
    act_ds = binarize(act_ds)
    print(igpt.shape)
    act_igpt = jnp.mean(jnp.array(igpt), axis=2)
    print(act_igpt.shape)
    ## act_igpt = flatten_tokens_features(act_igpt)
    act_igpt = binarize(act_igpt)
    mean_igpt_1, mean_igpt_2 = information_imbalance_fn(act_ds, act_igpt)

    fig, ax = plt.subplots()
    ax.plot(try_tokens, tokens, label="DS to IGPT", marker='o')
    ax.plot(try_tokens, r_tokens, label="IGPT to DS", marker='8')
    ax.axhline(y=mean_igpt_2, linestyle="-.", color="b", label="mean(IGPT) to DS")
    ax.axhline(y=mean_igpt_1, linestyle="--", color="b", label="DS to mean(IGPT)")
    ax.legend()
    ax.set_title("II - Tokens")
    plt.savefig("tokens-ii-debug.pdf", dpi=300)


    
    return 0

if __name__ == "__main__":
    main()



