#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from tqdm import tqdm

from compute_ii_mean import load_mean_activations


# In[8]:


try:
    from dadapy import data
except ModuleNotFoundError:
    print("Local version: it won't work")


# In[3]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_prefix", type=str, default="results/twoNN")
    args = parser.parse_args()

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("Loading activations...")
    act = load_mean_activations(args.input_path, subset_size=args.subset_size)
    print(f"ImageGPT: {len(act)} layers, embed={act['layer_0'].shape[1]}")

    # ------------------------------------------------------------------
    # twoNN sweep
    # ------------------------------------------------------------------
    def sweep(act):
        n_layers = len(act)
        results = []
        radii = []
        for i in tqdm(range(n_layers), desc="twoNN layers"):
            raw = jnp.array(act[f"layer_{i}"])
            # initialise the Data class
            _data = data.Data(np.array(raw))
            # estimate ID
            id_twoNN, _, r = _data.compute_id_2NN()
            results.append(id_twoNN)
            radii.append(r)
        return jnp.array(results), jnp.array(radii)

    print("\nComputing twoNN...")
    results, radii = sweep(act)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dict = {
        "two_nn": np.array(results),
        "radii": np.array(radii),
    }
    np.savez(f"{args.output_prefix}.npz", **save_dict)
    print(f"Saved {args.output_prefix}.npz")


    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    n_layers = len(act)
    rel = np.arange(n_layers) / n_layers
    ax.plot(rel, results, "-o")
    ax.set_xlabel("Relative depth", fontsize=10)
    ax.set_ylabel("twoNN", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.output_prefix}.{ext}", dpi=300)
    print(f"Saved {args.output_prefix}.pdf/.png")




if __name__ == "__main__":
    main()
