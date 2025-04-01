import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance
import jax
import jax.numpy as jnp
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, binarize
import pickle
from tqdm import tqdm
from pathlib import Path
import torch

import sys

def main(input_path_A, input_path_B):

    activations_B = {}
    with safe_open(input_path_B, framework="pt", device="cpu") as f:
        for k in f.keys():
            activations_B[k] = f.get_tensor(k)

    sample_size = activations_B["0"].shape[0]
    n_layers = len(list(activations_B.keys()))
    print(activations_B.keys())

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    inf_imb_matrix = []

    row_indices = [str(i) for i in range(1, n_layers)]
    col_indices = [str(i) for i in range(1, n_layers)]
    
    for row_index in tqdm(row_indices, leave=False, desc="Computing row"):
        with safe_open(input_path_A, framework="pt", device="cpu") as f:
            activations_A = f.get_tensor(row_index)
        row = []    
        for b_idx in (pbar := tqdm(col_indices, desc="Computing column")):
            act_A = bf16_torch_to_jax(activations_A)
            act_A = flatten_tokens_features(act_A)
            act_A = binarize(act_A)

            act_B = bf16_torch_to_jax(activations_B[b_idx])
            act_B = flatten_tokens_features(act_B)
            act_B = binarize(act_B)

            inf_imb = information_imbalance_fn(act_A, act_B)
            row.append(inf_imb.item())
            pbar.set_description(f"Computing column (II {inf_imb:.4f})")

        inf_imb_matrix.append(row)

    inf_imb = np.array(inf_imb_matrix)
    np.save(file="inf-imb-llama.npy", arr=inf_imb)

if __name__ == "__main__":
    input_path_A = "/scratch/amascare/llama/llama_flickr_cpt_1.safetensors"
    input_path_B = "/scratch/amascare/llama/llama_flickr_cpt_3.safetensors"

    n_tokens = 10

    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    main(
        input_path_A=input_path_A,
        input_path_B=input_path_B
    )
