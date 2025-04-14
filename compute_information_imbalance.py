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

import os
import argparse

def collect_data(input_path, filter_layer=None, n_loaded_tokens=40, n_files=10):
    files = list_folder(input_path, desc="chunk_")[:n_files]

    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            assert hidden_states.shape[1] >= n_loaded_tokens
            hidden_states = hidden_states[:, -n_loaded_tokens:]

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                if filter_layer is not None:
                    if layer_idx!=filter_layer: continue

                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)
        # print("Layer=", layer, "activations shape= ", all_hidden_states[layer].shape, flush=True)

    return all_hidden_states

def compute_inf_imb_matrix(
        all_activations_A, 
        all_activations_B, 
        n_tokens_list,
        output_folder,
        ):
    sample_size = all_activations_B["layer_0"].shape[0]

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    n_layers_A = len(list(all_activations_A.keys()))
    n_layers_B = len(list(all_activations_B.keys()))
    rows = [f"layer_{i}" for i in range(1, n_layers_A)]
    cols = [f"layer_{i}" for i in range(1, n_layers_B)]

    for n_tokens in tqdm(n_tokens_list,desc='Computing token'):
        
        inf_imb_matrix = []
        reciprocal_inf_imb_matrix = []
        
        for layer_A in tqdm(rows, desc="Computing layer_A",leave=False):
            activations_A = all_activations_A[layer_A]

            row = []
            reciprocal_row = []
            for layer_B in (pbar := tqdm(cols, desc="Computing layer_B",leave=False)):
                act_A = bf16_torch_to_jax(activations_A[:,-n_tokens:,:])
                act_A = flatten_tokens_features(act_A)
                act_A = binarize(act_A)

                act_B = bf16_torch_to_jax(all_activations_B[layer_B][:,-n_tokens:,:])
                act_B = flatten_tokens_features(act_B)
                act_B = binarize(act_B)

                inf_imb,reciprocal_inf_imb = information_imbalance_fn(act_A, act_B)
                row.append(inf_imb.item())
                reciprocal_row.append(reciprocal_inf_imb.item())
                pbar.set_description(f"Computing layer_B (II(A-->B) {inf_imb:.4f} ; II(B-->A) {reciprocal_inf_imb:.4f})")

            inf_imb_matrix.append(row)
            reciprocal_inf_imb_matrix.append(reciprocal_row)

        inf_imb = np.array(inf_imb_matrix)
        reciprocal_inf_imb = np.array(reciprocal_inf_imb_matrix)
        np.save(file=f"{output_folder}/n_tokens_{n_tokens}_AB.npy", arr=inf_imb)
        np.save(file=f"{output_folder}/n_tokens_{n_tokens}_BA.npy", arr=reciprocal_inf_imb)

def main(
        input_path_A,
        input_path_B,
        output_folder,
        n_loaded_tokens,
        n_files,
):
    n_tokens_list = np.arange(1, n_loaded_tokens+1, 1, dtype=int)

    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    all_activations_A = collect_data(input_path_A, n_loaded_tokens=n_loaded_tokens, n_files=n_files)
    all_activations_B = collect_data(input_path_B, n_loaded_tokens=n_loaded_tokens, n_files=n_files)

    compute_inf_imb_matrix(
        all_activations_A=all_activations_A,
        all_activations_B=all_activations_B,
        n_tokens_list=n_tokens_list,
        output_folder=output_folder,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Information Imbalance")
    parser.add_argument("--input_path_A", type=str, required=True, help="Path to activations for dataset A")
    parser.add_argument("--input_path_B", type=str, required=True, help="Path to activations for dataset B")
    parser.add_argument("--output_folder", type=str, required=True, help="Path of the folder where to store the results")
    args = parser.parse_args()
    
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    n_loaded_tokens = 9
    n_files = 50

    main(
        input_path_A=args.input_path_A, 
        input_path_B=args.input_path_B,
        output_folder=args.output_folder,
        n_loaded_tokens=n_loaded_tokens,
        n_files=n_files,
        )
