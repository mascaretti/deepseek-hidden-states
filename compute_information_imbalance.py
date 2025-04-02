from collections import defaultdict
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

def collect_data(input_path, filter_layer=None, n_tokens=40,  max_files=10):
    files = list_folder(input_path, desc="chunk_")[:max_files]

    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            assert hidden_states.shape[1] > n_tokens
            hidden_states = hidden_states[:, :n_tokens]

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                if filter_layer is not None:
                    if layer_idx!=filter_layer: continue

                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)
        # print("Layer=", layer, "activations shape= ", all_hidden_states[layer].shape, flush=True)

    return all_hidden_states

def main(input_path_A, input_path_B, n_tokens):
    all_activations_B = collect_data(input_path_B, n_tokens=n_tokens)
    all_activations_A = collect_data(input_path_A, n_tokens=n_tokens)

    sample_size = all_activations_B["layer_0"].shape[0]
    n_layers = len(list(all_activations_B.keys()))
    #print(activations_B.keys())

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    inf_imb_matrix = []

    rows = [f"layer_{i}" for i in range(1, n_layers)]
    cols = [f"layer_{i}" for i in range(1, n_layers)]
    
    for row in tqdm(rows, leave=False, desc="Computing row"):
        activations_A = all_activations_A[row]

        row = []    
        for col in (pbar := tqdm(cols, desc="Computing column")):
            act_A = bf16_torch_to_jax(activations_A)
            act_A = flatten_tokens_features(act_A)
            act_A = binarize(act_A)

            act_B = bf16_torch_to_jax(all_activations_B[col])
            act_B = flatten_tokens_features(act_B)
            act_B = binarize(act_B)

            inf_imb = information_imbalance_fn(act_A, act_B)
            row.append(inf_imb.item())
            pbar.set_description(f"Computing column (II {inf_imb:.4f})")

        inf_imb_matrix.append(row)

    inf_imb = np.array(inf_imb_matrix)
    np.save(file="inf-imb-llama.npy", arr=inf_imb)

if __name__ == "__main__":
    input_path_A = "/home/laios/activations/flickr/deepseek_captions1"
    input_path_B = "/home/laios/activations/flickr/deepseek_captions3"

    n_tokens = 8

    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    main(
        input_path_A=input_path_A,
        input_path_B=input_path_B,
        n_tokens=n_tokens
    )
