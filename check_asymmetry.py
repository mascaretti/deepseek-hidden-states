from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance, build_information_matrix
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
from jax import random as random
from safetensors.numpy import save_file

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def collect_data(input_path, filter_layer=None, n_loaded_tokens=9, subset_size=3000):
    files = list_folder(input_path, desc="chunk_")

    all_hidden_states = defaultdict(list)

    file_counter = 0
    for file in tqdm(files, desc="Collect File", leave=False):
        if file_counter >= subset_size:
            break
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))
        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            assert hidden_states.shape[1] >= n_loaded_tokens
            hidden_states = hidden_states[:, -n_loaded_tokens:]
            file_counter += 1

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                if filter_layer is not None:
                    if layer_idx!=filter_layer: continue

                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)[:subset_size]
    
    print(all_hidden_states["layer_0"].shape)
    return all_hidden_states

def collect_images(input_path, n_files, last_tokens, desc="chunk_"):
    files = list_folder(input_path, desc=desc)[:n_files]
    all_hidden_states = defaultdict(list)
    for file in tqdm(files, desc="Collect images"):
        batch_activations = CPU_Unpickler(open(Path(input_path, file.name), 'rb')).load()
        for layer_key in batch_activations.keys():
            all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.cat(tensors, dim=0)
    return all_hidden_states


def compute_inf_imb_matrix(
        all_activations_A, 
        all_activations_B, 
        output_folder
):
    
    key = jax.random.key(42)
    print(f"Initial key: {key}")
    
    n_layers_A = len(list(all_activations_A.keys()))
    n_layers_B = len(list(all_activations_B.keys()))

    cols = [f"layer_{i}" for i in range(1, n_layers_B)]

    sample_size = all_activations_A["layer_1"].shape[0]
    print("Sample size: ", sample_size)
    sample_size_B = all_activations_B["layer_1"].shape[0]
    assert sample_size == sample_size_B
        
    key, subkey = jax.random.split(key)
    information_matrix_fn = build_information_matrix(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    layer_A = "layer_52"
    print(f"Layer A: {layer_A}")
    activations_A = all_activations_A[layer_A]
    
    for layer_B in (pbar := tqdm(cols, desc="Computing layer_B",leave=False)):
        act_A = bf16_torch_to_jax(activations_A[:,-9:,:])
        act_A = flatten_tokens_features(act_A)
        act_A = binarize(act_A)
                 
        act_B = jnp.array(all_activations_B[layer_B])
        act_B = flatten_tokens_features(act_B)
        act_B = binarize(act_B)

        inf_imb = information_matrix_fn(act_A, act_B)
        save_file(inf_imb, Path(output_folder, f"{layer_B}.safetensors"))
        

def main(
        input_path_A,
        input_path_B,
        output_folder,
        n_files_b
):
    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    print("Output: ", output_folder)
    
    all_activations_A = collect_data(input_path_A, n_loaded_tokens=9, subset_size=subset_size_a)
    sample_size_A = all_activations_A["layer_0"].shape[0]
    print(f"Sample size A: {sample_size_A}")
    n_text_tokens =  all_activations_A["layer_0"].shape[1]
    print(f"Tokens: {n_text_tokens}")
   
    print(f"N_files_b: {n_files_b}")
    all_activations_B = collect_images(input_path_B, n_files=n_files_b, last_tokens=200, desc="image-chunk_")
    try:
        sample_size_B = all_activations_B["layer_0"].shape[0]
        print(f"Sample size B: {sample_size_B}")
    except AttributeError:
        sample_size_B = len(all_activations_B["layer_0"])
        print(f"Sample size B: {sample_size_B}")

    compute_inf_imb_matrix(
        all_activations_A=all_activations_A,
        all_activations_B=all_activations_B,
        output_folder=output_folder
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Information Imbalance")
    parser.add_argument("--input_path_A", type=str, required=True, help="Path to activations for dataset A")
    parser.add_argument("--input_path_B", type=str, required=True, help="Path to activations for dataset B")
    parser.add_argument("--output_folder", type=str, required=True, help="Path of the folder where to store the results")
    args = parser.parse_args()
    
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    subset_size_a = 5000
    n_files_b = int(subset_size_a / 10)
    print(f"Subset size: {subset_size_a}")
    print(f"n_files_b: {n_files_b}")

    main(
        input_path_A=args.input_path_A, 
        input_path_B=args.input_path_B,
        output_folder=args.output_folder,
        n_files_b=n_files_b
    )
