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

import sys


def collect_data(input_path, 
                 filter_layer=None, 
                 n_loaded_tokens=40, 
                 n_files=10,
                 mask=0,
                 ):
    files = list_folder(input_path, desc="chunk_")[:n_files]

    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            assert hidden_states.shape[1] >= n_loaded_tokens
            hidden_states = hidden_states[:, -n_loaded_tokens-mask:-mask]            

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                if filter_layer is not None:
                    if layer_idx!=filter_layer: continue

                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)
        # print("Layer=", layer, "activations shape= ", all_hidden_states[layer].shape, flush=True)

    return all_hidden_states

def main(input_path_A, 
         input_path_B, 
         n_loaded_tokens, 
         mask,
         n_files,
         n_tokens_list,
         output_folder,
         ):
    all_activations_A = collect_data(input_path_A, n_loaded_tokens=n_loaded_tokens, n_files=n_files,mask=mask,)
    all_activations_B = collect_data(input_path_B, n_loaded_tokens=n_loaded_tokens, n_files=n_files,mask=mask,)

    sample_size = all_activations_B["layer_0"].shape[0]
    n_layers = len(list(all_activations_B.keys()))
    #print(activations_B.keys())

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    rows = [f"layer_{i}" for i in range(1, n_layers)]
    cols = [f"layer_{i}" for i in range(1, n_layers)]

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

if __name__ == "__main__":

    mask = 2 # greater than zero
    n_loaded_tokens = 40 - mask  # the mask excludes final points and quoutes
    
    dbg = int(sys.argv[1])

    # model = 'llama'
    # languageA = 'english'
    # languageB = 'spanish'

    model = sys.argv[2]
    languageA = sys.argv[3]
    languageB = sys.argv[4]

    # languageA = 'german'
    # languageB = 'hungarian'

    input_paths = {}
    input_paths["deepseek"] = {}
    input_paths["llama"] = {}
    input_paths["deepseek"]["english"] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_english"
    input_paths["deepseek"]["spanish"] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_spanish"
    input_paths["llama"]["english"] = "/home/laios/activations/opus_english_spanish/llama-8b/llama_english"
    input_paths["llama"]["spanish"] = "/home/laios/activations/opus_english_spanish/llama-8b/llama_spanish"
    input_paths["deepseek"]["german"] = f"/home/laios/activations/opus_{languageA}_{languageB}/deepseek_{languageA}"
    input_paths["deepseek"]["hungarian"] = f"/home/laios/activations/opus_{languageA}_{languageB}/deepseek_{languageB}"

    if dbg == 0:
        n_files = 500
        # n_tokens_list = np.array([n_loaded_tokens,30,20,10,5,1,15,25,35])
        n_tokens_list = np.array([2,3,4,6,7,8,9,11,12,13,14])
    elif dbg == 1:
        n_files = 2
        n_tokens_list = np.array([40,1])

    input_path_A = input_paths[model][languageA]
    input_path_B = input_paths[model][languageB]
    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    output_folder = makefolder(base=f'./results/santiago/',
                               create_folder=True,
                               model=model,
                               n_files=n_files,
                               languageA=languageA,
                               languageB=languageB,
                               mask=mask,
                               )
    main(
        input_path_A=input_path_A,
        input_path_B=input_path_B,
        n_loaded_tokens=n_loaded_tokens,
        mask=mask,
        n_files=n_files,
        n_tokens_list=n_tokens_list,
        output_folder=output_folder,
    )
