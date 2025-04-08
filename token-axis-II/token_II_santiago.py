from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm
from pathlib import Path
import torch

import os,sys
import argparse

sys.path.append('../')
from II_santiago import collect_data
from datapaths import *
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, binarize, makefolder


def compute_inf_imb_matrix(
        all_activations_A,
        all_activations_B, 
        output_folder,
        ):
    sample_size = all_activations_A["layer_0"].shape[0]
    n_tokens = all_activations_A[f"layer_0"].shape[1]

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    n_layers_A = len(list(all_activations_A.keys()))
    n_layers_B = len(list(all_activations_B.keys()))
    layer_A_ids = [f"layer_{i}" for i in range(1, n_layers_A)]
    layer_B_ids = [f"layer_{i}" for i in range(1, n_layers_B)]

    inf_imb_matrix = np.zeros(shape=(2,
                                    len(layer_A_ids),
                                    n_tokens)
                    )


    all_activations = all_activations_A # Note that we have only one set of activations here...
    for layer_A_id,layer_A in enumerate(layer_A_ids):
        for delta_t_id,delta_t in enumerate(np.arange(1,n_tokens+1,dtype=int)):
            aux = -delta_t+1
            if aux == 0:
                aux = None

            act_A = all_activations[layer_A][:,-1:None,:]
            act_A = bf16_torch_to_jax(act_A)
            act_A = flatten_tokens_features(act_A)
            act_A = binarize(act_A)

            layer_B = layer_A
            act_B = all_activations[layer_B][:,-delta_t:aux,:]
            act_B = bf16_torch_to_jax(act_B)
            act_B = flatten_tokens_features(act_B)
            act_B = binarize(act_B)

            AB,BA = information_imbalance_fn(act_A, act_B)
            inf_imb_matrix[:,layer_A_id,delta_t_id] = np.array([AB.item(),BA.item()])

        np.save(file=f"{output_folder}/n_tokens_{n_tokens}.npy", arr=inf_imb_matrix)

def main(
        input_path,
        output_folder,
        n_files,
        n_loaded_tokens,
        mask,
):

    print("Input path = ", input_path, flush=True)
    
    all_activations = collect_data(input_path, 
                                n_loaded_tokens=n_loaded_tokens, 
                                n_files=n_files,
                                mask=mask)

    compute_inf_imb_matrix(
        all_activations_A=all_activations,
        all_activations_B=all_activations,
        output_folder=output_folder,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Information Imbalance")
    parser.add_argument("--input_path", type=str, required=True, help="Path to activations for dataset A")
    # parser.add_argument("--input_path_B", type=str, required=True, help="Path to activations for dataset B")
    parser.add_argument("--output_folder", type=str, required=True, help="Path of the folder where to store the results")
    parser.add_argument("--n_files",type=int,required=True)
    parser.add_argument("--mask",type=int,required=True)
    parser.add_argument("--n_loaded_tokens",type=int,required=True)
    args = parser.parse_args()
    
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)


    main(
        input_path=args.input_path, 
        # input_path_B=args.input_path_B,
        output_folder=args.output_folder,
        n_files=args.n_files,
        n_loaded_tokens=args.n_loaded_tokens,
        mask=args.mask,
        )