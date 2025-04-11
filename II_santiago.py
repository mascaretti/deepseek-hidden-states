import sys
sys.path.append('../')
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open
from information_imbalance_module import (normalized_L2_distance, 
                                          build_information_imbalance, 
                                          hamming_distance, 
                                          build_mutual_k_NN_alignment, 
                                          get_quantiles,
                                          )
import jax
import jax.numpy as jnp
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, binarize, makefolder
import pickle
from tqdm import tqdm
from pathlib import Path
import torch
from datapaths import *
import argparse

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
         k_list,
         output_folder0,
         avg_tokens_II,
         avg_tokens_A,
         diagonal_constraint,
         ):
    
    all_activations_A = collect_data(input_path_A, n_loaded_tokens=n_loaded_tokens, n_files=n_files,mask=mask,)
    all_activations_B = collect_data(input_path_B, n_loaded_tokens=n_loaded_tokens, n_files=n_files,mask=mask,)

    sample_size = all_activations_B["layer_0"].shape[0]
    n_layers = len(list(all_activations_B.keys()))

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)

    rows = [f"layer_{i}" for i in range(1, n_layers)]
    cols = [f"layer_{i}" for i in range(1, n_layers)]

    for n_tokens in n_tokens_list:
        print(f'{n_tokens=}')
        for k_id,k in enumerate(k_list):
            print(f'{k=}')
            information_imbalance_fn = build_information_imbalance(key=subkey, 
                                                                sample_size=sample_size, 
                                                                distance_fn=hamming_distance,
                                                                k=k)
            mutual_k_aligmment_fn = build_mutual_k_NN_alignment(key=subkey, 
                                                                sample_size=sample_size, 
                                                                distance_fn=normalized_L2_distance,
                                                                k=k)
            
            inf_imb_matrix = np.zeros(shape=(len(rows),len(cols)))
            reciprocal_inf_imb_matrix = np.zeros(shape=inf_imb_matrix.shape)
            alignment = np.zeros(shape=(len(rows),
                                        len(cols))
                                        )
            clipping_inf_imb = np.zeros(shape=(2,len(rows),len(cols)))
            
            for A_counter,layer_A in enumerate(rows):
                activations_A = all_activations_A[layer_A]

                for B_counter,layer_B in enumerate(cols):
                    if diagonal_constraint == 1 and layer_B != layer_A:
                        continue
                    
                    act_A = bf16_torch_to_jax(activations_A[:,-n_tokens:,:])
                    act_B = bf16_torch_to_jax(all_activations_B[layer_B][:,-n_tokens:,:])
                    
                    ### binarizing 
                    # start_II = time()
                    if avg_tokens_II == 1:
                        II_act_A = binarize(act_A.mean(axis=1))
                        II_act_B = binarize(act_B.mean(axis=1))
                    else:
                        II_act_A = binarize(flatten_tokens_features(act_A))
                        II_act_B = binarize(flatten_tokens_features(act_B))
                    inf_imb,reciprocal_inf_imb = information_imbalance_fn(II_act_A,II_act_B)

                    # print(f'{(time()-start_II)/60.} m')
                    inf_imb_matrix[A_counter,B_counter] = inf_imb.item()
                    reciprocal_inf_imb_matrix[A_counter,B_counter] = reciprocal_inf_imb.item()
                    
                    ### clipping 
                    if avg_tokens_A:
                        act_A = act_A.mean(axis=1)
                        act_B = act_B.mean(axis=1)
                    else:
                        act_A = flatten_tokens_features(act_A)
                        act_B = flatten_tokens_features(act_B)

                    alphamin = 0.05
                    alphamax = 0.95
                    qmin_A,qmax_A = get_quantiles(act_A,alphamin,alphamax)
                    qmin_B,qmax_B = get_quantiles(act_B,alphamin,alphamax)
                    
                    act_A = jnp.clip(act_A.T,min=qmin_A,max=qmax_A).T
                    act_B = jnp.clip(act_B.T,min=qmin_B,max=qmax_B).T

                    (alignment[A_counter,B_counter],
                     clipping_inf_imb[0,A_counter,B_counter],
                     clipping_inf_imb[1,A_counter,B_counter]) = mutual_k_aligmment_fn(act_A,act_B)

            output_folder = makefolder(base=output_folder0,
                                    create_folder=True,
                                    k=k,
                                    )
            np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens_II}_AB.npy", arr=inf_imb_matrix)
            np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens_II}_BA.npy", arr=reciprocal_inf_imb_matrix)
            np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens_A}_alignment.npy", arr=alignment)
            np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens_A}_clipping_AB.npy", arr=clipping_inf_imb[0])
            np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens_A}_clipping_BA.npy", arr=clipping_inf_imb[1])

        
if __name__ == "__main__":

    mask = 2 # greater than zero
    min_token_length = 40
    n_loaded_tokens = min_token_length - mask  # the mask excludes final points and quotes
    diagonal_constraint = 1

    parser = argparse.ArgumentParser(description="II")
    parser.add_argument("dbg",type=int)
    parser.add_argument("model",type=str)
    parser.add_argument("languageA",type=str)
    parser.add_argument("languageB",type=str)
    parser.add_argument("avg_tokens_II",type=int)
    parser.add_argument("avg_tokens_A",type=int)
    args = parser.parse_args()

    if args.dbg == 0:
        n_files = 500
        # n_tokens_list = np.array([n_loaded_tokens,30,20,10,5,1,15,25,35])
        n_tokens_list = np.array([20])
        k_list = [1,10]
    elif args.dbg == 1:
        n_files = 2
        n_tokens_list = np.array([20,1])
        k_list = [1,2]

    input_path_A = input_paths[args.model][args.languageA][str(min_token_length)]
    input_path_B = input_paths[args.model][args.languageB][str(min_token_length)]
    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    output_folder0 = makefolder(base=f'./results/santiago/',
                               create_folder=True,
                               model=args.model,
                               n_files=n_files,
                               languageA=args.languageA,
                               languageB=args.languageB,
                               mask=mask,
                               )
    main(
        input_path_A=input_path_A,
        input_path_B=input_path_B,
        n_loaded_tokens=n_loaded_tokens,
        mask=mask,
        n_files=n_files,
        n_tokens_list=n_tokens_list,
        k_list=k_list,
        output_folder0=output_folder0,
        avg_tokens_II=args.avg_tokens_II,
        avg_tokens_A=args.avg_tokens_A,
        diagonal_constraint=diagonal_constraint,
    )
