import sys,os
sys.path.append('../')
dbg = int(sys.argv[1])
if dbg:
    os.environ["JAX_PLATFORMS"] = "cpu"
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open
from information_imbalance_module import (normalized_L2_distance, 
                                          build_information_imbalance, 
                                          hamming_distance, 
                                          build_mutual_k_NN_alignment, 
                                          )
import jax
import jax.numpy as jnp
from utils import (bf16_torch_to_jax, 
                   flatten_tokens_features, 
                   list_folder, 
                   binarize, 
                   clip,
                   makefolder,
                   )
import pickle
from tqdm import tqdm
from pathlib import Path
import torch
from datapaths import *
import argparse

def take_last_hidden_states(hidden_states,mask,n_loaded_tokens):
    return hidden_states[:, -n_loaded_tokens-mask:-mask]

def take_first_hidden_states(hidden_states,mask,n_loaded_tokens):
    return hidden_states[:, mask:n_loaded_tokens+mask]

def collect_data(input_path, 
                 n_loaded_tokens, 
                 n_files,
                 mask,
                 take_last,
                 filter_layer=None, 
                 ):
    files = list_folder(input_path, desc="chunk_")[:n_files]

    all_hidden_states = defaultdict(list)
    
    if take_last:
        slice_hidden_states = take_last_hidden_states
    else:
        slice_hidden_states = take_first_hidden_states

    for file in tqdm(files, desc="Collect File"):
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            assert hidden_states.shape[1] >= n_loaded_tokens
            hidden_states = slice_hidden_states(hidden_states,mask,n_loaded_tokens)            
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
        #  input_path_B, 
         n_loaded_tokens, 
         mask,
         n_files,
         seed_list,
         k_list,
         output_folder0,
        #  avg_flags,
         Nbits_list,
        #  diagonal_constraint,
        take_last,
         ):
    
    all_activations_A = collect_data(input_path_A, 
                                     n_loaded_tokens=n_loaded_tokens, 
                                     n_files=n_files,
                                     mask=mask,
                                     take_last=take_last,
                                     )
    all_activations_B = all_activations_A

    sample_size = all_activations_B["layer_0"].shape[0]
    n_layers = len(list(all_activations_B.keys()))

    rows = [f"layer_{i}" for i in range(1, n_layers)]

    for seed_id, seed in enumerate(seed_list):
        print(f'{seed=}')
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)

        for k_id, k in enumerate(k_list):
            print(f'{k=}')

            for Nbits_id, Nbits in enumerate(Nbits_list):
                print(f'{Nbits=}')

                distance_fn = hamming_distance if Nbits else normalized_L2_distance 
                mutual_k_aligmment_fn = build_mutual_k_NN_alignment(key=subkey, 
                                                                    sample_size=sample_size, 
                                                                    distance_fn=distance_fn,
                                                                    k=k)
                
                alignment = np.zeros(shape=(len(rows),
                                            n_loaded_tokens)
                                            )
                inf_imb = np.zeros(shape=(2,len(rows),
                                          n_loaded_tokens))
                
                for A_counter,layer_A in enumerate(rows):
                    layer_B = layer_A

                    for delta_t_id,delta_t in enumerate(np.arange(1,n_loaded_tokens+1,dtype=int)):
                        aux = -delta_t+1
                        if aux == 0:
                            aux = None
                        act_A = all_activations_A[layer_A][:,-1:None,:]
                        act_B = all_activations_B[layer_B][:,-delta_t:aux,:]

                        act_A = bf16_torch_to_jax(act_A)
                        act_B = bf16_torch_to_jax(act_B)
                        
                        act_A = flatten_tokens_features(act_A)
                        act_B = flatten_tokens_features(act_B)

                        if Nbits == 1:
                            act_A = binarize(act_A)
                            act_B = binarize(act_B)
                        else:
                            act_A = clip(act_A)
                            act_B = clip(act_B)

                        (alignment[A_counter,delta_t_id],
                        inf_imb[0,A_counter,delta_t_id],
                        inf_imb[1,A_counter,delta_t_id]) = mutual_k_aligmment_fn(act_A,act_B)
                            
                output_folder = makefolder(base=output_folder0,
                                        create_folder=True,
                                        k=k,
                                        Nbits=Nbits,
                                        seed=seed,
                                        )
                n_tokens = 1
                avg_tokens = 0
                np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_alignment.npy", arr=alignment)
                np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_AB.npy", arr=inf_imb[0])
                np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_BA.npy", arr=inf_imb[1])


if __name__ == "__main__":

    print(f'{sys.argv=}')
    
    parser = argparse.ArgumentParser(description="II")
    parser.add_argument("dbg",type=int)
    parser.add_argument("min_token_length",type=int)
    parser.add_argument("modelA",type=str)
    parser.add_argument("language",type=str)
    args = parser.parse_args()

    mask = 2 # greater than zero
    n_loaded_tokens = args.min_token_length - mask  # the mask excludes final points and quotes
    take_last = 1

    if args.dbg == 0:
        n_files = 500
        seed_list = np.arange(0,5,dtype=int)
        k_list = [10]
        Nbits_list = [1,0]
        avg_flags = [0]

    elif args.dbg == 1:
        n_files = 1
        seed_list = np.arange(0,1+1,dtype=int)
        k_list = [10]
        Nbits_list = [0,1]
        avg_flags = [0]

    if args.language == "english":
        input_path_A = input_paths[args.modelA][args.language]["spanish"]["source"][str(args.min_token_length)]
    if args.language == "spanish":
        input_path_A = input_paths[args.modelA]["english"][args.language]["target"][str(args.min_token_length)]
    if args.language == "italian":
        input_path_A = input_paths[args.modelA]["english"][args.language]["target"][str(args.min_token_length)]
    

    print("Input path A = ", input_path_A, flush=True)
    
    output_folder0 = makefolder(base=f'./results/santiago/',
                               create_folder=True,
                               take_last=take_last,
                               modelA=args.modelA,
                               modelB=args.modelA,
                               n_files=n_files,
                               min_token_length=args.min_token_length,
                               languageA=args.language,
                               mask=mask,
                               )
    main(
        input_path_A=input_path_A,
        n_loaded_tokens=n_loaded_tokens,
        mask=mask,
        n_files=n_files,
        seed_list=seed_list,
        k_list=k_list,
        output_folder0=output_folder0,
        Nbits_list=Nbits_list,
        take_last=take_last,
    )
