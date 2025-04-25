import sys,os
sys.path.append('../../')
# dbg = int(sys.argv[1])
# if dbg:
#     os.environ["JAX_PLATFORMS"] = "cpu"
from collections import defaultdict
import numpy as np

import jax
from utils import (bf16_torch_to_jax, 
                   flatten_tokens_features, 
                   list_folder, 
                   binarize, 
                   clip,
                   makefolder,
                   )
import pickle
from tqdm import tqdm
import torch
from datapaths import *
import argparse

def collect_data(input_path, 
                 filter_layer=None, 
                 n_loaded_tokens=40, 
                 n_files=10,
                 mask=None,
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
         avg_flags,
         Nbits_list,
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

    for n_tokens_id, n_tokens in enumerate(n_tokens_list):
        print(f'{n_tokens=}')
        for k_id, k in enumerate(k_list):
            print(f'{k=}')
            for Nbits_id, Nbits in enumerate(Nbits_list):
                print(f'{Nbits=}')

                distance_fn = hamming_distance if Nbits else normalized_L2_distance 
                mutual_k_aligmment_fn = build_mutual_k_NN_alignment(key=subkey, 
                                                                    sample_size=sample_size, 
                                                                    distance_fn=distance_fn,
                                                                    k=k)
                
                for avg_tokens in avg_flags:
                    print(f'{avg_tokens=}')
                    alignment = np.zeros(shape=(len(rows),
                                                len(cols))
                                                )
                    inf_imb = np.zeros(shape=(2,len(rows),len(cols)))

                    for A_counter,layer_A in enumerate(rows):
                        activations_A = all_activations_A[layer_A]

                        for B_counter,layer_B in enumerate(cols):
                            activations_B = all_activations_B[layer_B]

                            if diagonal_constraint == 1 and layer_B != layer_A:
                                continue

                            act_A = bf16_torch_to_jax(activations_A[:,-n_tokens:,:])
                            act_B = bf16_torch_to_jax(activations_B[:,-n_tokens:,:])

                            if Nbits == 1:
                                if avg_tokens == 1:
                                    act_A = act_A.mean(axis=1,keepdims=True)
                                    act_B = act_B.mean(axis=1,keepdims=True)
                                act_A = binarize(act_A)
                                act_B = binarize(act_B)

                            elif Nbits == 0:
                                act_A = clip(act_A)
                                act_B = clip(act_B)
                                if avg_tokens == 1:
                                    act_A = act_A.mean(axis=1,keepdims=True)
                                    act_B = act_B.mean(axis=1,keepdims=True)

                            act_A = flatten_tokens_features(act_A)
                            act_B = flatten_tokens_features(act_B)

                            (alignment[A_counter,B_counter],
                            inf_imb[0,A_counter,B_counter],
                            inf_imb[1,A_counter,B_counter]) = mutual_k_aligmment_fn(act_A,act_B)
                            
                    output_folder = makefolder(base=output_folder0,
                                            create_folder=True,
                                            k=k,
                                            Nbits=Nbits,
                                            )
                    np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_alignment.npy", arr=alignment)
                    np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_AB.npy", arr=inf_imb[0])
                    np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_BA.npy", arr=inf_imb[1])

        
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="II")
    parser.add_argument("dbg",type=int)
    parser.add_argument("min_token_length",type=int)
    parser.add_argument("modelA",type=str)
    # parser.add_argument("modelB",type=str)
    parser.add_argument("languageA",type=str)
    parser.add_argument("languageB",type=str)
    args = parser.parse_args()

    mask = 2 # greater than zero
    n_loaded_tokens = args.min_token_length - mask  # the mask excludes final points and quotes

    if args.dbg == 0:
        n_files = 500
        if args.min_token_length == 40:
            n_tokens_list = np.array([n_loaded_tokens,30,20,10,5,1,15,25,35])
        elif args.min_token_length == 100:
            n_tokens_list = np.array([1,5,10,20,30,40,50,60,70,80,90,n_loaded_tokens])
        k_list = [10]
        Nbits_list = [1,0]
        avg_flags = [0,1]
        diagonal_constraint = 1

    elif args.dbg == 1:
        n_files = 1
        n_tokens_list = np.array([2])
        k_list = [10]
        Nbits_list = [0,1]
        avg_flags = [0,1]
        diagonal_constraint = 1

    print(f'{k_list=}')
    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{diagonal_constraint=}')
    

    input_path_A = input_paths[args.modelA][args.languageA][str(args.min_token_length)]
    input_path_B = input_paths[args.modelA][args.languageB][str(args.min_token_length)]
    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    output_folder0 = makefolder(base=f'./results/santiago/',
                               create_folder=True,
                               modelA=args.modelA,
                               modelB=args.modelA,
                               n_files=n_files,
                               min_token_length=args.min_token_length,
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
        avg_flags=avg_flags,
        Nbits_list=Nbits_list,
        diagonal_constraint=diagonal_constraint,
    )
