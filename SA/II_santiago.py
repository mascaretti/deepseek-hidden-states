import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from datetime import datetime
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

import sys,os
sys.path.append('../')
dbg = int(sys.argv[1])
if dbg:
    os.environ["JAX_PLATFORMS"] = "cpu"
from collections import defaultdict
import numpy as np

from information_imbalance_module import (normalized_L2_distance, 
                                          hamming_distance, 
                                          build_mutual_k_NN_alignment, 
                                          build_information_imbalance
                                          )
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
from time import time



def reshuffle_batch_axis(act, key):
    """
    Reshuffles the activations along the batch axis.

    Args:
        act (jnp.ndarray): Activation matrix of shape (batch_size, ...).
        key (jax.random.PRNGKey): A PRNG key for randomness.

    Returns:
        jnp.ndarray: Shuffled activations.
    """
    batch_size = act.shape[0]
    perm = jax.random.permutation(key, batch_size)
    return act[perm]


def reduce_list_half_preserve_extremes(lst):
    """
    Reduces the input list to approximately half its original size,
    preserving the first and last elements, and sampling uniformly
    from the intermediate elements.
    
    Parameters:
    -----------
    lst : list
        The input list to reduce.
    
    Returns:
    --------
    list
        A reduced list with approximately half the points,
        preserving the first and last elements.
    """
    N = len(lst)
    if N <= 2:
        return lst.copy()
    
    half_N = max(N // 2, 2)
    num_points_to_sample = half_N - 2
    
    new_lst = [lst[0]]
    
    if num_points_to_sample > 0:
        # Calculate the indices to sample from intermediates
        step = (N - 2) / (num_points_to_sample + 1)
        intermediate_indices = [int(round(1 + i * step)) for i in range(num_points_to_sample)]
        new_lst.extend([lst[i] for i in intermediate_indices])
    
    new_lst.append(lst[-1])
    return new_lst

def reshuffle_batch_and_tokens(act, key):
    """
    Reshuffles the activations along the batch and token axes independently.

    Args:
        act (jnp.ndarray): Activation tensor of shape (batch_size, seq_len, ...).
        key (jax.random.PRNGKey): A PRNG key for randomness.

    Returns:
        jnp.ndarray: Shuffled activations.
    """
    key_batch, key_token = jax.random.split(key)
    
    # Shuffle batch
    batch_size = act.shape[0]
    batch_perm = jax.random.permutation(key_batch, batch_size)
    act = act[batch_perm]
    
    # Shuffle tokens
    seq_len = act.shape[1]
    token_perm = jax.random.permutation(key_token, seq_len)
    act = act[:, token_perm]
    
    return act



def collect_data(input_path, 
                 filter_layer=None, 
                 n_loaded_tokens=40, 
                 n_files=10,
                 mask=None,
                #  seed=42,
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
         seed_list,
         n_subsamples,
         batch_shuffle,
         ):
    start_time = time()
    
    all_activations_A = collect_data(input_path_A,
                                     n_loaded_tokens=n_loaded_tokens, 
                                     n_files=n_files,
                                     mask=mask,
                                     )
    all_activations_B = collect_data(input_path_B,
                                     n_loaded_tokens=n_loaded_tokens, 
                                     n_files=n_files,
                                     mask=mask,
                                     )
    

    total_sample_size = all_activations_B["layer_0"].shape[0]

    #dbg:
    print(f'{total_sample_size=}', flush=True)
    print(f'{n_subsamples=}', flush=True)

    n_layers_A = len(list(all_activations_A.keys()))
    n_layers_B = len(list(all_activations_B.keys()))

    key_distances = jax.random.key(42)
    key_distances, subkey_distances = jax.random.split(key_distances)

    A_batch_key = jax.random.key(101)
    B_batch_key = jax.random.key(102)

    rows = [f"layer_{i}" for i in range(1, n_layers_A)]
    cols = [f"layer_{i}" for i in range(1, n_layers_B)]

    if 1:
        cols = reduce_list_half_preserve_extremes(cols)
        rows = reduce_list_half_preserve_extremes(rows)

    for jack_seed_id,jack_seed in enumerate(seed_list):
        jack_key = jax.random.key(jack_seed)
        jack_indices = jax.random.choice(key=jack_key,a=total_sample_size,shape=(n_subsamples,),replace=False)
        print(f"{jack_seed=}")

        for n_tokens_id, n_tokens in enumerate(n_tokens_list):
            print(f'{n_tokens=}')
            for k_id, k in enumerate(k_list):
                print(f'{k=}')
                for Nbits_id, Nbits in enumerate(Nbits_list):
                    print(f'{Nbits=}')

                    distance_fn = hamming_distance if Nbits else normalized_L2_distance 
                    mutual_k_aligmment_fn = build_mutual_k_NN_alignment(key=subkey_distances, 
                                                                        sample_size=n_subsamples, 
                                                                        distance_fn=distance_fn,
                                                                        k=k)
                    # information_imbalance_fn = build_information_imbalance(key=subkey_distances, 
                    #                                                        sample_size=n_subsamples, 
                    #                                                        distance_fn=distance_fn, 
                    #                                                        k=k)
                    
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
                                
                                act_A = jax.numpy.take(act_A, jack_indices, axis=0)
                                act_B = jax.numpy.take(act_B, jack_indices, axis=0)

                                if batch_shuffle == 1:
                                    print(f'batch_shuffling!!! B')
                                    act_B = reshuffle_batch_axis(act_B, B_batch_key)

                                if Nbits == 1:
                                    if avg_tokens == 1:
                                        act_A = act_A.mean(axis=1,keepdims=True)
                                        act_B = act_B.mean(axis=1,keepdims=True)
                                    act_A = binarize(act_A)
                                    act_B = binarize(act_B)

                                elif Nbits == 0:
                                    act_A = clip(act_A).astype(jnp.double)# promote them to double to break massive degeneracies due to small precision
                                    act_B = clip(act_B).astype(jnp.double)
                                    if avg_tokens == 1:
                                        act_A = act_A.mean(axis=1,keepdims=True)
                                        act_B = act_B.mean(axis=1,keepdims=True)

                                act_A = flatten_tokens_features(act_A)
                                act_B = flatten_tokens_features(act_B)

                                (alignment[A_counter,B_counter],
                                inf_imb[0,A_counter,B_counter],
                                inf_imb[1,A_counter,B_counter]) = mutual_k_aligmment_fn(act_A,act_B)
                                # (inf_imb[0,A_counter,B_counter],
                                # inf_imb[1,A_counter,B_counter]) = information_imbalance_fn(act_A,act_B)
                                print(f'{layer_A=};{layer_B=};{inf_imb[0,A_counter,B_counter],inf_imb[1,A_counter,B_counter]}')

                                
                        output_folder = makefolder(base=output_folder0,
                                                create_folder=True,
                                                k=k,
                                                Nbits=Nbits,
                                                seed=jack_seed,
                                                )
                        np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_alignment.npy", arr=alignment)
                        np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_AB.npy", arr=inf_imb[0])
                        np.save(file=f"{output_folder}/n_tokens_{n_tokens}_avg_{avg_tokens}_BA.npy", arr=inf_imb[1])
    print(f'this took {(time()-start_time)/60.} m ')

        
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="II")
    parser.add_argument("dbg",type=int)
    parser.add_argument("min_token_length",type=int)
    parser.add_argument("modelA",type=str) # llama or deepseek
    parser.add_argument("modelB",type=str)
    parser.add_argument("aux_A",type=str) # source or target
    parser.add_argument("aux_B",type=str)
    parser.add_argument("languageA",type=str)
    parser.add_argument("languageB",type=str)
    args = parser.parse_args()

    batch_size = 10
    mask = 2 # greater than zero
    n_loaded_tokens = args.min_token_length - mask  # the mask excludes final points and quotes
    seed_list = [0]

    if args.dbg == 0:
        batch_shuffle = 0
        n_files = 500
        n_subsamples = 2500
        if args.min_token_length == 40:
            n_tokens_list = np.array([n_loaded_tokens,30,20,10,5,1,15,25,35])
        elif args.min_token_length == 100:
            n_tokens_list = np.array([1,5,10,20,30,40,50,60,70,80,90,n_loaded_tokens])
        k_list = [10]
        Nbits_list = [0,1]
        avg_flags = [0,1]
        diagonal_constraint = 0

    elif args.dbg == 1:
        batch_shuffle = 0
        n_files = 2
        n_subsamples = 10
        n_tokens_list = np.array([1])
        k_list = [10]
        Nbits_list = [0,1]
        avg_flags = [0,1]
        diagonal_constraint = 0
    
    if args.dbg == 2: 
        batch_shuffle = 0
        n_files = 200
        n_subsamples = n_files * batch_size
        n_tokens_list = np.array([20])
        k_list = [10] 
        Nbits_list = [1,0]
        avg_flags = [0]
        diagonal_constraint = 1

    print(f'{k_list=}')
    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{diagonal_constraint=}')

    input_path_A = input_paths[args.modelA][args.languageA][args.languageB][args.aux_A]['40']
    input_path_B = input_paths[args.modelB][args.languageA][args.languageB][args.aux_B]['40']
    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    
    output_folder0 = makefolder(base=f'./results/santiago/',
                               create_folder=True,
                               batch_shuffle=batch_shuffle,
                               modelA=args.modelA,
                               modelB=args.modelB,
                               aux_A=args.aux_A,
                               aux_B=args.aux_B,
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
        seed_list=seed_list,
        n_subsamples=n_subsamples,
        batch_shuffle=batch_shuffle,
    )
