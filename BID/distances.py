import sys,os
os.environ["JAX_ENABLE_X64"] = "True"
sys.path.append('../')
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
                   makefolder,
                   )
import pickle
from tqdm import tqdm
import torch
from datapaths import *
import argparse
from Spins.spins import *
from time import time

depths = {"deepseek":61,
          "llama":32}
emb_dims = {"deepseek":7168,
          "llama":4096}

def collect_data(input_path, 
                 filter_layer=None, 
                 n_loaded_tokens=40, 
                 n_files=10,
                 ):
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
        # print(layer, "activations shape= ", all_hidden_states[layer].shape, flush=True)

    return all_hidden_states

def main(
         model,
         input_path, 
         n_loaded_tokens, 
         n_files,
         n_tokens_list,
         output_folder0,
         avg_flags,
         Nbits_list,
         seed_list,
         n_subsamples,
         ):
    n_layers = depths[model]
    layers = np.arange(1, n_layers+1, dtype=int)

    for Nbits_id, Nbits in enumerate(Nbits_list):
        print(f'{Nbits=}')

        for avg_tokens in avg_flags:
            print(f'{avg_tokens=}')

            for A_counter,layer_idx in enumerate(layers):
                print(f"{layer_idx=}")
                
                activations_A = collect_data(
                                            input_path=input_path, 
                                            filter_layer=layer_idx,
                                            n_loaded_tokens=n_loaded_tokens, 
                                            n_files=n_files,
                                            )[f"layer_{layer_idx}"]
                
                for jack_seed_id,jack_seed in enumerate(seed_list):
                    print(f"{jack_seed=}")
                    
                    for n_tokens_id, n_tokens in enumerate(n_tokens_list):
                        print(f'{n_tokens=}')
                        
                        H = Hamming(crossed_distances=1)
                        histfolder = makefolder(base=output_folder0 + '/hist/',
                                            n_tokens=n_tokens,
                                            layer_idx=layer_idx,
                                            seed=jack_seed,
                                            )
                        
                        act_A = bf16_torch_to_jax(activations_A[:,-n_tokens:,:])
                        total_sample_size = act_A.shape[0]

                        jack_key = jax.random.key(jack_seed)
                        jack_indices = jax.random.choice(key=jack_key,
                                                        a=total_sample_size,
                                                        shape=(n_subsamples,),
                                                        replace=False)
                        act_A = jax.numpy.take(act_A, jack_indices, axis=0)

                        if Nbits == 1:
                            if avg_tokens == 1:
                                act_A = act_A.mean(axis=1,keepdims=True)
                            act_A = binarize(act_A)
                        act_A = flatten_tokens_features(act_A)
                        H.coordinates = act_A
                        H.compute_distances()
                        H.D_histogram(compute_flag=1,          # if 0 the histograms are loaded instead of computed
                                    save=True,                 # we compute the histograms once and save time in the future
                                    resultsfolder=histfolder,  # folder where the histograms are saved
                                    ) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="II")
    parser.add_argument("dbg",type=int)
    parser.add_argument("min_token_length",type=int)
    parser.add_argument("modelA",type=str)
    parser.add_argument("language",type=str) # "source or target"
    args = parser.parse_args()

    n_loaded_tokens = args.min_token_length
    compute_BID = False

    if args.dbg == 0:
        n_files = 500
        n_subsamples = 2500
        seed_list = [0,1,2,3,4,5,6,7,8,9]
        if args.min_token_length == 40:
            n_tokens_list = np.array([15])
        elif args.min_token_length == 100:
            n_tokens_list = np.array([50])
        Nbits_list = [1]
        avg_flags = [0]

    elif args.dbg == 1:
        n_files = 2
        n_subsamples = 10
        seed_list = [0,1,2]
        n_tokens_list = np.array([10])
        Nbits_list = [1]
        avg_flags = [0]

    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    print(f'{n_tokens_list=}')

    
    languageA = 'english'
    languageB = 'spanish'
    input_path = input_paths[args.modelA][languageA][languageB][args.language][str(args.min_token_length)]
    print("Input path = ", input_path, flush=True)
    
    output_folder0 = makefolder(base=f'./results/santiago/',
                               create_folder=True,
                               modelA=args.modelA,
                               modelB=args.modelA,
                               n_files=n_files,
                               min_token_length=args.min_token_length,
                               languageA=languageA,
                               languageB=languageB,
                               language=args.language,
                               )
    start = time()
    main(
        model=args.modelA,
        input_path=input_path,
        n_loaded_tokens=n_loaded_tokens,
        n_files=n_files,
        n_tokens_list=n_tokens_list,
        output_folder0=output_folder0,
        avg_flags=avg_flags,
        Nbits_list=Nbits_list,
        seed_list=seed_list,
        n_subsamples=n_subsamples,
    )
    print(f"This took  {(time()-start)/60.} m ", flush=True)
