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
         input_path_A, 
         n_loaded_tokens, 
         n_files,
         n_tokens_list,
         output_folder0,
         avg_flags,
         Nbits_list,
         ):
    n_layers = depths[model]
    # layers = np.arange(1, n_layers, dtype=int)
    layers = np.array([61])

    for n_tokens_id, n_tokens in enumerate(n_tokens_list):
        print(f'{n_tokens=}')
        n_features = n_tokens * emb_dims[model]
        for Nbits_id, Nbits in enumerate(Nbits_list):
            print(f'{Nbits=}')

            for avg_tokens in avg_flags:
                print(f'{avg_tokens=}')

                for A_counter,layer_idx in enumerate(layers):
                    print(f"{layer_idx=}")
                    
                    H = Hamming(crossed_distances=1)
                    histfolder = makefolder(base=output_folder0 + '/hist/',
                                            n_tokens=n_tokens,
                                            layer_idx=layer_idx)
                    try:
                        H.D_histogram(compute_flag=0,
                                    resultsfolder=histfolder,
                                    )
                    except:
                        activations_A = collect_data(
                                                input_path=input_path_A, 
                                                filter_layer=layer_idx,
                                                n_loaded_tokens=n_loaded_tokens, 
                                                n_files=n_files,
                                                )[f"layer_{layer_idx}"]
                        act_A = bf16_torch_to_jax(activations_A[:,-n_tokens:,:])
                        if Nbits == 1:
                            if avg_tokens == 1:
                                act_A = act_A.mean(axis=1,keepdims=True)
                            act_A = binarize(act_A)
                        act_A = flatten_tokens_features(act_A)
                        assert n_features == act_A.shape[1]
                        H.coordinates = act_A
                        H.compute_distances()
                        H.D_histogram(compute_flag=1,            # if 0 the histograms are loaded instead of computed
                                    save=True,                 # we compute the histograms once and save time in the future
                                    resultsfolder=histfolder,  # folder where the histograms are saved
                                    )
                    if compute_BID:
                        # PARAMETER DEFINITIONS FOR OPTIMIZATION
                        alphamin = 0        
                        alphamax_list = np.arange(.15,.45+eps,.05)        
                        Nsteps = int(1e5)   
                        export_results = 1  
                        export_logKLs = 1   
                        optfolder0 = makefolder(base=output_folder0 + '/opt/',
                                            n_tokens=n_tokens,
                                            layer_idx=layer_idx)
                        steps_initial = jnp.array([5E-1,1E-1,5E-2,1E-2],dtype=jnp.double)
                        for alphamax_id,alphamax in enumerate(alphamax_list):
                            B = BID(
                                H=H,
                                alphamin=alphamin,
                                alphamax=alphamax,
                                Nsteps=Nsteps,
                                export_results=export_results,
                                export_logKLs=export_logKLs,
                                optfolder0=optfolder0,
                                L=n_features,
                                steps_initial=steps_initial,
                            )
                            B.computeBID() 
        
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="II")
    parser.add_argument("dbg",type=int)
    parser.add_argument("min_token_length",type=int)
    parser.add_argument("modelA",type=str)
    parser.add_argument("languageA",type=str)
    args = parser.parse_args()

    n_loaded_tokens = args.min_token_length
    compute_BID = False

    if args.dbg == 0:
        n_files = 500
        if args.min_token_length == 40:
            n_tokens_list = np.array([10])
        elif args.min_token_length == 100:
            n_tokens_list = np.array([50])
        Nbits_list = [1]
        avg_flags = [0]

    elif args.dbg == 1:
        n_files = 2
        n_tokens_list = np.array([2])
        Nbits_list = [1]
        avg_flags = [0]

    print(f'{Nbits_list=}')
    print(f'{avg_flags=}')
    

    input_path_A = input_paths[args.modelA][args.languageA][str(args.min_token_length)]
    print("Input path A = ", input_path_A, flush=True)
    
    output_folder0 = makefolder(base=f'./results/santiago/',
                               create_folder=True,
                               modelA=args.modelA,
                               modelB=args.modelA,
                               n_files=n_files,
                               min_token_length=args.min_token_length,
                               languageA=args.languageA,
                               )
    main(
        model=args.modelA,
        input_path_A=input_path_A,
        n_loaded_tokens=n_loaded_tokens,
        n_files=n_files,
        n_tokens_list=n_tokens_list,
        output_folder0=output_folder0,
        avg_flags=avg_flags,
        Nbits_list=Nbits_list,
    )
