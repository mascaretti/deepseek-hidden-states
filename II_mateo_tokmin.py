from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance
import jax
# import jax.numpy as jnp
from utils import bf16_torch_to_jax, flatten_tokens_features, binarize
import pickle
from tqdm import tqdm
from pathlib import Path
import torch

import seaborn as sns
from typing import List

import typer
import re
import warnings

app=typer.Typer()
@app.command()
def get_II(input_path, 
        #  n_tokens_list:List[int]=[1],
         output_folder:str=None,
         do_tokens:bool=True,
         average_tokens:bool=False,
         n_files:int=30,
         diagonal_only:bool=True,
         rank:int=0,
         ):
    # collect all pickles
    files = sorted(Path(input_path).glob("*_10_*.pkl"))
    # shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(files)
    assert len(files) > n_files* (rank+1), f"Not enough files to process. {len(files)} files found, but {n_files*(rank+1)} files are required."
    files = files[n_files*rank:n_files*(rank+1)]
    # if not do_tokens and n_tokens_list != [1]:
    #     n_tokens_list = [1]
    #     warnings.warn(f"`n_tokens_list` is set to [1] because `do_tokens` is False. Overriding the input value {n_tokens_list}.")

    acts = []
    for file in tqdm(list(files), desc="Collect File"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            _a = {k: torch.tensor(v) for k, v in data[0].items()}
            # _b = {k: torch.tensor(v) for k, v in data[1].items()}
            acts.append((_a, None))

    all_activations_A = {f"layer_{i}": torch.cat([act[0][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}
    # all_activations_B = {f"layer_{i}": torch.cat([act[1][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}

    n_tokens_list = [i for i in range(all_activations_A["layer_0"].shape[1])]
    print(f"max token = {n_tokens_list[-1]}")
    sample_size = all_activations_A["layer_0"].shape[0]
    rows = list(all_activations_A.keys())
    rows=[rows[1], rows[len(rows)//2], rows[-1]]
    # cols = all_activations_B.keys()

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)
        

        
    for layer_A in tqdm(rows, desc="Computing layer_A", leave=False):
        if diagonal_only and layer_A not in rows:
            continue  # Skip non-diagonal elements if diagonal_only is True
        # print(f"shape of layer {layer_A} = {all_activations_A[layer_A].shape}")
        inf_imb_matrix = []
        reciprocal_inf_imb_matrix = []
        act_A = all_activations_A[layer_A][:, -1, :].clone()
        act_A = act_A[:, None, ...]
        act_A = bf16_torch_to_jax(act_A.contiguous())
        act_A = flatten_tokens_features(act_A)
        act_A = binarize(act_A)
        for layer_B in (pbar := tqdm(rows, desc="Computing layer_B", leave=False)):
            if diagonal_only and layer_A != layer_B:
                continue  # Skip non-diagonal elements if diagonal_only is True
            row = []
            reciprocal_row = []
            for n_tokens in tqdm(n_tokens_list, desc='Computing token'):            
                act_B = all_activations_A[layer_B][:,-n_tokens,...] if do_tokens else all_activations_A[layer_B]
                act_B = act_B[:,None,...]
                act_B = act_B.mean(dim=1, keepdim=True) if average_tokens else act_B
                # act_B = act_B.view(act_B.shape[0], act_B.shape[1], -1) if do_tokens and not average_tokens else act_B[:,None,...].view(act_B.shape[0], act_B.shape[1], -1)
                act_B = bf16_torch_to_jax(act_B.contiguous())
                act_B = flatten_tokens_features(act_B)
                act_B = binarize(act_B)

                inf_imb, reciprocal_inf_imb = information_imbalance_fn(act_A, act_B)
                row.append(inf_imb.item())
                reciprocal_row.append(reciprocal_inf_imb.item())
                pbar.set_description(f"Computing layer_B (II(A-->B) {inf_imb:.4f} ; II(B-->A) {reciprocal_inf_imb:.4f})")
            inf_imb_matrix.append(row)
            reciprocal_inf_imb_matrix.append(reciprocal_row)

        inf_imb = np.array(inf_imb_matrix)
        # min_IIs.append((n_tokens, min(inf_imb)))
        reciprocal_inf_imb = np.array(reciprocal_inf_imb_matrix)
        np.save(file=f"{output_folder}/tt_{layer_A}_AB_{rank}.npy", arr=inf_imb)
        np.save(file=f"{output_folder}/tt_{layer_A}_BA_{rank}.npy", arr=reciprocal_inf_imb)
    # np.save(file=f"{output_folder}/min_IIs.npy", arr=np.array(min_IIs))

if __name__ == "__main__":
    app()