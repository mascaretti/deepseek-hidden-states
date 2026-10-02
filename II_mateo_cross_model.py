from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms

from safetensors import safe_open
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance
import jax
import jax.numpy as jnp
from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, binarize, makefolder
import pickle
from tqdm import tqdm
from pathlib import Path
import torch

import sys
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import seaborn as sns
from typing import List

import typer
import re
import warnings
from functools import partial
from itertools import combinations

app=typer.Typer()

class ImageGPTProcessor(torch.nn.Module):
    def __init__(self, model, processor, device):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = device
        self.n_layer = model.config.n_layer

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return {f"layer_{i}": hidden_state for i, hidden_state in enumerate(outputs.hidden_states)}

@app.command()
def main(input_path1:str,
        input_path2:str, 
         n_tokens_list:List[int]=[1],
         output_folder:str=None,
         do_tokens:bool=True,
         average_tokens:bool=False,
         n_files:int=30,
         diagonal_only:bool=False,  # New option to compute only the diagonal
         rank:int=0,
         ):
    # collect all pickles
    files = sorted(Path(input_path1).glob("*0_*.pkl"))
    # shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(files)
    assert len(files) > n_files* (rank+1), f"Not enough files to process. {len(files)} files found, but {n_files*(rank+1)} files are required."
    files = files[n_files*rank:n_files*(rank+1)]
    if not do_tokens and n_tokens_list != [1]:
        n_tokens_list = [1]
        warnings.warn(f"`n_tokens_list` is set to [1] because `do_tokens` is False. Overriding the input value {n_tokens_list}.")

    acts = []
    for file in tqdm(list(files), desc="Collect File"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            _a = {k: torch.tensor(v) for k, v in data[0].items()}
            _b = {}
            acts.append((_a, _b))

    all_activations_A = {f"layer_{i}": torch.cat([act[0][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}
    files2 = sorted(Path(input_path2).glob("*0_*.pkl"))
    files2 = files2[n_files*rank:n_files*(rank+1)]
    acts = []
    for file in tqdm(list(files2), desc="Collect File"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            _a = {k: torch.tensor(v) for k, v in data[0].items()}
            _b = {}
            acts.append((_a, _b))
    all_activations_B = {f"layer_{i}": torch.cat([act[0][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}
    
    sample_size = all_activations_B["layer_0"].shape[0]
    rows = all_activations_A.keys()
    cols = all_activations_B.keys()
    if diagonal_only and len(rows) != len(cols):
        min_len = min(len(rows), len(cols))
        rows = {f"layer_{i}": all_activations_A[f"layer_{i}"] for i in range(min_len)}
        cols = {f"layer_{i}": all_activations_B[f"layer_{i}"] for i in range(min_len)}


    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)
    for n_tokens in tqdm(n_tokens_list, desc='Computing token'):
        
        inf_imb_matrix = []
        reciprocal_inf_imb_matrix = []
        
        for layer_A in tqdm(rows, desc="Computing layer_A", leave=False):
            if diagonal_only and layer_A not in cols:
                continue  # Skip non-diagonal elements if diagonal_only is True

            activations_A = all_activations_A[layer_A]
            # tqdm.write(f"number of available tokens:{activations_A.shape[1]}")

            row = []
            reciprocal_row = []
            for layer_B in (pbar := tqdm(cols, desc="Computing layer_B", leave=False)):
                if diagonal_only and layer_A != layer_B:
                    continue  # Skip non-diagonal elements if diagonal_only is True

                act_A = activations_A[:,-n_tokens:,...] if do_tokens else activations_A
                act_A = act_A.mean(dim=1, keepdim=True) if average_tokens else act_A
                act_A = act_A.view(act_A.shape[0], act_A.shape[1], -1) if do_tokens and not average_tokens else act_A[:,None,...].view(act_A.shape[0], act_A.shape[1], -1)
                act_A = bf16_torch_to_jax(act_A.contiguous())
                act_A = flatten_tokens_features(act_A)
                act_A = binarize(act_A)
                act_B = all_activations_B[layer_B][:,-n_tokens:,...] if do_tokens else all_activations_B[layer_B]
                act_B = act_B.mean(dim=1, keepdim=True) if average_tokens else act_B
                act_B = act_B.view(act_B.shape[0], act_B.shape[1], -1) if do_tokens and not average_tokens else act_B[:,None,...].view(act_B.shape[0], act_B.shape[1], -1)
                act_B = bf16_torch_to_jax(act_B.contiguous())
                act_B = flatten_tokens_features(act_B)
                act_B = binarize(act_B)

                # input("ATTENTION: ")
                inf_imb, reciprocal_inf_imb = information_imbalance_fn(act_A, act_B)
                row.append(inf_imb.item())
                reciprocal_row.append(reciprocal_inf_imb.item())
                pbar.set_description(f"Computing layer_B (II(A-->B) {inf_imb:.4f} ; II(B-->A) {reciprocal_inf_imb:.4f})")
            inf_imb_matrix.append(row)
            reciprocal_inf_imb_matrix.append(reciprocal_row)

        inf_imb = np.array(inf_imb_matrix)
        reciprocal_inf_imb = np.array(reciprocal_inf_imb_matrix)
        np.save(file=f"{output_folder}/cm_n_tokens_{n_tokens}_AB.npy", arr=inf_imb)
        np.save(file=f"{output_folder}/cm_n_tokens_{n_tokens}_BA.npy", arr=reciprocal_inf_imb)

if __name__ == "__main__":
    app()