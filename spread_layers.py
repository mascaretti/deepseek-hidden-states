from safetensors import safe_open
from safetensors.torch import save_file
import pickle
from tqdm import tqdm
from pathlib import Path
import torch
from utils import list_folder

import sys

def collect_activations(input_path, n_tokens):
    chunks = list_folder(input_path, desc="chunk_")
  
    chunk = chunks[0]
    data = pickle.load(open(Path(input_path, chunk.name), 'rb'))
    sentence = data[0]
    hidden_states = sentence['meta_info']['hidden_states'][0]
    n_layers, _, emb_dim = hidden_states.shape
    print(f"n_layers: {n_layers}; embedding dimension: {emb_dim}.")

    all_hidden_states = [[] for _ in range(n_layers)]

    for chunk in tqdm(chunks):
        data = pickle.load(open(Path(input_path, chunk.name), 'rb'))
        for sentence in data:
            hidden_states = sentence['meta_info']['hidden_states'][0]
            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                all_hidden_states[layer_idx].append(layer_tensor[:, -n_tokens:, :])
                
    all_hidden_states = {str(layer_idx):torch.cat(layer_states, dim=0) for layer_idx, layer_states in enumerate(all_hidden_states)}
    return all_hidden_states


def main(input_path, output_path, n_tokens=10):
    hidden_states = collect_activations(input_path, n_tokens=n_tokens)
    for key in hidden_states.keys():
        print(key, hidden_states[key].shape)
    save_file(hidden_states, output_path)

if __name__ == "__main__":
    
    input_path = "/scratch/amascare/llama_flickr_cpt_1/"
    output_path = "/scratch/amascare/llama/llama_flickr_cpt_1.safetensors"
    n_tokens = 10

    print("Input path: ", input_path, flush=True)
    main(input_path=input_path, output_path=output_path, n_tokens=n_tokens)
