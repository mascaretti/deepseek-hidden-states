import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance
import jax
import jax.numpy as jnp
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder
import pickle
from tqdm import tqdm
from natsort import natsorted

import sys


def collect_layer_activations(input_path, n_layer, n_tokens, max_batch=None):
    """
    The last n_tokens are collected.
    """
    tensors = []

    folders = list_folder(input_path, desc="chunk_")

    for folder in tqdm(folders, desc="Loading files", leave=False):
        #* limit the max batch
        if max_batch and len(tensors) >= max_batch: break

        chunk_folder = input_path + "/" + folder.name
        natural_sorted_chunks = natsorted([file.stem for file in list_folder(chunk_folder, desc="input_")])
        chunks = filter(lambda k: k.split("_")[-1] == str(n_layer), natural_sorted_chunks)

        for chunk in chunks:
            chunk_path = chunk_folder + f"/{chunk}.pkl"

            with open(chunk_path, "rb") as f:
                tensor = pickle.load(f)

            tensor = tensor[-n_tokens:, :]
            tensor = bf16_torch_to_jax(tensor)
            tensors.append(tensor)

    tensors = jnp.array(tensors)
    return flatten_tokens_features(tensors)


def main(input_path_A, input_path_B, row_number, n_tokens, max_batch=4000):
    information_imbalance_fn = build_information_imbalance(L2_distance)

    activations_A = collect_layer_activations(input_path_A, n_layer=row_number, n_tokens=n_tokens, max_batch=max_batch)

    row = []
    for j in (pbar := tqdm(range(1, 62, 3), desc="Computing column")):
        activations_B = collect_layer_activations(input_path_B, n_layer=j, n_tokens=n_tokens, max_batch=max_batch)
        inf_imb = information_imbalance_fn(activations_A, activations_B)
        row.append(inf_imb.item())

        pbar.set_description(f"Computing column (II {inf_imb:.4f})")

    matrix = np.array(row)
    np.savetxt(f"row_{row_number}.txt", matrix, fmt="%.5f")

if __name__ == "__main__":
    input_path_A = "/home/laios/activations/flickr/deepseek_captions1_spread"
    input_path_B = "/home/laios/activations/flickr/deepseek_captions3_spread"

    row_number = int(sys.argv[1])
    n_tokens = 1

    print("Input path A = ", input_path_A, flush=True)
    print("Input path B = ", input_path_B, flush=True)
    main(input_path_A=input_path_A,
         input_path_B=input_path_B,
         row_number=row_number,
         n_tokens=n_tokens)