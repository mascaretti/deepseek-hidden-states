#!/usr/bin/env python
"""
Unified asymmetry computation script
Handles all comparison types with minimal changes from original code
"""

from collections import defaultdict
import numpy as np
from safetensors.numpy import save_file
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance, build_information_matrix
import jax
import jax.numpy as jnp
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, binarize
import pickle
from tqdm import tqdm
from pathlib import Path
import torch
import io
import argparse
from jax import random as random

# CHANGE: Made CPU_Unpickler available globally since it's used in multiple collection functions
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def collect_deepseek_data(input_path, n_loaded_tokens=9, subset_size=5000):
    """Collect DeepSeek text data - unchanged from original"""
    files = list_folder(input_path, desc="chunk_")
    all_hidden_states = defaultdict(list)

    file_counter = 0
    for file in tqdm(files, desc="Collect DeepSeek File", leave=False):
        if file_counter >= subset_size:
            break
        # CHANGE: Added context manager for file handling (critical fix)
        with open(input_path + "/" + file.name, 'rb') as f:
            data = pickle.load(f)
        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            # CHANGE: Warning instead of assert (critical fix)
            if hidden_states.shape[1] < n_loaded_tokens:
                print(f"Warning: Skipping sentence with only {hidden_states.shape[1]} tokens")
                continue
            hidden_states = hidden_states[:, -n_loaded_tokens:]
            file_counter += 1

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)[:subset_size]
    
    print(f"DeepSeek data shape: {all_hidden_states['layer_0'].shape}")
    return all_hidden_states


def collect_image_model_data(input_path, n_files, last_tokens=200):
    """Collect DinoV2/ImageGPT data - unchanged from original"""
    files = list_folder(input_path, desc="chunk_")[:n_files]
    all_hidden_states = defaultdict(list)
    for file in tqdm(files, desc="Collect image model data"):
        # CHANGE: Added context manager for file handling (critical fix)
        with open(Path(input_path, file.name), 'rb') as f:
            batch_activations = CPU_Unpickler(f).load()
        for layer_key in batch_activations.keys():
            all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.cat(tensors, dim=0)
    print(f"Image model data shape: {all_hidden_states['layer_0'].shape}")
    return all_hidden_states


def collect_clip_image_data(input_path, n_files, last_tokens=200):
    """Collect CLIP image data - uses 'image-chunk_' prefix"""
    files = list_folder(input_path, desc="image-chunk_")[:n_files]
    all_hidden_states = defaultdict(list)
    for file in tqdm(files, desc="Collect CLIP image data"):
        # CHANGE: Added context manager for file handling (critical fix)
        with open(Path(input_path, file.name), 'rb') as f:
            batch_activations = CPU_Unpickler(f).load()
        for layer_key in batch_activations.keys():
            all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.cat(tensors, dim=0)
    print(f"CLIP image data shape: {all_hidden_states['layer_0'].shape}")
    return all_hidden_states


def collect_clip_text_data(input_path, n_files, last_tokens=30):
    """
    Collect CLIP text data
    CHANGE: New function for CLIP text embeddings
    Assumes CLIP text has max 30 tokens based on extract_hidden_clip.py
    """
    files = list_folder(input_path, desc="text-chunk_")[:n_files]
    all_hidden_states = defaultdict(list)
    for file in tqdm(files, desc="Collect CLIP text data"):
        with open(Path(input_path, file.name), 'rb') as f:
            batch_activations = CPU_Unpickler(f).load()
        for layer_key in batch_activations.keys():
            # CLIP text might have fewer tokens, take what's available
            all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.cat(tensors, dim=0)
    print(f"CLIP text data shape: {all_hidden_states['layer_0'].shape}")
    return all_hidden_states


def compute_inf_imb_matrix(
        all_activations_A, 
        all_activations_B, 
        output_folder,
        layer_A_id=52,  # CHANGE: Made configurable, default DeepSeek layer
        comparison_type="text_image"  # CHANGE: Added to handle different preprocessing
):
    """
    Compute information imbalance matrix
    CHANGE: Added layer_A_id parameter and comparison_type for flexibility
    """
    key = jax.random.key(42)
    print(f"Initial key: {key}")
    
    n_layers_B = len(list(all_activations_B.keys()))
    cols = [f"layer_{i}" for i in range(1, n_layers_B)]

    sample_size = all_activations_A["layer_1"].shape[0]
    print("Sample size: ", sample_size)
    sample_size_B = all_activations_B["layer_1"].shape[0]
    assert sample_size == sample_size_B
        
    key, subkey = jax.random.split(key)
    information_matrix_fn = build_information_matrix(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)

    layer_A = f"layer_{layer_A_id}"
    # CHANGE: Added validation (critical fix)
    if layer_A not in all_activations_A:
        raise ValueError(f"Layer {layer_A} not found in activations A. Available: {list(all_activations_A.keys())}")
    
    print(f"Layer A: {layer_A}")
    activations_A = all_activations_A[layer_A]
    print(f"dtype of activations_A is: {activations_A.dtype}")
    print(f"*** COMPARISON TYPE: {comparison_type} ***")
    if comparison_type == "text_image" or comparison_type == "text_text":
        print("Text comparison...")
        if activations_A.dtype == torch.bfloat16:
            print("Converting to Jax...")
            act_A = bf16_torch_to_jax(activations_A[:, -9:, :])
        else:
            print("Creating array...")
            act_A = jnp.array(activations_A[:, -9:, :])
    else:
        print("Image comparison...")
        act_A = jnp.array(activations_A)
    print(f"dtype of act_A: {act_A.dtype}")

    act_A = flatten_tokens_features(act_A)
    act_A = binarize(act_A)
    
    for layer_B in (pbar := tqdm(cols, desc="Computing layer_B", leave=False)):
        act_B_torch = all_activations_B[layer_B]
        if hasattr(act_B_torch, 'dtype') and act_B_torch.dtype == torch.bfloat16:
            act_B = bf16_torch_to_jax(act_B_torch)
        else:
            act_B = jnp.array(act_B_torch)
        
        act_B = flatten_tokens_features(act_B)
        act_B = binarize(act_B)

        inf_imb = information_matrix_fn(act_A, act_B)
        save_file(inf_imb, Path(output_folder, f"{layer_B}.safetensors"))


def main(args):
    """
    Main function handling all comparison types
    CHANGE: Complete rewrite to handle multiple comparison types
    """
    print(f"Comparison type: {args.comparison}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Output: {args.output_folder}")
    
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    # Configuration based on comparison type
    subset_size = 5000
    n_files = int(subset_size / 10)
    
    # CHANGE: Routing logic for different comparisons

    if args.comparison == "deepseek_clip_text":
        all_activations_A = collect_deepseek_data(
            args.input_path_A, n_loaded_tokens=9, subset_size=subset_size
        )
        all_activations_B = collect_clip_text_data(
            args.input_path_B, n_files=n_files, last_tokens=30
        )
        layer_A_id = 52  # DeepSeek layer
        
    elif args.comparison == "clip_text_image":
        all_activations_A = collect_clip_text_data(
            args.input_path_A, n_files=n_files, last_tokens=30
        )
        all_activations_B = collect_clip_image_data(
            args.input_path_B, n_files=n_files, last_tokens=200
        )
        # CHANGE: Use last CLIP text layer (layer 12 for CLIP text with 13 layers total: 0-12)
        layer_A_id = 12  # Last layer for CLIP text
        
    elif args.comparison == "dino_clip_image":
        all_activations_A = collect_image_model_data(
            args.input_path_A, n_files=n_files, last_tokens=200
        )
        all_activations_B = collect_clip_image_data(
            args.input_path_B, n_files=n_files, last_tokens=200
        )
        layer_A_id = 24  # DinoV2 layer
        
    elif args.comparison == "deepseek_dino":
        all_activations_A = collect_deepseek_data(
            args.input_path_A, n_loaded_tokens=9, subset_size=subset_size
        )
        all_activations_B = collect_image_model_data(
            args.input_path_B, n_files=n_files, last_tokens=200
        )
        layer_A_id = 52  # DeepSeek layer
        
    elif args.comparison == "deepseek_clip_image":
        all_activations_A = collect_deepseek_data(
            args.input_path_A, n_loaded_tokens=9, subset_size=subset_size
        )
        all_activations_B = collect_clip_image_data(
            args.input_path_B, n_files=n_files, last_tokens=200
        )
        layer_A_id = 52  # DeepSeek layer
    if args.comparison == "dino_igpt":
        all_activations_A = collect_image_model_data(
            args.input_path_A, n_files=n_files, last_tokens=50
        )
        all_activations_B = collect_image_model_data(
            args.input_path_B, n_files=n_files, last_tokens=50
        )
        layer_A_id = 24
    if args.comparison == "igpt_dino":
        all_activations_A = collect_image_model_data(
            args.input_path_A, n_files=n_files, last_tokens=50
        )
        all_activations_B = collect_image_model_data(
            args.input_path_B, n_files=n_files, last_tokens=50
        )
        layer_A_id = 24
        
    else:
        raise ValueError(f"Unknown comparison type: {args.comparison}")
    
    # Verify data loaded correctly
    sample_size_A = all_activations_A["layer_0"].shape[0]
    sample_size_B = all_activations_B["layer_0"].shape[0]
    print(f"Sample size A: {sample_size_A}")
    print(f"Sample size B: {sample_size_B}")
    
    # Determine comparison type for preprocessing
    if "text" in args.comparison and "image" in args.comparison:
        comp_type = "text_image"
    elif args.comparison in {"deepseek_clip_text", "deepseek_clip_image"}:
        comp_type = "text_image"
    elif "text" in args.comparison and args.comparison.count("text") == 2:
        comp_type = "text_text"
    elif args.comparison == "deepseek_dino":
        comp_type = "text_text"
    else:
        comp_type = "image_image"
    
    # Compute information imbalance
    compute_inf_imb_matrix(
        all_activations_A=all_activations_A,
        all_activations_B=all_activations_B,
        output_folder=args.output_folder,
        layer_A_id=layer_A_id,
        comparison_type=comp_type
    )
    
    print(f"Results saved to: {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Information Imbalance - Unified")
    parser.add_argument("--comparison", type=str, required=True, 
                       choices=["deepseek_clip_text", "clip_text_image", "dino_clip_image", 
                               "deepseek_dino", "deepseek_clip_image", "dino_igpt", "igpt_dino"],
                       help="Type of comparison to perform")
    parser.add_argument("--model_a", type=str, required=True, help="Model A name for output")
    parser.add_argument("--model_b", type=str, required=True, help="Model B name for output")
    parser.add_argument("--input_path_A", type=str, required=True, help="Path to model A activations")
    parser.add_argument("--input_path_B", type=str, required=True, help="Path to model B activations")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for results")
    
    args = parser.parse_args()
    main(args)
