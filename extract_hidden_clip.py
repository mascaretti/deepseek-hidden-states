#!/usr/bin/env python
# coding: utf-8

import sys
import os
import multiprocessing
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoModel, ImageGPTImageProcessor, ImageGPTForCausalImageModeling, CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils import binarize
from pathlib import Path
from tqdm import tqdm
import pickle
import time

def extract_image_hidden_states(model_name: str, dataset: Dataset, subset_size: int = 2500, batch_size: int = 10, device: torch.device = None, output_directory=Path("/home/laios/image_captions_deepseek"), max_length: int = 30):
    """Extract (and store as side effect) the sign of the hidden states for an image transformer.

    Args:
        model_name: The Hugging Face image transformer.
        dataset: The data to analyse.
        img_column: The relevant image column of the dataset.
        subset_size: Number of samples to process.
        batch_size: The size of the batch to process.
        device: The relevant device, either CUDA or CPU.
        output_directory: Storing data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_model.to(device)

    (output_directory / "text").mkdir(exist_ok=True, parents=True)

    print(f"Output: {output_directory}")
    print(f"Device: {device}")

    total_iterations = np.floor(subset_size / batch_size).astype(int)
    print(f"Batch size: {batch_size}; Total iterations: {total_iterations}.")
    

    ## n_layers = clip_model.config.vision_config.num_hidden_layers + 1
    n_layers = clip_model.config.text_config.num_hidden_layers + 1
    
    print("Compute activations for images")
    for i in tqdm(range(total_iterations), desc="Processing image batch.", leave=False):
        chunk_path = Path(output_directory, "text", f"chunk_{i}.pkl")
        if chunk_path.exists():
            continue
        clip_inputs = clip_processor(
                text=dataset['txt'][batch_size*i:batch_size*(i+1)],
                images=dataset['jpg'][batch_size*i:batch_size*(i+1)],
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
                padding='max_length',
                max_length=max_length).to(device)
        with torch.no_grad():
            clip_output = clip_model(**clip_inputs, output_hidden_states=True)
        all_hidden_states = {}
        for j in range(0, n_layers):
            all_hidden_states[f"layer_{j}"] = clip_output.text_model_output.hidden_states[j]
            ## all_hidden_states[f"layer_{j}"] = clip_output.vision_model_output.hidden_states[j]
        with open(chunk_path, "wb") as f:
            pickle.dump(all_hidden_states, f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    subset_size = args.subset_size
    print(f"Subset size: {subset_size}")
    image_model_name = "openai/clip-vit-large-patch14"
    print(f"Model: {image_model_name}")
    ## dataset_name = "clip-benchmark/wds_flickr30k"
    ## dataset = load_dataset(dataset_name)
    dataset = load_from_disk("/home/datasets/flickr30k/")
    print(f"dataset length: {len(dataset)}")
    batch_size = 10

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    output_dir_name = image_model_name.replace('/', '-') + args.output_suffix
    extract_image_hidden_states(
        model_name=image_model_name,
        dataset=dataset,
        batch_size=batch_size,
        subset_size=subset_size,
        device=DEVICE,
        output_directory=Path("/home/laios/image-information-imbalance", output_dir_name),
        max_length=args.max_length,
    )
    
if __name__ == "__main__":
    main()


