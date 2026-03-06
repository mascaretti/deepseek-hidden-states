#!/usr/bin/env python
# coding: utf-8

import sys
import os
import multiprocessing
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoModel, ImageGPTImageProcessor, ImageGPTForCausalImageModeling
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils import binarize
from pathlib import Path
from tqdm import tqdm
import pickle
import time

def extract_image_hidden_states(model_name: str, dataset: Dataset, img_column: str, subset_size: int = 2500, batch_size: int = 10, device: torch.device = None, output_directory=Path("/home/laios/image_captions_deepseek")):
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

    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    image_model = AutoModel.from_pretrained(model_name)
    image_model.to(device)

    output_directory.mkdir(exist_ok=True, parents=True)

    print(output_directory)
    print(device)

    total_iterations = np.floor(subset_size / batch_size).astype(int)
    print(f"Batch size: {batch_size}; Total iterations: {total_iterations}.")
    
    n_layers = image_model.config.num_hidden_layers + 1

    print("Compute activations for images")
    for i in tqdm(range(total_iterations), desc="Processing image batch.", leave=False):
        image_tkns = image_processor(images=dataset[img_column][batch_size*i:batch_size*(i+1)], return_tensors="pt").to(device)
        with torch.no_grad():
            image_output = image_model(**image_tkns, output_hidden_states=True)
        all_hidden_states = {}
        for j in range(0, n_layers):
            all_hidden_states[f"layer_{j}"] = image_output["hidden_states"][j]
        with open(Path(output_directory, f"chunk_{i}.pkl"), "wb") as f:
            pickle.dump(all_hidden_states, f)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract image hidden states from a vision transformer.")
    parser.add_argument("--model", type=str, default="openai/imagegpt-small",
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: /home/laios/image-information-imbalance/<model>)")
    parser.add_argument("--dataset", type=str, default="/home/datasets/flickr30k/",
                        help="Path to the dataset on disk")
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    image_model_name = args.model
    output_directory = Path(args.output) if args.output else Path("/home/laios/image-information-imbalance", image_model_name.replace('/', '-'))

    print(f"Model: {image_model_name}")
    print(f"Output: {output_directory}")
    dataset = load_from_disk(args.dataset)
    print(f"Dataset size: {len(dataset)}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    extract_image_hidden_states(model_name=image_model_name, dataset=dataset, img_column='jpg',
                                batch_size=args.batch_size, subset_size=args.subset_size,
                                device=DEVICE, output_directory=output_directory)
    
if __name__ == "__main__":
    main()


