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

def extract_image_hidden_states(model_name: str, dataset: Dataset, img_column: str, batch_size: int = 10, device: torch.device = None, output_directory=Path("/home/laios/image_captions_deepseek")):

    """Extract (and store as side effect) the sign of the hidden states for an image transformer.


    Args:
        model_name: The Hugging Face image transformer.
        dataset: The data to analyse.
        img_column: The relevant image column of the dataset.
        batch_size: The size of the batch to process.
        device: The relevant device, either CUDA or CPU.
        output_directory: Storing data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    image_tokens = image_processor(images=dataset[img_column], return_tensors="pt")
    image_tokens = Dataset.from_dict(image_tokens)
    image_tokens.set_format(type='torch')
    image_model = AutoModel.from_pretrained(model_name)

    dataloader = DataLoader(image_tokens, batch_size=batch_size)
    output_directory.mkdir(exist_ok=True, parents=True)

    print(output_directory)
    print(device)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = image_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            all_hidden_states = {}
            for layer_idx, layer_states in enumerate(hidden_states):
                all_hidden_states[f"layer_{layer_idx}"] = layer_states.cpu()
                print(all_hidden_states)

            print(Path(output_directory, f"chunk_{batch_idx}.pkl"))
            with open(Path(output_directory, f"chunk_{batch_idx}.pkl"), "wb") as f:
                pickle.dump(all_hidden_states, f)

def main():
    subset_size = 5_000
    image_model_name = "openai/imagegpt-large"
    print(image_model_name, flush=True)
    dataset = load_from_disk("/home/datasets/flickr30k/").select(np.arange(subset_size))
    print(len(dataset), flush=True)
    batch_size = 10

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE, flush=True)

    extract_image_hidden_states(model_name=image_model_name, dataset=dataset, img_column='jpg', batch_size=batch_size, device=DEVICE, output_directory=Path("/home/laios/imagegpt_activations"))
    
if __name__ == "__main__":
    main()
