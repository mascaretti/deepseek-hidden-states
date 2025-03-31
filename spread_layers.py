import pickle

from pathlib import Path
import re

import torch
from safetensors.torch import save_file

def extract_index(file):
    match = re.search(r'chunk_(\d+)\.pkl', file.name)
    return int(match.group(1)) if match else float('inf')  # push non-matching files to the end

def main(input_path="/home/laios/activations/flickr/deepseek_captions5"):
    folder = Path(input_path)

    #* create folder in output_path
    output_path= input_path + "_safetensors"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    #* collect all the files starting with *chunk_* in the input_path
    files = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.name.startswith("chunk_")],
        key=extract_index)

    for file in files:
        print(file.name)
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

        all_hidden_states = {}
        for i, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                all_hidden_states[f"input_{i}_layer_{layer_idx}"] = layer_tensor.squeeze(0)

        new_file_name = file.name.split('.')[0] + ".safetensors"
        save_file(all_hidden_states, output_path + "/" + new_file_name)

if __name__ == "__main__":
    main()