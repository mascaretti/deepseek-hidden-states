import pickle
from pathlib import Path

import torch
from safetensors.torch import save_file

from tqdm import tqdm

from utils import list_chuncks_folder

def main(input_path):
    files = list_chuncks_folder(input_path)

    #* create folder in output_path
    output_path= input_path + "_safetensors"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print("Output path = ", output_path, flush=True)

    for file in tqdm(files):
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
    """
    Example usage:
    
    python3 spread_layers.py "/home/laios/activations/flickr/deepseek_captions1"
    """
    import sys
    input_path = sys.argv[1]

    print("Input path = ", input_path, flush=True)
    main(input_path=input_path)