import pickle
from pathlib import Path

from tqdm import tqdm

from utils import list_folder

def main(input_path):
    files = list_folder(input_path,  desc="chunk_")

    #* create folder in output_path
    output_path= input_path + "_spread"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print("Output path = ", output_path, flush=True)

    for file in tqdm(files):
        print(file.name)
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))
        chunk_path = output_path + f"/{file.stem}"
        Path(chunk_path).mkdir(parents=True, exist_ok=True)

        for i, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                with open(chunk_path + f"/input_{i}_layer_{layer_idx}.pkl", 'wb') as f:
                    pickle.dump(layer_tensor.squeeze(0), f)

if __name__ == "__main__":
    """
    Example usage:
    
    python3 spread_layers.py "/home/laios/activations/flickr/deepseek_captions1"
    """
    import sys
    input_path = sys.argv[1]

    print("Input path = ", input_path, flush=True)
    main(input_path=input_path)