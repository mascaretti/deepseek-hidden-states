from collections import defaultdict
import pickle

from tqdm import tqdm
import torch

from utils import list_folder


def collect_data(input_path,
                 filter_layer=None,
                 n_loaded_tokens=40,
                 n_files=10,
                 mask=None,
                #  seed=42,
                 ):

    files = list_folder(input_path, desc="chunk_")[:n_files]
    all_hidden_states = defaultdict(list)

    for file in tqdm(files, desc="Collect File"):
        data = pickle.load(open(input_path + "/" + file.name, 'rb'))

        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            # assert hidden_states.shape[1] >= n_loaded_tokens
            hidden_states = hidden_states[:, -n_loaded_tokens-mask:-mask]

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                if filter_layer is not None:
                    if layer_idx!=filter_layer: continue

                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{layer_idx}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)
        # print("Layer=", layer, "activations shape= ", all_hidden_states[layer].shape, flush=True)

    return all_hidden_states


def collect_data_aligned(input_path_A,
                         input_path_B,
                         filter_layer=None,
                         n_loaded_tokens=40,
                         n_files=10,
                         mask=None):
    """
    Load activations from A and B enforcing sample alignment.

    Alignment guarantees:
    1) The exact same chunk file names are used in both paths.
    2) Each paired chunk has the same number of sentences.
    3) Sentences are paired by index within each chunk.
    """
    files_A = list_folder(input_path_A, desc="chunk_")[:n_files]
    files_B = {f.name: f for f in list_folder(input_path_B, desc="chunk_")}

    all_hidden_states_A = defaultdict(list)
    all_hidden_states_B = defaultdict(list)

    for file_A in tqdm(files_A, desc="Collect Paired Files"):
        if file_A.name not in files_B:
            raise ValueError(
                f"Missing aligned chunk in B: {file_A.name}. "
                "A/B paths are not sample-aligned."
            )
        file_B = files_B[file_A.name]

        data_A = pickle.load(open(input_path_A + "/" + file_A.name, "rb"))
        data_B = pickle.load(open(input_path_B + "/" + file_B.name, "rb"))

        if len(data_A) != len(data_B):
            raise ValueError(
                f"Sentence count mismatch in chunk {file_A.name}: "
                f"A={len(data_A)} vs B={len(data_B)}"
            )

        for sentence_A, sentence_B in zip(data_A, data_B):
            hidden_states_A = sentence_A["meta_info"]["hidden_states"][0]
            hidden_states_B = sentence_B["meta_info"]["hidden_states"][0]

            hidden_states_A = hidden_states_A[:, -n_loaded_tokens-mask:-mask]
            hidden_states_B = hidden_states_B[:, -n_loaded_tokens-mask:-mask]

            for layer_idx, layer_tensor_A in enumerate(hidden_states_A.split(1, dim=0)):
                if filter_layer is not None and layer_idx != filter_layer:
                    continue
                all_hidden_states_A[f"layer_{layer_idx}"].append(layer_tensor_A.squeeze(0))

            for layer_idx, layer_tensor_B in enumerate(hidden_states_B.split(1, dim=0)):
                if filter_layer is not None and layer_idx != filter_layer:
                    continue
                all_hidden_states_B[f"layer_{layer_idx}"].append(layer_tensor_B.squeeze(0))

    for layer, tensors in all_hidden_states_A.items():
        all_hidden_states_A[layer] = torch.stack(tensors)
    for layer, tensors in all_hidden_states_B.items():
        all_hidden_states_B[layer] = torch.stack(tensors)

    return all_hidden_states_A, all_hidden_states_B
