from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms

from safetensors import safe_open
from information_imbalance_module import L2_distance, build_information_imbalance, hamming_distance, normalized_L2_distance
import jax
import jax.numpy as jnp
from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, makefolder
import pickle
from tqdm import tqdm
from pathlib import Path
import torch

import sys
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import seaborn as sns
from typing import List

import typer
import re
import warnings
from functools import partial
from itertools import combinations

app=typer.Typer()

class ImageGPTProcessor(torch.nn.Module):
    def __init__(self, model, processor, device):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = device
        self.n_layer = model.config.n_layer

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return {f"layer_{i}": hidden_state for i, hidden_state in enumerate(outputs.hidden_states)}


def register_hooks(model, return_nodes):
    # Register hooks to capture intermediate outputs
    intermediate_outputs = {}

    def hook_fn(module, input, output, layer_name):
        intermediate_outputs[layer_name] = output.detach().cpu()

    hooks = []
    for layer_name, alias in return_nodes.items():
        layer = dict(model.named_modules())[layer_name]
        hooks.append(layer.register_forward_hook(partial(hook_fn, layer_name=alias)))

    # Ensure hooks are removed after processing
    def remove_hooks():
        for hook in hooks:
            hook.remove()

    return intermediate_outputs, remove_hooks

def same_class_datapoints(imagenet_dl, output_path=None, n_files=10000):
    # get classes
    classes = imagenet_dl.classes
    sampled_pairs = []
    for _ in range(n_files):
        # choose a random class
        class_idx = np.random.randint(0, len(classes))
        # get the images of the class
        class_images = [imagenet_dl.samples[i][0] for i in range(len(imagenet_dl)) if imagenet_dl.targets[i] == class_idx]
        # sample two images
        sampled_images = np.random.choice(class_images, 2, replace=False)
        sampled_pairs.append(sampled_images)
    if output_path is not None:
        # output as a text file with two columns
        with open(output_path, 'w') as f:
            for pair in sampled_pairs:
                f.write(f"{pair[0]} {pair[1]}\n")
    return sampled_pairs

def closest_from_class_datapoints(imagenet_dl, transformations, output_path=None, n_files:int=10000, rank:int=0, n_ranks:int=1):
    sampled_pairs = []
    classes = imagenet_dl.classes
    n_pairs_per_class = n_files // len(classes)
    n_pairs_per_class = n_pairs_per_class if n_pairs_per_class > 0 else 1
    print(f"n_pairs_per_class={n_pairs_per_class}, rank={rank}, n_ranks={n_ranks}, len(classes)={len(classes)}", flush=True)

    # Divide classes among ranks
    classes_per_rank = len(classes) // n_ranks
    start_class = rank * classes_per_rank
    end_class = (rank + 1) * classes_per_rank if rank < n_ranks - 1 else len(classes)

    for class_idx in tqdm(range(start_class, end_class), desc=f"Processing classes (Rank {rank})"):
        # get the images of the class
        class_images = [imagenet_dl.samples[i][0] for i in range(len(imagenet_dl)) if imagenet_dl.targets[i] == class_idx]
        if len(class_images) < 2:
            continue  # skip classes with fewer than 2 images

        # Load all images of the class
        class_tensors = []
        for img_path in class_images:
            img = imagenet_dl.loader(img_path)
            img_tensor = transformations(img).unsqueeze(0)
            class_tensors.append(img_tensor)
        class_tensors = torch.cat(class_tensors, dim=0)

        # Compute cosine similarity
        class_tensors_flat = class_tensors.view(class_tensors.size(0), -1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        class_tensors_flat = class_tensors_flat.to(device)
        similarity_matrix = []
        batch_size = 2  # Adjust batch size based on available memory
        for i in range(0, class_tensors_flat.size(0), batch_size):
            batch = class_tensors_flat[i:i + batch_size]
            batch_similarity = torch.nn.functional.cosine_similarity(
                batch.unsqueeze(1), class_tensors_flat.unsqueeze(0), dim=-1
            )
            similarity_matrix.append(batch_similarity.cpu())
        similarity_matrix = torch.cat(similarity_matrix, dim=0)

        # Find the most distant pairs
        for _ in range(n_pairs_per_class):
            most_distant_pair_indices = torch.nonzero(similarity_matrix == similarity_matrix.min(), as_tuple=False)[0]
            most_distant_pair = [class_images[most_distant_pair_indices[0].item()], class_images[most_distant_pair_indices[1].item()]]
            sampled_pairs.append(most_distant_pair)

            # Mask the selected pair to avoid re-selection
            similarity_matrix[most_distant_pair_indices[0], most_distant_pair_indices[1]] = float('inf')
            similarity_matrix[most_distant_pair_indices[1], most_distant_pair_indices[0]] = float('inf')

    if output_path is not None:
        # output as a text file with two columns
        print(f"Saving sampled pairs to {output_path}_rank_{rank}.txt")
        print(sampled_pairs)
        with open(f"{output_path}_rank_{rank}.txt", 'w') as f:
            for pair in sampled_pairs:
                f.write(f"{pair[0]} {pair[1]}\n")
    return sampled_pairs

def different_class_datapoints(imagenet_dl, output_path=None, n_files=10000):
    sampled_pairs = []
    for _ in range(n_files):
        # get classes
        classes = imagenet_dl.classes
        # choose two random classes
        class_idx_A = np.random.randint(0, len(classes))
        class_idx_B = np.random.randint(0, len(classes))
        while class_idx_A == class_idx_B:
            class_idx_B = np.random.randint(0, len(classes))
        # get the images of the classes
        class_images_A = [imagenet_dl.samples[i][0] for i in range(len(imagenet_dl)) if imagenet_dl.targets[i] == class_idx_A]
        class_images_B = [imagenet_dl.samples[i][0] for i in range(len(imagenet_dl)) if imagenet_dl.targets[i] == class_idx_B]
        # sample one image from each class
        sampled_image_A = np.random.choice(class_images_A)
        sampled_image_B = np.random.choice(class_images_B)
        sampled_pairs.append([sampled_image_A, sampled_image_B])
    if output_path is not None:
        # output as a text file with two columns
        with open(output_path, 'w') as f:
            for pair in sampled_pairs:
                f.write(f"{pair[0]} {pair[1]}\n")
    return sampled_pairs

def single_class_datapoints(imagenet_dl, output_path=None, class_idx:int=None):
    sampled_pairs = []
    # get classes
    classes = imagenet_dl.classes
    # choose a random class
    class_idx = np.random.randint(0, len(classes)) if class_idx is None else class_idx
    print(f"Class index: {class_idx}", flush=True)
    # get all image pairs of the class
    class_images = [imagenet_dl.samples[i][0] for i in range(len(imagenet_dl)) if imagenet_dl.targets[i] == class_idx]
    sampled_pairs = []
    sampled_pairs = list(combinations(class_images, 2))
    if output_path is not None:
        # output as a text file with one column
        with open(output_path, 'w') as f:
            for pair in sampled_pairs:
                f.write(f"{pair[0]}\n")
    return sampled_pairs, class_idx

def furthest_single_class_datapoints(imagenet_dl, output_path=None, class_idx:int=None, n_files:int=1000, transformations=None):
    sampled_pairs = []
    # get classes
    classes = imagenet_dl.classes
    # choose a random class
    class_idx = np.random.randint(0, len(classes)) if class_idx is None else class_idx
    # get all image pairs of the class
    class_images = [imagenet_dl.samples[i][0] for i in range(len(imagenet_dl)) if imagenet_dl.targets[i] == class_idx]
    print(f"Class index: {class_idx} number of images: {len(class_images)} selecting {n_files} pairs", flush=True)
    sampled_pairs = []
    sampled_pairs = list(combinations(class_images, 2))
    # get the most distant pair
    sampled_pairs = sorted(sampled_pairs, key=lambda x: torch.nn.functional.cosine_similarity(
        transformations(imagenet_dl.loader(x[0])).view(1, -1),
        transformations(imagenet_dl.loader(x[1])).view(1, -1),
        dim=1
    ).item(), reverse=False)[:n_files]
    if output_path is not None:
        # output as a text file with one column
        with open(output_path, 'w') as f:
            for pair in sampled_pairs:
                f.write(f"{pair[0]}\n")
    return sampled_pairs, class_idx

@app.command()
def data_selection(n_files:int=10000, rank:int=0, n_rank:int=1):
    print(f"n_files {n_files}, rank {rank} out of {n_rank}", flush=True)
    imagenet_path = "/projects/colt/imagenet21k_resized/imagenet21k_train/"
    transformations = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_dl = datasets.ImageFolder(imagenet_path, transform=transformations)

    output_path1 = "./imagenet_same_class.txt"
    output_path2 = "./imagenet_different_class.txt"
    output_path3 = "./imagenet_closest_class"
    output_path_4 = "./imagenet_single_class"
    output_path_5 = "./imagenet_furthest_single_class"
    same_class_datapoints(imagenet_dl, output_path=output_path1, n_files=n_files)
    # different_class_datapoints(imagenet_dl, output_path=output_path2, n_files=n_files)
    # closest_from_class_datapoints(imagenet_dl, transformations, output_path=output_path3, n_files=n_files, rank=rank, n_ranks=n_rank)
    # _, class_idx=single_class_datapoints(imagenet_dl, output_path=output_path_4, class_idx=None)
    # furthest_single_class_datapoints(imagenet_dl, output_path=output_path_5, class_idx=class_idx, n_files=n_files, transformations=transformations)

@app.command()
def data_pickle(path_file, output_folder, imagenet_path, model_name, rank:int=0, n_ranks:int=100, batch_size:int=32):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    # imagenet_path = "/projects/colt/imagenet21k_resized/imagenet21k_train/"
    # path_file = "./imagenet_same_class.txt"
    # batch_size = 32
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) if model_name != "igpt" else None
    imagenet_dl = datasets.ImageFolder(imagenet_path, transform=transformations)
    # only load the rank images from path_file
    with open(path_file, 'r') as f:
        lines = f.readlines()
        lines = lines[rank::n_ranks]
    # process the images through the model        
    if model_name in ["vit_b_16", "resnet50"]:
        model = torch.hub.load("pytorch/vision", model_name, pretrained=True)
    elif model_name == "dino2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
        return_nodes = {f"blocks.{i}": f"layer_{i}" for i in range(40)}
        print("return_nodes = ", return_nodes, flush=True)

        # Register hooks for the current model
        intermediate_outputs, _ = register_hooks(model, return_nodes)

    
    if model_name == "vit_b_16":
        return_nodes = {f"encoder.layers.encoder_layer_{i}": f"layer_{i}" for i in range(12)}
        print("return_nodes = ", return_nodes, flush=True)
    elif model_name == "igpt":
        processor = ImageGPTImageProcessor.from_pretrained('openai/imagegpt-large')
        model = ImageGPTForCausalImageModeling.from_pretrained('openai/imagegpt-large')
        model = ImageGPTProcessor(model, processor, "cpu")

        
    elif model_name == "resnet50":
        # _, eval_nodes = get_graph_node_names(model)

        eval_nodes = [node for node in eval_nodes if re.match(r"layer\d+\.\d+", node)]
        eval_nodes = [node.split(".")[0] + "." + node.split(".")[1] for node in eval_nodes]
        eval_nodes = list(set(eval_nodes))
        # order using regex for first and second integer in the name
        return_nodes = sorted(eval_nodes, key=lambda x: (int(re.search(r'\d+', x).group()), int(re.search(r'\d+', x.split(".")[1]).group())))
        return_nodes = {node: "layer_" + str(i) for i, node in enumerate(return_nodes)}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "dino2": # here we're using the hook function
        model.eval()
        model.to(device)
        feature_extractor = model
    elif model_name == "igpt":
        model.model.eval()
        model.model.to(device)
        model.device = device
        feature_extractor = model
        return_nodes = {f"layer_{i}": f"layer_{i}" for i in range(model.n_layer)}
    else:
        feature_extractor = create_feature_extractor(model, return_nodes)
        feature_extractor.eval()
        feature_extractor.to(device)
    img1_minibatch = []
    img2_minibatch = []
    for i, line in enumerate(lines):
        img1_path, img2_path = line.strip().split()
        img1 = imagenet_dl.loader(img1_path)
        img2 = imagenet_dl.loader(img2_path)
        img1_tensor = transformations(img1).unsqueeze(0) if model_name != "igpt" else img1
        img2_tensor = transformations(img2).unsqueeze(0) if model_name != "igpt" else img2
        img1_minibatch.append(img1_tensor)
        img2_minibatch.append(img2_tensor)

        if len(img1_minibatch) >= batch_size:
            # process the batch
            img1_minibatch = torch.cat(img1_minibatch, dim=0).to(device) if model_name != "igpt" else img1_minibatch
            with torch.no_grad():
                features = feature_extractor(img1_minibatch)
                if model_name == "dino2":
                    # get features from the hook
                    print("intermediate_outputs = ", intermediate_outputs, flush=True)
                    features = {node : intermediate_outputs[node].cpu().detach().numpy() for node in return_nodes.values()} 
                else:
                    features = {node: features[node].cpu().detach().numpy() for node in return_nodes.values()}

            img1_minibatch = []
            img2_minibatch = torch.cat(img2_minibatch, dim=0).to(device) if model_name != "igpt" else img2_minibatch
            with torch.no_grad():
                features2 = feature_extractor(img2_minibatch)
                if model_name == "dino2":
                    # get features from the hook
                    features2 = {node : intermediate_outputs[node].cpu().detach().numpy() for node in return_nodes.values()}
                    
                else:
                    features2 = {node: features2[node].cpu().detach().numpy() for node in return_nodes.values()}                

            img2_minibatch = []
            # save the hidden states
            paired_hidden_states = (features, features2)
            with open(f"{output_folder}/{rank}_{n_ranks}_{i}.pkl", 'wb') as f:
                pickle.dump(paired_hidden_states, f)

@app.command()
def main(input_path, 
         n_tokens_list:List[int]=[1],
         output_folder:str=None,
         do_tokens:bool=True,
         average_tokens:bool=True,
         n_files:int=30,
         diagonal_only:bool=False,  # New option to compute only the diagonal
         rank:int=0,
         ):
    # collect all pickles
    files = sorted(Path(input_path).glob("*_10_*.pkl"))
    # shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(files)
    assert len(files) > n_files* (rank+1), f"Not enough files to process. {len(files)} files found, but {n_files*(rank+1)} files are required."
    files = files[n_files*rank:n_files*(rank+1)]
    if not do_tokens and n_tokens_list != [1]:
        n_tokens_list = [1]
        warnings.warn(f"`n_tokens_list` is set to [1] because `do_tokens` is False. Overriding the input value {n_tokens_list}.")

    acts = []
    for file in tqdm(list(files), desc="Collect File"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            _a = {k: torch.tensor(v) for k, v in data[0].items()}
            _b = {k: torch.tensor(v) for k, v in data[1].items()}
            acts.append((_a, _b))

    all_activations_A = {f"layer_{i}": torch.cat([act[0][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}
    all_activations_B = {f"layer_{i}": torch.cat([act[1][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}

    sample_size = all_activations_B["layer_0"].shape[0]
    rows = all_activations_A.keys()
    cols = all_activations_B.keys()

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=sample_size, distance_fn=normalized_L2_distance)
    for n_tokens in tqdm(n_tokens_list, desc='Computing token'):
        
        inf_imb_matrix = []
        reciprocal_inf_imb_matrix = []
        
        for layer_A in tqdm(rows, desc="Computing layer_A", leave=False):
            if diagonal_only and layer_A not in cols:
                continue  # Skip non-diagonal elements if diagonal_only is True

            activations_A = all_activations_A[layer_A]
            # tqdm.write(f"number of available tokens:{activations_A.shape[1]}")

            row = []
            reciprocal_row = []
            for layer_B in (pbar := tqdm(cols, desc="Computing layer_B", leave=False)):
                if diagonal_only and layer_A != layer_B:
                    continue  # Skip non-diagonal elements if diagonal_only is True

                act_A = activations_A[:,-n_tokens:,...] if do_tokens else activations_A
                act_A = act_A.mean(dim=1, keepdim=True) if average_tokens else act_A
                act_A = act_A.view(act_A.shape[0], act_A.shape[1], -1) if do_tokens and not average_tokens else act_A[:,None,...].view(act_A.shape[0], act_A.shape[1], -1)
                act_A = bf16_torch_to_jax(act_A.contiguous())
                act_A = flatten_tokens_features(act_A)
                act_B = all_activations_B[layer_B][:,-n_tokens:,...] if do_tokens else all_activations_B[layer_B]
                act_B = act_B.mean(dim=1, keepdim=True) if average_tokens else act_B
                act_B = act_B.view(act_B.shape[0], act_B.shape[1], -1) if do_tokens and not average_tokens else act_B[:,None,...].view(act_B.shape[0], act_B.shape[1], -1)
                act_B = bf16_torch_to_jax(act_B.contiguous())
                act_B = flatten_tokens_features(act_B)

                inf_imb, reciprocal_inf_imb = information_imbalance_fn(act_A, act_B)
                row.append(inf_imb.item())
                reciprocal_row.append(reciprocal_inf_imb.item())
                pbar.set_description(f"Computing layer_B (II(A-->B) {inf_imb:.4f} ; II(B-->A) {reciprocal_inf_imb:.4f})")
            inf_imb_matrix.append(row)
            reciprocal_inf_imb_matrix.append(reciprocal_row)

        inf_imb = np.array(inf_imb_matrix)
        reciprocal_inf_imb = np.array(reciprocal_inf_imb_matrix)
        np.save(file=f"{output_folder}/n_tokens_{n_tokens}_AB.npy", arr=inf_imb)
        np.save(file=f"{output_folder}/n_tokens_{n_tokens}_BA.npy", arr=reciprocal_inf_imb)

@app.command()
def draw_plot(np_file, output_file):
    # Load the data
    data = np.load(np_file)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=False, cmap="viridis", cbar=True)
    plt.title("Heatmap of Information Imbalance")
    plt.xlabel("Layer B")
    plt.ylabel("Layer A")
    plt.savefig(output_file)
    print(f"Saved heatmap to {output_file}", flush=True)
    plt.close()

@app.command()
def draw_plot2(np_file_list:List[str]=None, output_file="diagonal_plot.png"):
    # Load the data
    plt.figure(figsize=(10, 8))

    for np_file in np_file_list:

        # Extract metadata from the filename
        # match = re.search(r"(\d+)", Path(np_file).stem)
        # if not match:
        #     print(f"Skipping file {np_file} due to unmatched pattern", flush=True)
        #     continue
        # n_tokens = int(match.group(1))

        direction_match = re.search(r"(AB|BA)", Path(np_file).stem)
        direction = direction_match.group(1) if direction_match else "AB"  # Default to AB if not found
        model_name = "dino2" if "dino2" in np_file else "igpt" if "igpt" in np_file else "unknown"
        if model_name == "unknown":
            print(f"Skipping file {np_file} due to unmatched model name", flush=True)
            continue

        # Load the data and extract the diagonal
        data = np.load(np_file)
        x = np.linspace(0, 1, len(data))  # Normalize x-axis between 0 and 1
        diagonal = np.diag(data) if data.shape[0] == data.shape[1] else data[:, 0]

        # Plot the diagonal
        linestyle = '--' if direction == "BA" else '-'
        colour = 'blue' if model_name == "dino2" else 'orange' if model_name == "igpt" else 'green'
        plt.plot(x, diagonal, label=f"{model_name} {direction}", linestyle=linestyle, color=colour)

    plt.title("Diagonal of Information Imbalance Matrices")
    plt.xlabel("relative Layer Depth")
    plt.ylabel("Information Imbalance")
    plt.legend()
    plt.savefig(output_file)
    print(f"Saved diagonal plot to {output_file}", flush=True)
    plt.close()

if __name__ == "__main__":
    # data_selection()
    # Path("./imagenet_same_class").mkdir(parents=True, exist_ok=True)
    # data_pickle(
    #     path_file="./imagenet_same_class.txt",
    #     output_folder="./imagenet_same_class_resnet50",
    #     imagenet_path="/projects/colt/imagenet21k_resized/imagenet21k_train/",
    #     model_name="resnet50",
    #     # model_name="vit_b_16",
    #     rank=0,
    #     n_ranks=100,
    #     batch_size=32
    # )
    # main(
    #     input_path="./imagenet_same_class",
    #     n_tokens_list=[1, 2, 4, 8, 16, 32],
    #     output_folder="./imagenet_same_class",
    # )
    # for n_tokens in [1, 2, 4, 8, 16, 32]:
    #     draw_plot(
    #         np_file=f"./imagenet_different_class/n_tokens_{n_tokens}_AB.npy",
    #         output_file=f"./imagenet_different_class/n_tokens_{n_tokens}_AB.png"
    #     )
    #     draw_plot(
    #         np_file=f"./imagenet_different_class/n_tokens_{n_tokens}_BA.npy",
    #         output_file=f"./imagenet_different_class/n_tokens_{n_tokens}_BA.png"
    #     )
    app()