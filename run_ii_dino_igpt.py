#!/usr/bin/env python
"""
End-to-end Information Imbalance computation for DinoV2 vs iGPT
with jackknife procedure (5 replicas of 500 samples each).
Uses normalized L2 distance with token averaging.
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling
import jax
import jax.numpy as jnp
from functools import partial


def winsorized_mean(tensor, q_low=0.05, q_high=0.95, dim=1):
    """
    Computes the mean after clamping values to the q_low/q_high quantiles.

    Quantiles are computed per sample over all tokens × features jointly
    (one scalar bound per sample), so inter-feature relative scales are
    preserved.  Input must be 3-D with the token axis at `dim`.
    """
    tensor = tensor.to(torch.float32)
    B, T, E = tensor.shape

    # Flatten tokens × features → one threshold per sample
    flat = tensor.reshape(B, T * E)
    low_val  = torch.quantile(flat, q_low,  dim=1)  # (B,)
    high_val = torch.quantile(flat, q_high, dim=1)  # (B,)

    # Clip with per-sample scalar bounds, then mean over tokens
    clamped = torch.clamp(tensor,
                          min=low_val[:, None, None],
                          max=high_val[:, None, None])
    if not torch.isfinite(clamped).all():
        raise ValueError("Non-finite values after winsorization in chunk.")
    return clamped.mean(dim=dim).to(torch.float32)


# ========== Configuration ==========
N_REPLICAS = 5
SAMPLES_PER_REPLICA = 500
TOTAL_PAIRS = N_REPLICAS * SAMPLES_PER_REPLICA  # 2500
IMAGENET_PATH = "/projects/colt/imagenet21k_resized/imagenet21k_train/"
OUTPUT_DIR = Path("/home/mmahaut/projects/emecom/ii_dino_igpt_normalized_l2")
BATCH_SIZE = 8
SEED = 42

# ========== Information Imbalance Functions ==========
def normalized_L2_distance(x, y):
    """Compute L2 distance after normalizing each vector to unit norm."""
    x_norm = x / (jnp.linalg.norm(x) + 1e-8)
    y_norm = y / (jnp.linalg.norm(y) + 1e-8)
    diff = x_norm - y_norm
    return jnp.sqrt(jnp.sum(diff * diff))

def pairwise_distances(distance_fn, xs, ys):
    return jax.vmap(lambda x: jax.vmap(lambda y: distance_fn(x, y))(xs))(ys).T

def compute_relative_rank(x_dist, y_dist):
    from jax.scipy.stats import rankdata
    x_rank = rankdata(x_dist, method="min")
    y_rank = rankdata(y_dist, method="min")
    rel_ranks = jnp.where(x_rank == 1, y_rank - 1, -1)
    return jnp.sum(rel_ranks, where=rel_ranks != -1), jnp.asarray(rel_ranks != -1).sum()

batch_relative_ranks = jax.vmap(compute_relative_rank)

def build_information_imbalance(key, sample_size, distance_fn):
    from jax import random
    n_row_samples = jnp.array(100) if sample_size >= 500 else jnp.floor(0.2 * sample_size).astype(int)
    n_col_samples = (sample_size - n_row_samples).astype(int)
    
    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=sample_size, shape=(n_row_samples,), replace=False)
    remaining_indices = jnp.setdiff1d(jnp.arange(sample_size), indices_rows)
    key, subkey = random.split(key)
    indices_columns = random.choice(key=subkey, a=remaining_indices, shape=(n_col_samples,), replace=False)
    
    def information_imbalance(X, Y):
        assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
        d_X = pairwise_distances(distance_fn, X[indices_rows], X[indices_columns])
        d_Y = pairwise_distances(distance_fn, Y[indices_rows], Y[indices_columns])
        ranks, cards = batch_relative_ranks(d_X, d_Y)
        inf_imb = 2. * (jnp.sum(ranks) / jnp.sum(cards)) / (n_col_samples - 1.)
        reciprocal_ranks, reciprocal_cards = batch_relative_ranks(d_Y, d_X)
        reciprocal_inf_imb = 2. * (jnp.sum(reciprocal_ranks) / jnp.sum(reciprocal_cards)) / (n_col_samples - 1.)
        return inf_imb, reciprocal_inf_imb
    
    return jax.jit(information_imbalance)

# ========== Data Generation ==========
def generate_same_class_pairs(imagenet_dl, n_pairs, seed=42):
    """Generate pairs of images from the same class."""
    rng = np.random.default_rng(seed)
    classes = imagenet_dl.classes
    
    # Build class-to-indices mapping (faster than scanning each time)
    print("Building class index mapping...")
    class_to_indices = {i: [] for i in range(len(classes))}
    for idx, (_, class_idx) in enumerate(tqdm(imagenet_dl.samples, desc="Indexing")):
        class_to_indices[class_idx].append(idx)
    
    # Filter classes with at least 2 images
    valid_classes = [c for c, indices in class_to_indices.items() if len(indices) >= 2]
    print(f"Found {len(valid_classes)} classes with >= 2 images")
    
    pairs = []
    for _ in tqdm(range(n_pairs), desc="Generating pairs"):
        class_idx = rng.choice(valid_classes)
        indices = class_to_indices[class_idx]
        i1, i2 = rng.choice(len(indices), 2, replace=False)
        pairs.append((imagenet_dl.samples[indices[i1]][0], imagenet_dl.samples[indices[i2]][0]))
    
    return pairs

# ========== Feature Extraction ==========
def register_hooks(model, return_nodes):
    intermediate_outputs = {}
    
    def hook_fn(module, input, output, layer_name):
        intermediate_outputs[layer_name] = output.detach().cpu()
    
    hooks = []
    for layer_name, alias in return_nodes.items():
        layer = dict(model.named_modules())[layer_name]
        hooks.append(layer.register_forward_hook(partial(hook_fn, layer_name=alias)))
    
    def remove_hooks():
        for hook in hooks:
            hook.remove()
    
    return intermediate_outputs, remove_hooks

def extract_dino_features(image_paths, imagenet_dl, device, batch_size=8):
    """Extract DinoV2 features with token averaging."""
    print("Loading DinoV2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model.eval()
    model.to(device)
    
    return_nodes = {f"blocks.{i}": f"layer_{i}" for i in range(40)}
    intermediate_outputs, remove_hooks = register_hooks(model, return_nodes)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    all_features = {f"layer_{i}": [] for i in range(40)}
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="DinoV2 extraction"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            img = imagenet_dl.loader(path)
            images.append(transform(img))
        batch = torch.stack(images).to(device)
        
        with torch.no_grad():
            _ = model(batch)
        
        for layer_name, feat in intermediate_outputs.items():
            # Token averaging with winsorization: feat shape is (batch, n_tokens, embed_dim)
            feat_avg = winsorized_mean(feat)  # Winsorized mean across tokens -> (batch, embed_dim)
            all_features[layer_name].append(feat_avg)
    
    remove_hooks()
    
    # Concatenate all batches
    for k in all_features:
        all_features[k] = torch.cat(all_features[k], dim=0)
    
    return all_features

def extract_igpt_features(image_paths, imagenet_dl, device, batch_size=8):
    """Extract iGPT features with token averaging."""
    print("Loading iGPT model...")
    processor = ImageGPTImageProcessor.from_pretrained('openai/imagegpt-large')
    model = ImageGPTForCausalImageModeling.from_pretrained('openai/imagegpt-large')
    model.eval()
    model.to(device)
    
    n_layers = model.config.n_layer + 1
    all_features = {f"layer_{i}": [] for i in range(n_layers)}
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="iGPT extraction"):
        batch_paths = image_paths[i:i+batch_size]
        images = [imagenet_dl.loader(path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        for j, hidden_state in enumerate(outputs.hidden_states):
            # Token averaging with winsorization
            feat_avg = winsorized_mean(hidden_state).cpu()  # (batch, embed_dim)
            all_features[f"layer_{j}"].append(feat_avg)
    
    # Concatenate all batches
    for k in all_features:
        all_features[k] = torch.cat(all_features[k], dim=0)
    
    return all_features

# ========== Main Pipeline ==========
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load ImageNet and generate pairs
    print("=== Step 1: Generate same-class image pairs ===")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imagenet_dl = datasets.ImageFolder(IMAGENET_PATH, transform=transform)
    
    pairs = generate_same_class_pairs(imagenet_dl, TOTAL_PAIRS, seed=SEED)
    print(f"Generated {len(pairs)} pairs")
    
    # Save pairs for reference
    with open(OUTPUT_DIR / "image_pairs.txt", 'w') as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
    
    # Get unique image paths (each pair has 2 images)
    all_paths_A = [p[0] for p in pairs]
    all_paths_B = [p[1] for p in pairs]
    
    # Step 2: Extract DinoV2 features
    print("\n=== Step 2: Extract DinoV2 features ===")
    dino_features_A = extract_dino_features(all_paths_A, imagenet_dl, device, BATCH_SIZE)
    dino_features_B = extract_dino_features(all_paths_B, imagenet_dl, device, BATCH_SIZE)
    
    # Save DinoV2 features
    torch.save({'A': dino_features_A, 'B': dino_features_B}, OUTPUT_DIR / "dino_features.pt")
    print(f"DinoV2 features saved. Shape per layer: {dino_features_A['layer_0'].shape}")
    
    # Step 3: Extract iGPT features
    print("\n=== Step 3: Extract iGPT features ===")
    igpt_features_A = extract_igpt_features(all_paths_A, imagenet_dl, device, BATCH_SIZE)
    igpt_features_B = extract_igpt_features(all_paths_B, imagenet_dl, device, BATCH_SIZE)
    
    # Save iGPT features
    torch.save({'A': igpt_features_A, 'B': igpt_features_B}, OUTPUT_DIR / "igpt_features.pt")
    print(f"iGPT features saved. Shape per layer: {igpt_features_A['layer_0'].shape}")
    
    # Step 4: Compute II with jackknife for DinoV2
    print("\n=== Step 4: Compute II for DinoV2 (jackknife) ===")
    dino_ii_results = []
    for replica in range(N_REPLICAS):
        print(f"DinoV2 replica {replica}")
        start_idx = replica * SAMPLES_PER_REPLICA
        end_idx = start_idx + SAMPLES_PER_REPLICA
        
        key = jax.random.key(SEED + replica)
        ii_fn = build_information_imbalance(key, SAMPLES_PER_REPLICA, normalized_L2_distance)
        
        n_layers = len(dino_features_A)
        ii_matrix = np.zeros((n_layers, n_layers))
        ii_matrix_reciprocal = np.zeros((n_layers, n_layers))
        
        for i in tqdm(range(n_layers), desc=f"DinoV2 R{replica}", leave=False):
            act_A = dino_features_A[f"layer_{i}"][start_idx:end_idx].numpy()
            act_A = jnp.array(act_A)
            
            for j in range(n_layers):
                act_B = dino_features_B[f"layer_{j}"][start_idx:end_idx].numpy()
                act_B = jnp.array(act_B)
                
                ii_ab, ii_ba = ii_fn(act_A, act_B)
                ii_matrix[i, j] = ii_ab.item()
                ii_matrix_reciprocal[i, j] = ii_ba.item()
        
        dino_ii_results.append({'AB': ii_matrix, 'BA': ii_matrix_reciprocal})
    
    # Save DinoV2 II results
    np.save(OUTPUT_DIR / "dino_ii_jackknife.npy", dino_ii_results)
    
    # Step 5: Compute II with jackknife for iGPT
    print("\n=== Step 5: Compute II for iGPT (jackknife) ===")
    igpt_ii_results = []
    for replica in range(N_REPLICAS):
        print(f"iGPT replica {replica}")
        start_idx = replica * SAMPLES_PER_REPLICA
        end_idx = start_idx + SAMPLES_PER_REPLICA
        
        key = jax.random.key(SEED + replica + 100)
        ii_fn = build_information_imbalance(key, SAMPLES_PER_REPLICA, normalized_L2_distance)
        
        n_layers = len(igpt_features_A)
        ii_matrix = np.zeros((n_layers, n_layers))
        ii_matrix_reciprocal = np.zeros((n_layers, n_layers))
        
        for i in tqdm(range(n_layers), desc=f"iGPT R{replica}", leave=False):
            act_A = igpt_features_A[f"layer_{i}"][start_idx:end_idx].numpy()
            act_A = jnp.array(act_A)
            
            for j in range(n_layers):
                act_B = igpt_features_B[f"layer_{j}"][start_idx:end_idx].numpy()
                act_B = jnp.array(act_B)
                
                ii_ab, ii_ba = ii_fn(act_A, act_B)
                ii_matrix[i, j] = ii_ab.item()
                ii_matrix_reciprocal[i, j] = ii_ba.item()
        
        igpt_ii_results.append({'AB': ii_matrix, 'BA': ii_matrix_reciprocal})
    
    # Save iGPT II results
    np.save(OUTPUT_DIR / "igpt_ii_jackknife.npy", igpt_ii_results)
    
    # Compute mean and std across replicas
    print("\n=== Summary (Within-Model) ===")
    dino_ab_mean = np.mean([r['AB'] for r in dino_ii_results], axis=0)
    dino_ab_std = np.std([r['AB'] for r in dino_ii_results], axis=0)
    igpt_ab_mean = np.mean([r['AB'] for r in igpt_ii_results], axis=0)
    igpt_ab_std = np.std([r['AB'] for r in igpt_ii_results], axis=0)
    
    np.save(OUTPUT_DIR / "dino_ii_mean.npy", dino_ab_mean)
    np.save(OUTPUT_DIR / "dino_ii_std.npy", dino_ab_std)
    np.save(OUTPUT_DIR / "igpt_ii_mean.npy", igpt_ab_mean)
    np.save(OUTPUT_DIR / "igpt_ii_std.npy", igpt_ab_std)
    
    print(f"DinoV2 diagonal II (mean ± std): {np.diag(dino_ab_mean).mean():.4f} ± {np.diag(dino_ab_std).mean():.4f}")
    print(f"iGPT diagonal II (mean ± std): {np.diag(igpt_ab_mean).mean():.4f} ± {np.diag(igpt_ab_std).mean():.4f}")
    
    # Step 6: Cross-model II (DinoV2 <-> iGPT)
    print("\n=== Step 6: Compute Cross-Model II (DinoV2 <-> iGPT) ===")
    n_dino_layers = len(dino_features_A)
    n_igpt_layers = len(igpt_features_A)
    
    cross_dino_to_igpt_results = []  # DinoV2 -> iGPT
    cross_igpt_to_dino_results = []  # iGPT -> DinoV2
    
    for replica in range(N_REPLICAS):
        print(f"Cross-model replica {replica}")
        start_idx = replica * SAMPLES_PER_REPLICA
        end_idx = start_idx + SAMPLES_PER_REPLICA
        
        key = jax.random.key(SEED + replica + 200)
        ii_fn = build_information_imbalance(key, SAMPLES_PER_REPLICA, normalized_L2_distance)
        
        # DinoV2 -> iGPT: II(dino_layer_i, igpt_layer_j)
        ii_dino_to_igpt = np.zeros((n_dino_layers, n_igpt_layers))
        ii_igpt_to_dino = np.zeros((n_igpt_layers, n_dino_layers))
        
        for i in tqdm(range(n_dino_layers), desc=f"DinoV2->iGPT R{replica}", leave=False):
            dino_A = jnp.array(dino_features_A[f"layer_{i}"][start_idx:end_idx].numpy())
            for j in range(n_igpt_layers):
                igpt_B = jnp.array(igpt_features_B[f"layer_{j}"][start_idx:end_idx].numpy())
                ii_d2i, ii_i2d = ii_fn(dino_A, igpt_B)
                ii_dino_to_igpt[i, j] = ii_d2i.item()
        
        for i in tqdm(range(n_igpt_layers), desc=f"iGPT->DinoV2 R{replica}", leave=False):
            igpt_A = jnp.array(igpt_features_A[f"layer_{i}"][start_idx:end_idx].numpy())
            for j in range(n_dino_layers):
                dino_B = jnp.array(dino_features_B[f"layer_{j}"][start_idx:end_idx].numpy())
                ii_i2d, ii_d2i = ii_fn(igpt_A, dino_B)
                ii_igpt_to_dino[i, j] = ii_i2d.item()
        
        cross_dino_to_igpt_results.append(ii_dino_to_igpt)
        cross_igpt_to_dino_results.append(ii_igpt_to_dino)
    
    # Compute mean and std
    cross_d2i_mean = np.mean(cross_dino_to_igpt_results, axis=0)
    cross_d2i_std = np.std(cross_dino_to_igpt_results, axis=0)
    cross_i2d_mean = np.mean(cross_igpt_to_dino_results, axis=0)
    cross_i2d_std = np.std(cross_igpt_to_dino_results, axis=0)
    
    np.save(OUTPUT_DIR / "cross_dino_to_igpt_mean.npy", cross_d2i_mean)
    np.save(OUTPUT_DIR / "cross_dino_to_igpt_std.npy", cross_d2i_std)
    np.save(OUTPUT_DIR / "cross_igpt_to_dino_mean.npy", cross_i2d_mean)
    np.save(OUTPUT_DIR / "cross_igpt_to_dino_std.npy", cross_i2d_std)
    
    # Diagonal-like comparison: interpolate to match layers
    dino_x = np.linspace(0, 1, n_dino_layers)
    igpt_x = np.linspace(0, 1, n_igpt_layers)
    
    # Get "diagonal" by matching normalized depth
    cross_diag_d2i = np.array([cross_d2i_mean[i, int(i * (n_igpt_layers-1) / (n_dino_layers-1))] for i in range(n_dino_layers)])
    cross_diag_i2d = np.array([cross_i2d_mean[i, int(i * (n_dino_layers-1) / (n_igpt_layers-1))] for i in range(n_igpt_layers)])
    
    print(f"\n=== Cross-Model Summary ===")
    print(f"DinoV2 -> iGPT (diagonal mean): {cross_diag_d2i.mean():.4f}")
    print(f"iGPT -> DinoV2 (diagonal mean): {cross_diag_i2d.mean():.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
