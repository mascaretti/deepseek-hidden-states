#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pathlib import Path
from safetensors.numpy import load_file, save_file
from tqdm import tqdm
import argparse
import json
from matplotlib import pyplot as plt
from functools import partial
import scienceplots  # Requires LaTeX


# In[2]:


try:
    plt.style.use(['nature', 'science'])  # Match notebook styling
except Exception:
    pass


# In[3]:


def make_path(data_path):
    try:
        assert Path.is_dir(data_path) == True
    except AssertionError:
        data_path = Path("..", data_path)
        assert Path.is_dir(data_path) == True, f"{data_path} is not a valid path."
    return data_path


# In[4]:


# Model colors - consistent across all plots
MODEL_COLORS = {
    'dino': '#0A5EB0',        # Blue
    'clip_image': '#DC143C',   # Crimson
    'clip_text': '#FF8C00',    # Dark Orange
    'deepseek': '#2E8B57',     # Sea Green
    'igpt': '#399918',          # Green (for future),
    'llama': '#F54927',
    'dino_base' : '#5585A3',
    'dino_small' : '#ACCCE0'
}

# Model display names
MODEL_NAMES = {
    'dino': 'DinoV2',
    'clip_image': 'CLIP-Image',
    'clip_text': 'CLIP-Text',
    'deepseek': 'DeepSeek-V3',
    'igpt': 'ImageGPT',
    'llama': 'LLama3.1-8B',
    'dino_base': 'DinoV2-base',
    'dino_small': 'DinoV2-small'
}

# Anchor layer for (to_model, from_model): the layer of to_model held fixed
# when sweeping from_model layers.
# LLM anchors fixed at ~55% relative depth; vision models at sweep minima.
ANCHOR_LAYER = {
    ('deepseek', 'igpt'): 33,       # ~55% of 60
    ('deepseek', 'dino'): 33,       # ~55% of 60
    ('deepseek', 'dino_base'): 33,  # ~55% of 60
    ('deepseek', 'dino_small'): 33, # ~55% of 60
    ('deepseek', 'llama'): 33,      # ~55% of 60
    ('llama', 'igpt'): 18,          # ~55% of 32
    ('llama', 'dino'): 18,          # ~55% of 32
    ('dino', 'igpt'): 24,
    ('igpt', 'dino'): 20,
    ('clip_image', 'clip_text'): 20,  # sweep: CLIP↔CLIP min at (20, 11)
    ('clip_image', 'dino'): 24,
    # Reverse directions (image/text model anchor, sweep LLM layers)
    ('dino', 'deepseek'): 24,
    ('igpt', 'deepseek'): 20,
    ('dino', 'llama'): 24,
    ('igpt', 'llama'): 20,
    ('llama', 'deepseek'): 18,      # ~55% of 32
}

N_LAYERS = {
    'deepseek' : 60,
    'llama' : 32,
    'dino' : 24,
    'igpt' : 48,
    'clip_image' : 24,
    'clip_text' : 12,
    'dino_base' : 12,
    'dino_small': 12
}


# In[5]:


def load_inf_imb_data(data_path, n_layers):
    """Load information imbalance data from safetensors files."""
    inf_imb = {}
    for i in range(1, n_layers + 1):
        inf_imb[f"layer_{i}"] = load_file(Path(data_path, f"layer_{i}.safetensors"))
    return inf_imb


# In[6]:


def extract_matrices(inf_imb, n_layers):
    """Extract information and multiplicity matrices from loaded data."""
    n_rows = inf_imb.get("layer_1").get('ranks').shape[0]

    inf_matrix = np.empty((n_layers, n_rows, 2))
    mult_matrix = np.empty((n_layers, n_rows, 2))

    for i in range(n_layers):
        inf_matrix[i, :, 0] = inf_imb[f"layer_{i+1}"]["ranks"]
        inf_matrix[i, :, 1] = inf_imb[f"layer_{i+1}"]["r_ranks"]
        mult_matrix[i, :, 0] = inf_imb[f"layer_{i+1}"]["cards"]
        mult_matrix[i, :, 1] = inf_imb[f"layer_{i+1}"]["r_cards"]

    return inf_matrix, mult_matrix, n_rows


def compute_mean_inf_imb(inf_matrix, mult_matrix, n_layers, n_rows, sample_size):
    """Compute mean information imbalance for each layer."""
    n_cols = sample_size - n_rows

    # Forward direction
    forward_means = np.zeros(n_layers)
    for i in range(n_layers):
        vals = inf_matrix[i, :, 0]
        cards = mult_matrix[i, :, 0]
        forward_means[i] = 2. * (np.sum(vals) / np.sum(cards)) / (n_cols - 2.)

    # Backward direction
    backward_means = np.zeros(n_layers)
    for i in range(n_layers):
        vals = inf_matrix[i, :, 1]
        cards = mult_matrix[i, :, 1]
        backward_means[i] = 2. * (np.sum(vals) / np.sum(cards)) / (n_cols - 2.)

    return forward_means, backward_means


def bootstrap(rng, inf_matrix, mult_matrix, n_replicas, bootstrap_size, n_layers, n_rows, sample_size):
    """Perform bootstrap analysis on information imbalance data."""
    n_cols = sample_size - n_rows

    inf_forward = np.empty((n_layers, n_replicas))
    inf_backward = np.empty((n_layers, n_replicas))

    for layer_id in tqdm(np.arange(n_layers), desc="Bootstrap layers"):
        ranks_forward = inf_matrix[layer_id, :, 0]
        ranks_backward = inf_matrix[layer_id, :, 1]
        cards_forward = mult_matrix[layer_id, :, 0]
        cards_backward = mult_matrix[layer_id, :, 1]

        for i in range(n_replicas):
            indices = rng.choice(n_rows, size=bootstrap_size, replace=True)
            inf_forward[layer_id, i] = 2. * (np.sum(ranks_forward[indices]) / np.sum(cards_forward[indices])) / (n_cols - 2.)
            inf_backward[layer_id, i] = 2. * (np.sum(ranks_backward[indices]) / np.sum(cards_backward[indices])) / (n_cols - 2.)

    return inf_forward, inf_backward


# In[7]:


def save_bootstrap_results(output_path, forward_data, backward_data, config):
    """Save bootstrap results and configuration."""
    # Save numpy arrays
    np.save(Path(output_path, f"{config['model_a']}_{config['model_b']}_{config['layer']}.npy"), forward_data)
    np.save(Path(output_path, f"{config['model_b']}_{config['model_a']}_{config['layer']}.npy"), backward_data)

    # Save configuration
    with open(Path(output_path, "bootstrap_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Save summary statistics
    summary = {
        'forward_mean': np.mean(forward_data, axis=1).tolist(),
        'forward_std': np.std(forward_data, axis=1).tolist(),
        'backward_mean': np.mean(backward_data, axis=1).tolist(),
        'backward_std': np.std(backward_data, axis=1).tolist(),
        'n_layers': forward_data.shape[0],
        'n_replicas': forward_data.shape[1]
    }

    with open(Path(output_path, f"summary_{config['model_a']}_{config['model_b']}.json"), 'w') as f:
        json.dump(summary, f, indent=2)


# In[8]:


def make_comparison(to_model, from_model, rng, n_replicas = 100, bootstrap_size = 750, sample_size = 5_000, base_dir=None, hamming=False):
    if base_dir is None:
        base_dir = Path("data")
    metric = "hamming" if hamming else "l2_centered_normalized"
    pair = (to_model, from_model)
    # Try sweep path first (layer×layer data), fall back to legacy single-slice
    if pair in ANCHOR_LAYER:
        sweep_path = Path(base_dir, "sweep", f"{to_model}_{from_model}",
                          f"layer_{ANCHOR_LAYER[pair]}", metric)
        if sweep_path.is_dir():
            AtoB = sweep_path
        else:
            AtoB = make_path(Path(base_dir, f"{to_model}_{from_model}", metric))
            print(f"  (no sweep data, using legacy path)")
    else:
        AtoB = make_path(Path(base_dir, f"{to_model}_{from_model}", metric))
    print(f"Loading {AtoB}")
    inf_imb = load_inf_imb_data(AtoB, N_LAYERS[from_model])
    inf_matrix, mult_matrix, n_rows = extract_matrices(inf_imb, N_LAYERS[from_model])
    inf_forward, inf_backward = bootstrap(rng, inf_matrix, mult_matrix, n_replicas, bootstrap_size, N_LAYERS[from_model], n_rows, sample_size)
    forward_mean = np.mean(inf_forward, axis=1)
    forward_std = np.std(inf_forward, axis=1)
    backward_mean = np.mean(inf_backward, axis=1)
    backward_std = np.std(inf_backward, axis=1)
    return [forward_mean, forward_std], [backward_mean, backward_std]


# In[9]:


rng = np.random.default_rng(2019)
n_replicas = 2_000
bootstrap_size = 200
sample_size = 5_000
base_dir = make_path(Path("data"))
assert Path(base_dir).is_dir() == True, f"{base_dir} doesn't exist!"


# In[10]:


DO_HAMMING = False


# In[11]:


compare_deepseek = partial(make_comparison, to_model="deepseek", rng=rng, base_dir=base_dir, hamming=DO_HAMMING)


# In[12]:


igpt_fwd, igpt_bwd = compare_deepseek(from_model="igpt")


# In[13]:


dino_fwd, dino_bwd = compare_deepseek(from_model="dino")


# In[14]:


llama_fwd, llama_bwd = compare_deepseek(from_model="llama")


# In[15]:


dino_igpt_fwd, dino_igpt_bwd = make_comparison(to_model="dino", from_model="igpt", rng=rng, hamming=DO_HAMMING)


# In[16]:


igpt_dino_fwd, igpt_dino_bwd = make_comparison(to_model="igpt", from_model="dino", rng=rng, hamming=DO_HAMMING)


# In[17]:


try:
    imagenet_path = make_path(Path("data", "ImageNet"))
    HAS_IMAGENET = True
except AssertionError:
    HAS_IMAGENET = False
    print("WARNING: ImageNet data not found, skipping ImageNet panels")


# In[18]:


if HAS_IMAGENET:
    dino_igpt_layer = 39
    imagenet_dino_igpt_fwd = []
    imagenet_dino_igpt_fwd.append(np.load(Path(imagenet_path, "dino_ii_mean.npy"))[dino_igpt_layer, 1:])
    imagenet_dino_igpt_fwd.append(np.load(Path(imagenet_path, "dino_ii_std.npy"))[dino_igpt_layer, 1:])

    igpt_layer_dino = 20
    imagenet_igpt_dino_fwd = []
    imagenet_igpt_dino_fwd.append(np.load(Path(imagenet_path, "igpt_ii_mean.npy"))[igpt_layer_dino, 1:])
    imagenet_igpt_dino_fwd.append(np.load(Path(imagenet_path, "igpt_ii_std.npy"))[igpt_layer_dino, 1:])

    # Cross-model ImageNet: DinoV2-huge (40 layers) ↔ ImageGPT (49 layers)
    cross_dino_to_igpt_mean = np.load(Path(imagenet_path, "cross_dino_to_igpt_mean.npy"))
    cross_dino_to_igpt_std = np.load(Path(imagenet_path, "cross_dino_to_igpt_std.npy"))
    cross_igpt_to_dino_mean = np.load(Path(imagenet_path, "cross_igpt_to_dino_mean.npy"))
    cross_igpt_to_dino_std = np.load(Path(imagenet_path, "cross_igpt_to_dino_std.npy"))

    # Fix dino at last layer (39), sweep igpt layers (skip layer 0)
    cross_dino_igpt_fwd = [cross_dino_to_igpt_mean[dino_igpt_layer, 1:], cross_dino_to_igpt_std[dino_igpt_layer, 1:]]
    cross_dino_igpt_bwd = [cross_igpt_to_dino_mean[1:, dino_igpt_layer], cross_igpt_to_dino_std[1:, dino_igpt_layer]]

    # Fix igpt at layer 20, sweep dino layers (skip layer 0)
    cross_igpt_dino_fwd = [cross_igpt_to_dino_mean[igpt_layer_dino, 1:], cross_igpt_to_dino_std[igpt_layer_dino, 1:]]
    cross_igpt_dino_bwd = [cross_dino_to_igpt_mean[1:, igpt_layer_dino], cross_dino_to_igpt_std[1:, igpt_layer_dino]]
else:
    raise ValueError


# In[20]:


ylim = (0.0, 1)
figsize = (4.5, 4.5)
fontsize_label = 15
x = {key:np.linspace(0, 1, N_LAYERS[key]) for key in N_LAYERS.keys()}


# In[21]:


if HAS_IMAGENET:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    label_forward = f"{MODEL_NAMES['igpt']}"
    ax.plot(x['igpt'], imagenet_igpt_dino_fwd[0], '-', color=MODEL_COLORS['igpt'],
            label=label_forward, linewidth=2, marker='o',
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(x['igpt'], imagenet_igpt_dino_fwd[0] - imagenet_igpt_dino_fwd[1], imagenet_igpt_dino_fwd[0] + imagenet_igpt_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
    label_forward = f"{MODEL_NAMES['dino']}"
    ax.plot(np.arange(39)/39, imagenet_dino_igpt_fwd[0], '-', color=MODEL_COLORS['dino'],
            label=label_forward, linewidth=2, marker='o',
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(np.arange(39)/39, imagenet_dino_igpt_fwd[0] - imagenet_dino_igpt_fwd[1], imagenet_dino_igpt_fwd[0] + imagenet_dino_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
    title = f"Dataset: ImageNet1k"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Relative depth", fontsize=fontsize_label)
    ax.set_ylabel("Information Imbalance", fontsize=fontsize_label)
    ax.tick_params(axis='both', labelsize=fontsize_label)
    ax.set_ylim(ylim)
    ax.legend(fontsize=fontsize_label, loc='best')
    plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/fig4a.pdf"), dpi = 300)
else:
    raise ValueError

# In[22]:


fig, ax = plt.subplots(1, 1, figsize=figsize)

label_forward = f"{MODEL_NAMES['igpt']}"
ax.plot(x['igpt'], igpt_fwd[0], '-', color=MODEL_COLORS['igpt'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['igpt'], igpt_fwd[0] - igpt_fwd[1], igpt_fwd[0] + igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
label_backward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['deepseek']}"
ax.plot(x['igpt'], igpt_bwd[0], '--', color=MODEL_COLORS['igpt'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_bwd[0] - igpt_bwd[1], igpt_bwd[0] + igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])

label_forward = f"{MODEL_NAMES['dino']}"
ax.plot(x['dino'], dino_fwd[0], '-', color=MODEL_COLORS['dino'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['dino'], dino_fwd[0] - dino_fwd[1], dino_fwd[0] + dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
label_backward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['deepseek']}"
ax.plot(x['dino'], dino_bwd[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_bwd[0] - dino_bwd[1], dino_bwd[0] + dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
title = f"Image-Captions on Flickr30k"
ax.set_title(title, fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')
## ax.grid(True, alpha=0.3)


plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/fig4b.pdf"), dpi = 300)
## plt.close()


# In[23]:


fig, ax = plt.subplots(1, 1, figsize=figsize)
label_forward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['igpt']}"
ax.plot(x['igpt'], dino_igpt_fwd[0], '-', color=MODEL_COLORS['llama'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['igpt'], dino_igpt_fwd[0] - dino_igpt_fwd[1], dino_igpt_fwd[0] + dino_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['llama'])
label_backward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['dino']}"
ax.plot(x['igpt'], dino_igpt_bwd[0], '--', color=MODEL_COLORS['llama'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], dino_igpt_bwd[0] - dino_igpt_bwd[1], dino_igpt_bwd[0] + dino_igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['llama'])
title = "Image-image on Flickr30k"
ax.set_title(title, fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.set_ylim(ylim)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.legend(fontsize=fontsize_label, loc='best')
## ax.grid(True, alpha=0.3)
plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/fig4c.pdf"), dpi = 300)
## plt.close()


# In[24]:


clipi_clipt_fwd, clipi_clipt_bwd = make_comparison(to_model="clip_image", from_model="clip_text", rng=rng, n_replicas=n_replicas, bootstrap_size=bootstrap_size, sample_size=sample_size, base_dir=base_dir, hamming=DO_HAMMING)
clipi_dino_fwd, clipi_dino_bwd = make_comparison(to_model="clip_image", from_model="dino", rng=rng, n_replicas=n_replicas, bootstrap_size=bootstrap_size, sample_size=sample_size, base_dir=base_dir, hamming=DO_HAMMING)

# ---- Figure 4a: ImageNet1k within-model + Image-Captions Flickr30k ----
if HAS_IMAGENET:
    x_igpt_imagenet = np.arange(48) / 48  # igpt 49 layers, skip layer 0 → 48 points
    x_dino_imagenet = np.arange(39) / 39  # dino-huge 40 layers, skip layer 0 → 39 points

fig, axes = plt.subplots(1, 2, figsize=(3.5 * figsize[0], figsize[1]), sharey=True)

for i, ax in enumerate(axes):
    ax.text(-0.12, 1.08, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

# a) ImageNet1k — within-model
ax = axes[0]
if HAS_IMAGENET:
    label_forward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['igpt']}"
    ax.plot(x_igpt_imagenet, imagenet_igpt_dino_fwd[0], '-', color=MODEL_COLORS['igpt'],
            label=label_forward, linewidth=2, marker='o',
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(x_igpt_imagenet, imagenet_igpt_dino_fwd[0] - imagenet_igpt_dino_fwd[1], imagenet_igpt_dino_fwd[0] + imagenet_igpt_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])

    label_forward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['dino']}"
    ax.plot(x_dino_imagenet, imagenet_dino_igpt_fwd[0], '-', color=MODEL_COLORS['dino'],
            label=label_forward, linewidth=2, marker='o',
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(x_dino_imagenet, imagenet_dino_igpt_fwd[0] - imagenet_dino_igpt_fwd[1], imagenet_dino_igpt_fwd[0] + imagenet_dino_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
else:
    ax.text(0.5, 0.5, "ImageNet data\nnot available", transform=ax.transAxes,
            ha='center', va='center', fontsize=12, color='gray')

ax.set_title("Same-class images (ImageNet-1k)", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("Information Imbalance", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

# b) Image-Captions on Flickr30k: DeepSeek ↔ igpt/dino + CLIP-Image → CLIP-Text
ax = axes[1]
label_forward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['igpt']}"
ax.plot(x['igpt'], igpt_fwd[0], '-', color=MODEL_COLORS['igpt'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_fwd[0] - igpt_fwd[1], igpt_fwd[0] + igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
ax.plot(x['igpt'], igpt_bwd[0], '--', color=MODEL_COLORS['igpt'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_bwd[0] - igpt_bwd[1], igpt_bwd[0] + igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])

label_forward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['dino']}"
ax.plot(x['dino'], dino_fwd[0], '-', color=MODEL_COLORS['dino'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_fwd[0] - dino_fwd[1], dino_fwd[0] + dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
ax.plot(x['dino'], dino_bwd[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_bwd[0] - dino_bwd[1], dino_bwd[0] + dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])

# CLIP-Image → CLIP-Text
label_forward = f"{MODEL_NAMES['clip_image']} → {MODEL_NAMES['clip_text']}"
ax.plot(x['clip_text'], clipi_clipt_fwd[0], '-', color=MODEL_COLORS['clip_text'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['clip_text'], clipi_clipt_fwd[0] - clipi_clipt_fwd[1], clipi_clipt_fwd[0] + clipi_clipt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['clip_text'])
ax.plot(x['clip_text'], clipi_clipt_bwd[0], '--', color=MODEL_COLORS['clip_text'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['clip_text'], clipi_clipt_bwd[0] - clipi_clipt_bwd[1], clipi_clipt_bwd[0] + clipi_clipt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['clip_text'])

ax.set_title("Image--caption pairs (Flickr30k)", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

plt.tight_layout()
plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/fig4a.pdf"), dpi=300)


# In[25]:


compare_llama = partial(make_comparison, to_model="llama", rng=rng, base_dir=base_dir, hamming=DO_HAMMING)


# In[26]:


igpt_fwd_ll, igpt_bwd_ll = compare_llama(from_model="igpt")
dino_fwd_ll, dino_bwd_ll = compare_llama(from_model="dino")


# In[28]:


dino_fwd_base, dino_bwd_base = compare_deepseek(from_model="dino_base")
dino_fwd_small, dino_bwd_small = compare_deepseek(from_model="dino_small")


# In[29]:


# ---- Figure 4b: Cross-model ImageNet + Cross-model Flickr30k + CLIP-Image↔DinoV2 ----
fig, axes = plt.subplots(1, 3, figsize=(3.5 * figsize[0], figsize[1]), sharey=True)
ylim = (0.0, 1)

for i, ax in enumerate(axes):
    ax.text(-0.14, 1.10, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

# a) ImageNet1k — cross-model
ax = axes[0]
if HAS_IMAGENET:
    label_forward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['igpt']}"
    ax.plot(x_igpt_imagenet, cross_dino_igpt_fwd[0], '-', color=MODEL_COLORS['igpt'],
            label=label_forward, linewidth=2, marker='o',
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(x_igpt_imagenet, cross_dino_igpt_fwd[0] - cross_dino_igpt_fwd[1], cross_dino_igpt_fwd[0] + cross_dino_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
    ax.plot(x_igpt_imagenet, cross_dino_igpt_bwd[0], '--', color=MODEL_COLORS['igpt'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(x_igpt_imagenet, cross_dino_igpt_bwd[0] - cross_dino_igpt_bwd[1], cross_dino_igpt_bwd[0] + cross_dino_igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])

    label_forward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['dino']}"
    ax.plot(x_dino_imagenet, cross_igpt_dino_fwd[0], '-', color=MODEL_COLORS['dino'],
            label=label_forward, linewidth=2, marker='o',
            markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(x_dino_imagenet, cross_igpt_dino_fwd[0] - cross_igpt_dino_fwd[1], cross_igpt_dino_fwd[0] + cross_igpt_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
    ax.plot(x_dino_imagenet, cross_igpt_dino_bwd[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    ax.fill_between(x_dino_imagenet, cross_igpt_dino_bwd[0] - cross_igpt_dino_bwd[1], cross_igpt_dino_bwd[0] + cross_igpt_dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
else:
    ax.text(0.5, 0.5, "ImageNet data\nnot available", transform=ax.transAxes,
            ha='center', va='center', fontsize=12, color='gray')

ax.set_title("ImageNet1k (cross-model)", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("Information Imbalance", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

# b) Image-image on Flickr30k
ax = axes[1]
label_forward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['igpt']}"
ax.plot(x['igpt'], dino_igpt_fwd[0], '-', color=MODEL_COLORS['igpt'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], dino_igpt_fwd[0] - dino_igpt_fwd[1], dino_igpt_fwd[0] + dino_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
ax.plot(x['igpt'], dino_igpt_bwd[0], '--', color=MODEL_COLORS['igpt'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], dino_igpt_bwd[0] - dino_igpt_bwd[1], dino_igpt_bwd[0] + dino_igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])

label_forward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['dino']}"
ax.plot(x['dino'], igpt_dino_fwd[0], '-', color=MODEL_COLORS['dino'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], igpt_dino_fwd[0] - igpt_dino_fwd[1], igpt_dino_fwd[0] + igpt_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
ax.plot(x['dino'], igpt_dino_bwd[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], igpt_dino_bwd[0] - igpt_dino_bwd[1], igpt_dino_bwd[0] + igpt_dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])

ax.set_title("Image-image on Flickr30k", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

# c) CLIP-Image ↔ DinoV2
ax = axes[2]
label_forward = f"{MODEL_NAMES['clip_image']} → {MODEL_NAMES['dino']}"
ax.plot(x['clip_image'], clipi_dino_fwd[0], '-', color=MODEL_COLORS['dino'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['clip_image'], clipi_dino_fwd[0] - clipi_dino_fwd[1], clipi_dino_fwd[0] + clipi_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
ax.plot(x['dino'], clipi_dino_bwd[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['clip_image'], clipi_dino_bwd[0] - clipi_dino_bwd[1], clipi_dino_bwd[0] + clipi_dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])

ax.set_title("CLIP-Image vs DinoV2", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

plt.tight_layout()
plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/fig4b.pdf"), dpi=300)

# ---- Figure 4c: Target Llama + DinoV2 sizes + Sweep DeepSeek layers ----
fig, axes = plt.subplots(1, 3, figsize=(3.5 * figsize[0], figsize[1]), sharey=True)

for i, ax in enumerate(axes):
    ax.text(-0.12, 1.08, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

# a) Target: Llama
ax = axes[0]
label_forward = f"{MODEL_NAMES['llama']} → {MODEL_NAMES['igpt']}"
ax.plot(x['igpt'], igpt_fwd_ll[0], '-', color=MODEL_COLORS['igpt'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_fwd_ll[0] - igpt_fwd_ll[1], igpt_fwd_ll[0] + igpt_fwd_ll[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
label_backward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['llama']}"
ax.plot(x['igpt'], igpt_bwd_ll[0], '--', color=MODEL_COLORS['igpt'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_bwd_ll[0] - igpt_bwd_ll[1], igpt_bwd_ll[0] + igpt_bwd_ll[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])

label_forward = f"{MODEL_NAMES['llama']} → {MODEL_NAMES['dino']}"
ax.plot(x['dino'], dino_fwd_ll[0], '-', color=MODEL_COLORS['dino'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_fwd_ll[0] - dino_fwd_ll[1], dino_fwd_ll[0] + dino_fwd_ll[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
label_backward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['llama']}"
ax.plot(x['dino'], dino_bwd_ll[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_bwd_ll[0] - dino_bwd_ll[1], dino_bwd_ll[0] + dino_bwd_ll[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])

ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("Information Imbalance", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

# b) Target: DeepSeek — dino large/base/small (was fig4_appendix c)
ax = axes[1]
## Dino Large
label_forward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['dino']}"
ax.plot(x['dino'], dino_fwd[0], '-', color=MODEL_COLORS['dino'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_fwd[0] - dino_fwd[1], dino_fwd[0] + dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
ax.plot(x['dino'], dino_bwd[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_bwd[0] - dino_bwd[1], dino_bwd[0] + dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])

## Dino Base
label_forward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['dino_base']}"
ax.plot(x['dino_base'], dino_fwd_base[0], '-', color=MODEL_COLORS['dino_base'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino_base'], dino_fwd_base[0] - dino_fwd_base[1], dino_fwd_base[0] + dino_fwd_base[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino_base'])
ax.plot(x['dino_base'], dino_bwd_base[0], '--', color=MODEL_COLORS['dino_base'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino_base'], dino_bwd_base[0] - dino_bwd_base[1], dino_bwd_base[0] + dino_bwd_base[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino_base'])

## Dino Small
label_forward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['dino_small']}"
ax.plot(x['dino_small'], dino_fwd_small[0], '-', color=MODEL_COLORS['dino_small'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino_small'], dino_fwd_small[0] - dino_fwd_small[1], dino_fwd_small[0] + dino_fwd_small[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino_small'])
ax.plot(x['dino_small'], dino_bwd_small[0], '--', color=MODEL_COLORS['dino_small'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino_small'], dino_bwd_small[0] - dino_bwd_small[1], dino_bwd_small[0] + dino_bwd_small[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino_small'])

ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

# c) Image models → DeepSeek (all layers)
ds_dino_fwd, ds_dino_bwd = make_comparison(to_model="dino", from_model="deepseek", rng=rng, n_replicas=n_replicas, bootstrap_size=bootstrap_size, sample_size=sample_size, base_dir=base_dir, hamming=DO_HAMMING)
ds_igpt_fwd, ds_igpt_bwd = make_comparison(to_model="igpt", from_model="deepseek", rng=rng, n_replicas=n_replicas, bootstrap_size=bootstrap_size, sample_size=sample_size, base_dir=base_dir, hamming=DO_HAMMING)
ds_llama_fwd, ds_llama_bwd = make_comparison(to_model="llama", from_model="deepseek", rng=rng, n_replicas=n_replicas, bootstrap_size=bootstrap_size, sample_size=sample_size, base_dir=base_dir, hamming=DO_HAMMING)

ax = axes[2]
# DinoV2 (last layer) → DeepSeek[layer]
label_forward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_dino_fwd[0], '-', color=MODEL_COLORS['dino'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_dino_fwd[0] - ds_dino_fwd[1], ds_dino_fwd[0] + ds_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
ax.plot(x['deepseek'], ds_dino_bwd[0], '--', color=MODEL_COLORS['dino'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_dino_bwd[0] - ds_dino_bwd[1], ds_dino_bwd[0] + ds_dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])

# ImageGPT (~42% depth) → DeepSeek[layer]
label_forward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_igpt_fwd[0], '-', color=MODEL_COLORS['igpt'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_igpt_fwd[0] - ds_igpt_fwd[1], ds_igpt_fwd[0] + ds_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
ax.plot(x['deepseek'], ds_igpt_bwd[0], '--', color=MODEL_COLORS['igpt'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_igpt_bwd[0] - ds_igpt_bwd[1], ds_igpt_bwd[0] + ds_igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])

ax.set_xlabel("DeepSeek-V3 relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')

plt.tight_layout()
plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/fig4c.pdf"), dpi=300)


# In[30]:


# In[31]:


fig, axes = plt.subplots(1, 3, figsize=(3.2 * figsize[0], figsize[1]))

for i, ax in enumerate(axes):
    ax.text(-0.14, 1.10, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

ax = axes[0]
label_forward = f"{MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_igpt_fwd[0], '-', color=MODEL_COLORS['deepseek'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['deepseek'], ds_igpt_fwd[0] - ds_igpt_fwd[1], ds_igpt_fwd[0] + ds_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_igpt_bwd[0], '--', color=MODEL_COLORS['deepseek'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_igpt_bwd[0] - ds_igpt_bwd[1], ds_igpt_bwd[0] + ds_igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')
title = f"Target: {MODEL_NAMES['igpt']}[{ANCHOR_LAYER[('igpt', 'deepseek')]}]"
ax.set_title(title, fontsize=12)

ax = axes[1]
label_forward = f"{MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_dino_fwd[0], '-', color=MODEL_COLORS['deepseek'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['deepseek'], ds_dino_fwd[0] - ds_dino_fwd[1], ds_dino_fwd[0] + ds_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_dino_bwd[0], '--', color=MODEL_COLORS['deepseek'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_dino_bwd[0] - ds_dino_bwd[1], ds_dino_bwd[0] + ds_dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])

## title = f"Image-Captions on Flickr30k"
## ax.set_title(title, fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')
title = f"Target: {MODEL_NAMES['dino']}[{ANCHOR_LAYER[('dino', 'deepseek')]}]"
ax.set_title(title, fontsize=12)
## ax.grid(True, alpha=0.3)

ax = axes[2]
label_forward = f"{MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_llama_fwd[0], '-', color=MODEL_COLORS['deepseek'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['deepseek'], ds_llama_fwd[0] - ds_llama_fwd[1], ds_llama_fwd[0] + ds_llama_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_llama_bwd[0], '--', color=MODEL_COLORS['deepseek'], linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_llama_bwd[0] - ds_llama_bwd[1], ds_llama_bwd[0] + ds_llama_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')
title = f"Target: {MODEL_NAMES['llama']}[{ANCHOR_LAYER[('llama', 'deepseek')]}]"
ax.set_title(title, fontsize=12)
## ax.grid(True, alpha=0.3)

# plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/fig4b.pdf"), dpi = 300)
## plt.close()


# In[32]:


fig, axes = plt.subplots(1, 2, figsize=(2.5 * figsize[0], figsize[1]))

for i, ax in enumerate(axes):
    ax.text(-0.14, 1.10, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

ax = axes[0]
label_forward = f"{MODEL_NAMES['deepseek']}[{ANCHOR_LAYER[('deepseek', 'dino')]}] → {MODEL_NAMES['dino']}"
ax.plot(x['dino'], dino_fwd[0], '-', color=MODEL_COLORS['dino'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_fwd[0] - dino_fwd[1], dino_fwd[0] + dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
label_backward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['deepseek']}[{ANCHOR_LAYER[('deepseek', 'dino')]}]"
ax.plot(x['dino'], dino_bwd[0], '--', color=MODEL_COLORS['dino'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_bwd[0] - dino_bwd[1], dino_bwd[0] + dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
ax.vlines(x = ANCHOR_LAYER[('dino', 'deepseek')]/N_LAYERS['dino'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_dino_bwd[0][ANCHOR_LAYER[('deepseek', 'dino')]], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_dino_fwd[0][ANCHOR_LAYER[('deepseek', 'dino')]], xmin=-0., xmax=1.05, alpha=0.6, color="red")

ax.set_title("Image-Captions on Flickr30k", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')



ax = axes[1]
label_forward = f"{MODEL_NAMES['dino']}[{ANCHOR_LAYER[('dino', 'deepseek')]}] → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_dino_fwd[0], '-', color=MODEL_COLORS['deepseek'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_dino_fwd[0] - ds_dino_fwd[1], ds_dino_fwd[0] + ds_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['dino']}[{ANCHOR_LAYER[('dino', 'deepseek')]}]"
ax.plot(x['deepseek'], ds_dino_bwd[0], '--', color=MODEL_COLORS['deepseek'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_dino_bwd[0] - ds_dino_bwd[1], ds_dino_bwd[0] + ds_dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.vlines(x = ANCHOR_LAYER[('deepseek', 'dino')]/N_LAYERS['deepseek'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_dino_bwd[0][ANCHOR_LAYER[('deepseek', 'dino')]], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_dino_fwd[0][ANCHOR_LAYER[('deepseek', 'dino')]], xmin=-0., xmax=1.05, alpha=0.6, color="red")
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')


# In[33]:


fig, axes = plt.subplots(1, 2, figsize=(2.5 * figsize[0], figsize[1]))

for i, ax in enumerate(axes):
    ax.text(-0.14, 1.10, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

ax = axes[0]
label_forward = f"{MODEL_NAMES['deepseek']}[{ANCHOR_LAYER[('deepseek', 'igpt')]}] → {MODEL_NAMES['igpt']}"
ax.plot(x['igpt'], igpt_fwd[0], '-', color=MODEL_COLORS['igpt'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_fwd[0] - igpt_fwd[1], igpt_fwd[0] + igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
label_backward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['deepseek']}[{ANCHOR_LAYER[('deepseek', 'igpt')]}]"
ax.plot(x['igpt'], igpt_bwd[0], '--', color=MODEL_COLORS['igpt'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_bwd[0] - igpt_bwd[1], igpt_bwd[0] + igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
ax.vlines(x = ANCHOR_LAYER[('igpt', 'deepseek')]/N_LAYERS['igpt'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_igpt_bwd[0][ANCHOR_LAYER[('deepseek', 'igpt')]], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_igpt_fwd[0][ANCHOR_LAYER[('deepseek', 'igpt')]], xmin=-0., xmax=1.05, alpha=0.6, color="red")

## ax.set_title("Image-Captions on Flickr30k", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')



ax = axes[1]
label_forward = f"{MODEL_NAMES['igpt']}[{ANCHOR_LAYER[('igpt', 'deepseek')]}] → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_igpt_fwd[0], '-', color=MODEL_COLORS['deepseek'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_igpt_fwd[0] - ds_igpt_fwd[1], ds_igpt_fwd[0] + ds_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['igpt']}[{ANCHOR_LAYER[('igpt', 'deepseek')]}]"
ax.plot(x['deepseek'], ds_igpt_bwd[0], '--', color=MODEL_COLORS['deepseek'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_igpt_bwd[0] - ds_igpt_bwd[1], ds_igpt_bwd[0] + ds_igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.vlines(x = ANCHOR_LAYER[('deepseek', 'igpt')]/N_LAYERS['deepseek'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_igpt_bwd[0][ANCHOR_LAYER[('deepseek', 'igpt')]], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_igpt_fwd[0][ANCHOR_LAYER[('deepseek', 'igpt')]], xmin=-0., xmax=1.05, alpha=0.6, color="red")
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')


# In[34]:


fig, axes = plt.subplots(1, 2, figsize=(2.5 * figsize[0], figsize[1]))

for i, ax in enumerate(axes):
    ax.text(-0.14, 1.10, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

ax = axes[0]
label_forward = f"{MODEL_NAMES['deepseek']}[{ANCHOR_LAYER[('deepseek', 'llama')]}] → {MODEL_NAMES['llama']}"
ax.plot(x['llama'], llama_fwd[0], '-', color=MODEL_COLORS['llama'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['llama'], llama_fwd[0] - llama_fwd[1], llama_fwd[0] + llama_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['llama'])
label_backward = f"{MODEL_NAMES['llama']} → {MODEL_NAMES['deepseek']}[{ANCHOR_LAYER[('deepseek', 'llama')]}]"
ax.plot(x['llama'], llama_bwd[0], '--', color=MODEL_COLORS['llama'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['llama'], llama_bwd[0] - llama_bwd[1], llama_bwd[0] + llama_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['llama'])
ax.vlines(x = ANCHOR_LAYER[('llama', 'deepseek')]/N_LAYERS['llama'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_llama_bwd[0][ANCHOR_LAYER[('deepseek', 'llama')]], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_llama_fwd[0][ANCHOR_LAYER[('deepseek', 'llama')]], xmin=-0., xmax=1.05, alpha=0.6, color="red")

## ax.set_title("Image-Captions on Flickr30k", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')



ax = axes[1]
label_forward = f"{MODEL_NAMES['llama']}[{ANCHOR_LAYER[('llama', 'deepseek')]}] → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_llama_fwd[0], '-', color=MODEL_COLORS['deepseek'],
        label=label_forward, linewidth=2, marker='o',
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_llama_fwd[0] - ds_llama_fwd[1], ds_llama_fwd[0] + ds_llama_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['llama']}[{ANCHOR_LAYER[('llama', 'deepseek')]}]"
ax.plot(x['deepseek'], ds_llama_bwd[0], '--', color=MODEL_COLORS['deepseek'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_llama_bwd[0] - ds_llama_bwd[1], ds_llama_bwd[0] + ds_llama_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.vlines(x = ANCHOR_LAYER[('deepseek', 'llama')]/N_LAYERS['deepseek'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_llama_bwd[0][ANCHOR_LAYER[('deepseek', 'llama')]], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_llama_fwd[0][ANCHOR_LAYER[('deepseek', 'llama')]], xmin=-0., xmax=1.05, alpha=0.6, color="red")
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')


# In[ ]:


# ---- Heatmaps: layer x layer II matrices from sweep data ----
from safetensors.numpy import load_file as sf_load

def load_sweep_matrix(sweep_dir):
    """Load full layer_A x layer_B forward II matrix from sweep results."""
    layer_dirs = sorted(Path(sweep_dir).glob('layer_*'), key=lambda p: int(p.name.split('_')[1]))
    layer_A_indices = [int(p.name.split('_')[1]) for p in layer_dirs]

    metric_dir = layer_dirs[0] / 'l2_centered_normalized'
    layer_B_files = sorted(metric_dir.glob('layer_*.safetensors'), key=lambda f: int(f.stem.split('_')[1]))
    layer_B_indices = [int(f.stem.split('_')[1]) for f in layer_B_files]

    fwd = np.full((len(layer_A_indices), len(layer_B_indices)), np.nan)
    bwd = np.full((len(layer_A_indices), len(layer_B_indices)), np.nan)

    for i, ld in enumerate(layer_dirs):
        metric_dir = ld / 'l2_centered_normalized'
        for j, lb in enumerate(layer_B_indices):
            f = metric_dir / f'layer_{lb}.safetensors'
            if f.exists():
                data = sf_load(str(f))
                fwd[i, j] = float(data['inf_imb'])
                bwd[i, j] = float(data['reciprocal_inf_imb'])

    return fwd, bwd, np.array(layer_A_indices), np.array(layer_B_indices)

sweep_base = make_path(Path("data", "sweep"))

# Load matrices — DeepSeek sweeps (always available)
ds_igpt_hm_fwd, ds_igpt_hm_bwd, ds_igpt_la, ds_igpt_lb = load_sweep_matrix(Path(sweep_base, "deepseek_igpt"))
ds_dino_hm_fwd, ds_dino_hm_bwd, ds_dino_la, ds_dino_lb = load_sweep_matrix(Path(sweep_base, "deepseek_dino"))

ds_igpt_hm_avg = (ds_igpt_hm_fwd + ds_igpt_hm_bwd) / 2
ds_dino_hm_avg = (ds_dino_hm_fwd + ds_dino_hm_bwd) / 2

min_igpt = np.unravel_index(np.nanargmin(ds_igpt_hm_avg), ds_igpt_hm_avg.shape)
min_dino = np.unravel_index(np.nanargmin(ds_dino_hm_avg), ds_dino_hm_avg.shape)

print(f"DeepSeek-ImageGPT min avg II: {ds_igpt_hm_avg[min_igpt]:.4f} at DS layer {ds_igpt_la[min_igpt[0]]}, IGPT layer {ds_igpt_lb[min_igpt[1]]}")
print(f"DeepSeek-DinoV2   min avg II: {ds_dino_hm_avg[min_dino]:.4f} at DS layer {ds_dino_la[min_dino[0]]}, Dino layer {ds_dino_lb[min_dino[1]]}")

# Load matrices — Llama and CLIP sweeps (may not be available yet)
sweep_pairs_extra = {
    'llama_igpt': ('llama', 'igpt', 'LLama3.1-8B', 'ImageGPT'),
    'llama_dino': ('llama', 'dino', 'LLama3.1-8B', 'DinoV2'),
    'clip_text_clip_image': ('clip_text', 'clip_image', 'CLIP-Text', 'CLIP-Image'),
}

extra_heatmaps = {}
for sweep_name, (model_a, model_b, label_a, label_b) in sweep_pairs_extra.items():
    sweep_dir = Path(sweep_base, sweep_name)
    if sweep_dir.exists() and any(sweep_dir.glob('layer_*')):
        try:
            hm_fwd, hm_bwd, la, lb = load_sweep_matrix(sweep_dir)
            hm_avg = (hm_fwd + hm_bwd) / 2
            hm_min = np.unravel_index(np.nanargmin(hm_avg), hm_avg.shape)
            extra_heatmaps[sweep_name] = {
                'avg': hm_avg, 'la': la, 'lb': lb, 'min': hm_min,
                'model_a': model_a, 'model_b': model_b,
                'label_a': label_a, 'label_b': label_b,
            }
            print(f"{label_a}-{label_b} min avg II: {hm_avg[hm_min]:.4f} at {label_a} layer {la[hm_min[0]]}, {label_b} layer {lb[hm_min[1]]}")
        except Exception as e:
            print(f"WARNING: Could not load {sweep_name} sweep: {e}")

n_heatmaps = 2 + len(extra_heatmaps)

# Plot heatmaps
fig, axes = plt.subplots(1, n_heatmaps, figsize=(1.4 * n_heatmaps * figsize[0], figsize[1]))
if n_heatmaps == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.text(-0.12, 1.08, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

# a) DeepSeek vs ImageGPT
ax = axes[0]
x_rel = ds_igpt_la / N_LAYERS['deepseek']
y_rel = ds_igpt_lb / N_LAYERS['igpt']
im = ax.pcolormesh(x_rel, y_rel, ds_igpt_hm_avg.T, shading='auto', cmap='viridis_r')
ax.plot(x_rel[min_igpt[0]], y_rel[min_igpt[1]], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xlabel("DeepSeek-V3 relative depth", fontsize=fontsize_label)
ax.set_ylabel("ImageGPT relative depth", fontsize=fontsize_label)
ax.set_title("DeepSeek-V3 vs ImageGPT", fontsize=12)
ax.tick_params(axis='both', labelsize=fontsize_label)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Information Imbalance (avg)", fontsize=fontsize_label - 2)

# b) DeepSeek vs DinoV2
ax = axes[1]
x_rel = ds_dino_la / N_LAYERS['deepseek']
y_rel = ds_dino_lb / N_LAYERS['dino']
im = ax.pcolormesh(x_rel, y_rel, ds_dino_hm_avg.T, shading='auto', cmap='viridis_r')
ax.plot(x_rel[min_dino[0]], y_rel[min_dino[1]], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xlabel("DeepSeek-V3 relative depth", fontsize=fontsize_label)
ax.set_ylabel("DinoV2 relative depth", fontsize=fontsize_label)
ax.set_title("DeepSeek-V3 vs DinoV2", fontsize=12)
ax.tick_params(axis='both', labelsize=fontsize_label)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Information Imbalance (avg)", fontsize=fontsize_label - 2)

# Extra heatmaps (Llama×IGPT, Llama×DinoV2, CLIP-Text×CLIP-Image)
for panel_idx, (sweep_name, hm) in enumerate(extra_heatmaps.items(), start=2):
    ax = axes[panel_idx]
    x_rel = hm['la'] / N_LAYERS[hm['model_a']]
    y_rel = hm['lb'] / N_LAYERS[hm['model_b']]
    im = ax.pcolormesh(x_rel, y_rel, hm['avg'].T, shading='auto', cmap='viridis_r')
    ax.plot(x_rel[hm['min'][0]], y_rel[hm['min'][1]], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=0.5)
    ax.set_xlabel(f"{hm['label_a']} relative depth", fontsize=fontsize_label)
    ax.set_ylabel(f"{hm['label_b']} relative depth", fontsize=fontsize_label)
    ax.set_title(f"{hm['label_a']} vs {hm['label_b']}", fontsize=12)
    ax.tick_params(axis='both', labelsize=fontsize_label)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Information Imbalance (avg)", fontsize=fontsize_label - 2)

plt.tight_layout()
plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/heatmap_ii.pdf"), dpi=300)


# In[ ]:


# ==========================================================================
# Aggregation comparison: DinoV2 image-side token pooling strategies
# Parallels the text-side last/concat/avg comparison (Fig. last-concat-avg)
# ==========================================================================

AGG_RESULTS = Path("data", "aggregation_test")

# Average variants: (label, subfolder, n_dino_layers)
AVG_VARIANTS = [
    ("Last 1 token",     "deepseek_dino_avg_last1",    24),
    ("CLS only",         "deepseek_dino_avg_cls",      24),
    ("Last 10 tokens",   "deepseek_dino_avg_last10",   24),
    ("Last 50 tokens",   "deepseek_dino_avg_last50",   24),
    ("Last 100 tokens",  "deepseek_dino_avg_last100",  24),
    ("All 257 tokens",   "deepseek_dino_avg_all",      24),
]

# Concatenation variants
CONCAT_VARIANTS = [
    ("Last 1 (baseline)", "deepseek_dino_concat_last1",   24),
    ("Concat 10 tokens",  "deepseek_dino_concat_last10",  24),
    ("Concat 50 tokens",  "deepseek_dino_concat_last50",  24),
    ("Concat 100 tokens", "deepseek_dino_concat_last100", 24),
]


def load_agg_ii(base_dir, variant_dir, n_layers):
    """Load II results for an aggregation variant, return (forward_mean, forward_se) arrays."""
    result_dir = make_path(Path(base_dir, variant_dir, "l2_centered_normalized"))
    fwd_means = np.zeros(n_layers)
    fwd_ses = np.zeros(n_layers)
    for i in range(1, n_layers + 1):
        data = load_file(Path(result_dir, f"layer_{i}.safetensors"))
        ranks = data["ranks"]
        cards = data["cards"]
        n_rows = len(ranks)
        n_col = sample_size - n_rows
        total_r = float(ranks.sum())
        total_c = float(cards.sum())
        if total_c > 0:
            ii_full = 2.0 * total_r / total_c / (n_col - 1)
        else:
            ii_full = float("nan")
        # Jackknife SE
        loo_r = total_r - ranks
        loo_c = total_c - cards
        ii_loo = np.where(loo_c > 0, 2.0 * loo_r / loo_c / (n_col - 1), np.nan)
        pseudo = n_rows * ii_full - (n_rows - 1) * ii_loo
        se = float(np.nanstd(pseudo, ddof=1) / np.sqrt(n_rows))
        fwd_means[i - 1] = ii_full
        fwd_ses[i - 1] = se
    return fwd_means, fwd_ses


# In[ ]:


# Load all variants
try:
    agg_base = make_path(AGG_RESULTS)
    HAS_AGG = True
except (AssertionError, AssertionError):
    HAS_AGG = False
    print("WARNING: Aggregation test results not found, skipping aggregation plot")

if HAS_AGG:
    avg_data = {}
    for label, subdir, nl in AVG_VARIANTS:
        try:
            avg_data[label] = load_agg_ii(agg_base, subdir, nl)
        except Exception as e:
            print(f"  Skipping avg variant '{label}': {e}")

    concat_data = {}
    for label, subdir, nl in CONCAT_VARIANTS:
        try:
            concat_data[label] = load_agg_ii(agg_base, subdir, nl)
        except Exception as e:
            print(f"  Skipping concat variant '{label}': {e}")

    print(f"Loaded {len(avg_data)} average variants, {len(concat_data)} concat variants")


# In[ ]:


if HAS_AGG and (avg_data or concat_data):
    x_dino = np.linspace(0, 1, N_LAYERS['dino'])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    # Color gradient: fewer tokens = lighter, more tokens = darker
    avg_colors = ['#D4A5A5', '#C0392B', '#F39C12', '#3498DB', '#8E44AD', '#0A5EB0']
    concat_colors = ['#D4A5A5', '#F39C12', '#3498DB', '#8E44AD']

    # Panel a) Average
    ax = axes[0]
    ax.text(-0.12, 1.08, "a)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    for i, (label, (means, ses)) in enumerate(avg_data.items()):
        color = avg_colors[i % len(avg_colors)]
        ax.plot(x_dino, means, label=label, color=color)
        ax.fill_between(x_dino, means - ses, means + ses, alpha=0.15, color=color)
    ax.set_xlabel("DinoV2 relative depth", fontsize=fontsize_label)
    ax.set_ylabel("Information Imbalance", fontsize=fontsize_label)
    ax.set_title("Average pooling", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=fontsize_label)

    # Panel b) Concatenation
    ax = axes[1]
    ax.text(-0.12, 1.08, "b)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    for i, (label, (means, ses)) in enumerate(concat_data.items()):
        color = concat_colors[i % len(concat_colors)]
        ax.plot(x_dino, means, label=label, color=color)
        ax.fill_between(x_dino, means - ses, means + ses, alpha=0.15, color=color)
    # Also plot average-all as reference
    if "All 257 tokens" in avg_data:
        ref_means, ref_ses = avg_data["All 257 tokens"]
        ax.plot(x_dino, ref_means, label="Avg all (ref)", color='#0A5EB0', linestyle='--')
        ax.fill_between(x_dino, ref_means - ref_ses, ref_means + ref_ses, alpha=0.1, color='#0A5EB0')
    ax.set_xlabel("DinoV2 relative depth", fontsize=fontsize_label)
    ax.set_title("Concatenation", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=fontsize_label)

    plt.tight_layout()
    plt.savefig(Path("/home/masca/GitHub/deepseek-hidden-states/plots/aggregation_comparison.pdf"), dpi=300)
    print("Saved: plots/aggregation_comparison.pdf")


