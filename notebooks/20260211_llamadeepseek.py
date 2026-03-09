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

TARGET_LAYERS = {
    'deepseek' : 52,
    'llama' : 20,
    'dino' : 24,
    'clip_text' : 12,
    'igpt' : 20
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
        base_dir = Path("data", "results-mean")
    AtoB = make_path(Path(base_dir, f"{to_model}_{from_model}", "l2_centered_normalized")) if hamming is False else make_path(Path(base_dir, f"{to_model}_{from_model}", "hamming"))
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
base_dir = make_path(Path("data", "results-mean"))
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


imagenet_path = make_path(Path("data", "imagenet"))


# In[18]:


dino_igpt_layer = 39
imagenet_dino_igpt_fwd = []
imagenet_dino_igpt_fwd.append(np.load(Path(imagenet_path, "dino_ii_mean.npy"))[dino_igpt_layer, 1:])
imagenet_dino_igpt_fwd.append(np.load(Path(imagenet_path, "dino_ii_std.npy"))[dino_igpt_layer, 1:])


# In[19]:


igpt_layer_dino = 20
imagenet_igpt_dino_fwd = []
imagenet_igpt_dino_fwd.append(np.load(Path(imagenet_path, "igpt_ii_mean.npy"))[igpt_layer_dino, 1:])
imagenet_igpt_dino_fwd.append(np.load(Path(imagenet_path, "igpt_ii_std.npy"))[igpt_layer_dino, 1:])

# Cross-model ImageNet: DinoV2-huge (40 layers) ↔ ImageGPT (49 layers)
# cross_dino_to_igpt[dino_layer, igpt_layer] = II(dino → igpt)
# cross_igpt_to_dino[igpt_layer, dino_layer] = II(igpt → dino)
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


# In[20]:


ylim = (0.0, 1)
figsize = (4.5, 4.5)
fontsize_label = 15
x = {key:np.linspace(0, 1, N_LAYERS[key]) for key in N_LAYERS.keys()}


# In[21]:


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
x_igpt_imagenet = np.arange(48) / 48  # igpt 49 layers, skip layer 0 → 48 points
x_dino_imagenet = np.arange(39) / 39  # dino-huge 40 layers, skip layer 0 → 39 points

fig, axes = plt.subplots(1, 2, figsize=(3.5 * figsize[0], figsize[1]), sharey=True)

for i, ax in enumerate(axes):
    ax.text(-0.12, 1.08, f"{chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

# a) ImageNet1k — within-model
ax = axes[0]
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
title = f"Target: {MODEL_NAMES['igpt']}[{TARGET_LAYERS['igpt']}]"
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
title = f"Target: {MODEL_NAMES['dino']}[{TARGET_LAYERS['dino']}]"
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
title = f"Target: {MODEL_NAMES['llama']}[{TARGET_LAYERS['llama']}]"
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
label_forward = f"{MODEL_NAMES['deepseek']}[{TARGET_LAYERS['deepseek']}] → {MODEL_NAMES['dino']}"
ax.plot(x['dino'], dino_fwd[0], '-', color=MODEL_COLORS['dino'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['dino'], dino_fwd[0] - dino_fwd[1], dino_fwd[0] + dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
label_backward = f"{MODEL_NAMES['dino']} → {MODEL_NAMES['deepseek']}[{TARGET_LAYERS['deepseek']}]"
ax.plot(x['dino'], dino_bwd[0], '--', color=MODEL_COLORS['dino'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['dino'], dino_bwd[0] - dino_bwd[1], dino_bwd[0] + dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['dino'])
ax.vlines(x = TARGET_LAYERS['dino']/N_LAYERS['dino'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_dino_bwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_dino_fwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="red")

ax.set_title("Image-Captions on Flickr30k", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')



ax = axes[1]
label_forward = f"{MODEL_NAMES['dino']}[{TARGET_LAYERS['dino']}] → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_dino_fwd[0], '-', color=MODEL_COLORS['deepseek'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['deepseek'], ds_dino_fwd[0] - ds_dino_fwd[1], ds_dino_fwd[0] + ds_dino_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['dino']}[{TARGET_LAYERS['dino']}]"
ax.plot(x['deepseek'], ds_dino_bwd[0], '--', color=MODEL_COLORS['deepseek'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_dino_bwd[0] - ds_dino_bwd[1], ds_dino_bwd[0] + ds_dino_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.vlines(x = TARGET_LAYERS['deepseek']/N_LAYERS['deepseek'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_dino_bwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_dino_fwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="red")
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
label_forward = f"{MODEL_NAMES['deepseek']}[{TARGET_LAYERS['deepseek']}] → {MODEL_NAMES['igpt']}"
ax.plot(x['igpt'], igpt_fwd[0], '-', color=MODEL_COLORS['igpt'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['igpt'], igpt_fwd[0] - igpt_fwd[1], igpt_fwd[0] + igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
label_backward = f"{MODEL_NAMES['igpt']} → {MODEL_NAMES['deepseek']}[{TARGET_LAYERS['deepseek']}]"
ax.plot(x['igpt'], igpt_bwd[0], '--', color=MODEL_COLORS['igpt'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['igpt'], igpt_bwd[0] - igpt_bwd[1], igpt_bwd[0] + igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['igpt'])
ax.vlines(x = TARGET_LAYERS['igpt']/N_LAYERS['igpt'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_igpt_bwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_igpt_fwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="red")

## ax.set_title("Image-Captions on Flickr30k", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')



ax = axes[1]
label_forward = f"{MODEL_NAMES['igpt']}[{TARGET_LAYERS['igpt']}] → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_igpt_fwd[0], '-', color=MODEL_COLORS['deepseek'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['deepseek'], ds_igpt_fwd[0] - ds_igpt_fwd[1], ds_igpt_fwd[0] + ds_igpt_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['igpt']}[{TARGET_LAYERS['igpt']}]"
ax.plot(x['deepseek'], ds_igpt_bwd[0], '--', color=MODEL_COLORS['deepseek'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_igpt_bwd[0] - ds_igpt_bwd[1], ds_igpt_bwd[0] + ds_igpt_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.vlines(x = TARGET_LAYERS['deepseek']/N_LAYERS['deepseek'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_igpt_bwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_igpt_fwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="red")
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
label_forward = f"{MODEL_NAMES['deepseek']}[{TARGET_LAYERS['deepseek']}] → {MODEL_NAMES['llama']}"
ax.plot(x['llama'], llama_fwd[0], '-', color=MODEL_COLORS['llama'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['llama'], llama_fwd[0] - llama_fwd[1], llama_fwd[0] + llama_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['llama'])
label_backward = f"{MODEL_NAMES['llama']} → {MODEL_NAMES['deepseek']}[{TARGET_LAYERS['deepseek']}]"
ax.plot(x['llama'], llama_bwd[0], '--', color=MODEL_COLORS['llama'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['llama'], llama_bwd[0] - llama_bwd[1], llama_bwd[0] + llama_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['llama'])
ax.vlines(x = TARGET_LAYERS['llama']/N_LAYERS['llama'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_llama_bwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_llama_fwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="red")

## ax.set_title("Image-Captions on Flickr30k", fontsize=12)
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')



ax = axes[1]
label_forward = f"{MODEL_NAMES['llama']}[{TARGET_LAYERS['llama']}] → {MODEL_NAMES['deepseek']}"
ax.plot(x['deepseek'], ds_llama_fwd[0], '-', color=MODEL_COLORS['deepseek'], 
        label=label_forward, linewidth=2, marker='o', 
        markersize=3, markeredgecolor='black', markeredgewidth=0.3)    
ax.fill_between(x['deepseek'], ds_llama_fwd[0] - ds_llama_fwd[1], ds_llama_fwd[0] + ds_llama_fwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
label_backward = f"{MODEL_NAMES['deepseek']} → {MODEL_NAMES['llama']}[{TARGET_LAYERS['llama']}]"
ax.plot(x['deepseek'], ds_llama_bwd[0], '--', color=MODEL_COLORS['deepseek'], label = label_backward, linewidth=2, markersize=3, markeredgecolor='black', markeredgewidth=0.3)
ax.fill_between(x['deepseek'], ds_llama_bwd[0] - ds_llama_bwd[1], ds_llama_bwd[0] + ds_llama_bwd[1], alpha=0.2, linewidth=0, color=MODEL_COLORS['deepseek'])
ax.vlines(x = TARGET_LAYERS['deepseek']/N_LAYERS['deepseek'], ymin=ylim[0], ymax=ylim[1], alpha=0.2, color="black")
ax.hlines(y = ds_llama_bwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="black")
ax.hlines(y = ds_llama_fwd[0][TARGET_LAYERS['deepseek']], xmin=-0., xmax=1.05, alpha=0.6, color="red")
ax.set_xlabel("Relative depth", fontsize=fontsize_label)
ax.set_ylabel("", fontsize=fontsize_label)
ax.tick_params(axis='both', labelsize=fontsize_label)
ax.set_ylim(ylim)
ax.legend(fontsize=fontsize_label, loc='best')


# In[ ]:




