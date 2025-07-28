#!/usr/bin/env python
"""
Comprehensive multi-layer analysis script
Runs all model pair comparisons with multiple layers
Includes bootstrapping and plotting
"""

import os
import sys
from pathlib import Path
import subprocess
import time
import json

# Configuration for all model pairs
MODEL_PAIRS = [
    {
        'name': 'deepseek_clip_text',
        'model_a': 'deepseek',
        'model_b': 'clip_text',
        'path_a': '/home/laios/activations/flickr/deepseek_all',
        'path_b': '/home/laios/image-information-imbalance/openai-clip-vit-large-patch14',
        'layers_a': {'early': 10, 'middle': 30, 'semantic': 52},
        'n_layers_b': 13,
        'tokens': 9,
        'comparison_type': 'text_text'
    },
    {
        'name': 'deepseek_clip_image',
        'model_a': 'deepseek',
        'model_b': 'clip_image',
        'path_a': '/home/laios/activations/flickr/deepseek_all',
        'path_b': '/home/laios/image-information-imbalance/openai-clip-vit-large-patch14',
        'layers_a': {'early': 10, 'middle': 30, 'semantic': 52},
        'n_layers_b': 25,
        'tokens': 200,
        'comparison_type': 'text_image'
    },
    {
        'name': 'deepseek_dino',
        'model_a': 'deepseek', 
        'model_b': 'dino',
        'path_a': '/home/laios/activations/flickr/deepseek_all',
        'path_b': '/home/laios/image-information-imbalance/facebook-dinov2-large',
        'layers_a': {'early': 10, 'middle': 30, 'semantic': 52},
        'n_layers_b': 25,
        'tokens': 200,
        'comparison_type': 'text_image'
    },
    {
        'name': 'clip_text_clip_image',
        'model_a': 'clip_text',
        'model_b': 'clip_image',
        'path_a': '/home/laios/image-information-imbalance/openai-clip-vit-large-patch14',
        'path_b': '/home/laios/image-information-imbalance/openai-clip-vit-large-patch14',
        'layers_a': {'early': 1, 'middle': 6, 'last': 12},
        'n_layers_b': 25,
        'tokens': 30,
        'comparison_type': 'text_image'
    },
    {
        'name': 'dino_clip_image',
        'model_a': 'dino',
        'model_b': 'clip_image',
        'path_a': '/home/laios/image-information-imbalance/facebook-dinov2-large',
        'path_b': '/home/laios/image-information-imbalance/openai-clip-vit-large-patch14',
        'layers_a': {'early': 1, 'middle': 12, 'last': 24},
        'n_layers_b': 25,
        'tokens': 200,
        'comparison_type': 'image_image'
    }
]


def create_multilayer_script(model_pair):
    """Create a custom multi-layer analysis script for each model pair"""
    
    script_content = f'''#!/usr/bin/env python
"""
Multi-layer analysis for {model_pair['model_a']} vs {model_pair['model_b']}
Auto-generated script
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import jax.numpy as jnp
from tqdm import tqdm
import pickle
import io

from safetensors.numpy import save_file
from information_imbalance_module import hamming_distance, build_information_matrix
from utils import bf16_torch_to_jax, flatten_tokens_features, list_folder, binarize
import jax

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# Data collection functions
def collect_deepseek_data(input_path, n_loaded_tokens=9, subset_size=5000):
    """Collect DeepSeek text data"""
    files = list_folder(input_path, desc="chunk_")
    all_hidden_states = defaultdict(list)

    file_counter = 0
    for file in tqdm(files, desc="Collect DeepSeek File", leave=False):
        if file_counter >= subset_size:
            break
        with open(input_path + "/" + file.name, 'rb') as f:
            data = pickle.load(f)
        for _, sentence in enumerate(data):
            hidden_states = sentence['meta_info']['hidden_states'][0]
            if hidden_states.shape[1] < n_loaded_tokens:
                print(f"Warning: Skipping sentence with only {{hidden_states.shape[1]}} tokens")
                continue
            hidden_states = hidden_states[:, -n_loaded_tokens:]
            file_counter += 1

            for layer_idx, layer_tensor in enumerate(hidden_states.split(1, dim=0)):
                layer_tensor = layer_tensor.squeeze(0)
                all_hidden_states[f"layer_{{layer_idx}}"].append(layer_tensor)

    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.stack(tensors)[:subset_size]
    
    return all_hidden_states

def collect_image_data(input_path, n_files, last_tokens=200):
    """Collect image model data"""
    files = list_folder(input_path, desc="chunk_")[:n_files]
    all_hidden_states = defaultdict(list)
    for file in tqdm(files, desc="Collect image data"):
        with open(Path(input_path, file.name), 'rb') as f:
            batch_activations = CPU_Unpickler(f).load()
        for layer_key in batch_activations.keys():
            all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.cat(tensors, dim=0)
    return all_hidden_states

def collect_clip_text_data(input_path, n_files, last_tokens=30):
    """Collect CLIP text data"""
    files = list_folder(input_path, desc="text-chunk_")[:n_files]
    all_hidden_states = defaultdict(list)
    for file in tqdm(files, desc="Collect CLIP text data"):
        with open(Path(input_path, file.name), 'rb') as f:
            batch_activations = CPU_Unpickler(f).load()
        for layer_key in batch_activations.keys():
            all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.cat(tensors, dim=0)
    return all_hidden_states

def collect_clip_image_data(input_path, n_files, last_tokens=200):
    """Collect CLIP image data"""
    files = list_folder(input_path, desc="image-chunk_")[:n_files]
    all_hidden_states = defaultdict(list)
    for file in tqdm(files, desc="Collect CLIP image data"):
        with open(Path(input_path, file.name), 'rb') as f:
            batch_activations = CPU_Unpickler(f).load()
        for layer_key in batch_activations.keys():
            all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
    for layer, tensors in all_hidden_states.items():
        all_hidden_states[layer] = torch.cat(tensors, dim=0)
    return all_hidden_states

# Main analysis
def main():
    print("Multi-layer analysis: {model_pair['model_a']} vs {model_pair['model_b']}")
    print("="*60)
    
    subset_size = 5000
    n_files = 500
    
    # Load data based on model types
'''

    # Add data loading logic based on model types
    if model_pair['model_a'] == 'deepseek':
        script_content += f'''
    print("Loading DeepSeek data...")
    all_activations_A = collect_deepseek_data("{model_pair['path_a']}", n_loaded_tokens=9, subset_size=subset_size)
'''
    elif model_pair['model_a'] == 'clip_text':
        script_content += f'''
    print("Loading CLIP text data...")
    all_activations_A = collect_clip_text_data("{model_pair['path_a']}", n_files=n_files, last_tokens=30)
'''
    elif model_pair['model_a'] in ['dino', 'clip_image']:
        script_content += f'''
    print("Loading {model_pair['model_a']} data...")
    all_activations_A = collect_image_data("{model_pair['path_a']}", n_files=n_files, last_tokens=200)
'''

    # Model B loading
    if model_pair['model_b'] == 'clip_text':
        script_content += f'''
    print("Loading CLIP text data...")
    all_activations_B = collect_clip_text_data("{model_pair['path_b']}", n_files=n_files, last_tokens=30)
'''
    elif model_pair['model_b'] == 'clip_image':
        script_content += f'''
    print("Loading CLIP image data...")
    all_activations_B = collect_clip_image_data("{model_pair['path_b']}", n_files=n_files, last_tokens=200)
'''
    elif model_pair['model_b'] in ['dino']:
        script_content += f'''
    print("Loading {model_pair['model_b']} data...")
    all_activations_B = collect_image_data("{model_pair['path_b']}", n_files=n_files, last_tokens=200)
'''

    # Add computation logic
    layers_str = str(model_pair['layers_a'])
    script_content += f'''
    
    # Verify data
    print(f"Model A samples: {{all_activations_A['layer_0'].shape[0]}}")
    print(f"Model B samples: {{all_activations_B['layer_0'].shape[0]}}")
    
    # Define layers to analyze
    layers_A = {layers_str}
    
    # Process each layer
    for layer_name, layer_id in layers_A.items():
        print(f"\\nProcessing {{layer_name}} layer ({{layer_id}})...")
        
        output_dir = Path("/home/laios/image-information-imbalance/results/{model_pair['name']}-multilayer-5000") / f"{model_pair['model_a']}_{{layer_name}}_layer{{layer_id}}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get activations for this layer
        layer_A = f"layer_{{layer_id}}"
        if layer_A not in all_activations_A:
            print(f"WARNING: Layer {{layer_A}} not found!")
            continue
            
        activations_A = all_activations_A[layer_A]
        sample_size = activations_A.shape[0]
        
        # Initialize information matrix computation
        key = jax.random.key(42)
        key, subkey = jax.random.split(key)
        information_matrix_fn = build_information_matrix(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)
        
        # Process model A activations based on type
        if "{model_pair['comparison_type']}" in ["text_text", "text_image"]:
            # Text model: take last tokens
            if hasattr(activations_A, 'dtype') and activations_A.dtype == torch.bfloat16:
                act_A = bf16_torch_to_jax(activations_A[:, -{model_pair['tokens']}:, :])
            else:
                act_A = jnp.array(activations_A[:, -{model_pair['tokens']}:, :])
        else:
            # Image model
            act_A = jnp.array(activations_A)
        
        act_A = flatten_tokens_features(act_A)
        act_A = binarize(act_A)
        
        # Process each layer of model B
        n_layers_B = {model_pair['n_layers_b']}
        for layer_b_id in tqdm(range(1, n_layers_B), desc="Processing model B layers"):
            layer_B = f"layer_{{layer_b_id}}"
            if layer_B not in all_activations_B:
                continue
                
            act_B_torch = all_activations_B[layer_B]
            
            # Convert based on dtype
            if hasattr(act_B_torch, 'dtype') and act_B_torch.dtype == torch.bfloat16:
                act_B = bf16_torch_to_jax(act_B_torch)
            else:
                act_B = jnp.array(act_B_torch)
            
            act_B = flatten_tokens_features(act_B)
            act_B = binarize(act_B)
            
            # Compute information matrix
            inf_imb = information_matrix_fn(act_A, act_B)
            
            # Save results
            save_file(inf_imb, output_dir / f"{{layer_B}}.safetensors")
        
        print(f"Completed {{layer_name}} layer")
    
    print("\\nAll layers processed!")

if __name__ == "__main__":
    main()
'''
    
    return script_content


def run_bootstrap_for_pair(model_pair, base_results_dir):
    """Run bootstrap analysis for all layers of a model pair"""
    
    multilayer_dir = Path(base_results_dir) / f"{model_pair['name']}-multilayer-5000"
    if not multilayer_dir.exists():
        print(f"WARNING: No results found for {model_pair['name']}")
        return
    
    output_dir = Path(base_results_dir) / "bootstrap" / f"{model_pair['name']}-multilayer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run bootstrap_multilayer.py
    cmd = [
        "python", "bootstrap_multilayer.py",
        "--input_dir", str(multilayer_dir),
        "--output_dir", str(output_dir),
        "--sample_size", "5000",
        "--n_layers", str(model_pair['n_layers_b'])
    ]
    
    print(f"Running bootstrap for {model_pair['name']}...")
    subprocess.run(cmd, check=True)


def generate_plots_for_pair(model_pair, results_dir, plots_dir):
    """Generate plots for a model pair"""
    
    # Create plotting script specifically for this pair
    plot_script = f'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load bootstrap results
bootstrap_dir = Path("{results_dir}/bootstrap/{model_pair['name']}-multilayer")
summary_file = bootstrap_dir / "multilayer_summary.json"

if not summary_file.exists():
    print("No bootstrap results found")
    exit(1)

with open(summary_file, 'r') as f:
    results = json.load(f)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

layers = {str(model_pair['layers_a'])}
colors = {{'early': '#87CEEB', 'middle': '#4682B4', 'last': '#000080', 'semantic': '#2E8B57'}}

for layer_name, layer_id in layers.items():
    key = f"{model_pair['model_a']}_{{layer_name}}_layer{{layer_id}}"
    if key in results:
        data = results[key]
        forward_mean = np.array(data['forward_mean'])
        backward_mean = np.array(data['backward_mean'])
        n_layers = len(forward_mean)
        x = np.linspace(0., 1., n_layers)
        
        color = colors.get(layer_name, '#666666')
        
        ax1.plot(x, forward_mean, '-o', color=color, label=f'Layer {{layer_id}}', 
                markersize=3, markevery=5)
        ax2.plot(x, backward_mean, '-s', color=color, label=f'Layer {{layer_id}}',
                markersize=3, markevery=5)

ax1.set_title('{model_pair['model_a']} → {model_pair['model_b']}')
ax1.set_xlabel('Target Layer (normalized)')
ax1.set_ylabel('Information Imbalance')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_title('{model_pair['model_b']} → {model_pair['model_a']}')
ax2.set_xlabel('Target Layer (normalized)')
ax2.set_ylabel('Information Imbalance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('{model_pair['model_a']} vs {model_pair['model_b']} Multi-Layer Analysis\\n5000 samples')
plt.tight_layout()

output_path = Path("{plots_dir}") / "{model_pair['name']}_multilayer.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved plot: {{output_path}}")
'''
    
    # Save and run plot script
    plot_script_path = Path(f"plot_{model_pair['name']}_temp.py")
    with open(plot_script_path, 'w') as f:
        f.write(plot_script)
    
    subprocess.run(["python", str(plot_script_path)], check=True)
    plot_script_path.unlink()  # Clean up


def main():
    """Run complete multi-layer analysis pipeline"""
    
    print("="*80)
    print("COMPREHENSIVE MULTI-LAYER ANALYSIS")
    print("="*80)
    
    base_results_dir = "/home/laios/image-information-imbalance/results"
    plots_dir = "/home/mascaretti/deepseek-hidden-states/plots/multilayer"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Track timing
    start_time = time.time()
    
    for i, model_pair in enumerate(MODEL_PAIRS):
        print(f"\n[{i+1}/{len(MODEL_PAIRS)}] Processing {model_pair['name']}")
        print("-"*60)
        
        try:
            # Step 1: Create and run analysis script
            print("Step 1: Running multi-layer analysis...")
            script_path = Path(f"multilayer_{model_pair['name']}_temp.py")
            script_content = create_multilayer_script(model_pair)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run the analysis
            subprocess.run(["python", str(script_path)], check=True)
            script_path.unlink()  # Clean up
            
            # Step 2: Run bootstrap
            print("\nStep 2: Running bootstrap analysis...")
            run_bootstrap_for_pair(model_pair, base_results_dir)
            
            # Step 3: Generate plots
            print("\nStep 3: Generating plots...")
            generate_plots_for_pair(model_pair, base_results_dir, plots_dir)
            
            print(f"\n✓ Completed {model_pair['name']}")
            
        except Exception as e:
            print(f"\n✗ Error processing {model_pair['name']}: {e}")
            continue
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE")
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Results saved to: {base_results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print("="*80)
    
    # Create final summary plot comparing all pairs
    create_summary_comparison_plot(base_results_dir, plots_dir)


def create_summary_comparison_plot(results_dir, plots_dir):
    """Create a summary plot comparing all model pairs"""
    
    print("\nCreating summary comparison plot...")
    
    # This would create a comprehensive figure showing all comparisons
    # Implementation depends on specific visualization needs
    pass


if __name__ == "__main__":
    main()
