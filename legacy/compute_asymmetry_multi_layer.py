#!/usr/bin/env python
"""
Enhanced asymmetry computation with support for multiple layer pairs
Priority: DinoV2 vs CLIP image
"""

from compute_asymmetry_unified import *
import json

# Layer configurations for each model
LAYER_CONFIGS = {
    "deepseek": {
        "total": 62,
        "first": 10,
        "middle": 30,
        "semantic": 52,
        "last": 61
    },
    "dino": {
        "total": 25,
        "first": 1,
        "middle": 12,
        "last": 24
    },
    "clip_image": {
        "total": 25,
        "first": 1,
        "middle": 12,
        "last": 24
    },
    "clip_text": {
        "total": 13,
        "first": 1,
        "middle": 6,
        "last": 12
    }
}


def compute_inf_imb_matrix_multi_layer(
        all_activations_A, 
        all_activations_B, 
        output_folder,
        layers_A,  # List of layers to use from model A
        model_a_name="model_a",
        model_b_name="model_b",
        comparison_type="image_image"
):
    """
    Compute information imbalance matrix for multiple source layers
    """
    key = jax.random.key(42)
    print(f"Initial key: {key}")
    
    n_layers_B = len(list(all_activations_B.keys()))
    cols = [f"layer_{i}" for i in range(1, n_layers_B)]

    sample_size = all_activations_A["layer_1"].shape[0]
    print("Sample size: ", sample_size)
    sample_size_B = all_activations_B["layer_1"].shape[0]
    assert sample_size == sample_size_B
    
    # Process each source layer
    for layer_name, layer_id in layers_A.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_a_name} {layer_name} (layer {layer_id})")
        print(f"{'='*60}")
        
        layer_A = f"layer_{layer_id}"
        if layer_A not in all_activations_A:
            print(f"WARNING: Layer {layer_A} not found in {model_a_name}. Skipping.")
            continue
        
        # Create subfolder for this layer
        layer_output = Path(output_folder) / f"{model_a_name}_{layer_name}_layer{layer_id}"
        layer_output.mkdir(parents=True, exist_ok=True)
        
        key, subkey = jax.random.split(key)
        information_matrix_fn = build_information_matrix(
            key=subkey, sample_size=sample_size, distance_fn=hamming_distance
        )
        
        activations_A = all_activations_A[layer_A]
        
        for layer_B in (pbar := tqdm(cols, desc=f"Computing {layer_name} vs layer_B", leave=False)):
            # Handle preprocessing based on model type
            if model_a_name == "deepseek" or model_a_name == "clip_text":
                if activations_A.dim() == 3 and activations_A.shape[1] > 1:
                    if hasattr(activations_A, 'dtype') and activations_A.dtype == torch.bfloat16:
                        act_A = bf16_torch_to_jax(activations_A[:, -9:, :])
                    else:
                        act_A = jnp.array(activations_A[:, -9:, :])
                else:
                    if hasattr(activations_A, 'dtype') and activations_A.dtype == torch.bfloat16:
                        act_A = bf16_torch_to_jax(activations_A)
                    else:
                        act_A = jnp.array(activations_A)
                act_A = flatten_tokens_features(act_A)
            else:
                act_A = jnp.array(activations_A)
                act_A = flatten_tokens_features(act_A)
            act_A = binarize(act_A)
            
            # Process B
            act_B_torch = all_activations_B[layer_B]
            if hasattr(act_B_torch, 'dtype') and act_B_torch.dtype == torch.bfloat16:
                act_B = bf16_torch_to_jax(act_B_torch)
            else:
                act_B = jnp.array(act_B_torch)
            
            act_B = flatten_tokens_features(act_B)
            act_B = binarize(act_B)

            inf_imb = information_matrix_fn(act_A, act_B)
            save_file(inf_imb, layer_output / f"{layer_B}.safetensors")
        
        # Save metadata
        metadata = {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "layer_a_name": layer_name,
            "layer_a_id": layer_id,
            "sample_size": sample_size,
            "comparison_type": comparison_type
        }
        with open(layer_output / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Completed {layer_name} layer {layer_id}")


def main_multi_layer():
    """
    Priority run: DinoV2 vs CLIP image with multiple layers
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=5000, 
                       help="Number of samples (use 500 for quick test)")
    parser.add_argument("--layers", type=str, default="first,middle,last",
                       help="Comma-separated layer names (first,middle,last,all)")
    args = parser.parse_args()
    
    # Configuration
    subset_size = args.subset_size
    n_files = int(subset_size / 10)
    
    print(f"Running DinoV2 vs CLIP image comparison")
    print(f"Subset size: {subset_size}")
    print(f"Layers: {args.layers}")
    
    # Load data
    print("\nLoading DinoV2...")
    all_activations_A = collect_deepseek_data(
        "/home/laios/image-information-imbalance/facebook-dinov2-large",
        n_files=n_files,
        last_tokens=200
    )
    
    print("\nLoading CLIP image...")
    all_activations_B = collect_clip_image_data(
        "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14",
        n_files=n_files,
        last_tokens=200
    )
    
    # Determine which layers to use
    if args.layers == "all":
        layers_A = LAYER_CONFIGS["dino"]
    else:
        layer_names = args.layers.split(",")
        layers_A = {name: LAYER_CONFIGS["dino"][name] 
                   for name in layer_names 
                   if name in LAYER_CONFIGS["dino"]}
    
    print(f"\nUsing DinoV2 layers: {layers_A}")
    
    # Compute for multiple layers
    output_base = f"/home/laios/image-information-imbalance/results/dino-clip-image-multilayer-{subset_size}"
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    compute_inf_imb_matrix_multi_layer(
        all_activations_A=all_activations_A,
        all_activations_B=all_activations_B,
        output_folder=output_base,
        layers_A=layers_A,
        model_a_name="dino",
        model_b_name="clip_image",
        comparison_type="image_image"
    )
    
    print(f"\n✓ All computations complete!")
    print(f"Results saved to: {output_base}")
    
    # Print summary
    print("\nSummary of outputs:")
    for layer_name, layer_id in layers_A.items():
        path = Path(output_base) / f"dino_{layer_name}_layer{layer_id}"
        if path.exists():
            n_files = len(list(path.glob("*.safetensors")))
            print(f"  - {layer_name} (layer {layer_id}): {n_files} files")


def main_other():
    """
    Run extra comparisons
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=5000, 
                       help="Number of samples (use 500 for quick test)")
    parser.add_argument("--layers", type=str, default="first,middle,last",
                       help="Comma-separated layer names (first,middle,last,all)")
    args = parser.parse_args()
    
    # Configuration
    subset_size = args.subset_size
    n_files = int(subset_size / 10)
    
    print(f"Running DeepSeek-V3 vs CLIP text comparison")
    print(f"Subset size: {subset_size}")
    print(f"Layers: {args.layers}")
    
    # Load data
    print("\nLoading DeepSeekV3...")
    all_activations_A = collect_image_model_data(
        "/home/laios/image-information-imbalance/facebook-dinov2-large",
        n_files=n_files,
        last_tokens=200
    )
    
    print("\nLoading CLIP text...")
    all_activations_B = collect_clip_image_data(
        "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14",
        n_files=n_files,
        last_tokens=200
    )
    
    # Determine which layers to use
    if args.layers == "all":
        layers_A = LAYER_CONFIGS["dino"]
    else:
        layer_names = args.layers.split(",")
        layers_A = {name: LAYER_CONFIGS["dino"][name] 
                   for name in layer_names 
                   if name in LAYER_CONFIGS["dino"]}
    
    print(f"\nUsing DinoV2 layers: {layers_A}")
    
    # Compute for multiple layers
    output_base = f"/home/laios/image-information-imbalance/results/dino-clip-image-multilayer-{subset_size}"
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    compute_inf_imb_matrix_multi_layer(
        all_activations_A=all_activations_A,
        all_activations_B=all_activations_B,
        output_folder=output_base,
        layers_A=layers_A,
        model_a_name="dino",
        model_b_name="clip_image",
        comparison_type="image_image"
    )
    
    print(f"\n✓ All computations complete!")
    print(f"Results saved to: {output_base}")
    
    # Print summary
    print("\nSummary of outputs:")
    for layer_name, layer_id in layers_A.items():
        path = Path(output_base) / f"dino_{layer_name}_layer{layer_id}"
        if path.exists():
            n_files = len(list(path.glob("*.safetensors")))
            print(f"  - {layer_name} (layer {layer_id}): {n_files} files")

if __name__ == "__main__":
    main_multi_layer()
