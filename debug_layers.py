#!/usr/bin/env python
"""
Quick debug script to check available layers in the data files
"""

import pickle
from pathlib import Path
import torch
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def check_layers(file_path, file_type="auto"):
    """Check layers in a pickle file"""
    print(f"\nChecking: {file_path}")
    
    if not Path(file_path).exists():
        print(f"  ERROR: File not found!")
        return
    
    try:
        with open(file_path, 'rb') as f:
            if file_type == "deepseek":
                data = pickle.load(f)
                # DeepSeek has different structure
                if isinstance(data, list) and len(data) > 0:
                    hidden_states = data[0]['meta_info']['hidden_states'][0]
                    n_layers = hidden_states.shape[0]
                    print(f"  DeepSeek format: {n_layers} layers (0-{n_layers-1})")
                    print(f"  Shape: {hidden_states.shape}")
            else:
                data = CPU_Unpickler(f).load()
                layers = sorted(data.keys())
                print(f"  Layers found: {len(layers)}")
                print(f"  Layer names: {layers[:5]}...{layers[-5:]}")
                if layers:
                    sample_shape = data[layers[0]].shape
                    print(f"  Sample shape: {sample_shape}")
                    print(f"  Data type: {data[layers[0]].dtype}")
    except Exception as e:
        print(f"  ERROR: {e}")

def main():
    print("Checking layer structure in data files...")
    print("=" * 60)
    
    # Check DeepSeek
    deepseek_path = Path("/home/laios/activations/flickr/deepseek_all")
    if deepseek_path.exists():
        files = sorted(deepseek_path.glob("chunk_*.pkl"))
        if files:
            check_layers(files[0], "deepseek")
    
    # Check DinoV2
    dino_path = Path("/home/laios/image-information-imbalance/facebook-dinov2-large")
    if dino_path.exists():
        files = sorted(dino_path.glob("chunk_*.pkl"))
        if files:
            check_layers(files[0], "image")
    
    # Check CLIP image
    clip_path = Path("/home/laios/image-information-imbalance/openai-clip-vit-large-patch14")
    if clip_path.exists():
        # Check image chunks
        files = sorted(clip_path.glob("image-chunk_*.pkl"))
        if files:
            check_layers(files[0], "image")
        
        # Check text chunks
        files = sorted(clip_path.glob("text-chunk_*.pkl"))
        if files:
            check_layers(files[0], "text")
    
    print("\n" + "=" * 60)
    print("Note: Most models have layers 0-23 (24 total layers)")
    print("The last layer (23) is typically the most semantic")

if __name__ == "__main__":
    main()
