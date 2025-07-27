#!/usr/bin/env python
"""
Quick verification script to check that all data files exist before running the main analysis
"""

from pathlib import Path
import sys

def verify_data_exists():
    """Verify all required data directories and files exist"""
    
    # Define paths
    paths_to_check = {
        "DeepSeek": Path("/home/laios/activations/flickr/deepseek_all"),
        "DinoV2": Path("/home/laios/image-information-imbalance/facebook-dinov2-large"),
        "CLIP": Path("/home/laios/image-information-imbalance/openai-clip-vit-large-patch14"),
    }
    
    print("Verifying data directories...")
    print("=" * 60)
    
    all_good = True
    
    for name, path in paths_to_check.items():
        if path.exists():
            # Count files
            if name == "DeepSeek":
                files = list(path.glob("chunk_*.pkl"))
            elif name == "CLIP":
                image_files = list(path.glob("image-chunk_*.pkl"))
                text_files = list(path.glob("text-chunk_*.pkl"))
                files = image_files + text_files
                print(f"✓ {name}: {path}")
                print(f"  - Image chunks: {len(image_files)}")
                print(f"  - Text chunks: {len(text_files)}")
                continue
            else:
                files = list(path.glob("chunk_*.pkl"))
            
            print(f"✓ {name}: {path}")
            print(f"  - Files found: {len(files)}")
        else:
            print(f"✗ {name}: {path} NOT FOUND")
            all_good = False
    
    print("=" * 60)
    
    # Check for CLIP text/image separation
    clip_path = Path("/home/laios/image-information-imbalance/openai-clip-vit-large-patch14")
    if clip_path.exists():
        print("\nChecking CLIP file structure:")
        image_chunks = list(clip_path.glob("image-chunk_*.pkl"))
        text_chunks = list(clip_path.glob("text-chunk_*.pkl"))
        
        if not image_chunks:
            print("⚠️  WARNING: No image-chunk_*.pkl files found for CLIP")
            print("   You may need to rename chunk_*.pkl to image-chunk_*.pkl")
            regular_chunks = list(clip_path.glob("chunk_*.pkl"))
            if regular_chunks:
                print(f"   Found {len(regular_chunks)} chunk_*.pkl files that might be image chunks")
        
        if not text_chunks:
            print("⚠️  WARNING: No text-chunk_*.pkl files found for CLIP")
            print("   You may need to extract CLIP text embeddings separately")
    
    print("\nOutput directories:")
    output_base = Path("/home/laios/image-information-imbalance/results")
    repo_plots = Path("/home/mascaretti/deepseek-hidden-states/plots")
    
    print(f"Results will be saved to: {output_base}")
    print(f"Plots will be saved to: {repo_plots}")
    
    if not output_base.exists():
        print(f"  - Creating {output_base}")
        output_base.mkdir(parents=True, exist_ok=True)
    
    if all_good:
        print("\n✓ All data directories found! Ready to run analysis.")
        return 0
    else:
        print("\n✗ Some data directories are missing. Please check the paths.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_data_exists())
