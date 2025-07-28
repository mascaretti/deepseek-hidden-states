#!/usr/bin/env python
"""
Verify that all multi-layer analyses completed successfully
"""

from pathlib import Path
import json

def check_results():
    """Check all expected results exist"""
    
    results_base = Path("/home/laios/image-information-imbalance/results")
    plots_base = Path("/home/mascaretti/deepseek-hidden-states/plots/multilayer")
    
    model_pairs = [
        ("deepseek_clip_text", ["early", "middle", "semantic"], 13),
        ("deepseek_clip_image", ["early", "middle", "semantic"], 25),
        ("deepseek_dino", ["early", "middle", "semantic"], 25),
        ("clip_text_clip_image", ["early", "middle", "last"], 25),
        ("dino_clip_image", ["early", "middle", "last"], 25)
    ]
    
    print("MULTI-LAYER ANALYSIS VERIFICATION")
    print("=" * 60)
    
    all_good = True
    
    for pair_name, layers, n_layers_b in model_pairs:
        print(f"\n{pair_name}:")
        print("-" * 40)
        
        # Check raw results
        multilayer_dir = results_base / f"{pair_name}-multilayer-5000"
        if multilayer_dir.exists():
            print(f"✓ Analysis directory exists")
            
            # Check each layer subdirectory
            for layer in layers:
                layer_dirs = list(multilayer_dir.glob(f"*_{layer}_layer*"))
                if layer_dirs:
                    layer_dir = layer_dirs[0]
                    n_files = len(list(layer_dir.glob("*.safetensors")))
                    expected = n_layers_b - 1  # layers 1 to n-1
                    if n_files == expected:
                        print(f"  ✓ {layer} layer: {n_files} files")
                    else:
                        print(f"  ✗ {layer} layer: {n_files}/{expected} files")
                        all_good = False
                else:
                    print(f"  ✗ {layer} layer: directory not found")
                    all_good = False
        else:
            print(f"✗ Analysis directory missing")
            all_good = False
        
        # Check bootstrap results
        bootstrap_dir = results_base / "bootstrap" / f"{pair_name}-multilayer"
        if bootstrap_dir.exists():
            summary_file = bootstrap_dir / "multilayer_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                print(f"✓ Bootstrap complete: {len(summary)} layer results")
            else:
                print(f"✗ Bootstrap summary missing")
                all_good = False
        else:
            print(f"✗ Bootstrap directory missing")
            all_good = False
        
        # Check plots
        plot_file = plots_base / f"{pair_name}_multilayer_analysis.png"
        if plot_file.exists():
            print(f"✓ Plot generated")
        else:
            print(f"✗ Plot missing")
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✓ ALL ANALYSES COMPLETE!")
    else:
        print("✗ Some analyses are incomplete. Check the logs.")
    
    # Show any bootstrap errors
    print("\nChecking for errors in logs...")
    log_dir = Path("/home/mascaretti/deepseek-hidden-states/logs")
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            with open(log_file, 'r') as f:
                content = f.read()
                if "error" in content.lower() or "exception" in content.lower():
                    print(f"⚠️  Errors found in {log_file.name}")
    
    return all_good


if __name__ == "__main__":
    check_results()
