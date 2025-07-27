#!/usr/bin/env python
"""
Test script for the remaining unchecked items:
1. DeepSeek bfloat16 handling
2. CLIP text with 13 layers
"""

import sys
sys.path.insert(0, '.')
from compute_asymmetry_unified import *
import time

def test_deepseek_loading():
    """Test DeepSeek data loading with bfloat16"""
    print("\n" + "="*60)
    print("TEST 1: DeepSeek bfloat16 handling")
    print("="*60)
    
    start = time.time()
    try:
        # Load just 100 samples for quick test
        deepseek_data = collect_deepseek_data(
            "/home/laios/activations/flickr/deepseek_all",
            n_loaded_tokens=9,
            subset_size=50
        )
        
        print(f"✓ DeepSeek loaded successfully in {time.time()-start:.1f}s")
        print(f"  Shape: {deepseek_data['layer_0'].shape}")
        print(f"  Dtype: {deepseek_data['layer_0'].dtype}")
        print(f"  Layers available: {len(deepseek_data)} (layer_0 to layer_{len(deepseek_data)-1})")
        print(f" hasattribute: {hasattr(deepseek_data['layer_0'], 'dtype')}")
        
        # Test conversion
        test_tensor = deepseek_data['layer_52'][:10, -9:, :]
        if test_tensor.dtype == torch.bfloat16:
            print("  Testing bfloat16 conversion...")
            converted = bf16_torch_to_jax(test_tensor)
            print(f"  ✓ Conversion successful: {converted.shape}, {converted.dtype}")
        else:
            print(f"  Note: DeepSeek is using {test_tensor.dtype}, not bfloat16")
            
        return True
        
    except Exception as e:
        print(f"✗ DeepSeek loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_text_loading():
    """Test CLIP text loading with 13 layers"""
    print("\n" + "="*60)
    print("TEST 2: CLIP text with 13 layers")
    print("="*60)
    
    start = time.time()
    try:
        # Load just 10 files for quick test
        clip_text_data = collect_clip_text_data(
            "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14",
            n_files=10,
            last_tokens=30
        )
        
        print(f"✓ CLIP text loaded successfully in {time.time()-start:.1f}s")
        print(f"  Shape: {clip_text_data['layer_0'].shape}")
        print(f"  Dtype: {clip_text_data['layer_0'].dtype}")
        print(f"  Layers available: {len(clip_text_data)} (layer_0 to layer_{len(clip_text_data)-1})")
        
        # Verify layer 12 exists (last layer)
        if 'layer_12' in clip_text_data:
            print(f"  ✓ Layer 12 (last layer) exists: {clip_text_data['layer_12'].shape}")
        else:
            print(f"  ✗ Layer 12 not found! Available: {sorted(clip_text_data.keys())}")
            
        return True
        
    except Exception as e:
        print(f"✗ CLIP text loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_combinations():
    """Test different layer combinations"""
    print("\n" + "="*60)
    print("TEST 3: Layer combination suggestions")
    print("="*60)
    
    print("\nRecommended layer combinations:")
    print("\n1. DinoV2 (25 layers, 0-24):")
    print("   - First: layer_1")
    print("   - Middle: layer_12")
    print("   - Last: layer_24")
    
    print("\n2. CLIP Image (25 layers, 0-24):")
    print("   - First: layer_1")
    print("   - Middle: layer_12")
    print("   - Last: layer_24")
    
    print("\n3. CLIP Text (13 layers, 0-12):")
    print("   - First: layer_1")
    print("   - Middle: layer_6")
    print("   - Last: layer_12")
    
    print("\n4. DeepSeek (62 layers, 0-61):")
    print("   - Early: layer_10")
    print("   - Middle: layer_30")
    print("   - Semantic: layer_52 (your default)")
    print("   - Last: layer_61")


if __name__ == "__main__":
    print("Running remaining tests...")
    
    # Test 1: DeepSeek
    deepseek_ok = test_deepseek_loading()
    
    # Test 2: CLIP text
    clip_text_ok = test_clip_text_loading()
    
    # Test 3: Show layer suggestions
    test_layer_combinations()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"DeepSeek loading: {'✓ PASS' if deepseek_ok else '✗ FAIL'}")
    print(f"CLIP text loading: {'✓ PASS' if clip_text_ok else '✗ FAIL'}")
    print("="*60)
