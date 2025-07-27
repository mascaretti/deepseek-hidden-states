#!/bin/bash
# 15-minute test with reduced data

echo "Starting 15-minute test run..."
echo "=============================="

# Activate environment
source /home/mascaretti/myjax/bin/activate
cd /home/mascaretti/deepseek-hidden-states

# Create test script with reduced parameters
cat > test_run.py << 'EOF'
import sys
sys.path.insert(0, '.')
from compute_asymmetry_unified import *

# Override the subset size
print("Overriding parameters for quick test:")
print("  subset_size: 5000 -> 500")
print("  n_files: 500 -> 50")

# Monkey patch the main function
original_main = main

def test_main(args):
    global subset_size, n_files
    # These are accessed in main()
    import compute_asymmetry_unified
    # Temporarily change the module-level variables
    original_subset = compute_asymmetry_unified.subset_size if hasattr(compute_asymmetry_unified, 'subset_size') else 5000
    
    # Inject the values into the main function's local scope
    import inspect
    frame = inspect.currentframe()
    
    # Call original but with modified values
    def modified_main(args):
        subset_size = 500  # Override here
        n_files = 50
        
        # Rest of the main function logic
        print(f"Comparison type: {args.comparison}")
        print(f"Model A: {args.model_a}")
        print(f"Model B: {args.model_b}")
        print(f"Output: {args.output_folder}")
        print(f"Using subset_size: {subset_size}")
        print(f"Using n_files: {n_files}")
        
        Path(args.output_folder).mkdir(parents=True, exist_ok=True)
        
        # Just test DinoV2 vs CLIP image (simplest case)
        all_activations_A = collect_image_model_data(
            args.input_path_A, n_files=n_files, last_tokens=200
        )
        all_activations_B = collect_clip_image_data(
            args.input_path_B, n_files=n_files, last_tokens=200
        )
        layer_A_id = 24  # DinoV2 layer
        
        sample_size_A = all_activations_A["layer_0"].shape[0]
        sample_size_B = all_activations_B["layer_0"].shape[0]
        print(f"Sample size A: {sample_size_A}")
        print(f"Sample size B: {sample_size_B}")
        
        compute_inf_imb_matrix(
            all_activations_A=all_activations_A,
            all_activations_B=all_activations_B,
            output_folder=args.output_folder,
            layer_A_id=layer_A_id,
            comparison_type="image_image"
        )
        
        print(f"Results saved to: {args.output_folder}")
    
    modified_main(args)

# Replace main
main = test_main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=str, default="dino_clip_image")
    parser.add_argument("--model_a", type=str, default="dino")
    parser.add_argument("--model_b", type=str, default="clip_image")
    parser.add_argument("--input_path_A", type=str, 
                       default="/home/laios/image-information-imbalance/facebook-dinov2-large")
    parser.add_argument("--input_path_B", type=str,
                       default="/home/laios/image-information-imbalance/openai-clip-vit-large-patch14")
    parser.add_argument("--output_folder", type=str,
                       default="/home/laios/image-information-imbalance/test_15min")
    
    args = parser.parse_args()
    main(args)
EOF

# Run the test
echo ""
echo "Running test..."
time python test_run.py

# Check results
echo ""
echo "Checking results..."
if [ -d "/home/laios/image-information-imbalance/test_15min" ]; then
    echo "Files created:"
    ls -la /home/laios/image-information-imbalance/test_15min/*.safetensors | head -5
    echo "..."
    echo "Total files: $(ls -1 /home/laios/image-information-imbalance/test_15min/*.safetensors | wc -l)"
else
    echo "ERROR: No output directory created"
fi

# Cleanup
rm -f test_run.py

echo ""
echo "Test complete!"
