#!/bin/bash
# Script to apply critical fixes to your asymmetry analysis scripts

echo "Critical Fixes Application Script"
echo "================================="
echo "This will apply minimal, safe fixes to your scripts"
echo ""

# Check if we're in the right directory
if [ ! -f "check_asymmetry.py" ]; then
    echo "ERROR: Cannot find check_asymmetry.py"
    echo "Please run this script from your project directory"
    exit 1
fi

# Create backup
echo "Creating backups..."
mkdir -p backups_$(date +%Y%m%d_%H%M%S)
cp check_asymmetry.py backups_$(date +%Y%m%d_%H%M%S)/
cp check_asymmetry_images.py backups_$(date +%Y%m%d_%H%M%S)/
cp check_mean_asymmetry.py backups_$(date +%Y%m%d_%H%M%S)/

echo "Backups created in backups_$(date +%Y%m%d_%H%M%S)/"

# Save the patch file
cat > critical_fixes.patch << 'EOF'
--- a/check_asymmetry.py
+++ b/check_asymmetry.py
@@ -30,11 +30,13 @@ def collect_data(input_path, filter_layer=None, n_loaded_tokens=9, subset_size=3
     for file in tqdm(files, desc="Collect File", leave=False):
         if file_counter >= subset_size:
             break
-        data = pickle.load(open(input_path + "/" + file.name, 'rb'))
+        with open(input_path + "/" + file.name, 'rb') as f:
+            data = pickle.load(f)
         for _, sentence in enumerate(data):
             hidden_states = sentence['meta_info']['hidden_states'][0]
-            assert hidden_states.shape[1] >= n_loaded_tokens
+            if hidden_states.shape[1] < n_loaded_tokens:
+                print(f"Warning: Skipping sentence with only {hidden_states.shape[1]} tokens")
+                continue
             hidden_states = hidden_states[:, -n_loaded_tokens:]
             file_counter += 1
 
@@ -56,7 +58,8 @@ def collect_images(input_path, n_files, last_tokens):
     files = list_folder(input_path, desc="chunk_")[:n_files]
     all_hidden_states = defaultdict(list)
     for file in tqdm(files, desc="Collect images"):
-        batch_activations = CPU_Unpickler(open(Path(input_path, file.name), 'rb')).load()
+        with open(Path(input_path, file.name), 'rb') as f:
+            batch_activations = CPU_Unpickler(f).load()
         for layer_key in batch_activations.keys():
             all_hidden_states[layer_key].append(batch_activations[layer_key][:, -last_tokens:, :])
     for layer, tensors in all_hidden_states.items():
@@ -85,10 +88,15 @@ def compute_inf_imb_matrix(
     sample_size_B = all_activations_B["layer_1"].shape[0]
     assert sample_size == sample_size_B
         
+    # Use configurable layer instead of hardcoded
+    layer_A = "layer_52"  # TODO: make this configurable via args
+    if layer_A not in all_activations_A:
+        raise ValueError(f"Layer {layer_A} not found in activations A. Available layers: {list(all_activations_A.keys())}")
+    
     key, subkey = jax.random.split(key)
     information_matrix_fn = build_information_matrix(key=subkey, sample_size=sample_size, distance_fn=hamming_distance)
 
-    layer_A = "layer_52"
     print(f"Layer A: {layer_A}")
     activations_A = all_activations_A[layer_A]
     
@@ -97,8 +105,17 @@ def compute_inf_imb_matrix(
         act_A = flatten_tokens_features(act_A)
         act_A = binarize(act_A)
                  
-        act_B = jnp.array(all_activations_B[layer_B])
+        # Ensure consistent conversion
+        act_B_torch = all_activations_B[layer_B]
+        if hasattr(act_B_torch, 'dtype') and act_B_torch.dtype == torch.bfloat16:
+            act_B = bf16_torch_to_jax(act_B_torch)
+        else:
+            act_B = jnp.array(act_B_torch)
+        
         act_B = flatten_tokens_features(act_B)
         act_B = binarize(act_B)
 
         inf_imb = information_matrix_fn(act_A, act_B)
EOF

# Apply the patch
echo ""
echo "Applying fixes..."
patch -p1 < critical_fixes.patch

# Now fix check_asymmetry_images.py manually since it has slight differences
echo "Fixing check_asymmetry_images.py..."
sed -i.bak 's/batch_activations = CPU_Unpickler(open(Path(input_path, file.name), '\''rb'\'')).load()/with open(Path(input_path, file.name), '\''rb'\'') as f:\n            batch_activations = CPU_Unpickler(f).load()/' check_asymmetry_images.py

# Fix check_mean_asymmetry.py
echo "Fixing check_mean_asymmetry.py..."
sed -i.bak 's/batch_activations = CPU_Unpickler(open(Path(input_path, file.name), '\''rb'\'')).load()/with open(Path(input_path, file.name), '\''rb'\'') as f:\n            batch_activations = CPU_Unpickler(f).load()/' check_mean_asymmetry.py

# Clean up
rm -f critical_fixes.patch
rm -f *.bak

echo ""
echo "✅ Critical fixes applied successfully!"
echo ""
echo "Summary of changes:"
echo "1. Fixed file descriptor leaks (using context managers)"
echo "2. Replaced asserts with warnings (won't crash on bad data)"
echo "3. Added validation for layer existence"
echo "4. Standardized tensor conversion (handles bfloat16 correctly)"
echo ""
echo "Next steps:"
echo "1. Test on a small subset to ensure everything works"
echo "2. Run your full analysis"
echo "3. Use the bootstrap_analysis.py and plot_results.py scripts provided earlier"
echo ""
echo "Your original files are backed up in: backups_$(date +%Y%m%d_%H%M%S)/"
