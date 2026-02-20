#!/usr/bin/env python
"""
Mean-pool activations over the token dimension for any model.
Includes:
1. Winsorization (clipping to 5th/95th quantiles) to remove spikes.
2. Option to select only the last K tokens (DeepSeek).

Supports two on-disk formats (auto-detected from the first chunk):

  DeepSeek:  chunk_N.pkl → list of sentence dicts
             sentence['meta_info']['hidden_states'][0] → (n_layers, tokens, embed) bf16
             tokens are variable-length across sentences.

  Image models (DinoV2, ImageGPT, CLIP, ...):  chunk_N.pkl → dict
             {layer_N: (batch, tokens, embed)} float32
             tokens are fixed within a model.

Output (both cases):
  mean_chunk_N.safetensors → {layer_N: (n_samples, embed)} float32
"""

import argparse
import io
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class CPU_Unpickler(pickle.Unpickler):
    """Unpickle GPU-saved tensors onto CPU."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _extract_index(file):
    match = re.search(r"chunk_(\d+)\.pkl$", file.name)
    return int(match.group(1)) if match else float("inf")


def list_chunks(path, prefix="chunk_"):
    folder = Path(path)
    return sorted(
        [f for f in folder.iterdir() if f.name.startswith(prefix)],
        key=_extract_index,
    )


def winsorized_mean(tensor, q_low=0.05, q_high=0.95, dim=1):
    """
    Computes the mean after clamping values to the q_low/q_high quantiles.

    Quantiles are computed per sample over all tokens × features jointly
    (one scalar bound per sample), so inter-feature relative scales are
    preserved.  Input must be 3-D with the token axis at `dim`.
    """
    tensor = tensor.to(torch.float32)
    B, T, E = tensor.shape

    # Flatten tokens × features → one threshold per sample
    flat = tensor.reshape(B, T * E)
    low_val  = torch.quantile(flat, q_low,  dim=1)  # (B,)
    high_val = torch.quantile(flat, q_high, dim=1)  # (B,)

    # Clip with per-sample scalar bounds, then mean over tokens
    clamped = torch.clamp(tensor,
                          min=low_val[:, None, None],
                          max=high_val[:, None, None])
    if not torch.isfinite(clamped).all():
        raise ValueError("Non-finite values after winsorization in chunk.")
    return clamped.mean(dim=dim).to(torch.float32)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(chunk_path):
    """Return 'deepseek' or 'image' based on the first chunk's structure."""
    with open(chunk_path, "rb") as f:
        data = CPU_Unpickler(f).load()
    if isinstance(data, list) and isinstance(data[0], dict) and "meta_info" in data[0]:
        return "deepseek", data
    if isinstance(data, dict) and "layer_0" in data:
        return "image", data
    raise ValueError(f"Unrecognised chunk format in {chunk_path}")


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------
def process_deepseek_chunk(data, last_k=None, start_token_idx=0):
    """Mean-pool a DeepSeek/Llama chunk.
    
    Args:
        data: List of sentence dictionaries.
        last_k: If integer > 0, only use the last k tokens.
        start_token_idx: Index to start pooling from (use 1 to skip BOS).
    """
    layer_means = defaultdict(list)
    for sentence in data:
        hs = sentence["meta_info"]["hidden_states"][0]  # (layers, tokens, embed)
        
        # 1. CRITICAL: Remove the BOS token (Attention Sink)
        if start_token_idx > 0:
            hs = hs[:, start_token_idx:, :]

        # 2. Slice last k tokens if requested (applied after removing BOS)
        if last_k is not None and last_k > 0:
            hs = hs[:, -last_k:, :]

        # Winsorized mean
        mean = winsorized_mean(hs, dim=1)  # (layers, embed)
        
        for layer_idx in range(mean.shape[0]):
            layer_means[f"layer_{layer_idx}"].append(mean[layer_idx].numpy())
    return {k: np.stack(v) for k, v in layer_means.items()}

def process_image_chunk(data, last_k=None):
    """Mean-pool an image-model chunk (dict of layer tensors)."""
    result = {}
    for layer_key, tensor in data.items():
        # tensor: (batch, tokens, embed)
        if last_k is not None and last_k > 0:
            tensor = tensor[:, -last_k:, :]
        result[layer_key] = winsorized_mean(tensor, dim=1).numpy()
    return result

def load_and_process(chunk_path, fmt, last_k=None, start_token_idx=0):
    """Load a chunk and return mean-pooled activations."""
    if fmt == "deepseek":
        with open(chunk_path, "rb") as f:
            data = pickle.load(f)
        # Pass the argument here!
        return process_deepseek_chunk(data, last_k=last_k, start_token_idx=start_token_idx)
    else:
        with open(chunk_path, "rb") as f:
            data = CPU_Unpickler(f).load()
        return process_image_chunk(data, last_k=last_k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mean-pool activations with Winsorization and optional token slicing."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Directory containing chunk_N.pkl files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory for mean_chunk_N.safetensors output")
    parser.add_argument("--prefix", type=str, default="chunk_",
                        help="File prefix to match (default: chunk_)")
    parser.add_argument("--last_k", type=int, default=0,
                        help="Use last k tokens before pooling. 0 = use all tokens (default).")
    parser.add_argument("--skip-first", dest="skip_first", action="store_true", 
                        help="Skip the first token (BOS) before pooling.")
    parser.add_argument("--verify", action="store_true",
                        help="After processing, verify first chunk against original")
    args = parser.parse_args()

    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_chunks(input_dir, prefix=args.prefix)
    if not files:
        print(f"No files matching '{args.prefix}*' in {input_dir}")
        return

    # Detect format from first file
    fmt, first_data = detect_format(files[0])
    print(f"Detected format: {fmt}")
    if args.last_k > 0:
        print(f"Pooling strategy: Winsorized Mean over LAST {args.last_k} tokens.")
    else:
        print(f"Pooling strategy: Winsorized Mean over ALL tokens.")

    print(f"Found {len(files)} chunk files in {input_dir}")

    print("Set first token:")
    start_idx = 1 if args.skip_first else 0
    print(f"We start from {start_idx}.")
    
    total_samples = 0
    for i, file in enumerate(tqdm(files, desc="Processing chunks")):
        out_name = f"mean_{file.stem}.safetensors"
        out_path = output_dir / out_name

        if out_path.exists():
            # If skipping, we just peek at shape to track total_samples
            existing = load_file(str(out_path))
            total_samples += existing["layer_0"].shape[0]
            continue

        # Reuse already-loaded first chunk
        if i == 0:
            layer_means = (process_deepseek_chunk(first_data, last_k=args.last_k, start_token_idx=start_idx) 
                           if fmt == "deepseek"
                           else process_image_chunk(first_data, last_k=args.last_k if args.last_k > 0 else None))
        else:
            layer_means = load_and_process(file, fmt, last_k=args.last_k, start_token_idx=start_idx)

        total_samples += layer_means["layer_0"].shape[0]
        save_file(layer_means, str(out_path))

    # Read back shape info from first output file
    first_out = output_dir / f"mean_{files[0].stem}.safetensors"
    sample = load_file(str(first_out))
    n_layers = len(sample)
    embed = sample["layer_0"].shape[1]
    print(f"\nDone. {total_samples} samples, {n_layers} layers, embed={embed}.")
    print(f"Output in: {output_dir}")

    # --- Verification ---    
    if args.verify:
        print("\n=== Verification ===")
        out_path = output_dir / f"mean_{files[0].stem}.safetensors"
        
        # Re-compute expected
        if fmt == "deepseek":
            expected = process_deepseek_chunk(first_data, last_k=args.last_k, start_token_idx=start_idx)
        else:
            expected = process_image_chunk(first_data, last_k=args.last_k if args.last_k > 0 else None)
            
        saved = load_file(str(out_path))

        all_ok = True
        for layer_key in sorted(expected.keys(), key=lambda k: int(k.split("_")[1])):
            diff = np.abs(expected[layer_key] - saved[layer_key]).max()
            if diff > 1e-6:
                print(f"  MISMATCH {layer_key}: shape={saved[layer_key].shape}, max_diff={diff:.2e}")
                all_ok = False

        n_s = saved["layer_0"].shape[0]
        n_l = len(saved)
        e = saved["layer_0"].shape[1]
        print(f"  Shape: ({n_s} samples, {n_l} layers, {e} embed)")
        print(f"  dtype: {saved['layer_0'].dtype}")
        print(f"  {'All layers match. PASSED.' if all_ok else 'FAILED — see above.'}")


if __name__ == "__main__":
    main()
