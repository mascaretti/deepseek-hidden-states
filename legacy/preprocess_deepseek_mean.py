#!/usr/bin/env python
"""
Preprocess DeepSeek activations: mean-pool over the token dimension.

Input:  chunk_N.pkl  — list of sentences, each with hidden_states (62, variable_tokens, 7168) bfloat16
Output: mean_chunk_N.safetensors — per layer: (n_sentences, 7168) float32
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import re

import numpy as np
import torch
from safetensors.numpy import save_file, load_file
from tqdm import tqdm


def _extract_index(file):
    match = re.search(r"chunk_(\d+)\.pkl", file.name)
    return int(match.group(1)) if match else float("inf")


def list_folder(path, desc="chunk_"):
    folder = Path(path)
    return sorted(
        [f for f in folder.iterdir() if f.name.startswith(desc)],
        key=_extract_index,
    )


def process_chunk(chunk_path):
    """Load a single chunk and return mean-pooled activations per layer.

    Returns:
        dict[str, np.ndarray]: {layer_0: (n_sentences, 7168), ...} float32
    """
    with open(chunk_path, "rb") as f:
        data = pickle.load(f)

    layer_means = defaultdict(list)
    for sentence in data:
        hs = sentence["meta_info"]["hidden_states"][0]  # (62, tokens, 7168) bf16
        # mean over token dim → (62, 7168), then convert to float32
        mean = hs.float().mean(dim=1)  # (62, 7168) float32
        for layer_idx in range(mean.shape[0]):
            layer_means[f"layer_{layer_idx}"].append(mean[layer_idx].numpy())

    return {k: np.stack(v) for k, v in layer_means.items()}


def main():
    parser = argparse.ArgumentParser(description="Mean-pool DeepSeek activations over tokens")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Directory containing chunk_N.pkl files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory for mean_chunk_N.safetensors output")
    parser.add_argument("--verify", action="store_true",
                        help="After processing, verify first chunk against original")
    args = parser.parse_args()

    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_folder(input_dir, desc="chunk_")
    print(f"Found {len(files)} chunk files in {input_dir}")

    total_sentences = 0
    for file in tqdm(files, desc="Mean-pooling chunks"):
        out_name = f"mean_{file.stem}.safetensors"
        out_path = output_dir / out_name

        if out_path.exists():
            # Load to count sentences without reprocessing
            existing = load_file(str(out_path))
            total_sentences += existing["layer_0"].shape[0]
            continue

        layer_means = process_chunk(file)
        n_sent = layer_means["layer_0"].shape[0]
        total_sentences += n_sent

        save_file(layer_means, str(out_path))

    print(f"\nDone. Processed {total_sentences} sentences across {len(files)} chunks.")
    print(f"Output in: {output_dir}")

    # --- Verification ---
    if args.verify and len(files) > 0:
        print("\n=== Verification ===")
        first_file = files[0]
        out_path = output_dir / f"mean_{first_file.stem}.safetensors"

        # Recompute from original
        expected = process_chunk(first_file)
        # Load saved
        saved = load_file(str(out_path))

        all_ok = True
        for layer_key in sorted(expected.keys(), key=lambda k: int(k.split("_")[1])):
            diff = np.abs(expected[layer_key] - saved[layer_key]).max()
            shape = saved[layer_key].shape
            if diff > 1e-6:
                print(f"  MISMATCH {layer_key}: shape={shape}, max_diff={diff:.2e}")
                all_ok = False

        n_layers = len(saved)
        n_sent = saved["layer_0"].shape[0]
        embed = saved["layer_0"].shape[1]
        print(f"  Shape: ({n_sent} sentences, {n_layers} layers, {embed} embed)")
        print(f"  dtype: {saved['layer_0'].dtype}")

        if all_ok:
            print("  All layers match. Verification PASSED.")
        else:
            print("  Verification FAILED — see mismatches above.")


if __name__ == "__main__":
    main()
