#!/usr/bin/env python
"""
Compute information imbalance between two models using mean-pooled activations.

Input:  mean_chunk_N.safetensors from preprocess_mean.py
        Each file: {layer_N: (n_samples, embed)} float32

Pipeline: load → concatenate chunks → binarize (Hamming) / raw or normalise (L2) → II
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file
from tqdm import tqdm

from information_imbalance_module import (
    L2_distance,
    build_information_matrix,
    hamming_distance,
)
from utils import binarize


# ---------------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------------

def jackknife_ii(ranks, cards, n_col):
    """Compute II with jackknife standard error from per-row ranks and cards.

    Args:
        ranks: (n_rows,) array of per-row rank sums.
        cards:  (n_rows,) array of per-row cardinalities.
        n_col:  number of column samples used in the II computation.

    Returns:
        (ii, se): full-sample II estimate and jackknife standard error.
    """
    n = len(ranks)
    total_r = float(ranks.sum())
    total_c = float(cards.sum())
    if total_c == 0:
        return float("nan"), float("nan")

    ii_full = 2.0 * total_r / total_c / (n_col - 1)

    # Leave-one-out estimates
    loo_r = total_r - ranks
    loo_c = total_c - cards
    ii_loo = np.where(loo_c > 0, 2.0 * loo_r / loo_c / (n_col - 1), np.nan)

    # Pseudo-values → SE
    pseudo = n * ii_full - (n - 1) * ii_loo
    se = float(np.nanstd(pseudo, ddof=1) / np.sqrt(n))
    return float(ii_full), se



# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _extract_index(f):
    match = re.search(r"chunk_(\d+)", f.name)
    return int(match.group(1)) if match else float("inf")


def load_mean_activations(input_path, subset_size=5000, prefix="mean_"):
    """Load mean-pooled activations and concatenate across chunks.

    Returns:
        dict[str, np.ndarray]: {layer_N: (min(total, subset_size), embed)}
    """
    folder = Path(input_path)
    files = sorted(
        [f for f in folder.iterdir() if f.name.startswith(prefix) and f.suffix == ".safetensors"],
        key=_extract_index,
    )
    if not files:
        raise FileNotFoundError(f"No {prefix}*.safetensors files in {input_path}")

    layer_data = defaultdict(list)
    total = 0
    for f in tqdm(files, desc=f"Loading {folder.name}", leave=False):
        if total >= subset_size:
            break
        chunk = load_file(str(f))
        for layer_key, arr in chunk.items():
            layer_data[layer_key].append(arr)
        total += chunk["layer_0"].shape[0]

    result = {}
    for layer_key, arrays in layer_data.items():
        result[layer_key] = np.concatenate(arrays, axis=0)[:subset_size]

    for layer_key, arr in result.items():
        assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype} in {layer_key}"
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in {layer_key} from {folder.name}")

    sample = result["layer_0"]
    print(f"  {folder.name}: {sample.shape[0]} samples, "
          f"{len(result)} layers, embed={sample.shape[1]}")
    return result


# ---------------------------------------------------------------------------
# L2 normalisation
# ---------------------------------------------------------------------------

def l2_normalize(a):
    norms = jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a / jnp.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute information imbalance from mean-pooled activations"
    )
    parser.add_argument("--input_path_A", type=str, required=True,
                        help="Directory with mean-pooled safetensors for model A")
    parser.add_argument("--input_path_B", type=str, required=True,
                        help="Directory with mean-pooled safetensors for model B")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--layer_A", type=int, nargs="+", required=True,
                        help="Fixed layer index(es) for model A (multiple = sweep)")
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--l2_normalize", action="store_true",
                        help="L2-normalise vectors before computing L2 distance (cosine mode)")
    parser.add_argument("--mean_center", action="store_true",
                        help="Subtract per-layer mean before computing L2 distance")
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["hamming", "l2"],
                        help="Which metrics to compute (default: hamming l2)")
    args = parser.parse_args()

    metrics = set(args.metrics)
    l2_label = None
    if "l2" in metrics:
        l2_parts = ["l2"]
        if args.mean_center:
            l2_parts.append("centered")
        if args.l2_normalize:
            l2_parts.append("normalized")
        l2_label = "_".join(l2_parts)

    # Load once
    print("Loading activations...")
    act_A = load_mean_activations(args.input_path_A, subset_size=args.subset_size)
    act_B = load_mean_activations(args.input_path_B, subset_size=args.subset_size)

    # Validate all requested layers exist
    layer_A_list = args.layer_A
    for la in layer_A_list:
        if f"layer_{la}" not in act_A:
            available = sorted(act_A.keys(), key=lambda k: int(k.split("_")[1]))
            raise ValueError(f"layer_{la} not in model A. Available: {available}")

    sample_size_A = act_A["layer_0"].shape[0]
    sample_size_B = act_B["layer_0"].shape[0]
    if sample_size_A != sample_size_B:
        raise ValueError(f"Sample size mismatch: A={sample_size_A}, B={sample_size_B}")
    sample_size = sample_size_A
    n_layers_B = len(act_B)

    print(f"\nModel A target layers: {layer_A_list}")
    print(f"Model B layers: {n_layers_B} ({act_B['layer_0'].shape[1]}d)")
    print(f"Sample size: {sample_size}")

    # Loop over target layers (data already loaded)
    for layer_A_idx in layer_A_list:
        layer_A_key = f"layer_{layer_A_idx}"

        # Output folder: use subfolder per layer_A when sweeping multiple
        if len(layer_A_list) > 1:
            out_folder = Path(args.output_folder, f"layer_{layer_A_idx}")
        else:
            out_folder = Path(args.output_folder)

        if "hamming" in metrics:
            Path(out_folder, "hamming").mkdir(parents=True, exist_ok=True)

        if l2_label:
            Path(out_folder, l2_label).mkdir(parents=True, exist_ok=True)

        print(f"\n--- layer_A = {layer_A_idx} ({act_A[layer_A_key].shape[1]}d) ---")

        raw_a = jnp.array(act_A[layer_A_key])

        # Deterministic per-metric keys: each metric always gets the same
        # subkey regardless of which other metrics are enabled.
        base_key = jax.random.key(42)
        hamming_key = jax.random.fold_in(base_key, 0)
        l2_key = jax.random.fold_in(base_key, 1)

        hamming_fn = l2_fn = None
        n_col = None
        if "hamming" in metrics or "l2" in metrics:
            n_row = int(np.floor(0.2 * sample_size))
            n_col = sample_size - n_row

        if "hamming" in metrics:
            hamming_fn = build_information_matrix(
                key=hamming_key, sample_size=sample_size, distance_fn=hamming_distance
            )
        if "l2" in metrics:
            l2_fn = build_information_matrix(
                key=l2_key, sample_size=sample_size, distance_fn=L2_distance
            )

        # Prepare model A transforms
        a_hamming = binarize(raw_a) if "hamming" in metrics else None
        a_l2 = None
        if "l2" in metrics:
            if args.l2_normalize:
                print("L2 Normalisation")
            if args.mean_center:
                print("Mean centering")
            a_l2 = raw_a
            if args.mean_center:
                a_l2 = a_l2 - a_l2.mean(axis=0)
            if args.l2_normalize:
                a_l2 = l2_normalize(a_l2)

        # Sweep over model B layers
        print(f"Metrics: {', '.join(sorted(metrics))}")
        cols = [f"layer_{i}" for i in range(1, n_layers_B)]
        for layer_B_key in tqdm(cols, desc=f"layer_A={layer_A_idx}"):
            raw_b = jnp.array(act_B[layer_B_key])

            if "hamming" in metrics:
                b_hamming = binarize(raw_b)
                result = hamming_fn(a_hamming, b_hamming)
                result_np = {k: np.asarray(v) for k, v in result.items()}
                ii_h, se_h = jackknife_ii(result_np["ranks"], result_np["cards"], n_col)
                rii_h, rse_h = jackknife_ii(result_np["r_ranks"], result_np["r_cards"], n_col)
                result_np["inf_imb"] = np.float32(ii_h)
                result_np["inf_imb_se"] = np.float32(se_h)
                result_np["reciprocal_inf_imb"] = np.float32(rii_h)
                result_np["reciprocal_inf_imb_se"] = np.float32(rse_h)
                save_file(result_np, str(Path(out_folder, "hamming", f"{layer_B_key}.safetensors")))


            if "l2" in metrics:
                b_l2 = raw_b
                if args.mean_center:
                    b_l2 = b_l2 - b_l2.mean(axis=0)
                if args.l2_normalize:
                    b_l2 = l2_normalize(b_l2)
                result = l2_fn(a_l2, b_l2)
                result_np = {k: np.asarray(v) for k, v in result.items()}
                ii_l2, se_l2 = jackknife_ii(result_np["ranks"], result_np["cards"], n_col)
                rii_l2, rse_l2 = jackknife_ii(result_np["r_ranks"], result_np["r_cards"], n_col)
                result_np["inf_imb"] = np.float32(ii_l2)
                result_np["inf_imb_se"] = np.float32(se_l2)
                result_np["reciprocal_inf_imb"] = np.float32(rii_l2)
                result_np["reciprocal_inf_imb_se"] = np.float32(rse_l2)
                save_file(result_np, str(Path(out_folder, l2_label, f"{layer_B_key}.safetensors")))

        print(f"Results saved to: {out_folder}")


if __name__ == "__main__":
    main()
