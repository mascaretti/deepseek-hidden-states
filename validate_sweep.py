#!/usr/bin/env python
"""
Validate sweep results against existing single-slice results.

Compares II values from the sweep experiment with the original run_ii_mean results
for matching (model, layer) pairs. Should produce identical values since the random
seed is hardcoded (key=42 in compute_ii_mean.py).

Usage:
    python validate_sweep.py [--base_dir BASE_DIR]
"""

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def load_ii_values(result_dir, metric="l2_centered_normalized"):
    """Load II forward and backward values from a result directory."""
    metric_dir = Path(result_dir, metric)
    if not metric_dir.exists():
        return None, None

    files = sorted(metric_dir.glob("layer_*.safetensors"),
                   key=lambda f: int(f.stem.split("_")[1]))
    if not files:
        return None, None

    forward = []
    backward = []
    for f in files:
        data = load_file(str(f))
        forward.append(float(data["inf_imb"]))
        backward.append(float(data["reciprocal_inf_imb"]))

    return np.array(forward), np.array(backward)


def compare(name, original_dir, sweep_dir, metric="l2_centered_normalized"):
    """Compare II values between original and sweep results."""
    orig_fwd, orig_bwd = load_ii_values(original_dir, metric)
    sweep_fwd, sweep_bwd = load_ii_values(sweep_dir, metric)

    if orig_fwd is None:
        print(f"  {name}: SKIP (original not found at {original_dir})")
        return
    if sweep_fwd is None:
        print(f"  {name}: SKIP (sweep not found at {sweep_dir})")
        return

    if len(orig_fwd) != len(sweep_fwd):
        print(f"  {name}: FAIL (different number of layers: {len(orig_fwd)} vs {len(sweep_fwd)})")
        return

    max_diff_fwd = np.max(np.abs(orig_fwd - sweep_fwd))
    max_diff_bwd = np.max(np.abs(orig_bwd - sweep_bwd))

    status = "PASS" if max_diff_fwd < 1e-6 and max_diff_bwd < 1e-6 else "FAIL"
    print(f"  {name}: {status}  (max diff forward={max_diff_fwd:.2e}, backward={max_diff_bwd:.2e})")

    if status == "FAIL":
        # Show per-layer diffs for debugging
        for i, (of, sf, ob, sb) in enumerate(zip(orig_fwd, sweep_fwd, orig_bwd, sweep_bwd)):
            if abs(of - sf) > 1e-6 or abs(ob - sb) > 1e-6:
                print(f"    layer {i+1}: fwd {of:.6f} vs {sf:.6f}, bwd {ob:.6f} vs {sb:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Validate sweep against original results")
    parser.add_argument("--base_dir", type=str,
                        default="/home/laios/image-information-imbalance/results-mean",
                        help="Base directory for results")
    args = parser.parse_args()

    base = Path(args.base_dir)
    sweep = base / "sweep"

    print("Validating sweep results against original single-slice results...\n")

    # Pairs to check: (description, original_dir, sweep_dir)
    checks = [
        (
            "ImageGPT[20] -> sweep DeepSeek",
            base / "igpt_deepseek",
            sweep / "igpt_deepseek" / "layer_20",
        ),
        (
            "DinoV2[24] -> sweep DeepSeek",
            base / "dino_deepseek",
            sweep / "dino_deepseek" / "layer_24",
        ),
    ]

    for name, orig, swp in checks:
        compare(name, orig, swp)

    print("\nNote: DeepSeek[52] is not in the every-5th sweep grid (50, 55 are).")
    print("Add 52 to LAYERS_A in run_ii_sweep.sbatch to also validate:")
    print("  deepseek_igpt  vs  sweep/deepseek_igpt/layer_52")
    print("  deepseek_dino  vs  sweep/deepseek_dino/layer_52")


if __name__ == "__main__":
    main()
