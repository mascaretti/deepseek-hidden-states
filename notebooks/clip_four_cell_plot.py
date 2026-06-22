"""Four-cell CLIP↔CLIP vs DeepSeek↔DinoV2 rebuttal figure.

2x2 grid for the architectural-cap A/B/C/D experiment, forward direction
Δ(text → image). Each panel overlays DeepSeek-V3(L36)↔DinoV2 as a baseline:

    rows: max_length = 30 (top), max_length = 77 (bottom)
    cols: all 5,000 images (left), ≤77 BPE subset of 3,330 (right)

DeepSeek↔DinoV2 does not depend on CLIP's max_length, so its curve is the
same within each column (all vs filtered).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file

try:
    import scienceplots  # noqa: F401
    plt.style.use(["nature", "science"])
except Exception:
    pass


def load_layer_sweep(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(
        folder.glob("layer_*.safetensors"),
        key=lambda p: int(re.search(r"layer_(\d+)", p.name).group(1)),
    )
    layers, ii, se = [], [], []
    for f in files:
        d = load_file(str(f))
        layers.append(int(re.search(r"layer_(\d+)", f.name).group(1)))
        ii.append(float(d["inf_imb"]))
        se.append(float(d["inf_imb_se"]))
    return np.array(layers), np.array(ii), np.array(se)


PANELS = [
    [
        ("A: max\\_length=30, all 5{,}000",                "clip_text_clip_image",                "deepseek_dino"),
        ("C: max\\_length=30, $\\leq$77 BPE (3{,}330)",    "clip_text_clip_image_filtered",       "deepseek_dino_filtered"),
    ],
    [
        ("B: max\\_length=77, all 5{,}000",                "clip_text_max77_clip_image",          "deepseek_dino"),
        ("D: max\\_length=77, $\\leq$77 BPE (3{,}330)",    "clip_text_max77_clip_image_filtered", "deepseek_dino_filtered"),
    ],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--metric_subdir", type=str, default="l2_centered_normalized")
    parser.add_argument("--output", type=str,
                        default="paper/figs-tmlr/rebuttal/clip_four_cell")
    args = parser.parse_args()

    results = Path(args.results_dir)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=False)
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            title, clip_dir, ds_dir = PANELS[row][col]

            clip_folder = results / clip_dir / args.metric_subdir
            if clip_folder.exists():
                x, y, se = load_layer_sweep(clip_folder)
                ax.plot(x, y, color="tab:blue", marker="o", markersize=3.5,
                        linewidth=1.6,
                        label=r"CLIP-text (L11) $\to$ CLIP-image")
                ax.fill_between(x, y - se, y + se, color="tab:blue",
                                alpha=0.15, linewidth=0)
            else:
                print(f"  missing: {clip_folder}")

            ds_folder = results / ds_dir / args.metric_subdir
            if ds_folder.exists():
                x, y, se = load_layer_sweep(ds_folder)
                ax.plot(x, y, color="tab:orange", marker="s", markersize=3.5,
                        linewidth=1.6,
                        label=r"DeepSeek-V3 (L36) $\to$ DinoV2")
                ax.fill_between(x, y - se, y + se, color="tab:orange",
                                alpha=0.15, linewidth=0)
            else:
                print(f"  missing: {ds_folder}")

            ax.set_title(title)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if row == 1:
                ax.set_xlabel("Image-encoder layer")
            if col == 0:
                ax.set_ylabel(
                    r"$\Delta(\mathrm{text} \to \mathrm{image})$"
                )
            ax.legend(loc="best", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
