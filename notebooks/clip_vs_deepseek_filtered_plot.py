"""Plot CLIP-text↔CLIP-image vs DeepSeek↔DinoV2 — full vs filtered.

Direct rebuttal of reviewer 3yju's truncation hypothesis. Forward direction
only (Δ(text → image)), max=30 throughout (= published extraction). Two
panels:

  Left  — all 5,000 images (republishes paper Fig. 4 restricted to the two
          pairs in question).
  Right — 3,330-image subset whose 5-caption concat fits in CLIP's 77-token
          architectural cap (--keep_indices=data/clip_max77_keep.npy).

If the CLIP↔CLIP > DeepSeek↔DinoV2 ordering holds in both panels, the
published comparison is not a 30-token truncation artefact.

DeepSeek-V3 fixed at layer 36 (≈ 60% relative depth, paper main.tex:405).
CLIP-text fixed at layer 11.

Usage (from repo root, after rclone'ing the four dirs into data/results-mean):
    .venv/bin/python notebooks/clip_vs_deepseek_filtered_plot.py \\
        --results_dir data/results-mean \\
        --output paper/figs-tmlr/rebuttal/clip_vs_deepseek_filtered
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
    """Return (layer_idx, ii, ii_se) sorted by layer index."""
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


PANELS = {
    "all":      (r"All 5{,}000 images",        ""),
    "filtered": (r"$\leq$77 BPE subset (3{,}330 images)", "_filtered"),
}

PAIRS = [
    (r"CLIP-text (L11) $\to$ CLIP-image",       "clip_text_clip_image", "tab:blue"),
    (r"DeepSeek-V3 (L36) $\to$ DinoV2",         "deepseek_dino",        "tab:orange"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--metric_subdir", type=str, default="l2_centered_normalized")
    parser.add_argument("--output", type=str,
                        default="paper/figs-tmlr/rebuttal/clip_vs_deepseek_filtered")
    args = parser.parse_args()

    results = Path(args.results_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (panel_key, (panel_title, suffix)) in zip(axes, PANELS.items()):
        for label, base, color in PAIRS:
            folder = results / f"{base}{suffix}" / args.metric_subdir
            if not folder.exists():
                print(f"  missing: {folder}")
                continue
            x, y, se = load_layer_sweep(folder)
            ax.plot(x, y, color=color, marker="o", markersize=3.5,
                    linewidth=1.6, label=label)
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.15, linewidth=0)

        ax.set_title(panel_title)
        ax.set_xlabel("Image-encoder layer")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel(r"Information imbalance $\Delta(\mathrm{text} \to \mathrm{image})$")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
