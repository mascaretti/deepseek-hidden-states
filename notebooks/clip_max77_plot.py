"""Plot the max=77 vs max=30 CLIP-text rebuttal A/B.

Reads the four cells produced by run_ii_clip_max{30,77}.sbatch and renders
a 2-panel figure (forward / backward direction) with four curves each:

    A — max=30, all 5,000 images          (published Fig. 4 baseline)
    B — max=77, all 5,000 images          (extraction-time fix only)
    C — max=30, filtered to 3,330 ≤77 BPE (apples-to-apples vs D)
    D — max=77, filtered to 3,330 ≤77 BPE (untruncated for these images)

Headline: C vs D shows whether the published II changes once the extraction
truncation is removed for images that fit in 77 BPE.

Usage (assumes results-mean has been rsync'd locally, or run on cluster):
    python notebooks/clip_max77_plot.py \\
        --results_dir /home/laios/image-information-imbalance/results-mean \\
        --output paper/figs-tmlr/rebuttal/clip_max77
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
    """Return (layer_idx, ii, ii_se) sorted by layer index for one cell+direction."""
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


CELLS = {
    "forward": [
        ("A — max=30, all 5,000",     "clip_text_clip_image",                    "tab:blue", "-"),
        ("B — max=77, all 5,000",     "clip_text_max77_clip_image",              "tab:red",  "-"),
        ("C — max=30, filtered 3,330", "clip_text_clip_image_filtered",          "tab:blue", "--"),
        ("D — max=77, filtered 3,330", "clip_text_max77_clip_image_filtered",    "tab:red",  "--"),
    ],
    "backward": [
        ("A — max=30, all 5,000",     "clip_image_clip_text",                    "tab:blue", "-"),
        ("B — max=77, all 5,000",     "clip_image_clip_text_max77",              "tab:red",  "-"),
        ("C — max=30, filtered 3,330", "clip_image_clip_text_filtered",          "tab:blue", "--"),
        ("D — max=77, filtered 3,330", "clip_image_clip_text_max77_filtered",    "tab:red",  "--"),
    ],
}

PANEL_TITLES = {
    "forward":  r"$\Delta(\mathrm{CLIP\text{-}text}\to\mathrm{CLIP\text{-}image})$",
    "backward": r"$\Delta(\mathrm{CLIP\text{-}image}\to\mathrm{CLIP\text{-}text})$",
}
X_LABELS = {
    "forward":  "CLIP-image layer (text fixed at layer 11)",
    "backward": "CLIP-text layer (image fixed at layer 20)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Parent of the eight clip_*_image{,_filtered} dirs.")
    parser.add_argument("--metric_subdir", type=str, default="l2_centered_normalized")
    parser.add_argument("--output", type=str,
                        default="paper/figs-tmlr/rebuttal/clip_max77",
                        help="Output stem; .pdf and .png are written.")
    args = parser.parse_args()

    results = Path(args.results_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, (direction, cells) in zip(axes, CELLS.items()):
        for label, dirname, color, ls in cells:
            folder = results / dirname / args.metric_subdir
            if not folder.exists():
                print(f"  missing: {folder}")
                continue
            x, y, se = load_layer_sweep(folder)
            ax.plot(x, y, color=color, linestyle=ls, marker="o", markersize=3,
                    linewidth=1.4, label=label)
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.12, linewidth=0)

        ax.set_title(PANEL_TITLES[direction])
        ax.set_xlabel(X_LABELS[direction])
        ax.set_ylabel("Information imbalance")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
