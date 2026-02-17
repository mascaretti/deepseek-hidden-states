#!/usr/bin/env python
"""
Plot smoke-test results for three II computation runs.

Produces a 1x3 figure comparing Hamming II and mean-centred L2 II
across layers of model B for each (model_A_layer, model_B) pair.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS = [
    {
        "dir": "results/smoke_dino_igpt",
        "title": "DinoV2 L12 vs ImageGPT",
        "layers": range(1, 49),
    },
    {
        "dir": "results/smoke_dino_dino",
        "title": "DinoV2 L12 vs DinoV2",
        "layers": range(1, 25),
    },
    {
        "dir": "results/smoke_igpt_igpt",
        "title": "ImageGPT L20 vs ImageGPT",
        "layers": range(1, 49),
    },
]

# Hamming normalisation constant.
# The computation used build_information_matrix with sample_size=410,
# giving n_row_samples=82 and n_col_samples=328.
N_COL_SAMPLES = 328


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hamming_ii(directory: Path, layers):
    """Compute Hamming II scalars from stored ranks / cards."""
    ii_vals, rii_vals, layer_ids = [], [], []
    for layer in layers:
        path = directory / "hamming" / f"layer_{layer}.safetensors"
        if not path.exists():
            continue
        with safe_open(str(path), framework="pt") as f:
            ranks = f.get_tensor("ranks").float()
            cards = f.get_tensor("cards").float()
            r_ranks = f.get_tensor("r_ranks").float()
            r_cards = f.get_tensor("r_cards").float()
        ii = 2.0 * (ranks.sum() / cards.sum()) / (N_COL_SAMPLES - 1)
        rii = 2.0 * (r_ranks.sum() / r_cards.sum()) / (N_COL_SAMPLES - 1)
        layer_ids.append(layer)
        ii_vals.append(ii.item())
        rii_vals.append(rii.item())
    return np.array(layer_ids), np.array(ii_vals), np.array(rii_vals)


def load_l2_ii(directory: Path, layers):
    """Load L2 II scalars directly from safetensors."""
    ii_vals, rii_vals, layer_ids = [], [], []
    for layer in layers:
        path = directory / "l2_centered_normalized" / f"layer_{layer}.safetensors"
        if not path.exists():
            continue
        with safe_open(str(path), framework="pt") as f:
            ii = f.get_tensor("inf_imb").item()
            rii = f.get_tensor("reciprocal_inf_imb").item()
        layer_ids.append(layer)
        ii_vals.append(ii)
        rii_vals.append(rii)
    return np.array(layer_ids), np.array(ii_vals), np.array(rii_vals)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for ax, cfg in zip(axes, RESULTS):
        d = Path(cfg["dir"])
        layers = cfg["layers"]

        h_layers, h_ii, h_rii = load_hamming_ii(d, layers)
        l_layers, l_ii, l_rii = load_l2_ii(d, layers)

        # A -> B  (inf_imb)
        ax.plot(h_layers, h_ii, "o-", ms=3, lw=1.2, color="C0",
                label="Hamming  A->B")
        ax.plot(l_layers, l_ii, "s-", ms=3, lw=1.2, color="C1",
                label="L2 cent.  A->B")

        # B -> A  (reciprocal_inf_imb)
        ax.plot(h_layers, h_rii, "o--", ms=3, lw=1.0, color="C0", alpha=0.5,
                label="Hamming  B->A")
        ax.plot(l_layers, l_rii, "s--", ms=3, lw=1.0, color="C1", alpha=0.5,
                label="L2 cent.  B->A")

        ax.set_title(cfg["title"], fontsize=11)
        ax.set_xlabel("Layer of model B")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Information Imbalance")
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle("Smoke-test: Information Imbalance across layers", fontsize=13, y=1.02)
    fig.tight_layout()

    out = Path("plots/smoke_test.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
