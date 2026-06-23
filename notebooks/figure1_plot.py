#!/usr/bin/env python
"""
Plot Figure 1 (+ its appendix companion) from precomputed .npz (produced by
figure1_compute.py).

Main figure (`{output}.pdf`):
    Panel a — II(X→Y), II(Y→X), CKA, Neighborhood Overlap vs rank index r.
    Panel b — shared-features curves at the *first* p value (p_values[0]).

Appendix figure (`{output}_appendix.pdf`):
    The remaining p value(s) (p_values[1:]) as a standalone figure — the
    right-most shared-features panel that used to sit in Fig. 1.

Usage:
    python notebooks/figure1_plot.py --input notebooks/figure1_data.npz \
        --output notebooks/figure1_ii_cka
"""

import argparse
import os
from pathlib import Path

mpl_cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots
    plt.style.use(["nature", "science"])
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 1 from .npz")
    parser.add_argument("--input", type=str, required=True, help="Path to .npz from figure1_compute.py")
    parser.add_argument("--output", type=str, default="notebooks/figure1_ii_cka",
                        help="Output path (no extension)")
    parser.add_argument("--no_spearman", action="store_true",
                        help="Suppress the Spearman/RSA curve even if present in the npz")
    args = parser.parse_args()

    d = np.load(args.input, allow_pickle=True)
    dim = int(d["dim"])
    L = int(d["L"])
    p_values = d["p_values"]

    # Colors
    col_xy   = "#364F6B"
    col_yx   = "#609966"
    col_cka  = "#C44536"
    col_no   = "#FC5185"
    col_spm  = "#E89C31"
    ms, mew = 3, 0.3

    have_spearman = "a_mean_spearman" in d.files and not args.no_spearman

    def draw_panel_b(ax, p):
        """Render one shared-features sub-panel (II/CKA/NO/Spearman vs fraction)."""
        p = int(p)
        fracs = d[f"b_{p}_fracs"]
        panel_b_lines = [
            (f"b_{p}_mean_cka", "CKA", col_cka),
            (f"b_{p}_mean_fwd", r"$\Delta(Y \rightarrow Y_{\mathrm{sub}})$", col_xy),
            (f"b_{p}_mean_bwd", r"$\Delta(Y_{\mathrm{sub}} \rightarrow Y)$", col_yx),
            (f"b_{p}_mean_no", "NO", col_no),
        ]
        if have_spearman:
            panel_b_lines.insert(1, (f"b_{p}_mean_spearman", "Spearman", col_spm))
        for key_base, label, color in panel_b_lines:
            mean = d[key_base]
            se = d[key_base.replace("mean", "se")]
            ax.plot(fracs, mean, color=color, markersize=ms,
                    markeredgecolor="black", markeredgewidth=mew, label=label)
            ax.fill_between(fracs, mean - se, mean + se, color=color, alpha=0.2)
        ax.set_xlabel("Fraction of shared features", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=11)
        ax.set_title(f"$p = {p}$", fontsize=11)

    # ==================================================================
    # Main figure — Panel a (rank mapping) + Panel b (first p value)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for i, ax in enumerate(axes):
        ax.text(-0.14, 1.10, f"{chr(ord('a') + i)})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # Panel a
    # ------------------------------------------------------------------
    ax = axes[0]
    rel = np.arange(L) / L

    panel_a_lines = [
        ("a_mean_XY", r"$\Delta(X \rightarrow Y)$", col_xy),
        ("a_mean_YX", r"$\Delta(Y \rightarrow X)$", col_yx),
        ("a_mean_cka", "CKA", col_cka),
        ("a_mean_no", "NO", col_no),
    ]
    if have_spearman:
        panel_a_lines.append(("a_mean_spearman", "Spearman", col_spm))
    for key_prefix, label, color in panel_a_lines:
        mean = d[key_prefix]
        se = d[key_prefix.replace("mean", "se")]
        ax.plot(rel, mean, "-o", color=color, markersize=ms,
                markeredgecolor="black", markeredgewidth=mew, label=label)
        ax.fill_between(rel, mean - se, mean + se, color=color, alpha=0.2)

    ax.set_xlabel("Index $r/p$", fontsize=11)
    ax.set_ylabel("II / CKA / NO", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title(f"$p = {dim}$", fontsize=11)

    # ------------------------------------------------------------------
    # Panel b — first p value only
    # ------------------------------------------------------------------
    draw_panel_b(axes[1], p_values[0])

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.output}.{ext}", dpi=300)
    print(f"Saved {args.output}.pdf and {args.output}.png")

    # ==================================================================
    # Appendix figure — the remaining p value(s), standalone
    # ==================================================================
    appendix_ps = p_values[1:]
    if len(appendix_ps) > 0:
        fig_a, axes_a = plt.subplots(1, len(appendix_ps),
                                     figsize=(3.5 * len(appendix_ps), 2.5))
        if not hasattr(axes_a, "__len__"):
            axes_a = [axes_a]
        for ax, p in zip(axes_a, appendix_ps):
            draw_panel_b(ax, p)
        axes_a[0].set_ylabel("II / CKA / NO", fontsize=11)
        fig_a.tight_layout()
        out_app = f"{args.output}_appendix"
        for ext in ("pdf", "png"):
            fig_a.savefig(f"{out_app}.{ext}", dpi=300)
        print(f"Saved {out_app}.pdf and {out_app}.png")


if __name__ == "__main__":
    main()
