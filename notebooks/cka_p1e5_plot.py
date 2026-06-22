"""Render the p=10^5 CKA Bernoulli/Gaussian curves from the cluster pickles.

Reads infimbgauss/cka_{bernoulli,gaussian}_p_100000.pickle (10 replicas x 200
feature-fraction grid points each, sample size n=2000) and renders a 1x2 panel
figure with the replica-mean curve and a +/- SE band.

Grid recovered from cka.compute_cka: jj = arange(1, p, p//n_data) with
p=100000, n_data=200 -> step=500 -> frac = jj/p in [1e-5, ~0.995].
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(["nature", "science"])
except Exception:
    pass


def stack_replicas(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.stack([np.asarray(d[k]["2000"]) for k in sorted(d, key=int)])


PANELS = [
    ("Gaussian", "infimbgauss/cka_gaussian_p_100000.pickle", "tab:blue"),
    ("Bernoulli", "infimbgauss/cka_bernoulli_p_100000.pickle", "tab:red"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=100000)
    parser.add_argument("--n_data", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default="paper/figs-tmlr/rebuttal/cka_p1e5")
    args = parser.parse_args()

    step = args.p // args.n_data
    jj = np.arange(1, args.p, step)
    frac = jj / args.p

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (title, pkl, color) in zip(axes, PANELS):
        arr = stack_replicas(Path(pkl))
        n_rep = arr.shape[0]
        # truncate grid to whatever was saved (defensive: dict carries 200)
        m = min(arr.shape[1], frac.size)
        x = frac[:m]
        mean = arr[:, :m].mean(axis=0)
        se = arr[:, :m].std(axis=0, ddof=1) / np.sqrt(n_rep)

        ax.plot(x, mean, color=color, linewidth=1.5, label=f"CKA  ({n_rep} replicas)")
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.25, linewidth=0)
        ax.set_title(rf"{title}, $p=10^5$, $n=2000$")
        ax.set_xlabel(r"\% of shared features  ($j/p$)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("CKA")

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
