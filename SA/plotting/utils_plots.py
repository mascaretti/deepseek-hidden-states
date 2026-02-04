import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# Ensure parent repo (where utils.py lives) is importable when utils_plots is imported from SA
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in os.sys.path:
    os.sys.path.append(_repo_root)

from utils import makefolder, reduce_list_half_preserve_extremes


def apply_plot_style(rcpsize=12):
    """Apply consistent plotting style and return (colors, markers, linestyles)."""
    plt.rcParams['xtick.labelsize'] = rcpsize
    plt.rcParams['ytick.labelsize'] = rcpsize
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = rcpsize
    plt.rcParams.update({'figure.autolayout': True})

    colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    markers = ['o', '*']
    linestyles = ['-', '--', 'dotted']
    return colors, markers, linestyles


def get_layers(model_name, depths):
    layers = np.arange(1, depths[model_name] + 1, dtype=float)
    layers = np.array(reduce_list_half_preserve_extremes(layers), dtype=float)
    return layers


def load_ii_curves(
    *,
    base,
    batch_shuffle,
    model,
    aux_A,
    aux_B,
    n_files,
    min_token_length,
    languageA,
    languageB,
    mask,
    k,
    Nbits,
    seeds,
    n_tokens,
    avg_tokens,
):
    curves = []
    for seed in seeds:
        folder = makefolder(
            base=base,
            create_folder=False,
            batch_shuffle=batch_shuffle,
            modelA=model,
            modelB=model,
            aux_A=aux_A,
            aux_B=aux_B,
            n_files=n_files,
            min_token_length=min_token_length,
            languageA=languageA,
            languageB=languageB,
            mask=mask,
            k=k,
            Nbits=Nbits,
            seed=seed,
        )
        ab = np.load(folder + f"n_tokens_{n_tokens}_avg_{avg_tokens}_AB.npy").diagonal()
        ba = np.load(folder + f"n_tokens_{n_tokens}_avg_{avg_tokens}_BA.npy").diagonal()
        curves.append(np.stack([ab, ba], axis=0))  # (2, L)

    curves = np.stack(curves, axis=0)   # (S, 2, L)
    mean_curves = curves.mean(axis=0)   # (2, L)
    std_curves = curves.std(axis=0)     # (2, L)
    return mean_curves, std_curves


def linestyle_for_language(lang, linestyles):
    return linestyles[0] if lang.lower() == 'english' else linestyles[1]
