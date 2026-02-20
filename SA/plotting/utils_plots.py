import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# Ensure parent repo (where utils.py lives) is importable when utils_plots is imported from SA
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in os.sys.path:
    os.sys.path.append(_repo_root)

try:
    from utils import makefolder, reduce_list_half_preserve_extremes
except ImportError:
    # Fallbacks for environments where ../../utils.py cannot import optional deps (e.g., jax).
    def reduce_list_half_preserve_extremes(lst):
        n = len(lst)
        if n <= 2:
            return lst.copy()
        half_n = max(n // 2, 2)
        num_points = half_n - 2
        reduced = [lst[0]]
        if num_points > 0:
            step = (n - 2) / (num_points + 1)
            indices = [int(round(1 + i * step)) for i in range(num_points)]
            reduced.extend([lst[i] for i in indices])
        reduced.append(lst[-1])
        return reduced

    def makefolder(base='./', create_folder=False, precision=5, **kwargs):
        folder = base
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, float):
                folder += key + f'_{value:.{precision}f}/'
            else:
                folder += key + f'_{value}/'
        if create_folder:
            os.makedirs(folder, exist_ok=True)
        return folder

def _min_token_length(model, languageA,languageB):
    condition0 = (model == 'llama')
    conditio1 = (languageA == 'english' and languageB == 'spanish')
    condition2 = (languageA == 'german' and languageB == 'english')
    if condition0 and (conditio1 or condition2):
        return 40
    elif model == 'deepseek':
        return 40
    else:
        return 35

def apply_plot_style(rcpsize=14):
    """Apply consistent plotting style and return (colors, markers, linestyles)."""
    plt.rcParams['xtick.labelsize'] = rcpsize
    plt.rcParams['ytick.labelsize'] = rcpsize
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = rcpsize
    plt.rcParams.update({'figure.autolayout': True})

    if 'seaborn-v0_8-dark-palette' in plt.style.library:
        style_name = 'seaborn-v0_8-dark-palette'
    elif 'seaborn-dark-palette' in plt.style.library:
        style_name = 'seaborn-dark-palette'
    else:
        style_name = 'default'
    colors = plt.style.library[style_name]['axes.prop_cycle'].by_key()['color']
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
