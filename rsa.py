"""RSA / Spearman correlation between pairwise-distance matrices.

Spearman correlation of two distance matrices = Pearson correlation of the
ranks of their entries. We rank the **strict upper triangle** of each matrix
(symmetric distance matrices' diagonal zeros and below-diagonal mirrors carry
no extra information) on the host with ``scipy.stats.rankdata``. The host
path avoids ``jax.scipy.stats.rankdata``, whose internal scatter-add over an
``s32[n^2]`` index array triggers slow XLA constant-folding (>10 s per shape)
on n=2500-scale matrices.
"""
import numpy as np
from scipy.stats import rankdata as _np_rankdata


def upper_tri_ranks(D):
    """Average-method ranks of D's strict upper triangle, flattened (float32 numpy)."""
    D_np = np.asarray(D)
    iu = np.triu_indices(D_np.shape[0], k=1)
    return _np_rankdata(D_np[iu], method="average").astype(np.float32)


def spearman_from_ranks(rx, ry):
    """Pearson correlation between two pre-ranked numpy vectors."""
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(rx @ ry / (np.sqrt(rx @ rx) * np.sqrt(ry @ ry)))


def spearman_distance(dx, dy):
    """Spearman correlation between two pairwise-distance matrices (upper triangle)."""
    return spearman_from_ranks(upper_tri_ranks(dx), upper_tri_ranks(dy))
