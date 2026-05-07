"""RSA / Spearman correlation between pairwise-distance matrices.

Spearman correlation of two distance matrices = Pearson correlation of their
flattened ranks, i.e. the standard rank-based RSA alignment metric. We
implement it as ``cka_distance`` on ranks so the relationship to the
centred-Frobenius CKA in ``cka.py`` is explicit.
"""
import jax
import jax.numpy as jnp
from jax.scipy.stats import rankdata as _rankdata

from cka import cka_distance


@jax.jit
def rankdata_flat(d):
    """Average-method ranks of a flattened (distance) matrix."""
    return _rankdata(d.reshape(-1))


def spearman_distance(dx, dy):
    """Spearman correlation between two pairwise-distance matrices.

    Equivalent to ``scipy.stats.spearmanr`` over the flattened entries.
    """
    return cka_distance(rankdata_flat(dx), rankdata_flat(dy))
