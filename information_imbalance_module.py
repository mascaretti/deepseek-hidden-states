import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import rankdata
import numpy as np
from functools import partial

def binarize(a):
    a = np.sign(a)
    a[np.asarray(a == 0).nonzero()] = -1 # occasional zeros are set to -1
    a = a.astype(np.int8)
    return a

def hamming_distance(x, y):
    return jnp.count_nonzero(x != y)

def L2_distance(x, y, significant_digits=2):
    u = (x-y)
    normalized_distance = jnp.sqrt(jnp.sum(u*u) / (jnp.sum(x*x) * jnp.sum(y*y)))
    scale = 10**significant_digits
    return (scale*normalized_distance).astype(jnp.int32)

def pairwise_distances(distance, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: distance(x, y))(xs))(ys).T

def compute_relative_rank(x_dist, y_dist):
    x_rank = rankdata(x_dist, method="min")
    y_rank = rankdata(y_dist, method="min")
    rel_ranks = jnp.where(x_rank == 1, y_rank - 1, -1)
    return jnp.sum(rel_ranks, where=rel_ranks != -1), jnp.asarray(rel_ranks != -1).sum()

batch_relative_ranks = jax.vmap(compute_relative_rank)

def build_information_imbalance(distance_fn):
    """Construct function to compute the information imbalance between sets X and Y.

    Args:
       key: the JAX random key.
       X: sets from which we compute the information imbalance.
       Y: sets to which we compute the information imbalance.
       distance_fn: function to compute the distance in spaces X and Y.
       frac_rows: fraction of rows that are selected as origins.
    """
    def information_imbalance(X, Y):
        """Function to compute the information imbalance from X to Y.

        Args:
            X: the origin space.
            Y: the destination space.

        Returns:
            The value of the information imbalance from X to Y.
        """
        assert X.shape[0] == Y.shape[0]

        M = X.shape[0]
        d_X = pairwise_distances(distance_fn, X, X)
        d_Y = pairwise_distances(distance_fn, Y, Y)
        ranks, cards = batch_relative_ranks(d_X, d_Y)
        inf_imb = 2. * (jnp.sum(ranks) / jnp.sum(cards)) / (M - 2.)
        return inf_imb
    
    return jax.jit(information_imbalance)
