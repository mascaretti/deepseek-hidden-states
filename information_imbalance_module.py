import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import rankdata
import numpy as np
from functools import partial
from jax import random

def hamming_distance(x, y):
    return jnp.count_nonzero(x != y)

def L2_distance(x, y):
    return jnp.linalg.norm(x=x - y, ord=None)

def L1_distance(x, y):
    return jnp.sum(jnp.abs(x - y))

def Lhalf_distance(x, y):
    """L_{1/2} pseudo-metric: (sum sqrt|x_i - y_i|)^2. Less sensitive to concentration than L2."""
    return jnp.sum(jnp.sqrt(jnp.abs(x - y))) ** 2

def pairwise_distances(distance, xs, ys):
  if distance is L2_distance:
    sq_xs = jnp.sum(xs * xs, axis=1)
    sq_ys = jnp.sum(ys * ys, axis=1)
    d2 = sq_xs[:, None] + sq_ys[None, :] - 2.0 * (xs @ ys.T)
    return jnp.sqrt(jnp.maximum(d2, 0.0))
  return jax.vmap(lambda x: jax.vmap(lambda y: distance(x, y))(xs))(ys).T

def compute_relative_rank(x_dist, y_dist):
    x_rank = rankdata(x_dist, method="min")
    y_rank = rankdata(y_dist, method="min")
    rel_ranks = jnp.where(x_rank == 1, y_rank - 1, -1)
    return jnp.sum(rel_ranks, where=rel_ranks != -1), jnp.asarray(rel_ranks != -1).sum()

batch_relative_ranks = jax.vmap(compute_relative_rank)

def build_information_imbalance(key, sample_size, distance_fn):
    """Construct function to compute the information imbalance between sets X and Y.

    Args:
       key: the JAX random key.
       X: sets from which we compute the information imbalance.
       Y: sets to which we compute the information imbalance.
       distance_fn: function to compute the distance in spaces X and Y.
    """
    n_row_samples = jnp.array(500) if sample_size >= 2500 else jnp.floor(0.2 * sample_size).astype(int)
    n_col_samples = (sample_size - n_row_samples).astype(int)
    
    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=sample_size, shape=(n_row_samples,), replace=False)

    remaining_indices = jnp.setdiff1d(jnp.arange(sample_size), indices_rows)
    key, subkey = random.split(key)
    indices_columns = random.choice(key=subkey, a=remaining_indices, shape=(n_col_samples,), replace=False)
    
    def information_imbalance(X, Y):
        """Function to compute the information imbalance from X to Y.

        Args:
            X: the origin space.
            Y: the destination space.

        Returns:
            The value of the information imbalance from X to Y.
        """
        assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
        M = X.shape[0]
        d_X = pairwise_distances(distance_fn, X[indices_rows], X[indices_columns])
        d_Y = pairwise_distances(distance_fn, Y[indices_rows], Y[indices_columns])
        ranks, cards = batch_relative_ranks(d_X, d_Y)
        total_cards = jnp.sum(cards)
        inf_imb = jnp.where(
            total_cards > 0,
            2. * (jnp.sum(ranks) / total_cards) / (n_col_samples - 1.),
            jnp.nan,
        )
        reciprocal_ranks, reciprocal_cards = batch_relative_ranks(d_Y, d_X)
        total_reciprocal_cards = jnp.sum(reciprocal_cards)
        reciprocal_inf_imb = jnp.where(
            total_reciprocal_cards > 0,
            2. * (jnp.sum(reciprocal_ranks) / total_reciprocal_cards) / (n_col_samples - 1.),
            jnp.nan,
        )
        return inf_imb, reciprocal_inf_imb
    
    return jax.jit(information_imbalance)


def build_information_matrix(key, sample_size, distance_fn):
    """Construct function to compute the information imbalance between sets X and Y.

    Args:
       key: the JAX random key.
       X: sets from which we compute the information imbalance.
       Y: sets to which we compute the information imbalance.
       distance_fn: function to compute the distance in spaces X and Y.
    """
    n_row_samples = jnp.floor(0.2 * sample_size).astype(int)
    n_col_samples = (sample_size - n_row_samples).astype(int)
    
    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=sample_size, shape=(n_row_samples,), replace=False)

    remaining_indices = jnp.setdiff1d(jnp.arange(sample_size), indices_rows)
    key, subkey = random.split(key)
    indices_columns = random.choice(key=subkey, a=remaining_indices, shape=(n_col_samples,), replace=False)
    
    def information_matrix(X, Y):
        """Function to compute the information imbalance from X to Y.

        Args:
            X: the origin space.
            Y: the destination space.

        Returns:
            The value of the information imbalance from X to Y.
        """
        assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
        M = X.shape[0]
        d_X = pairwise_distances(distance_fn, X[indices_rows], X[indices_columns])
        d_Y = pairwise_distances(distance_fn, Y[indices_rows], Y[indices_columns])
        ranks, cards = batch_relative_ranks(d_X, d_Y)
        reciprocal_ranks, reciprocal_cards = batch_relative_ranks(d_Y, d_X)
        return {'ranks':ranks, 'cards':cards, 'r_ranks':reciprocal_ranks, 'r_cards':reciprocal_cards}
    
    return jax.jit(information_matrix)
