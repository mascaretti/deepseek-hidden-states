import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import rankdata
import numpy as np
from functools import partial
from jax import random

def hamming_distance(x, y):
    return jnp.count_nonzero(x != y)

def modified_L2_distance(x, y, significant_digits=2):
    u = (x-y)
    normalized_distance = jnp.sqrt(jnp.sum(u*u) / (jnp.sum(x*x) * jnp.sum(y*y)))
    scale = 10**significant_digits
    return (scale*normalized_distance).astype(jnp.int32)

def normalized_L2_distance(x, y):
    x /= jnp.linalg.norm(x)
    y /= jnp.linalg.norm(y)
    u = (x-y)
    return jnp.sqrt(jnp.sum(u*u))

def pairwise_distances(distance, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: distance(x, y))(xs))(ys).T

def compute_ranks(x_dist,y_dist):
    x_rank = rankdata(x_dist, method="min")
    y_rank = rankdata(y_dist, method="min")
    return x_rank,y_rank
mapped_compute_ranks = jax.vmap(compute_ranks)

def build_get_ranks(key, sample_size, distance_fn,):
    """

    Args:
       key: the JAX random key.
       X: sets from which we compute the information imbalance.
       Y: sets to which we compute the information imbalance.
       distance_fn: function to compute the distance in spaces X and Y.
    """
    indices_rows,indices_columns = separate_samples(key,sample_size)
    def _get_ranks(X, Y):
        assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
        d_X = pairwise_distances(distance_fn, X[indices_rows], X[indices_columns])
        d_Y = pairwise_distances(distance_fn, Y[indices_rows], Y[indices_columns])
        
        return mapped_compute_ranks(d_X,d_Y)
    
    return jax.jit(_get_ranks)

def get_relative_ranks(x_rank, y_rank, k):
    """ k = 1 corresponds to first neighbours"""
    rel_ranks = jnp.where(x_rank <= k, y_rank - 1, -1)
    return jnp.sum(rel_ranks, where=rel_ranks != -1), jnp.asarray(rel_ranks != -1).sum()

def mapped_relative_ranks(k):
    return jax.vmap(partial(get_relative_ranks,k=k))

def separate_samples(key, sample_size):
    n_row_samples = int(sample_size // 2)
    n_col_samples = int(sample_size - n_row_samples)
    
    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=sample_size, shape=(n_row_samples,), replace=False)

    remaining_indices = jnp.setdiff1d(jnp.arange(sample_size), indices_rows)
    key, subkey = random.split(key)
    indices_columns = random.choice(key=subkey, a=remaining_indices, shape=(n_col_samples,), replace=False)
    
    return indices_rows,indices_columns


def build_information_imbalance(k=1):


    def _information_imbalance(ranks_X,ranks_Y,k=k):
        assert ranks_X.shape == ranks_Y.shape

        max_rank = ranks_X.shape[1]
        relative_ranks,num_neighbours = mapped_relative_ranks(k)(ranks_X,ranks_Y)
        inf_imb = 2. * (jnp.sum(relative_ranks) / jnp.sum(num_neighbours)) / (max_rank - 1.)

        reciprocal_ranks, reciprocal_num_neighbours = mapped_relative_ranks(k)(ranks_Y,ranks_X)
        reciprocal_inf_imb = 2. * (jnp.sum(reciprocal_ranks) / jnp.sum(reciprocal_num_neighbours)) / (max_rank - 1.)

        return inf_imb,reciprocal_inf_imb
    return jax.jit(_information_imbalance)

### Platonic-functions

def _compute_neighbourhood_overlap(x_rank, y_rank,k):
    x_nns = jnp.where(x_rank <= k,1,0).astype(jnp.int8) # due to degeneracies there might be more than k points here
    y_nns = jnp.where(y_rank <= k,1,0).astype(jnp.int8)

    alignment = jnp.float32(jnp.bitwise_and(x_nns,y_nns).sum()) / ((x_nns.sum()+y_nns.sum())/2) # defined like this, it is always <= 1.

    rel_ranks_xy = jnp.where(x_nns.astype(jnp.bool), y_rank - 1, -1)
    rel_ranks_yx = jnp.where(y_nns.astype(jnp.bool), x_rank - 1, -1)

    return (alignment, 
            jnp.sum(rel_ranks_xy, where=rel_ranks_xy != -1), 
            jnp.asarray(rel_ranks_xy != -1).sum(),
            jnp.sum(rel_ranks_yx, where=rel_ranks_yx != -1), 
            jnp.asarray(rel_ranks_yx != -1).sum(),
            )

def mapped_compute_neighbourhood_overlaps(k):
    return jax.vmap(partial(_compute_neighbourhood_overlap,k=k))

def build_mutual_k_NN_alignment(key, sample_size, distance_fn, k=10):
    """
    """
    indices_rows,indices_columns = separate_samples(key,sample_size)
    
    def mutual_k_NN_alignment(X,Y,k=k):
        """

        Args:
            X: the origin space.
            Y: the destination space.

        Returns:
         jitted function to output the mutual-k-NN-alignment
        """
        assert X.shape[0] == Y.shape[0], "Sample size must be equal across X and Y."
        
        d_X = pairwise_distances(distance_fn, X[indices_rows], X[indices_columns])
        d_Y = pairwise_distances(distance_fn, Y[indices_rows], Y[indices_columns])
        ranks_X,ranks_Y = mapped_compute_ranks(d_X,d_Y)

        (alignment, 
        relative_ranks_XY,
        num_neighbors_X, 
        relative_ranks_YX, 
        num_neighbors_Y) = mapped_compute_neighbourhood_overlaps(k)(ranks_X,ranks_Y)

        alignment = alignment.mean()
        inf_imb_XY = 2. * (jnp.sum(relative_ranks_XY) / jnp.sum(num_neighbors_X)) / (len(indices_columns) - 1.)
        inf_imb_YX = 2. * (jnp.sum(relative_ranks_YX) / jnp.sum(num_neighbors_Y)) / (len(indices_columns) - 1.)

        return alignment, inf_imb_XY, inf_imb_YX
    
    return jax.jit(mutual_k_NN_alignment)
