import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import rankdata
import numpy as np
from functools import partial

key = jrandom.PRNGKey(42)  # Change 42 for different randomness

def binarize(a):
    a = np.sign(a)
    a[np.asarray(a == 0).nonzero()] = -1 # occasional zeros are set to -1
    a = a.astype(np.int8)
    return a

def compute_rank_matrix(batch_rows, batch_columns):
    """
    """
    diffs = batch_rows[:, jnp.newaxis, :] - batch_columns[jnp.newaxis, :, :]
    dist2_matrix = jnp.sum(diffs * diffs, axis=-1)

    rank_matrix = dist2_matrix.argsort(axis=1).argsort(axis=1)
    return rank_matrix

def compute_imbalance_and_error(
        data_A, data_B, indices_rows, indices_columns, k
    ):
        """
        """
        N = indices_rows.shape[0]
        max_rank = indices_columns.shape[0]

        batch_A_ranks = compute_rank_matrix( # compute rank matrix A
            batch_rows=data_A[indices_rows],
            batch_columns=data_A[indices_columns],
        ) + 1

        batch_B_ranks = compute_rank_matrix( # compute rank matrix B
            batch_rows=data_B[indices_rows],
            batch_columns=data_B[indices_columns],
        ) + 1
    
        mask = jnp.where(batch_A_ranks <= k, 1.0, 0.0)
        
        # compute imbalance and error
        values_average = 2.0 / ((max_rank + 1)*k) * jnp.sum(batch_B_ranks * mask, axis=1)
        imbalance = jnp.mean(values_average)
        error_imbalance = jnp.std(values_average, ddof=1) / jnp.sqrt(N)

        return imbalance, error_imbalance

def hamming_distance(x, y):
    return jnp.count_nonzero(x != y)

def L2_distance(x, y):
    u = (x-y)
    # u = (10*x-10*y).astype(jnp.int32)
    # return jnp.dot(u,u)
    return (100*jnp.sqrt(jnp.dot(u,u))).astype(jnp.int32)

def pairwise_distances(distance, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: distance(x, y))(xs))(ys).T


def compute_relative_rank(x_dist, y_dist):
    x_rank = rankdata(x_dist, method="min")
    y_rank = rankdata(y_dist, method="min")
    rel_ranks = jnp.where(x_rank == 1, y_rank - 1, -1)
    return jnp.sum(rel_ranks, where=rel_ranks != -1), jnp.asarray(rel_ranks != -1).sum()

batch_relative_ranks = jax.vmap(compute_relative_rank)

def build_information_imbalance(key, X, Y, distance_fn, frac_rows=0.20):
    """Construct function to compute the information imbalance between sets X and Y.

    Args:
       key: the JAX random key.
       X: sets from which we compute the information imbalance.
       Y: sets to which we compute the information imbalance.
       distance_fn: function to compute the distance in spaces X and Y.
       frac_rows: fraction of rows that are selected as origins.
    """
    n_samples = X.shape[0]
    n_row_samples = jnp.floor(n_samples * frac_rows).astype(int)
    n_col_samples = (n_samples - n_row_samples).astype(int)

    indices = jnp.arange(subset_size)
    key, subkey = random.split(key)
    origin_rows = random.choice(key=subkey, a=subset_size, shape=(n_row_samples,), replace=False)

    remaining_indices = jnp.setdiff1d(indices, indices_rows)
    key, subkey = random.split(key)
    destination_rows = random.permutation(subkey, x=remaining_indices)
    
    def information_imbalance(X, Y):
        """Function to compute the information imbalance from X to Y.

        Args:
            X: the origin space.
            Y: the destination space.

        Returns:
            The value of the information imbalance from X to Y.
        """
    
        d_X = pairwise_distances(distance_fn, X[origin_rows], X[destination_rows])
        d_Y = pairwise_distances(distance_fn, Y[origin_rows], Y[destination_rows])
        ranks, cards = batch_relative_ranks(d_X, d_Y)
        inf_imb = 2. * (jnp.sum(ranks) / jnp.sum(cards)) / (m - 2.)
        return inf_imb
    return jax.jit(information_imbalance)
