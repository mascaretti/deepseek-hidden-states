import jax
import jax.numpy as jnp
from jax.scipy.stats import rankdata
from jax.nn import softmax
from functools import partial
import optax
from jax import random
from typing import Dict, Tuple, Callable
import math
import itertools

@jax.jit
def compute_rank_matrix(batch_rows, batch_columns):
    """
    """
    diffs = batch_rows[:, jnp.newaxis, :] - batch_columns[jnp.newaxis, :, :]
    dist2_matrix = jnp.sum(diffs * diffs, axis=-1)

    rank_matrix = dist2_matrix.argsort(axis=1).argsort(axis=1)
    return rank_matrix

@jax.jit
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

@jax.jit
def compute_imbalance_standard(
        data_A, data_B, k
    ):
        """
        """
        N = data_A.shape[0]
        max_rank = data_A.shape[0] - 1

        batch_A_ranks = compute_rank_matrix( # compute rank matrix A
            batch_rows=data_A,
            batch_columns=data_A,
        )
        batch_A_ranks = batch_A_ranks.at[jnp.arange(N), jnp.arange(N)].set(
            max_rank + 1
        )

        batch_B_ranks = _compute_rank_matrix( # compute rank matrix B
            batch_rows=data_B,
            batch_columns=data_B,
        )
        batch_B_ranks = batch_B_ranks.at[jnp.arange(N), jnp.arange(N)].set(
            max_rank + 1
        )
    
        mask = jnp.where(batch_A_ranks <= k, 1.0, 0.0)
        
        # compute imbalance and error
        values_average = 2.0 / ((max_rank + 1) * k) * jnp.sum(batch_B_ranks * mask, axis=1)
        std_imbalance = jnp.mean(values_average)

        return std_imbalance


@jax.jit
def hamming_distance(x, y):
    return jnp.count_nonzero(x != y)

def pairwise_distances(dist, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: dist(x, y))(xs))(ys).T

@jax.jit
def compute_relative_rank(x_dist, y_dist):
    x_rank = rankdata(x_dist, method="min")
    y_rank = rankdata(y_dist, method="min")
    rel_ranks = jnp.where(x_rank == 1, y_rank - 1, -1)
    return jnp.sum(rel_ranks, where=rel_ranks != -1), jnp.asarray(rel_ranks != -1).sum()

batch_relative_ranks = jax.vmap(compute_relative_rank)

@jax.jit
def spin_information_imbalance(X, Y, from_subset, to_subset):
    """
    """
    n = len(from_subset)
    m = len(to_subset)
    d_X = pairwise_distances(hamming_distance, X[from_subset], X[to_subset])
    d_Y = pairwise_distances(hamming_distance, Y[from_subset], Y[to_subset])
    ranks, cards = batch_relative_ranks(d_X, d_Y)
    inf_imb = 2 * (jnp.sum(ranks) / jnp.sum(cards)) / (m - 2.)
    return inf_imb

@jax.jit
def weighted_manhattan_distance(alpha, x, y):
    return jnp.sum(alpha * jnp.abs(x - y))

def pairwise_manhattan(alpha, xs, ys):
    return jax.vmap(lambda x: jax.vmap(lambda y:  weighted_manhattan_distance(alpha, x, y))(xs))(ys).T

def token_manhattan(alpha, embedding_dimension, n_tokens, xs, ys):
    weights = jnp.repeat(a=alpha, repeats=embedding_dimension, total_repeat_length=embedding_dimension*n_tokens)
    distances = pairwise_manhattan(alpha=weights, xs=xs, ys=ys)
    return distances
    
def coordinate_manhattan(alpha, embedding_dimension, n_tokens, xs, ys):
    weights = jnp.tile(A=alpha, reps=n_tokens)
    distances = pairwise_manhattan(alpha=weights, xs=xs, ys=ys)
    return distances

def token_spin_imbalance(params, X, Y, from_subset, to_subset, embedding_dimension, n_tokens, lambda_factor=0.0001):
    n = len(from_subset)
    m = len(to_subset)
    d_X = token_manhattan(alpha=params, embedding_dimension=embedding_dimension, n_tokens=n_tokens, xs=X[from_subset], ys=X[to_subset])
    L = lambda_factor * jnp.ones(d_X.shape[0])
    c_X = softmax(x=-d_X/L[:, jnp.newaxis], axis=1)
    c_X /= jnp.max(c_X, axis=1)[:, jnp.newaxis]
    d_Y = pairwise_manhattan(jnp.ones(Y.shape[1]), Y[from_subset], Y[to_subset])
    r_Y = rankdata(a=d_Y, method="min", axis=1) - 1.
    return 2. * jnp.sum(c_X * r_Y) / jnp.sum(c_X) / (m - 2.)
    

@jax.jit
def differentiable_spin_imbalance(params, X, Y, from_subset, to_subset, L=1e-10):
    n = len(from_subset)
    m = len(to_subset)
    d_X = pairwise_manhattan(params, X[from_subset], X[to_subset])
    c_X = softmax(x=-d_X/L, axis=1)
    c_X /= jnp.max(c_X, axis=1)[:, jnp.newaxis]
    d_Y = pairwise_manhattan(jnp.zeros(Y.shape[1]), Y[from_subset], Y[to_subset])
    r_Y = rankdata(a=d_Y, method="min", axis=1) - 1.
    return 2. * jnp.sum(c_X * r_Y) / jnp.sum(c_X) / (m - 2.)

def update_lambda(k, d_X, lambda_factor, lambda_offset=0.01):
    n, m = d_X.shape
    k_max = jnp.min(jnp.array([100, m]))
    smallest_distances, _ = jax.lax.top_k(-d_X, k_max)
    return (-1. * smallest_distances[:, k - 1] + lambda_offset) * lambda_factor

@partial(jax.jit, static_argnames=['optimizer'])
def update_weights(optimizer, params, opt_state, loss_value, grads, init_l2):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    params_l2 = jnp.linalg.norm(x=params, ord=2)
    params = (init_l2 / params_l2) * params
    return params, opt_state

def spin_imbalance(d_X, r_Y, lambda_factor=0.01):
    n = len(from_subset)
    m = len(to_subset)
    L = lambda_factor * jnp.ones(n)
    c_X = softmax(x=-d_X/L[:, jnp.newaxis], axis=1)
    c_X /= jnp.max(c_X, axis=1)[:, jnp.newaxis]
    return 2. * jnp.sum(c_X * r_Y) / jnp.sum(c_X) / (m - 2.)

def loss_function():
    return -1

def minimise_spin(X, Y, from_subset, to_subset, embedding_dimension, n_tokens, lambda_factor, learning_rate, n_iter, init_params, k_init_value, k_alpha):
    
    params = init_params
    assert params.shape == (n_tokens,), "Params shape should be equal to the number of tokens"
    init_l2 = jnp.linalg.norm(x=init_params, ord=2)
    
    L = lambda_factor * jnp.ones(len(from_subset))
    
    d_X = token_manhattan(params, embedding_dimension, n_tokens, X[from_subset], X[to_subset])
    d_Y = token_manhattan(jnp.ones(n_tokens), embedding_dimension, n_tokens, Y[from_subset], Y[to_subset])
    
    r_Y = rankdata(a=d_Y, method="min", axis=1) - 1.
    
    loss_value = spin_imbalance(d_X=d_X, r_Y=r_Y, lambda_factor=0.01)
    print(f"Initial value: {loss_value}.")

    # Initialise optimiser -------
    optimizer = optax.chain(optax.adam(learning_rate=learning_rate), optax.keep_params_nonnegative())
    opt_state = optimizer.init(params)
    update_k = jax.jit(cosine_decay_schedule(init_value=k_init_value, decay_steps=n_iter, alpha=k_alpha))
    
    # Allocate memory to store intermediate results -------
    params_training = jnp.empty(shape=(n_iter + 1, len(init_params)))
    imbs_training = jnp.empty(shape=(n_iter + 1,))

    # Store initial vals ------
    params_training = params_training.at[0].set(params)
    imbs_training = imbs_training.at[0].set(loss_value)

    for ii in tqdm.tqdm(range(1, n_iter + 1)):
        d_X = token_manhattan(params, embedding_dimension, n_tokens, X[from_subset], X[to_subset])
        k = jnp.rint(update_k(ii)).astype(int)
        L = update_lambda(k, d_X, lambda_factor)
        loss_value, grads = jax.value_and_grad(spin_imbalance)(params, d_X, r_Y, from_subset, to_subset, L)
        params, opt_state = update_weights(optimizer, params, opt_state, loss_value, grads, init_l2)
        params_training = params_training.at[ii].set(params)
        imbs_training = imbs_training.at[ii].set(loss_value)

    return imbs_training, params_training

def compute_sii_transformers(
    key: jax._src.prng.PRNGKeyArray,
    model_one: Dict[str, float],
    model_two: Dict[str, float],
    frac_row: float = 0.20,
    n_tokens_one: int = None,
    n_tokens_two: int = None,
    preprocess_one: Callable[jnp.ndarray, jnp.ndarray] = None,
    preprocess_two: Callable[jnp.ndarray, jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the spin information imbalance between the spin activations of two transformers.

    Args:
        key: JAX random key.
        model_one: Dictionary of the activations for model 1: keys are the layers.
        model_two: Dictionary of the activations for model 2: keys are the layers.
        frac_row: fraction of rows to consider for splitting the dataset.
        n_tokens_one: number of tokens for model 1.
        n_tokens_two: number of tokens for model 2.
        preprocess_one: function to preprocess activations of model 1 (e.g. mean).
        preprocess_two: function to preprocess activations of model 2 (e.g. mean).

    Returns:
        A tuple with the information imbalance matrices from 1 to 2 and from 2 to 1.
    """
    model_one_layers = len(model_one.keys())
    subset_size, max_length_one, emb_dim_one = model_one["0"].shape
    if n_tokens_one is not None:
        assert n_tokens_one > 0, "Number of tokens for model 1 must be positive."
        assert n_tokens_one <= max_length_one, f"Asking for {n_tokens_one}, but there are {max_length_one} tokens."
    else:
        n_tokens_one = max_length_one
    
    model_two_layers = len(model_two.keys())
    subset_size_two, max_length_two, emb_dim_two = model_two["0"].shape
    assert subset_size == subset_size_two, "Model one and model two have different subset sizes"
    if n_tokens_two is not None:
        assert n_tokens_one > 0, "Number of tokens for model 2 must be positive."
        assert n_tokens_two <= max_length_two, f"Asking for {n_tokens_two}, but there are {max_length_two} tokens."
    else:
        n_tokens_two = max_length_two

    n_row_samples = jnp.floor(subset_size * frac_row).astype(int)
    n_col_samples = (subset_size - n_row_samples).astype(int)

    indices = jnp.arange(subset_size)
    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=subset_size, shape=(n_row_samples,), replace=False)
    remaining_indices = jnp.setdiff1d(indices, indices_rows)
    key, subkey = random.split(key)
    indices_columns = random.choice(key=subkey, a=remaining_indices, shape=(n_col_samples,), replace=False)

    one_to_two = jnp.zeros((model_one_layers, model_two_layers))
    two_to_one = jnp.zeros((model_two_layers, model_one_layers))

    for ii, jj in itertools.product(range(model_one_layers), range(model_two_layers)):   
        model_one_token = model_one[f"{ii}"][:, -n_tokens_one:, :].reshape(subset_size, n_tokens_one * emb_dim_one) if preprocess_one is None else preprocess_one(model_one[f"{ii}"][:, -n_tokens_one:, :])
        model_two_token = model_two[f"{jj}"][:, -n_tokens_two:, :].reshape(subset_size, n_tokens_two * emb_dim_two) if preprocess_two is None else preprocess_two(model_one[f"{ii}"][:, -n_tokens_two:, :])
        one_to_two = one_to_two.at[ii, jj].set(
            spin_information_imbalance(model_one_token, model_two_token, indices_rows, indices_columns)
        )
        two_to_one = two_to_one.at[jj, ii].set(
            spin_information_imbalance(model_two_token, model_one_token, indices_rows, indices_columns)
        )
    return (one_to_two, two_to_one)

@jax.jit
def token_mean(layer: jnp.ndarray) -> jnp.ndarray:
    """Return the token-wise mean.

    Args:
        layer: the layer we take the mean on.

    Returns:
        The token-wise mean.
    """
    subset_size, n_token, emb_dim = layer.shape
    return jnp.mean(layer, axis=1).reshape(subset_size, emb_dim)