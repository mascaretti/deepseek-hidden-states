import jax
from jax import random
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, *args, **kwargs):
        return iterable

import jax.numpy as jnp


def build_cka_distance(X, Y, distance_fn, chunk_size=100):
    """Compute CKA with memory-efficient chunking"""
    n, p = X.shape
    assert Y.shape[0] == n
    
    # Compute distance matrices in chunks to avoid OOM
    def compute_distance_matrix_chunked(Z):
        n = Z.shape[0]
        distances = []
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            chunk_distances = []
            for j in range(0, n, chunk_size):
                j_end = min(j + chunk_size, n)
                # Compute distances for this chunk
                chunk = jax.vmap(lambda x: jax.vmap(lambda y: distance_fn(x, y))(Z[j:j_end]))(Z[i:i_end])
                chunk_distances.append(chunk.flatten())
            distances.append(jnp.concatenate(chunk_distances))
        return jnp.concatenate(distances)
    
    distance_X = compute_distance_matrix_chunked(X)
    distance_Y = compute_distance_matrix_chunked(Y)
    
    # Center the distances
    distance_X = distance_X - jnp.mean(distance_X)
    distance_Y = distance_Y - jnp.mean(distance_Y)
    
    # Compute CKA
    cka_distance = jnp.sum(distance_X * distance_Y) / (jnp.sqrt(jnp.sum(distance_X ** 2)) * jnp.sqrt(jnp.sum(distance_Y ** 2)))
    return cka_distance


def compute_cka(key, Y, distance_fn, n_data=200):
    n, p = Y.shape
    if p < n_data:
        print(f"Warning: p={p} < n_data={n_data}, setting n_data=p")
        n_data = p
    key, subkey = random.split(key)
    # information_imbalance_fn = build_information_imbalance(key=subkey, sample_size=n, distance_fn=distance_fn)
    cka = []
    for jj in tqdm(jnp.arange(1, p, p//n_data)):
        cka.append(np.array(build_cka_distance(Y, Y[:, :jj], distance_fn=distance_fn)))
    return np.asarray(cka)
