#!/usr/bin/env python
# coding: utf-8

# # Asymmetry of II
import os
from pathlib import Path

# Ensure matplotlib cache stays inside the workspace (avoids permission issues)
mpl_cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import jax
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # Requires LaTeX
from information_imbalance.imbalance import build_information_imbalance, l2_distance as np_l2_distance

from cka import build_cka_distance


def l2_distance_jax(x, y):
    """JAX-friendly L2 distance for CKA computations."""
    return jnp.linalg.norm(x - y)

try:
    plt.style.use(['nature', 'science'])  # Match notebook styling
except Exception:
    pass

plt.figure(figsize=(3.5, 2.))


D = 10
L_DEFAULT = 10

def rank_schedule_middle_high(L: int, min_rank: int = 3, max_rank: int = 10) -> jnp.ndarray:
    """
    Rank is LOW at l=0 and l=L-1, and HIGH around the middle.
    """
    l = jnp.arange(L)
    center = (L - 1) / 2.0
    dist = jnp.abs(l - center)
    max_dist = center if center > 0 else 1.0

    # dist = 0 => max_rank, dist = max_dist => min_rank
    ranks = max_rank - (max_rank - min_rank) * (dist / max_dist)
    ranks = jnp.rint(ranks).astype(jnp.int32)

    # enforce borders explicitly
    ranks = ranks.at[0].set(min_rank)
    ranks = ranks.at[L - 1].set(min_rank)
    return ranks

def noise_schedule_middle_low(L: int,
                              sigma_center: float = 0.02,
                              sigma_border: float = 0.2) -> jnp.ndarray:
    """
    Noise is LOW in the middle layers and HIGH at the borders.
    sigma_center: noise at center (best predictability)
    sigma_border: noise at l=0 and l=L-1 (worst predictability)
    """
    l = jnp.arange(L)
    center = (L - 1) / 2.0
    dist = jnp.abs(l - center)
    max_dist = center if center > 0 else 1.0

    # dist = 0 => sigma_center, dist = max_dist => sigma_border
    sigmas = sigma_center + (sigma_border - sigma_center) * (dist / max_dist)

    # enforce borders explicitly
    sigmas = sigmas.at[0].set(sigma_border)
    sigmas = sigmas.at[L - 1].set(sigma_border)
    return sigmas

def make_rank_matrix(key, rank: int, dim: int = 10) -> jnp.ndarray:
    key_u, key_v = random.split(key)
    U = random.normal(key_u, (dim, rank))
    V = random.normal(key_v, (rank, dim))
    return U @ V


def build_B_stack(key,
                  L: int = L_DEFAULT,
                  min_rank: int = 3,
                  max_rank: int = 10,
                  dim: int = 10) -> jnp.ndarray:
    """
    Build B_l with low ranks at borders and high ranks in the middle.
    Shape: (L, dim, dim)
    """
    ranks = rank_schedule_middle_high(L, min_rank=min_rank, max_rank=max_rank)
    keys = random.split(key, L)

    Bs = []
    for l in range(L):
        r = int(ranks[l])
        k = keys[l]
        Bs.append(make_rank_matrix(k, r, dim=dim))

    B = jnp.stack(Bs, axis=0)
    return B, ranks


def simulate_XY(
    key,
    L: int = L_DEFAULT,
    n_samples: int = 500,
    tau: float = 1.0,
    dim: int = 10,
    mu: jnp.ndarray | None = None,
    min_rank: int = 3,
    max_rank: int = 10,
    sigma_center: float = 0.02,
    sigma_border: float = 0.2,
):
    """
    Simulate:
      X_l ~ N(mu_l, tau^2 I)
      Y_l = B_l X_l + eps_l
      eps_l ~ N(0, sigma_l^2 I), with sigma_l low in the middle and high at borders.

    Returns:
      X: (L, N, dim)
      Y: (L, N, dim)
      B: (L, dim, dim)
      mu: (L, dim)
      ranks_B: (L,)
      sigma_l: (L,)
    """
    key_B, key_X, key_eps = random.split(key, 3)

    B, ranks_B = build_B_stack(key_B, L=L, min_rank=min_rank, max_rank=max_rank, dim=dim)
    sigma_l = noise_schedule_middle_low(L, sigma_center=sigma_center, sigma_border=sigma_border)

    if mu is None:
        mu = jnp.zeros((L, dim))

    # X
    X_noise = random.normal(key_X, (L, n_samples, dim))
    X = mu[:, None, :] + tau * X_noise

    # eps with layer-wise noise
    eps_noise = random.normal(key_eps, (L, n_samples, dim))
    eps = sigma_l[:, None, None] * eps_noise  # broadcast over samples & dims

    # Y_l = B_l X_l + eps_l
    Y = jnp.einsum("lij,lnj->lni", B, X) + eps

    return X, Y, B, mu, ranks_B, sigma_l

def compute_imbalance_over_layers(
    X_lnd: jnp.ndarray,
    Y_lnd: jnp.ndarray,
    k: int = 1,
    maxk: int | None = 80,
    rng_seed: int = 0,
):
    L, N, _ = X_lnd.shape

    if maxk is None:
        maxk = N - 1

    imb_fn = build_information_imbalance(
        distance_fn=np_l2_distance,
        k=k,
        maxk=maxk,
        rng=rng_seed,
    )

    imbal_X_to_Y = np.zeros(L)
    imbal_Y_to_X = np.zeros(L)

    for l in range(L):
        Xl = np.asarray(X_lnd[l])
        Yl = np.asarray(Y_lnd[l])
        d_xy, d_yx = imb_fn(Xl, Yl)
        imbal_X_to_Y[l] = d_xy
        imbal_Y_to_X[l] = d_yx

    return imbal_X_to_Y, imbal_Y_to_X

key = random.PRNGKey(0)
L = 10

# 1) simulate with layer-wise noise
X_lnd, Y_lnd, B_lij, mu, ranks_B, sigma_l = simulate_XY(
    key,
    L=L,
    n_samples=500,
    tau=1.0,
    dim=10,
    min_rank=3,
    max_rank=10,
    sigma_center=0.02,
    sigma_border=0.2,
)

# 2) information imbalance
imb_XY, imb_YX = compute_imbalance_over_layers(
    X_lnd,
    Y_lnd,
    k=1,
    maxk=80,
    rng_seed=0,
)

layers = np.arange(L)

# 3) cka - compute CKA between X and Y within each layer
cka = []
for l in range(L):
    cka_val = build_cka_distance(X_lnd[l], Y_lnd[l], distance_fn=l2_distance_jax)
    cka.append(float(cka_val))
cka = np.array(cka)

# Plot all on same y axis with notebook-style settings
rel_depth = layers / L
plt.plot(rel_depth, imb_XY, '-o', color="#364F6B", markersize=3, markeredgecolor='black', markeredgewidth=0.3, label=r"$\Delta(X \rightarrow Y)$")
plt.plot(rel_depth, imb_YX, '-o', color="#609966", markersize=3, markeredgecolor='black', markeredgewidth=0.3, label=r"$\Delta(Y \rightarrow X)$")
plt.plot(rel_depth, cka, '-o', color="#C44536", markersize=3, markeredgecolor='black', markeredgewidth=0.3, label="CKA")
plt.xlabel("Relative depth", fontsize=11)
plt.ylabel("Information imbalance", fontsize=11)
plt.ylim(0, 1.0)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.legend(fontsize=6, loc="upper right")
plt.tight_layout()
plt.savefig("f1_a.pdf", dpi=300)
plt.savefig("f1_a.png", dpi=300)
