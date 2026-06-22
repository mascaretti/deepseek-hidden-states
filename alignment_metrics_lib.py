"""
Building blocks for comparing Information Imbalance against CKA, mutual-kNN
(Neighborhood Overlap), mean CCA, and RSA, under several data-generating
processes:

    DGP-G : Y = alpha X + sqrt(1-alpha^2) eps,  eps ~ N(0, I_p)
    DGP-T : Y = alpha X + sqrt(1-alpha^2) eps,  eps ~ t_nu(0, I_p)  (iid coords)
    DGP-A : Y = alpha X + sqrt(1-alpha^2) eps,  eps ~ N(0, diag(d))   (anisotropic)
    DGP-N : Y = f(X) + sigma eps,               f(x) = x**2 (non-bijective)

For DGP-{G, T, A}: H(Y|X) = h(eps) + (p/2) log(1 - alpha^2).
For DGP-N:        H(Y|X) = h(sigma * eps)  (independent of f, since f(X) is
                  determined by X). Here sigma is the alignment knob.

Nothing in this file runs a full experiment -- the bottom of the script does a
smoke test on tiny data to verify each component works.
"""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import scipy.special as sp


# =============================================================================
# 1. Building blocks
# =============================================================================

@jit
def pairwise_sq_dist(X):
    sq = jnp.sum(X ** 2, axis=1)
    return sq[:, None] + sq[None, :] - 2.0 * X @ X.T

@jit
def ranks_per_row(D):
    """ranks[i, j] = rank of j as neighbor of i (1 = closest); diag -> N."""
    N = D.shape[0]
    Dm = D + jnp.eye(N) * 1e10
    order = jnp.argsort(Dm, axis=1)
    rows = jnp.arange(N)[:, None]
    cols = jnp.arange(N)[None, :]
    ranks = jnp.zeros_like(order)
    return ranks.at[rows, order].set(cols + 1)

@jit
def ordinal_rank(x):
    """Ordinal ranks of a 1D vector (1 = smallest)."""
    return jnp.argsort(jnp.argsort(x)) + 1

@jit
def pearson_corr(x, y):
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    return jnp.sum(x * y) / (jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y)) + 1e-12)


# =============================================================================
# 2. Alignment metrics
# =============================================================================

@jit
def information_imbalance(X, Y):
    """Δ(X → Y) ∈ [0, 1].  Lower means X better predicts Y."""
    DX = pairwise_sq_dist(X)
    DY = pairwise_sq_dist(Y)
    N = X.shape[0]
    nn_X = jnp.argmin(DX + jnp.eye(N) * 1e10, axis=1)
    R_Y  = ranks_per_row(DY)
    r_y  = R_Y[jnp.arange(N), nn_X]
    return 2.0 / (N - 1) * jnp.mean(r_y)

@partial(jit, static_argnums=(2,))
def neighborhood_overlap(X, Y, k=10):
    """Mutual-kNN agreement (Huh et al. 2024 'NO'). Symmetric."""
    N = X.shape[0]
    DX = pairwise_sq_dist(X) + jnp.eye(N) * 1e10
    DY = pairwise_sq_dist(Y) + jnp.eye(N) * 1e10
    nn_X = jnp.argsort(DX, axis=1)[:, :k]
    nn_Y = jnp.argsort(DY, axis=1)[:, :k]
    rows = jnp.arange(N)[:, None]
    mask_X = jnp.zeros((N, N), dtype=bool).at[rows, nn_X].set(True)
    mask_Y = jnp.zeros((N, N), dtype=bool).at[rows, nn_Y].set(True)
    return jnp.mean(jnp.sum(mask_X & mask_Y, axis=1) / k)

@jit
def linear_cka(X, Y):
    """Linear CKA via centered Gram matrices (Kornblith et al. 2019b)."""
    N  = X.shape[0]
    Hc = jnp.eye(N) - jnp.ones((N, N)) / N
    K  = Hc @ (X @ X.T) @ Hc
    L  = Hc @ (Y @ Y.T) @ Hc
    return jnp.sum(K * L) / jnp.sqrt(jnp.sum(K * K) * jnp.sum(L * L) + 1e-12)

@jit
def mean_cca(X, Y):
    """Mean canonical correlation. Centered SVD-based formulation."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Ux, _, _ = jnp.linalg.svd(Xc, full_matrices=False)
    Uy, _, _ = jnp.linalg.svd(Yc, full_matrices=False)
    sv = jnp.linalg.svd(Ux.T @ Uy, compute_uv=False)
    return jnp.mean(sv)

@jit
def rsa_spearman(X, Y):
    """Representational Similarity Analysis: Spearman rank correlation
    between the upper triangles of the two pairwise-distance matrices."""
    N  = X.shape[0]
    DX = pairwise_sq_dist(X)
    DY = pairwise_sq_dist(Y)
    iu = jnp.triu_indices(N, k=1)
    dx = DX[iu]
    dy = DY[iu]
    return pearson_corr(ordinal_rank(dx).astype(jnp.float32),
                        ordinal_rank(dy).astype(jnp.float32))

ALL_METRICS = {
    "II_x_to_y": lambda X, Y: information_imbalance(X, Y),
    "II_y_to_x": lambda X, Y: information_imbalance(Y, X),
    "NO":       lambda X, Y: neighborhood_overlap(X, Y, 10),
    "CKA":      lambda X, Y: linear_cka(X, Y),
    "mean_CCA": lambda X, Y: mean_cca(X, Y),
    "RSA":      lambda X, Y: rsa_spearman(X, Y),
}


# =============================================================================
# 3. Data generators
# =============================================================================

def _split(key, n):
    return jax.random.split(key, n)

def sample_gaussian_pair(key, N, p, alpha):
    """DGP-G: Y = αX + sqrt(1-α²) ε, X, ε iid N(0, I_p)."""
    k1, k2 = _split(key, 2)
    X   = jax.random.normal(k1, (N, p))
    eps = jax.random.normal(k2, (N, p))
    Y   = alpha * X + jnp.sqrt(1.0 - alpha ** 2) * eps
    return X, Y

def sample_tstudent_pair(key, N, p, alpha, nu):
    """DGP-T: same as DGP-G but ε iid Student-t_ν per coordinate, rescaled
    to unit variance so that α controls SNR identically to the Gaussian case.
    Requires ν > 2 (otherwise variance is undefined)."""
    if nu <= 2:
        raise ValueError("nu must be > 2 for unit-variance normalization")
    k1, k2 = _split(key, 2)
    X    = jax.random.normal(k1, (N, p))
    raw  = jax.random.t(k2, df=nu, shape=(N, p))
    eps  = raw * jnp.sqrt((nu - 2) / nu)            # now Var(eps) = 1
    Y    = alpha * X + jnp.sqrt(1.0 - alpha ** 2) * eps
    return X, Y

def sample_anisotropic_pair(key, N, p, alpha, scales):
    """DGP-A: Gaussian noise but anisotropic (scales is shape (p,))."""
    k1, k2 = _split(key, 2)
    X   = jax.random.normal(k1, (N, p))
    eps = jax.random.normal(k2, (N, p)) * scales[None, :]
    Y   = alpha * X + jnp.sqrt(1.0 - alpha ** 2) * eps
    return X, Y

def sample_nonlinear_pair(key, N, p, sigma, kind="square"):
    """DGP-N: Y = f(X) + σ ε, with f non-bijective per-coordinate.
    kind ∈ {'square', 'abs', 'sin'}.  All are non-injective."""
    k1, k2 = _split(key, 2)
    X = jax.random.normal(k1, (N, p))
    if kind == "square":
        fX = X ** 2 - 1.0           # subtract mean for centering
    elif kind == "abs":
        fX = jnp.abs(X) - jnp.sqrt(2.0 / jnp.pi)
    elif kind == "sin":
        fX = jnp.sin(2.0 * X)
    else:
        raise ValueError(kind)
    eps = jax.random.normal(k2, (N, p))
    Y   = fX + sigma * eps
    return X, Y


# =============================================================================
# 4. Analytical conditional entropy
# =============================================================================

def _h_univariate_t(nu):
    """Differential entropy of a univariate Student-t with df=ν, scale=1."""
    log_beta = sp.gammaln(nu / 2) + sp.gammaln(0.5) - sp.gammaln((nu + 1) / 2)
    return (0.5 * np.log(nu)
            + log_beta
            + (nu + 1) / 2 * (sp.digamma((nu + 1) / 2) - sp.digamma(nu / 2)))

H_GAUSS_UNIV = 0.5 * np.log(2 * np.pi * np.e)

def H_cond_gaussian(alpha, p):
    return p * H_GAUSS_UNIV + 0.5 * p * np.log(1.0 - alpha ** 2)

def H_cond_tstudent(alpha, p, nu):
    """H(Y|X) for unit-variance-normalized t_ν noise (matching the sampler).
    h(c · ε) = h(ε) + p·log(c), and ε here is t_ν rescaled by √((ν-2)/ν),
    which contributes -p·(1/2)·log(ν/(ν-2)) relative to raw t_ν."""
    h_eps = p * (_h_univariate_t(nu) - 0.5 * np.log(nu / (nu - 2)))
    return h_eps + 0.5 * p * np.log(1.0 - alpha ** 2)

def H_cond_anisotropic(alpha, scales):
    """scales ∈ R^p, Gaussian noise per coord with given std-devs."""
    p = len(scales)
    return p * H_GAUSS_UNIV + np.sum(np.log(scales)) + 0.5 * p * np.log(1.0 - alpha ** 2)

def H_cond_nonlinear(sigma, p):
    """For Y = f(X) + σ ε with ε ~ N(0,I), H(Y|X) = h(σε) (independent of f)."""
    return p * H_GAUSS_UNIV + p * np.log(sigma)


# =============================================================================
# 5. Experiment runners (parametric — not executed automatically)
# =============================================================================

def run_sweep(sample_fn, knobs, N, p, n_seeds, metric_names=None,
              extra_args=()):
    """Generic sweep over one knob.

    sample_fn(key, N, p, knob, *extra_args) -> (X, Y)
    knobs: array of values for the swept variable
    Returns dict {metric_name: array of shape (n_seeds, len(knobs))}.
    """
    if metric_names is None:
        metric_names = list(ALL_METRICS.keys())
    out = {m: np.zeros((n_seeds, len(knobs))) for m in metric_names}
    key = jax.random.PRNGKey(0)
    for s in range(n_seeds):
        for ki, knob in enumerate(knobs):
            key, subkey = _split(key, 2)
            X, Y = sample_fn(subkey, N, p, knob, *extra_args)
            for m in metric_names:
                out[m][s, ki] = float(ALL_METRICS[m](X, Y))
    return out

# Convenience runners (return all metrics + analytical H(Y|X))

def experiment_gaussian(N=1500, p=10, n_alphas=15, n_seeds=4):
    alphas = np.linspace(0.0, 0.97, n_alphas)
    H = np.array([H_cond_gaussian(a, p) for a in alphas])
    out = run_sweep(sample_gaussian_pair, alphas, N, p, n_seeds)
    return alphas, H, out

def experiment_tstudent(N=1500, p=10, nu=3.0, n_alphas=15, n_seeds=4):
    alphas = np.linspace(0.0, 0.97, n_alphas)
    H = np.array([H_cond_tstudent(a, p, nu) for a in alphas])
    out = run_sweep(sample_tstudent_pair, alphas, N, p, n_seeds,
                    extra_args=(nu,))
    return alphas, H, out

def experiment_anisotropic(N=1500, p=10, condition=20.0,
                           n_alphas=15, n_seeds=4):
    """Diagonal scales spread geometrically with given condition number."""
    scales = jnp.geomspace(1.0, condition, p)
    alphas = np.linspace(0.0, 0.97, n_alphas)
    H = np.array([H_cond_anisotropic(a, np.array(scales)) for a in alphas])
    out = run_sweep(sample_anisotropic_pair, alphas, N, p, n_seeds,
                    extra_args=(scales,))
    return alphas, H, out

def experiment_nonlinear(N=1500, p=10, kind="square",
                         n_sigmas=15, sigma_min=0.05, sigma_max=3.0, n_seeds=4):
    sigmas = np.geomspace(sigma_min, sigma_max, n_sigmas)
    H = np.array([H_cond_nonlinear(s, p) for s in sigmas])
    out = run_sweep(sample_nonlinear_pair, sigmas, N, p, n_seeds,
                    extra_args=(kind,))
    return sigmas, H, out


# =============================================================================
# 6. Smoke test (tiny N, just to verify everything wires up)
# =============================================================================

def _smoke():
    print("=== smoke test (N=200) ===")
    N, p = 200, 6
    key = jax.random.PRNGKey(42)

    # one (X, Y) sample of each kind, all metrics evaluated
    samplers = {
        "Gaussian   alpha=0.7":      lambda k: sample_gaussian_pair(k, N, p, 0.7),
        "t-Student  alpha=0.7 ν=3":  lambda k: sample_tstudent_pair(k, N, p, 0.7, 3.0),
        "Anisotropic alpha=0.7 κ=20": lambda k: sample_anisotropic_pair(
                                          k, N, p, 0.7, jnp.geomspace(1.0, 20.0, p)),
        "Nonlinear  sigma=0.3 sq":   lambda k: sample_nonlinear_pair(k, N, p, 0.3, "square"),
        "Nonlinear  sigma=2.0 sq":   lambda k: sample_nonlinear_pair(k, N, p, 2.0, "square"),
    }
    rows = []
    for name, sampler in samplers.items():
        key, sub = _split(key, 2)
        X, Y = sampler(sub)
        vals = {m: float(ALL_METRICS[m](X, Y)) for m in ALL_METRICS}
        rows.append((name, vals))

    # print as table
    metric_names = list(ALL_METRICS.keys())
    header = f"{'setting':35s}  " + "  ".join(f"{m:>9s}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for name, vals in rows:
        cells = "  ".join(f"{vals[m]:>9.3f}" for m in metric_names)
        print(f"{name:35s}  {cells}")

    # check analytical entropies
    print("\nanalytical H(Y|X)  [nats]:")
    print(f"  Gaussian  α=0.7 p=6:                   {H_cond_gaussian(0.7, 6):.3f}")
    print(f"  t-Student α=0.7 ν=3 p=6:               {H_cond_tstudent(0.7, 6, 3):.3f}")
    print(f"  Nonlinear σ=0.3 p=6:                   {H_cond_nonlinear(0.3, 6):.3f}")
    print(f"  Nonlinear σ=2.0 p=6:                   {H_cond_nonlinear(2.0, 6):.3f}")


if __name__ == "__main__":
    _smoke()
