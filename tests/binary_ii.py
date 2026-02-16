"""
Smoke tests for Information Imbalance: Hamming vs L2 (raw and normalised).

Gaussian setup:
    Y ~ N(0, I_p)  with p=100, N=2000
    X shares the first floor(alpha * p) components with Y,
    the rest are independent Gaussian noise.

As alpha goes from 0 to 1, II should decrease monotonically from ~1 to 0.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from information_imbalance_module import (
    L2_distance,
    build_information_imbalance,
    hamming_distance,
)
from utils import binarize


def l2_normalize(a):
    norms = jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a / jnp.maximum(norms, 1e-8)


def gaussian_ii(distance_fn, preprocess, N=2000, p=100, n_alphas=11, seed=0):
    """Compute II(X->Y) for a sweep of alpha values.

    Returns:
        alphas: (n_alphas,) array of alpha values
        ii_fwd: (n_alphas,) array of II(X->Y)
        ii_rev: (n_alphas,) array of II(Y->X)
    """
    alphas = np.linspace(0, 1, n_alphas)

    key = jax.random.key(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)

    Y = jax.random.normal(k1, (N, p))
    Z = jax.random.normal(k2, (N, p))

    ii_fn = build_information_imbalance(
        key=k3, sample_size=N, distance_fn=distance_fn
    )

    ii_fwd, ii_rev = [], []
    for alpha in alphas:
        n_shared = int(round(alpha * p))
        X = jnp.concatenate([Y[:, :n_shared], Z[:, n_shared:]], axis=1)
        Xp, Yp = preprocess(X), preprocess(Y)
        fwd, rev = ii_fn(Xp, Yp)
        ii_fwd.append(float(fwd))
        ii_rev.append(float(rev))

    return alphas, np.array(ii_fwd), np.array(ii_rev)


def check_monotonic_decrease(values, label, tolerance=0.05):
    """Check that values are roughly monotonically decreasing.

    Allow small violations up to `tolerance` to account for finite-sample noise.
    """
    violations = []
    for i in range(len(values) - 1):
        increase = values[i + 1] - values[i]
        if increase > tolerance:
            violations.append((i, increase))
    if violations:
        print(f"  WARN [{label}]: {len(violations)} monotonicity violations > {tolerance}:")
        for idx, inc in violations:
            print(f"    alpha index {idx} -> {idx+1}: +{inc:.4f}")
    else:
        print(f"  OK   [{label}]: monotonically decreasing (tol={tolerance})")
    return len(violations) == 0


def check_endpoints(ii_fwd, ii_rev, label):
    """Check that II ~ 1 at alpha=0 and II ~ 0 at alpha=1."""
    ok = True
    if ii_fwd[0] < 0.7:
        print(f"  WARN [{label} fwd]: II at alpha=0 is {ii_fwd[0]:.4f}, expected > 0.7")
        ok = False
    if ii_fwd[-1] > 0.05:
        print(f"  WARN [{label} fwd]: II at alpha=1 is {ii_fwd[-1]:.4f}, expected < 0.05")
        ok = False
    if ii_rev[0] < 0.7:
        print(f"  WARN [{label} rev]: II at alpha=0 is {ii_rev[0]:.4f}, expected > 0.7")
        ok = False
    if ii_rev[-1] > 0.05:
        print(f"  WARN [{label} rev]: II at alpha=1 is {ii_rev[-1]:.4f}, expected < 0.05")
        ok = False
    if ok:
        print(f"  OK   [{label}]: endpoints correct "
              f"(fwd: {ii_fwd[0]:.3f}->{ii_fwd[-1]:.3f}, "
              f"rev: {ii_rev[0]:.3f}->{ii_rev[-1]:.3f})")
    return ok


def main():
    configs = {
        "hamming": (hamming_distance, binarize),
        "l2_raw": (L2_distance, lambda x: x),
        "l2_normalized": (L2_distance, l2_normalize),
    }

    results = {}
    for name, (dist_fn, preproc) in configs.items():
        print(f"\n--- {name} ---")
        alphas, ii_fwd, ii_rev = gaussian_ii(dist_fn, preproc)
        results[name] = (alphas, ii_fwd, ii_rev)

        check_endpoints(ii_fwd, ii_rev, name)
        check_monotonic_decrease(ii_fwd, f"{name} fwd")
        check_monotonic_decrease(ii_rev, f"{name} rev")

    # Print comparison table
    alphas = results["hamming"][0]
    print(f"\n{'alpha':>6}", end="")
    for name in configs:
        print(f"  {name + ' fwd':>16}  {name + ' rev':>16}", end="")
    print()
    print("-" * (6 + len(configs) * 36))
    for i, a in enumerate(alphas):
        print(f"{a:>6.2f}", end="")
        for name in configs:
            _, fwd, rev = results[name]
            print(f"  {fwd[i]:>16.4f}  {rev[i]:>16.4f}", end="")
        print()


if __name__ == "__main__":
    main()
