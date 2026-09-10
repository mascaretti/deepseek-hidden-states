"""
Tests for compute_pca_mean.py — per-layer PCA of mean-pooled activations.

Covers both decomposition branches (covariance side d<=N, Gram side d>N), the
Var(score_k) == lambda_k invariant, agreement with the SVD reference and with
the pre-existing spectral_entropy() in test_distance_strategies.py, and an
end-to-end CLI run whose projections are read back through the loader
compute_ii_mean.py uses.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from safetensors.numpy import save_file

from compute_pca_mean import pca_layer
from compute_ii_mean import load_mean_activations
from test_distance_strategies import spectral_entropy

REPO = Path(__file__).resolve().parent.parent


def _low_rank(n, d, rank, seed=0):
    """(n, d) matrix of exact rank `rank` (plus a non-zero mean offset)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, d))
    return (x + 5.0 * rng.standard_normal(d)).astype(np.float32)


# ---------------------------------------------------------------------------
# pca_layer
# ---------------------------------------------------------------------------

class TestPCALayer:
    @pytest.mark.parametrize("n,d", [(60, 12), (12, 60)])   # covariance side, Gram side
    def test_matches_svd_reference(self, n, d):
        x = _low_rank(n, d, rank=5)
        eig, scores = pca_layer(x, n_components=4)

        xc = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)
        _, s, _ = np.linalg.svd(xc, full_matrices=False)
        expected = s ** 2 / (n - 1)

        assert eig.shape == (min(n, d),)
        np.testing.assert_allclose(eig, expected[:min(n, d)], rtol=1e-8, atol=1e-10)
        assert scores.shape == (n, 4)

    @pytest.mark.parametrize("n,d", [(60, 12), (12, 60)])
    def test_score_variance_equals_eigenvalue(self, n, d):
        """Var(score_k) == lambda_k is the whole point of the covariance convention."""
        x = _low_rank(n, d, rank=5)
        eig, scores = pca_layer(x, n_components=min(n, d))
        np.testing.assert_allclose(scores.var(axis=0, ddof=1), eig, rtol=1e-7, atol=1e-10)

    @pytest.mark.parametrize("n,d", [(60, 12), (12, 60)])
    def test_rank_is_recovered(self, n, d):
        """An exactly rank-5 matrix has 5 non-negligible eigenvalues, no negatives."""
        x = _low_rank(n, d, rank=5)
        eig, _ = pca_layer(x, n_components=0)
        assert (eig >= 0).all()
        assert np.sum(eig > 1e-8 * eig[0]) == 5
        assert np.all(np.diff(eig) <= 1e-12)               # descending

    def test_full_projection_is_an_isometry(self):
        """Keeping every PC preserves pairwise distances of the centred data."""
        x = _low_rank(40, 15, rank=15)
        _, scores = pca_layer(x, n_components=15)
        xc = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)

        d_orig = np.linalg.norm(xc[:, None] - xc[None, :], axis=-1)
        d_pca = np.linalg.norm(scores[:, None] - scores[None, :], axis=-1)
        np.testing.assert_allclose(d_pca, d_orig, rtol=1e-8, atol=1e-8)

    def test_no_components_requested(self):
        eig, scores = pca_layer(_low_rank(30, 8, rank=3), n_components=0)
        assert scores is None
        assert eig.shape == (8,)

    def test_entropy_matches_existing_implementation(self):
        """Our spectrum-based H must equal test_distance_strategies.spectral_entropy."""
        x = _low_rank(50, 20, rank=20)
        eig, _ = pca_layer(x, n_components=0)
        p = eig / eig.sum()
        p = p[p > 0]
        ours = -np.sum(p * np.log(p))
        np.testing.assert_allclose(ours, spectral_entropy(x), rtol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_projections_are_readable_by_the_ii_loader(self, tmp_path):
        n_layers, n_per_chunk, d, k = 3, 20, 16, 5
        in_dir = tmp_path / "mean"
        in_dir.mkdir()
        full = {f"layer_{i}": _low_rank(2 * n_per_chunk, d, rank=6, seed=i)
                for i in range(n_layers)}
        for c in range(2):
            sl = slice(c * n_per_chunk, (c + 1) * n_per_chunk)
            save_file({key: np.ascontiguousarray(v[sl]) for key, v in full.items()},
                      str(in_dir / f"mean_chunk_{c}.safetensors"))

        npz_path = tmp_path / "pca.npz"
        proj_dir = tmp_path / "proj"
        result = subprocess.run(
            [sys.executable, str(REPO / "compute_pca_mean.py"),
             "--input_path", str(in_dir),
             "--output_npz", str(npz_path),
             "--subset_size", str(2 * n_per_chunk),
             "--project_k", str(k),
             "--project_out", str(proj_dir)],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert result.returncode == 0, result.stderr

        out = np.load(npz_path)
        assert out["eigenvalues"].shape == (n_layers, min(2 * n_per_chunk, d))
        assert out["evr_cumsum"][:, -1] == pytest.approx(1.0, abs=1e-5)
        # rank-6 data: 90% of the variance well inside the first 6 PCs
        assert (out["n_pcs_90"] <= 6).all()
        assert (out["participation_ratio"] > 1.0).all()

        # The projected folder must be consumable by compute_ii_mean.py unchanged
        loaded = load_mean_activations(str(proj_dir), subset_size=2 * n_per_chunk)
        assert len(loaded) == n_layers
        assert loaded["layer_0"].shape == (2 * n_per_chunk, k)
        assert loaded["layer_0"].dtype == np.float32

        # ...and must match a direct call on the concatenated input
        eig, scores = pca_layer(full["layer_0"], n_components=k)
        np.testing.assert_allclose(loaded["layer_0"], scores.astype(np.float32),
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out["eigenvalues"][0], eig.astype(np.float32),
                                   rtol=1e-4, atol=1e-6)
