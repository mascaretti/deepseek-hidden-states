"""
Tests for:
  - winsorized_mean / process_image_chunk  (preprocess_mean.py, torch-based)
  - mean_center / l2_normalize            (test_distance_strategies.py, jax-based)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from preprocess_mean import winsorized_mean, process_image_chunk
from test_distance_strategies import mean_center, l2_normalize
from information_imbalance_module import build_information_imbalance, L2_distance


# ---------------------------------------------------------------------------
# winsorized_mean
# ---------------------------------------------------------------------------

class TestWinsorizedMean:
    def _t(self, arr):
        """Helper: numpy array → float32 torch tensor."""
        return torch.tensor(arr, dtype=torch.float32)

    def test_output_shape(self):
        x = self._t(np.random.randn(4, 20, 8))   # (batch, tokens, embed)
        out = winsorized_mean(x, dim=1)
        assert out.shape == (4, 8)

    def test_uniform_tokens_unchanged(self):
        """If every token is identical the mean equals that value."""
        val = np.array([1.0, -2.0, 3.0])
        x = self._t(np.tile(val, (2, 15, 1)))     # (2, 15, 3) — all tokens equal
        out = winsorized_mean(x, dim=1)
        np.testing.assert_allclose(out.numpy(), np.tile(val, (2, 1)), atol=1e-5)

    def test_outlier_is_clipped(self):
        """A single extreme token should shift the mean less than a raw mean would."""
        rng = np.random.default_rng(0)
        base = rng.standard_normal((1, 50, 4)).astype(np.float32)   # (1, 50, 4)

        # Spike one token to a huge value
        spiked = base.copy()
        spiked[0, 0, :] = 1000.0

        raw_mean   = spiked.mean(axis=1)           # strongly pulled by spike
        wins_mean  = winsorized_mean(self._t(spiked), dim=1).numpy()
        clean_mean = base.mean(axis=1)

        # Winsorized mean should be closer to the clean mean than the raw mean is
        err_wins = np.abs(wins_mean - clean_mean).mean()
        err_raw  = np.abs(raw_mean  - clean_mean).mean()
        assert err_wins < err_raw, (
            f"Winsorized mean (err={err_wins:.4f}) should be closer to clean mean "
            f"than raw mean (err={err_raw:.4f})"
        )

    def test_output_within_data_range(self):
        """Output values must not exceed the input's extreme values."""
        x = self._t(np.random.randn(3, 30, 6).astype(np.float32))
        out = winsorized_mean(x, dim=1)
        assert float(out.min()) >= float(x.min()) - 1e-5
        assert float(out.max()) <= float(x.max()) + 1e-5

    def test_dtype_is_float(self):
        x = self._t(np.random.randn(2, 10, 4))
        out = winsorized_mean(x, dim=1)
        assert out.dtype in (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# process_image_chunk (last_k slicing)
# ---------------------------------------------------------------------------

class TestProcessImageChunk:
    def _chunk(self, batch=4, tokens=20, embed=8, n_layers=3):
        """Synthetic image chunk: {layer_i: (batch, tokens, embed)}."""
        rng = np.random.default_rng(1)
        return {
            f"layer_{i}": torch.tensor(
                rng.standard_normal((batch, tokens, embed)), dtype=torch.float32
            )
            for i in range(n_layers)
        }

    def test_output_shape_all_tokens(self):
        chunk = self._chunk(batch=4, tokens=20, embed=8, n_layers=3)
        result = process_image_chunk(chunk)
        for key, arr in result.items():
            assert arr.shape == (4, 8), f"{key}: expected (4, 8), got {arr.shape}"

    def test_output_shape_last_k(self):
        chunk = self._chunk(batch=4, tokens=20, embed=8, n_layers=3)
        result = process_image_chunk(chunk, last_k=5)
        for key, arr in result.items():
            assert arr.shape == (4, 8), f"{key}: expected (4, 8), got {arr.shape}"

    def test_last_k_uses_correct_tokens(self):
        """process_image_chunk with last_k=k should equal winsorized_mean of the last k tokens."""
        batch, tokens, embed = 2, 10, 4
        rng = np.random.default_rng(2)
        raw = torch.tensor(rng.standard_normal((batch, tokens, embed)), dtype=torch.float32)
        chunk = {"layer_0": raw}

        k = 3
        result_k = process_image_chunk(chunk, last_k=k)["layer_0"]
        expected  = winsorized_mean(raw[:, -k:, :], dim=1).numpy().astype(np.float32)
        np.testing.assert_allclose(result_k, expected, atol=1e-5)

    def test_last_k_none_uses_all_tokens(self):
        """last_k=None and last_k=0 should both use all tokens."""
        chunk = self._chunk(batch=2, tokens=15, embed=4, n_layers=2)
        r_none = process_image_chunk(chunk, last_k=None)
        r_zero = process_image_chunk(chunk, last_k=0)
        r_all  = process_image_chunk(chunk)
        for key in r_none:
            np.testing.assert_array_equal(r_none[key], r_all[key])
            np.testing.assert_array_equal(r_zero[key], r_all[key])

    def test_last_k_differs_from_all(self):
        """With a heterogeneous token sequence, last_k result should differ from full mean."""
        batch, tokens, embed = 2, 20, 4
        rng = np.random.default_rng(3)
        # Make last 3 tokens very different from the rest
        raw = torch.zeros(batch, tokens, embed)
        raw[:, -3:, :] = 100.0
        chunk = {"layer_0": raw}

        r_all    = process_image_chunk(chunk)["layer_0"]
        r_last3  = process_image_chunk(chunk, last_k=3)["layer_0"]
        assert not np.allclose(r_all, r_last3), \
            "last_k=3 should give a different result from using all tokens"


# ---------------------------------------------------------------------------
# mean_center / l2_normalize
# ---------------------------------------------------------------------------

class TestMeanCenter:
    def test_column_mean_is_zero(self):
        rng = np.random.default_rng(4)
        a = jnp.array(rng.standard_normal((50, 16)).astype(np.float32))
        out = mean_center(a)
        col_means = np.array(out).mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-5)

    def test_shape_preserved(self):
        a = jnp.ones((30, 12))
        assert mean_center(a).shape == (30, 12)

    def test_constant_column_becomes_zero(self):
        """A column that is constant should be all zeros after centering."""
        a = jnp.array(np.full((10, 4), 5.0, dtype=np.float32))
        out = np.array(mean_center(a))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)


class TestL2Normalize:
    def test_row_norms_are_one(self):
        rng = np.random.default_rng(5)
        a = jnp.array(rng.standard_normal((40, 32)).astype(np.float32))
        out = np.array(l2_normalize(a))
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_shape_preserved(self):
        a = jnp.ones((20, 8))
        assert l2_normalize(a).shape == (20, 8)

    def test_zero_vector_does_not_explode(self):
        """Near-zero rows should be handled gracefully (clamped denominator)."""
        a = jnp.zeros((5, 4))
        out = np.array(l2_normalize(a))
        assert np.all(np.isfinite(out)), "l2_normalize should not produce inf/nan on zero input"


class TestCenteredNormL2:
    """Tests for the composed transform: mean_center → l2_normalize."""

    def _transform(self, a):
        return l2_normalize(mean_center(a))

    def test_row_norms_are_one(self):
        rng = np.random.default_rng(6)
        a = jnp.array(rng.standard_normal((60, 24)).astype(np.float32))
        out = np.array(self._transform(a))
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_output_is_unit_sphere_subset(self):
        """After centering+normalizing, every row norm should be in (0, 1]."""
        rng = np.random.default_rng(7)
        a = jnp.array(rng.standard_normal((50, 16)).astype(np.float32))
        norms = np.linalg.norm(np.array(self._transform(a)), axis=-1)
        assert np.all(norms > 0) and np.all(norms <= 1.0 + 1e-5)

    def test_shape_preserved(self):
        a = jnp.array(np.random.randn(25, 10).astype(np.float32))
        assert self._transform(a).shape == (25, 10)


# ---------------------------------------------------------------------------
# TestNumericalRobustness
# ---------------------------------------------------------------------------

class TestNumericalRobustness:
    def _chunk(self, batch=4, tokens=20, embed=8, n_layers=2, dtype=torch.float32):
        rng = np.random.default_rng(42)
        return {
            f"layer_{i}": torch.tensor(
                rng.standard_normal((batch, tokens, embed)), dtype=dtype
            )
            for i in range(n_layers)
        }

    # --- dtype tests ---

    def test_process_image_chunk_dtype_is_float32(self):
        """process_image_chunk output arrays must be float32."""
        chunk = self._chunk()
        result = process_image_chunk(chunk)
        for key, arr in result.items():
            assert arr.dtype == np.float32, f"{key}: expected float32, got {arr.dtype}"

    def test_winsorized_mean_dtype_is_float32(self):
        """winsorized_mean must return a float32 tensor."""
        x = torch.randn(3, 20, 8, dtype=torch.float32)
        out = winsorized_mean(x, dim=1)
        assert out.dtype == torch.float32

    # --- NaN/Inf input raises ---

    def test_nan_input_raises(self):
        """A chunk containing NaN must raise ValueError."""
        chunk = self._chunk()
        chunk["layer_0"][0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="Non-finite"):
            process_image_chunk(chunk)

    def test_inf_input_is_clipped(self):
        """A single Inf value is winsorized away: it lies above the 95th percentile
        of the flattened per-sample distribution and is clipped to that bound.
        Output must be finite."""
        chunk = self._chunk()
        chunk["layer_0"][0, 5, 3] = float("inf")
        result = process_image_chunk(chunk)
        assert np.isfinite(result["layer_0"]).all(), "Inf should be clipped, output must be finite"

    # --- finite output for normal input ---

    def test_winsorized_output_is_finite(self):
        """For random finite input, winsorized_mean output is always finite."""
        rng = np.random.default_rng(99)
        x = torch.tensor(rng.standard_normal((8, 50, 16)), dtype=torch.float32)
        out = winsorized_mean(x, dim=1)
        assert torch.isfinite(out).all()

    # --- l2_normalize dtype ---

    def test_l2_normalize_output_is_float32(self):
        """l2_normalize must preserve float32 dtype (JAX path)."""
        a = jnp.array(np.random.randn(20, 16).astype(np.float32))
        out = l2_normalize(a)
        assert out.dtype == jnp.float32

    # --- II with constant activations returns NaN ---

    def test_ii_constant_activations_does_not_crash(self):
        """Constant activations → all pairwise distances equal (all 0).
        rankdata(method='min') assigns rank 1 to all tied elements, so sum(cards) > 0
        and the formula returns 0.0 (finite).  Must not raise or return Inf/NaN.
        """
        sample_size = 100
        embed = 8
        key = jax.random.key(0)
        ii_fn = build_information_imbalance(
            key=key, sample_size=sample_size, distance_fn=L2_distance
        )
        X = jnp.ones((sample_size, embed), dtype=jnp.float32)
        Y = jnp.ones((sample_size, embed), dtype=jnp.float32) * 2.0
        inf_imb, rec_imb = ii_fn(X, Y)
        assert jnp.isfinite(inf_imb), f"Expected finite inf_imb, got {inf_imb}"
        assert jnp.isfinite(rec_imb), f"Expected finite rec_imb, got {rec_imb}"
