from __future__ import annotations

import numpy as np
import pytest

from SLAM.numerics import _normalize_sampling_probabilities


def test_float32_softmax_rows_are_safe_for_numpy_sampling() -> None:
    # A valid float32 softmax row can violate Generator.choice's stricter
    # float64 sum tolerance after a plain cast.
    raw = np.full((2, 3), np.float32(1.0 / 3.0), dtype=np.float32)
    cast_error = np.abs(
        np.sum(raw.astype(np.float64), axis=-1) - 1.0
    )
    assert np.all(cast_error > np.sqrt(np.finfo(np.float64).eps))

    probabilities = _normalize_sampling_probabilities(raw, epsilon=1e-7)
    assert probabilities.dtype == np.float64
    np.testing.assert_allclose(
        np.sum(probabilities, axis=-1, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    )

    rng = np.random.default_rng(7)
    for row in probabilities:
        for _ in range(100):
            sampled = rng.choice(3, p=row)
            assert 0 <= sampled < 3


def test_sampling_normalization_rejects_non_finite_rows() -> None:
    with pytest.raises(FloatingPointError):
        _normalize_sampling_probabilities(
            np.asarray([[0.5, np.nan, 0.5]]), epsilon=1e-7
        )
