"""Numerical utilities shared by training and inference."""
from __future__ import annotations

import numpy as np


def _normalize_sampling_probabilities(
    probabilities: np.ndarray, epsilon: float
) -> np.ndarray:
    """Return float64 probability rows safe for NumPy categorical sampling.

    TensorFlow softmax commonly produces float32 rows such as three copies of
    ``0.33333334``. Casting that row directly to float64 preserves a sum of
    about ``1.00000003``, which exceeds ``Generator.choice``'s float64
    tolerance even though the distribution is valid at float32 precision.
    Renormalizing after conversion to float64 removes that dtype-boundary
    failure.
    """
    if not np.isfinite(epsilon) or not 0.0 < float(epsilon) < 1.0:
        raise ValueError("epsilon must be finite and in (0, 1)")

    normalized = np.asarray(probabilities, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError(
            f"probability array must be rank 2, got shape {normalized.shape}"
        )
    if normalized.shape[1] < 1:
        raise ValueError("probability rows must contain at least one action")
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError(
            "actor produced non-finite action probabilities"
        )

    normalized = np.clip(normalized, float(epsilon), 1.0)
    row_sums = np.sum(
        normalized, axis=-1, keepdims=True, dtype=np.float64
    )
    if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise FloatingPointError(
            "actor produced an invalid action-probability row"
        )
    normalized = normalized / row_sums

    # Absorb the final round-off residual into the largest component. This
    # keeps every entry non-negative and makes the exact float64 rows passed to
    # NumPy sum to one within machine precision.
    residuals = 1.0 - np.sum(
        normalized, axis=-1, dtype=np.float64
    )
    pivots = np.argmax(normalized, axis=-1)
    normalized[np.arange(normalized.shape[0]), pivots] += residuals

    final_sums = np.sum(normalized, axis=-1, dtype=np.float64)
    tolerance = 8.0 * np.finfo(np.float64).eps
    if np.any(normalized < 0.0) or not np.all(
        np.abs(final_sums - 1.0) <= tolerance
    ):
        raise FloatingPointError(
            "could not normalize action probabilities for sampling"
        )
    return normalized
