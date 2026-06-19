"""Scoring estimates against ground truth, and plotting helpers.

Because the data is synthetic with a known pdf/CDF, every estimate is comparable to the truth.
Report these consistently for both the Parzen estimate and the neural estimate.
"""

from __future__ import annotations

import numpy as np


def ks_distance(cdf_true: np.ndarray, cdf_est: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance: ``max |F_true - F_est|`` on a shared grid."""
    return float(np.max(np.abs(np.asarray(cdf_true) - np.asarray(cdf_est))))


def mse(true: np.ndarray, est: np.ndarray) -> float:
    """Mean squared error between two evaluations on a shared grid."""
    return float(np.mean((np.asarray(true) - np.asarray(est)) ** 2))


def integrates_to_one(pdf_values: np.ndarray, grid: np.ndarray) -> float:
    """Numerically integrate a pdf over ``grid`` (should be ~1). Useful as a sanity check."""
    return float(np.trapezoid(np.asarray(pdf_values), np.asarray(grid)))


def monotonicity_violation_fraction(cdf_on_grid: np.ndarray) -> float:
    """Fraction of adjacent grid steps where the CDF estimate *decreases* (a monotonicity defect).

    The grid is assumed sorted in increasing order. Returns 0.0 for a perfectly monotone curve.
    """
    diffs = np.diff(np.asarray(cdf_on_grid))
    return float(np.mean(diffs < 0)) if diffs.size else 0.0
