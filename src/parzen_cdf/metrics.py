"""Scoring estimates against ground truth, and plotting helpers.

Because the data is synthetic with a known pdf/CDF, every estimate is comparable to the truth.
Report these consistently for both the Parzen estimate and the neural estimate.
"""

from __future__ import annotations

import numpy as np


def ks_distance(cdf_true: np.ndarray, cdf_est: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance: ``max |F_true - F_est|`` on a shared grid."""
    raise NotImplementedError


def mse(true: np.ndarray, est: np.ndarray) -> float:
    """Mean squared error between two evaluations on a shared grid."""
    raise NotImplementedError


def integrates_to_one(pdf_values: np.ndarray, grid: np.ndarray) -> float:
    """Numerically integrate a pdf over ``grid`` (should be ~1). Useful as a sanity check."""
    raise NotImplementedError
