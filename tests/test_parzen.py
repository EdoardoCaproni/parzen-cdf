"""Sanity checks for the Parzen-window CDF/pdf estimator.

These encode properties the math guarantees: the CDF estimate is monotone and in [0, 1], and the
pdf estimate (its exact derivative) integrates to ~1.
"""

import numpy as np
import pytest

from parzen_cdf import metrics
from parzen_cdf.parzen import parzen_cdf, parzen_pdf


@pytest.fixture
def samples() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=200)


def test_cdf_monotone_and_bounded(samples: np.ndarray) -> None:
    grid = np.linspace(-5, 5, 500)
    f = parzen_cdf(grid, samples, h=0.5)
    assert np.all((f >= 0) & (f <= 1)), "CDF estimate must lie in [0, 1]"
    assert np.all(np.diff(f) >= -1e-9), "CDF estimate must be non-decreasing"


def test_pdf_integrates_to_one(samples: np.ndarray) -> None:
    grid = np.linspace(-8, 8, 4000)
    f = parzen_pdf(grid, samples, h=0.5)
    area = metrics.integrates_to_one(f, grid)
    assert abs(area - 1.0) < 1e-2, f"pdf should integrate to ~1, got {area:.4f}"
