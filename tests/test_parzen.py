"""Sanity checks for the Parzen-window CDF/pdf estimator.

These encode properties the math guarantees, so they double as a specification while the
implementation in ``parzen_cdf.parzen`` is being written. They are skipped until the estimator
is implemented (the stubs raise ``NotImplementedError``).
"""

import numpy as np
import pytest

from parzen_cdf.parzen import parzen_cdf, parzen_pdf


@pytest.fixture
def samples() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=200)


def _implemented(fn, *args) -> bool:
    try:
        fn(*args)
        return True
    except NotImplementedError:
        return False


def test_cdf_monotone_and_bounded(samples: np.ndarray) -> None:
    grid = np.linspace(-5, 5, 500)
    if not _implemented(parzen_cdf, grid, samples, 0.5):
        pytest.skip("parzen_cdf not implemented yet")
    f = parzen_cdf(grid, samples, h=0.5)
    assert np.all((f >= 0) & (f <= 1)), "CDF estimate must lie in [0, 1]"
    assert np.all(np.diff(f) >= -1e-9), "CDF estimate must be non-decreasing"


def test_pdf_integrates_to_one(samples: np.ndarray) -> None:
    grid = np.linspace(-8, 8, 4000)
    if not _implemented(parzen_pdf, grid, samples, 0.5):
        pytest.skip("parzen_pdf not implemented yet")
    f = parzen_pdf(grid, samples, h=0.5)
    area = np.trapezoid(f, grid)
    assert abs(area - 1.0) < 1e-2, f"pdf should integrate to ~1, got {area:.4f}"
