"""Sanity checks for the Parzen-window CDF/pdf estimator.

These encode properties the math guarantees: the CDF estimate is monotone and in [0, 1], and the
pdf estimate (its exact derivative) integrates to ~1.
"""

import numpy as np
import pytest

from parzen_cdf import metrics
from parzen_cdf import parzen as parzen_module
from parzen_cdf.parzen import KERNELS, parzen_cdf, parzen_pdf


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


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_every_kernel_is_a_valid_window(samples: np.ndarray, kernel: str) -> None:
    grid = np.linspace(-8, 8, 4000)
    cdf = parzen_cdf(grid, samples, h=0.5, kernel=kernel)
    pdf = parzen_pdf(grid, samples, h=0.5, kernel=kernel)
    assert np.all((cdf >= 0) & (cdf <= 1)) and np.all(np.diff(cdf) >= -1e-9)
    assert cdf[0] < 1e-3 and cdf[-1] > 1 - 1e-3
    assert abs(metrics.integrates_to_one(pdf, grid) - 1.0) < 1e-2
    # pdf must be the derivative of the CDF (matches a finite-difference check)
    fd = np.gradient(cdf, grid)
    assert np.max(np.abs(fd - pdf)) < 0.02


def test_chunked_eval_matches_direct(samples: np.ndarray, monkeypatch) -> None:
    grid = np.linspace(-5, 5, 777)
    expected_cdf = parzen_cdf(grid, samples, h=0.5)
    expected_pdf = parzen_pdf(grid, samples, h=0.5)
    monkeypatch.setattr(parzen_module, "_CHUNK_ELEMENTS", 1000)  # force many small chunks
    assert np.allclose(parzen_cdf(grid, samples, h=0.5), expected_cdf)
    assert np.allclose(parzen_pdf(grid, samples, h=0.5), expected_pdf)
    # per-sample bandwidths (the adaptive path) must survive chunking too
    h_arr = np.full(samples.size, 0.5)
    assert np.allclose(parzen_pdf(grid, samples, h_arr), expected_pdf)
