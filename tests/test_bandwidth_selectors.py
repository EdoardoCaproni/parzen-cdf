"""Tests for the bandwidth-selection methods (all truth-free)."""

import numpy as np
import pytest

from parzen_cdf import metrics, parzen


@pytest.fixture
def samples() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=300)


def test_variance_matched_is_silverman_scaled(samples) -> None:
    h_silv = parzen.silverman_bandwidth(samples)
    h_vm = parzen.variance_matched_bandwidth(samples)
    assert h_vm == pytest.approx(h_silv * np.sqrt(3) / np.pi)
    assert h_vm < h_silv  # the correction sharpens (logistic kernel is wider than Gaussian)


@pytest.mark.parametrize("selector", [parzen.likelihood_cv_bandwidth, parzen.lscv_bandwidth])
def test_cv_selectors_return_h_within_candidate_range(samples, selector) -> None:
    cands = parzen.candidate_bandwidths(samples)
    h = selector(samples)
    assert isinstance(h, float)
    assert cands.min() <= h <= cands.max()


def test_likelihood_cv_recovers_sane_bandwidth_for_gaussian() -> None:
    # For a standard normal, a good bandwidth is a modest fraction of the std (~O(0.1-0.5)).
    rng = np.random.default_rng(1)
    s = rng.normal(size=500)
    h = parzen.likelihood_cv_bandwidth(s)
    assert 0.05 < h < 1.0


def test_adaptive_bandwidths_are_positive_and_per_sample(samples) -> None:
    h = parzen.adaptive_bandwidths(samples)
    assert h.shape == samples.shape
    assert np.all(h > 0)


def test_parzen_pdf_accepts_per_sample_bandwidth(samples) -> None:
    """A per-sample bandwidth array must still yield a valid density (integrates to ~1)."""
    h = parzen.adaptive_bandwidths(samples)
    grid = np.linspace(-8, 8, 4000)
    pdf = parzen.parzen_pdf(grid, samples, h)
    assert np.all(pdf >= 0)
    assert abs(metrics.integrates_to_one(pdf, grid) - 1.0) < 1e-2
