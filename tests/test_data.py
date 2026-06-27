"""Sanity checks for the synthetic Gaussian-mixture distributions.

These verify the analytic ground truth itself: a valid CDF (monotone, in [0,1], limits 0->1), a
pdf that integrates to 1, and a sampler whose empirical mean matches the analytic mean.
"""

import numpy as np
import pytest

from parzen_cdf import data, metrics

ALL_MIXTURES = [
    data.single_gaussian(),
    data.symmetric_bimodal(),
    data.asymmetric_trimodal(),
    data.spike_in_broad(),
    data.default_mixture(),
]


@pytest.mark.parametrize("mix", ALL_MIXTURES)
def test_cdf_is_valid(mix: data.GaussianMixture1D) -> None:
    grid = np.linspace(-15, 15, 4000)
    f = mix.cdf(grid)
    assert np.all((f >= 0) & (f <= 1)), "CDF must lie in [0, 1]"
    assert np.all(np.diff(f) >= -1e-12), "CDF must be non-decreasing"
    assert f[0] < 1e-3 and f[-1] > 1 - 1e-3, "CDF must approach 0 and 1 at the tails"


@pytest.mark.parametrize("mix", ALL_MIXTURES)
def test_pdf_integrates_to_one(mix: data.GaussianMixture1D) -> None:
    grid = np.linspace(-20, 20, 8000)
    area = metrics.integrates_to_one(mix.pdf(grid), grid)
    assert abs(area - 1.0) < 1e-4, f"pdf should integrate to ~1, got {area:.6f}"


@pytest.mark.parametrize("mix", ALL_MIXTURES)
def test_sample_mean_matches_analytic_mean(mix: data.GaussianMixture1D) -> None:
    rng = np.random.default_rng(0)
    samples = mix.sample(200_000, rng)
    analytic_mean = float(mix.weights @ mix.means)
    assert abs(samples.mean() - analytic_mean) < 0.05


def test_invalid_weights_rejected() -> None:
    with pytest.raises(ValueError):
        data.GaussianMixture1D(weights=[0.5, 0.4], means=[0.0, 1.0], stds=[1.0, 1.0])


def test_random_mixture_is_valid() -> None:
    rng = np.random.default_rng(0)
    m = None
    for _ in range(20):
        m = data.random_mixture(rng)
        assert 3 <= m.n_components <= 6  # __post_init__ already enforces valid weights/stds
    grid = np.linspace(-30, 30, 12000)
    assert abs(metrics.integrates_to_one(m.pdf(grid), grid) - 1.0) < 1e-2
