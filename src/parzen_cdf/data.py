"""Synthetic data with known density and CDF.

We deliberately use distributions whose true pdf and CDF are available in closed form, so that
every estimate (Parzen-window or neural) can be scored against ground truth. The target family is
the **mixture of Gaussians**: its pdf is a weighted sum of normal pdfs, its CDF a weighted sum of
normal CDFs, and sampling is exact -- so every quantity we need for evaluation is analytic.

A small ladder of distributions of increasing difficulty is provided (see the factory functions),
matching the brief's request for a *variety* of synthetic data:

- ``single_gaussian``   -- sanity rung; validates the whole pipeline on a known-easy case.
- ``symmetric_bimodal`` -- first non-trivial CDF (a plateau between two modes).
- ``asymmetric_trimodal`` (the default) -- unequal weights, different widths, partial overlap.
- ``spike_in_broad``    -- a sharp peak inside a broad mode; stresses a single global bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class GaussianMixture1D:
    """A 1-D Gaussian mixture with known pdf, CDF, and sampler.

    Parameters
    ----------
    weights : mixture weights (must be non-negative and sum to 1).
    means   : component means.
    stds    : component standard deviations (must be positive).
    """

    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.stds = np.asarray(self.stds, dtype=float)
        if not (self.weights.shape == self.means.shape == self.stds.shape):
            raise ValueError("weights, means, and stds must have the same shape")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {self.weights.sum():.6f}")
        if np.any(self.stds <= 0):
            raise ValueError("stds must be positive")

    @property
    def n_components(self) -> int:
        return self.weights.size

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """True density at ``x`` (weighted sum of normal pdfs)."""
        x = np.asarray(x, dtype=float)
        # Broadcast x against the components along a trailing axis, then collapse it.
        per_component = norm.pdf(x[..., None], loc=self.means, scale=self.stds)
        return per_component @ self.weights

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """True CDF at ``x`` (weighted sum of normal CDFs)."""
        x = np.asarray(x, dtype=float)
        per_component = norm.cdf(x[..., None], loc=self.means, scale=self.stds)
        return per_component @ self.weights

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` i.i.d. samples from the mixture (shape ``(n,)``)."""
        components = rng.choice(self.n_components, size=n, p=self.weights)
        return rng.normal(loc=self.means[components], scale=self.stds[components])


def single_gaussian() -> GaussianMixture1D:
    """Standard normal -- the sanity rung."""
    return GaussianMixture1D(weights=[1.0], means=[0.0], stds=[1.0])


def symmetric_bimodal() -> GaussianMixture1D:
    """Two equal, well-separated modes; the CDF shows a clear plateau between them."""
    return GaussianMixture1D(weights=[0.5, 0.5], means=[-2.0, 2.0], stds=[0.7, 0.7])


def asymmetric_bimodal() -> GaussianMixture1D:
    """Unequal weights and widths with partial overlap; one gentle step past symmetric_bimodal."""
    return GaussianMixture1D(weights=[0.65, 0.35], means=[0.0, 3.0], stds=[1.0, 0.6])


def asymmetric_trimodal() -> GaussianMixture1D:
    """Unequal weights, different widths, and partial overlap -- the main running example."""
    return GaussianMixture1D(
        weights=[0.3, 0.5, 0.2],
        means=[-2.0, 1.0, 4.0],
        stds=[0.5, 1.0, 0.3],
    )


def spike_in_broad() -> GaussianMixture1D:
    """A sharp peak sitting inside a broad mode; stresses a single global bandwidth."""
    return GaussianMixture1D(weights=[0.6, 0.4], means=[0.0, 0.5], stds=[1.5, 0.2])


def default_mixture() -> GaussianMixture1D:
    """The reference multimodal mixture used as the running example in Step 1."""
    return asymmetric_trimodal()
