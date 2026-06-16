"""Synthetic data with known density and CDF.

We deliberately use distributions whose true pdf and CDF are available in closed form, so that
every estimate (Parzen-window or neural) can be scored against ground truth. The default target
is a non-trivial **mixture of Gaussians**: multimodal enough to be interesting, yet with an
analytic pdf and CDF (a weighted sum of normal pdfs / CDFs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianMixture1D:
    """A 1-D Gaussian mixture with known pdf, CDF, and sampler.

    Parameters
    ----------
    weights : array of mixture weights (must sum to 1).
    means   : array of component means.
    stds    : array of component standard deviations.
    """

    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """True density at ``x`` (weighted sum of normal pdfs)."""
        raise NotImplementedError

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """True CDF at ``x`` (weighted sum of normal CDFs)."""
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` i.i.d. samples from the mixture."""
        raise NotImplementedError


def default_mixture() -> GaussianMixture1D:
    """A reference multimodal mixture used as the running example in Step 1."""
    raise NotImplementedError
