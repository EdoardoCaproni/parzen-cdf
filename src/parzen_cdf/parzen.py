"""Parzen-window estimation with logistic kernels.

The logistic kernel is chosen because its integral is the logistic sigmoid, giving a *closed-form*
CDF estimate. With samples ``x_1..x_n``, bandwidth ``h`` and ``z_i = (x - x_i) / h``::

    F_hat(x) = (1/n)  * sum_i sigmoid(z_i)
    f_hat(x) = (1/nh) * sum_i sigmoid(z_i) * (1 - sigmoid(z_i))   # = dF_hat/dx

``F_hat`` is smooth, strictly increasing, and valued in (0, 1) -- an ideal regression target, and
``f_hat`` is exactly its derivative.

Note on the bandwidth: ``silverman_bandwidth`` uses Silverman's rule, whose constant is derived
for the *Gaussian* kernel (variance 1). The standard logistic kernel has variance ``pi**2 / 3``,
so the same ``h`` smooths more here; treat the returned value as a starting point and tune ``h``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

# Standard-logistic spread, used to variance-match the bandwidth to a Gaussian kernel.
LOGISTIC_KERNEL_STD = float(np.sqrt(np.pi**2 / 3))      # ~= 1.8138
VARIANCE_MATCH_SCALE = 1.0 / LOGISTIC_KERNEL_STD        # = sqrt(3)/pi ~= 0.5513


def silverman_bandwidth(samples: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth as a starting point for ``h``.

    ``h = 0.9 * min(std, IQR / 1.349) * n ** (-1/5)``, falling back to the std when the IQR is 0.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n < 2:
        raise ValueError("need at least 2 samples to estimate a bandwidth")
    std = samples.std(ddof=1)
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = q75 - q25
    spread = min(std, iqr / 1.349) if iqr > 0 else std
    if spread <= 0:
        raise ValueError("samples have zero spread; cannot choose a bandwidth")
    return float(0.9 * spread * n ** (-1 / 5))


def _scaled_diffs(x: np.ndarray, samples: np.ndarray, h) -> np.ndarray:
    """Return the ``(..., n)`` matrix of standardized differences ``(x - x_i) / h``.

    ``h`` may be a scalar (fixed bandwidth) or a length-``n`` array (a per-sample bandwidth, as
    used by the adaptive estimator), in which case it broadcasts over the sample axis.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h <= 0):
        raise ValueError("bandwidth h must be positive")
    x = np.asarray(x, dtype=float)
    samples = np.asarray(samples, dtype=float)
    return (x[..., None] - samples) / h


def parzen_cdf(x: np.ndarray, samples: np.ndarray, h) -> np.ndarray:
    """Logistic-kernel Parzen CDF estimate evaluated at ``x``.

    Implements ``(1/n) * sum_i sigmoid((x - x_i) / h_i)``. ``h`` is a scalar or a per-sample array.
    """
    z = _scaled_diffs(x, samples, h)
    return expit(z).mean(axis=-1)


def parzen_pdf(x: np.ndarray, samples: np.ndarray, h) -> np.ndarray:
    """Logistic-kernel Parzen density estimate at ``x`` (the exact derivative of ``parzen_cdf``).

    Implements ``(1/n) * sum_i sigmoid(z_i) * (1 - sigmoid(z_i)) / h_i`` with ``z_i = (x - x_i)/h_i``.
    Dividing by ``h`` *inside* the sum keeps it correct when ``h`` is a per-sample array.
    """
    z = _scaled_diffs(x, samples, h)
    s = expit(z)
    return (s * (1.0 - s) / np.asarray(h, dtype=float)).mean(axis=-1)


# --------------------------------------------------------------------------------------------------
# Bandwidth selection. All selectors below use ONLY the samples (never the true pdf). They are the
# deployable methods; an oracle (truth-minimizing) bandwidth lives in the comparison script, not
# here, precisely because it is not a real selector.
# --------------------------------------------------------------------------------------------------


def variance_matched_bandwidth(samples: np.ndarray) -> float:
    """Silverman's bandwidth rescaled so the logistic kernel matches a unit-variance (Gaussian)
    kernel: ``h_silverman * sqrt(3)/pi``. A principled, truth-free correction for over-smoothing."""
    return silverman_bandwidth(samples) * VARIANCE_MATCH_SCALE


def candidate_bandwidths(samples: np.ndarray, n_grid: int = 40,
                         lo_scale: float = 0.1, hi_scale: float = 2.0) -> np.ndarray:
    """A geometric grid of candidate bandwidths spanning ``[lo, hi] * Silverman h`` for CV searches."""
    h0 = silverman_bandwidth(samples)
    return h0 * np.geomspace(lo_scale, hi_scale, n_grid)


def _logistic_density(u: np.ndarray) -> np.ndarray:
    """Standard logistic density ``k(u) = sigmoid(u) * (1 - sigmoid(u))``."""
    s = expit(u)
    return s * (1.0 - s)


def _loo_density_at_samples(samples: np.ndarray, h: float) -> np.ndarray:
    """Leave-one-out density ``f_hat_{-i}(x_i)`` at each sample (excludes its own kernel)."""
    n = samples.size
    diffs = samples[:, None] - samples[None, :]      # (n, n)
    k = _logistic_density(diffs / h)
    np.fill_diagonal(k, 0.0)
    return k.sum(axis=1) / ((n - 1) * h)


def likelihood_cv_bandwidth(samples: np.ndarray, candidates: np.ndarray | None = None) -> float:
    """Leave-one-out maximum-likelihood CV: pick ``h`` maximizing ``mean_i log f_hat_{-i}(x_i)``.

    Truth-free; targets a Kullback-Leibler objective. Only evaluates the kernel (no self-convolution).
    """
    samples = np.asarray(samples, dtype=float)
    if candidates is None:
        candidates = candidate_bandwidths(samples)
    best_h, best_score = float(candidates[0]), -np.inf
    for h in candidates:
        f_loo = _loo_density_at_samples(samples, h)
        score = np.log(np.maximum(f_loo, 1e-300)).mean()
        if score > best_score:
            best_score, best_h = score, float(h)
    return best_h


def lscv_bandwidth(samples: np.ndarray, candidates: np.ndarray | None = None,
                   grid: np.ndarray | None = None) -> float:
    """Least-squares (unbiased) CV: minimize ``int f_hat^2 - (2/n) sum_i f_hat_{-i}(x_i)``.

    This is an unbiased estimate of the integrated squared error up to an ``h``-independent constant,
    computable from the samples alone. ``int f_hat^2`` is evaluated numerically on ``grid``.
    """
    samples = np.asarray(samples, dtype=float)
    if candidates is None:
        candidates = candidate_bandwidths(samples)
    if grid is None:
        h0 = silverman_bandwidth(samples)
        grid = np.linspace(samples.min() - 5 * h0, samples.max() + 5 * h0, 4000)
    best_h, best_score = float(candidates[0]), np.inf
    for h in candidates:
        f = parzen_pdf(grid, samples, h)
        term1 = np.trapezoid(f**2, grid)
        term2 = 2.0 * _loo_density_at_samples(samples, h).mean()
        score = term1 - term2
        if score < best_score:
            best_score, best_h = score, float(h)
    return best_h


def adaptive_bandwidths(samples: np.ndarray, pilot_h: float | None = None) -> np.ndarray:
    """Abramson variable bandwidth: ``h_i = pilot_h * (f_pilot(x_i) / g)^(-1/2)`` with ``g`` the
    geometric mean of the pilot density at the samples. Smaller where data is dense, wider in the
    tails -- the structural fix for densities with disparate scales. Returns a length-``n`` array.
    """
    samples = np.asarray(samples, dtype=float)
    if pilot_h is None:
        pilot_h = variance_matched_bandwidth(samples)
    f_pilot = parzen_pdf(samples, samples, pilot_h)
    log_g = np.log(np.maximum(f_pilot, 1e-300)).mean()
    g = np.exp(log_g)
    lam = (np.maximum(f_pilot, 1e-300) / g) ** (-0.5)
    return pilot_h * lam
