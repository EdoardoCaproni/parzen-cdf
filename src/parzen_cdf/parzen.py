"""Parzen-window estimation with a registry of window shapes (kernels).

The default window is the **logistic** kernel, chosen because its integral is the logistic
sigmoid, giving a *closed-form* CDF estimate. With samples ``x_1..x_n``, window size ``h`` and
``z_i = (x - x_i) / h``::

    F_hat(x) = (1/n)  * sum_i K(z_i)          # K = the window's CDF (sigmoid for logistic)
    f_hat(x) = (1/nh) * sum_i k(z_i)          # k = the window's pdf  (= dK/du)

``F_hat`` is smooth (for smooth windows), non-decreasing, and valued in [0, 1] -- an ideal
regression target -- and ``f_hat`` is exactly its derivative.

Other window shapes live in :data:`KERNELS` (gaussian, box, epanechnikov, triangular); all
estimator and selector functions accept a ``kernel`` name and default to ``"logistic"``.

Note on the window size: ``silverman_bandwidth`` uses Silverman's rule, whose constant is
derived for the *Gaussian* kernel (variance 1). A kernel with std ``s`` smooths ``s`` times
more at the same ``h``; ``variance_matched_bandwidth`` divides by ``s`` to correct for this
(a no-op for the gaussian window).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from scipy.special import erf, expit

# ``np.trapezoid`` e' il nome NumPy >= 2.0 di ``np.trapz`` (rimosso in NumPy 2.4).
# NB: il guard deve essere PIGRO. ``getattr(np, "trapezoid", np.trapz)`` valuterebbe
# ``np.trapz`` comunque, sollevando AttributeError proprio dove non serve.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# --------------------------------------------------------------------------------------------------
# Window shapes. Each kernel is (CDF K(u), pdf k(u) = K'(u), std of the kernel as a density).
# To add a shape, append an entry here; every estimator and selector picks it up by name.
# --------------------------------------------------------------------------------------------------


class Kernel(NamedTuple):
    cdf: Callable[[np.ndarray], np.ndarray]
    pdf: Callable[[np.ndarray], np.ndarray]
    std: float


def _logistic_pdf(u: np.ndarray) -> np.ndarray:
    s = expit(u)
    return s * (1.0 - s)


def _gaussian_cdf(u: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(u / np.sqrt(2.0)))


def _gaussian_pdf(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)


def _box_cdf(u: np.ndarray) -> np.ndarray:
    return np.clip((u + 1.0) / 2.0, 0.0, 1.0)


def _box_pdf(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1.0, 0.5, 0.0)


def _epanechnikov_cdf(u: np.ndarray) -> np.ndarray:
    t = np.clip(u, -1.0, 1.0)
    return 0.5 + 0.75 * t - 0.25 * t**3


def _epanechnikov_pdf(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def _triangular_cdf(u: np.ndarray) -> np.ndarray:
    t = np.clip(u, -1.0, 1.0)
    return np.where(t < 0.0, 0.5 * (t + 1.0) ** 2, 1.0 - 0.5 * (1.0 - t) ** 2)


def _triangular_pdf(u: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 - np.abs(u), 0.0)


KERNELS: dict[str, Kernel] = {
    "logistic": Kernel(expit, _logistic_pdf, float(np.sqrt(np.pi**2 / 3))),
    "gaussian": Kernel(_gaussian_cdf, _gaussian_pdf, 1.0),
    "box": Kernel(_box_cdf, _box_pdf, float(1 / np.sqrt(3))),
    "epanechnikov": Kernel(_epanechnikov_cdf, _epanechnikov_pdf, float(1 / np.sqrt(5))),
    "triangular": Kernel(_triangular_cdf, _triangular_pdf, float(1 / np.sqrt(6))),
}

# Standard-logistic spread, used to variance-match the window size to a Gaussian kernel.
LOGISTIC_KERNEL_STD = KERNELS["logistic"].std                   # ~= 1.8138
VARIANCE_MATCH_SCALE = 1.0 / LOGISTIC_KERNEL_STD                # = sqrt(3)/pi ~= 0.5513


def _kernel(name: str) -> Kernel:
    try:
        return KERNELS[name]
    except KeyError:
        raise ValueError(f"unknown kernel {name!r}; choose from {list(KERNELS)}") from None


def sqrt_n_bandwidth(samples: np.ndarray, h1: float = 1.0) -> float:
    """The classic fixed (deterministic) consistency schedule: ``h_n = h1 / sqrt(n)``.

    ``h1`` is chosen once; the window then shrinks with the sample count. This is the
    course's reference rule (Duda & Hart); see the study for the calibration of ``h1``.
    """
    n = np.asarray(samples).size
    if n < 1:
        raise ValueError("need at least 1 sample")
    if h1 <= 0:
        raise ValueError("h1 must be positive")
    return float(h1 / np.sqrt(n))


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


# Cap the (len(x), n) temporaries at ~160 MB of float64: evaluating at the samples themselves
# (adaptive pilot, training labels) would otherwise allocate an n x n matrix (3.2 GB at n=20k).
_CHUNK_ELEMENTS = 20_000_000


def _chunked_eval(fn, x: np.ndarray, n: int) -> np.ndarray:
    """Apply ``fn`` (a 1-D-batch evaluator) over row-chunks of ``x``, bounding peak memory."""
    x = np.asarray(x, dtype=float)
    flat = x.reshape(-1)
    rows = max(1, _CHUNK_ELEMENTS // max(n, 1))
    if flat.size <= rows:
        return fn(flat).reshape(x.shape)
    out = np.empty(flat.size)
    for i in range(0, flat.size, rows):
        out[i:i + rows] = fn(flat[i:i + rows])
    return out.reshape(x.shape)


def parzen_cdf(x: np.ndarray, samples: np.ndarray, h, kernel: str = "logistic") -> np.ndarray:
    """Parzen CDF estimate at ``x``: ``(1/n) * sum_i K((x - x_i) / h_i)``.

    ``h`` is a scalar or a per-sample array; ``kernel`` names a window shape in :data:`KERNELS`.
    """
    k = _kernel(kernel)
    samples = np.asarray(samples, dtype=float)
    return _chunked_eval(lambda xc: k.cdf(_scaled_diffs(xc, samples, h)).mean(axis=-1),
                         x, samples.size)


def parzen_pdf(x: np.ndarray, samples: np.ndarray, h, kernel: str = "logistic") -> np.ndarray:
    """Parzen density estimate at ``x`` (the exact derivative of ``parzen_cdf``).

    Implements ``(1/n) * sum_i k(z_i) / h_i`` with ``z_i = (x - x_i)/h_i``. Dividing by ``h``
    *inside* the sum keeps it correct when ``h`` is a per-sample array.
    """
    k = _kernel(kernel)
    samples = np.asarray(samples, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    return _chunked_eval(lambda xc: (k.pdf(_scaled_diffs(xc, samples, h)) / h_arr).mean(axis=-1),
                         x, samples.size)


# --------------------------------------------------------------------------------------------------
# Bandwidth selection. All selectors below use ONLY the samples (never the true pdf). They are the
# deployable methods; an oracle (truth-minimizing) bandwidth lives in the comparison script, not
# here, precisely because it is not a real selector.
# --------------------------------------------------------------------------------------------------


def variance_matched_bandwidth(samples: np.ndarray, kernel: str = "logistic") -> float:
    """Silverman's bandwidth rescaled so the chosen kernel matches a unit-variance (Gaussian)
    kernel: ``h_silverman / kernel_std``. A principled, truth-free correction for over-smoothing
    (identity for the gaussian window)."""
    return silverman_bandwidth(samples) / _kernel(kernel).std


def candidate_bandwidths(samples: np.ndarray, n_grid: int = 40,
                         lo_scale: float = 0.1, hi_scale: float = 2.0) -> np.ndarray:
    """A geometric grid of candidate bandwidths spanning ``[lo, hi] * Silverman h`` for CV searches."""
    h0 = silverman_bandwidth(samples)
    return h0 * np.geomspace(lo_scale, hi_scale, n_grid)


def _loo_density_at_samples(samples: np.ndarray, h: float, kernel: str = "logistic") -> np.ndarray:
    """Leave-one-out density ``f_hat_{-i}(x_i)`` at each sample (excludes its own kernel)."""
    n = samples.size
    diffs = samples[:, None] - samples[None, :]      # (n, n)
    k = _kernel(kernel).pdf(diffs / h)
    np.fill_diagonal(k, 0.0)
    return k.sum(axis=1) / ((n - 1) * h)


def likelihood_cv_bandwidth(samples: np.ndarray, candidates: np.ndarray | None = None,
                            kernel: str = "logistic") -> float:
    """Leave-one-out maximum-likelihood CV: pick ``h`` maximizing ``mean_i log f_hat_{-i}(x_i)``.

    Truth-free; targets a Kullback-Leibler objective. Only evaluates the kernel (no self-convolution).
    """
    samples = np.asarray(samples, dtype=float)
    if candidates is None:
        candidates = candidate_bandwidths(samples)
    best_h, best_score = float(candidates[0]), -np.inf
    for h in candidates:
        f_loo = _loo_density_at_samples(samples, h, kernel)
        score = np.log(np.maximum(f_loo, 1e-300)).mean()
        if score > best_score:
            best_score, best_h = score, float(h)
    return best_h


def lscv_bandwidth(samples: np.ndarray, candidates: np.ndarray | None = None,
                   grid: np.ndarray | None = None, kernel: str = "logistic") -> float:
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
        f = parzen_pdf(grid, samples, h, kernel)
        term1 = _trapezoid(f**2, grid)
        term2 = 2.0 * _loo_density_at_samples(samples, h, kernel).mean()
        score = term1 - term2
        if score < best_score:
            best_score, best_h = score, float(h)
    return best_h


def adaptive_bandwidths(samples: np.ndarray, pilot_h: float | None = None,
                        kernel: str = "logistic") -> np.ndarray:
    """Abramson variable bandwidth: ``h_i = pilot_h * (f_pilot(x_i) / g)^(-1/2)`` with ``g`` the
    geometric mean of the pilot density at the samples. Smaller where data is dense, wider in the
    tails -- the structural fix for densities with disparate scales. Returns a length-``n`` array.
    """
    samples = np.asarray(samples, dtype=float)
    if pilot_h is None:
        pilot_h = variance_matched_bandwidth(samples, kernel)
    f_pilot = parzen_pdf(samples, samples, pilot_h, kernel)
    log_g = np.log(np.maximum(f_pilot, 1e-300)).mean()
    g = np.exp(log_g)
    lam = (np.maximum(f_pilot, 1e-300) / g) ** (-0.5)
    return pilot_h * lam
