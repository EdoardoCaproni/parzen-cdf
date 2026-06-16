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


def _scaled_diffs(x: np.ndarray, samples: np.ndarray, h: float) -> np.ndarray:
    """Return the ``(..., n)`` matrix of standardized differences ``(x - x_i) / h``."""
    if h <= 0:
        raise ValueError("bandwidth h must be positive")
    x = np.asarray(x, dtype=float)
    samples = np.asarray(samples, dtype=float)
    return (x[..., None] - samples) / h


def parzen_cdf(x: np.ndarray, samples: np.ndarray, h: float) -> np.ndarray:
    """Logistic-kernel Parzen CDF estimate evaluated at ``x``.

    Implements ``(1/n) * sum_i sigmoid((x - x_i) / h)``.
    """
    z = _scaled_diffs(x, samples, h)
    return expit(z).mean(axis=-1)


def parzen_pdf(x: np.ndarray, samples: np.ndarray, h: float) -> np.ndarray:
    """Logistic-kernel Parzen density estimate at ``x`` (the exact derivative of ``parzen_cdf``).

    Implements ``(1/nh) * sum_i sigmoid(z) * (1 - sigmoid(z))`` with ``z = (x - x_i) / h``.
    """
    z = _scaled_diffs(x, samples, h)
    s = expit(z)
    return (s * (1.0 - s)).mean(axis=-1) / h
