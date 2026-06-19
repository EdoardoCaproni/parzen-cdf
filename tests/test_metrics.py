"""Tests for the scoring helpers."""

import numpy as np

from parzen_cdf import metrics


def test_ks_distance() -> None:
    a = np.array([0.0, 0.5, 1.0])
    b = np.array([0.0, 0.3, 1.0])
    assert metrics.ks_distance(a, b) == 0.2


def test_mse() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    assert metrics.mse(a, b) == 4.0 / 3.0


def test_integrates_to_one_on_gaussian() -> None:
    grid = np.linspace(-10, 10, 5000)
    pdf = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
    assert abs(metrics.integrates_to_one(pdf, grid) - 1.0) < 1e-4


def test_monotonicity_violation_fraction() -> None:
    assert metrics.monotonicity_violation_fraction(np.array([0.0, 0.1, 0.2, 1.0])) == 0.0
    # one decrease out of three steps
    assert metrics.monotonicity_violation_fraction(np.array([0.0, 0.5, 0.3, 1.0])) == 1.0 / 3.0
