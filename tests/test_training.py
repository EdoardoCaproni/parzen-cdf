"""Tests for the training pipeline and density recovery."""

import numpy as np
import torch

from parzen_cdf import parzen
from parzen_cdf.data import default_mixture
from parzen_cdf.models import CDFNet
from parzen_cdf import metrics
from parzen_cdf.training import (
    TrainConfig,
    density_from_cdf,
    make_training_set,
    rectify_cdf,
    train_cdf,
)


def test_rectify_cdf_is_monotone_and_unit_mass() -> None:
    grid = np.linspace(-5, 5, 1000)
    wobbly = 0.5 * (1 + np.tanh(grid)) + 0.02 * np.sin(5 * grid)  # non-monotone CDF-ish curve
    rc, pdf = rectify_cdf(wobbly, grid)
    assert np.all(np.diff(rc) >= -1e-12), "rectified CDF must be non-decreasing"
    assert rc[0] == 0.0 and abs(rc[-1] - 1.0) < 1e-12, "rectified CDF must span [0, 1]"
    assert np.all(pdf >= 0)
    assert abs(metrics.integrates_to_one(pdf, grid) - 1.0) < 1e-2


def _toy_problem(n_train=256):
    rng = np.random.default_rng(0)
    mix = default_mixture()
    samples = mix.sample(1000, rng)
    h = parzen.silverman_bandwidth(samples)
    inputs, targets = make_training_set(samples, h, n_train, rng)
    return samples, h, inputs, targets


def test_make_training_set_shapes_and_domain() -> None:
    samples, h, inputs, targets = _toy_problem(n_train=300)
    assert inputs.shape == (300,)
    assert targets.shape == (300,)
    assert torch.all((targets >= 0) & (targets <= 1))
    lo, hi = samples.min() - 3 * h, samples.max() + 3 * h
    assert inputs.min() >= lo - 1e-6 and inputs.max() <= hi + 1e-6


def test_density_matches_finite_difference() -> None:
    """density_from_cdf should equal the finite-difference derivative of the network's CDF."""
    torch.manual_seed(0)
    net = CDFNet(in_dim=1, hidden_sizes=(16,))
    grid = torch.linspace(-4, 4, 400)
    pdf = density_from_cdf(net, grid, clamp=False).detach().numpy()
    cdf = net(grid).detach().numpy()
    fd = np.gradient(cdf, grid.numpy())
    assert np.max(np.abs(pdf - fd)) < 1e-2


def test_training_reduces_loss_and_fits_targets() -> None:
    _, _, inputs, targets = _toy_problem()
    net = CDFNet(in_dim=1, hidden_sizes=(16,))
    cfg = TrainConfig(epochs=400, seed=0)
    net, history = train_cdf(net, inputs, targets, cfg)
    assert history[-1] < history[0], "loss did not decrease"
    final = net(inputs).detach()
    # Loose tolerance: this is a deliberately short (400-epoch) run just to confirm it learns.
    assert torch.mean((final - targets) ** 2).item() < 5e-3, "did not fit the Parzen targets"
