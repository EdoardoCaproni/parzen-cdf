"""Tests for the CDFNet architecture."""

import numpy as np
import pytest
import torch

from parzen_cdf.models import CDFNet


@pytest.mark.parametrize("hidden_sizes", [(4,), (16,), (8, 8)])
def test_output_in_unit_interval_and_shape(hidden_sizes) -> None:
    net = CDFNet(in_dim=1, hidden_sizes=hidden_sizes)
    x = torch.linspace(-5, 5, 50)
    y = net(x)
    assert y.shape == (50,)
    assert torch.all((y > 0) & (y < 1)), "CDF output must lie in (0, 1)"


def test_monotone_net_is_monotone() -> None:
    """A monotone-by-construction net must be non-decreasing for any weights."""
    torch.manual_seed(1)
    net = CDFNet(in_dim=1, hidden_sizes=(16,), monotone=True)
    # Perturb away from initialisation so the test isn't trivially satisfied.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p))
    grid = torch.linspace(-8, 8, 1000)
    y = net(grid).detach().numpy()
    assert np.all(np.diff(y) >= -1e-7), "monotone net produced a decreasing CDF"


def test_monotone_rejects_non_monotone_activation() -> None:
    with pytest.raises(ValueError):
        CDFNet(monotone=True, activation="silu")


def test_unknown_activation_rejected() -> None:
    with pytest.raises(ValueError):
        CDFNet(activation="relu")
