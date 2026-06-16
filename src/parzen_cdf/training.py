"""Training the CDF regressor and recovering the density.

Loss = data term (regress the Parzen targets) + monotonicity penalty. The monotonicity penalty
discourages a negative input gradient of the network output, evaluated at points sampled across
the domain. The density is then the input gradient of the trained network.

Both the penalty and the density rely on ``torch.autograd.grad`` with ``create_graph=True`` so the
gradient is itself differentiable during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import CDFNet


@dataclass
class TrainConfig:
    """Hyperparameters for a training run (record these alongside any reported result)."""

    epochs: int = 2000
    lr: float = 1e-3
    monotonicity_weight: float = 1.0
    seed: int = 0


def monotonicity_penalty(model: CDFNet, x: torch.Tensor) -> torch.Tensor:
    """Penalty for a negative input gradient ``dF/dx`` at the points ``x``.

    For 1-D, penalise ``relu(-dF/dx)``. For N-D, see the *N-increasing* note in ``CLAUDE.md``:
    per-coordinate monotonicity is necessary but not sufficient for a valid joint CDF.
    """
    raise NotImplementedError


def density_from_cdf(model: CDFNet, x: torch.Tensor) -> torch.Tensor:
    """Recover the pdf as the (mixed) derivative of the CDF network at ``x``.

    1-D: ``dF/dx``. N-D: the mixed partial ``d^N F / dx_1...dx_N``. Negative values may be clamped
    to zero as a worst-case fallback.
    """
    raise NotImplementedError


def train_cdf(
    model: CDFNet,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: TrainConfig,
) -> CDFNet:
    """Fit ``model`` to ``(inputs, targets)`` Parzen pairs under the monotonicity penalty."""
    raise NotImplementedError
