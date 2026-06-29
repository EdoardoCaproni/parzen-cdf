"""Training the CDF regressor and recovering the density.

Loss = data term (regress the Parzen CDF targets) + an optional monotonicity penalty. The penalty
discourages a negative input gradient of the network output, evaluated at collocation points
spanning the domain. The density is then the input gradient of the trained network.

Both the penalty and the density rely on ``torch.autograd.grad``; the penalty uses
``create_graph=True`` so it is itself differentiable during training. Everything here is written
for the 1-D case (Step 1); the N-D density is a mixed partial derivative (Step 2).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from . import parzen
from .models import CDFNet


@dataclass
class TrainConfig:
    """Hyperparameters for a training run (record these alongside any reported result)."""

    epochs: int = 2000
    lr: float = 1e-2
    monotonicity_weight: float = 0.0  # 0 = unconstrained baseline; > 0 = soft penalty
    weight_decay: float = 0.0  # L2 regularization (wrong smoothness prior for a CDF net; see follow-up)
    curvature_weight: float = 0.0  # > 0 penalizes |d2F/dx2|: a CDF-appropriate smoothness prior
    optimizer: str = "adam"  # "sgd" (simplest, fixed lr) or "adam"
    n_penalty_points: int = 256
    seed: int = 0


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sample_training_set(samples: np.ndarray, h) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(inputs, targets)`` from the data samples themselves -- the constraint-correct regime.

    Per the project pipeline the MLP is trained ONLY on the drawn sample points ``x_i`` with label
    ``F_hat(x_i)`` (the logistic Parzen CDF). No collocation, no synthetic ``x``, no augmentation:
    the network sees exactly the data and its Parzen-CDF value there. ``h`` is a scalar or a
    per-sample array. ``F_hat(x_i)`` is the full estimate (it includes ``x_i``'s own kernel).
    """
    samples = np.asarray(samples, dtype=float)
    targets = parzen.parzen_cdf(samples, samples, h)
    inputs_t = torch.as_tensor(samples, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    return inputs_t, targets_t


def make_training_set(
    samples: np.ndarray,
    h: float,
    n_points: int,
    rng: np.random.Generator,
    k_pad: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(inputs, targets)`` from uniform collocation points over a bounded domain.

    Inputs are drawn uniformly from ``[min(samples) - k_pad*h, max(samples) + k_pad*h]`` (even
    coverage, including the saturating tails); targets are the logistic Parzen CDF at those points.

    NOTE: this collocation scheme samples the Parzen CDF at synthetic ``x`` and so deviates from the
    project constraint (train on the data points only). Kept as an out-of-constraint reference;
    constraint-correct experiments use :func:`make_sample_training_set`.
    """
    lo = float(samples.min() - k_pad * h)
    hi = float(samples.max() + k_pad * h)
    inputs = rng.uniform(lo, hi, size=n_points)
    targets = parzen.parzen_cdf(inputs, samples, h)
    inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    return inputs_t, targets_t


def monotonicity_penalty(model: CDFNet, x: torch.Tensor) -> torch.Tensor:
    """Penalty for a negative input gradient ``dF/dx`` at the points ``x``: ``mean(relu(-dF/dx))``.

    For N-D, recall the *N-increasing* requirement: per-coordinate monotonicity is necessary but
    not sufficient for a valid joint CDF.
    """
    x = x.detach().clone().requires_grad_(True)
    f = model(x)
    grad = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    return torch.relu(-grad).mean()


def curvature_penalty(model: CDFNet, x: torch.Tensor) -> torch.Tensor:
    """Mean squared second derivative ``(d2F/dx2)^2`` at the points ``x``: a smoothness prior.

    The true CDF is smooth, whereas the finite-sample Parzen labels are wiggly; penalizing curvature
    encourages the network to fit the underlying trend instead of interpolating that wiggle. Unlike
    weight decay (which pulls the output toward a flat ``sigmoid(0)=0.5``), this targets smoothness
    of the *function*, so it does not bias the CDF toward a constant.
    """
    x = x.detach().clone().requires_grad_(True)
    f = model(x)
    g1 = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    g2 = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
    return (g2 ** 2).mean()


def density_from_cdf(model: CDFNet, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Recover the 1-D pdf as ``dF/dx`` via autograd; optionally clamp negatives to 0."""
    x = x.detach().clone().requires_grad_(True)
    f = model(x)
    grad = torch.autograd.grad(f.sum(), x, create_graph=False)[0]
    return grad.clamp_min(0.0) if clamp else grad


def rectify_cdf(cdf_on_grid: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downstream monotonicity fix on an evaluated CDF curve: take the cumulative max (so it is
    non-decreasing) and rescale to [0, 1]; return the rectified CDF and its density (the clamped
    gradient, which then integrates to 1). The chosen, cheapest monotonicity strategy (see the study).
    """
    rc = np.maximum.accumulate(np.asarray(cdf_on_grid, dtype=float))
    lo, hi = float(rc[0]), float(rc[-1])
    if hi > lo:
        rc = (rc - lo) / (hi - lo)
    pdf = np.clip(np.gradient(rc, np.asarray(grid, dtype=float)), 0.0, None)
    return rc, pdf


def train_cdf(
    model: CDFNet,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: TrainConfig,
    penalty_points: torch.Tensor | None = None,
) -> tuple[CDFNet, list[float]]:
    """Fit ``model`` to the ``(inputs, targets)`` Parzen pairs, returning the model and loss history.

    When ``config.monotonicity_weight > 0`` a soft penalty on negative ``dF/dx`` is added, evaluated
    at ``penalty_points`` (defaults to uniform points over the input range).
    """
    set_seed(config.seed)
    if (config.monotonicity_weight > 0 or config.curvature_weight > 0) and penalty_points is None:
        lo, hi = inputs.min().item(), inputs.max().item()
        penalty_points = torch.linspace(lo, hi, config.n_penalty_points)

    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = torch.nn.MSELoss()
    history: list[float] = []

    for _ in range(config.epochs):
        optimizer.zero_grad()
        loss = mse(model(inputs), targets)
        if config.monotonicity_weight > 0:
            loss = loss + config.monotonicity_weight * monotonicity_penalty(model, penalty_points)
        if config.curvature_weight > 0:
            loss = loss + config.curvature_weight * curvature_penalty(model, penalty_points)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return model, history
