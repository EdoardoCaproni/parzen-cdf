"""The neural CDF regressor.

A small MLP ``F_theta : R^d -> (0, 1)`` that approximates the Parzen CDF estimate. In 1-D it maps
a scalar to a scalar; in N-D it maps a point to a single scalar (the joint CDF value). A final
sigmoid squashes the output to (0, 1) to respect the range of a CDF.

Two modes share one code path so the comparison is fair:

- ``monotone=False`` -- a plain MLP. Monotonicity is only *encouraged* via a loss penalty (see
  ``training``); this covers both the unconstrained baseline and the soft-penalty variant.
- ``monotone=True``  -- monotone *by construction* (Sill, 1998): every weight matrix is passed
  through ``softplus`` (a smooth non-negativity reparameterization), so with monotone activations
  and a final sigmoid the network is non-decreasing in its input and its derivative -- the pdf --
  is guaranteed non-negative.

Activations must be **smooth**, because the pdf is obtained by differentiating the output; a ReLU
network would yield a piecewise-constant, discontinuous pdf. For the monotone mode the activation
must additionally be monotone (so ``silu`` is allowed only in the non-monotone mode).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Smooth activations only. The flag marks those that are also monotone increasing.
_ACTIVATIONS = {
    "sigmoid": (torch.sigmoid, True),
    "tanh": (torch.tanh, True),
    "softplus": (nn.functional.softplus, True),
    "silu": (nn.functional.silu, False),
}


class CDFNet(nn.Module):
    """MLP mapping an input point to a CDF value in (0, 1).

    Parameters
    ----------
    in_dim       : input dimensionality (1 for Step 1, N for Step 2).
    hidden_sizes : widths of the hidden layers.
    activation   : smooth activation name ("sigmoid", "tanh", "softplus", "silu").
    monotone     : if True, enforce monotonicity by construction (non-negative weights).
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden_sizes: Sequence[int] = (16,),
        activation: str = "sigmoid",
        monotone: bool = False,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(_ACTIVATIONS)}")
        act_fn, is_monotone_act = _ACTIVATIONS[activation]
        if monotone and not is_monotone_act:
            raise ValueError(f"activation {activation!r} is not monotone; cannot use with monotone=True")

        self.in_dim = in_dim
        self.monotone = monotone
        self._activation = act_fn

        dims = [in_dim, *hidden_sizes, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            bias = torch.zeros(out_f)
            if monotone:
                # Initialise so that softplus(raw) starts at a modest, positive Xavier-like scale.
                scale = (6.0 / (in_f + out_f)) ** 0.5
                effective = torch.rand(out_f, in_f) * scale
                weight = torch.log(torch.expm1(effective.clamp_min(1e-6)))  # inverse softplus
            else:
                weight = torch.empty(out_f, in_f)
                nn.init.xavier_uniform_(weight)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))

    def _weight(self, raw: torch.Tensor) -> torch.Tensor:
        """Effective weight matrix: non-negative (via softplus) in monotone mode, else raw."""
        return nn.functional.softplus(raw) if self.monotone else raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the estimated CDF value(s) for ``x`` (shape ``[batch]`` or ``[batch, in_dim]``)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = x
        last = len(self.weights) - 1
        for i, (raw_w, b) in enumerate(zip(self.weights, self.biases)):
            h = nn.functional.linear(h, self._weight(raw_w), b)
            if i < last:
                h = self._activation(h)
        return torch.sigmoid(h).squeeze(-1)
