"""The neural CDF regressor.

A small MLP ``F_theta : R^d -> [0, 1]`` that approximates the Parzen CDF estimate. In 1-D it maps
a scalar to a scalar; in N-D it maps a point to a single scalar (the joint CDF value). The output
is squashed to (0, 1) (e.g. a final sigmoid) to respect the range of a CDF.

Monotonicity is *not* baked into the architecture in the baseline; it is encouraged via a loss
penalty (see ``training``). An architectural-monotonicity variant can be added later as a
speculative direction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CDFNet(nn.Module):
    """MLP mapping an input point to a CDF value in (0, 1).

    Parameters
    ----------
    in_dim       : input dimensionality (1 for Step 1, N for Step 2).
    hidden_sizes : widths of the hidden layers.
    """

    def __init__(self, in_dim: int = 1, hidden_sizes: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the estimated CDF value(s) for ``x`` (shape ``[batch, in_dim]``)."""
        raise NotImplementedError
