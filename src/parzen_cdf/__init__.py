"""Parzen CDF: neural estimation of CDFs and densities from Parzen-window targets.

The package is organised into small, single-responsibility modules:

- ``data``     -- synthetic distributions with known pdf/CDF and samplers.
- ``parzen``   -- logistic-kernel Parzen-window CDF/pdf estimator and bandwidth selection.
- ``models``   -- the MLP that regresses the CDF.
- ``training`` -- training loop, monotonicity penalty, and pdf via autograd.
- ``metrics``  -- comparison against ground truth (KS distance, MSE) and plotting helpers.
"""

__version__ = "0.1.0"
