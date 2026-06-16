"""Reproducible Step 1 run: synthetic data -> Parzen CDF -> MLP -> recovered pdf.

This is the end-to-end entry point for the univariate experiment. Fill in the steps as the
library modules are implemented. Keep all hyperparameters and the RNG seed here so a run is fully
reproducible from the command line.

    python scripts/run_step1.py
"""

from __future__ import annotations


def main() -> None:
    # 1. Build the known mixture and draw samples (parzen_cdf.data).
    # 2. Choose bandwidth h and compute the Parzen CDF targets (parzen_cdf.parzen).
    # 3. Assemble (x, F_hat(x)) training pairs.
    # 4. Train the CDFNet with the monotonicity penalty (parzen_cdf.training).
    # 5. Recover the pdf as the derivative; score CDF/pdf against truth (parzen_cdf.metrics).
    raise NotImplementedError("Step 1 pipeline not implemented yet.")


if __name__ == "__main__":
    main()
