"""Phase A / single Gaussian -- estimate the CDF with a Parzen Window, built up one step at a time.

Terminology: the estimator is a *Parzen Window*; its smoothing parameter is the *window size*. The
CDF estimate is the integral of the estimated pdf -- in closed form, a mean of logistic sigmoids.
Each step is motivated by the previous step's shortcoming; we judge by the CDF gap (KS) versus the
known truth N(0,1). Steps are added here as we go.

    python scripts/01_parzen_gaussian.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, metrics, parzen

SEED = 0
N = 2000
RESULTS_DIR = "results"

mix = data.single_gaussian()                       # known CDF: a single Gaussian N(0,1)
grid = np.linspace(-6, 6, 2000)
true_pdf, true_cdf = mix.pdf(grid), mix.cdf(grid)
samples = mix.sample(N, np.random.default_rng(SEED))


def estimate(window_size):
    """Parzen Window estimate; window_size is a scalar (fixed) or a per-point array (adaptive)."""
    est_pdf = parzen.parzen_pdf(grid, samples, window_size)
    est_cdf = parzen.parzen_cdf(grid, samples, window_size)   # == integral of est_pdf (closed form)
    return est_pdf, est_cdf, metrics.ks_distance(true_cdf, est_cdf)


# Step 1 -- simplest possible: a fixed, trivial window size.
pdf1, cdf1, ks1 = estimate(1.0)

# Step 2 -- adaptive (per-point) window size, using the Step-1 fixed window as the pilot.
win_adapt = parzen.adaptive_bandwidths(samples, pilot_h=1.0)
pdf2, cdf2, ks2 = estimate(win_adapt)

# Step 3 -- a data-driven global window size (Silverman's rule) instead of the trivial fixed one.
win_silv = parzen.silverman_bandwidth(samples)
pdf3, cdf3, ks3 = estimate(win_silv)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Phase A -- single Gaussian N(0,1), n={N}")
    print(f"  Step 1  fixed window size = 1.0            : CDF gap (KS) = {ks1:.4f}")
    print(f"  Step 2  adaptive window size (pilot = 1.0) : CDF gap (KS) = {ks2:.4f}"
          f"   [per-point window: mean {win_adapt.mean():.3f}, range {win_adapt.min():.3f}-{win_adapt.max():.3f}]")
    print(f"  Step 3  Silverman window size = {win_silv:.3f}     : CDF gap (KS) = {ks3:.4f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(grid, true_pdf, "k-", lw=2, label="true")
    a.plot(grid, pdf1, "--", color="tab:red", label=f"step 1: fixed 1.0 (KS {ks1:.3f})")
    a.plot(grid, pdf2, ":", color="tab:orange", label=f"step 2: adaptive (KS {ks2:.3f})")
    a.plot(grid, pdf3, "-", color="tab:green", label=f"step 3: Silverman (KS {ks3:.3f})")
    a.set_title("estimated pdf"); a.legend(fontsize=8)
    b.plot(grid, true_cdf, "k-", lw=2, label="true")
    b.plot(grid, cdf1, "--", color="tab:red", label="step 1: fixed 1.0")
    b.plot(grid, cdf2, ":", color="tab:orange", label="step 2: adaptive")
    b.plot(grid, cdf3, "-", color="tab:green", label="step 3: Silverman")
    b.set_title("estimated CDF (integral of the pdf)"); b.legend(fontsize=8)
    fig.suptitle(f"Phase A / single Gaussian: Parzen Window window-size strategies (n={N})")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_parzen_gaussian.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
