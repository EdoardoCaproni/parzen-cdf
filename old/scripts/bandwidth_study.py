"""Bandwidth sensitivity study for the logistic Parzen estimator (no neural training).

The bandwidth -- not the sample count -- is the dominant lever on Parzen accuracy. Two facts:

1. Silverman's constant is tuned for the *Gaussian* kernel (std = h). The standard *logistic*
   kernel has std sqrt(pi**2 / 3) ~= 1.81, so our estimate is ~1.81x over-smoothed at Silverman's
   h. The variance-matched scale is sqrt(3)/pi ~= 0.551.
2. Silverman's h ~ n**(-1/5), so the sample count is a weak lever (2x samples -> ~13% smaller h).

For each distribution we scale Silverman's h by a grid of factors and score the Parzen estimate
against the known truth (KS on the CDF, MSE on the pdf), to locate the KS-optimal scale and check
whether the variance-matched scale helps.

    python scripts/bandwidth_study.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, metrics, parzen

SEED = 0
N_SAMPLES = 2000
SCALES = [0.3, 0.4, 0.551, 0.7, 1.0, 1.5, 2.0]
VARIANCE_MATCHED = np.sqrt(3) / np.pi  # ~= 0.551, std-match the logistic kernel to a Gaussian one
RESULTS_DIR = "results"

DISTS = {
    "single_gaussian": (data.single_gaussian(), "N(0,1)"),
    "symmetric_bimodal": (data.symmetric_bimodal(), "bimodal"),
    "asymmetric_trimodal": (data.asymmetric_trimodal(), "trimodal"),
    "spike_in_broad": (data.spike_in_broad(), "spike-in-broad"),
}


def parzen_scores(samples, h, grid, true_cdf, true_pdf):
    cdf = parzen.parzen_cdf(grid, samples, h)
    pdf = parzen.parzen_pdf(grid, samples, h)
    return metrics.ks_distance(true_cdf, cdf), metrics.mse(true_pdf, pdf)


def run_one(ax, name, mix):
    rng = np.random.default_rng(SEED)
    samples = mix.sample(N_SAMPLES, rng)
    h_silverman = parzen.silverman_bandwidth(samples)
    lo = float(samples.min() - 3 * h_silverman)
    hi = float(samples.max() + 3 * h_silverman)
    grid = np.linspace(lo, hi, 4000)
    true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)

    ks_list, mse_list = [], []
    for s in SCALES:
        ks, mse = parzen_scores(samples, s * h_silverman, grid, true_cdf, true_pdf)
        ks_list.append(ks)
        mse_list.append(mse)

    best_i = int(np.argmin(ks_list))
    print(f"\n=== {name} (Silverman h={h_silverman:.4f}) ===")
    print(f"{'scale':>7}{'h':>9}{'KS':>9}{'MSE(pdf)':>11}")
    for s, ks, mse in zip(SCALES, ks_list, mse_list):
        tag = "  <- Silverman" if s == 1.0 else ("  <- var-matched" if s == 0.551 else "")
        print(f"{s:>7.3f}{s * h_silverman:>9.4f}{ks:>9.4f}{mse:>11.5f}{tag}")
    print(f"KS-optimal scale = {SCALES[best_i]:.3f} (KS={ks_list[best_i]:.4f}); "
          f"Silverman KS={ks_list[SCALES.index(1.0)]:.4f}")

    ax.plot(SCALES, ks_list, "o-", label="KS(CDF)")
    ax.axvline(1.0, ls=":", color="gray"); ax.text(1.0, ax.get_ylim()[1], "Silverman", fontsize=7, ha="center", va="bottom")
    ax.axvline(VARIANCE_MATCHED, ls="--", color="tab:green")
    ax.text(VARIANCE_MATCHED, ax.get_ylim()[1], "var-matched", fontsize=7, ha="center", va="bottom", color="tab:green")
    ax.scatter([SCALES[best_i]], [ks_list[best_i]], color="red", zorder=5, label=f"KS-opt ({SCALES[best_i]:.2f})")
    ax.set_title(name); ax.set_xlabel("bandwidth scale x Silverman h"); ax.set_ylabel("KS vs true CDF")
    ax.legend(fontsize=7)


def sample_size_check():
    """Show the sample count is a weak lever vs the bandwidth (trimodal)."""
    mix = data.asymmetric_trimodal()
    print("\n=== sample-size check (trimodal): KS vs true CDF ===")
    print(f"{'n':>8}{'Silverman':>12}{'var-matched':>14}")
    for n in (500, 2000, 8000):
        rng = np.random.default_rng(SEED)
        samples = mix.sample(n, rng)
        h = parzen.silverman_bandwidth(samples)
        lo, hi = samples.min() - 3 * h, samples.max() + 3 * h
        grid = np.linspace(lo, hi, 4000)
        true_cdf = mix.cdf(grid)
        ks_silv = metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, h))
        ks_vm = metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, VARIANCE_MATCHED * h))
        print(f"{n:>8}{ks_silv:>12.4f}{ks_vm:>14.4f}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (name, (mix, _)) in zip(axes.flat, DISTS.items()):
        run_one(ax, name, mix)
    fig.suptitle(
        f"Logistic Parzen: KS vs bandwidth scale (n={N_SAMPLES}). "
        f"Variance-matched scale = sqrt(3)/pi = {VARIANCE_MATCHED:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "bandwidth_study.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nsaved -> {path}")
    sample_size_check()
