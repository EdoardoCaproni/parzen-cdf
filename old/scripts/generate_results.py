"""Generate one results figure per distribution: CDF overlay, pdf overlay, KS-vs-width, and
MSE-vs-width, with the full parameter set in the title. Figures are written to the tracked
``results/`` directory.

    python scripts/generate_results.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig,
    density_from_cdf,
    make_training_set,
    set_seed,
    train_cdf,
)

# ---- Fixed experiment parameters (identical across all distributions) ----
N_SAMPLES = 2000
N_TRAIN = 1024          # uniform collocation points
EPOCHS = 2000
LR = 1e-2
SOFT_LAMBDA = 5.0
SEED = 0
WIDTHS = [4, 8, 16, 32]
OVERLAY_WIDTH = 4        # width shown in the CDF/pdf overlay panels (best-performing small net)
VARIANTS = {"baseline": (False, 0.0), "soft": (False, SOFT_LAMBDA), "sill": (True, 0.0)}
RESULTS_DIR = "results"

DISTS = {
    "single_gaussian": (data.single_gaussian(), "N(0,1)"),
    "symmetric_bimodal": (data.symmetric_bimodal(), "0.5 N(-2,0.7) + 0.5 N(2,0.7)"),
    "asymmetric_trimodal": (data.asymmetric_trimodal(), "0.3 N(-2,0.5) + 0.5 N(1,1.0) + 0.2 N(4,0.3)"),
    "spike_in_broad": (data.spike_in_broad(), "0.6 N(0,1.5) + 0.4 N(0.5,0.2)"),
}


def run_one(name, mix, formula):
    rng = np.random.default_rng(SEED)
    samples = mix.sample(N_SAMPLES, rng)
    h = parzen.silverman_bandwidth(samples)
    inputs, targets = make_training_set(samples, h, N_TRAIN, rng)
    lo, hi = inputs.min().item(), inputs.max().item()
    grid = np.linspace(lo, hi, 2000)
    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)
    pcdf, ppdf = parzen.parzen_cdf(grid, samples, h), parzen.parzen_pdf(grid, samples, h)
    parzen_ks = metrics.ks_distance(true_cdf, pcdf)
    parzen_mse = metrics.mse(true_pdf, ppdf)

    models, ks_curve, mse_curve = {}, {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
    print(f"\n=== {name}: {formula} ===")
    print(f"h={h:.4f}  domain=[{lo:.2f},{hi:.2f}]   parzen: KS={parzen_ks:.4f} MSE(pdf)={parzen_mse:.5f}")
    print(f"{'variant':<10}{'width':>6}{'KS':>9}{'MSE(pdf)':>11}{'viol%':>8}{'mass':>8}")
    for w in WIDTHS:
        for v, (mono, wt) in VARIANTS.items():
            set_seed(SEED)
            m = CDFNet(1, (w,), monotone=mono)
            m, _ = train_cdf(m, inputs, targets, TrainConfig(epochs=EPOCHS, lr=LR, monotonicity_weight=wt, seed=SEED))
            models[(v, w)] = m
            lc = m(grid_t).detach().numpy()
            lp = density_from_cdf(m, grid_t, clamp=True).detach().numpy()
            ks_curve[v].append(metrics.ks_distance(true_cdf, lc))
            mse_curve[v].append(metrics.mse(true_pdf, lp))
            print(f"{v:<10}{w:>6}{ks_curve[v][-1]:>9.4f}{mse_curve[v][-1]:>11.5f}"
                  f"{100 * metrics.monotonicity_violation_fraction(lc):>7.1f}%"
                  f"{metrics.integrates_to_one(lp, grid):>8.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(grid, true_cdf, "k-", lw=2, label="true")
    ax[0, 0].plot(grid, pcdf, "--", color="gray", label="parzen")
    ax[0, 1].plot(grid, true_pdf, "k-", lw=2, label="true")
    ax[0, 1].plot(grid, ppdf, "--", color="gray", label="parzen")
    for v in VARIANTS:
        m = models[(v, OVERLAY_WIDTH)]
        ax[0, 0].plot(grid, m(grid_t).detach().numpy(), label=v, alpha=0.85)
        ax[0, 1].plot(grid, density_from_cdf(m, grid_t).detach().numpy(), label=v, alpha=0.85)
    ax[0, 0].set_title(f"CDF overlay (width={OVERLAY_WIDTH})"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].set_title(f"pdf = dF/dx (width={OVERLAY_WIDTH})"); ax[0, 1].legend(fontsize=8)

    for v in VARIANTS:
        ax[1, 0].plot(WIDTHS, ks_curve[v], "o-", label=v)
        ax[1, 1].plot(WIDTHS, mse_curve[v], "o-", label=v)
    ax[1, 0].axhline(parzen_ks, ls="--", color="gray", label="parzen")
    ax[1, 1].axhline(parzen_mse, ls="--", color="gray", label="parzen")
    ax[1, 0].set_title("KS(CDF) vs width"); ax[1, 0].set_xlabel("hidden width"); ax[1, 0].set_xticks(WIDTHS); ax[1, 0].legend(fontsize=8)
    ax[1, 1].set_title("MSE(pdf) vs width"); ax[1, 1].set_xlabel("hidden width"); ax[1, 1].set_xticks(WIDTHS); ax[1, 1].legend(fontsize=8)

    params = (
        f"{name}:  {formula}\n"
        f"n_samples={N_SAMPLES} | h (Silverman)={h:.3f} | domain=[{lo:.2f}, {hi:.2f}] | "
        f"{N_TRAIN} uniform collocation points\n"
        f"arch: 1 hidden layer + sigmoid, final sigmoid | Adam lr={LR} | epochs={EPOCHS} | "
        f"soft penalty lambda={SOFT_LAMBDA} | seed={SEED}"
    )
    fig.suptitle(params, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(RESULTS_DIR, f"results_{name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, (mix, formula) in DISTS.items():
        run_one(name, mix, formula)
