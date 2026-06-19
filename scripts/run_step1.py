"""Reproducible Step 1 run: synthetic data -> Parzen CDF -> MLP -> recovered pdf.

Trains three monotonicity variants (unconstrained baseline, soft derivative penalty, and
monotone-by-construction / Sill) across a sweep of (small) hidden widths, then scores each against
the *true* mixture CDF/pdf. The headline output is a variant x width table -- the "performance at
small size" story.

    python scripts/run_step1.py                 # full sweep, saves a plot to outputs/
    python scripts/run_step1.py --no-plot       # skip plotting
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from parzen_cdf import metrics, parzen
from parzen_cdf.data import default_mixture
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig,
    density_from_cdf,
    make_training_set,
    set_seed,
    train_cdf,
)

# Variant name -> (monotone-by-construction?, monotonicity-penalty weight).
VARIANTS = {
    "baseline": (False, 0.0),
    "soft": (False, 5.0),
    "sill": (True, 0.0),
}


def evaluate(model: CDFNet, grid: np.ndarray, true_cdf: np.ndarray, true_pdf: np.ndarray) -> dict:
    """Score a trained model on a dense grid against the known truth."""
    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    learned_cdf = model(grid_t).detach().numpy()
    learned_pdf = density_from_cdf(model, grid_t, clamp=True).detach().numpy()
    return {
        "ks_cdf": metrics.ks_distance(true_cdf, learned_cdf),
        "mse_pdf": metrics.mse(true_pdf, learned_pdf),
        "viol": metrics.monotonicity_violation_fraction(learned_cdf),
        "pdf_mass": metrics.integrates_to_one(learned_pdf, grid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-train", type=int, default=1024, help="number of collocation points")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    # 1. Known mixture + samples.
    rng = np.random.default_rng(args.seed)
    mix = default_mixture()
    samples = mix.sample(args.n_samples, rng)

    # 2. Bandwidth and Parzen CDF targets on uniform collocation points.
    h = parzen.silverman_bandwidth(samples)
    inputs, targets = make_training_set(samples, h, args.n_train, rng)

    # 3. Dense evaluation grid over the same bounded domain, with the known truth.
    lo, hi = inputs.min().item(), inputs.max().item()
    grid = np.linspace(lo, hi, 2000)
    true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)

    # Parzen reference row (the target the networks are fitting).
    parzen_cdf_grid = parzen.parzen_cdf(grid, samples, h)
    parzen_pdf_grid = parzen.parzen_pdf(grid, samples, h)
    print(f"n_samples={args.n_samples}  h={h:.4f}  domain=[{lo:.2f}, {hi:.2f}]  epochs={args.epochs}")
    print(f"{'reference':<12}{'width':>6}{'KS(cdf)':>10}{'MSE(pdf)':>11}{'viol%':>8}{'pdf_mass':>10}")
    print(
        f"{'parzen':<12}{'-':>6}"
        f"{metrics.ks_distance(true_cdf, parzen_cdf_grid):>10.4f}"
        f"{metrics.mse(true_pdf, parzen_pdf_grid):>11.5f}"
        f"{100 * metrics.monotonicity_violation_fraction(parzen_cdf_grid):>7.1f}%"
        f"{metrics.integrates_to_one(parzen_pdf_grid, grid):>10.4f}"
    )
    print("-" * 57)

    # 4. Train each variant x width and score it.
    results = {}
    for width in args.widths:
        for name, (monotone, weight) in VARIANTS.items():
            set_seed(args.seed)
            model = CDFNet(in_dim=1, hidden_sizes=(width,), monotone=monotone)
            cfg = TrainConfig(epochs=args.epochs, monotonicity_weight=weight, seed=args.seed)
            model, _ = train_cdf(model, inputs, targets, cfg)
            res = evaluate(model, grid, true_cdf, true_pdf)
            results[(width, name)] = (model, res)
            print(
                f"{name:<12}{width:>6}{res['ks_cdf']:>10.4f}{res['mse_pdf']:>11.5f}"
                f"{100 * res['viol']:>7.1f}%{res['pdf_mass']:>10.4f}"
            )
        print("-" * 57)

    if not args.no_plot:
        _save_plot(grid, true_cdf, true_pdf, parzen_cdf_grid, parzen_pdf_grid, results, args.widths)


def _save_plot(grid, true_cdf, true_pdf, parzen_cdf, parzen_pdf, results, widths):
    """Overlay true / Parzen / neural CDF and pdf for the largest swept width."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = max(widths)
    fig, (ax_cdf, ax_pdf) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_cdf.plot(grid, true_cdf, "k-", lw=2, label="true")
    ax_cdf.plot(grid, parzen_cdf, "--", color="gray", label="parzen")
    ax_pdf.plot(grid, true_pdf, "k-", lw=2, label="true")
    ax_pdf.plot(grid, parzen_pdf, "--", color="gray", label="parzen")

    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    for name in VARIANTS:
        model, _ = results[(width, name)]
        ax_cdf.plot(grid, model(grid_t).detach().numpy(), label=name)
        ax_pdf.plot(grid, density_from_cdf(model, grid_t).detach().numpy(), label=name)

    ax_cdf.set_title(f"CDF (width={width})"); ax_cdf.legend()
    ax_pdf.set_title(f"pdf = dF/dx (width={width})"); ax_pdf.legend()
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", f"step1_width{width}.png")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"saved plot -> {path}")


if __name__ == "__main__":
    main()
