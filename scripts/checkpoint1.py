"""Checkpoint 1 -- the consolidated end-to-end univariate pipeline.

Brings together everything from Phase A and B into one reproducible run:
  samples  ->  Parzen CDF (consolidated window: LSCV if n<=2000, else variance-matched)
           ->  labels (x_i, F_hat(x_i)) at the data points only
           ->  MLP (one hidden layer, width 32, sigmoid, Adam, fixed lr, full batch, no val split)
           ->  CDF = network, pdf = its derivative
           ->  downstream rectification (cumulative-max + rescale): a monotone CDF and unit-mass pdf.

Demonstrated on a ladder (single Gaussian, the two bimodals, the asymmetric trimodal) at both carried
budgets (under-2k n=1000, overall n=20000). Scored against the known truth. The trained overall-budget
models are saved to results/checkpoint1.pt.

    python scripts/checkpoint1.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, rectify_cdf, set_seed, train_cdf

RESULTS_DIR = "results"
SEED = 0
WIDTH = 32
LR = 0.03
EPOCHS = 5000
BUDGETS = {"under_2k": 1000, "overall": 20000}
DISTS = {
    "single_gaussian": data.single_gaussian(),
    "symmetric_bimodal": data.symmetric_bimodal(),
    "asymmetric_bimodal": data.asymmetric_bimodal(),
    "asymmetric_trimodal": data.asymmetric_trimodal(),
}


def consolidated_window(samples):
    """The Phase-A consolidated truth-free window: LSCV at small n, variance-matched at large n."""
    return parzen.lscv_bandwidth(samples) if samples.size <= 2000 else parzen.variance_matched_bandwidth(samples)


def run(mix, n, grid, grid_t, true_cdf, true_pdf):
    samples = mix.sample(n, np.random.default_rng(SEED))
    h = consolidated_window(samples)
    inputs, targets = make_sample_training_set(samples, h)
    parzen_ks = metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, h))

    set_seed(SEED)
    model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), activation="sigmoid", monotone=False)
    model, _ = train_cdf(model, inputs, targets, TrainConfig(optimizer="adam", lr=LR, epochs=EPOCHS, seed=SEED))
    raw_cdf = model(grid_t).detach().numpy()
    cdf, pdf = rectify_cdf(raw_cdf, grid)        # monotone CDF + unit-mass pdf
    return {
        "model": model, "parzen_ks": parzen_ks,
        "net_ks": metrics.ks_distance(true_cdf, cdf), "pdf_mse": metrics.mse(true_pdf, pdf),
        "viol": metrics.monotonicity_violation_fraction(cdf), "mass": metrics.integrates_to_one(pdf, grid),
        "cdf": cdf, "pdf": pdf,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results, overall_states = {}, {}
    print("Checkpoint 1 -- consolidated pipeline (MLP w32, sigmoid, Adam, rectified). KS vs truth:")
    print(f"  {'distribution':<20}{'budget':<10}{'target':>9}{'net':>9}{'pdfMSE':>9}{'viol%':>7}{'mass':>8}")
    for name, mix in DISTS.items():
        lo = float((mix.means - 5 * mix.stds).min()); hi = float((mix.means + 5 * mix.stds).max())
        grid = np.linspace(lo, hi, 2000); grid_t = torch.as_tensor(grid, dtype=torch.float32)
        true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)
        for tag, n in BUDGETS.items():
            r = run(mix, n, grid, grid_t, true_cdf, true_pdf)
            results[(name, tag)] = {**r, "grid": grid, "true_cdf": true_cdf, "true_pdf": true_pdf}
            if tag == "overall":
                overall_states[name] = r["model"].state_dict()
            print(f"  {name:<20}{tag:<10}{r['parzen_ks']:>9.4f}{r['net_ks']:>9.4f}"
                  f"{r['pdf_mse']:>9.5f}{100*r['viol']:>6.1f}%{r['mass']:>8.4f}")

    torch.save(overall_states, os.path.join(RESULTS_DIR, "checkpoint1.pt"))
    print(f"\n  saved models -> {os.path.join(RESULTS_DIR, 'checkpoint1.pt')}")

    # summary figure at the overall budget
    fig, axes = plt.subplots(len(DISTS), 2, figsize=(12, 14))
    for row, name in enumerate(DISTS):
        d = results[(name, "overall")]
        ax_c, ax_p = axes[row]
        ax_c.plot(d["grid"], d["true_cdf"], "k-", lw=2, label="true")
        ax_c.plot(d["grid"], d["cdf"], "-", color="tab:blue", label=f"pipeline (KS {d['net_ks']:.3f})")
        ax_c.set_title(f"{name}: CDF"); ax_c.legend(fontsize=8)
        ax_p.plot(d["grid"], d["true_pdf"], "k-", lw=2, label="true")
        ax_p.plot(d["grid"], d["pdf"], "-", color="tab:blue", label="pipeline pdf")
        ax_p.set_title(f"{name}: pdf"); ax_p.legend(fontsize=8)
    fig.suptitle("Checkpoint 1: consolidated univariate pipeline at the overall budget (n=20000)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(RESULTS_DIR, "checkpoint1.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
