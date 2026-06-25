"""Experiment 0 -- Naive baseline, constraint-correct regime.

The canonical pipeline: sample a known Gaussian mixture, fit a fixed-bandwidth (Silverman) logistic
Parzen estimator, then train a small sigmoidal MLP on the DATA POINTS ONLY -- inputs = the n drawn
samples x_i, targets = F_hat(x_i). No collocation, no synthetic x, no augmentation. The pdf is the
autograd derivative of the learned CDF. Monotonicity (a hard constraint) is handled three ways, so
this baseline doubles as the monotonicity-strategy anchor for the ablation:

    baseline : unconstrained MLP (monotonicity only observed, not enforced) -- shows the problem.
    soft     : + soft penalty mean(relu(-dF/dx)) on a uniform grid over the observed data range.
    sill     : monotone by construction (Sill 1998; non-negative weights), zero extra points.

Scored against the *known* truth (the referee): KS on the CDF, MSE on the pdf, plus the monotonicity
violation fraction and the recovered pdf mass. The Parzen row is the target the MLP is fitting (the
ceiling for this h).

    python scripts/experiments/e0_naive_baseline.py
"""

import json
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
    make_sample_training_set,
    set_seed,
    train_cdf,
)

SEED = 0
N_SAMPLES = 2000
WIDTH = 16
EPOCHS = 2000
LR = 1e-2
SOFT_LAMBDA = 5.0
VARIANTS = {"baseline": (False, 0.0), "soft": (False, SOFT_LAMBDA), "sill": (True, 0.0)}
RESULTS_DIR = "results"

DISTS = {
    "single_gaussian": (data.single_gaussian(), "N(0,1)"),
    "symmetric_bimodal": (data.symmetric_bimodal(), "0.5 N(-2,0.7) + 0.5 N(2,0.7)"),
    "asymmetric_trimodal": (data.asymmetric_trimodal(), "0.3 N(-2,0.5) + 0.5 N(1,1.0) + 0.2 N(4,0.3)"),
    "spike_in_broad": (data.spike_in_broad(), "0.6 N(0,1.5) + 0.4 N(0.5,0.2)"),
}


def evaluate(model, grid, grid_t, true_cdf, true_pdf):
    learned_cdf = model(grid_t).detach().numpy()
    learned_pdf = density_from_cdf(model, grid_t, clamp=True).detach().numpy()
    return {
        "ks_cdf": metrics.ks_distance(true_cdf, learned_cdf),
        "mse_pdf": metrics.mse(true_pdf, learned_pdf),
        "viol_frac": metrics.monotonicity_violation_fraction(learned_cdf),
        "pdf_mass": metrics.integrates_to_one(learned_pdf, grid),
    }


def run_one(name, mix):
    rng = np.random.default_rng(SEED)
    samples = mix.sample(N_SAMPLES, rng)
    h = parzen.silverman_bandwidth(samples)
    inputs, targets = make_sample_training_set(samples, h)

    lo = float(samples.min() - 3 * h)
    hi = float(samples.max() + 3 * h)
    grid = np.linspace(lo, hi, 2000)
    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)

    parzen_cdf_g = parzen.parzen_cdf(grid, samples, h)
    parzen_pdf_g = parzen.parzen_pdf(grid, samples, h)
    rows = {
        "parzen": {
            "ks_cdf": metrics.ks_distance(true_cdf, parzen_cdf_g),
            "mse_pdf": metrics.mse(true_pdf, parzen_pdf_g),
            "viol_frac": metrics.monotonicity_violation_fraction(parzen_cdf_g),
            "pdf_mass": metrics.integrates_to_one(parzen_pdf_g, grid),
        }
    }

    models = {}
    for v, (mono, wt) in VARIANTS.items():
        set_seed(SEED)
        m = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), monotone=mono)
        m, hist = train_cdf(m, inputs, targets, TrainConfig(epochs=EPOCHS, lr=LR, monotonicity_weight=wt, seed=SEED))
        models[v] = m
        res = evaluate(m, grid, grid_t, true_cdf, true_pdf)
        res["train_mse"] = float(torch.mean((m(inputs).detach() - targets) ** 2))
        rows[v] = res

    return dict(h=float(h), n=N_SAMPLES, grid=grid, true_cdf=true_cdf, true_pdf=true_pdf,
                parzen_cdf=parzen_cdf_g, parzen_pdf=parzen_pdf_g, models=models, grid_t=grid_t, rows=rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}
    for name, (mix, formula) in DISTS.items():
        out = run_one(name, mix)
        summary[name] = {"formula": formula, "h": out["h"], "n": out["n"], "rows": out["rows"]}
        print(f"\n=== {name}: {formula}  (h_silverman={out['h']:.4f}, n={out['n']}, train on data points only) ===")
        print(f"{'variant':<10}{'KS(cdf)':>9}{'MSE(pdf)':>11}{'viol%':>8}{'pdf_mass':>10}{'train_MSE':>11}")
        for v in ["parzen", "baseline", "soft", "sill"]:
            r = out["rows"][v]
            tmse = f"{r['train_mse']:.6f}" if "train_mse" in r else "-"
            print(f"{v:<10}{r['ks_cdf']:>9.4f}{r['mse_pdf']:>11.5f}{100 * r['viol_frac']:>7.2f}%{r['pdf_mass']:>10.4f}{tmse:>11}")
        if name == "asymmetric_trimodal":
            _save_figure(out, name, formula)

    with open(os.path.join(RESULTS_DIR, "e0_naive_baseline.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {os.path.join(RESULTS_DIR, 'e0_naive_baseline.json')}")
    _print_markdown(summary)


def _save_figure(out, name, formula):
    grid, grid_t = out["grid"], out["grid_t"]
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax_c.plot(grid, out["true_cdf"], "k-", lw=2, label="true")
    ax_c.plot(grid, out["parzen_cdf"], "--", color="gray", label="parzen (target)")
    ax_p.plot(grid, out["true_pdf"], "k-", lw=2, label="true")
    ax_p.plot(grid, out["parzen_pdf"], "--", color="gray", label="parzen (target)")
    for v, m in out["models"].items():
        ax_c.plot(grid, m(grid_t).detach().numpy(), alpha=0.85, label=v)
        ax_p.plot(grid, density_from_cdf(m, grid_t).detach().numpy(), alpha=0.85, label=v)
    ax_c.set_title("CDF"); ax_c.legend(fontsize=8)
    ax_p.set_title("pdf = dF/dx (autograd, clamped)"); ax_p.legend(fontsize=8)
    fig.suptitle(f"E0 naive baseline (data points only) -- {name}: {formula}\n"
                 f"n={out['n']} | h (Silverman)={out['h']:.3f} | width={WIDTH} sigmoid | Adam lr={LR} | epochs={EPOCHS}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    path = os.path.join(RESULTS_DIR, f"e0_naive_{name}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"saved -> {path}")


def _print_markdown(summary):
    print("\n----- markdown -----")
    for name, d in summary.items():
        print(f"\n**{name}** (h={d['h']:.3f}, n={d['n']}):\n")
        print("| variant | KS(CDF) | MSE(pdf) | viol% | pdf mass |")
        print("|---|---|---|---|---|")
        for v in ["parzen", "baseline", "soft", "sill"]:
            r = d["rows"][v]
            print(f"| {v} | {r['ks_cdf']:.4f} | {r['mse_pdf']:.5f} | {100 * r['viol_frac']:.2f}% | {r['pdf_mass']:.4f} |")


if __name__ == "__main__":
    main()
