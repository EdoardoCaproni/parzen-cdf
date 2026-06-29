"""Phase B / capacity push -- can a bigger / longer-trained MLP fit the hard trimodal?

Checkpoint 1 was faithful up to the bimodals but blunted the asymmetric trimodal (a sharp,
well-separated, low-weight mode of width 0.3): net KS ~0.08 vs a Parzen target of ~0.015-0.027 with
the width-32 / 5000-epoch network. Here we scale capacity and training on that exact case to see what
it takes to reach the target. Same pipeline otherwise (consolidated Parzen target, Adam lr 0.03,
downstream rectification). Both budgets.

    python scripts/mlp_capacity.py
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
LR = 0.03
BUDGETS = {"under_2k": (1000, "lscv"), "overall": (20000, "variance_matched")}
CONFIGS = [  # label -> (hidden_sizes, epochs)
    ("w32 e5000 (checkpoint)", (32,), 5000),
    ("w64 e15000", (64,), 15000),
    ("w128 e15000", (128,), 15000),
    ("w32x2 e15000", (32, 32), 15000),
]

mix = data.asymmetric_trimodal()
lo = float((mix.means - 5 * mix.stds).min()); hi = float((mix.means + 5 * mix.stds).max())
GRID = np.linspace(lo, hi, 2000); GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF, TRUE_PDF = mix.cdf(GRID), mix.pdf(GRID)


def window(samples, selector):
    return parzen.lscv_bandwidth(samples) if selector == "lscv" else parzen.variance_matched_bandwidth(samples)


def run(samples, inputs, targets, hidden, epochs):
    set_seed(SEED)
    model = CDFNet(in_dim=1, hidden_sizes=hidden, activation="sigmoid", monotone=False)
    model, _ = train_cdf(model, inputs, targets, TrainConfig(optimizer="adam", lr=LR, epochs=epochs, seed=SEED))
    cdf, pdf = rectify_cdf(model(GRID_T).detach().numpy(), GRID)
    return cdf, pdf, metrics.ks_distance(TRUE_CDF, cdf), metrics.mse(TRUE_PDF, pdf)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_overall = {}
    for tag, (n, selector) in BUDGETS.items():
        samples = mix.sample(n, np.random.default_rng(SEED))
        h = window(samples, selector)
        inputs, targets = make_sample_training_set(samples, h)
        parzen_ks = metrics.ks_distance(TRUE_CDF, parzen.parzen_cdf(GRID, samples, h))
        print(f"\n[{tag} n={n}, target={selector}]  Parzen target KS={parzen_ks:.4f}")
        for label, hidden, epochs in CONFIGS:
            cdf, pdf, ks, pmse = run(samples, inputs, targets, hidden, epochs)
            print(f"   {label:<24} net KS={ks:.4f}  pdf MSE={pmse:.5f}")
            if tag == "overall":
                best_overall[label] = (cdf, pdf, ks)

    # figure: overall-budget pdf, checkpoint config vs the best scaled config vs truth/target
    samples = mix.sample(20000, np.random.default_rng(SEED))
    h = window(samples, "variance_matched")
    parzen_pdf_g = parzen.parzen_pdf(GRID, samples, h)
    base_label = CONFIGS[0][0]
    best_label = min(best_overall, key=lambda k: best_overall[k][2])
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(13, 5))
    ax_c.plot(GRID, TRUE_CDF, "k-", lw=2, label="true")
    ax_c.plot(GRID, best_overall[base_label][0], "-", color="tab:red", alpha=0.8,
              label=f"{base_label} (KS {best_overall[base_label][2]:.3f})")
    ax_c.plot(GRID, best_overall[best_label][0], "-", color="tab:green",
              label=f"{best_label} (KS {best_overall[best_label][2]:.3f})")
    ax_c.set_title("trimodal CDF (n=20000)"); ax_c.legend(fontsize=8)
    ax_p.plot(GRID, TRUE_PDF, "k-", lw=2, label="true")
    ax_p.plot(GRID, parzen_pdf_g, "--", color="gray", label="Parzen target")
    ax_p.plot(GRID, best_overall[base_label][1], "-", color="tab:red", alpha=0.8, label=base_label)
    ax_p.plot(GRID, best_overall[best_label][1], "-", color="tab:green", label=best_label)
    ax_p.set_title("trimodal pdf (n=20000)"); ax_p.legend(fontsize=8)
    fig.suptitle("Capacity push on the asymmetric trimodal: checkpoint width-32 vs a scaled network")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "mlp_capacity_trimodal.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n  best overall config: {best_label}")
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
