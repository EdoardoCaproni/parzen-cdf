"""Phase B / step 1 -- the simplest possible MLP learning the Parzen CDF of the single Gaussian.

The network trains ONLY on the data points: inputs are the n samples x_i, labels are the Parzen CDF
value F_hat(x_i) there (constraint-correct; no collocation, no augmentation). We use the consolidated
Parzen target from Phase A; on the single Gaussian Silverman is excellent, so the target is the
Silverman Parzen CDF. We carry the two budgets directly (no 2000 step): best under-2k n=1000 and best
overall n=20000.

Simplest network: one hidden layer (width 16), sigmoidal activations (to mimic the logistic-window CDF
shape), a fixed learning rate, plain SGD, full batch. The pdf is the derivative of the trained network
(autograd). Improvements (Adam, more width, ...) come in later steps. Score = CDF gap (KS) vs the
known truth; the Parzen target's own gap is the ceiling the network can hope to match.

    python scripts/mlp_gaussian.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, density_from_cdf, make_sample_training_set, set_seed, train_cdf

RESULTS_DIR = "results"
SEED = 0
WIDTH = 16
LR = 1.0          # fixed learning rate, plain SGD
EPOCHS = 5000
BUDGETS = {"under_2k": 1000, "overall": 20000}

mix = data.single_gaussian()
GRID = np.linspace(-6, 6, 2000)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF, TRUE_PDF = mix.cdf(GRID), mix.pdf(GRID)


def run_budget(n):
    samples = mix.sample(n, np.random.default_rng(SEED))
    h = parzen.silverman_bandwidth(samples)
    inputs, targets = make_sample_training_set(samples, h)        # labels = Parzen CDF at the data points
    parzen_cdf_g = parzen.parzen_cdf(GRID, samples, h)
    parzen_ks = metrics.ks_distance(TRUE_CDF, parzen_cdf_g)        # the ceiling

    set_seed(SEED)
    model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), activation="sigmoid", monotone=False)
    cfg = TrainConfig(optimizer="sgd", lr=LR, epochs=EPOCHS, seed=SEED)
    model, hist = train_cdf(model, inputs, targets, cfg)

    learned_cdf = model(GRID_T).detach().numpy()
    learned_pdf = density_from_cdf(model, GRID_T, clamp=True).detach().numpy()
    return {
        "h": float(h), "parzen_ks": parzen_ks, "parzen_cdf": parzen_cdf_g,
        "net_ks": metrics.ks_distance(TRUE_CDF, learned_cdf),
        "net_pdf_mse": metrics.mse(TRUE_PDF, learned_pdf),
        "viol": metrics.monotonicity_violation_fraction(learned_cdf),
        "mass": metrics.integrates_to_one(learned_pdf, GRID),
        "train_mse": float(hist[-1]),
        "learned_cdf": learned_cdf, "learned_pdf": learned_pdf,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Phase B / step 1 -- simplest MLP (1 hidden, width {WIDTH}, sigmoid, SGD lr={LR}, "
          f"{EPOCHS} epochs), target = Silverman Parzen CDF on the data points")
    out = {}
    for tag, n in BUDGETS.items():
        r = run_budget(n)
        out[(tag, n)] = r
        print(f"\n  [{tag} n={n}]  Parzen target KS={r['parzen_ks']:.4f}  ->  "
              f"net KS={r['net_ks']:.4f}  (train MSE {r['train_mse']:.6f})")
        print(f"      net pdf MSE={r['net_pdf_mse']:.5f}  viol={100*r['viol']:.2f}%  mass={r['mass']:.4f}")

    fig, axes = plt.subplots(len(BUDGETS), 2, figsize=(12, 8))
    for row, (tag, n) in enumerate(BUDGETS.items()):
        r = out[(tag, n)]
        ax_c, ax_p = axes[row]
        ax_c.plot(GRID, TRUE_CDF, "k-", lw=2, label="true")
        ax_c.plot(GRID, r["parzen_cdf"], "--", color="gray", label=f"Parzen target (KS {r['parzen_ks']:.3f})")
        ax_c.plot(GRID, r["learned_cdf"], "-", color="tab:orange", label=f"MLP (KS {r['net_ks']:.3f})")
        ax_c.set_title(f"{tag} n={n}: CDF"); ax_c.legend(fontsize=8)
        ax_p.plot(GRID, TRUE_PDF, "k-", lw=2, label="true")
        ax_p.plot(GRID, parzen.parzen_pdf(GRID, mix.sample(n, np.random.default_rng(SEED)), r["h"]),
                  "--", color="gray", label="Parzen target")
        ax_p.plot(GRID, r["learned_pdf"], "-", color="tab:orange", label="MLP (dF/dx)")
        ax_p.set_title(f"{tag} n={n}: pdf = dF/dx"); ax_p.legend(fontsize=8)
    fig.suptitle(f"Phase B step 1: simplest MLP (1 hidden width {WIDTH}, sigmoid, SGD lr={LR}) "
                 "vs the Parzen target and the truth")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "mlp_gaussian_simplest.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    main()
