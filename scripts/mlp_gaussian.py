"""Phase B / steps 1-2 -- the MLP learning the Parzen CDF of the single Gaussian.

The network trains ONLY on the data points: inputs are the n samples x_i, labels are the Parzen CDF
value F_hat(x_i) there (constraint-correct; no collocation, no augmentation). Target = the Silverman
Parzen CDF. We carry the two budgets directly: under-2k n=1000 and overall n=20000.

Network: one hidden layer (width 16), sigmoidal activations (to mimic the logistic-window CDF shape),
fixed learning rate, full batch. We compare two optimizers:
  step 1 -- the simplest: plain SGD at a fixed lr;
  step 2 -- the obvious improvement: Adam at a fixed lr.
The pdf is the derivative of the trained network. Score = CDF gap (KS) vs the known truth; the Parzen
target's own gap is the ceiling the network can hope to match.

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
EPOCHS = 5000
BUDGETS = {"under_2k": 1000, "overall": 20000}
METHODS = {  # name -> (optimizer, fixed lr, colour)
    "SGD (lr 1.0)": ("sgd", 1.0, "tab:orange"),
    "Adam (lr 0.03)": ("adam", 0.03, "tab:blue"),
}

mix = data.single_gaussian()
GRID = np.linspace(-6, 6, 2000)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF, TRUE_PDF = mix.cdf(GRID), mix.pdf(GRID)


def train_one(inputs, targets, optimizer, lr):
    set_seed(SEED)
    model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), activation="sigmoid", monotone=False)
    model, hist = train_cdf(model, inputs, targets, TrainConfig(optimizer=optimizer, lr=lr, epochs=EPOCHS, seed=SEED))
    learned_cdf = model(GRID_T).detach().numpy()
    learned_pdf = density_from_cdf(model, GRID_T, clamp=True).detach().numpy()
    return {
        "ks": metrics.ks_distance(TRUE_CDF, learned_cdf),
        "pdf_mse": metrics.mse(TRUE_PDF, learned_pdf),
        "viol": metrics.monotonicity_violation_fraction(learned_cdf),
        "mass": metrics.integrates_to_one(learned_pdf, GRID),
        "train_mse": float(hist[-1]),
        "cdf": learned_cdf, "pdf": learned_pdf,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    for tag, n in BUDGETS.items():
        samples = mix.sample(n, np.random.default_rng(SEED))
        h = parzen.silverman_bandwidth(samples)
        inputs, targets = make_sample_training_set(samples, h)
        parzen_cdf_g, parzen_pdf_g = parzen.parzen_cdf(GRID, samples, h), parzen.parzen_pdf(GRID, samples, h)
        parzen_ks = metrics.ks_distance(TRUE_CDF, parzen_cdf_g)
        runs = {name: train_one(inputs, targets, opt, lr) for name, (opt, lr, _) in METHODS.items()}
        out[(tag, n)] = {"h": float(h), "parzen_ks": parzen_ks, "parzen_cdf": parzen_cdf_g,
                         "parzen_pdf": parzen_pdf_g, "runs": runs}
        print(f"\n  [{tag} n={n}]  Parzen target KS={parzen_ks:.4f}")
        for name, r in runs.items():
            print(f"      {name:<16} net KS={r['ks']:.4f}  pdf MSE={r['pdf_mse']:.5f}  "
                  f"viol={100*r['viol']:.2f}%  mass={r['mass']:.4f}  trainMSE={r['train_mse']:.6f}")

    _fig_simplest(out)          # step 1 figure (SGD only) -- keeps the report Part IV figure valid
    _fig_compare(out)           # step 2 figure (SGD vs Adam vs target)


def _fig_simplest(out):
    fig, axes = plt.subplots(len(BUDGETS), 2, figsize=(12, 8))
    for row, (tag, n) in enumerate(BUDGETS.items()):
        d = out[(tag, n)]; r = d["runs"]["SGD (lr 1.0)"]
        ax_c, ax_p = axes[row]
        ax_c.plot(GRID, TRUE_CDF, "k-", lw=2, label="true")
        ax_c.plot(GRID, d["parzen_cdf"], "--", color="gray", label=f"Parzen target (KS {d['parzen_ks']:.3f})")
        ax_c.plot(GRID, r["cdf"], "-", color="tab:orange", label=f"MLP (KS {r['ks']:.3f})")
        ax_c.set_title(f"{tag} n={n}: CDF"); ax_c.legend(fontsize=8)
        ax_p.plot(GRID, TRUE_PDF, "k-", lw=2, label="true")
        ax_p.plot(GRID, d["parzen_pdf"], "--", color="gray", label="Parzen target")
        ax_p.plot(GRID, r["pdf"], "-", color="tab:orange", label="MLP (dF/dx)")
        ax_p.set_title(f"{tag} n={n}: pdf = dF/dx"); ax_p.legend(fontsize=8)
    fig.suptitle(f"Phase B step 1: simplest MLP (1 hidden width {WIDTH}, sigmoid, SGD lr=1.0) "
                 "vs the Parzen target and the truth")
    fig.tight_layout(); _save(fig, "mlp_gaussian_simplest.png")


def _fig_compare(out):
    fig, axes = plt.subplots(len(BUDGETS), 2, figsize=(12, 8))
    for row, (tag, n) in enumerate(BUDGETS.items()):
        d = out[(tag, n)]
        ax_c, ax_p = axes[row]
        ax_c.plot(GRID, TRUE_CDF, "k-", lw=2, label="true")
        ax_c.plot(GRID, d["parzen_cdf"], "--", color="gray", label=f"Parzen target ({d['parzen_ks']:.3f})")
        ax_p.plot(GRID, TRUE_PDF, "k-", lw=2, label="true")
        ax_p.plot(GRID, d["parzen_pdf"], "--", color="gray", label="Parzen target")
        for name, (_, _, colour) in METHODS.items():
            r = d["runs"][name]
            ax_c.plot(GRID, r["cdf"], "-", color=colour, label=f"{name} ({r['ks']:.3f})")
            ax_p.plot(GRID, r["pdf"], "-", color=colour, label=name)
        ax_c.set_title(f"{tag} n={n}: CDF"); ax_c.legend(fontsize=8)
        ax_p.set_title(f"{tag} n={n}: pdf = dF/dx"); ax_p.legend(fontsize=8)
    fig.suptitle(f"Phase B step 2: SGD vs Adam (1 hidden width {WIDTH}, sigmoid, fixed lr) "
                 "against the Parzen target")
    fig.tight_layout(); _save(fig, "mlp_gaussian_sgd_vs_adam.png")


def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
