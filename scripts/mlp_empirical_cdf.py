"""Phase B / side study -- can the MLP learn the CDF directly from real data (the empirical CDF)?

So far the network's labels are the Parzen CDF at the data points. Here we ask whether the Parzen
step is even needed: we train the same network directly on the *empirical CDF* of the samples, the
rawest estimate of the true CDF from real data. At each sample x_i the label is its plotting position
F_n(x_i) = (rank - 0.5)/n (a noisy, step-like sampling of the true CDF). The smooth network regresses
through these points, so it effectively smooths the empirical CDF on its own.

We compare, on the single Gaussian at both budgets (n=1000, n=20000):
  - net trained on the Parzen target (the current pipeline);
  - net trained on the empirical CDF (no Parzen);
both with the same simplest network (1 hidden width 16, sigmoid, Adam lr 0.03, full batch, no
train/val split). Score = CDF gap (KS) vs the known truth; pdf via the network derivative.

    python scripts/mlp_empirical_cdf.py
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
SEEDS = range(5)
WIDTH = 16
LR = 0.03
EPOCHS = 5000
BUDGETS = {"under_2k": 1000, "overall": 20000}

mix = data.single_gaussian()
GRID = np.linspace(-6, 6, 2000)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF, TRUE_PDF = mix.cdf(GRID), mix.pdf(GRID)


def empirical_cdf_labels(samples):
    """Plotting-position empirical CDF at each sample: (rank - 0.5)/n, in (0, 1)."""
    n = samples.size
    ranks = np.empty(n)
    ranks[np.argsort(samples)] = np.arange(n)
    return (ranks + 0.5) / n


def train_on(inputs_t, targets_t, seed):
    set_seed(seed)
    model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), activation="sigmoid", monotone=False)
    model, _ = train_cdf(model, inputs_t, targets_t, TrainConfig(optimizer="adam", lr=LR, epochs=EPOCHS, seed=seed))
    cdf = model(GRID_T).detach().numpy()
    pdf = density_from_cdf(model, GRID_T, clamp=True).detach().numpy()
    return cdf, pdf, metrics.ks_distance(TRUE_CDF, cdf), metrics.integrates_to_one(pdf, GRID)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    for tag, n in BUDGETS.items():
        acc = {k: [] for k in ("parzen_ks", "emp_ks", "ks_p", "ks_e", "mass_e")}
        seed0 = None
        for seed in SEEDS:
            samples = mix.sample(n, np.random.default_rng(seed))
            h = parzen.silverman_bandwidth(samples)
            inp_parzen, tgt_parzen = make_sample_training_set(samples, h)            # Parzen CDF labels
            inp_emp = torch.as_tensor(samples, dtype=torch.float32)
            tgt_emp = torch.as_tensor(empirical_cdf_labels(samples), dtype=torch.float32)  # empirical labels
            sorted_s = np.sort(samples)
            emp_on_grid = np.searchsorted(sorted_s, GRID, side="right") / n
            acc["parzen_ks"].append(metrics.ks_distance(TRUE_CDF, parzen.parzen_cdf(GRID, samples, h)))
            acc["emp_ks"].append(metrics.ks_distance(TRUE_CDF, emp_on_grid))
            cdf_p, pdf_p, ks_p, _ = train_on(inp_parzen, tgt_parzen, seed)
            cdf_e, pdf_e, ks_e, mass_e = train_on(inp_emp, tgt_emp, seed)
            acc["ks_p"].append(ks_p); acc["ks_e"].append(ks_e); acc["mass_e"].append(mass_e)
            if seed == 0:
                seed0 = dict(emp_on_grid=emp_on_grid, parzen_cdf=parzen.parzen_cdf(GRID, samples, h),
                             cdf_p=cdf_p, pdf_p=pdf_p, ks_p=ks_p, cdf_e=cdf_e, pdf_e=pdf_e, ks_e=ks_e,
                             emp_ks=acc["emp_ks"][0])
        stats = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
        out[(tag, n)] = {"stats": stats, **seed0}
        print(f"\n  [{tag} n={n}]  mean +/- std over {len(list(SEEDS))} seeds")
        print(f"      target Parzen CDF   KS = {stats['parzen_ks'][0]:.4f} +/- {stats['parzen_ks'][1]:.4f}")
        print(f"      target empirical CDF KS = {stats['emp_ks'][0]:.4f} +/- {stats['emp_ks'][1]:.4f}")
        print(f"      net on Parzen        KS = {stats['ks_p'][0]:.4f} +/- {stats['ks_p'][1]:.4f}")
        print(f"      net on empirical     KS = {stats['ks_e'][0]:.4f} +/- {stats['ks_e'][1]:.4f}   (mass {stats['mass_e'][0]:.4f})")

    fig, axes = plt.subplots(len(BUDGETS), 2, figsize=(12, 8))
    for row, (tag, n) in enumerate(BUDGETS.items()):
        d = out[(tag, n)]
        ax_c, ax_p = axes[row]
        ax_c.plot(GRID, d["emp_on_grid"], "-", color="lightgray", lw=1, label=f"empirical CDF (KS {d['emp_ks']:.3f})")
        ax_c.plot(GRID, TRUE_CDF, "k-", lw=2, label="true")
        ax_c.plot(GRID, d["cdf_p"], "-", color="tab:blue", label=f"net on Parzen (KS {d['ks_p']:.3f})")
        ax_c.plot(GRID, d["cdf_e"], "-", color="tab:green", label=f"net on empirical (KS {d['ks_e']:.3f})")
        ax_c.set_title(f"{tag} n={n}: CDF"); ax_c.legend(fontsize=8)
        ax_p.plot(GRID, TRUE_PDF, "k-", lw=2, label="true")
        ax_p.plot(GRID, d["pdf_p"], "-", color="tab:blue", label="net on Parzen (dF/dx)")
        ax_p.plot(GRID, d["pdf_e"], "-", color="tab:green", label="net on empirical (dF/dx)")
        ax_p.set_title(f"{tag} n={n}: pdf = dF/dx"); ax_p.legend(fontsize=8)
    fig.suptitle("Can the MLP learn the CDF directly from real data? "
                 "Net trained on the empirical CDF vs on the Parzen target (single Gaussian, Adam)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "mlp_empirical_cdf.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    main()
