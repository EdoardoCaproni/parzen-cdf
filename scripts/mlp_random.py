"""Phase B / step 4 -- the MLP on the battery of 10 random complex mixtures.

Does the "faithful learner" result hold on varied, complex shapes? We train the network (one hidden
layer, width 32, sigmoid, Adam lr 0.03) on each of the 10 random mixtures from Phase A step 3, only on
the data points, with the consolidated Parzen target per budget (LSCV at under-2k n=1000,
variance-matched at overall n=20000) and the chosen monotonicity strategy (downstream rectification:
cumulative-max + rescale). One sample set per (mixture, budget), seeded by mixture index.

We report, across the 10 mixtures: the Parzen target gap (ceiling), the network's gap after
rectification, and how often the raw (pre-rectification) network violated monotonicity. Score = CDF
gap (KS) vs the known truth.

    python scripts/mlp_random.py
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
N_MIXTURES = 10
WIDTH = 32
LR = 0.03
EPOCHS = 5000
BUDGETS = {"under_2k": (1000, "lscv"), "overall": (20000, "variance_matched")}


def make_mixtures():
    rng = np.random.default_rng(0)
    return [data.random_mixture(rng) for _ in range(N_MIXTURES)]


def grid_for(mix):
    lo = float((mix.means - 5 * mix.stds).min()); hi = float((mix.means + 5 * mix.stds).max())
    return np.linspace(lo, hi, 1500)


def window(samples, selector):
    return parzen.lscv_bandwidth(samples) if selector == "lscv" else parzen.variance_matched_bandwidth(samples)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mixes = make_mixtures()
    agg = {b: {"parzen": [], "net": [], "raw_viol": []} for b in BUDGETS}
    gallery = []  # (grid, true_pdf, net_pdf at overall)

    for i, mix in enumerate(mixes):
        grid = grid_for(mix); grid_t = torch.as_tensor(grid, dtype=torch.float32)
        true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)
        print(f"\nmixture {i}: k={mix.n_components}")
        for b, (n, selector) in BUDGETS.items():
            samples = mix.sample(n, np.random.default_rng(i))
            h = window(samples, selector)
            inputs, targets = make_sample_training_set(samples, h)
            parzen_ks = metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, h))

            set_seed(i)
            model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), activation="sigmoid", monotone=False)
            model, _ = train_cdf(model, inputs, targets, TrainConfig(optimizer="adam", lr=LR, epochs=EPOCHS, seed=i))
            raw_cdf = model(grid_t).detach().numpy()
            raw_viol = metrics.monotonicity_violation_fraction(raw_cdf)
            rect_cdf, rect_pdf = rectify_cdf(raw_cdf, grid)               # chosen monotonicity strategy
            net_ks = metrics.ks_distance(true_cdf, rect_cdf)

            agg[b]["parzen"].append(parzen_ks); agg[b]["net"].append(net_ks); agg[b]["raw_viol"].append(raw_viol)
            print(f"  [{b:8} n={n:5}] target KS={parzen_ks:.4f}  net(rectified) KS={net_ks:.4f}  "
                  f"raw viol={100*raw_viol:.2f}%")
            if b == "overall":
                gallery.append((grid, true_pdf, rect_pdf))

    print("\n===== mean across the 10 mixtures =====")
    for b, (n, sel) in BUDGETS.items():
        a = agg[b]
        print(f"  {b} (n={n}, target={sel}): Parzen {np.mean(a['parzen']):.4f}  ->  "
              f"net(rectified) {np.mean(a['net']):.4f}   (mean raw viol {100*np.mean(a['raw_viol']):.2f}%)")

    _scatter(agg)
    _gallery(gallery)


def _scatter(agg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (b, (n, sel)) in zip(axes, BUDGETS.items()):
        x, y = agg[b]["parzen"], agg[b]["net"]
        ax.scatter(x, y, color="tab:blue")
        lim = max(max(x), max(y)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="net = target")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Parzen target CDF gap"); ax.set_ylabel("net (rectified) CDF gap")
        ax.set_title(f"{b} (n={n}, {sel})"); ax.legend(fontsize=8)
    fig.suptitle("Phase B step 4: net vs its Parzen target across 10 random mixtures "
                 "(points on the diagonal = faithful learner)")
    fig.tight_layout(); _save(fig, "mlp_random_vs_target.png")


def _gallery(gallery):
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for ax, (grid, true_pdf, net_pdf) in zip(axes.flat, gallery):
        ax.plot(grid, true_pdf, "k-", lw=1.4)
        ax.plot(grid, net_pdf, "-", color="tab:blue", lw=1.3)
        ax.set_yticks([])
    axes.flat[0].legend(["true", "MLP (rectified dF/dx)"], fontsize=7)
    fig.suptitle("The 10 random mixtures: true density (black) vs MLP recovered density (blue), "
                 "n=20000")
    fig.tight_layout(); _save(fig, "mlp_random_gallery.png")


def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
