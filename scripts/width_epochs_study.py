"""Diagnostic: is the larger network worse because it overfits the (noisy) Parzen target, or
because it is undertrained?

The two hypotheses are distinguishable by comparing the *training* MSE (network vs the Parzen
target it is fit to) against the *test* KS (network CDF vs the true CDF):

- Overfitting  : training MSE keeps falling with width while KS vs truth rises.
- Underfitting : training MSE is *higher* at large width (the bigger net has not converged).

We measure both, on the two hard cases (trimodal, spike-in-broad), with the unconstrained
baseline net. We then sweep the number of epochs at width 32 to test whether more training fixes
it.

    python scripts/width_epochs_study.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_training_set, set_seed, train_cdf

SEED = 0
N_SAMPLES = 2000
N_TRAIN = 1024
LR = 1e-2
BASE_EPOCHS = 2000
WIDTHS = [4, 8, 16, 32]
EPOCH_GRID = [2000, 5000, 10000]
RESULTS_DIR = "results"

DISTS = {
    "trimodal": data.asymmetric_trimodal(),
    "spike_in_broad": data.spike_in_broad(),
}


def setup(mix):
    rng = np.random.default_rng(SEED)
    samples = mix.sample(N_SAMPLES, rng)
    h = parzen.silverman_bandwidth(samples)
    inputs, targets = make_training_set(samples, h, N_TRAIN, rng)
    lo, hi = inputs.min().item(), inputs.max().item()
    grid = np.linspace(lo, hi, 2000)
    return inputs, targets, grid, mix.cdf(grid)


def train(inputs, targets, width, epochs):
    set_seed(SEED)
    model = CDFNet(1, (width,), monotone=False)
    model, history = train_cdf(model, inputs, targets, TrainConfig(epochs=epochs, lr=LR, seed=SEED))
    return model, history[-1]  # final training MSE (vs the Parzen target)


def ks_vs_truth(model, grid, true_cdf):
    learned = model(torch.as_tensor(grid, dtype=torch.float32)).detach().numpy()
    return metrics.ks_distance(true_cdf, learned)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, len(DISTS), figsize=(12, 9))

    for col, (name, mix) in enumerate(DISTS.items()):
        inputs, targets, grid, true_cdf = setup(mix)

        # ---- width sweep at BASE_EPOCHS ----
        train_mse, ks = [], []
        print(f"\n=== {name}: width sweep ({BASE_EPOCHS} epochs) ===")
        print(f"{'width':>6}{'train_MSE':>12}{'KS(truth)':>11}")
        for w in WIDTHS:
            model, tmse = train(inputs, targets, w, BASE_EPOCHS)
            k = ks_vs_truth(model, grid, true_cdf)
            train_mse.append(tmse); ks.append(k)
            print(f"{w:>6}{tmse:>12.6f}{k:>11.4f}")

        ax = axes[0, col]
        ax.plot(WIDTHS, ks, "o-", color="tab:blue", label="KS vs truth")
        ax.set_ylabel("KS vs true CDF", color="tab:blue"); ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(WIDTHS, train_mse, "s--", color="tab:red", label="train MSE")
        ax2.set_ylabel("train MSE vs Parzen", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"{name}: width sweep"); ax.set_xlabel("hidden width"); ax.set_xticks(WIDTHS)

        # ---- epochs sweep at width 32 ----
        ep_train_mse, ep_ks = [], []
        print(f"=== {name}: epochs sweep at width 32 ===")
        print(f"{'epochs':>7}{'train_MSE':>12}{'KS(truth)':>11}")
        for ep in EPOCH_GRID:
            model, tmse = train(inputs, targets, 32, ep)
            k = ks_vs_truth(model, grid, true_cdf)
            ep_train_mse.append(tmse); ep_ks.append(k)
            print(f"{ep:>7}{tmse:>12.6f}{k:>11.4f}")

        ax = axes[1, col]
        ax.plot(EPOCH_GRID, ep_ks, "o-", color="tab:blue", label="KS vs truth")
        ax.set_ylabel("KS vs true CDF", color="tab:blue"); ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(EPOCH_GRID, ep_train_mse, "s--", color="tab:red", label="train MSE")
        ax2.set_ylabel("train MSE vs Parzen", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"{name}: width 32, epochs sweep"); ax.set_xlabel("epochs"); ax.set_xticks(EPOCH_GRID)

    fig.suptitle("Width / epochs diagnostic (unconstrained baseline): KS vs truth (blue) and "
                 "training MSE vs Parzen target (red)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "width_epochs_study.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    run()
