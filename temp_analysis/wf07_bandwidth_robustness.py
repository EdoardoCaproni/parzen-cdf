"""wf07: robustness to the target window-size choice.

Question: is the MLP-on-Parzen net LESS sensitive to the window size h than the
Parzen estimate is to its own h? We sweep h over a wide range on the symmetric
bimodal (n=2000, 3 seeds), measure the CDF gap (KS vs the true CDF) for both the
Parzen estimate and the net trained on Parzen(x_i) labels, and compare the SPREAD
of those gaps across windows.

Mechanism under test: at small h the Parzen CDF is a noisy near-step function; the
smooth, limited-capacity sigmoidal net regresses through that noise (implicit
regularization), so its gap should degrade less than Parzen's at small h. At large
h both over-smooth and should track each other.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Machine is heavily oversubscribed; pin to 1 thread to avoid contention thrash.
torch.set_num_threads(1)

from parzen_cdf import data, parzen, metrics
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig, make_sample_training_set, train_cdf, rectify_cdf, set_seed,
)

N = 2000
SEEDS = [0, 1, 2]
EPOCHS = 800
HIDDEN = (32,)
GRID = np.linspace(-6.0, 6.0, 1000)

dist = data.symmetric_bimodal()
cdf_true = dist.cdf(GRID)
grid_t = torch.as_tensor(GRID, dtype=torch.float32).reshape(-1, 1)

# Wide window sweep, anchored on the per-dataset Silverman h as multiplicative factors.
FACTORS = np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 7.0])


def net_cdf_on_grid(samples, h, seed):
    inputs_t, targets_t = make_sample_training_set(samples, h)
    inputs_t = inputs_t.reshape(-1, 1)
    set_seed(seed)
    model = CDFNet(in_dim=1, hidden_sizes=HIDDEN, activation="sigmoid", monotone=False)
    train_cdf(model, inputs_t, targets_t,
              TrainConfig(optimizer="adam", lr=0.03, epochs=EPOCHS, seed=seed))
    with torch.no_grad():
        raw = model(grid_t).reshape(-1).numpy()
    # Enforce the hard monotonicity constraint downstream (rectification).
    cdf_rect, _ = rectify_cdf(raw, GRID)
    return cdf_rect


# results[seed][factor] = (parzen_ks, net_ks)
parzen_ks = np.zeros((len(SEEDS), len(FACTORS)))
net_ks = np.zeros((len(SEEDS), len(FACTORS)))

for si, seed in enumerate(SEEDS):
    rng = np.random.default_rng(seed)
    samples = dist.sample(N, rng)
    h_silver = parzen.silverman_bandwidth(samples)
    for fi, f in enumerate(FACTORS):
        h = f * h_silver
        p_cdf = parzen.parzen_cdf(GRID, samples, h)
        parzen_ks[si, fi] = metrics.ks_distance(cdf_true, p_cdf)
        net_ks[si, fi] = metrics.ks_distance(cdf_true, net_cdf_on_grid(samples, h, seed))
        print(f"seed={seed} f={f:>4} h={h:.4f}  parzen_KS={parzen_ks[si,fi]:.4f}  net_KS={net_ks[si,fi]:.4f}")

p_mean = parzen_ks.mean(axis=0)
n_mean = net_ks.mean(axis=0)

print("\n=== mean CDF gap (KS) across seeds, per window factor ===")
print(f"{'factor':>8} {'parzen_KS':>11} {'net_KS':>9}")
for fi, f in enumerate(FACTORS):
    print(f"{f:>8} {p_mean[fi]:>11.4f} {n_mean[fi]:>9.4f}")

# Spread of the gap across the swept windows (per seed, then averaged).
def spread_stats(arr):
    # arr: (seeds, factors). Per-seed spread across factors, then mean over seeds.
    per_seed_std = arr.std(axis=1, ddof=1)
    per_seed_range = arr.max(axis=1) - arr.min(axis=1)
    per_seed_best = arr.min(axis=1)
    per_seed_worst = arr.max(axis=1)
    return (per_seed_std.mean(), per_seed_range.mean(),
            per_seed_best.mean(), per_seed_worst.mean())

p_std, p_range, p_best, p_worst = spread_stats(parzen_ks)
n_std, n_range, n_best, n_worst = spread_stats(net_ks)

print("\n=== SPREAD of CDF gap across the window sweep (mean over seeds) ===")
print(f"{'':<10}{'std':>10}{'range':>10}{'best':>10}{'worst':>10}")
print(f"{'parzen':<10}{p_std:>10.4f}{p_range:>10.4f}{p_best:>10.4f}{p_worst:>10.4f}")
print(f"{'net':<10}{n_std:>10.4f}{n_range:>10.4f}{n_best:>10.4f}{n_worst:>10.4f}")
print(f"\nstd ratio   (net/parzen) = {n_std / p_std:.3f}")
print(f"range ratio (net/parzen) = {n_range / p_range:.3f}")
print(f"worst ratio (net/parzen) = {n_worst / p_worst:.3f}")

# Small-window subset: where the regularization story should bite hardest.
small = FACTORS <= 0.5
ps_worst = parzen_ks[:, small].max(axis=1).mean()
ns_worst = net_ks[:, small].max(axis=1).mean()
print(f"\nSmall-window subset (factor<=0.5): mean worst parzen_KS={ps_worst:.4f}  net_KS={ns_worst:.4f}  (net/parzen={ns_worst/ps_worst:.3f})")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(FACTORS, p_mean, yerr=parzen_ks.std(axis=0), marker="o", label="Parzen CDF gap (KS)", capsize=3)
ax.errorbar(FACTORS, n_mean, yerr=net_ks.std(axis=0), marker="s", label="Net CDF gap (KS)", capsize=3)
ax.set_xscale("log")
ax.set_xlabel("window factor (x Silverman h)")
ax.set_ylabel("KS distance to true CDF")
ax.set_title("Sensitivity of CDF gap to window size (symmetric bimodal, n=2000)")
ax.axvline(1.0, color="gray", ls=":", alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig("temp_analysis/wf07_bandwidth_robustness.png", dpi=110)
print("\nsaved temp_analysis/wf07_bandwidth_robustness.png")
