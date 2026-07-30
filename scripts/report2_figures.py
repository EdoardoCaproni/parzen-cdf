"""Explanatory figures for the report (report/report2.tex).

  1. report2_kernels.png  : the window shapes in the library, all at h = 1, with their spreads
  2. report2_schedule.png : a window that never shrinks vs the h1/sqrt(n) schedule
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, parzen
from study2_common import grid_for, ks, ks_floor, sigma_hat

# ------------------------------------------------------------------ fig 1: window shapes
fig, ax = plt.subplots(figsize=(7.6, 4.0))
u = np.linspace(-4.5, 4.5, 800)
for name, k in parzen.KERNELS.items():
    ax.plot(u, k.pdf(u), lw=1.6, label=f"{name}  (std = {k.std:.2f})")
ax.set_xlabel("u")
ax.set_ylabel(r"window function $\varphi(u)$")
ax.set_title("The window shapes, all at h = 1: same area (1), different spreads")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/report2_kernels.png", dpi=120)

# ------------------------------------------------------------------ fig 2: shrink or freeze
TRI = data.asymmetric_trimodal()
GRID = grid_for(TRI)
T_CDF = TRI.cdf(GRID)
NS = [125, 250, 500, 1000, 2000]
SEEDS = range(5)

rules = {
    "h = 1.0 at every n (window never shrinks)": lambda s, n: 1.0,
    "schedule h = 1.5 sigma-hat / sqrt(n)": lambda s, n: 1.5 * sigma_hat(s) / np.sqrt(n),
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
for name, rule in rules.items():
    hs, kss = [], []
    for n in NS:
        h_seed, k_seed = [], []
        for seed in SEEDS:
            s = TRI.sample(n, np.random.default_rng(seed))
            h = rule(s, n)
            h_seed.append(h)
            k_seed.append(ks(parzen.parzen_cdf(GRID, s, h), T_CDF))
        hs.append(np.mean(h_seed))
        kss.append(np.mean(k_seed))
    axes[0].loglog(NS, hs, "o-", ms=4, label=name)
    axes[1].loglog(NS, kss, "o-", ms=4, label=name)
axes[0].set_xlabel("number of samples n")
axes[0].set_ylabel("window size h")
axes[0].set_title("The window size each rule picks")
axes[1].plot(NS, [ks_floor(n) for n in NS], "k:", lw=1.2,
             label="statistical floor 0.87/sqrt(n)")
axes[1].set_xlabel("number of samples n")
axes[1].set_ylabel("CDF gap (KS), mean of 5 seeds")
axes[1].set_title("The resulting error (three-mode mixture)")
for ax in axes:
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/report2_schedule.png", dpi=120)
print("figures -> results/report2_kernels.png, results/report2_schedule.png")
