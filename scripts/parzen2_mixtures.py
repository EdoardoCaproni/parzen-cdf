"""Study 2, step 2: the h1 stress test on multimodal mixtures.

Same protocol as step 1, on the two bimodals of the original ladder plus the asymmetric
trimodal (the Professor's example). The question this step answers: does the optimal h1
still sit at ~3.5 sigma-hat as on the Gaussian, or does multimodality pull it down?
The answer calibrates the truth-free heuristic h1 = c * sigma-hat.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, parzen
from study2_common import (BUDGETS, SEEDS, ecdf_ks_mean, grid_for, ks_floor, plateau,
                           sigma_hat, stress_h1)

CASES = [
    ("symmetric bimodal", data.symmetric_bimodal()),
    ("asymmetric bimodal", data.asymmetric_bimodal()),
    ("asymmetric trimodal", data.asymmetric_trimodal()),
]
H1_GRID = np.geomspace(0.05, 30, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
summary = []
for col, (name, mix) in enumerate(CASES):
    res = stress_h1(mix, H1_GRID)
    sig = np.mean([sigma_hat(mix.sample(1000, np.random.default_rng(s))) for s in SEEDS])
    print("=" * 76)
    print(f"{name}   (sigma-hat = {sig:.3f})")
    print(f"{'n':>6} {'h1*':>7} {'KS*':>8} {'KS @h1=1':>9} {'ECDF':>8} {'floor':>7} "
          f"{'plateau 10%':>16} {'h1*/sigma':>10}")
    for n in BUDGETS:
        i = int(np.argmin(res[n]["ks"]))
        lo, hi = plateau(H1_GRID, res[n]["ks"])
        at1 = res[n]["ks"][int(np.argmin(np.abs(H1_GRID - 1.0)))]
        print(f"{n:>6} {H1_GRID[i]:>7.2f} {res[n]['ks'][i]:>8.4f} {at1:>9.4f} "
              f"{ecdf_ks_mean(mix, n):>8.4f} {ks_floor(n):>7.4f} "
              f"[{lo:>5.2f}, {hi:>6.2f}] {H1_GRID[i] / sig:>10.2f}")
        summary.append((name, n, H1_GRID[i] / sig))

    ax = axes[0][col]
    for n in BUDGETS:
        line, = ax.semilogx(H1_GRID, res[n]["ks"], label=f"n = {n}")
        ax.axhline(ks_floor(n), color=line.get_color(), ls=":", lw=0.8)
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.axvline(sig, color="gray", ls="-.", lw=0.8)
    ax.set_title(f"{name} · KS vs h1")
    ax.set_xlabel("h1 (gray: h1=1 dashed, h1=sigma dash-dot)")
    if col == 0:
        ax.set_ylabel("mean KS (10 seeds)")
        ax.legend(fontsize=8)

    ax = axes[1][col]
    n_show = 500
    smp = mix.sample(n_show, np.random.default_rng(0))
    g = grid_for(mix)
    ax.plot(g, mix.pdf(g), "k--", lw=1.4, label="truth")
    i = int(np.argmin(res[n_show]["ks"]))
    for h1, lab in ((1.0, "h1 = 1"), (float(H1_GRID[i]), f"h1* = {H1_GRID[i]:.2f}")):
        ax.plot(g, parzen.parzen_pdf(g, smp, h1 / np.sqrt(n_show)), lw=1.3, label=lab)
    ax.set_title(f"pdf at n = {n_show}")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("results/parzen2_mixtures_stress.png", dpi=120)
print()
print("h1*/sigma-hat across cases:", ", ".join(f"{v:.2f}" for _, _, v in summary))
print("figure -> results/parzen2_mixtures_stress.png")
