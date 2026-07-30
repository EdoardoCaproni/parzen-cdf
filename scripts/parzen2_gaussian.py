"""Study 2, step 1: the single Gaussian under the h_n = h1/sqrt(n) framework.

The didactic ladder, corrected:
  (a) start from the trivial h1 = 1.0 (the study's new "first symbolic value");
  (b) STRESS h1 across three orders of magnitude at every budget (n = 500, 1000, 2000);
  (c) read the optimum in units of the sample scale sigma-hat, and note where the pdf
      optimum sits relative to the CDF optimum.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data
from study2_common import (BUDGETS, SEEDS, ecdf_ks_mean, eval_window, grid_for, ks_floor,
                           plateau, sigma_hat, stress_h1)

MIX = data.single_gaussian()
H1_GRID = np.geomspace(0.05, 30, 100)

grid = grid_for(MIX)
t_cdf, t_pdf = MIX.cdf(grid), MIX.pdf(grid)

# ------------------------------------------------------------------ (a) the trivial start
print("=" * 76)
print("(a) h1 = 1.0, the trivial start   [single Gaussian N(0,1)]")
print(f"{'n':>6} {'h = 1/sqrt(n)':>14} {'KS mean':>9} {'ECDF KS':>9} {'floor':>7}")
for n in BUDGETS:
    h = 1.0 / np.sqrt(n)
    vals = [eval_window(MIX, grid, t_cdf, t_pdf, MIX.sample(n, np.random.default_rng(s)), h)["ks"]
            for s in SEEDS]
    print(f"{n:>6} {h:>14.4f} {np.mean(vals):>9.4f} {ecdf_ks_mean(MIX, n):>9.4f} {ks_floor(n):>7.4f}")

# ------------------------------------------------------------------ (b) the h1 stress test
print()
print("(b) h1 stress test (mean KS over 10 seeds), optimum and 10% plateau")
res = stress_h1(MIX, H1_GRID)
sig = np.mean([sigma_hat(MIX.sample(n, np.random.default_rng(s))) for n in BUDGETS for s in SEEDS])
print(f"    mean sigma-hat = {sig:.3f}")
print(f"{'n':>6} {'h1* (KS)':>9} {'KS*':>8} {'plateau (10%)':>18} {'h1*/sigma':>10} {'h1* (pdf ISE)':>14}")
for n in BUDGETS:
    i = int(np.argmin(res[n]["ks"]))
    j = int(np.argmin(res[n]["ise"]))
    lo, hi = plateau(H1_GRID, res[n]["ks"])
    print(f"{n:>6} {H1_GRID[i]:>9.2f} {res[n]['ks'][i]:>8.4f} "
          f"[{lo:>6.2f}, {hi:>6.2f}] {H1_GRID[i] / sig:>10.2f} {H1_GRID[j]:>14.2f}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for n in BUDGETS:
    line, = axes[0].semilogx(H1_GRID, res[n]["ks"], label=f"n = {n}")
    axes[0].axhline(ks_floor(n), color=line.get_color(), ls=":", lw=0.8)
axes[0].axvline(1.0, color="gray", ls="--", lw=0.8, label="h1 = 1 (start)")
axes[0].set_xlabel("h1  (window size h = h1/√n)")
axes[0].set_ylabel("mean KS vs truth (10 seeds)")
axes[0].set_title("N(0,1) · CDF gap vs h1 (dotted: ECDF floor per n)")
axes[0].legend(fontsize=8)

n_show, seed_show = 500, 0
smp = MIX.sample(n_show, np.random.default_rng(seed_show))
from parzen_cdf import parzen
axes[1].plot(grid, t_pdf, "k--", lw=1.5, label="truth")
for h1 in (1.0, float(H1_GRID[int(np.argmin(res[n_show]['ks']))])):
    axes[1].plot(grid, parzen.parzen_pdf(grid, smp, h1 / np.sqrt(n_show)),
                 lw=1.4, label=f"h1 = {h1:.2f}")
axes[1].set_title(f"pdf at the baseline budget n = {n_show}")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/parzen2_gaussian_stress.png", dpi=120)
print("\nfigure -> results/parzen2_gaussian_stress.png")
