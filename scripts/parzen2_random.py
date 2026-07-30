"""Study 2, step 3: the battery of 10 random complex mixtures at n = 500/1000/2000.

Selectors compared (all truth-free except the oracle):
  start        h1 = 1.0                 (the trivial schedule start)
  sigma-rule   h1 = 1.5 * sigma-hat     (the heuristic calibrated in steps 1-2, frozen there)
  lscv         least-squares cross-validation, candidate windows on a sigma-hat-based
               geometric grid (0.01 to 0.5 sigma-hat, 40 values)
  oracle-h     the fixed window minimizing the measured gap to the reference CDF, per sample
               set (an upper bound no data-driven method can use)
plus the empirical CDF and the statistical floor as references.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, parzen
from study2_common import BUDGETS, ecdf_on_grid, grid_for, ks, ks_floor, sigma_hat

N_MIXTURES = 10
SEEDS_PER = 3
C_STAR = 1.5                       # frozen in steps 1-2
H_ORACLE = np.geomspace(0.005, 1.5, 60)


def make_mixtures():
    rng = np.random.default_rng(0)
    return [data.random_mixture(rng) for _ in range(N_MIXTURES)]


def selector_h(name, samples, n):
    sig = sigma_hat(samples)
    if name == "start":
        return 1.0 / np.sqrt(n)
    if name == "sigma-rule":
        return C_STAR * sig / np.sqrt(n)
    if name == "lscv":
        return parzen.lscv_bandwidth(samples, candidates=sig * np.geomspace(0.01, 0.5, 40))
    raise ValueError(name)


SELECTORS = ["start", "sigma-rule", "lscv"]

mixtures = make_mixtures()
results = {sel: {n: [] for n in BUDGETS} for sel in SELECTORS + ["oracle-h", "ecdf"]}

for mi, mix in enumerate(mixtures):
    grid = grid_for(mix)
    t_cdf = mix.cdf(grid)
    for n in BUDGETS:
        for s in range(SEEDS_PER):
            samples = mix.sample(n, np.random.default_rng(100 * mi + s))
            for sel in SELECTORS:
                h = selector_h(sel, samples, n)
                results[sel][n].append(ks(parzen.parzen_cdf(grid, samples, h), t_cdf))
            results["oracle-h"][n].append(
                min(ks(parzen.parzen_cdf(grid, samples, h), t_cdf) for h in H_ORACLE))
            results["ecdf"][n].append(ks(ecdf_on_grid(samples, grid), t_cdf))
    print(f"mixture {mi + 1}/10 done")

print()
print("=" * 76)
print("mean KS over the 10 mixtures x 3 seeds (lower is better)")
header = f"{'selector':>12} | " + " ".join(f"{f'n={n}':>8}" for n in BUDGETS)
print(header)
print("-" * len(header))
for sel in ["start", "sigma-rule", "lscv", "oracle-h", "ecdf"]:
    row = " ".join(f"{np.mean(results[sel][n]):>8.4f}" for n in BUDGETS)
    print(f"{sel:>12} | {row}")
print(f"{'floor':>12} | " + " ".join(f"{ks_floor(n):>8.4f}" for n in BUDGETS))

print()
print("worst mixture-mean KS (robustness):")
for sel in SELECTORS:
    worst = []
    for n in BUDGETS:
        per_mix = [np.mean(results[sel][n][SEEDS_PER * mi:SEEDS_PER * (mi + 1)])
                   for mi in range(N_MIXTURES)]
        worst.append(max(per_mix))
    print(f"{sel:>12} | " + " ".join(f"{w:>8.4f}" for w in worst))

# ------------------------------------------------------------------ selector figure
fig, ax = plt.subplots(figsize=(7.2, 4.4))
styles = {"start": "o-", "sigma-rule": "s-", "lscv": "^-", "oracle-h": "d:", "ecdf": "*:"}
for sel, st in styles.items():
    ax.plot(BUDGETS, [np.mean(results[sel][n]) for n in BUDGETS], st, label=sel, ms=5)
ax.plot(BUDGETS, [ks_floor(n) for n in BUDGETS], "k:", lw=1, label="floor 0.87/√n")
ax.set_xscale("log")
ax.set_xticks(BUDGETS)
ax.set_xticklabels([str(n) for n in BUDGETS])
ax.minorticks_off()
ax.set_xlabel("samples n")
ax.set_ylabel("mean KS vs reference CDF (10 mixtures × 3 seeds)")
ax.set_title("Battery of 10 random mixtures · window-size selectors, h = h1/√n")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("results/parzen2_random_selectors.png", dpi=120)

# ------------------------------------------------------------------ gallery figure
fig, axes = plt.subplots(2, 5, figsize=(16, 5.6))
n_show = 500
for mi, (mix, ax) in enumerate(zip(mixtures, axes.ravel())):
    grid = grid_for(mix)
    samples = mix.sample(n_show, np.random.default_rng(100 * mi))
    h = C_STAR * sigma_hat(samples) / np.sqrt(n_show)
    ax.plot(grid, mix.pdf(grid), "k--", lw=1.2)
    ax.plot(grid, parzen.parzen_pdf(grid, samples, h), lw=1.3)
    ax.set_title(f"mixture {mi + 1}", fontsize=9)
    ax.tick_params(labelsize=7)
axes[0][0].legend(["reference density", "Parzen, σ-rule"], fontsize=7)
fig.suptitle(f"The 10 random mixtures at n = {n_show}: Parzen with the σ-rule window", y=1.0)
fig.tight_layout()
fig.savefig("results/parzen2_random_gallery.png", dpi=120)
print("\nfigures -> results/parzen2_random_selectors.png, results/parzen2_random_gallery.png")
