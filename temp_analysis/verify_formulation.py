"""Verification of the Parzen-window formulation and of the mixture sampling, prompted by the
Professor's remark (2026-07-03): "with a well-chosen FIXED window, a trimodal at n=1000 should
already give excellent results; you should not need 20000 samples".

Four checks:
  A. formulation sanity: h -> 0 recovers the empirical CDF; pdf integrates to 1; the
     gaussian-kernel estimate matches scipy.stats.gaussian_kde pointwise.
  B. sampling correctness: the mixture sampler is exact ancestral sampling (categorical
     component choice + numpy's ziggurat normal, NOT inverse-CDF); KS-test p-values across
     seeds must look uniform.
  C. the Professor's experiment: asymmetric trimodal, n=1000, sweep of FIXED window sizes,
     logistic vs gaussian kernel, against the empirical CDF and the statistical floor
     E[KS] ~ 0.8687 / sqrt(n).
  D. the suspected scale bug: Silverman's constant is derived for a unit-variance (gaussian)
     kernel; the logistic kernel has std pi/sqrt(3) ~ 1.814, so Silverman-h with a logistic
     window over-smooths by ~1.8x unless variance-matched.
"""

import numpy as np
from scipy.stats import gaussian_kde, kstest

from parzen_cdf import data, parzen

TRI = data.asymmetric_trimodal()
GRID = np.linspace(-5, 7, 4001)
T_CDF = TRI.cdf(GRID)
T_PDF = TRI.pdf(GRID)


def ecdf_on_grid(samples, grid):
    return np.searchsorted(np.sort(samples), grid, side="right") / samples.size


def ks(a, b):
    return float(np.max(np.abs(a - b)))


print("=" * 78)
print("A. FORMULATION")
rng = np.random.default_rng(0)
s = TRI.sample(1000, rng)

# A1: h -> 0 recovers the empirical CDF (each window collapses to a step at x_i)
tiny = parzen.parzen_cdf(GRID, s, 1e-9)
print(f"A1  max|Parzen(h→0) − ECDF| = {ks(tiny, ecdf_on_grid(s, GRID)):.2e}   (expect ~0)")

# A2: the pdf integrates to 1 and is the derivative of the CDF
for k in ("logistic", "gaussian", "box", "epanechnikov", "triangular"):
    p = parzen.parzen_pdf(GRID, s, 0.25, kernel=k)
    c = parzen.parzen_cdf(GRID, s, 0.25, kernel=k)
    fd = np.gradient(c, GRID)
    print(f"A2  {k:12s} mass = {np.trapz(p, GRID):.6f}   max|pdf − dCDF/dx| = {np.max(np.abs(p - fd)):.2e}")

# A3: gaussian kernel vs scipy.stats.gaussian_kde at the same bandwidth
h = 0.3
ours = parzen.parzen_pdf(GRID, s, h, kernel="gaussian")
kde = gaussian_kde(s, bw_method=h / s.std(ddof=1))
print(f"A3  max|ours − scipy.gaussian_kde| = {np.max(np.abs(ours - kde(GRID))):.2e}   (expect ~1e-15)")

print()
print("=" * 78)
print("B. SAMPLING (ancestral: categorical component + numpy ziggurat normal; no inverse CDF)")
pvals = [kstest(TRI.sample(1000, np.random.default_rng(seed)), TRI.cdf).pvalue for seed in range(40)]
pvals = np.array(pvals)
print(f"B1  KS-test vs true mixture CDF over 40 seeds: p<0.05 in {int((pvals < 0.05).sum())}/40 "
      f"(expect ~2), median p = {np.median(pvals):.2f}")

print()
print("=" * 78)
print("C. THE PROFESSOR'S EXPERIMENT: trimodal, n = 1000, FIXED window size")
n = 1000
seeds = range(10)
h_grid = np.geomspace(0.02, 1.5, 50)
floor = 0.8687 / np.sqrt(n)

results = {}
for kernel in ("logistic", "gaussian"):
    curves = []
    for seed in seeds:
        smp = TRI.sample(n, np.random.default_rng(seed))
        curves.append([ks(parzen.parzen_cdf(GRID, smp, h, kernel=kernel), T_CDF) for h in h_grid])
    mean_ks = np.mean(curves, axis=0)
    best = int(np.argmin(mean_ks))
    results[kernel] = (h_grid[best], mean_ks[best], mean_ks)
    print(f"C1  {kernel:9s} best FIXED h = {h_grid[best]:.3f}  ->  mean KS = {mean_ks[best]:.4f}")

emp = np.mean([ks(ecdf_on_grid(TRI.sample(n, np.random.default_rng(sd)), GRID), T_CDF) for sd in seeds])
print(f"C2  empirical CDF (no smoothing)        mean KS = {emp:.4f}")
print(f"C3  statistical floor E[KS] = 0.8687/sqrt(n)  = {floor:.4f}")
print("     -> no estimator whatsoever can be reliably better than this at n = 1000")

print()
print("=" * 78)
print("D. THE SCALE MISMATCH (Silverman constant vs logistic kernel std)")
sil_ks, vm_ks, sil_h, vm_h = [], [], [], []
for seed in seeds:
    smp = TRI.sample(n, np.random.default_rng(seed))
    hs = parzen.silverman_bandwidth(smp)
    hv = parzen.variance_matched_bandwidth(smp)
    sil_h.append(hs); vm_h.append(hv)
    sil_ks.append(ks(parzen.parzen_cdf(GRID, smp, hs), T_CDF))
    vm_ks.append(ks(parzen.parzen_cdf(GRID, smp, hv), T_CDF))
print(f"D1  logistic kernel std = {parzen.LOGISTIC_KERNEL_STD:.4f} (Silverman's constant assumes 1.0)")
print(f"D2  Silverman h = {np.mean(sil_h):.3f} -> effective gaussian-equivalent smoothing "
      f"{np.mean(sil_h) * parzen.LOGISTIC_KERNEL_STD:.3f}")
print(f"D3  trimodal n=1000, logistic + Silverman h:          mean KS = {np.mean(sil_ks):.4f}")
print(f"D4  trimodal n=1000, logistic + variance-matched h:    mean KS = {np.mean(vm_ks):.4f}")
print(f"D5  trimodal n=1000, logistic + best fixed h:          mean KS = {results['logistic'][1]:.4f}")
print(f"D6  study's 'fixed trivial window' h = 1.0:            mean KS = "
      f"{np.mean([ks(parzen.parzen_cdf(GRID, TRI.sample(n, np.random.default_rng(sd)), 1.0), T_CDF) for sd in seeds]):.4f}")

# the same sweep at n = 20000 to quantify what the extra samples actually buy
n2 = 20000
smp = TRI.sample(n2, np.random.default_rng(0))
best20 = min(ks(parzen.parzen_cdf(GRID, smp, h), T_CDF) for h in h_grid)
print(f"D7  n = 20000, best fixed h:                           KS = {best20:.4f} "
      f"(floor {0.8687 / np.sqrt(n2):.4f})")

# ------------------------------------------------------------------ figure for the Professor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

smp = TRI.sample(n, np.random.default_rng(0))
h_best = results["logistic"][0]
h_sil = parzen.silverman_bandwidth(smp)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(GRID, T_PDF, "k--", lw=1.5, label="truth")
axes[0].plot(GRID, parzen.parzen_pdf(GRID, smp, h_sil), lw=1.5,
             label=f"logistic + Silverman h={h_sil:.2f} (mis-scaled)")
axes[0].plot(GRID, parzen.parzen_pdf(GRID, smp, h_best), lw=1.5,
             label=f"logistic + fixed h={h_best:.2f}")
axes[0].set_title(f"pdf · asymmetric trimodal · n = {n}")
axes[0].legend(fontsize=8)
axes[1].semilogx(h_grid, results["logistic"][2], label="logistic kernel")
axes[1].semilogx(h_grid, results["gaussian"][2], label="gaussian kernel")
axes[1].axhline(emp, color="gray", ls=":", label=f"empirical CDF ({emp:.3f})")
axes[1].axhline(floor, color="k", ls=":", label=f"floor 0.87/√n ({floor:.3f})")
axes[1].axvline(np.mean(sil_h), color="C0", ls="--", alpha=0.5, label="Silverman h")
axes[1].set_xlabel("fixed window size h"); axes[1].set_ylabel("mean KS vs truth (10 seeds)")
axes[1].set_title("KS vs fixed h · the two kernels differ only by scale")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("temp_analysis/verify_formulation.png", dpi=120)
print("\nfigure -> temp_analysis/verify_formulation.png")
