"""Phase A / single Gaussian -- estimate the CDF with a Parzen Window, built up one step at a time.

Terminology: the estimator is a *Parzen Window*; its smoothing parameter is the *window size*. The
CDF estimate is the integral of the estimated pdf -- in closed form, a mean of logistic sigmoids. We
judge by the CDF gap (Kolmogorov-Smirnov distance) versus the known truth N(0,1).

Produces three clearly-named figures in results/:
  parzen_gaussian_window_strategies.png  -- fixed vs adaptive vs Silverman window size (n=2000)
  parzen_gaussian_silverman_check.png    -- visual check of the chosen (Silverman) estimate
  parzen_gaussian_sample_size.png        -- CDF gap and pdf-peak vs the number of samples

    python scripts/parzen_gaussian.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, metrics, parzen

SEED = 0
N = 2000
RESULTS_DIR = "results"

mix = data.single_gaussian()                       # known CDF: a single Gaussian N(0,1)
GRID = np.linspace(-6, 6, 2000)
TRUE_PDF, TRUE_CDF = mix.pdf(GRID), mix.cdf(GRID)


def estimate(samples, window_size, grid=GRID, true_cdf=TRUE_CDF):
    """Parzen Window estimate. window_size is a scalar (fixed) or per-point array (adaptive)."""
    est_pdf = parzen.parzen_pdf(grid, samples, window_size)
    est_cdf = parzen.parzen_cdf(grid, samples, window_size)   # == integral of est_pdf (closed form)
    return est_pdf, est_cdf, metrics.ks_distance(true_cdf, est_cdf)


def window_strategies():
    """Step 1->3: fixed trivial -> adaptive (per-point) -> Silverman window size, at n=2000."""
    samples = mix.sample(N, np.random.default_rng(SEED))
    pdf1, cdf1, ks1 = estimate(samples, 1.0)                                  # fixed, trivial
    win_adapt = parzen.adaptive_bandwidths(samples, pilot_h=1.0)              # adaptive, pilot = 1.0
    pdf2, cdf2, ks2 = estimate(samples, win_adapt)
    win_silv = parzen.silverman_bandwidth(samples)                           # data-driven global
    pdf3, cdf3, ks3 = estimate(samples, win_silv)

    print(f"Phase A -- single Gaussian N(0,1), n={N}")
    print(f"  fixed window size = 1.0             : CDF gap (KS) = {ks1:.4f}")
    print(f"  adaptive window size (pilot 1.0)    : CDF gap (KS) = {ks2:.4f}"
          f"   [per-point: mean {win_adapt.mean():.3f}, range {win_adapt.min():.3f}-{win_adapt.max():.3f}]")
    print(f"  Silverman window size = {win_silv:.3f}      : CDF gap (KS) = {ks3:.4f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(GRID, TRUE_PDF, "k-", lw=2, label="true")
    a.plot(GRID, pdf1, "--", color="tab:red", label=f"fixed 1.0 (KS {ks1:.3f})")
    a.plot(GRID, pdf2, ":", color="tab:orange", label=f"adaptive (KS {ks2:.3f})")
    a.plot(GRID, pdf3, "-", color="tab:green", label=f"Silverman (KS {ks3:.3f})")
    a.set_title("estimated pdf"); a.legend(fontsize=8)
    b.plot(GRID, TRUE_CDF, "k-", lw=2, label="true")
    b.plot(GRID, cdf1, "--", color="tab:red", label="fixed 1.0")
    b.plot(GRID, cdf2, ":", color="tab:orange", label="adaptive")
    b.plot(GRID, cdf3, "-", color="tab:green", label="Silverman")
    b.set_title("estimated CDF (integral of the pdf)"); b.legend(fontsize=8)
    fig.suptitle(f"Parzen Window window-size strategies -- single Gaussian (n={N})")
    fig.tight_layout()
    _save(fig, "parzen_gaussian_window_strategies.png")
    return win_silv


def silverman_check(win_silv):
    """Visual check of the chosen estimate (Silverman window) at n=2000: pdf, CDF, and the CDF error."""
    samples = mix.sample(N, np.random.default_rng(SEED))
    pdf, cdf, ks = estimate(samples, win_silv)
    err = np.abs(cdf - TRUE_CDF); ksx = GRID[err.argmax()]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(GRID, TRUE_PDF, "k-", lw=2, label="true pdf")
    ax[0].plot(GRID, pdf, "-", color="tab:green", lw=1.8, label="Parzen (Silverman)")
    ax[0].hist(samples, bins=60, density=True, alpha=0.25, color="gray", label="samples")
    ax[0].set_title("pdf: estimate vs truth"); ax[0].legend(fontsize=8)
    ax[1].plot(GRID, TRUE_CDF, "k-", lw=2, label="true CDF")
    ax[1].plot(GRID, cdf, "-", color="tab:green", lw=1.8, label="Parzen (Silverman)")
    ax[1].set_title("CDF: estimate vs truth (overlapping)"); ax[1].legend(fontsize=8)
    ax[2].plot(GRID, err, "-", color="tab:purple"); ax[2].axvline(ksx, ls=":", color="red")
    ax[2].annotate(f"KS = {ks:.4f} @ x={ksx:.2f}", (ksx, ks), textcoords="offset points",
                   xytext=(8, -4), fontsize=9, color="red")
    ax[2].set_title("CDF error |estimate - truth|"); ax[2].set_ylim(bottom=0)
    fig.suptitle(f"Visual check -- single Gaussian, Silverman window size = {win_silv:.3f}, n={N} (KS = {ks:.4f})")
    fig.tight_layout()
    _save(fig, "parzen_gaussian_silverman_check.png")


def sample_size_sweep(seeds=range(5)):
    """CDF gap and pdf-peak vs sample count, for three window-size methods: the naive fixed window,
    Silverman, and adaptive (pilot Silverman). Key contrast: a fixed (too-large) window does NOT
    improve with more samples (it never shrinks), while the data-driven windows do. Mean over seeds.
    """
    ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    grid = np.linspace(-6, 6, 1000)                 # coarser grid keeps memory modest at large n
    true_cdf = mix.cdf(grid)
    true_peak = float(mix.pdf(np.array([0.0]))[0])   # ~0.3989
    methods = ["fixed 1.0", "Silverman", "adaptive"]
    colors = {"fixed 1.0": "tab:red", "Silverman": "tab:green", "adaptive": "tab:blue"}
    ks = {m: [] for m in methods}; ks_std = {m: [] for m in methods}; peak = {m: [] for m in methods}
    print(f"\n  sample-size sweep (CDF gap, mean over {len(list(seeds))} seeds):")
    print(f"  {'n':>7}" + "".join(f"{m:>12}" for m in methods))
    for n in ns:
        per = {m: [] for m in methods}; perp = {m: [] for m in methods}
        for s in seeds:
            samples = mix.sample(n, np.random.default_rng(s))
            h = parzen.silverman_bandwidth(samples)
            wins = {"fixed 1.0": 1.0, "Silverman": h,
                    "adaptive": parzen.adaptive_bandwidths(samples, pilot_h=h)}
            for m, w in wins.items():
                per[m].append(metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, w)))
                perp[m].append(float(parzen.parzen_pdf(np.array([0.0]), samples, w)[0]))
        for m in methods:
            ks[m].append(np.mean(per[m])); ks_std[m].append(np.std(per[m])); peak[m].append(np.mean(perp[m]))
        print(f"  {n:>7}" + "".join(f"{np.mean(per[m]):>12.4f}" for m in methods))

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
    for m in methods:
        a.errorbar(ns, ks[m], yerr=ks_std[m], marker="o", capsize=3, color=colors[m], label=m)
        b.plot(ns, peak[m], "o-", color=colors[m], label=m)
    a.set_xscale("log"); a.set_title("CDF gap (KS) vs number of samples")
    a.set_xlabel("n"); a.set_ylabel("CDF gap (KS)"); a.legend(fontsize=8)
    b.axhline(true_peak, ls="--", color="k", label=f"true peak {true_peak:.3f}")
    b.set_xscale("log"); b.set_title("pdf peak height vs number of samples")
    b.set_xlabel("n"); b.legend(fontsize=8)
    fig.suptitle("Single Gaussian: fixed vs data-driven window sizes as the sample count grows")
    fig.tight_layout()
    _save(fig, "parzen_gaussian_sample_size.png")


def sample_size_progression():
    """Progressive view: the Silverman estimate (pdf top row, CDF bottom row) at growing n."""
    ns = [50, 100, 200, 500, 2000, 20000]
    fig, axes = plt.subplots(2, len(ns), figsize=(19, 6), sharex=True)
    for j, n in enumerate(ns):
        samples = mix.sample(n, np.random.default_rng(SEED))
        h = parzen.silverman_bandwidth(samples)
        est_pdf, est_cdf, ks = estimate(samples, h)
        axes[0, j].plot(GRID, TRUE_PDF, "k-", lw=1.4)
        axes[0, j].plot(GRID, est_pdf, "-", color="tab:green", lw=1.4)
        axes[0, j].set_title(f"n={n}\nwindow {h:.3f} · KS {ks:.4f}", fontsize=9)
        axes[1, j].plot(GRID, TRUE_CDF, "k-", lw=1.4)
        axes[1, j].plot(GRID, est_cdf, "-", color="tab:green", lw=1.4)
    axes[0, 0].set_ylabel("pdf"); axes[1, 0].set_ylabel("CDF")
    axes[0, -1].legend(["true", "Parzen (Silverman)"], fontsize=7)
    fig.suptitle("Single Gaussian: the Parzen Window estimate sharpens with more samples "
                 "(black = truth, green = Silverman estimate)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "parzen_gaussian_sample_size_progression.png")


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    win_silv = window_strategies()
    silverman_check(win_silv)
    sample_size_sweep()
    sample_size_progression()


if __name__ == "__main__":
    main()
