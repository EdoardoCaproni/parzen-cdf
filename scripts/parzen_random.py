"""Phase A / step 3 -- a battery of ~10 random complex Gaussian mixtures.

We generate 10 random mixtures (3 to 6 modes, varied means and widths) and test every truth-free
window-size selector on all of them, at the two carried budgets:
  - best under 2k: n = 1000  -- the full selector set (Silverman, variance-matched, adaptive,
    likelihood-CV, least-squares-CV);
  - best overall : n = 20000 -- only the practical selectors, because cross-validation is
    O(n^2 * candidates) and impractical at this sample count.

The point is to see which truth-free window size is most solid across varied shapes, not just on the
two hand-picked bimodals. Score = CDF gap (Kolmogorov-Smirnov) vs the known truth; one sample set per
(mixture, budget), seeded by the mixture index. Terminology: Parzen Window, window size.

    python scripts/parzen_random.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, metrics, parzen

RESULTS_DIR = "results"
N_MIXTURES = 10
N_UNDER2K = 1000
N_OVERALL = 20000
ALL_SELECTORS = ["silverman", "variance_matched", "adaptive", "likelihood_cv", "lscv"]
CHEAP_SELECTORS = ["silverman", "variance_matched", "adaptive"]


def make_mixtures():
    rng = np.random.default_rng(0)
    return [data.random_mixture(rng) for _ in range(N_MIXTURES)]


def grid_for(mix):
    lo = float((mix.means - 5 * mix.stds).min())
    hi = float((mix.means + 5 * mix.stds).max())
    return np.linspace(lo, hi, 1000)


def windows(samples, names):
    h = parzen.silverman_bandwidth(samples)
    out = {}
    for nm in names:
        if nm == "silverman":
            out[nm] = h
        elif nm == "variance_matched":
            out[nm] = parzen.variance_matched_bandwidth(samples)
        elif nm == "adaptive":
            out[nm] = parzen.adaptive_bandwidths(samples, pilot_h=h)
        elif nm == "likelihood_cv":
            out[nm] = parzen.likelihood_cv_bandwidth(samples)
        elif nm == "lscv":
            out[nm] = parzen.lscv_bandwidth(samples)
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mixes = make_mixtures()
    budgets = {"under_2k": (N_UNDER2K, ALL_SELECTORS), "overall": (N_OVERALL, CHEAP_SELECTORS)}
    ks = {b: {s: [] for s in sel} for b, (n, sel) in budgets.items()}
    gallery = []  # (grid, true_pdf, adaptive_pdf_overall) per mixture

    for i, mix in enumerate(mixes):
        grid = grid_for(mix)
        true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)
        print(f"\nmixture {i}: k={mix.n_components}, "
              f"means={np.round(mix.means, 2)}, stds={np.round(mix.stds, 2)}")
        for b, (n, sel) in budgets.items():
            samples = mix.sample(n, np.random.default_rng(i))
            ws = windows(samples, sel)
            line = []
            for s in sel:
                k = metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, ws[s]))
                ks[b][s].append(k)
                line.append(f"{s}={k:.4f}")
                if b == "overall" and s == "adaptive":
                    gallery.append((grid, true_pdf, parzen.parzen_pdf(grid, samples, ws[s])))
            print(f"  [{b:8} n={n:5}] " + "  ".join(line))

    print("\n===== mean CDF gap across the 10 mixtures =====")
    for b, (n, sel) in budgets.items():
        print(f"  {b} (n={n}):")
        for s in sel:
            arr = np.array(ks[b][s])
            print(f"    {s:<18} mean {arr.mean():.4f}   (worst {arr.max():.4f})")

    _bar_fig(ks, budgets)
    _gallery_fig(gallery)


def _bar_fig(ks, budgets):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (b, (n, sel)) in zip(axes, budgets.items()):
        means = [np.mean(ks[b][s]) for s in sel]
        stds = [np.std(ks[b][s]) for s in sel]
        ax.bar(range(len(sel)), means, yerr=stds, capsize=3, color="tab:blue")
        ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{b} (n={n})"); ax.set_ylabel("mean CDF gap (KS) over 10 mixtures")
    fig.suptitle("Window-size selectors on 10 random mixtures (mean +/- std across mixtures)")
    fig.tight_layout(); _save(fig, "parzen_random_selectors.png")


def _gallery_fig(gallery):
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for ax, (grid, true_pdf, est_pdf) in zip(axes.flat, gallery):
        ax.plot(grid, true_pdf, "k-", lw=1.4)
        ax.plot(grid, est_pdf, "-", color="tab:blue", lw=1.3)
        ax.set_yticks([])
    axes.flat[0].legend(["true", "adaptive"], fontsize=7)
    fig.suptitle("The 10 random mixtures: true density (black) vs adaptive Parzen estimate "
                 "(blue), n=20000")
    fig.tight_layout(); _save(fig, "parzen_random_gallery.png")


def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
