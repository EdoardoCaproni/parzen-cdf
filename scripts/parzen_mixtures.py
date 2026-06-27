"""Phase A / Gaussian mixtures -- Parzen Window estimation, carrying two sample budgets.

State of the art from the single-Gaussian study: a data-driven Silverman window size. Mixtures are
harder -- Silverman tends to over-smooth the valley between modes -- so we also try an *adaptive*
(per-point) window size with a Silverman pilot (narrower where data is dense, wider in the tails).

Per the owner's request we carry TWO operating points throughout the rest of the study:
  - best overall : abundant data, n = 20000
  - best under 2k: a scarce-data budget, n = 1000 (n = 500 is reported too)

Terminology: Parzen Window; smoothing parameter = window size; the CDF estimate is the integral of
the estimated pdf (closed form). Score = the CDF gap (Kolmogorov-Smirnov distance) vs the known truth.

    python scripts/parzen_mixtures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, metrics, parzen

RESULTS_DIR = "results"
SEED = 0
N_UNDER2K = 1000
N_OVERALL = 20000
NS = [100, 200, 500, 1000, 2000, 5000, 20000]
SWEEP_SEEDS = range(3)

MIXES = {
    "symmetric_bimodal": (data.symmetric_bimodal(), "0.5 N(-2,0.7) + 0.5 N(2,0.7)"),
    "asymmetric_bimodal": (data.asymmetric_bimodal(), "0.65 N(0,1) + 0.35 N(3,0.6)"),
}


def grid_for(mix, n_points=2000):
    lo = float((mix.means - 5 * mix.stds).min())
    hi = float((mix.means + 5 * mix.stds).max())
    return np.linspace(lo, hi, n_points)


def window(samples, strategy):
    """Return the window size for a strategy: a scalar (Silverman) or per-point array (adaptive)."""
    h = parzen.silverman_bandwidth(samples)
    return h if strategy == "silverman" else parzen.adaptive_bandwidths(samples, pilot_h=h)


def ks_for(mix, samples, grid, true_cdf, strategy):
    return metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, window(samples, strategy)))


def window_strategies_fig(name, mix, formula):
    """Silverman vs adaptive window size at n=2000, pdf and CDF vs truth."""
    grid = grid_for(mix)
    true_pdf, true_cdf = mix.pdf(grid), mix.cdf(grid)
    samples = mix.sample(2000, np.random.default_rng(SEED))
    h = parzen.silverman_bandwidth(samples)
    wa = parzen.adaptive_bandwidths(samples, pilot_h=h)
    cdf_s = parzen.parzen_cdf(grid, samples, h); pdf_s = parzen.parzen_pdf(grid, samples, h)
    cdf_a = parzen.parzen_cdf(grid, samples, wa); pdf_a = parzen.parzen_pdf(grid, samples, wa)
    ks_s = metrics.ks_distance(true_cdf, cdf_s); ks_a = metrics.ks_distance(true_cdf, cdf_a)
    print(f"  [{name}] n=2000: Silverman KS={ks_s:.4f} (window {h:.3f}), "
          f"adaptive KS={ks_a:.4f} (window mean {wa.mean():.3f})")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(grid, true_pdf, "k-", lw=2, label="true")
    a.plot(grid, pdf_s, "--", color="tab:red", label=f"Silverman (KS {ks_s:.3f})")
    a.plot(grid, pdf_a, "-", color="tab:blue", label=f"adaptive (KS {ks_a:.3f})")
    a.set_title("estimated pdf"); a.legend(fontsize=8)
    b.plot(grid, true_cdf, "k-", lw=2, label="true")
    b.plot(grid, cdf_s, "--", color="tab:red", label="Silverman")
    b.plot(grid, cdf_a, "-", color="tab:blue", label="adaptive")
    b.set_title("estimated CDF (integral of the pdf)"); b.legend(fontsize=8)
    fig.suptitle(f"{name}: {formula} -- Silverman vs adaptive window size (n=2000)")
    fig.tight_layout(); _save(fig, f"parzen_{name}_window_strategies.png")
    return ks_s, ks_a


def sample_size_fig(name, mix, formula):
    """CDF gap vs n for both strategies, mean over seeds; the two carried budgets are marked."""
    grid = grid_for(mix, 1000)
    true_cdf = mix.cdf(grid)
    curves = {"silverman": [], "adaptive": []}
    rows = {}
    for n in NS:
        per = {"silverman": [], "adaptive": []}
        for s in SWEEP_SEEDS:
            samples = mix.sample(n, np.random.default_rng(s))
            for strat in per:
                per[strat].append(ks_for(mix, samples, grid, true_cdf, strat))
        for strat in per:
            curves[strat].append(np.mean(per[strat]))
        rows[n] = {strat: np.mean(per[strat]) for strat in per}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(NS, curves["silverman"], "o--", color="tab:red", label="Silverman")
    ax.plot(NS, curves["adaptive"], "o-", color="tab:blue", label="adaptive")
    for nb, lbl in [(N_UNDER2K, "under-2k (1000)"), (N_OVERALL, "best overall (20000)")]:
        ax.axvline(nb, ls=":", color="gray"); ax.text(nb, ax.get_ylim()[1], lbl, rotation=90,
                                                       va="top", ha="right", fontsize=7, color="gray")
    ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("CDF gap (KS)")
    ax.set_title(f"{name}: CDF gap vs samples (mean over {len(list(SWEEP_SEEDS))} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, f"parzen_{name}_sample_size.png")
    return rows


def progression_fig(name, mix, formula, strategy="adaptive"):
    """The chosen strategy's estimate (pdf top, CDF bottom) at growing n."""
    ns = [100, 500, 1000, 2000, 20000]
    grid = grid_for(mix, 1500)
    true_pdf, true_cdf = mix.pdf(grid), mix.cdf(grid)
    fig, axes = plt.subplots(2, len(ns), figsize=(17, 6), sharex=True)
    for j, n in enumerate(ns):
        samples = mix.sample(n, np.random.default_rng(SEED))
        w = window(samples, strategy)
        ec = parzen.parzen_cdf(grid, samples, w); ep = parzen.parzen_pdf(grid, samples, w)
        ks = metrics.ks_distance(true_cdf, ec)
        axes[0, j].plot(grid, true_pdf, "k-", lw=1.3); axes[0, j].plot(grid, ep, "-", color="tab:blue", lw=1.3)
        axes[0, j].set_title(f"n={n}\nKS {ks:.4f}", fontsize=9)
        axes[1, j].plot(grid, true_cdf, "k-", lw=1.3); axes[1, j].plot(grid, ec, "-", color="tab:blue", lw=1.3)
    axes[0, 0].set_ylabel("pdf"); axes[1, 0].set_ylabel("CDF")
    axes[0, -1].legend(["true", f"Parzen ({strategy})"], fontsize=7)
    fig.suptitle(f"{name}: {formula} -- {strategy} window size, estimate vs samples "
                 "(black = truth, blue = estimate)")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); _save(fig, f"parzen_{name}_progression.png")


def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, (mix, formula) in MIXES.items():
        print(f"\n##### {name}: {formula}")
        window_strategies_fig(name, mix, formula)
        rows = sample_size_fig(name, mix, formula)
        progression_fig(name, mix, formula, strategy="adaptive")
        print(f"  carried operating points (best strategy = adaptive):")
        for nb, tag in [(N_UNDER2K, "under-2k"), (N_OVERALL, "overall")]:
            print(f"    n={nb:<6} ({tag:8}): Silverman KS={rows[nb]['silverman']:.4f}  "
                  f"adaptive KS={rows[nb]['adaptive']:.4f}")
        print(f"    [n=500 ref          : Silverman KS={rows[500]['silverman']:.4f}  "
              f"adaptive KS={rows[500]['adaptive']:.4f}]")


if __name__ == "__main__":
    main()
