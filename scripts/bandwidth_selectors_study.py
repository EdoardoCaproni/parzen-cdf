"""Comparative study of bandwidth-selection methods for the logistic Parzen estimator.

Every selector uses ONLY the samples (it is a method you could actually deploy). The true pdf is
used solely as the referee -- to score each result with KS (on the CDF) and MSE (on the pdf) -- and
once more for the ORACLE bandwidth (the KS-minimizing h), which is *not* a real selector but an
upper bound: the best a global bandwidth could possibly do if you knew the truth.

We run all selectors across the full distribution ladder and report which truth-free method gets
closest to the oracle.

    python scripts/bandwidth_selectors_study.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, metrics, parzen

SEED = 0
N_SAMPLES = 2000
RESULTS_DIR = "results"

DISTS = {
    "single_gaussian": data.single_gaussian(),
    "symmetric_bimodal": data.symmetric_bimodal(),
    "asymmetric_trimodal": data.asymmetric_trimodal(),
    "spike_in_broad": data.spike_in_broad(),
}

# Truth-free selectors return a scalar h or (adaptive) a per-sample array.
TRUTH_FREE = {
    "silverman": parzen.silverman_bandwidth,
    "var_matched": parzen.variance_matched_bandwidth,
    "likelihood_cv": parzen.likelihood_cv_bandwidth,
    "lscv": parzen.lscv_bandwidth,
    "adaptive": parzen.adaptive_bandwidths,
}
ORDER = ["silverman", "var_matched", "likelihood_cv", "lscv", "adaptive", "oracle"]


def oracle_bandwidth(samples, grid, true_cdf):
    """KS-minimizing global h (uses the truth -- an upper bound, not a deployable selector)."""
    h0 = parzen.silverman_bandwidth(samples)
    scales = np.geomspace(0.1, 2.0, 60)
    ks = [metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, s * h0)) for s in scales]
    return float(scales[int(np.argmin(ks))] * h0)


def score(samples, h, grid, true_cdf, true_pdf):
    cdf = parzen.parzen_cdf(grid, samples, h)
    pdf = parzen.parzen_pdf(grid, samples, h)
    return metrics.ks_distance(true_cdf, cdf), metrics.mse(true_pdf, pdf)


def h_label(h):
    h = np.asarray(h, dtype=float)
    return f"{float(h.mean()):.3f}*" if h.ndim else f"{float(h):.3f}"  # '*' marks a per-sample mean


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    all_ks = {}  # name -> {selector: ks}

    for ax, (name, mix) in zip(axes.flat, DISTS.items()):
        rng = np.random.default_rng(SEED)
        samples = mix.sample(N_SAMPLES, rng)
        h0 = parzen.silverman_bandwidth(samples)
        grid = np.linspace(samples.min() - 5 * h0, samples.max() + 5 * h0, 4000)
        true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)

        bandwidths = {k: fn(samples) for k, fn in TRUTH_FREE.items()}
        bandwidths["oracle"] = oracle_bandwidth(samples, grid, true_cdf)

        print(f"\n=== {name} (Silverman h={h0:.4f}) ===")
        print(f"{'selector':<15}{'h':>10}{'KS':>9}{'MSE(pdf)':>11}")
        ks_by_sel = {}
        for sel in ORDER:
            ks, mse = score(samples, bandwidths[sel], grid, true_cdf, true_pdf)
            ks_by_sel[sel] = ks
            tag = "  (oracle: uses truth)" if sel == "oracle" else ""
            print(f"{sel:<15}{h_label(bandwidths[sel]):>10}{ks:>9.4f}{mse:>11.5f}{tag}")
        all_ks[name] = ks_by_sel

        colors = ["tab:gray", "tab:olive", "tab:blue", "tab:cyan", "tab:purple", "tab:red"]
        ax.bar(range(len(ORDER)), [ks_by_sel[s] for s in ORDER], color=colors)
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{name}  (Silverman h={h0:.3f})", fontsize=10)
        ax.set_ylabel("KS vs true CDF")

    fig.suptitle(f"Bandwidth selectors vs the oracle (n={N_SAMPLES}, seed={SEED}). "
                 "All but 'oracle' are truth-free.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "bandwidth_selectors_study.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\nsaved -> {path}")

    _overlay_trimodal()


def _overlay_trimodal():
    """Visual: how each selector's Parzen pdf tracks the true trimodal density."""
    mix = data.asymmetric_trimodal()
    rng = np.random.default_rng(SEED)
    samples = mix.sample(N_SAMPLES, rng)
    h0 = parzen.silverman_bandwidth(samples)
    grid = np.linspace(samples.min() - 5 * h0, samples.max() + 5 * h0, 4000)
    true_pdf, true_cdf = mix.pdf(grid), mix.cdf(grid)
    bandwidths = {k: fn(samples) for k, fn in TRUTH_FREE.items()}
    bandwidths["oracle"] = oracle_bandwidth(samples, grid, true_cdf)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(grid, true_pdf, "k-", lw=2.5, label="true")
    for sel in ORDER:
        ax.plot(grid, parzen.parzen_pdf(grid, samples, bandwidths[sel]), lw=1.2, alpha=0.85, label=sel)
    ax.set_title("Trimodal pdf: Parzen estimate under each bandwidth selector")
    ax.legend(fontsize=8)
    path = os.path.join(RESULTS_DIR, "bandwidth_selectors_trimodal_pdf.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    run()
