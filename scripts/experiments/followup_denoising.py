"""Follow-up (NOT a naive experiment) -- can the network BEAT the kernel estimate, not just copy it?

The naive experiments (e0_*, stage_a_*, stage_b_*) established that a well-trained network reproduces
its Parzen target but does not surpass it. This follow-up tests the two proposed power-ups inside the
strict rule (train only on (x_i, F_hat(x_i))):

  Power-up 1 -- smoothing / denoising. The Parzen labels are a noisy estimate of the true CDF; a
    network that fits their *trend* instead of interpolating their wiggle can average the noise out.
    A FIRST attempt with plain weight decay failed badly (it pulls the output toward a flat 0.5, the
    wrong prior for a CDF). Here we use the CDF-appropriate prior: a curvature penalty on d2F/dx2,
    evaluated at the data points only.
  Power-up 2 -- ensembling. Average several networks (different seeds) to cancel their variance. We
    estimate the ensemble's spread honestly by averaging over ALL size-5 subsets of 8 trained nets.

and the key combination: a SHARPER (less biased) bandwidth whose extra noise the ensemble cleans up,
which is the only candidate that could beat the best fixed-bandwidth kernel.

Two references: the Parzen estimate at Silverman's bandwidth (the "ceiling" the naive network matched)
and at the adaptive bandwidth (the best truth-free kernel from the naive study). The denoising effect
should be largest at small n (noisiest labels), so we sweep n. Everything is averaged over seeds.

    python scripts/experiments/followup_denoising.py
"""

import json
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, density_from_cdf, make_sample_training_set, set_seed, train_cdf

RESULTS_DIR = "results"
SEEDS = range(8)
ENSEMBLE_K = 5          # ensemble size; spread estimated over all C(8,5)=56 subsets
WIDTH = 32
LR = 3e-2               # safe rate (lr 0.1 diverged at large width in the naive study)
EPOCHS = 2500
CURV_GRID = (1e-3, 1e-2)
SHARP = 0.5             # combo bandwidth = SHARP * Silverman (less biased, noisier target)
NS = (500, 2000)

DISTS = {
    "single_gaussian": data.single_gaussian(),
    "symmetric_bimodal": data.symmetric_bimodal(),
    "asymmetric_bimodal": data.asymmetric_bimodal(),
}


def grid_for(mix):
    lo = float((mix.means - 5 * mix.stds).min())
    hi = float((mix.means + 5 * mix.stds).max())
    g = np.linspace(lo, hi, 2000)
    return g, torch.as_tensor(g, dtype=torch.float32)


def run_net(samples, h, curvw, grid_t, grid, true_cdf, true_pdf, seed):
    """Train one network (data points only); curvature penalty (if any) is evaluated at the data
    points. Returns (cdf_on_grid, KS, density_MSE)."""
    set_seed(seed)
    inputs, targets = make_sample_training_set(samples, h)
    model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), monotone=False)
    pts = inputs if curvw > 0 else None
    model, _ = train_cdf(model, inputs, targets,
                         TrainConfig(epochs=EPOCHS, lr=LR, curvature_weight=curvw, seed=seed),
                         penalty_points=pts)
    cdf = model(grid_t).detach().numpy()
    pdf = density_from_cdf(model, grid_t, clamp=True).detach().numpy()
    return cdf, metrics.ks_distance(true_cdf, cdf), metrics.mse(true_pdf, pdf)


def agg(xs):
    return {"ks_mean": float(np.mean(xs)), "ks_std": float(np.std(xs))}


def ensemble_stats(cdfs, true_cdf, k):
    """Mean +/- std of the ensemble KS over all size-k subsets of the trained nets."""
    kss = [metrics.ks_distance(true_cdf, np.mean([cdfs[i] for i in idx], axis=0))
           for idx in combinations(range(len(cdfs)), k)]
    return agg(kss)


def run_cell(mix, n, grid, grid_t, true_cdf, true_pdf):
    parzen_silv, parzen_adapt = [], []
    plain_silv_ks, plain_sharp_ks = [], []
    curv_ks = {w: [] for w in CURV_GRID}
    cdfs_silv, cdfs_sharp = [], []

    for seed in SEEDS:
        samples = mix.sample(n, np.random.default_rng(seed))
        h = parzen.silverman_bandwidth(samples)
        parzen_silv.append(metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, h)))
        parzen_adapt.append(metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, samples, parzen.adaptive_bandwidths(samples))))

        cdf, ks, _ = run_net(samples, h, 0.0, grid_t, grid, true_cdf, true_pdf, seed)
        plain_silv_ks.append(ks); cdfs_silv.append(cdf)
        for w in CURV_GRID:
            _, ksc, _ = run_net(samples, h, w, grid_t, grid, true_cdf, true_pdf, seed)
            curv_ks[w].append(ksc)
        cdf_s, ks_s, _ = run_net(samples, SHARP * h, 0.0, grid_t, grid, true_cdf, true_pdf, seed)
        plain_sharp_ks.append(ks_s); cdfs_sharp.append(cdf_s)

    return {
        "parzen silverman (ceiling)": agg(parzen_silv),
        "parzen adaptive (best kernel)": agg(parzen_adapt),
        "net plain @ silverman": agg(plain_silv_ks),
        "net + curvature 1e-3 @ silverman": agg(curv_ks[1e-3]),
        "net + curvature 1e-2 @ silverman": agg(curv_ks[1e-2]),
        f"ENSEMBLE @ silverman (any {ENSEMBLE_K} of 8)": ensemble_stats(cdfs_silv, true_cdf, ENSEMBLE_K),
        f"net plain @ {SHARP:g}x bandwidth": agg(plain_sharp_ks),
        f"ENSEMBLE @ {SHARP:g}x bandwidth (combo)": ensemble_stats(cdfs_sharp, true_cdf, ENSEMBLE_K),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for name, mix in DISTS.items():
        grid, grid_t = grid_for(mix)
        true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)
        results[name] = {}
        print(f"\n##### {name} | width={WIDTH} lr={LR:g} epochs={EPOCHS} | mean+/-std over {len(list(SEEDS))} seeds")
        for n in NS:
            cell = run_cell(mix, n, grid, grid_t, true_cdf, true_pdf)
            results[name][f"n={n}"] = cell
            ceil = cell["parzen silverman (ceiling)"]["ks_mean"]
            best_kernel = cell["parzen adaptive (best kernel)"]["ks_mean"]
            print(f"\n  n={n}  (Silverman ceiling KS={ceil:.4f}, adaptive kernel KS={best_kernel:.4f})")
            for k, r in cell.items():
                flag = ""
                if not k.startswith("parzen"):
                    if r["ks_mean"] < best_kernel:
                        flag = " <== beats BEST kernel"
                    elif r["ks_mean"] < ceil:
                        flag = " <- beats ceiling"
                print(f"    {k:<42}{r['ks_mean']:.4f} +/- {r['ks_std']:.4f}{flag}")

    with open(os.path.join(RESULTS_DIR, "followup_denoising.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {os.path.join(RESULTS_DIR, 'followup_denoising.json')}")
    _figure(results)


def _figure(results):
    order = ["net plain @ silverman", "net + curvature 1e-3 @ silverman",
             "net + curvature 1e-2 @ silverman", f"ENSEMBLE @ silverman (any {ENSEMBLE_K} of 8)",
             f"net plain @ {SHARP:g}x bandwidth", f"ENSEMBLE @ {SHARP:g}x bandwidth (combo)"]
    short = ["plain", "curv 1e-3", "curv 1e-2", "ensemble", "plain @0.5h", "ensemble @0.5h"]
    fig, axes = plt.subplots(1, len(DISTS), figsize=(17, 5.5))
    for ax, (name, byn) in zip(axes, results.items()):
        cell = byn["n=500"]
        means = [cell[k]["ks_mean"] for k in order]
        stds = [cell[k]["ks_std"] for k in order]
        ax.bar(range(len(order)), means, yerr=stds, capsize=3,
               color=["tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:blue", "tab:green"])
        ax.axhline(cell["parzen silverman (ceiling)"]["ks_mean"], ls="--", color="gray", label="Silverman ceiling")
        ax.axhline(cell["parzen adaptive (best kernel)"]["ks_mean"], ls=":", color="black", label="adaptive kernel (best)")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{name}  (n=500)"); ax.set_ylabel("CDF gap vs truth"); ax.legend(fontsize=7)
    fig.suptitle("Follow-up: curvature smoothing + ensembling vs the kernel references "
                 "(n=500, mean+/-std over seeds). Dashed = Silverman ceiling; dotted = best kernel.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "followup_denoising.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
