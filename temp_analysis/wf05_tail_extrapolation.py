"""wf05 -- tail / out-of-range behaviour: net vs Parzen.

Question: outside the data range the Parzen CDF saturates toward the sigmoid of the
extreme sample (NOT exactly 0/1), and its pdf decays. Does the trained sigmoidal MLP
extrapolate the CDF sensibly to 0 and 1, and how do tail CDF errors compare?

Setup: single Gaussian and symmetric bimodal, n=2000, variance-matched bandwidth.
We measure, per region:
  - "in-range"  : grid points inside [min(sample), max(sample)]
  - "tail-just" : just beyond the data range, out to where TRUE CDF hits ~1e-3 / 1-1e-3
  - "tail-far"  : far beyond (TRUE CDF effectively 0 / 1)
KS and max|err| of CDF vs the TRUE cdf, for Parzen and net. Plus the asymptotic
limit each estimate reaches at +-large x (does the net push to 0/1? does Parzen?).

Strict pipeline: net trains ONLY on (x_i, F_hat(x_i)).  3 seeds.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(2)  # ponytail: many jobs run concurrently; avoid CPU oversubscription

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed

N = 2000
SEEDS = range(3)
EPOCHS = 1500


def train_net(samples, h, seed):
    set_seed(seed)
    inp, tgt = make_sample_training_set(samples, h)
    m = CDFNet(1, (32,), activation="sigmoid", monotone=False)
    m, _ = train_cdf(m, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=EPOCHS, seed=seed))
    return m


def regions(grid, true_cdf, lo, hi):
    """Boolean masks for in-range / just-tail / far-tail."""
    in_range = (grid >= lo) & (grid <= hi)
    # tail = outside data range; split "just" (true cdf in [1e-3,1-1e-3]) vs "far"
    out = ~in_range
    near_extreme = ((true_cdf >= 1e-3) & (true_cdf <= 1 - 1e-3))
    tail_just = out & near_extreme
    tail_far = out & ~near_extreme
    return in_range, tail_just, tail_far


def err_in(true_c, est_c, mask):
    if mask.sum() == 0:
        return np.nan, np.nan
    t, e = true_c[mask], est_c[mask]
    return metrics.ks_distance(t, e), float(np.max(np.abs(t - e)))


def run2(mix, name, span):
    grid = np.linspace(-span, span, 1500)
    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    true_c = mix.cdf(grid)
    far = np.array([-span * 3, span * 3], dtype=float)
    far_t = torch.as_tensor(far, dtype=torch.float32)

    acc = {"parzen": [], "net": []}
    lims = {"parz_l": [], "parz_r": [], "net_l": [], "net_r": []}
    plot_cache = None

    for seed in SEEDS:
        print(f"  [{name}] seed {seed} ...", flush=True)
        s = mix.sample(N, np.random.default_rng(seed))
        lo, hi = float(s.min()), float(s.max())
        h = parzen.variance_matched_bandwidth(s)

        parz_c = parzen.parzen_cdf(grid, s, h)
        m = train_net(s, h, seed)
        net_c = m(grid_t).detach().numpy().ravel()

        in_r, tj, tf = regions(grid, true_c, lo, hi)
        for est_c, who in [(parz_c, "parzen"), (net_c, "net")]:
            acc[who].append([
                *err_in(true_c, est_c, in_r),
                *err_in(true_c, est_c, tj),
                *err_in(true_c, est_c, tf),
            ])

        parz_far = parzen.parzen_cdf(far, s, h)
        net_far = m(far_t).detach().numpy().ravel()
        lims["parz_l"].append(parz_far[0]); lims["parz_r"].append(parz_far[1])
        lims["net_l"].append(net_far[0]); lims["net_r"].append(net_far[1])

        if plot_cache is None:
            plot_cache = (grid, true_c, parz_c, net_c, lo, hi)

    print(f"\n=== {name}  (n={N}, {len(list(SEEDS))} seeds, variance-matched h) ===")
    print(f"  data range across seeds approx [{plot_cache[4]:.2f}, {plot_cache[5]:.2f}], grid span +-{span}")
    hdr = f"  {'est':>7}{'in_KS':>9}{'in_max':>9}{'tj_KS':>9}{'tj_max':>9}{'tf_KS':>9}{'tf_max':>9}"
    print(hdr)
    means = {}
    for who in ("parzen", "net"):
        a = np.array(acc[who])
        m = np.nanmean(a, axis=0)
        means[who] = m
        print(f"  {who:>7}" + "".join(f"{v:>9.4f}" for v in m))

    print("  asymptotic CDF limit far outside data (should be ~0 left, ~1 right):")
    print(f"    parzen: left={np.mean(lims['parz_l']):.4f}  right={np.mean(lims['parz_r']):.4f}")
    print(f"    net   : left={np.mean(lims['net_l']):.4f}  right={np.mean(lims['net_r']):.4f}")

    return plot_cache, means, {k: float(np.mean(v)) for k, v in lims.items()}


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
summary = {}
for ax_row, (mix, name, span) in zip(
    axes,
    [(data.single_gaussian(), "single_gaussian", 8.0),
     (data.symmetric_bimodal(), "symmetric_bimodal", 9.0)],
):
    plot_cache, means, lims = run2(mix, name, span)
    summary[name] = (means, lims)
    grid, true_c, parz_c, net_c, lo, hi = plot_cache

    axL, axR = ax_row
    axL.plot(grid, true_c, "k-", lw=2, label="true CDF")
    axL.plot(grid, parz_c, "C0--", lw=1.4, label="Parzen CDF")
    axL.plot(grid, net_c, "C3-", lw=1.4, label="net CDF")
    axL.axvspan(lo, hi, color="gray", alpha=0.12, label="data range")
    axL.set_title(f"{name}: CDF"); axL.legend(fontsize=8); axL.set_xlabel("x")

    # zoom on the right tail in log scale of (1 - CDF)
    axR.semilogy(grid, np.clip(1 - true_c, 1e-12, None), "k-", lw=2, label="1 - true")
    axR.semilogy(grid, np.clip(1 - parz_c, 1e-12, None), "C0--", lw=1.4, label="1 - Parzen")
    axR.semilogy(grid, np.clip(1 - net_c, 1e-12, None), "C3-", lw=1.4, label="1 - net")
    axR.axvline(hi, color="gray", ls=":", label="max sample")
    axR.set_title(f"{name}: right-tail 1-CDF (log)"); axR.legend(fontsize=8)
    axR.set_xlabel("x"); axR.set_ylim(1e-9, 2)
    axR.set_xlim(hi - 1, span)

plt.tight_layout()
out = "/home/ppianigi/parzen-cdf/temp_analysis/wf05_tail_extrapolation.png"
plt.savefig(out, dpi=110)
print(f"\nsaved plot -> {out}")
