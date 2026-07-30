"""Study 2, Phase B, step 1: the PNN recipe transplanted to the CDF (trimodal running example).

The Parzen Neural Network (Trentin) beats the Parzen Window it is trained from by combining
(a) leave-one-out ("unbiased") targets, (b) a deliberately sharp window, and (c) a network of
limited capacity whose smoothness averages the target's estimation noise away. Here the same
recipe is applied to the CDF: LOO logistic-Parzen CDF targets at the sample points only,
h_n = h1/sqrt(n-1), sigmoidal MLP, rectified downstream (which also guarantees mass = 1, the
validity the pdf-PNN lacks by construction).

Two experiments:
  (i)  capacity probe: the U-curve of net accuracy vs width on a noisy teacher;
  (ii) teacher-regime sweep: net vs the PW it learned from, across h1 = c * sigma-hat.
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parzen_cdf import data, parzen
from study2_common import (BUDGETS, ecdf_ks_mean, eval_cdf_net, eval_window, fit_cdf_net,
                           grid_for, ks_floor, loo_cdf_targets, pnn_window, sigma_hat)

MIX = data.asymmetric_trimodal()
GRID = grid_for(MIX)
T_CDF, T_PDF = MIX.cdf(GRID), MIX.pdf(GRID)
SEEDS = range(5)
C_SIGMA_RULE = 1.5


def run_once(n, seed, c_h1, width):
    samples = MIX.sample(n, np.random.default_rng(seed))
    h = pnn_window(samples, c_h1 * sigma_hat(samples))
    teacher = eval_window(MIX, GRID, T_CDF, T_PDF, samples, h)          # the PW it learns from
    model, final_loss = fit_cdf_net(samples, loo_cdf_targets(samples, h), width=width, seed=seed)
    net = eval_cdf_net(model, GRID, T_CDF, T_PDF)
    ref = eval_window(MIX, GRID, T_CDF, T_PDF, samples,                 # best-practice PW
                      C_SIGMA_RULE * sigma_hat(samples) / np.sqrt(n))
    return teacher, net, ref, final_loss


# ------------------------------------------------------------------ (i) capacity probe
print("=" * 84)
print("(i) capacity probe · trimodal, n = 1000, sharp teacher h1 = 0.5·sigma (5 seeds)")
print(f"{'width':>6} | {'net KS':>8} {'net ISE':>9} | teacher: KS 0.0(-) ISE (-)   [filled below]")
probe = {}
teach_probe = []
for width in (8, 16, 32, 64):
    rows = []
    for seed in SEEDS:
        teacher, net, _, _ = run_once(1000, seed, 0.5, width)
        rows.append((net["ks"], net["ise"]))
        if width == 8:
            teach_probe.append((teacher["ks"], teacher["ise"]))
    probe[width] = np.mean(rows, axis=0)
    print(f"{width:>6} | {probe[width][0]:>8.4f} {probe[width][1]:>9.5f}")
t_ks, t_ise = np.mean(teach_probe, axis=0)
print(f"{'PW':>6} | {t_ks:>8.4f} {t_ise:>9.5f}   (the teacher itself, same h)")
best_width = min(probe, key=lambda w: probe[w][1])
print(f"-> width kept for step (ii): {best_width} (best pdf ISE)")

# ------------------------------------------------------------------ (ii) teacher-regime sweep
print()
print("=" * 84)
print(f"(ii) net vs its teacher across the teacher's sharpness (width {best_width}, 5 seeds)")
C_GRID = (0.25, 0.5, 1.0, 1.5, 3.0)
sweep = {}
for n in BUDGETS:
    print(f"\n  n = {n}   (ECDF KS {ecdf_ks_mean(MIX, n, SEEDS):.4f}, floor {ks_floor(n):.4f})")
    print(f"  {'h1/sigma':>8} | {'PW KS':>8} {'net KS':>8} {'gain':>6} | "
          f"{'PW ISE':>9} {'net ISE':>9} {'gain':>6}")
    for c in C_GRID:
        rows = []
        for seed in SEEDS:
            teacher, net, ref, _ = run_once(n, seed, c, best_width)
            rows.append((teacher["ks"], net["ks"], teacher["ise"], net["ise"],
                         ref["ks"], ref["ise"]))
        m = np.mean(rows, axis=0)
        sweep[(n, c)] = m
        print(f"  {c:>8.2f} | {m[0]:>8.4f} {m[1]:>8.4f} {m[0] / m[1]:>5.1f}x | "
              f"{m[2]:>9.5f} {m[3]:>9.5f} {m[2] / m[3]:>5.1f}x")
    ref_ks, ref_ise = m[4], m[5]
    print(f"  {'σ-rule PW reference':>8}:  KS {ref_ks:.4f}   ISE {ref_ise:.5f}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# left: the PNN-style picture, pdf at n=500 with a sharp teacher
n_show, c_show, seed_show = 500, 0.5, 0
samples = MIX.sample(n_show, np.random.default_rng(seed_show))
h = pnn_window(samples, c_show * sigma_hat(samples))
model, _ = fit_cdf_net(samples, loo_cdf_targets(samples, h), width=best_width, seed=seed_show)
import torch
from parzen_cdf.training import rectify_cdf
with torch.no_grad():
    raw = model(torch.as_tensor(GRID, dtype=torch.float32)).numpy().astype(float)
_, net_pdf = rectify_cdf(raw, GRID)
axes[0].plot(GRID, T_PDF, "k--", lw=1.5, label="truth")
axes[0].plot(GRID, parzen.parzen_pdf(GRID, samples, h), lw=0.8, alpha=0.8,
             label=f"Parzen (teacher), h1 = {c_show}σ")
axes[0].plot(GRID, net_pdf, lw=1.8, label="MLP on LOO targets")
axes[0].set_title(f"pdf · trimodal · n = {n_show}")
axes[0].legend(fontsize=8)

# middle + right: gains vs teacher sharpness
for ax, (idx_pw, idx_net, name) in zip(axes[1:], [(0, 1, "CDF gap (KS)"), (2, 3, "pdf ISE")]):
    for n in BUDGETS:
        pw = [sweep[(n, c)][idx_pw] for c in C_GRID]
        net = [sweep[(n, c)][idx_net] for c in C_GRID]
        line, = ax.plot(C_GRID, pw, "o--", lw=1, ms=4, label=f"PW n={n}")
        ax.plot(C_GRID, net, "s-", lw=1.6, ms=4, color=line.get_color(), label=f"net n={n}")
    ax.set_xscale("log")
    ax.set_xticks(C_GRID)
    ax.set_xticklabels([str(c) for c in C_GRID])
    ax.set_yscale("log")
    ax.set_xlabel("teacher sharpness h1 / sigma-hat")
    ax.set_title(f"{name}: net (solid) vs its teacher (dashed)")
    ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig("results/mlp2_pnn_cdf.png", dpi=120)
print("\nfigure -> results/mlp2_pnn_cdf.png")
