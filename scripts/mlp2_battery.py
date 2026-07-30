"""Study 2, Phase B, step 2: the PNN-CDF recipe on the battery of 10 random mixtures.

Frozen recipe (from step B1, before touching the battery): leave-one-out logistic-Parzen CDF
targets, h_n = C_H1 * sigma-hat / sqrt(n-1), sigmoidal MLP of width WIDTH trained on the
sample points only, downstream rectification. Compared against:
  - the teacher PW itself (same window),
  - the sigma-rule PW (the best truth-free Parzen of Phase A),
  - the empirical CDF and the statistical floor.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parzen_cdf import data, parzen
from study2_common import (BUDGETS, ecdf_on_grid, eval_cdf_net, eval_window, fit_cdf_net,
                           grid_for, ks, ks_floor, loo_cdf_targets, pnn_window, sigma_hat)

WIDTH = 8         # frozen from step B1: small capacity IS the regularizer
C_H1 = 0.5        # frozen from step B1: the sharp-teacher regime
C_SIGMA_RULE = 1.5
SEEDS_PER = 3


def cases():
    rng = np.random.default_rng(0)
    return [(f"mixture {i + 1}", data.random_mixture(rng)) for i in range(10)]


rows = {}
for name, mix in cases():
    grid = grid_for(mix)
    t_cdf, t_pdf = mix.cdf(grid), mix.pdf(grid)
    for n in BUDGETS:
        for s in range(SEEDS_PER):
            samples = mix.sample(n, np.random.default_rng(1000 + 17 * s + len(name)))
            sig = sigma_hat(samples)
            h_teacher = pnn_window(samples, C_H1 * sig)
            teacher = eval_window(mix, grid, t_cdf, t_pdf, samples, h_teacher)
            model, _ = fit_cdf_net(samples, loo_cdf_targets(samples, h_teacher),
                                   width=WIDTH, seed=s)
            net = eval_cdf_net(model, grid, t_cdf, t_pdf)
            ref = eval_window(mix, grid, t_cdf, t_pdf, samples, C_SIGMA_RULE * sig / np.sqrt(n))
            e_ks = ks(ecdf_on_grid(samples, grid), t_cdf)
            rows.setdefault((name, n), []).append(
                (teacher["ks"], net["ks"], ref["ks"], e_ks,
                 teacher["ise"], net["ise"], ref["ise"]))
    print(f"{name} done")

LABELS = ["PW teacher", "net (PNN-CDF)", "PW σ-rule", "ECDF"]

print()
print("=" * 92)
print("mean over the 10 random mixtures (3 seeds each)")
for metric, sl in (("CDF gap (KS)", slice(0, 4)), ("pdf ISE", slice(4, 7))):
    print(f"\n--- {metric}")
    labels = LABELS[:sl.stop - sl.start]
    header = f"{'estimator':>14} | " + " ".join(f"{f'n={n}':>9}" for n in BUDGETS)
    print(header)
    print("-" * len(header))
    for k, lab in enumerate(labels):
        vals = []
        for n in BUDGETS:
            per = [np.mean(rows[(nm, n)], axis=0)[sl][k] for nm, _ in cases()]
            vals.append(np.mean(per))
        print(f"{lab:>14} | " + " ".join(f"{v:>9.5f}" for v in vals))
    if metric.startswith("CDF"):
        print(f"{'floor':>14} | " + " ".join(f"{ks_floor(n):>9.5f}" for n in BUDGETS))

print()
print("net-beats-teacher scoreboard (per mixture-budget mean):")
wins_ks = wins_ise = tot = 0
for (name, n), r in rows.items():
    m = np.mean(r, axis=0)
    tot += 1
    wins_ks += m[1] < m[0]
    wins_ise += m[5] < m[4]
print(f"  KS : net better than its teacher in {wins_ks}/{tot} cases")
print(f"  ISE: net better than its teacher in {wins_ise}/{tot} cases")

# ------------------------------------------------------------------ gallery figure (n = 500)
n_show = 500
fig, axes = plt.subplots(2, 5, figsize=(16, 5.6))
for mi, ((name, mix), ax) in enumerate(zip(cases(), axes.ravel())):
    grid = grid_for(mix)
    samples = mix.sample(n_show, np.random.default_rng(1000 + len(name)))
    h = pnn_window(samples, C_H1 * sigma_hat(samples))
    model, _ = fit_cdf_net(samples, loo_cdf_targets(samples, h), width=WIDTH, seed=0)
    import torch
    from parzen_cdf.training import rectify_cdf
    with torch.no_grad():
        raw = model(torch.as_tensor(grid, dtype=torch.float32)).numpy().astype(float)
    _, net_pdf = rectify_cdf(raw, grid)
    ax.plot(grid, mix.pdf(grid), "k--", lw=1.2)
    ax.plot(grid, parzen.parzen_pdf(grid, samples, h), lw=0.6, alpha=0.75)
    ax.plot(grid, net_pdf, lw=1.5)
    ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=7)
axes[0][0].legend(["reference density", "PW teacher (sharp)", "net (PNN-CDF)"], fontsize=7)
fig.suptitle(f"The battery at n = {n_show}: sharp Parzen teacher and the network trained on it", y=1.0)
fig.tight_layout()
fig.savefig("results/mlp2_battery_gallery.png", dpi=120)

# ------------------------------------------------------------------ scatter: net vs teacher ISE
fig, ax = plt.subplots(figsize=(5.2, 4.8))
markers = {500: "o", 1000: "s", 2000: "^"}
for n in BUDGETS:
    xs = [np.mean(rows[(nm, n)], axis=0)[4] for nm, _ in cases()]
    ys = [np.mean(rows[(nm, n)], axis=0)[5] for nm, _ in cases()]
    ax.scatter(xs, ys, marker=markers[n], s=30, label=f"n = {n}")
lim = [0.0005, 0.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("teacher PW density error (ISE)")
ax.set_ylabel("network density error (ISE)")
ax.set_title("Below the diagonal: the network beats its teacher")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/mlp2_battery_scatter.png", dpi=120)
print("\nfigures -> results/mlp2_battery_gallery.png, results/mlp2_battery_scatter.png")
