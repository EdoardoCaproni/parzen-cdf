"""Study 2, Phase B, step 3: enforcing monotonicity, in the loss or downstream.

A CDF must never decrease. The network's raw output can violate this locally, so something
must enforce it. Candidates, all trained with the frozen PNN-CDF recipe (leave-one-out
targets, sharp teacher h = 0.5 sigma-hat / sqrt(n-1), width 8):

  raw          nothing enforced (control: how bad is the violation, actually?)
  rectified    downstream fix: running maximum of the output, rescaled to [0, 1]
  penalty(l)   a term l * mean(relu(-dF/dx)) added to the loss at 256 collocation points,
               for l = 1, 10, 100, evaluated on the raw output
  sill         monotone by construction: all weights kept positive (softplus
               reparameterization), so the output cannot decrease anywhere

Metrics: CDF gap (KS), density error (ISE), fraction of grid steps where the raw output
decreases (viol), and the total area of the recovered density (mass; should be 1).
"""

import numpy as np
import torch

from parzen_cdf import data
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, rectify_cdf, train_cdf
from study2_common import (BUDGETS, grid_for, ise, ks, loo_cdf_targets, pnn_window, sigma_hat)

MIX = data.asymmetric_trimodal()
GRID = grid_for(MIX)
T_CDF, T_PDF = MIX.cdf(GRID), MIX.pdf(GRID)
SEEDS = range(5)
WIDTH = 8
C_H1 = 0.5

REGIMES = ["raw", "rectified", "penalty λ=1", "penalty λ=10", "penalty λ=100", "sill"]


def fit(samples, targets, seed, mono_weight=0.0, monotone=False):
    cfg = TrainConfig(epochs=6000, lr=0.03, optimizer="adam", seed=seed,
                      monotonicity_weight=mono_weight)
    model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), activation="sigmoid", monotone=monotone)
    model, _ = train_cdf(model,
                         torch.as_tensor(samples, dtype=torch.float32),
                         torch.as_tensor(targets, dtype=torch.float32), cfg)
    return model


def evaluate(model, rectify):
    with torch.no_grad():
        raw = model(torch.as_tensor(GRID, dtype=torch.float32)).numpy().astype(float)
    diffs = np.diff(raw)
    viol = float(np.mean(diffs < 0))
    if rectify:
        cdf, pdf = rectify_cdf(raw, GRID)
    else:
        cdf = raw
        pdf = np.clip(np.gradient(raw, GRID), 0.0, None)
    return {"ks": ks(cdf, T_CDF), "ise": ise(pdf, T_PDF, GRID),
            "viol": viol, "mass": float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(pdf, GRID))}


results = {}
for n in BUDGETS:
    for seed in SEEDS:
        samples = MIX.sample(n, np.random.default_rng(seed))
        h = pnn_window(samples, C_H1 * sigma_hat(samples))
        targets = loo_cdf_targets(samples, h)

        plain = fit(samples, targets, seed)
        results.setdefault(("raw", n), []).append(evaluate(plain, rectify=False))
        results.setdefault(("rectified", n), []).append(evaluate(plain, rectify=True))
        for lam in (1.0, 10.0, 100.0):
            m = fit(samples, targets, seed, mono_weight=lam)
            results.setdefault((f"penalty λ={lam:g}", n), []).append(evaluate(m, rectify=False))
        m = fit(samples, targets, seed, monotone=True)
        results.setdefault(("sill", n), []).append(evaluate(m, rectify=False))
    print(f"n = {n} done")

print()
print("=" * 88)
print("monotonicity enforcement · trimodal · PNN-CDF recipe (width 8, LOO, 5 seeds)")
for n in BUDGETS:
    print(f"\n  n = {n}")
    print(f"  {'regime':>14} | {'KS':>8} {'ISE':>9} {'viol %':>7} {'mass':>8}")
    for reg in REGIMES:
        r = results[(reg, n)]
        m = {k: np.mean([x[k] for x in r]) for k in r[0]}
        print(f"  {reg:>14} | {m['ks']:>8.4f} {m['ise']:>9.5f} "
              f"{100 * m['viol']:>7.2f} {m['mass']:>8.4f}")
