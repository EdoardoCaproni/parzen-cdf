"""wf04: CDF route vs direct-pdf route (Magdon-Ismail "CDF is easier" claim).

Claim under test: constraining a CDF (monotone + range [0,1]) is EASIER than
constraining a pdf (>= 0 AND integrates to 1). The asymmetry: for a CDF, BOTH
constraints are cheap and EXACT via a downstream rectification (cumulative max +
rescale to [0,1]) -> differentiate -> a density that is non-negative AND
integrates to exactly 1 by construction. For a direct-pdf net, non-negativity is
free (softplus output) but normalization (int f = 1) is a GLOBAL constraint that
cannot be enforced pointwise: you must either add a penalty (soft, never exact)
or renormalize post-hoc (exact mass, but the mass estimate is itself noisy and
the trained shape was fit under no mass constraint).

We hold everything else equal: same distributions, same net capacity, same
training-set rule (the project constraint: train ONLY on the n data points x_i,
labels = the Parzen estimate there), same optimizer/lr/epochs/seeds. We then
ask: which route gives a more VALID and more ACCURATE density, with less tuning?

Routes
  (a) CDF route:        CDFNet -> Parzen CDF targets -> rectify -> differentiate.
  (b1) direct-pdf + penalty:    softplus-output net -> Parzen pdf targets,
                                 + lambda * (int f - 1)^2 normalization penalty.
  (b2) direct-pdf + post-hoc:   same net, no penalty, renormalize on the grid.

Metrics vs TRUTH: density MSE, density KS-like sup error is not standard so we
report pdf-MSE and CDF-KS; plus VALIDITY: how far does each density's mass
deviate from 1 BEFORE any cheating, and negativity.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig,
    density_from_cdf,
    make_sample_training_set,
    rectify_cdf,
    set_seed,
    train_cdf,
)

N = 500
EPOCHS = 1000
LR = 0.03
HIDDEN = (24,)
SEEDS = (0, 1)
GRID_N = 800
PEN_GRID_N = 200  # coarse grid used ONLY for the per-epoch normalization penalty (keeps it cheap)


class PDFNet(nn.Module):
    """Direct-pdf MLP: same shape/capacity as CDFNet but a softplus output head so the
    density is non-negative by construction. Normalization is NOT built in."""

    def __init__(self, hidden=(24,)):
        super().__init__()
        dims = [1, *hidden, 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = x
        last = len(self.layers) - 1
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < last:
                h = torch.sigmoid(h)
        return nn.functional.softplus(h).squeeze(-1)  # >= 0 always


def train_pdf_net(samples, h, pen_grid_t, seed, norm_weight):
    """Train the direct-pdf net on (x_i, Parzen_pdf(x_i)). Optional normalization penalty
    enforced as (trapz(f, grid) - 1)^2 on a COARSE fixed grid (cheap). Returns the trained net."""
    set_seed(seed)
    samples = np.asarray(samples, float)
    targets = parzen.parzen_pdf(samples, samples, h)
    x_t = torch.as_tensor(samples, dtype=torch.float32)
    y_t = torch.as_tensor(targets, dtype=torch.float32)
    net = PDFNet(HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    mse = nn.MSELoss()
    gx = pen_grid_t  # coarse grid for the mass penalty only
    dg = (gx[1:] - gx[:-1])
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = mse(net(x_t), y_t)
        if norm_weight > 0:
            fg = net(gx)
            mass = (0.5 * (fg[1:] + fg[:-1]) * dg).sum()  # trapezoid on grid
            loss = loss + norm_weight * (mass - 1.0) ** 2
        loss.backward()
        opt.step()
    return net


def run_dist(name, dist):
    samples_rng = np.random.default_rng(1234)
    h = parzen.variance_matched_bandwidth(dist.sample(N, np.random.default_rng(99)))
    lo, hi = None, None
    rows = []
    per_seed = {"cdf": [], "pdf_penalty": [], "pdf_posthoc": []}
    last_curves = None

    for seed in SEEDS:
        samples = dist.sample(N, np.random.default_rng(1000 + seed))
        h_s = parzen.variance_matched_bandwidth(samples)
        lo = float(samples.min() - 4 * h_s)
        hi = float(samples.max() + 4 * h_s)
        grid = np.linspace(lo, hi, GRID_N)
        grid_t = torch.as_tensor(grid, dtype=torch.float32)
        pen_grid_t = torch.as_tensor(np.linspace(lo, hi, PEN_GRID_N), dtype=torch.float32)
        pdf_true = dist.pdf(grid)
        cdf_true = dist.cdf(grid)

        # ---- (a) CDF route -------------------------------------------------
        inputs_t, targets_t = make_sample_training_set(samples, h_s)
        model = CDFNet(in_dim=1, hidden_sizes=HIDDEN, activation="sigmoid", monotone=False)
        model, _ = train_cdf(model, inputs_t, targets_t,
                             TrainConfig(optimizer="adam", lr=LR, epochs=EPOCHS, seed=seed))
        with torch.no_grad():
            cdf_raw = model(grid_t).numpy()
        # raw density (before rectify) to measure native validity
        dens_raw = density_from_cdf(model, grid_t, clamp=False).detach().numpy()
        mass_cdf_raw = metrics.integrates_to_one(np.clip(dens_raw, 0, None), grid)
        neg_cdf_raw = float(np.mean(dens_raw < 0))
        # rectify: monotone + [0,1] EXACTLY -> density integrates to 1 by construction
        cdf_rect, pdf_cdf = rectify_cdf(cdf_raw, grid)
        mass_cdf = metrics.integrates_to_one(pdf_cdf, grid)
        per_seed["cdf"].append(dict(
            pdf_mse=metrics.mse(pdf_true, pdf_cdf),
            cdf_ks=metrics.ks_distance(cdf_true, cdf_rect),
            mass_native=mass_cdf_raw,         # raw-derivative mass, no fix
            mass_final=mass_cdf,              # after rectify
            neg_frac=neg_cdf_raw,
        ))

        # ---- (b1) direct-pdf + normalization penalty -----------------------
        net_pen = train_pdf_net(samples, h_s, pen_grid_t, seed, norm_weight=10.0)
        with torch.no_grad():
            f_pen = net_pen(grid_t).numpy()
        mass_pen = metrics.integrates_to_one(f_pen, grid)
        # validity: how far is mass from 1 with ONLY the penalty (no post-hoc fix)?
        # accuracy uses the as-trained density (the honest validity case)
        per_seed["pdf_penalty"].append(dict(
            pdf_mse=metrics.mse(pdf_true, f_pen),
            mass_native=mass_pen,
            neg_frac=0.0,  # softplus -> never negative
        ))

        # ---- (b2) direct-pdf, no penalty, post-hoc renormalize -------------
        net_raw = train_pdf_net(samples, h_s, pen_grid_t, seed, norm_weight=0.0)
        with torch.no_grad():
            f_raw = net_raw(grid_t).numpy()
        mass_raw = metrics.integrates_to_one(f_raw, grid)
        f_norm = f_raw / mass_raw if mass_raw > 0 else f_raw
        per_seed["pdf_posthoc"].append(dict(
            pdf_mse_native=metrics.mse(pdf_true, f_raw),     # before renorm
            pdf_mse=metrics.mse(pdf_true, f_norm),           # after renorm
            mass_native=mass_raw,
            neg_frac=0.0,
        ))

        if seed == SEEDS[-1]:
            last_curves = dict(grid=grid, pdf_true=pdf_true,
                               pdf_cdf=pdf_cdf, f_pen=f_pen, f_norm=f_norm)

    def agg(key, field):
        vals = [d[field] for d in per_seed[key] if field in d]
        return float(np.mean(vals)), float(np.std(vals))

    print(f"\n=== {name}  (n={N}, h~{h_s:.3f}, hidden={HIDDEN}, epochs={EPOCHS}, seeds={SEEDS}) ===", flush=True)
    print(f"  (a) CDF route:        pdf_MSE={agg('cdf','pdf_mse')[0]:.5f}+-{agg('cdf','pdf_mse')[1]:.5f}"
          f"  CDF_KS={agg('cdf','cdf_ks')[0]:.4f}"
          f"  mass(native deriv)={agg('cdf','mass_native')[0]:.4f}"
          f"  mass(after rectify)={agg('cdf','mass_final')[0]:.5f}"
          f"  neg_frac(native)={agg('cdf','neg_frac')[0]:.4f}", flush=True)
    print(f"  (b1) pdf+penalty:     pdf_MSE={agg('pdf_penalty','pdf_mse')[0]:.5f}+-{agg('pdf_penalty','pdf_mse')[1]:.5f}"
          f"  mass(native,penalty only)={agg('pdf_penalty','mass_native')[0]:.4f}"
          f"  neg_frac=0 (softplus)", flush=True)
    print(f"  (b2) pdf+post-hoc:    pdf_MSE(after renorm)={agg('pdf_posthoc','pdf_mse')[0]:.5f}+-{agg('pdf_posthoc','pdf_mse')[1]:.5f}"
          f"  pdf_MSE(native)={agg('pdf_posthoc','pdf_mse_native')[0]:.5f}"
          f"  mass(native,no fix)={agg('pdf_posthoc','mass_native')[0]:.4f}", flush=True)

    return name, per_seed, last_curves


def main():
    dists = [
        ("single_gaussian", data.single_gaussian()),
        ("asymmetric_bimodal", data.asymmetric_bimodal()),
        ("asymmetric_trimodal", data.asymmetric_trimodal()),
    ]
    results = [run_dist(n, d) for n, d in dists]

    # one figure: density overlays per distribution
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, _, cv) in zip(axes, results):
        g = cv["grid"]
        ax.plot(g, cv["pdf_true"], "k-", lw=2, label="truth")
        ax.plot(g, cv["pdf_cdf"], "C0-", lw=1.3, label="CDF route (rectified)")
        ax.plot(g, cv["f_pen"], "C1--", lw=1.1, label="pdf + penalty")
        ax.plot(g, cv["f_norm"], "C2:", lw=1.3, label="pdf + post-hoc norm")
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = "temp_analysis/wf04_pdf_vs_cdf_route.png"
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")

    # headline summary across distributions
    print("\n=== HEADLINE (mean over distributions & seeds) ===")
    def overall(key, field):
        v = []
        for _, ps, _ in results:
            v += [d[field] for d in ps[key] if field in d]
        return float(np.mean(v))
    print(f"  CDF route       mean pdf_MSE = {overall('cdf','pdf_mse'):.5f}   mass after rectify = {overall('cdf','mass_final'):.5f} (exact by construction)")
    print(f"  pdf + penalty   mean pdf_MSE = {overall('pdf_penalty','pdf_mse'):.5f}   mass with penalty only = {overall('pdf_penalty','mass_native'):.4f}")
    print(f"  pdf + post-hoc  mean pdf_MSE = {overall('pdf_posthoc','pdf_mse'):.5f}   mass before any fix = {overall('pdf_posthoc','mass_native'):.4f}")
    print(f"  (direct-pdf native mass spread tells you how badly 'integrate to 1' is violated without cheating)")


if __name__ == "__main__":
    main()
