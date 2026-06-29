"""Phase B / step on monotonicity -- enforce a non-decreasing CDF where the net actually violates it.

In step 3 the unconstrained net produced ~6% monotonicity violations on the symmetric bimodal at
n=20000, width 16. Here we compare, on that exact regime (3 seeds), the three enforcement routes we
have, and ask which removes the violations at the least accuracy cost:

  baseline    -- unconstrained (the violating net), for reference;
  soft pen.   -- a loss penalty mean(relu(-dF/dx)) during training (weights 10 and 100);
  Sill        -- monotone by construction (non-negative weights);
  rectified   -- downstream fix of the trained baseline: cumulative-max of the CDF, then rescale to
                 [0,1] (which also normalises the recovered density to integrate to 1).

Score = CDF gap (KS) vs truth, monotonicity violation fraction, pdf MSE, recovered mass.

    python scripts/mlp_monotonicity.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, density_from_cdf, make_sample_training_set, set_seed, train_cdf

RESULTS_DIR = "results"
SEEDS = range(3)
N = 20000
WIDTH = 16
LR = 0.03
EPOCHS = 5000

mix = data.symmetric_bimodal()
GRID = np.linspace(-5.5, 5.5, 2000)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF, TRUE_PDF = mix.cdf(GRID), mix.pdf(GRID)


def train(inputs, targets, seed, monotone, soft):
    set_seed(seed)
    model = CDFNet(in_dim=1, hidden_sizes=(WIDTH,), activation="sigmoid", monotone=monotone)
    model, _ = train_cdf(model, inputs, targets, TrainConfig(optimizer="adam", lr=LR, epochs=EPOCHS,
                                                             monotonicity_weight=soft, seed=seed))
    cdf = model(GRID_T).detach().numpy()
    pdf = density_from_cdf(model, GRID_T, clamp=True).detach().numpy()
    return cdf, pdf


def rectify(cdf):
    """Downstream fix: make the CDF non-decreasing (cumulative max) and rescale to [0,1]; the
    rescaled derivative then integrates to 1."""
    rc = np.maximum.accumulate(cdf)
    lo, hi = rc[0], rc[-1]
    if hi > lo:
        rc = (rc - lo) / (hi - lo)
    pdf = np.clip(np.gradient(rc, GRID), 0.0, None)
    return rc, pdf


def score(cdf, pdf):
    return dict(ks=metrics.ks_distance(TRUE_CDF, cdf), viol=metrics.monotonicity_violation_fraction(cdf),
                pdf_mse=metrics.mse(TRUE_PDF, pdf), mass=metrics.integrates_to_one(pdf, GRID))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = ["baseline", "soft 10", "soft 100", "Sill", "rectified"]
    acc = {s: {k: [] for k in ("ks", "viol", "pdf_mse", "mass")} for s in strategies}
    seed0 = {}
    for seed in SEEDS:
        samples = mix.sample(N, np.random.default_rng(seed))
        h = parzen.variance_matched_bandwidth(samples)
        inputs, targets = make_sample_training_set(samples, h)
        if seed == 0:
            seed0["parzen_cdf"] = parzen.parzen_cdf(GRID, samples, h)
            seed0["parzen_pdf"] = parzen.parzen_pdf(GRID, samples, h)

        cdf_b, pdf_b = train(inputs, targets, seed, monotone=False, soft=0.0)
        cdf_s10, pdf_s10 = train(inputs, targets, seed, monotone=False, soft=10.0)
        cdf_s100, pdf_s100 = train(inputs, targets, seed, monotone=False, soft=100.0)
        cdf_sill, pdf_sill = train(inputs, targets, seed, monotone=True, soft=0.0)
        cdf_rec, pdf_rec = rectify(cdf_b)

        results = {"baseline": (cdf_b, pdf_b), "soft 10": (cdf_s10, pdf_s10), "soft 100": (cdf_s100, pdf_s100),
                   "Sill": (cdf_sill, pdf_sill), "rectified": (cdf_rec, pdf_rec)}
        for s, (c, p) in results.items():
            sc = score(c, p)
            for k, v in sc.items():
                acc[s][k].append(v)
        if seed == 0:
            seed0["results"] = results

    print(f"symmetric_bimodal, n={N}, width={WIDTH}, Adam lr={LR}  (mean over {len(list(SEEDS))} seeds)")
    print(f"  {'strategy':<11}{'KS':>9}{'viol%':>8}{'pdf MSE':>10}{'mass':>8}")
    for s in strategies:
        a = acc[s]
        print(f"  {s:<11}{np.mean(a['ks']):>9.4f}{100*np.mean(a['viol']):>7.2f}%"
              f"{np.mean(a['pdf_mse']):>10.5f}{np.mean(a['mass']):>8.4f}")

    # figure (seed 0): baseline (violating) vs rectified vs Sill, against truth + target
    r = seed0["results"]
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(13, 5))
    ax_c.plot(GRID, TRUE_CDF, "k-", lw=2, label="true")
    ax_c.plot(GRID, seed0["parzen_cdf"], "--", color="gray", label="Parzen target")
    ax_c.plot(GRID, r["baseline"][0], "-", color="tab:red", alpha=0.8, label="baseline (violates)")
    ax_c.plot(GRID, r["rectified"][0], "-", color="tab:green", label="rectified")
    ax_c.plot(GRID, r["Sill"][0], "-", color="tab:purple", alpha=0.8, label="Sill")
    ax_c.set_title("CDF"); ax_c.legend(fontsize=8)
    ax_p.plot(GRID, TRUE_PDF, "k-", lw=2, label="true")
    ax_p.plot(GRID, r["baseline"][1], "-", color="tab:red", alpha=0.8, label="baseline")
    ax_p.plot(GRID, r["rectified"][1], "-", color="tab:green", label="rectified")
    ax_p.plot(GRID, r["Sill"][1], "-", color="tab:purple", alpha=0.8, label="Sill")
    ax_p.set_title("pdf = dF/dx"); ax_p.legend(fontsize=8)
    fig.suptitle(f"Monotonicity enforcement on the violating regime "
                 f"(symmetric bimodal, n={N}, width {WIDTH})")
    fig.tight_layout(); _save(fig, "mlp_monotonicity.png")


def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
