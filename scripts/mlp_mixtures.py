"""Phase B / step 3 -- the MLP on the two Gaussian mixtures, with a capacity probe.

Carry the simplest+Adam network to multimodal targets. Trained only on the data points
(x_i, F_hat(x_i)); the target is the consolidated Parzen window per budget (from Phase A step 3):
LSCV at the under-2k budget (n=1000), variance-matched at the overall budget (n=20000). We probe the
hidden width (16, 32, 64) to see whether the multimodal CDF needs more capacity than the Gaussian did,
or whether the network reaches its Parzen target regardless. Adam at a fixed lr, full batch; pdf via
the network derivative. Score = CDF gap (KS) vs the known truth; the Parzen target's gap is the ceiling.

    python scripts/mlp_mixtures.py
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
SEED = 0
LR = 0.03
EPOCHS = 5000
WIDTHS = [16, 32, 64]
BUDGETS = {"under_2k": (1000, "lscv"), "overall": (20000, "variance_matched")}
MIXES = {
    "symmetric_bimodal": data.symmetric_bimodal(),
    "asymmetric_bimodal": data.asymmetric_bimodal(),
}


def window(samples, selector):
    if selector == "lscv":
        return parzen.lscv_bandwidth(samples)
    if selector == "variance_matched":
        return parzen.variance_matched_bandwidth(samples)
    raise ValueError(selector)


def train_width(inputs, targets, grid_t, grid, true_cdf, true_pdf, width):
    set_seed(SEED)
    model = CDFNet(in_dim=1, hidden_sizes=(width,), activation="sigmoid", monotone=False)
    model, hist = train_cdf(model, inputs, targets, TrainConfig(optimizer="adam", lr=LR, epochs=EPOCHS, seed=SEED))
    cdf = model(grid_t).detach().numpy()
    pdf = density_from_cdf(model, grid_t, clamp=True).detach().numpy()
    return {"ks": metrics.ks_distance(true_cdf, cdf), "pdf_mse": metrics.mse(true_pdf, pdf),
            "viol": metrics.monotonicity_violation_fraction(cdf), "mass": metrics.integrates_to_one(pdf, grid),
            "train_mse": float(hist[-1]), "cdf": cdf, "pdf": pdf}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, mix in MIXES.items():
        lo = float((mix.means - 5 * mix.stds).min()); hi = float((mix.means + 5 * mix.stds).max())
        grid = np.linspace(lo, hi, 2000); grid_t = torch.as_tensor(grid, dtype=torch.float32)
        true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)
        print(f"\n##### {name}")
        panels = {}
        for tag, (n, selector) in BUDGETS.items():
            samples = mix.sample(n, np.random.default_rng(SEED))
            h = window(samples, selector)
            inputs, targets = make_sample_training_set(samples, h)
            parzen_cdf_g, parzen_pdf_g = parzen.parzen_cdf(grid, samples, h), parzen.parzen_pdf(grid, samples, h)
            parzen_ks = metrics.ks_distance(true_cdf, parzen_cdf_g)
            print(f"  [{tag} n={n}, target={selector}]  Parzen target KS={parzen_ks:.4f}")
            runs = {w: train_width(inputs, targets, grid_t, grid, true_cdf, true_pdf, w) for w in WIDTHS}
            for w in WIDTHS:
                r = runs[w]
                print(f"      width {w:>3}: net KS={r['ks']:.4f}  pdf MSE={r['pdf_mse']:.5f}  "
                      f"viol={100*r['viol']:.2f}%  mass={r['mass']:.4f}  trainMSE={r['train_mse']:.6f}")
            best_w = min(WIDTHS, key=lambda w: runs[w]["ks"])
            panels[tag] = dict(n=n, selector=selector, parzen_ks=parzen_ks, parzen_cdf=parzen_cdf_g,
                               parzen_pdf=parzen_pdf_g, best_w=best_w, run=runs[best_w])
            print(f"      -> best width {best_w} (net KS={runs[best_w]['ks']:.4f})")

        fig, axes = plt.subplots(len(BUDGETS), 2, figsize=(12, 8))
        for row, tag in enumerate(BUDGETS):
            p = panels[tag]; ax_c, ax_p = axes[row]
            ax_c.plot(grid, true_cdf, "k-", lw=2, label="true")
            ax_c.plot(grid, p["parzen_cdf"], "--", color="gray", label=f"Parzen target ({p['parzen_ks']:.3f})")
            ax_c.plot(grid, p["run"]["cdf"], "-", color="tab:blue", label=f"MLP w{p['best_w']} ({p['run']['ks']:.3f})")
            ax_c.set_title(f"{tag} n={p['n']} ({p['selector']}): CDF"); ax_c.legend(fontsize=8)
            ax_p.plot(grid, true_pdf, "k-", lw=2, label="true")
            ax_p.plot(grid, p["parzen_pdf"], "--", color="gray", label="Parzen target")
            ax_p.plot(grid, p["run"]["pdf"], "-", color="tab:blue", label=f"MLP w{p['best_w']} (dF/dx)")
            ax_p.set_title(f"{tag} n={p['n']}: pdf = dF/dx"); ax_p.legend(fontsize=8)
        fig.suptitle(f"Phase B step 3: MLP (sigmoid, Adam) on {name}, target = consolidated Parzen")
        fig.tight_layout(); _save(fig, f"mlp_{name}.png")


def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
