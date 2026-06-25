"""Stage A -- deep inner ablation on the single Gaussian N(0,1).

The constraint-correct regime throughout: the MLP trains ONLY on (x_i, F_hat(x_i)) at the n drawn
samples (``make_sample_training_set``); the dense grid + analytic truth are the referee only. We
vary one knob at a time off a naive baseline (width 16, lr 1e-2, 2000 epochs, Silverman h, n=2000,
unconstrained) and score against the known N(0,1):

    E1 capacity      -- hidden width, then depth at width 32
    E2a lr           -- learning rate
    E2b epochs       -- training length
    E3 bandwidth     -- Silverman / var-matched / CV / adaptive selectors + manual h scales
    E4 samples       -- n
    E5 monotonicity  -- unconstrained / soft-lambda / Sill

For the pdf we report MSE both raw and after mass-normalisation (dividing by the recovered mass),
to probe whether the neural pdf's lost mass (H6) is the accuracy bottleneck.

    python scripts/experiments/stage_a_single_gaussian.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig,
    density_from_cdf,
    make_sample_training_set,
    set_seed,
    train_cdf,
)

SEED = 0
RESULTS_DIR = "results"

# Naive baseline defaults (one knob is varied off these per axis).
BASE = dict(n=2000, hidden=(16,), lr=1e-2, epochs=2000, mono=False, soft=0.0)

MIX = data.single_gaussian()
GRID = np.linspace(-6.0, 6.0, 2000)          # fixed referee grid for N(0,1)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF, TRUE_PDF = MIX.cdf(GRID), MIX.pdf(GRID)


def baseline_samples():
    rng = np.random.default_rng(SEED)
    return MIX.sample(BASE["n"], rng)


def parzen_ref(samples, h):
    pc = parzen.parzen_cdf(GRID, samples, h)
    pp = parzen.parzen_pdf(GRID, samples, h)
    return metrics.ks_distance(TRUE_CDF, pc), metrics.mse(TRUE_PDF, pp)


def train_and_score(samples, h, hidden, lr, epochs, mono, soft):
    set_seed(SEED)
    inputs, targets = make_sample_training_set(samples, h)
    model = CDFNet(in_dim=1, hidden_sizes=hidden, monotone=mono)
    model, _ = train_cdf(model, inputs, targets, TrainConfig(epochs=epochs, lr=lr, monotonicity_weight=soft, seed=SEED))
    learned_cdf = model(GRID_T).detach().numpy()
    learned_pdf = density_from_cdf(model, GRID_T, clamp=True).detach().numpy()
    mass = metrics.integrates_to_one(learned_pdf, GRID)
    pdf_norm = learned_pdf / mass if mass > 1e-9 else learned_pdf
    return {
        "ks": metrics.ks_distance(TRUE_CDF, learned_cdf),
        "mse_pdf": metrics.mse(TRUE_PDF, learned_pdf),
        "mse_pdf_norm": metrics.mse(TRUE_PDF, pdf_norm),
        "viol": metrics.monotonicity_violation_fraction(learned_cdf),
        "mass": mass,
        "train_mse": float(torch.mean((model(inputs).detach() - targets) ** 2)),
    }


def cfg(**over):
    c = dict(BASE)
    c.update(over)
    return c


def run_axis(title, rows):
    """rows: list of (label, samples, h, cfg-dict). Returns list of (label, metrics)."""
    print(f"\n=== {title} ===")
    print(f"{'config':<24}{'KS':>8}{'MSE':>10}{'MSEn':>10}{'viol%':>8}{'mass':>8}{'trainMSE':>11}")
    out = []
    for label, samples, h, c in rows:
        r = train_and_score(samples, h, c["hidden"], c["lr"], c["epochs"], c["mono"], c["soft"])
        out.append((label, r))
        print(f"{label:<24}{r['ks']:>8.4f}{r['mse_pdf']:>10.5f}{r['mse_pdf_norm']:>10.5f}"
              f"{100 * r['viol']:>7.2f}%{r['mass']:>8.4f}{r['train_mse']:>11.6f}")
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    s = baseline_samples()
    h0 = parzen.silverman_bandwidth(s)
    pks, pmse = parzen_ref(s, h0)
    print(f"single_gaussian N(0,1) | n={BASE['n']} | Silverman h={h0:.4f} | "
          f"parzen ref: KS={pks:.4f} MSE(pdf)={pmse:.5f}")
    results = {"meta": {"h_silverman": float(h0), "n": BASE["n"], "parzen_ks": pks, "parzen_mse": pmse}}

    # E1 capacity: width sweep (depth 1), then depth sweep at width 32.
    e1 = run_axis("E1 capacity (width, depth)", [
        (f"width={w}", s, h0, cfg(hidden=(w,))) for w in (4, 8, 16, 32, 64, 128)
    ] + [
        ("depth=2 (32,32)", s, h0, cfg(hidden=(32, 32))),
        ("depth=3 (32,32,32)", s, h0, cfg(hidden=(32, 32, 32))),
    ])
    results["E1_capacity"] = {lbl: r for lbl, r in e1}

    # E2a learning rate.
    e2a = run_axis("E2a learning rate", [
        (f"lr={lr:g}", s, h0, cfg(lr=lr)) for lr in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    ])
    results["E2a_lr"] = {lbl: r for lbl, r in e2a}

    # E2b epochs.
    e2b = run_axis("E2b epochs", [
        (f"epochs={e}", s, h0, cfg(epochs=e)) for e in (1000, 2000, 5000, 10000)
    ])
    results["E2b_epochs"] = {lbl: r for lbl, r in e2b}

    # E3 bandwidth: manual scales + truth-free selectors. Each retrains on the new Parzen target.
    bw_rows = [(f"scale={sc:g}x", s, sc * h0, cfg()) for sc in (0.3, 0.5, 0.7, 1.0, 1.5)]
    selectors = {
        "var_matched": parzen.variance_matched_bandwidth(s),
        "likelihood_cv": parzen.likelihood_cv_bandwidth(s),
        "lscv": parzen.lscv_bandwidth(s),
        "adaptive": parzen.adaptive_bandwidths(s),
    }
    bw_rows += [(name, s, h, cfg()) for name, h in selectors.items()]
    e3 = run_axis("E3 bandwidth (target quality)", bw_rows)
    # record the scalar (or mean) h used for each
    e3_h = {f"scale={sc:g}x": float(sc * h0) for sc in (0.3, 0.5, 0.7, 1.0, 1.5)}
    e3_h.update({k: float(np.mean(v)) for k, v in selectors.items()})
    results["E3_bandwidth"] = {lbl: {**r, "h": e3_h[lbl]} for lbl, r in e3}

    # E4 sample count: re-sample, recompute Silverman h.
    e4_rows, e4_parzen = [], {}
    for n in (250, 500, 1000, 2000, 4000, 8000):
        rng = np.random.default_rng(SEED)
        sn = MIX.sample(n, rng)
        hn = parzen.silverman_bandwidth(sn)
        e4_rows.append((f"n={n}", sn, hn, cfg(n=n)))
        e4_parzen[f"n={n}"] = parzen_ref(sn, hn)[0]
    e4 = run_axis("E4 sample count", e4_rows)
    results["E4_samples"] = {lbl: {**r, "parzen_ks": e4_parzen[lbl]} for lbl, r in e4}

    # E5 monotonicity strategy.
    e5 = run_axis("E5 monotonicity", [
        ("baseline", s, h0, cfg()),
        ("soft lambda=1", s, h0, cfg(soft=1.0)),
        ("soft lambda=5", s, h0, cfg(soft=5.0)),
        ("soft lambda=20", s, h0, cfg(soft=20.0)),
        ("sill", s, h0, cfg(mono=True)),
    ])
    results["E5_monotonicity"] = {lbl: r for lbl, r in e5}

    with open(os.path.join(RESULTS_DIR, "stage_a_single_gaussian.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {os.path.join(RESULTS_DIR, 'stage_a_single_gaussian.json')}")
    _figure(e1, e2a, e2b, e3, e3_h, e4, e4_parzen, e5, pks)


def _figure(e1, e2a, e2b, e3, e3_h, e4, e4_parzen, e5, parzen_ks):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    widths = [4, 8, 16, 32, 64, 128]
    ax[0, 0].plot(widths, [dict(e1)[f"width={w}"]["ks"] for w in widths], "o-", label="width (depth 1)")
    ax[0, 0].axhline(parzen_ks, ls="--", color="gray", label="parzen")
    ax[0, 0].set_xscale("log", base=2); ax[0, 0].set_xticks(widths); ax[0, 0].set_xticklabels(widths)
    ax[0, 0].set_title("E1 capacity: KS vs width"); ax[0, 0].set_xlabel("hidden width"); ax[0, 0].set_ylabel("KS"); ax[0, 0].legend(fontsize=8)

    lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    ax[0, 1].plot(lrs, [dict(e2a)[f"lr={lr:g}"]["ks"] for lr in lrs], "o-")
    ax[0, 1].axhline(parzen_ks, ls="--", color="gray")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_title("E2a: KS vs learning rate"); ax[0, 1].set_xlabel("lr"); ax[0, 1].set_ylabel("KS")

    eps = [1000, 2000, 5000, 10000]
    ax[0, 2].plot(eps, [dict(e2b)[f"epochs={e}"]["ks"] for e in eps], "o-")
    ax[0, 2].axhline(parzen_ks, ls="--", color="gray")
    ax[0, 2].set_title("E2b: KS vs epochs"); ax[0, 2].set_xlabel("epochs"); ax[0, 2].set_ylabel("KS")

    e3d = dict(e3)
    hs = [e3_h[lbl] for lbl, _ in e3]
    ax[1, 0].scatter(hs, [r["ks"] for _, r in e3], color="tab:blue")
    for (lbl, r) in e3:
        ax[1, 0].annotate(lbl.replace("scale=", "").replace("x", ""), (e3_h[lbl], r["ks"]), fontsize=6)
    ax[1, 0].axhline(parzen_ks, ls="--", color="gray", label="parzen@silverman")
    ax[1, 0].set_title("E3 bandwidth: KS vs h"); ax[1, 0].set_xlabel("bandwidth h"); ax[1, 0].set_ylabel("KS"); ax[1, 0].legend(fontsize=8)

    ns = [250, 500, 1000, 2000, 4000, 8000]
    ax[1, 1].plot(ns, [dict(e4)[f"n={n}"]["ks"] for n in ns], "o-", label="neural")
    ax[1, 1].plot(ns, [e4_parzen[f"n={n}"] for n in ns], "s--", color="gray", label="parzen")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_title("E4: KS vs sample count"); ax[1, 1].set_xlabel("n"); ax[1, 1].set_ylabel("KS"); ax[1, 1].legend(fontsize=8)

    labels = [lbl for lbl, _ in e5]
    ax[1, 2].bar(range(len(labels)), [r["ks"] for _, r in e5], color="tab:purple")
    ax[1, 2].axhline(parzen_ks, ls="--", color="gray")
    ax[1, 2].set_xticks(range(len(labels))); ax[1, 2].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax[1, 2].set_title("E5 monotonicity: KS by strategy"); ax[1, 2].set_ylabel("KS")

    fig.suptitle("Stage A -- single Gaussian N(0,1): inner ablation (data points only). "
                 "Gray dashed = Parzen target ceiling.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "stage_a_single_gaussian.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
