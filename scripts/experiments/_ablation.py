"""Inner-ablation engine, shared by the per-distribution Stage scripts (A, B, ...).

Constraint-correct throughout: the MLP trains only on ``(x_i, F_hat(x_i))`` at the n samples
(``make_sample_training_set``); the dense grid + analytic truth are the referee. ``run`` varies one
knob at a time off a baseline config and sweeps E1 capacity, E2 lr/epochs, E3 bandwidth, E4 sample
count (optionally averaged over seeds), E5 monotonicity, then writes ``results/<out_name>.{png,json}``
and prints tables. The Stage scripts are thin callers that only supply the distribution and baseline.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from parzen_cdf import metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig,
    density_from_cdf,
    make_sample_training_set,
    set_seed,
    train_cdf,
)

RESULTS_DIR = "results"
WIDTHS = (4, 8, 16, 32, 64, 128)
LRS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
EPOCHS_GRID = (1000, 2000, 5000, 10000)
BW_SCALES = (0.3, 0.5, 0.7, 1.0, 1.5)
NS = (250, 500, 1000, 2000, 4000, 8000)


def _train_and_score(mix, samples, h, hidden, lr, epochs, mono, soft, grid, grid_t, true_cdf, true_pdf, seed):
    set_seed(seed)
    inputs, targets = make_sample_training_set(samples, h)
    model = CDFNet(in_dim=1, hidden_sizes=hidden, monotone=mono)
    model, _ = train_cdf(model, inputs, targets, TrainConfig(epochs=epochs, lr=lr, monotonicity_weight=soft, seed=seed))
    learned_cdf = model(grid_t).detach().numpy()
    learned_pdf = density_from_cdf(model, grid_t, clamp=True).detach().numpy()
    mass = metrics.integrates_to_one(learned_pdf, grid)
    pdf_norm = learned_pdf / mass if mass > 1e-9 else learned_pdf
    return {
        "ks": metrics.ks_distance(true_cdf, learned_cdf),
        "mse_pdf": metrics.mse(true_pdf, learned_pdf),
        "mse_pdf_norm": metrics.mse(true_pdf, pdf_norm),
        "viol": metrics.monotonicity_violation_fraction(learned_cdf),
        "mass": mass,
        "train_mse": float(torch.mean((model(inputs).detach() - targets) ** 2)),
    }


def run(name, mix, baseline, out_name, grid_lo=None, grid_hi=None, e4_seeds=(0,), seed=0):
    """Run the full inner ablation for one distribution. ``baseline`` keys: n, hidden, lr, epochs,
    mono, soft. Grid defaults to [min(mu-5sigma), max(mu+5sigma)] (covers the saturating tails)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if grid_lo is None:
        grid_lo = float((mix.means - 5 * mix.stds).min())
    if grid_hi is None:
        grid_hi = float((mix.means + 5 * mix.stds).max())
    grid = np.linspace(grid_lo, grid_hi, 2000)
    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    true_cdf, true_pdf = mix.cdf(grid), mix.pdf(grid)

    b = baseline
    rng = np.random.default_rng(seed)
    s = mix.sample(b["n"], rng)
    h0 = parzen.silverman_bandwidth(s)
    pks = metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, s, h0))
    pmse = metrics.mse(true_pdf, parzen.parzen_pdf(grid, s, h0))
    print(f"\n##### {name} | n={b['n']} | Silverman h={h0:.4f} | grid [{grid_lo:.1f},{grid_hi:.1f}] | "
          f"baseline hidden={b['hidden']} lr={b['lr']:g} epochs={b['epochs']} mono={b['mono']} soft={b['soft']:g} | "
          f"parzen ref KS={pks:.4f} MSE={pmse:.5f}")

    def ts(samples, h, hidden=None, lr=None, epochs=None, mono=None, soft=None, sd=seed):
        return _train_and_score(
            mix, samples, h,
            b["hidden"] if hidden is None else hidden,
            b["lr"] if lr is None else lr,
            b["epochs"] if epochs is None else epochs,
            b["mono"] if mono is None else mono,
            b["soft"] if soft is None else soft,
            grid, grid_t, true_cdf, true_pdf, sd)

    def axis(title, rows):
        print(f"\n=== {name}: {title} ===")
        print(f"{'config':<22}{'KS':>8}{'MSE':>10}{'MSEn':>10}{'viol%':>8}{'mass':>8}")
        out = {}
        for label, r in rows:
            out[label] = r
            print(f"{label:<22}{r['ks']:>8.4f}{r['mse_pdf']:>10.5f}{r['mse_pdf_norm']:>10.5f}"
                  f"{100 * r['viol']:>7.2f}%{r['mass']:>8.4f}")
        return out

    results = {"meta": {"name": name, "h_silverman": float(h0), "n": b["n"], "parzen_ks": pks,
                        "parzen_mse": pmse, "grid": [grid_lo, grid_hi],
                        "baseline": {k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()}}}

    results["E1_capacity"] = axis("E1 capacity", (
        [(f"width={w}", ts(s, h0, hidden=(w,))) for w in WIDTHS]
        + [("depth=2 (32,32)", ts(s, h0, hidden=(32, 32))),
           ("depth=3 (32,32,32)", ts(s, h0, hidden=(32, 32, 32)))]))

    results["E2a_lr"] = axis("E2a learning rate", [(f"lr={lr:g}", ts(s, h0, lr=lr)) for lr in LRS])
    results["E2b_epochs"] = axis("E2b epochs", [(f"epochs={e}", ts(s, h0, epochs=e)) for e in EPOCHS_GRID])

    selectors = {
        "var_matched": parzen.variance_matched_bandwidth(s),
        "likelihood_cv": parzen.likelihood_cv_bandwidth(s),
        "lscv": parzen.lscv_bandwidth(s),
        "adaptive": parzen.adaptive_bandwidths(s),
    }
    e3_h = {f"scale={sc:g}x": float(sc * h0) for sc in BW_SCALES}
    e3_h.update({k: float(np.mean(v)) for k, v in selectors.items()})
    e3 = axis("E3 bandwidth (target quality)",
              [(f"scale={sc:g}x", ts(s, sc * h0)) for sc in BW_SCALES]
              + [(k, ts(s, v)) for k, v in selectors.items()])
    results["E3_bandwidth"] = {lbl: {**r, "h": e3_h[lbl]} for lbl, r in e3.items()}

    print(f"\n=== {name}: E4 sample count (mean over {len(e4_seeds)} seed(s)) ===")
    print(f"{'config':<22}{'KS':>8}{'parzenKS':>10}{'mass':>8}")
    e4 = {}
    for n in NS:
        runs, pks_n = [], []
        for sd in e4_seeds:
            sn = mix.sample(n, np.random.default_rng(sd))
            hn = parzen.silverman_bandwidth(sn)
            runs.append(ts(sn, hn, sd=sd))
            pks_n.append(metrics.ks_distance(true_cdf, parzen.parzen_cdf(grid, sn, hn)))
        mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        mean["parzen_ks"] = float(np.mean(pks_n))
        e4[f"n={n}"] = mean
        print(f"{'n=' + str(n):<22}{mean['ks']:>8.4f}{mean['parzen_ks']:>10.4f}{mean['mass']:>8.4f}")
    results["E4_samples"] = e4

    results["E5_monotonicity"] = axis("E5 monotonicity", [
        ("baseline", ts(s, h0, mono=False, soft=0.0)),
        ("soft lambda=1", ts(s, h0, mono=False, soft=1.0)),
        ("soft lambda=5", ts(s, h0, mono=False, soft=5.0)),
        ("soft lambda=20", ts(s, h0, mono=False, soft=20.0)),
        ("sill", ts(s, h0, mono=True, soft=0.0)),
    ])

    with open(os.path.join(RESULTS_DIR, f"{out_name}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {os.path.join(RESULTS_DIR, out_name)}.json")
    _figure(results, out_name, name, len(e4_seeds))
    return results


def _figure(results, out_name, name, n_seeds):
    pks = results["meta"]["parzen_ks"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    DIVERGED = 0.5  # KS at/above this = a diverged run; keep it out of the line, annotate instead
    w_ks = [(w, results["E1_capacity"][f"width={w}"]["ks"]) for w in WIDTHS]
    conv_w = [(w, k) for w, k in w_ks if k < DIVERGED]
    ax[0, 0].plot([w for w, _ in conv_w], [k for _, k in conv_w], "o-", label="width (depth 1)")
    depths = {"depth=2 (32,32)": results["E1_capacity"]["depth=2 (32,32)"]["ks"],
              "depth=3 (32,32,32)": results["E1_capacity"]["depth=3 (32,32,32)"]["ks"]}
    dconv = [v for v in depths.values() if v < DIVERGED]
    ax[0, 0].scatter([32] * len(dconv), dconv, color="tab:red", marker="s", zorder=5, label="depth 2 / 3 @32")
    ax[0, 0].axhline(pks, ls="--", color="gray", label="parzen")
    ax[0, 0].set_xscale("log", base=2); ax[0, 0].set_xticks(WIDTHS); ax[0, 0].set_xticklabels(WIDTHS)
    diverged = [f"w={w}" for w, k in w_ks if k >= DIVERGED] + [l for l, v in depths.items() if v >= DIVERGED]
    if diverged:
        ax[0, 0].annotate("diverged: " + ", ".join(diverged), (0.5, 0.95), xycoords="axes fraction",
                          fontsize=6, color="red", ha="center")
    ax[0, 0].set_title("E1 capacity: KS vs width"); ax[0, 0].set_xlabel("hidden width"); ax[0, 0].set_ylabel("KS"); ax[0, 0].legend(fontsize=7)

    ax[0, 1].plot(LRS, [results["E2a_lr"][f"lr={lr:g}"]["ks"] for lr in LRS], "o-")
    ax[0, 1].axhline(pks, ls="--", color="gray"); ax[0, 1].set_xscale("log")
    ax[0, 1].set_title("E2a: KS vs learning rate"); ax[0, 1].set_xlabel("lr"); ax[0, 1].set_ylabel("KS")

    ax[0, 2].plot(EPOCHS_GRID, [results["E2b_epochs"][f"epochs={e}"]["ks"] for e in EPOCHS_GRID], "o-")
    ax[0, 2].axhline(pks, ls="--", color="gray")
    ax[0, 2].set_title("E2b: KS vs epochs"); ax[0, 2].set_xlabel("epochs"); ax[0, 2].set_ylabel("KS")

    e3 = results["E3_bandwidth"]
    ax[1, 0].scatter([e3[l]["h"] for l in e3], [e3[l]["ks"] for l in e3], color="tab:blue")
    for l in e3:
        ax[1, 0].annotate(l.replace("scale=", "").replace("x", ""), (e3[l]["h"], e3[l]["ks"]), fontsize=6)
    ax[1, 0].axhline(pks, ls="--", color="gray", label="parzen@silverman")
    ax[1, 0].set_title("E3 bandwidth: KS vs h"); ax[1, 0].set_xlabel("bandwidth h"); ax[1, 0].set_ylabel("KS"); ax[1, 0].legend(fontsize=7)

    ax[1, 1].plot(NS, [results["E4_samples"][f"n={n}"]["ks"] for n in NS], "o-", label="neural")
    ax[1, 1].plot(NS, [results["E4_samples"][f"n={n}"]["parzen_ks"] for n in NS], "s--", color="gray", label="parzen")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_title(f"E4: KS vs n (mean/{n_seeds} seed)"); ax[1, 1].set_xlabel("n"); ax[1, 1].set_ylabel("KS"); ax[1, 1].legend(fontsize=7)

    e5 = results["E5_monotonicity"]
    labels = list(e5)
    ax[1, 2].bar(range(len(labels)), [e5[l]["ks"] for l in labels], color="tab:purple")
    ax[1, 2].axhline(pks, ls="--", color="gray")
    ax[1, 2].set_xticks(range(len(labels))); ax[1, 2].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax[1, 2].set_title("E5 monotonicity: KS by strategy"); ax[1, 2].set_ylabel("KS")

    fig.suptitle(f"{name}: inner ablation (data points only). Gray dashed = Parzen target ceiling.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, f"{out_name}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"saved -> {path}")
