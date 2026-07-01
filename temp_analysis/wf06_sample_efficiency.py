"""wf06 -- sample efficiency at very small n.

Question: at n in {25,50,100,200}, does the smooth MLP-on-Parzen generalize
better than the raw Parzen CDF (lower CDF gap to truth)? Averaged over seeds,
on a single Gaussian and a symmetric bimodal.

Everything is fit ONLY from the n samples:
  - Silverman bandwidth h from the n samples
  - Parzen CDF target built from the n samples at h
  - net trains ONLY on (x_i, Fhat(x_i)) for the n sample points
We then evaluate KS (sup gap) and MSE of the CDF on a dense truth grid, and
also report the empirical CDF as the dumbest baseline.
"""
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed

GRID = np.linspace(-6, 6, 1200)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
NS = [25, 50, 100, 200]
SEEDS = range(8)
EPOCHS = 2000  # small n -> fast; keeps it light


def empirical_cdf(samples, grid):
    s = np.sort(samples)
    # fraction of samples <= grid point
    return np.searchsorted(s, grid, side="right") / len(s)


def net_cdf(samples, h, seed):
    set_seed(seed)
    inp, tgt = make_sample_training_set(samples, h)
    m = CDFNet(1, (32,), activation="sigmoid", monotone=False)
    m, _ = train_cdf(m, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=EPOCHS, seed=seed))
    return m(GRID_T).detach().numpy()


def run(mix, name):
    true = mix.cdf(GRID)
    print(f"\n=== {name} ===")
    print(f"  {'n':>5} | {'emp KS':>8}{'par KS':>8}{'net KS':>8} | "
          f"{'emp MSE':>9}{'par MSE':>9}{'net MSE':>9}")
    rows = []
    for n in NS:
        emp_ks, par_ks, net_ks = [], [], []
        emp_ms, par_ms, net_ms = [], [], []
        for seed in SEEDS:
            s = mix.sample(n, np.random.default_rng(1000 + seed))
            h = parzen.silverman_bandwidth(s)
            ce = empirical_cdf(s, GRID)
            cp = parzen.parzen_cdf(GRID, s, h)
            cn = net_cdf(s, h, seed)
            emp_ks.append(metrics.ks_distance(true, ce))
            par_ks.append(metrics.ks_distance(true, cp))
            net_ks.append(metrics.ks_distance(true, cn))
            emp_ms.append(metrics.mse(true, ce))
            par_ms.append(metrics.mse(true, cp))
            net_ms.append(metrics.mse(true, cn))
        print(f"  {n:>5} | {np.mean(emp_ks):>8.4f}{np.mean(par_ks):>8.4f}{np.mean(net_ks):>8.4f} | "
              f"{np.mean(emp_ms):>9.5f}{np.mean(par_ms):>9.5f}{np.mean(net_ms):>9.5f}")
        rows.append((n, np.mean(par_ks), np.mean(net_ks), np.std(net_ks),
                     np.mean(par_ms), np.mean(net_ms)))
    return rows


def verdict(rows, name):
    print(f"\n  verdict ({name}): net vs Parzen on KS")
    for n, pk, nk, nks, pm, nm in rows:
        d_ks = 100 * (pk - nk) / pk
        tag = "net better" if nk < pk - 1e-4 else ("worse" if nk > pk + 1e-4 else "tie")
        print(f"    n={n:>4}: parzen={pk:.4f} net={nk:.4f} ({d_ks:+.1f}% KS) [{tag}]  "
              f"MSE par={pm:.5f} net={nm:.5f}")


if __name__ == "__main__":
    rows_g = run(data.single_gaussian(), "single_gaussian")
    verdict(rows_g, "single_gaussian")
    rows_b = run(data.symmetric_bimodal(), "symmetric_bimodal")
    verdict(rows_b, "symmetric_bimodal")
