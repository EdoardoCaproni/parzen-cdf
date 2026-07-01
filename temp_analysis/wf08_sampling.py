"""wf08 -- inverse-CDF sampling quality: is the net a faithful, compact generator?

Random-variate generation by inverting a learned monotone CDF: x = F^{-1}(u), u ~ U(0,1).
We draw 20000 samples by inverting (a) the net CDF and (b) the Parzen CDF on the asymmetric
bimodal, and ask whether the drawn samples' empirical CDF matches the TRUE CDF. A fresh true
sample of the same size is the irreducible-noise reference. We average over a few seeds so the
verdict is not one lucky draw, and we report the generator's storage footprint (net params vs
n stored samples) because compactness is the only thing the net could plausibly buy here.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # ponytail: box is oversubscribed; 1 thread/proc avoids thrash
import numpy as np
import torch

torch.set_num_threads(1)

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, set_seed, train_cdf

mix = data.asymmetric_bimodal()
GRID = np.linspace(-6, 8, 4000)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF = mix.cdf(GRID)
QS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
TRUE_Q = mix.cdf_inverse(QS) if hasattr(mix, "cdf_inverse") else None


def invert(cdf_on_grid, u):
    """x = F^{-1}(u) on the monotone CDF grid, linear-interpolated."""
    cdf = np.maximum.accumulate(cdf_on_grid)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    idx = np.clip(np.searchsorted(cdf, u), 1, len(GRID) - 1)
    c0, c1 = cdf[idx - 1], cdf[idx]
    x0, x1 = GRID[idx - 1], GRID[idx]
    frac = np.where(c1 > c0, (u - c0) / (c1 - c0), 0.0)
    return x0 + frac * (x1 - x0)


def emp_ks(draws):
    sd = np.sort(draws)
    ecdf = np.searchsorted(sd, GRID, side="right") / len(draws)
    return metrics.ks_distance(TRUE_CDF, ecdf)


def true_quantiles():
    # invert the true CDF on the grid for a quantile ground truth
    return invert(TRUE_CDF, QS)


def main():
    n_train, n_draw = 2000, 20000
    seeds = [0, 1, 2]
    tq = true_quantiles()

    rows = {"net": [], "parzen": [], "true_ref": []}
    qabs = {"net": [], "parzen": []}
    net_params = None
    plot = {}
    for sd in seeds:
        s = mix.sample(n_train, np.random.default_rng(sd))
        h = parzen.silverman_bandwidth(s)

        set_seed(sd)
        inp, tgt = make_sample_training_set(s, h)
        net = CDFNet(1, (32,), activation="sigmoid", monotone=True)
        net, _ = train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=8000, seed=sd))
        if net_params is None:
            net_params = sum(p.numel() for p in net.parameters())
        net_cdf = net(GRID_T).detach().numpy()
        parzen_cdf = parzen.parzen_cdf(GRID, s, h)

        # use the SAME uniforms for net and parzen so the comparison is variance-free
        u = np.random.default_rng(100 + sd).uniform(size=n_draw)
        draws_net = invert(net_cdf, u)
        draws_parzen = invert(parzen_cdf, u)
        draws_true = mix.sample(n_draw, np.random.default_rng(200 + sd))

        rows["net"].append(emp_ks(draws_net))
        rows["parzen"].append(emp_ks(draws_parzen))
        rows["true_ref"].append(emp_ks(draws_true))
        qabs["net"].append(np.abs(np.quantile(draws_net, QS) - tq))
        qabs["parzen"].append(np.abs(np.quantile(draws_parzen, QS) - tq))
        if sd == seeds[0]:
            plot["net"], plot["true"] = draws_net, draws_true
        print(f"  seed {sd} done", flush=True)

    def ms(a):
        a = np.asarray(a)
        return a.mean(), a.std()

    print(f"asymmetric_bimodal  train n={n_train}  draw n={n_draw}  seeds={seeds}")
    print("KS gap (drawn empirical CDF vs TRUE CDF), mean +/- std over seeds:")
    print(f"  net    inverse-CDF : {ms(rows['net'])[0]:.4f} +/- {ms(rows['net'])[1]:.4f}")
    print(f"  parzen inverse-CDF : {ms(rows['parzen'])[0]:.4f} +/- {ms(rows['parzen'])[1]:.4f}")
    print(f"  fresh true sample  : {ms(rows['true_ref'])[0]:.4f} +/- {ms(rows['true_ref'])[1]:.4f}  (irreducible)")

    qn = np.mean(qabs["net"], axis=0)
    qp = np.mean(qabs["parzen"], axis=0)
    print(f"\nMean |quantile error| at {list(QS)}:")
    print(f"  true quantiles : {np.round(tq, 3)}")
    print(f"  net    err     : {np.round(qn, 4)}  (avg {qn.mean():.4f})")
    print(f"  parzen err     : {np.round(qp, 4)}  (avg {qp.mean():.4f})")

    print(f"\nGenerator footprint (storage to reproduce the sampler):")
    print(f"  net    : {net_params} float params")
    print(f"  parzen : {n_train} stored samples + 1 bandwidth = {n_train + 1} floats")
    print(f"  compression factor net vs parzen: {(n_train + 1) / net_params:.1f}x fewer floats")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    draws_net, draws_true = plot["net"], plot["true"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(draws_true, bins=90, density=True, alpha=0.4, color="gray", label="true samples")
    ax.hist(draws_net, bins=90, density=True, histtype="step", color="tab:blue", lw=1.6, label="net inverse-CDF draws")
    ax.plot(GRID, mix.pdf(GRID), "k-", lw=1.5, label="true pdf")
    ax.set_title("Sampling by inverting the learned monotone net CDF")
    ax.legend(fontsize=8)
    ax.set_xlim(-5, 7)
    fig.tight_layout()
    fig.savefig("temp_analysis/wf08_sampling.png", dpi=120)
    plt.close(fig)
    print("\nfigure -> temp_analysis/wf08_sampling.png")


if __name__ == "__main__":
    main()
