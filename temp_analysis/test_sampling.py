"""Test E -- generative use: sampling by inverting the learned CDF (random variate generation).

A learned, monotone, compact CDF can be inverted to draw samples: x = F^{-1}(u), u ~ Uniform(0,1). This
is the random-variate-generation use highlighted by Magdon-Ismail & Atiya. We invert the net's CDF (on
a fine grid) and check the drawn samples match the true distribution (via the CDF gap of their own
empirical CDF, and a quantile check). The net gives a compact closed-form generator; we contrast with
sampling from the Parzen estimate.
"""
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed

mix = data.asymmetric_bimodal()
GRID = np.linspace(-6, 8, 4000); GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE_CDF = mix.cdf(GRID)


def invert(cdf_on_grid, u):
    """x = F^{-1}(u) by searching the monotone CDF grid."""
    cdf = np.maximum.accumulate(cdf_on_grid)            # ensure monotone
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 1, len(GRID) - 1)
    # linear interpolation between grid points for a smoother inverse
    c0, c1 = cdf[idx - 1], cdf[idx]; x0, x1 = GRID[idx - 1], GRID[idx]
    frac = np.where(c1 > c0, (u - c0) / (c1 - c0), 0.0)
    return x0 + frac * (x1 - x0)


def main():
    n_train, n_draw = 2000, 20000
    s = mix.sample(n_train, np.random.default_rng(0))
    h = parzen.silverman_bandwidth(s)

    # train the net CDF on the data points
    set_seed(0)
    inp, tgt = make_sample_training_set(s, h)
    net = CDFNet(1, (32,), activation="sigmoid", monotone=False)
    net, _ = train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=5000, seed=0))
    net_cdf = net(GRID_T).detach().numpy()

    # draw via inverse-CDF from the net, and (for contrast) from the Parzen estimate
    u = np.random.default_rng(1).uniform(size=n_draw)
    draws_net = invert(net_cdf, u)
    draws_parzen = invert(parzen.parzen_cdf(GRID, s, h), u)
    draws_true = mix.sample(n_draw, np.random.default_rng(2))

    def emp_ks(draws):
        sd = np.sort(draws)
        ecdf = np.searchsorted(sd, GRID, side="right") / len(draws)
        return metrics.ks_distance(TRUE_CDF, ecdf)

    print(f"asymmetric_bimodal: train n={n_train}, draw n={n_draw}")
    print(f"  CDF gap of the drawn samples' empirical CDF vs the TRUE CDF:")
    print(f"    draws from the NET inverse-CDF    : {emp_ks(draws_net):.4f}")
    print(f"    draws from the Parzen inverse-CDF : {emp_ks(draws_parzen):.4f}")
    print(f"    a fresh true sample (reference)   : {emp_ks(draws_true):.4f}")
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"  quantiles {qs}:")
    print(f"    true : {np.round(np.quantile(draws_true, qs), 3)}")
    print(f"    net  : {np.round(np.quantile(draws_net, qs), 3)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(draws_true, bins=80, density=True, alpha=0.4, color="gray", label="true samples")
    ax.hist(draws_net, bins=80, density=True, histtype="step", color="tab:blue", lw=1.6,
            label="net inverse-CDF draws")
    ax.plot(GRID, mix.pdf(GRID), "k-", lw=1.5, label="true pdf")
    ax.set_title("Sampling by inverting the learned net CDF"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("temp_analysis/test_sampling.png", dpi=120); plt.close(fig)
    print("  figure -> temp_analysis/test_sampling.png")


if __name__ == "__main__":
    main()
