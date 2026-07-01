"""Test C -- the multivariate crux: a smooth differentiable joint density via mixed partials.

2D Gaussian mixture (known joint pdf). Train a net F: R^2 -> (0,1) on the data points only, with labels
= the joint Parzen CDF there (product logistic kernel). Recover the joint density as the MIXED partial
d^2F/dx dy via autograd. Compare to the product-kernel KDE and the truth.

Three points this makes:
  1. the net yields a smooth, differentiable joint density (mixed partial); the joint EMPIRICAL CDF is
     a step function whose mixed partial is a sum of point masses (spikes), useless as a density;
  2. the net's query cost is O(params), independent of n; the KDE density is O(n) per query point;
  3. accuracy: net density vs KDE density vs truth (MSE on a grid).
"""
import time

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.special import expit
import torch

from parzen_cdf import parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, train_cdf, set_seed

# --- a known 2D Gaussian mixture ----------------------------------------------------------------
W = np.array([0.4, 0.35, 0.25])
MU = np.array([[-1.5, -1.0], [1.5, 1.0], [0.0, 1.8]])
COV = np.array([[[0.6, 0], [0, 0.8]], [[0.7, 0], [0, 0.5]], [[0.4, 0], [0, 0.4]]])


def true_pdf(xy):
    return sum(w * mvn(mean=m, cov=c).pdf(xy) for w, m, c in zip(W, MU, COV))


def sample(n, rng):
    comp = rng.choice(3, size=n, p=W)
    return np.stack([rng.multivariate_normal(MU[k], COV[k]) for k in comp])


def joint_parzen_cdf(q, s, hx, hy):
    sx = expit((q[:, 0, None] - s[:, 0][None, :]) / hx)
    sy = expit((q[:, 1, None] - s[:, 1][None, :]) / hy)
    return (sx * sy).mean(axis=1)


def joint_kde_pdf(q, s, hx, hy):
    zx = (q[:, 0, None] - s[:, 0][None, :]) / hx; kx = expit(zx) * (1 - expit(zx)) / hx
    zy = (q[:, 1, None] - s[:, 1][None, :]) / hy; ky = expit(zy) * (1 - expit(zy)) / hy
    return (kx * ky).mean(axis=1)


def net_mixed_partial(model, xy_np):
    xy = torch.tensor(xy_np, dtype=torch.float32, requires_grad=True)
    f = model(xy)
    g = torch.autograd.grad(f.sum(), xy, create_graph=True)[0]   # [dF/dx, dF/dy]
    gxy = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=False)[0][:, 1]  # d2F/dxdy
    return gxy.clamp_min(0).detach().numpy()


def main():
    n = 2000
    rng = np.random.default_rng(0)
    s = sample(n, rng)
    hx, hy = parzen.silverman_bandwidth(s[:, 0]), parzen.silverman_bandwidth(s[:, 1])

    # labels = joint Parzen CDF at the data points; train the net
    labels = joint_parzen_cdf(s, s, hx, hy)
    set_seed(0)
    net = CDFNet(in_dim=2, hidden_sizes=(64, 64), activation="sigmoid", monotone=False)
    inp = torch.as_tensor(s, dtype=torch.float32); tgt = torch.as_tensor(labels, dtype=torch.float32)
    net, _ = train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=8000, seed=0))

    # evaluation grid
    gx = np.linspace(-4, 4, 80); gy = np.linspace(-4, 4, 80)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    true_d = true_pdf(grid)
    kde_d = joint_kde_pdf(grid, s, hx, hy)
    net_d = net_mixed_partial(net, grid)

    mse_kde = float(np.mean((true_d - kde_d) ** 2))
    mse_net = float(np.mean((true_d - net_d) ** 2))

    # query cost on the grid
    t_kde = time.perf_counter(); [joint_kde_pdf(grid, s, hx, hy) for _ in range(5)]; t_kde = (time.perf_counter() - t_kde) / 5
    t_net = time.perf_counter(); [net_mixed_partial(net, grid) for _ in range(5)]; t_net = (time.perf_counter() - t_net) / 5
    n_params = sum(p.numel() for p in net.parameters())

    print(f"2D mixture, n={n}, windows hx={hx:.3f} hy={hy:.3f}")
    print(f"  joint density MSE vs truth:  KDE {mse_kde:.5f}   net(mixed partial) {mse_net:.5f}")
    print(f"  grid query time ({grid.shape[0]} pts):  KDE {1000*t_kde:.1f} ms (O(n))   "
          f"net {1000*t_net:.1f} ms (O(params), {n_params} params)")
    print(f"  net stores {n_params} floats; KDE stores n={n} points (and is O(n) per query)")
    print("  note: the joint EMPIRICAL CDF's mixed partial is a sum of point masses (no usable "
          "density); the net (or product-Parzen) gives a smooth one.")

    # heatmaps
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (title, d) in zip(ax, [("true density", true_d), ("product KDE", kde_d),
                                  ("net: mixed partial d2F/dxdy", net_d)]):
        a.contourf(XX, YY, d.reshape(XX.shape), levels=20, cmap="viridis")
        a.scatter(s[:, 0], s[:, 1], s=2, c="white", alpha=0.15)
        a.set_title(title); a.set_xlim(-4, 4); a.set_ylim(-4, 4)
    fig.suptitle(f"2D joint density: truth vs product-KDE vs net mixed partial "
                 f"(MSE: KDE {mse_kde:.4f}, net {mse_net:.4f})")
    fig.tight_layout(); fig.savefig("temp_analysis/test_2d.png", dpi=120); plt.close(fig)
    print("  figure -> temp_analysis/test_2d.png")


if __name__ == "__main__":
    main()
