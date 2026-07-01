"""wf03 -- 3D joint density via the THIRD mixed partial d3F/dxdydz vs a product-KDE.

Extends the 2D mixed-partial idea (test_2d.py) to 3D. A net F: R^3 -> (0,1) is trained ONLY on the
n data points, label = the joint product-logistic Parzen CDF there. The joint density is recovered
as d3F/dx dy dz via three nested autograd passes. Compared to a 3D product-logistic KDE and the truth
on a coarse 20^3 grid.

The point of the angle is the structural one: the joint EMPIRICAL CDF (the multivariate step function
1/n * sum_i 1[x>=x_i componentwise]) has a third mixed partial that is a sum of Dirac point masses --
it is NOT a usable density. Smoothing (Parzen, or the smooth net surrogate) is what makes a
differentiable joint CDF, and the net then gives the density at O(params) per query, independent of n.
"""
import time

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.special import expit
import torch

from parzen_cdf import parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, train_cdf, set_seed

# --- a known 3D Gaussian mixture (diagonal covariances; product-form is exact per-component) -----
W = np.array([0.45, 0.35, 0.20])
MU = np.array([[-1.5, -1.0, -0.8], [1.5, 1.0, 1.2], [0.0, 1.8, -1.5]])
COV = np.array([
    [[0.6, 0, 0], [0, 0.8, 0], [0, 0, 0.5]],
    [[0.7, 0, 0], [0, 0.5, 0], [0, 0, 0.6]],
    [[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.7]],
])


def true_pdf(xyz):
    return sum(w * mvn(mean=m, cov=c).pdf(xyz) for w, m, c in zip(W, MU, COV))


def sample(n, rng):
    comp = rng.choice(3, size=n, p=W)
    return np.stack([rng.multivariate_normal(MU[k], COV[k]) for k in comp])


def joint_parzen_cdf(q, s, h):
    """Product-logistic joint CDF: F(q) = (1/n) sum_i prod_d sigmoid((q_d - s_id)/h_d)."""
    prod = np.ones((q.shape[0], s.shape[0]))
    for d in range(3):
        prod = prod * expit((q[:, d, None] - s[:, d][None, :]) / h[d])
    return prod.mean(axis=1)


def joint_kde_pdf(q, s, h):
    """Product-logistic KDE: f(q) = (1/n) sum_i prod_d k_d, k_d = sigmoid(z)(1-sigmoid(z))/h_d."""
    prod = np.ones((q.shape[0], s.shape[0]))
    for d in range(3):
        z = (q[:, d, None] - s[:, d][None, :]) / h[d]
        sg = expit(z)
        prod = prod * (sg * (1 - sg) / h[d])
    return prod.mean(axis=1)


def net_mixed_partial3(model, xyz_np, batch=2000):
    """d3F/dx dy dz via three nested autograd passes, batched to bound memory."""
    out = np.empty(xyz_np.shape[0])
    for b in range(0, xyz_np.shape[0], batch):
        chunk = xyz_np[b:b + batch]
        xyz = torch.tensor(chunk, dtype=torch.float32, requires_grad=True)
        f = model(xyz)
        gx = torch.autograd.grad(f.sum(), xyz, create_graph=True)[0][:, 0]          # dF/dx
        gxy = torch.autograd.grad(gx.sum(), xyz, create_graph=True)[0][:, 1]        # d2F/dxdy
        gxyz = torch.autograd.grad(gxy.sum(), xyz, create_graph=False)[0][:, 2]     # d3F/dxdydz
        out[b:b + batch] = gxyz.clamp_min(0).detach().numpy()
    return out


def empirical_cdf(q, s):
    """Multivariate empirical CDF: (1/n) sum_i 1[q >= s_i componentwise]. A step function."""
    ge = (q[:, None, :] >= s[None, :, :]).all(axis=2)
    return ge.mean(axis=1)


def main():
    n = 1000
    rng = np.random.default_rng(0)
    s = sample(n, rng)
    h = np.array([parzen.silverman_bandwidth(s[:, d]) for d in range(3)])

    # labels = joint product-Parzen CDF at the data points; train the net (constraint-correct)
    labels = joint_parzen_cdf(s, s, h)
    set_seed(0)
    net = CDFNet(in_dim=3, hidden_sizes=(64, 64), activation="sigmoid", monotone=False)
    inp = torch.as_tensor(s, dtype=torch.float32)
    tgt = torch.as_tensor(labels, dtype=torch.float32)
    net, hist = train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=2500, seed=0))

    # coarse 20^3 evaluation grid
    g = np.linspace(-4, 4, 20)
    XX, YY, ZZ = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)  # 8000 points

    true_d = true_pdf(grid)
    kde_d = joint_kde_pdf(grid, s, h)
    net_d = net_mixed_partial3(net, grid)

    mse_kde = float(np.mean((true_d - kde_d) ** 2))
    mse_net = float(np.mean((true_d - net_d) ** 2))

    # also report how well the net surrogate matches its OWN Parzen target's CDF (faithful regressor?)
    parzen_cdf_grid = joint_parzen_cdf(grid, s, h)
    with torch.no_grad():
        net_cdf_grid = net(torch.as_tensor(grid, dtype=torch.float32)).numpy()
    cdf_mse_net_vs_parzen = float(np.mean((net_cdf_grid - parzen_cdf_grid) ** 2))
    cdf_mse_parzen_vs_true = float(np.mean((parzen_cdf_grid - _true_cdf(grid)) ** 2))

    # density that the Parzen CDF itself implies (the smooth product KDE IS its mixed partial),
    # so net-vs-KDE is really net-surrogate vs its analytic target's density.

    # query cost on a 1000-point subgrid (1 rep; triple-autograd is heavy, this is representative)
    qgrid = grid[:1000]
    t_kde = time.perf_counter(); joint_kde_pdf(qgrid, s, h); t_kde = time.perf_counter() - t_kde
    t_net = time.perf_counter(); net_mixed_partial3(net, qgrid); t_net = time.perf_counter() - t_net
    n_params = sum(p.numel() for p in net.parameters())

    # empirical-CDF demonstration: its mixed partial is spikes. Show that finite differencing it on the
    # grid gives a near-all-zero, occasionally-spiking array (n nonzero cells out of 20^3), not a density.
    emp = empirical_cdf(grid, s).reshape(XX.shape)
    # third mixed finite difference (a discrete d3) -- count nonzero, compare integral behaviour
    d3 = np.diff(np.diff(np.diff(emp, axis=0), axis=1), axis=2)
    nonzero_cells = int(np.sum(np.abs(d3) > 1e-12))
    total_cells = d3.size

    print(f"3D mixture, n={n}, bandwidths h={np.round(h,3)}")
    print(f"  final train loss (MSE on Parzen CDF labels): {hist[-1]:.2e}")
    print(f"  grid {grid.shape[0]} pts (20^3)")
    print(f"  JOINT DENSITY MSE vs truth:   product-KDE {mse_kde:.6e}   net(d3F/dxdydz) {mse_net:.6e}")
    print(f"  ratio net/KDE density MSE: {mse_net / mse_kde:.3f}")
    print(f"  CDF MSE: net-surrogate vs its Parzen target {cdf_mse_net_vs_parzen:.6e}; "
          f"Parzen target vs true CDF {cdf_mse_parzen_vs_true:.6e}")
    print(f"  density integrates (Riemann sum, dx^3): truth {_riemann(true_d,g):.3f}  "
          f"KDE {_riemann(kde_d,g):.3f}  net {_riemann(net_d,g):.3f}")
    print(f"  QUERY COST ({qgrid.shape[0]} pts):  KDE {1000*t_kde:.1f} ms (O(n), n={n})   "
          f"net {1000*t_net:.1f} ms (O(params))")
    print(f"  net stores {n_params} floats (O(params)); KDE stores n*d = {n*3} floats and is O(n) per query")
    print(f"  EMPIRICAL CDF mixed partial (discrete d3): {nonzero_cells}/{total_cells} cells nonzero "
          f"(spikes at data, ~0 everywhere) -- NOT a usable density")
    print(f"  max |discrete d3 of empirical CDF| = {np.abs(d3).max():.4f} (a spike); "
          f"true density max on grid = {true_d.max():.4f}")

    return dict(mse_kde=mse_kde, mse_net=mse_net, t_kde=t_kde, t_net=t_net, n_params=n_params,
                cdf_mse_net_vs_parzen=cdf_mse_net_vs_parzen, nonzero_cells=nonzero_cells,
                total_cells=total_cells, n=n)


def _true_cdf(q):
    """True joint CDF of the diagonal mixture: sum_k w_k prod_d Phi((q_d-mu_kd)/sigma_kd)."""
    from scipy.stats import norm
    out = np.zeros(q.shape[0])
    for w, m, c in zip(W, MU, COV):
        p = np.ones(q.shape[0])
        for d in range(3):
            p = p * norm.cdf((q[:, d] - m[d]) / np.sqrt(c[d][d]))
        out = out + w * p
    return out


def _riemann(vals, g):
    dx = g[1] - g[0]
    return float(vals.sum() * dx ** 3)


if __name__ == "__main__":
    main()
