"""wf09_copula: Sklar/copula decomposition in 2D.

Question: in 2D, is it better to estimate the joint CDF with
  (a) a single joint neural CDF (MLP on the 2D Parzen-product CDF), or
  (b) 1D marginal neural CDFs (where the method is strong) + a copula
      (independence  C(u,v)=u*v, or a fitted Gaussian copula)?

Everything stays on the project's strict regime: each MLP trains ONLY on the
drawn sample points with label = (logistic) Parzen CDF value there. For the
joint that means the 2D product-logistic Parzen CDF at each sample; for the
marginals the usual 1D Parzen CDF on each coordinate.

Truth: 2D Gaussian-mixture components, so the true joint CDF is a weighted sum
of component bivariate-normal CDFs (scipy), and true marginals are 1D Gaussian
mixtures. We measure sup-distance (KS-style) and MSE of each estimated joint
CDF against truth on a grid.
"""
import numpy as np
import torch
torch.set_num_threads(2)  # ponytail: many jobs share this box; don't hog cores
from scipy.stats import multivariate_normal, norm
from scipy.special import expit

from parzen_cdf import parzen, metrics
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig, make_sample_training_set, train_cdf, set_seed,
)

# ---------------------------------------------------------------- 2D mixture
class GaussMix2D:
    """Bivariate Gaussian mixture: analytic joint CDF, marginals, sampling."""

    def __init__(self, weights, means, covs):
        self.w = np.asarray(weights, float)
        self.means = [np.asarray(m, float) for m in means]
        self.covs = [np.asarray(c, float) for c in covs]
        self.comps = [multivariate_normal(mean=m, cov=c)
                      for m, c in zip(self.means, self.covs)]

    def sample(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w)
        out = np.empty((n, 2))
        for i in range(len(self.w)):
            m = k == i
            if m.any():
                out[m] = rng.multivariate_normal(self.means[i], self.covs[i], m.sum())
        return out

    def cdf(self, pts):
        """True joint CDF F(x,y) at pts shape (...,2)."""
        pts = np.asarray(pts, float)
        acc = np.zeros(pts.shape[:-1])
        for wi, comp in zip(self.w, self.comps):
            acc += wi * comp.cdf(pts)
        return acc

    def marginal_cdf(self, vals, dim):
        """True 1D marginal CDF along coordinate dim."""
        vals = np.asarray(vals, float)
        acc = np.zeros_like(vals)
        for wi, m, c in zip(self.w, self.means, self.covs):
            acc += wi * norm.cdf(vals, loc=m[dim], scale=np.sqrt(c[dim, dim]))
        return acc

    def marginal_corr(self):
        """Population Pearson correlation of the mixture (for the Gaussian copula)."""
        # mixture mean
        mu = sum(wi * m for wi, m in zip(self.w, self.means))
        # mixture second moment E[xy], var
        cov = np.zeros((2, 2))
        for wi, m, c in zip(self.w, self.means, self.covs):
            cov += wi * (c + np.outer(m, m))
        cov -= np.outer(mu, mu)
        return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])


# --------------------------------------------------- 2D product-logistic Parzen
def joint_parzen_cdf(pts, samples, h):
    """(1/n) sum_i prod_d sigmoid((pts_d - X_i_d)/h). pts (...,2), samples (n,2)."""
    pts = np.asarray(pts, float)
    z0 = (pts[..., 0:1] - samples[:, 0]) / h          # (...,n)
    z1 = (pts[..., 1:2] - samples[:, 1]) / h
    return (expit(z0) * expit(z1)).mean(axis=-1)


def make_joint_training_set(samples, h):
    """Strict regime in 2D: inputs = the samples, targets = joint Parzen CDF there."""
    targets = joint_parzen_cdf(samples, samples, h)
    return (torch.as_tensor(samples, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32))


def silverman_2d(samples):
    """Per-dim Silverman, then a single scalar h (mean) for the product kernel."""
    n = len(samples)
    hs = [1.06 * np.std(samples[:, d]) * n ** (-1 / 5) for d in range(2)]
    return float(np.mean(hs))


# ---------------------------------------------------------------- train helper
def train_1d(samples_1d, h, seed, epochs):
    inp, tgt = make_sample_training_set(samples_1d, h)
    net = CDFNet(in_dim=1, hidden_sizes=(24,), activation="sigmoid")
    set_seed(seed)
    train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03,
                                         epochs=epochs, seed=seed))
    return net


def eval_1d(net, vals):
    with torch.no_grad():
        out = net(torch.as_tensor(vals, dtype=torch.float32)).cpu().numpy()
    return np.clip(out, 1e-6, 1 - 1e-6)


def train_joint(samples, h, seed, epochs):
    inp, tgt = make_joint_training_set(samples, h)
    net = CDFNet(in_dim=2, hidden_sizes=(40,), activation="sigmoid")
    set_seed(seed)
    train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03,
                                         epochs=epochs, seed=seed))
    return net


def eval_joint(net, pts):
    with torch.no_grad():
        out = net(torch.as_tensor(pts, dtype=torch.float32)).cpu().numpy()
    return out


# ---------------------------------------------------------------- copulas
def indep_copula(u, v):
    return u * v


def gaussian_copula(u, v, rho):
    """C(u,v) = Phi_rho(Phi^-1(u), Phi^-1(v)). Vectorized over a grid."""
    a = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    b = norm.ppf(np.clip(v, 1e-6, 1 - 1e-6))
    mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    pts = np.stack([a.ravel(), b.ravel()], axis=-1)
    return mvn.cdf(pts).reshape(a.shape)


# ---------------------------------------------------------------- experiment
def run_case(name, mix, n, rng, seed, epochs, grid_pts=35):
    samples = mix.sample(n, rng)
    h = silverman_2d(samples)

    # evaluation grid (covers the data well)
    lo = samples.min(0) - 1.0
    hi = samples.max(0) + 1.0
    gx = np.linspace(lo[0], hi[0], grid_pts)
    gy = np.linspace(lo[1], hi[1], grid_pts)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=-1)

    F_true = mix.cdf(grid)

    # ---- (a) single joint neural CDF
    jnet = train_joint(samples, h, seed, epochs)
    F_joint = eval_joint(jnet, grid)

    # ---- (b) marginals + copula
    netx = train_1d(samples[:, 0], h, seed, epochs)
    nety = train_1d(samples[:, 1], h, seed + 1, epochs)
    U = eval_1d(netx, grid[:, 0])
    V = eval_1d(nety, grid[:, 1])

    F_indep = indep_copula(U, V)
    rho_true = mix.marginal_corr()
    # rank-based (empirical) rho for the Gaussian copula, as a practitioner would
    from scipy.stats import spearmanr
    rho_emp = 2 * np.sin(np.pi / 6 * spearmanr(samples[:, 0], samples[:, 1]).statistic)
    F_gauss = gaussian_copula(U, V, rho_emp)

    # ---- also: copula on the TRUE marginals (isolates the copula model error)
    Ut = mix.marginal_cdf(grid[:, 0], 0)
    Vt = mix.marginal_cdf(grid[:, 1], 1)
    F_gauss_truemarg = gaussian_copula(Ut, Vt, rho_emp)

    def err(F):
        return dict(sup=float(np.max(np.abs(F - F_true))),
                    mse=float(metrics.mse(F, F_true)))

    # marginal quality of the 1D nets (the "method is strong here" claim)
    mx = float(np.max(np.abs(eval_1d(netx, gx) - mix.marginal_cdf(gx, 0))))
    my = float(np.max(np.abs(eval_1d(nety, gy) - mix.marginal_cdf(gy, 1))))

    return dict(name=name, n=n, h=round(h, 3), rho_true=round(rho_true, 3),
                rho_emp=round(float(rho_emp), 3),
                joint=err(F_joint), indep=err(F_indep), gauss=err(F_gauss),
                gauss_truemarg=err(F_gauss_truemarg),
                marg_sup=(round(mx, 4), round(my, 4)))


def main():
    n = 1200
    epochs = 800
    seed = 0
    rng = np.random.default_rng(7)

    cases = {}

    # 1) genuinely Gaussian, correlated -> Gaussian copula is the TRUE copula
    cases["gauss_corr0.6"] = GaussMix2D(
        [1.0], [[0, 0]], [[[1.0, 0.6], [0.6, 1.0]]])

    # 2) independent components -> independence copula is exact
    cases["gauss_indep"] = GaussMix2D(
        [1.0], [[0, 0]], [[[1.0, 0.0], [0.0, 1.0]]])

    # 3) two-blob mixture: marginals are bimodal, dependence is NON-Gaussian
    #    (blobs on a diagonal -> a Gaussian copula cannot capture it)
    cases["mix_diag"] = GaussMix2D(
        [0.5, 0.5],
        [[-2, -2], [2, 2]],
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])

    # 4) cross / X-shaped mixture: strong tail dependence, no linear corr
    cases["mix_cross"] = GaussMix2D(
        [0.5, 0.5],
        [[0, 0], [0, 0]],
        [[[2.0, 1.6], [1.6, 2.0]], [[2.0, -1.6], [-1.6, 2.0]]])

    results = []
    for name, mix in cases.items():
        r = run_case(name, mix, n, rng, seed, epochs)
        results.append(r)
        print(f"\n=== {r['name']}  (n={r['n']}, h={r['h']}, "
              f"rho_true={r['rho_true']}, rho_emp={r['rho_emp']}) ===", flush=True)
        print(f"  marginal-net sup err (x,y) : {r['marg_sup']}")
        print(f"  (a) JOINT neural CDF       : sup={r['joint']['sup']:.4f}  mse={r['joint']['mse']:.2e}")
        print(f"  (b) marg + INDEP copula    : sup={r['indep']['sup']:.4f}  mse={r['indep']['mse']:.2e}")
        print(f"  (b) marg + GAUSS copula    : sup={r['gauss']['sup']:.4f}  mse={r['gauss']['mse']:.2e}")
        print(f"      GAUSS on TRUE marginals: sup={r['gauss_truemarg']['sup']:.4f}  mse={r['gauss_truemarg']['mse']:.2e}", flush=True)

    # quick assert-style self check: on the exactly-Gaussian correlated case,
    # the Gaussian copula route must beat (or match) the raw joint net.
    g = next(r for r in results if r["name"] == "gauss_corr0.6")
    print("\n[self-check] gauss_corr0.6: gauss-copula sup <= joint sup ? ",
          g["gauss"]["sup"] <= g["joint"]["sup"] + 1e-9)

    return results


if __name__ == "__main__":
    main()
