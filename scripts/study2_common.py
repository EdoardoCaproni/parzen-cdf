"""Shared helpers for the corrected Phase A study (study2): the h_n = h1/sqrt(n) framework.

Budgets are deliberately small (the Professor's point): baseline n = 500, never beyond 2000.
Every experiment is scored against the known truth with the CDF gap (KS) as the primary
metric and the pdf integrated squared error (ISE) as the secondary, density-eye view.
"""

import numpy as np

from parzen_cdf import parzen

# ``np.trapezoid`` e' il nome NumPy >= 2.0 di ``np.trapz`` (rimosso in NumPy 2.4).
# NB: il guard deve essere PIGRO. ``getattr(np, "trapezoid", np.trapz)`` valuterebbe
# ``np.trapz`` comunque, sollevando AttributeError proprio dove non serve.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

BUDGETS = (500, 1000, 2000)
SEEDS = range(10)


def ks_floor(n: int) -> float:
    """E[KS] of the empirical CDF, ~ the statistical floor: 0.8687 / sqrt(n)."""
    return 0.8687 / np.sqrt(n)


def grid_for(mix, n_points: int = 2001) -> np.ndarray:
    lo = float((mix.means - 5 * mix.stds).min())
    hi = float((mix.means + 5 * mix.stds).max())
    return np.linspace(lo, hi, n_points)


def ecdf_on_grid(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(samples), grid, side="right") / samples.size


def ks(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def ise(pdf_est: np.ndarray, pdf_true: np.ndarray, grid: np.ndarray) -> float:
    d2 = (pdf_est - pdf_true) ** 2
    return float(_trapezoid(d2, grid))


def eval_window(mix, grid, truth_cdf, truth_pdf, samples, h) -> dict:
    """Score one logistic-Parzen estimate at window size ``h`` against the truth."""
    return {
        "ks": ks(parzen.parzen_cdf(grid, samples, h), truth_cdf),
        "ise": ise(parzen.parzen_pdf(grid, samples, h), truth_pdf, grid),
    }


def stress_h1(mix, h1_grid, budgets=BUDGETS, seeds=SEEDS) -> dict:
    """Mean KS and pdf-ISE over seeds, for every (budget, h1): the h1 stress test."""
    grid = grid_for(mix)
    t_cdf, t_pdf = mix.cdf(grid), mix.pdf(grid)
    out = {}
    for n in budgets:
        ks_rows, ise_rows = [], []
        for seed in seeds:
            samples = mix.sample(n, np.random.default_rng(seed))
            row_ks, row_ise = [], []
            for h1 in h1_grid:
                r = eval_window(mix, grid, t_cdf, t_pdf, samples, h1 / np.sqrt(n))
                row_ks.append(r["ks"])
                row_ise.append(r["ise"])
            ks_rows.append(row_ks)
            ise_rows.append(row_ise)
        out[n] = {"ks": np.mean(ks_rows, axis=0), "ise": np.mean(ise_rows, axis=0)}
    return out


def sigma_hat(samples: np.ndarray) -> float:
    """The truth-free scale used by the h1 heuristic: the sample standard deviation."""
    return float(np.asarray(samples).std(ddof=1))


def plateau(h1_grid, mean_ks, tol: float = 1.10) -> tuple:
    """The h1 range whose mean KS stays within ``tol`` of the optimum (robustness width)."""
    best = mean_ks.min()
    ok = h1_grid[mean_ks <= tol * best]
    return float(ok.min()), float(ok.max())


def ecdf_ks_mean(mix, n, seeds=SEEDS) -> float:
    grid = grid_for(mix)
    t_cdf = mix.cdf(grid)
    return float(np.mean([ks(ecdf_on_grid(mix.sample(n, np.random.default_rng(s)), grid), t_cdf)
                          for s in seeds]))


# ---------------------------------------------------------------------- Phase B (PNN-CDF)

def loo_cdf_targets(samples: np.ndarray, h: float) -> np.ndarray:
    """Leave-one-out logistic-Parzen CDF at the sample points (the PNN's 'unbiased' targets).

    The full estimate includes the sample's own window, which contributes K(0) = 1/2:
    y_i = (n * F_hat(x_i) - 1/2) / (n - 1).
    """
    n = samples.size
    full = parzen.parzen_cdf(samples, samples, h)
    return (n * full - 0.5) / (n - 1)


def pnn_window(samples: np.ndarray, h1: float) -> float:
    """The PNN schedule: h_n = h1 / sqrt(n - 1)."""
    return float(h1 / np.sqrt(samples.size - 1))


def fit_cdf_net(samples: np.ndarray, targets: np.ndarray, width: int = 32,
                epochs: int = 6000, seed: int = 0):
    """Train the sigmoidal CDF regressor on (x_i, y_i) only (full batch, Adam)."""
    import torch
    from parzen_cdf.models import CDFNet
    from parzen_cdf.training import TrainConfig, train_cdf

    cfg = TrainConfig(epochs=epochs, lr=0.03, optimizer="adam", seed=seed)
    model = CDFNet(in_dim=1, hidden_sizes=(width,), activation="sigmoid")
    model, hist = train_cdf(model,
                            torch.as_tensor(samples, dtype=torch.float32),
                            torch.as_tensor(targets, dtype=torch.float32), cfg)
    return model, hist[-1]


def eval_cdf_net(model, grid, truth_cdf, truth_pdf):
    """Rectified net CDF/pdf scored against the truth (KS + pdf ISE)."""
    import torch
    from parzen_cdf.training import rectify_cdf

    with torch.no_grad():
        raw = model(torch.as_tensor(grid, dtype=torch.float32)).numpy().astype(float)
    cdf, pdf = rectify_cdf(raw, grid)
    return {"ks": ks(cdf, truth_cdf), "ise": ise(pdf, truth_pdf, grid)}
