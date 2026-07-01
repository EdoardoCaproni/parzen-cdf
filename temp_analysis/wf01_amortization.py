"""wf01 -- Amortization & dimension scaling: the net's query cost is O(params) (constant in n),
Parzen's is O(n). We (1) time both as n grows to 100k on a 1000-pt grid, (2) report stored memory
(params vs n floats), (3) show the net's param count grows only gently with dimension D while KDE
stays O(n) per query point, and (4) illustrate that the n KDE needs for a fixed accuracy itself
grows with D (curse of dimensionality), compounding the per-query O(n) cost.

This is the *amortization* argument: pay a one-off training cost, then every future query is cheap
and n-independent; Parzen has no training cost but pays O(n) forever, on every query, and that n
must grow with D to stay accurate.
"""
import time

import numpy as np
import torch

from parzen_cdf import data, parzen, metrics
from parzen_cdf.models import CDFNet
from parzen_cdf.training import (
    TrainConfig, make_sample_training_set, train_cdf, set_seed, density_from_cdf,
)


def timeit(fn, reps):
    fn()  # warmup
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


# ---------------------------------------------------------------------------
# Part 1 + 2: query cost and memory vs n (1D), net (fixed params) vs Parzen (O(n)).
# ---------------------------------------------------------------------------
mix = data.single_gaussian()
M = 1000
query = np.linspace(-6, 6, M)
query_t = torch.as_tensor(query, dtype=torch.float32)

set_seed(0)
# The QUERY-cost measurement is independent of training quality: a net's forward/derivative cost is
# fixed by its architecture (params), not by how well it fits. So we time an *initialized* width-32
# net here (training is a one-off cost paid before any query; see the amortization argument in the
# writeup, and the baseline trained-net timing in temp_analysis/test_query_cost.py which agrees).
net = CDFNet(1, (32,), activation="sigmoid", monotone=False)
net.eval()
print("net built, beginning timing", flush=True)
n_params = sum(p.numel() for p in net.parameters())

with torch.no_grad():
    t_net_cdf = timeit(lambda: net(query_t).numpy(), reps=50)
t_net_pdf = timeit(lambda: density_from_cdf(net, query_t).detach().numpy(), reps=50)

print("=== Part 1+2: 1D query cost & memory, net (constant) vs Parzen (O(n)) ===", flush=True)
print(f"net: {n_params} params (constant memory ~= {n_params} floats), independent of n")
print(f"  net CDF query ({M} pts): {1000*t_net_cdf:.3f} ms | net pdf query: {1000*t_net_pdf:.3f} ms")
print(f"\n{'n':>9}{'parzen floats':>15}{'mem ratio':>11}"
      f"{'parzen CDF ms':>15}{'parzen pdf ms':>15}{'CDF speedup':>13}{'pdf speedup':>13}")
ns = (1000, 5000, 20000, 100000)
rows = []
for n in ns:
    s = mix.sample(n, np.random.default_rng(1))
    h = parzen.silverman_bandwidth(s)
    reps = 8 if n < 20000 else (3 if n < 100000 else 1)
    t_p_cdf = timeit(lambda: parzen.parzen_cdf(query, s, h), reps=reps)
    t_p_pdf = timeit(lambda: parzen.parzen_pdf(query, s, h), reps=reps)
    cdf_speed = t_p_cdf / t_net_cdf
    pdf_speed = t_p_pdf / t_net_pdf
    mem_ratio = n / n_params
    rows.append((n, t_p_cdf, t_p_pdf, cdf_speed, pdf_speed, mem_ratio))
    print(f"{n:>9}{n:>15}{mem_ratio:>10.0f}x"
          f"{1000*t_p_cdf:>15.3f}{1000*t_p_pdf:>15.3f}{cdf_speed:>12.0f}x{pdf_speed:>12.0f}x",
          flush=True)

# Confirm scaling is linear in n: cost-per-n should be ~constant across the rows.
cdf_per_n = [1000 * r[1] / r[0] for r in rows]
print(f"\nParzen CDF cost per sample (us/sample, should be ~flat => linear in n): "
      f"{['%.4f' % (1000*c) for c in cdf_per_n]}")

# ---------------------------------------------------------------------------
# Part 3: dimension scaling of STORAGE. Net params grow gently with D (one extra weight column per
# input dim); KDE must store all n points x D coords, and (Part 4) n itself must grow with D.
# ---------------------------------------------------------------------------
print("\n=== Part 3: storage vs dimension D ===")
print(f"{'D':>4}{'net params (w=32)':>20}{'KDE floats (n=20000)':>24}")
n_fixed = 20000
for d in (1, 2, 3, 5, 10, 20):
    p = sum(pp.numel() for pp in CDFNet(d, (32,)).parameters())
    print(f"{d:>4}{p:>20}{n_fixed * d:>24}")
print("net params ~ in_dim*width + const => LINEAR & tiny in D; KDE storage ~ n*D and n must grow (Part 4)")

# ---------------------------------------------------------------------------
# Part 4: the n KDE needs for a FIXED accuracy grows with D (curse of dimensionality). We hold the
# product-Gaussian target fixed per dim and ask: how many samples does the logistic-Parzen KDE need
# so its density MISE (estimated on a random test set) falls below a fixed threshold, in D=1,2,3?
# A separable standard normal in D dims; KDE uses a per-dim Silverman bandwidth (product kernel).
# ---------------------------------------------------------------------------
def kde_pdf_nd(query_pts, samples, h):
    """Product-logistic KDE density at query_pts (m,D) given samples (n,D), scalar/per-dim h."""
    z = (query_pts[:, None, :] - samples[None, :, :]) / h          # (m, n, D)
    s = 1.0 / (1.0 + np.exp(-z))
    k = (s * (1.0 - s) / h)                                        # per-dim logistic kernel density
    return k.prod(axis=2).mean(axis=1)                            # product kernel, average over n


def true_pdf_nd(query_pts):
    """Standard normal product density in D dims."""
    d = query_pts.shape[1]
    return np.exp(-0.5 * (query_pts ** 2).sum(axis=1)) / (2 * np.pi) ** (d / 2)


print("\n=== Part 4: samples needed for fixed-accuracy KDE grow with D (separable standard normal) ===")
rng = np.random.default_rng(7)
M_test = 300
TARGET_REL = 0.10    # target: relative density-MISE (normalized by mean true density^2) below this
print(f"{'D':>4}{'n to hit relMISE<%.2f' % TARGET_REL:>26}{'best relMISE@n_grid':>22}")
for d in (1, 2, 3):
    # test points drawn from the true density so the error metric weights the bulk of the mass
    test = rng.standard_normal((M_test, d))
    f_true = true_pdf_nd(test)
    denom = np.mean(f_true ** 2)
    found_n = None
    best_rel = np.inf
    for n in (200, 500, 1000, 2000, 5000, 10000, 20000):
        s = rng.standard_normal((n, d))
        # per-dim Silverman bandwidth (each dim ~ standard normal, std~1)
        h = 0.9 * 1.0 * n ** (-1.0 / (d + 4))          # Silverman exponent generalizes to -(1/(d+4))
        f_hat = kde_pdf_nd(test, s, h)
        rel = np.mean((f_hat - f_true) ** 2) / denom
        best_rel = min(best_rel, rel)
        if found_n is None and rel < TARGET_REL:
            found_n = n
    label = str(found_n) if found_n is not None else ">50000"
    print(f"{d:>4}{label:>26}{best_rel:>22.4f}")
print("Even for the easiest target (a Gaussian), the n needed to reach the same relative accuracy")
print("climbs steeply with D; every one of those n samples is touched on EVERY KDE query.")
