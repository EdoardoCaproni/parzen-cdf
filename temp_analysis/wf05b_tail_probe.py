"""wf05b -- deeper tail probe: does the net's CDF stay in [0,1] far out, does its
pdf -> 0, and where exactly does the net beat or lose to Parzen relative to truth.

Reuses the same strict pipeline. Single Gaussian + symmetric bimodal, n=2000, 1 seed
each (cheap probe; the headline numbers come from wf05). We look at:
  - far extrapolation: CDF and pdf at x = +-(span..3*span); does net overshoot/undershoot
    past 0/1, does its tail pdf stay ~0 or blow up?
  - the moderate tail (true cdf ~1e-3 .. 1e-1) where tail accuracy actually matters,
    measured as max abs CDF error there.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

torch.set_num_threads(2)

from parzen_cdf import data, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed, density_from_cdf

N = 2000
EPOCHS = 1500
SEED = 0


def fit(samples, h, seed):
    set_seed(seed)
    inp, tgt = make_sample_training_set(samples, h)
    m = CDFNet(1, (32,), activation="sigmoid", monotone=False)
    m, _ = train_cdf(m, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=EPOCHS, seed=seed))
    return m


for mix, name, span in [(data.single_gaussian(), "single_gaussian", 8.0),
                        (data.symmetric_bimodal(), "symmetric_bimodal", 9.0)]:
    s = mix.sample(N, np.random.default_rng(SEED))
    lo, hi = float(s.min()), float(s.max())
    h = parzen.variance_matched_bandwidth(s)
    m = fit(s, h, SEED)

    print(f"\n=== {name} (n={N}, seed {SEED}) data range [{lo:.2f},{hi:.2f}] ===")

    # far probes at increasing distance beyond the data
    probes = np.array([hi + 1, hi + 3, hi + 6, 2 * span, 3 * span,
                       lo - 1, lo - 3, lo - 6, -2 * span, -3 * span])
    pt = torch.as_tensor(probes, dtype=torch.float32)
    net_cdf = m(pt).detach().numpy().ravel()
    parz_cdf = parzen.parzen_cdf(probes, s, h)
    true_cdf = mix.cdf(probes)
    net_pdf = density_from_cdf(m, pt, clamp=False).detach().numpy().ravel()
    parz_pdf = parzen.parzen_pdf(probes, s, h)
    true_pdf = mix.pdf(probes)

    print("  far-tail CDF (true / parzen / net) and pdf:")
    print(f"  {'x':>8}{'trueF':>9}{'parzF':>9}{'netF':>9}  | {'truef':>9}{'parzf':>10}{'netf':>10}")
    for i in range(len(probes)):
        print(f"  {probes[i]:>8.1f}{true_cdf[i]:>9.5f}{parz_cdf[i]:>9.5f}{net_cdf[i]:>9.5f}  |"
              f" {true_pdf[i]:>9.2e}{parz_pdf[i]:>10.2e}{net_pdf[i]:>10.2e}")

    # does net CDF leave [0,1] or go non-monotone far out?
    xx = np.linspace(-3 * span, 3 * span, 1200)
    nc = m(torch.as_tensor(xx, dtype=torch.float32)).detach().numpy().ravel()
    print(f"  net CDF range over [+-3span]: [{nc.min():.5f}, {nc.max():.5f}]  "
          f"(outside [0,1]? {'YES' if nc.min() < -1e-4 or nc.max() > 1 + 1e-4 else 'no'})")
    dec = np.diff(nc)
    print(f"  net CDF max decrease (non-monotonicity) over [+-3span]: {min(dec.min(), 0.0):.2e}")

    # moderate tail where accuracy matters: true cdf in [1e-3, 0.1] (right) and [0.9,1-1e-3]
    g = np.linspace(-span, span, 1500)
    tc = mix.cdf(g)
    mask = ((tc > 1e-3) & (tc < 0.1)) | ((tc > 0.9) & (tc < 1 - 1e-3))
    ncg = m(torch.as_tensor(g, dtype=torch.float32)).detach().numpy().ravel()
    pcg = parzen.parzen_cdf(g, s, h)
    print(f"  moderate-tail (trueF in 1e-3..0.1 & .9..1-1e-3) max|CDF err|: "
          f"parzen={np.max(np.abs(tc[mask]-pcg[mask])):.4f}  net={np.max(np.abs(tc[mask]-ncg[mask])):.4f}")
