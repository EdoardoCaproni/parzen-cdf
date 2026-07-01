"""Test B -- denoising: can the net beat the BEST fixed-window Parzen?

A 1D statistical-advantage test. Fix n; sweep the target window size from sharp/noisy to smooth/biased.
For each window: the Parzen CDF gap vs truth, and the net (trained on that Parzen target) gap vs truth.
If the net's best over the sweep is below the Parzen's best over the sweep, the network's smoothness has
turned a sharp (low-bias, high-variance) target into something better than any single fixed-window
Parzen -- a genuine accuracy advantage, not just a faithful copy. Averaged over seeds.
"""
import numpy as np
import torch

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed

mix = data.symmetric_bimodal()        # bandwidth trade-off is clear on a bimodal
GRID = np.linspace(-6, 6, 2000); GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
TRUE = mix.cdf(GRID)
N = 2000
SCALES = [0.15, 0.25, 0.4, 0.55, 0.7, 1.0, 1.4]   # x Silverman
SEEDS = range(4)


def net_ks(samples, h, seed):
    set_seed(seed)
    inp, tgt = make_sample_training_set(samples, h)
    m = CDFNet(1, (32,), activation="sigmoid", monotone=False)
    m, _ = train_cdf(m, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=4000, seed=seed))
    return metrics.ks_distance(TRUE, m(GRID_T).detach().numpy())


parzen_ks = {sc: [] for sc in SCALES}
net_ks_d = {sc: [] for sc in SCALES}
for seed in SEEDS:
    s = mix.sample(N, np.random.default_rng(seed))
    h0 = parzen.silverman_bandwidth(s)
    for sc in SCALES:
        h = sc * h0
        parzen_ks[sc].append(metrics.ks_distance(TRUE, parzen.parzen_cdf(GRID, s, h)))
        net_ks_d[sc].append(net_ks(s, h, seed))

print(f"symmetric_bimodal, n={N}, mean over {len(list(SEEDS))} seeds")
print(f"  {'scale':>7}{'parzen KS':>12}{'net KS':>10}")
for sc in SCALES:
    print(f"  {sc:>7.2f}{np.mean(parzen_ks[sc]):>12.4f}{np.mean(net_ks_d[sc]):>10.4f}")
best_parzen = min(np.mean(parzen_ks[sc]) for sc in SCALES)
best_net = min(np.mean(net_ks_d[sc]) for sc in SCALES)
print(f"\n  best fixed-window Parzen KS = {best_parzen:.4f}")
print(f"  best net (over windows)  KS = {best_net:.4f}")
print(f"  -> net {'BEATS' if best_net < best_parzen else 'does NOT beat'} the best fixed-window Parzen"
      f" (by {100*(best_parzen-best_net)/best_parzen:+.1f}%)")
