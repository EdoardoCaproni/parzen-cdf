"""Is the 800-epoch net actually converged to its Parzen target, or underfit?
At a representative scale, train for increasing epochs and report:
  net-vs-truth KS,  net-vs-its-own-Parzen-target KS (faithfulness),  parzen-vs-truth KS.
If net-vs-target KS -> 0 but net-vs-truth stays above parzen-vs-truth, the net is
converged and genuinely cannot beat Parzen (verdict robust). If net-vs-truth keeps
dropping with epochs, 800 was underfit and the main result is unfair.
"""
import numpy as np, torch
torch.set_num_threads(2)
from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed

mix = data.symmetric_bimodal()           # the hardest case for the net
grid = np.linspace(-6, 6, 800); gt = torch.as_tensor(grid, dtype=torch.float32)
true = mix.cdf(grid)
s = mix.sample(2000, np.random.default_rng(0))
h = 0.35 * parzen.silverman_bandwidth(s)   # near Parzen's best scale
ptgt = parzen.parzen_cdf(grid, s, h)        # the net's target on the grid
print(f"symmetric_bimodal scale=0.35  parzen-vs-truth KS={metrics.ks_distance(true, ptgt):.4f}", flush=True)
print(f"  {'epochs':>7}{'net-truthKS':>13}{'net-targetKS':>14}", flush=True)
for ep in (800, 1500, 2500):
    set_seed(0)
    inp, tgt = make_sample_training_set(s, h)
    m = CDFNet(1, (32,), activation="sigmoid", monotone=False)
    m, _ = train_cdf(m, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=ep, seed=0))
    nc = m(gt).detach().numpy()
    print(f"  {ep:>7}{metrics.ks_distance(true, nc):>13.4f}{metrics.ks_distance(ptgt, nc):>14.4f}", flush=True)
