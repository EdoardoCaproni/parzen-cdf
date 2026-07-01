"""Test A/F -- amortization: query cost and memory, net (O(params), n-independent) vs Parzen (O(n)).

A trained net evaluates the CDF/pdf at any point in cost set by its parameter count, independent of the
training sample count n. The Parzen estimate must sum over all n stored samples at every query. We time
both as n grows, and report memory (stored floats).
"""
import time

import numpy as np
import torch

from parzen_cdf import data, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed, density_from_cdf

mix = data.single_gaussian()
M = 1000                       # query points
query = np.linspace(-6, 6, M)
query_t = torch.as_tensor(query, dtype=torch.float32)

# train the net ONCE (width 32) on a modest sample; its query cost is fixed thereafter
set_seed(0)
s0 = mix.sample(2000, np.random.default_rng(0))
inp, tgt = make_sample_training_set(s0, parzen.silverman_bandwidth(s0))
net = CDFNet(1, (32,), activation="sigmoid", monotone=False)
net, _ = train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=2000, seed=0))
n_params = sum(p.numel() for p in net.parameters())


def timeit(fn, reps=20):
    fn()  # warmup
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


# net query cost (fixed params, independent of n)
t_net_cdf = timeit(lambda: net(query_t).detach().numpy())
t_net_pdf = timeit(lambda: density_from_cdf(net, query_t).detach().numpy())

print(f"net: {n_params} params (constant memory), independent of n")
print(f"  net CDF query ({M} pts): {1000*t_net_cdf:.3f} ms   net pdf query: {1000*t_net_pdf:.3f} ms")
print(f"\n{'n':>9}{'parzen mem':>13}{'parzen CDF ms':>15}{'parzen pdf ms':>15}{'x slower than net CDF':>24}")
for n in (1000, 5000, 20000, 100000):
    s = mix.sample(n, np.random.default_rng(1))
    h = parzen.silverman_bandwidth(s)
    t_p_cdf = timeit(lambda: parzen.parzen_cdf(query, s, h), reps=10)
    t_p_pdf = timeit(lambda: parzen.parzen_pdf(query, s, h), reps=10)
    print(f"{n:>9}{n:>10} flt{1000*t_p_cdf:>15.3f}{1000*t_p_pdf:>15.3f}{t_p_cdf/t_net_cdf:>24.1f}")

print("\nTakeaway: the net's query cost is flat (set by params), Parzen's grows linearly in n, and "
      "the net stores ~%d floats vs n for Parzen." % n_params)
