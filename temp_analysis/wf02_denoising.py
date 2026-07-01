"""wf02_denoising: does the net beat the BEST fixed-window Parzen?

Statistical-advantage test. Fix n=2000; sweep the target window over scales of
Silverman. For each scale: Parzen CDF gap vs truth (KS and MSE), and the net
trained on that Parzen target, its gap vs truth. Question: does the net's
min-over-windows beat Parzen's min-over-windows (the oracle bandwidth)?

Run per distribution:  python temp_analysis/wf02_denoising.py <dist>
Or all three:          python temp_analysis/wf02_denoising.py all
Each row is flushed so partial results survive a timeout.
"""
import sys
import numpy as np
import torch

torch.set_num_threads(2)  # ponytail: cap threads, many jobs run concurrently

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed

N = 2000
SCALES = [0.35, 0.55, 0.8, 1.2]   # x Silverman bandwidth (Parzen optimum lives in here)
SEEDS = range(3)
EPOCHS = 2500   # net must reach its Parzen target; 800 underfits multimodal targets badly

DISTS = {
    "single_gaussian": (data.single_gaussian, (-6, 6)),
    "symmetric_bimodal": (data.symmetric_bimodal, (-6, 6)),
    "asymmetric_trimodal": (data.asymmetric_trimodal, (-8, 10)),
}


def net_cdf(samples, h, seed, grid_t):
    set_seed(seed)
    inp, tgt = make_sample_training_set(samples, h)
    m = CDFNet(1, (32,), activation="sigmoid", monotone=False)
    m, _ = train_cdf(m, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=EPOCHS, seed=seed))
    return m(grid_t).detach().numpy()


def run_dist(name):
    factory, (lo, hi) = DISTS[name]
    mix = factory()
    grid = np.linspace(lo, hi, 800)
    grid_t = torch.as_tensor(grid, dtype=torch.float32)
    true = mix.cdf(grid)

    pks = {sc: [] for sc in SCALES}
    pmse = {sc: [] for sc in SCALES}
    nks = {sc: [] for sc in SCALES}
    nmse = {sc: [] for sc in SCALES}

    samples = [mix.sample(N, np.random.default_rng(seed)) for seed in SEEDS]
    print(f"=== {name}, n={N}, {len(samples)} seeds, {EPOCHS} epochs ===", flush=True)
    print(f"  {'scale':>7}{'parzenKS':>10}{'netKS':>9}{'parzenMSE':>12}{'netMSE':>12}", flush=True)
    for sc in SCALES:
        for seed, s in zip(SEEDS, samples):
            h = sc * parzen.silverman_bandwidth(s)
            pc = parzen.parzen_cdf(grid, s, h)
            pks[sc].append(metrics.ks_distance(true, pc))
            pmse[sc].append(metrics.mse(true, pc))
            nc = net_cdf(s, h, seed, grid_t)
            nks[sc].append(metrics.ks_distance(true, nc))
            nmse[sc].append(metrics.mse(true, nc))
        print(f"  {sc:>7.2f}{np.mean(pks[sc]):>10.4f}{np.mean(nks[sc]):>9.4f}"
              f"{np.mean(pmse[sc]):>12.6f}{np.mean(nmse[sc]):>12.6f}", flush=True)

    bp_ks = min(np.mean(pks[sc]) for sc in SCALES)
    bn_ks = min(np.mean(nks[sc]) for sc in SCALES)
    bp_mse = min(np.mean(pmse[sc]) for sc in SCALES)
    bn_mse = min(np.mean(nmse[sc]) for sc in SCALES)
    print(f"  best fixed-window Parzen: KS={bp_ks:.4f}  MSE={bp_mse:.6f}", flush=True)
    print(f"  best net (over windows) : KS={bn_ks:.4f}  MSE={bn_mse:.6f}", flush=True)
    print(f"  net vs Parzen  KS: {'BEATS' if bn_ks < bp_ks else 'NO'} ({100*(bp_ks-bn_ks)/bp_ks:+.1f}%)"
          f"   MSE: {'BEATS' if bn_mse < bp_mse else 'NO'} ({100*(bp_mse-bn_mse)/bp_mse:+.1f}%)",
          flush=True)
    return {"name": name, "bp_ks": bp_ks, "bn_ks": bn_ks, "bp_mse": bp_mse, "bn_mse": bn_mse}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(DISTS) if which == "all" else [which]
    for nm in names:
        run_dist(nm)
