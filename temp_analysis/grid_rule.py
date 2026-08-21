"""E2b: quale ancoraggio per il dominio di valutazione, costruito dai soli campioni.

E2 ha mostrato che [min(x) - k*h, max(x) + k*h] NON funziona: dove h e' piccolo (densita'
con una componente stretta dentro una larga) il padding e' irrilevante e la massa fuori
griglia non scende aumentando k. Serve un ancoraggio legato alla CODA, non alla risoluzione.

Regole a confronto (tutte truth-free):
  A  min/max dei dati, nessun padding
  B  padding k*h            (h da LSCV)                      <- la candidata bocciata
  C  padding k*sigma_hat                                     <- ancoraggio alla dispersione
  D  padding k*scala robusta = k*min(sigma, IQR/1.349)
  E  quantili della CDF STIMATA: risolve F_hat(lo)=eps/2, F_hat(hi)=1-eps/2
     (la stima stessa dice dove sta la sua massa; non serve scegliere un k)

Metrica: massa VERA lasciata fuori dal dominio. Meno e' meglio; conta il CASO PEGGIORE.
"""
import sys
import numpy as np
sys.path.insert(0, "temp_analysis")
from bandwidth_study import DENSITIES, h_lscv, _scale, parzen_cdf

SEEDS = range(8)
N = 500
EPS = 1e-4


def invert_parzen(x, h, target, lo, hi):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if parzen_cdf(np.array([mid]), x, h)[0] < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


rules = ["A min/max", "B  +3h", "B  +10h", "C  +1sigma", "C  +3sigma",
         "D  +3robusta", f"E quantili F_hat (eps={EPS:g})"]
out = {r: [] for r in rules}
print(f"{'densita':24} " + "".join(f"{r:>16}" for r in rules))
for d in DENSITIES:
    acc = {r: [] for r in rules}
    for s in SEEDS:
        x = d.sample(N, np.random.default_rng(s))
        h = h_lscv(x)
        sd, rob = _scale(x)
        far_lo, far_hi = x.min() - 200 * sd, x.max() + 200 * sd
        doms = {
            "A min/max":    (x.min(), x.max()),
            "B  +3h":       (x.min() - 3 * h, x.max() + 3 * h),
            "B  +10h":      (x.min() - 10 * h, x.max() + 10 * h),
            "C  +1sigma":   (x.min() - sd, x.max() + sd),
            "C  +3sigma":   (x.min() - 3 * sd, x.max() + 3 * sd),
            "D  +3robusta": (x.min() - 3 * rob, x.max() + 3 * rob),
            f"E quantili F_hat (eps={EPS:g})": (
                invert_parzen(x, h, EPS / 2, far_lo, x.min()),
                invert_parzen(x, h, 1 - EPS / 2, x.max(), far_hi)),
        }
        for r, (lo, hi) in doms.items():
            acc[r].append(float(d.cdf(np.array([lo]))[0] + 1.0 - d.cdf(np.array([hi]))[0]))
    row = f"{d.name:24} "
    for r in rules:
        m = float(np.mean(acc[r])); out[r].append(m); row += f"{m:>16.2e}"
    print(row)

print()
print(f"{'regola':32} {'media':>12} {'CASO PEGGIORE':>16}")
for r in rules:
    v = np.array(out[r])
    print(f"{r:32} {v.mean():>12.2e} {v.max():>16.2e}")
print()
print("Nota: a k=0 la massa fuori vale ~2/(n+1) = %.1e per pure statistiche d'ordine," % (2/(N+1)))
print("indipendentemente dalla densita'. E' il pavimento contro cui misurare i padding.")
