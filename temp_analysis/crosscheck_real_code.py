"""Verifica incrociata: le conclusioni tratte dalle repliche NumPy reggono sul codice vero?

Tutta l'analisi di F1 e' stata fatta su reimplementazioni indipendenti in NumPy, per non
far dipendere il giudizio dal codice giudicato. Ora che il repo gira (B1/B6), ogni
affermazione che riguarda il COMPORTAMENTO del codice va rimisurata con il codice vero.

Si usa esclusivamente la libreria del repo: parzen_cdf.* e scripts/study2_common.py.

  V1  monotonia strutturale di CDFNet          <- confronta con redesign_network.md E1b
  V2  saturazione delle code dopo il training  <- confronta con E2 / T1
  V3  invarianza per traslazione               <- confronta con E7 / D-09
  V4  le violazioni di monotonia dichiarate dallo studio precedente si riproducono?
  V5  il clamp di density_from_cdf si attiva mai?
"""

import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parzen_cdf import data, metrics, parzen                      # noqa: E402
from parzen_cdf.models import CDFNet                              # noqa: E402
from parzen_cdf.training import (TrainConfig, density_from_cdf,   # noqa: E402
                                 rectify_cdf, train_cdf)
from study2_common import (fit_cdf_net, grid_for, ks,             # noqa: E402
                           loo_cdf_targets, pnn_window, sigma_hat)

MIX = data.asymmetric_trimodal()
GRID = grid_for(MIX)
T_CDF = MIX.cdf(GRID)


def net_curve(model, g):
    with torch.no_grad():
        return model(torch.as_tensor(g, dtype=torch.float32)).numpy().astype(float)


print("=" * 92)
print("V1  MONOTONIA STRUTTURALE DI CDFNet (1000 parametrizzazioni casuali estreme)")
print("    riferimento: redesign_network.md E1b, misurato sulla replica NumPy")
print(f"    {'modalita':28} {'non monotone / 1000':>22} {'replica NumPy':>16}")
gtest = torch.linspace(-40, 40, 5000)
for mono, atteso in ((False, "955/1000"), (True, "0/1000")):
    torch.manual_seed(7)
    bad = 0
    for _ in range(1000):
        net = CDFNet(in_dim=1, hidden_sizes=(8,), activation="sigmoid", monotone=mono)
        with torch.no_grad():
            for prm in net.parameters():
                prm.add_(torch.randn_like(prm) * 3.0)
            y = net(gtest).numpy()
        if np.any(np.diff(y) < -1e-9):
            bad += 1
    print(f"    monotone={str(mono):<20} {bad:>18}/1000 {atteso:>16}")

print()
print("=" * 92)
print("V2  SATURAZIONE DELLE CODE dopo il training (trimodale, n=1000, seme 0)")
print("    riferimento: E2 (replica NumPy: massa 0.9921 libera, 0.9898 monotona)")
n, seed = 1000, 0
x = MIX.sample(n, np.random.default_rng(seed))
h = pnn_window(x, 0.5 * sigma_hat(x))
y = loo_cdf_targets(x, h)
far = np.array([np.median(x) - 50 * x.std(ddof=1), np.median(x) + 50 * x.std(ddof=1)])
print(f"    {'modalita':16} {'F(-inf)':>12} {'1-F(+inf)':>12} {'massa fra asintoti':>20}")
trained = {}
for mono in (False, True):
    torch.manual_seed(seed)
    model = CDFNet(in_dim=1, hidden_sizes=(8,), activation="sigmoid", monotone=mono)
    model, _ = train_cdf(model, torch.as_tensor(x, dtype=torch.float32),
                         torch.as_tensor(y, dtype=torch.float32),
                         TrainConfig(epochs=6000, lr=0.03, optimizer="adam", seed=seed))
    trained[mono] = model
    lo_, hi_ = net_curve(model, far)
    print(f"    monotone={str(mono):<8} {lo_:>12.6f} {1 - hi_:>12.6f} {hi_ - lo_:>20.6f}")

print()
print("=" * 92)
print("V3  INVARIANZA PER TRASLAZIONE (trimodale, n=1000)")
print("    riferimento: E7 (replica NumPy: KS 0.028 -> 0.50, collasso alla costante)")
print(f"    {'traslazione':>14} " + "".join(f"{'seme ' + str(s):>12}" for s in range(3)))
for shift in (0.0, 100.0, 1000.0):
    row = f"    {shift:>14.0f} "
    for s in range(3):
        xs = MIX.sample(n, np.random.default_rng(s)) + shift
        hs = pnn_window(xs, 0.5 * sigma_hat(xs))
        ys = loo_cdf_targets(xs, hs)
        torch.manual_seed(s)
        m = CDFNet(in_dim=1, hidden_sizes=(8,), activation="sigmoid")
        m, _ = train_cdf(m, torch.as_tensor(xs, dtype=torch.float32),
                         torch.as_tensor(ys, dtype=torch.float32),
                         TrainConfig(epochs=6000, lr=0.03, optimizer="adam", seed=s))
        rc, _ = rectify_cdf(net_curve(m, GRID + shift), GRID + shift)
        row += f"{ks(rc, T_CDF):>12.4f}"
    print(row)

print()
print("=" * 92)
print("V4  LE VIOLAZIONI DI MONOTONIA DICHIARATE DALLO STUDIO PRECEDENTE?")
print("    docs/study2.md, rimosso alla pulizia e recuperabile da git log, riportava")
print("    1.74% di violazioni per il regime 'raw' a n = 1000.")
print("    Ricetta dichiarata: width 8, etichette LOO, h = 0.5*sigma/sqrt(n-1), Adam 6000 ep.")
print(f"    {'seme':>6} {'violazioni %':>14} {'KS grezzo':>11} {'KS rettificato':>16}")
viols = []
for s in range(5):
    xs = MIX.sample(1000, np.random.default_rng(s))
    hs = pnn_window(xs, 0.5 * sigma_hat(xs))
    model, _ = fit_cdf_net(xs, loo_cdf_targets(xs, hs), width=8, epochs=6000, seed=s)
    raw = net_curve(model, GRID)
    v = metrics.monotonicity_violation_fraction(raw) * 100
    viols.append(v)
    rc, _ = rectify_cdf(raw, GRID)
    print(f"    {s:>6} {v:>13.3f}% {ks(raw, T_CDF):>11.4f} {ks(rc, T_CDF):>16.4f}")
print(f"    media su 5 semi: {np.mean(viols):.3f}%   (docs/study2.md dichiara 1.74%)")

print()
print("=" * 92)
print("V5  IL CLAMP DI density_from_cdf SI ATTIVA MAI?")
print("    riferimento: E5 (replica NumPy: mai attivo, difetto latente non osservato)")
gt = torch.as_tensor(GRID, dtype=torch.float32)
for label, model in (("monotone=False", trained[False]), ("monotone=True", trained[True])):
    raw_d = density_from_cdf(model, gt, clamp=False).detach().numpy()
    cl_d = density_from_cdf(model, gt, clamp=True).detach().numpy()
    neg = int(np.sum(raw_d < 0))
    m_raw = float(np.trapezoid(raw_d, GRID))
    m_cl = float(np.trapezoid(cl_d, GRID))
    print(f"    {label:16} punti con densita' < 0: {neg:>5}/{GRID.size}   "
          f"int f = {m_raw:.6f}   int clamp(f) = {m_cl:.6f}   differenza {m_cl - m_raw:+.2e}")
