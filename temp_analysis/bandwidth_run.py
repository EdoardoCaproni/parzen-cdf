"""Esecuzione dello studio sulla finestra: validazioni, poi risultati.

Uso:  python temp_analysis/bandwidth_run.py [n] [n_semi]
"""

import sys
import numpy as np

sys.path.insert(0, "temp_analysis")
from bandwidth_study import (DENSITIES, H_SWEEP, K_STD, SEL_FAMILIES, SEL_FIXED,
                             _scale, calibrate_lodo, h_plugin, h_sigma_rule, ise, ks,
                             parzen_cdf, parzen_pdf, run, sweep, at)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SEEDS = range(NSEED)

# ============================================================== V  VALIDAZIONI
print("=" * 100)
print("V  VALIDAZIONI (se una fallisce, tutto il resto e' da buttare)")

print("  V1  ogni densita' integra a 1 e la CDF va da ~0 a ~1 sulla griglia usata")
worst_m, worst_c = 0.0, 0.0
for d in DENSITIES:
    g = d.grid()
    m = float(np.trapezoid(d.pdf(g), g))
    c0, c1 = float(d.cdf(g)[0]), float(d.cdf(g)[-1])
    worst_m = max(worst_m, abs(m - 1)); worst_c = max(worst_c, c0, abs(c1 - 1))
print(f"      scarto massimo sulla massa: {worst_m:.2e}   sugli estremi della CDF: {worst_c:.2e}")

print("  V2  il plug-in a due stadi su dati gaussiani deve dare ~1.06*s*n^(-1/5)")
for n in (200, 500, 2000):
    r = []
    for s in range(20):
        x = np.random.default_rng(s).normal(0, 1, n)
        r.append(h_plugin(x) * K_STD / (x.std(ddof=1) * n ** -0.2))   # riportato a kernel gaussiano
    print(f"      n={n:>5}: costante implicita = {np.mean(r):.4f} +- {np.std(r):.4f}   (atteso ~1.06)")

print("  V3  l'interpolazione della curva ISE(h) non introduce errore apprezzabile")
d = DENSITIES[9]
x, hs, I, K = sweep(d, N, 0)
g = d.grid(); tp = d.pdf(g)
err = []
for frac in (0.17, 0.33, 0.51, 0.72, 0.9):
    h = float(np.exp(np.interp(frac, [0, 1], [np.log(hs[0]), np.log(hs[-1])])))
    err.append(abs(at(hs, I, h) - ise(parzen_pdf(g, x, h), tp, g)) / ise(parzen_pdf(g, x, h), tp, g))
print(f"      errore relativo massimo dell'interpolazione: {max(err):.2%} (su {d.name})")

print("  V4  l'ottimo dell'oracolo non tocca i bordi della griglia di h (altrimenti e' troncato)")
edge = 0
for d in DENSITIES:
    _, hs, I, K = sweep(d, N, 0)
    if np.argmin(I) in (0, hs.size - 1) or np.argmin(K) in (0, hs.size - 1):
        edge += 1
        print(f"      ATTENZIONE bordo: {d.name}  argmin ISE={np.argmin(I)} argmin KS={np.argmin(K)}")
print(f"      densita' con ottimo sul bordo: {edge}/{len(DENSITIES)}")

# ============================================================== R  RISULTATI
print()
print("=" * 100)
print(f"R  RISULTATI  (n = {N}, {NSEED} semi, kernel logistico)")
print("   efficienza = errore(selettore) / errore(oracolo);  1.00 = ottimo, piu' alto = peggio")

res, curves = run(N, SEEDS)

for fam_name, (fac, cgrid) in SEL_FAMILIES.items():
    out, cstar, cbest = calibrate_lodo(curves, fac, cgrid, DENSITIES)
    for nm in res:
        res[nm][fam_name + " [LODO]"] = out[nm]
    print(f"\n   costante ottima per densita', famiglia '{fam_name}':")
    vals = np.array([cbest[d.name] for d in DENSITIES])
    print("      " + "  ".join(f"{d.name.split()[0]}={cbest[d.name]:.2f}" for d in DENSITIES))
    print(f"      -> min {vals.min():.2f}  max {vals.max():.2f}  rapporto max/min = "
          f"{vals.max() / vals.min():.1f}x   (se fosse una legge, sarebbero tutte uguali)")

SELS = list(SEL_FIXED) + [f + " [LODO]" for f in SEL_FAMILIES]

print()
print("   EFFICIENZA ISE DELLA pdf, per densita' (media sui semi)")
hdr = f"   {'densita':24}" + "".join(f"{s.split(' [')[0][:13]:>14}" for s in SELS)
print(hdr)
for d in DENSITIES:
    row = f"   {d.name:24}"
    for s in SELS:
        row += f"{np.mean([a for a, _ in res[d.name][s]]):>14.2f}"
    print(row)

print()
print("   SINTESI (su tutte le densita')")
print(f"   {'selettore':40} {'ISE media':>10} {'ISE p90':>9} {'ISE peggiore':>13} "
      f"{'KS media':>9} {'KS peggiore':>12} {'>2x':>6}")
summary = {}
for s in SELS:
    per_d = np.array([np.mean([a for a, _ in res[d.name][s]]) for d in DENSITIES])
    per_k = np.array([np.mean([b for _, b in res[d.name][s]]) for d in DENSITIES])
    allv = np.array([a for d in DENSITIES for a, _ in res[d.name][s]])
    bad = float(np.mean(allv > 2.0))
    summary[s] = (per_d.mean(), np.percentile(allv, 90), per_d.max(), per_k.mean(), per_k.max(), bad)
    print(f"   {s:40} {per_d.mean():>10.2f} {np.percentile(allv, 90):>9.2f} {per_d.max():>13.2f} "
          f"{per_k.mean():>9.2f} {per_k.max():>12.2f} {bad:>5.0%}")

print()
print("   VARIABILITA' DI h FRA CAMPIONI (coefficiente di variazione di h sui semi).")
print("   La critica classica alla CV e' proprio l'instabilita': va misurata, non assunta.")
print(f"   {'selettore':22} " + "".join(f"{d.name.split()[0]:>8}" for d in DENSITIES))
for s in ("plug-in 2 stadi", "LSCV", "MLCV (KL)", "Silverman robusto"):
    fn = SEL_FIXED[s]
    row = f"   {s:22} "
    for d in DENSITIES:
        hh = np.array([fn(x) for (x, *_r) in curves[d.name]])
        row += f"{hh.std(ddof=1) / hh.mean():>8.2f}"
    print(row)

best = min(summary, key=lambda s: summary[s][0])
print(f"\n   migliore per efficienza ISE media: {best}")
rob = min(summary, key=lambda s: summary[s][1])
print(f"   migliore per caso peggiore:        {rob}")

print()
print("   h1 EQUIVALENTE (h * sqrt(n)) del selettore raccomandato, per densita'")
print("   serve a consegnare il risultato nella forma h_n = h1/sqrt(n) richiesta dall'esercizio")
fn = SEL_FIXED.get(best)
if fn is not None:
    for d in DENSITIES:
        vals = [fn(x) * np.sqrt(N) for (x, *_r) in curves[d.name]]
        sig = [np.std(x, ddof=1) for (x, *_r) in curves[d.name]]
        print(f"      {d.name:24} h1 = {np.mean(vals):7.3f}   h1/sigma = "
              f"{np.mean(np.array(vals) / np.array(sig)):6.3f}")
