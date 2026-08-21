"""Evidenze per il re-design della pipeline (Q5): griglia, dominio, diagnostica truth-free.

Tre domande, tutte con lo stesso vincolo: sui campioni del docente NON abbiamo la verita'.

  E1  quanto cambia la stima consegnata al variare della griglia?
      (la rettifica riscala usando gli estremi della griglia: e' una dipendenza reale)
  E2  quale regola per costruire la griglia DAI DATI garantisce di non tagliare le code?
  E3  quali diagnostiche truth-free sono informative e quali sono TRAPPOLE?
      (senza verita', e' l'unico modo che avremo per sapere se la stima e' buona)
"""

import sys
import numpy as np
from scipy import stats

sys.path.insert(0, "temp_analysis")
from bandwidth_study import (DENSITIES, K_STD, _scale, h_lscv, ise, ks,
                             parzen_cdf, parzen_pdf)

SEEDS = range(8)
N = 500


def rectify_cdf(cdf_on_grid, grid):
    """Copia fedele di training.py:120-130."""
    rc = np.maximum.accumulate(np.asarray(cdf_on_grid, float))
    lo, hi = float(rc[0]), float(rc[-1])
    if hi > lo:
        rc = (rc - lo) / (hi - lo)
    return rc, np.clip(np.gradient(rc, np.asarray(grid, float)), 0.0, None)


def ecdf(x, g):
    return np.searchsorted(np.sort(x), g, side="right") / x.size


# =============================================================== E1
print("=" * 98)
print("E1  LA RETTIFICA E' SENSIBILE ALLA GRIGLIA")
print("    Test isolante: applichiamo `rectify_cdf` a una CDF di Parzen che e' GIA' valida")
print("    (monotona, in [0,1]). Su una stima valida dovrebbe essere un'operazione neutra.")
print("    Misuriamo quanto NON lo e', al variare dell'ampiezza della griglia.")
print()
print(f"    {'densita':24} {'k=1':>9} {'k=2':>9} {'k=3':>9} {'k=5':>9} {'k=10':>9}"
      f"   (k = quante h oltre min/max dei dati)")
KS = (1, 2, 3, 5, 10)
for d in DENSITIES[:8]:
    acc = {k: [] for k in KS}
    for s in SEEDS:
        x = d.sample(N, np.random.default_rng(s))
        h = h_lscv(x)                                  # non dipende da k: una volta sola
        for k in KS:
            g = np.linspace(x.min() - k * h, x.max() + k * h, 2001)
            base = parzen_cdf(g, x, h)                 # stima valida, gia' monotona
            rect, _ = rectify_cdf(base, g)             # dovrebbe essere neutra
            acc[k].append(np.max(np.abs(rect - base)))
    print(f"    {d.name:24}" + "".join(f"{np.mean(acc[k]):>9.4f}" for k in KS))
print()
print("    Il numero e' lo spostamento massimo introdotto dalla sola rettifica su una stima")
print("    che non aveva nulla da correggere. Va confrontato con il KS tipico (~0.03).")

# =============================================================== E2
print()
print("=" * 98)
print("E2  UNA REGOLA PER LA GRIGLIA COSTRUITA DAI DATI")
print("    Regola candidata: [min(x) - k*h, max(x) + k*h] con h scelto da LSCV.")
print("    Riportiamo la massa VERA che resta fuori dalla griglia (che la rettifica")
print("    ridistribuirebbe all'interno, cioe' un errore sistematico non visibile).")
print()
print(f"    {'densita':24} " + "".join(f"{'k=' + str(k):>11}" for k in (0, 1, 2, 3, 5, 10)))
worst = {k: 0.0 for k in (0, 1, 2, 3, 5, 10)}
K2 = (0, 1, 2, 3, 5, 10)
for d in DENSITIES:
    acc = {k: [] for k in K2}
    for s in SEEDS:
        x = d.sample(N, np.random.default_rng(s))
        h = h_lscv(x)                                  # una volta sola
        for k in K2:
            lo, hi = x.min() - k * h, x.max() + k * h
            acc[k].append(float(d.cdf(np.array([lo]))[0] + 1.0 - d.cdf(np.array([hi]))[0]))
    row = f"    {d.name:24}"
    for k in K2:
        m = float(np.mean(acc[k]))
        worst[k] = max(worst[k], m)
        row += f"{m:>11.2e}"
    print(row)
print()
print("    massa fuori griglia nel CASO PEGGIORE fra le densita':")
print("      " + "   ".join(f"k={k}: {v:.2e}" for k, v in worst.items()))

# =============================================================== E3
print()
print("=" * 98)
print("E3  DIAGNOSTICHE TRUTH-FREE: QUALI FUNZIONANO E QUALI SONO TRAPPOLE")
print("    Sui campioni del docente non avremo la verita'. Per ogni valore di h calcoliamo")
print("    l'errore VERO (ISE) e quattro indicatori calcolabili dai soli campioni; poi")
print("    misuriamo se gli indicatori ordinano gli h come li ordina la verita'.")
print()
SWEEP = np.geomspace(0.05, 3.0, 25)
names = ["LSCV (score)", "log-verosimiglianza LOO", "KS contro ECDF", "massa sulla griglia"]
rho = {nm: [] for nm in names}
eff = {nm: [] for nm in names}
for d in DENSITIES:
    for s in SEEDS:
        x = d.sample(N, np.random.default_rng(s))
        sd, rob = _scale(x)
        hs = SWEEP * min(sd, rob) * N ** (-0.2) / K_STD
        g = np.linspace(x.min() - 5 * hs[-1], x.max() + 5 * hs[-1], 1500)
        tp = d.pdf(g)
        emp = ecdf(x, g)
        true_ise, sc_lscv, sc_loo, sc_ecdf, sc_mass = [], [], [], [], []
        for h in hs:
            f = parzen_pdf(g, x, h)
            F = parzen_cdf(g, x, h)
            true_ise.append(ise(f, tp, g))
            u = 1.0 / (1.0 + np.exp(-(x[:, None] - x[None, :]) / h))
            kk = u * (1 - u)
            np.fill_diagonal(kk, 0.0)
            floo = kk.sum(axis=1) / ((x.size - 1) * h)
            sc_lscv.append(np.trapezoid(f ** 2, g) - 2.0 * floo.mean())
            sc_loo.append(-np.log(np.maximum(floo, 1e-300)).mean())   # segno: piu' basso = meglio
            sc_ecdf.append(ks(F, emp))
            sc_mass.append(abs(np.trapezoid(f, g) - 1.0))
        true_ise = np.array(true_ise)
        best = true_ise.min()
        for nm, sc in zip(names, (sc_lscv, sc_loo, sc_ecdf, sc_mass)):
            sc = np.array(sc)
            rho[nm].append(stats.spearmanr(sc, true_ise).statistic)
            eff[nm].append(true_ise[int(np.argmin(sc))] / best)

print(f"    {'indicatore':26} {'rho di Spearman con ISE vero':>30} {'efficienza se lo si usa':>26}")
for nm in names:
    r = np.array(rho[nm]); e = np.array(eff[nm])
    print(f"    {nm:26} {np.nanmean(r):>16.3f} (mediana {np.nanmedian(r):+.2f}) "
          f"{np.mean(e):>14.2f} (peggiore {np.max(e):.1f})")
print()
print("    rho vicino a +1 = l'indicatore ordina gli h come la verita' (utile).")
print("    rho vicino a  0 o negativo = l'indicatore non informa, o inganna.")
print("    'efficienza se lo si usa' = quanto costa scegliere h minimizzando quell'indicatore.")
