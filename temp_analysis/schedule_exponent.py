"""Lo schedule h_n = h1/sqrt(n): quanto e' distante dal tasso ottimo, e quanto costa.

Il docente e' legato alla forma h_n = h1/sqrt(n) (Duda & Hart), che soddisfa le condizioni
di consistenza (h_n -> 0 e n*h_n -> infinito). La domanda non e' se sia legittima -- lo e' --
ma due altre, misurabili:

  A) con che esponente scala davvero la finestra ottima?  h* ~ n^(-p): stimiamo p.
  B) quanto costa imporre p = 1/2 invece del p misurato, se h1 viene calibrato a un budget
     e usato a un altro? (a budget FISSO non costa nulla: e' il punto da dimostrare)

Nessun oracolo entra nella pipeline: serve solo a misurare il tasso.
"""

import sys
import numpy as np

sys.path.insert(0, "temp_analysis")
from bandwidth_study import DENSITIES, K_STD, _scale, ise, parzen_pdf

NS = (100, 250, 500, 1000, 2000)
SEEDS = range(6)
SWEEP = np.geomspace(0.003, 3.0, 45)
SUBSET = ["MW01 gaussian", "MW06 bimodal", "MW09 trimodal", "MW10 claw",
          "MW12 asymmetric claw", "REPO trimodale", "REPO 5 mode strette",
          "REPO 6 mode scale miste"]
DS = [d for d in DENSITIES if d.name in SUBSET]


def oracle_h(mix, n, seed):
    x = mix.sample(n, np.random.default_rng(seed))
    g = mix.grid()
    tp = mix.pdf(g)
    sd, rob = _scale(x)
    hs = SWEEP * min(sd, rob) * n ** (-0.2) / K_STD
    I = np.array([ise(parzen_pdf(g, x, h), tp, g) for h in hs])
    return float(hs[int(np.argmin(I))]), hs, I, x, g, tp


print("=" * 96)
print("A  ESPONENTE EMPIRICO: h* ~ n^(-p).  Lo schedule del corso impone p = 0.5;")
print("   la teoria MISE per la pdf dice p = 0.2.  Regressione di log h* su log n.")
print(f"   {'densita':24} " + "".join(f"{'n=' + str(n):>10}" for n in NS) + f"{'p stimato':>12}")
rows = {}
for d in DS:
    hstar = []
    for n in NS:
        hstar.append(np.mean([oracle_h(d, n, s)[0] for s in SEEDS]))
    p = -np.polyfit(np.log(NS), np.log(hstar), 1)[0]
    rows[d.name] = (hstar, p)
    print(f"   {d.name:24} " + "".join(f"{v:>10.4f}" for v in hstar) + f"{p:>12.3f}")
ps = np.array([v[1] for v in rows.values()])
print(f"   -> p medio = {ps.mean():.3f}  (intervallo {ps.min():.3f} - {ps.max():.3f});"
      f"  teoria pdf = 0.200,  schedule del corso = 0.500")

print()
print("=" * 96)
print("B  COSTO DI IMPORRE p = 1/2.  h1 viene calibrato all'ORACOLO su n_cal e poi usato")
print("   su n_uso tramite h = h1/sqrt(n_uso).  Efficienza ISE rispetto all'oracolo di n_uso.")
print("   ATTENZIONE alla lettura della diagonale (n_cal = n_uso): NON vale 1.00, perche' h1 e'")
print("   calibrato sull'ottimo MEDIO fra campioni mentre il denominatore e' l'ottimo del singolo")
print("   campione. Il suo scarto da 1 misura il costo di usare un unico h1 per tutti i campioni")
print("   allo stesso n, che e' un costo REALE e non un artefatto. Il costo dello SCHEDULE e'")
print("   invece la differenza fra le celle fuori diagonale e la diagonale della stessa colonna.")
print(f"   {'densita':24} {'n_cal':>6} " + "".join(f"{'uso ' + str(n):>10}" for n in NS))
for d in DS:
    hstar, _ = rows[d.name]
    for i, ncal in enumerate((100, 500, 2000)):
        h1 = hstar[NS.index(ncal)] * np.sqrt(ncal)          # h1 calibrato a n_cal
        line = f"   {d.name if i == 0 else '':24} {ncal:>6} "
        for j, nuse in enumerate(NS):
            effs = []
            for s in SEEDS:
                _, hs, I, x, g, tp = oracle_h(d, nuse, s)
                h_used = h1 / np.sqrt(nuse)
                e = ise(parzen_pdf(g, x, float(np.clip(h_used, hs[0], hs[-1]))), tp, g)
                effs.append(e / I.min())
            line += f"{np.mean(effs):>10.2f}"
        print(line)
