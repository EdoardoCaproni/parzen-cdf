"""Quanto deve essere stretta la finestra del maestro?

Il report dice che le etichette si costruiscono con meta' della finestra scelta per lo
stimatore, e spiega la direzione: una finestra stretta produce etichette rumorose ma poco
distorte, e una rete di capacita' limitata media via il rumore invece di seguirlo (ricetta
PNN, Trentin). La direzione e' di principio.

Il fattore due no: era una scelta tonda, mai misurata. Il gruppo l'ha chiesto in revisione,
e la sezione 6.2 prometteva perfino che il capitolo 9 lo riportasse, mentre il capitolo 9
misura un'altra cosa (la rete contro il suo maestro, non 0.5 contro 1.0). Questo script
chiude la promessa.

Si spazza ``teacher_scale`` su entrambi i lati di 0.5, sui tre bersagli della tabella di
capacita' e con cinque semi. Cinque perche' in una misura precedente il vantaggio del
maestro affilato oscillava fra +25% e -4% con tre semi: sotto questa soglia la quantita' non
e' misurabile e qualunque minimo sarebbe rumore.

    python temp_analysis/teacher_width.py > temp_analysis/teacher_width.txt
"""

import numpy as np

from parzen_cdf import data, parzen
from parzen_cdf.diagnostics import report_domain
from parzen_cdf.estimate import run_from_samples

N = 500
SEEDS = range(5)
SCALES = (0.25, 0.5, 0.75, 1.0, 1.5)
CASES = {
    "trimodale": data.asymmetric_trimodal(),
    "5 mode strette": data.GaussianMixture1D([.2] * 5, [-8, -4, 0, 4, 8], [.35] * 5),
    "6 mode scale miste": data.GaussianMixture1D([.25, .2, .15, .15, .15, .10],
                                                 [-6, -3, 0, 1.2, 4, 7],
                                                 [1.2, .4, .25, .8, .5, 1.5]),
}

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def ise(a, b, g):
    return float(_trapezoid((a - b) ** 2, g))


def ks(a, b):
    return float(np.max(np.abs(a - b)))


print("=" * 92)
print(f"FINESTRA DEL MAESTRO: teacher_scale volte la finestra scelta per lo stimatore")
print(f"n = {N}, {len(list(SEEDS))} semi, J = 12. Media su semi di ISE della densita' e KS "
      f"della CDF.")
print("Il maestro e' la CDF di Parzen leave-one-out; scale = 1 usa la stessa finestra dello")
print("stimatore, scale < 1 la stringe deliberatamente.")
print("=" * 92)

riassunto = {}
for nome, mix in CASES.items():
    prep = []
    for s in SEEDS:
        x = mix.sample(N, np.random.default_rng(s))
        lo, hi = report_domain(x)
        g = np.linspace(lo, hi, 3001)
        prep.append((x, g, mix.pdf(g), mix.cdf(g)))

    print()
    print(f"--- {nome}")
    print(f"   {'scale':>7} {'ISE pdf':>10} {'KS cdf':>10}   {'ISE/min':>8}")

    ises, kss = [], []
    for sc in SCALES:
        vi, vk = [], []
        for x, g, tp, tc in prep:
            est = run_from_samples(x, epochs=6000, seed=0, teacher_scale=sc)
            vi.append(ise(est.pdf(g), tp, g))
            vk.append(ks(est.cdf(g), tc))
        ises.append(float(np.mean(vi)))
        kss.append(float(np.mean(vk)))

    # il maestro da solo, come riferimento: la rete lo batte a ogni scala?
    vp = [ise(parzen.parzen_pdf(g, x, parzen.lscv_bandwidth(x)), tp, g) for x, g, tp, _ in prep]
    vpk = [ks(parzen.parzen_cdf(g, x, parzen.lscv_bandwidth(x)), tc) for x, g, _, tc in prep]

    base = min(ises)
    for sc, i, k in zip(SCALES, ises, kss):
        marchio = "  <- min" if i == base else ""
        print(f"   {sc:>7.2f} {i:>10.5f} {k:>10.4f}   {i / base:>8.2f}{marchio}")
    print(f"   {'Parzen':>7} {np.mean(vp):>10.5f} {np.mean(vpk):>10.4f}")
    riassunto[nome] = (SCALES[int(np.argmin(ises))], max(ises) / base, ises)

print()
print("=" * 92)
print("RIASSUNTO")
print(f"   {'caso':22} {'scale migliore':>15} {'peggio/migliore':>17}")
for nome, (best, spread, _) in riassunto.items():
    print(f"   {nome:22} {best:>15.2f} {spread:>17.2f}x")

# La domanda che conta non e' dove cade il minimo su un singolo caso, che con cinque semi
# resta rumoroso, ma se la scelta del fattore sposti l'errore in modo apprezzabile.
tutti = np.array([r[2] for r in riassunto.values()])          # casi x scale
peggiore = tutti.max(axis=0) / tutti.max(axis=0).min()
print()
print("   caso peggiore fra i tre bersagli, normalizzato al migliore fra le scale:")
for sc, v in zip(SCALES, peggiore):
    print(f"     scale {sc:.2f}: {v:.2f}x")
print()
print("   Il criterio e' lo stesso della scelta di J: il bersaglio e' una distribuzione sola")
print("   e ignota, quindi conta il caso peggiore, non la media fra i casi.")
