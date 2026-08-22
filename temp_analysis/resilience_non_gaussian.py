"""P-02: quanto regge la pipeline su distribuzioni che non sono misture di gaussiane.

Tutte le calibrazioni del progetto (finestra, capacita', dominio) sono state fatte su
misture di normali. Il docente ha parlato di multimodalita', quindi e' l'ipotesi piu'
probabile, ma non e' una garanzia. Questo studio cerca i punti in cui la pipeline si rompe
o degrada, e distingue i difetti *riparabili a costo zero* da quelli strutturali.

Tre sospetti concreti, formulati prima di misurare:

  S1  ``report_domain`` usa la deviazione standard campionaria. Su code pesanti (Student-t
      con pochi gradi di liberta') sigma_hat e' instabile e per df <= 2 la varianza vera non
      esiste affatto: il dominio potrebbe dilatarsi a dismisura, e con esso perdersi la
      risoluzione dove i dati stanno davvero.

  S2  Il nucleo logistico ha code esponenziali. Una distribuzione a code polinomiali non e'
      rappresentabile bene: e' un limite dello stimatore di Parzen, non del nostro
      re-design, ma va quantificato.

  S3  La mistura di CDF logistiche ha densita' strettamente positiva su tutto R (teorema T2).
      Non puo' quindi rappresentare un bordo netto: uniforme ed esponenziale hanno un salto
      di densita' che verra' inevitabilmente arrotondato.
"""

import sys
import pathlib

import numpy as np
import scipy.stats as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parzen_cdf import parzen                                    # noqa: E402
from parzen_cdf.diagnostics import report_domain                 # noqa: E402
from parzen_cdf.estimate import run_from_samples                 # noqa: E402

N, SEEDS = 500, range(5)


class Mix:
    """Mistura di distribuzioni scipy congelate: pdf, cdf e campionamento esatti."""

    def __init__(self, comps, weights=None):
        self.d = list(comps)
        w = np.ones(len(self.d)) if weights is None else np.asarray(weights, float)
        self.w = w / w.sum()

    def pdf(self, x):
        return sum(wi * di.pdf(x) for wi, di in zip(self.w, self.d))

    def cdf(self, x):
        return sum(wi * di.cdf(x) for wi, di in zip(self.w, self.d))

    def sample(self, n, rng):
        idx = rng.choice(len(self.d), size=n, p=self.w)
        out = np.empty(n)
        for i, di in enumerate(self.d):
            m = idx == i
            out[m] = di.rvs(size=int(m.sum()), random_state=rng)
        return out


CASES = {
    # riferimento: e' la famiglia su cui abbiamo calibrato tutto
    "RIF mistura gaussiana": Mix([st.norm(-2, .5), st.norm(1, 1), st.norm(4, .3)], [.3, .5, .2]),
    # S3: bordi netti
    "uniforme": Mix([st.uniform(-1, 2)]),
    "esponenziale": Mix([st.expon(0, 1)]),
    "due uniformi separate": Mix([st.uniform(-3, 1), st.uniform(2, 1.5)], [.5, .5]),
    # S1/S2: code pesanti
    "Student-t df=3": Mix([st.t(3)]),
    "Student-t df=2 (varianza infinita)": Mix([st.t(2)]),
    "Student-t df=1 (Cauchy, media infinita)": Mix([st.t(1)]),
    "lognormale": Mix([st.lognorm(0.7)]),
    # famiglie miste, il caso realistico se il docente non usa gaussiane
    "gauss + uniforme": Mix([st.norm(-2, .4), st.uniform(1, 2)], [.5, .5]),
    "gauss + esponenziale": Mix([st.norm(-2, .4), st.expon(1, .8)], [.5, .5]),
    "gauss + Student-t df=3": Mix([st.norm(-3, .4), st.t(3, loc=2)], [.5, .5]),
    "laplace + uniforme + gauss": Mix([st.laplace(-4, .6), st.uniform(-1, 2), st.norm(4, .5)],
                                      [.35, .35, .30]),
}


def evaluate(mix, x, cdf_fn, pdf_fn):
    lo, hi = report_domain(x)
    g = np.linspace(lo, hi, 3001)
    tc, tp = mix.cdf(g), mix.pdf(g)
    c, d = np.asarray(cdf_fn(g), float), np.asarray(pdf_fn(g), float)
    fuori = float(mix.cdf(np.array([lo]))[0] + 1.0 - mix.cdf(np.array([hi]))[0])
    return {
        "ks": float(np.max(np.abs(c - tc))),
        "ise": float(np.trapezoid((d - tp) ** 2, g)),
        "massa": float(np.trapezoid(d, g)),
        "ampiezza_dominio": float(hi - lo),
        "rapporto_dominio_dati": float((hi - lo) / (x.max() - x.min())),
        "massa_fuori": fuori,
    }


print("=" * 108)
print(f"P-02  RESILIENZA A DISTRIBUZIONI NON GAUSSIANE  (n = {N}, {len(list(SEEDS))} semi)")
print("      'dom/dati' = ampiezza del dominio diviso l'ampiezza dei dati: se esplode, il")
print("      sospetto S1 e' confermato. 'fuori' = massa vera lasciata fuori dal dominio.")
print()
hdr = (f"  {'caso':38} {'stim.':8} {'KS':>8} {'ISE':>9} {'massa':>9} "
       f"{'dom/dati':>9} {'fuori':>9}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

righe = {}
for name, mix in CASES.items():
    acc = {"rete": [], "parzen": []}
    for s in SEEDS:
        x = mix.sample(N, np.random.default_rng(s))
        est = run_from_samples(x, epochs=6000, seed=0)
        acc["rete"].append(evaluate(mix, x, est.cdf, est.pdf))
        h = parzen.lscv_bandwidth(x)
        acc["parzen"].append(evaluate(mix, x, lambda g: parzen.parzen_cdf(g, x, h),
                                      lambda g: parzen.parzen_pdf(g, x, h)))
    for k in ("parzen", "rete"):
        m = {kk: float(np.mean([r[kk] for r in acc[k]])) for kk in acc[k][0]}
        righe[(name, k)] = m
        etichetta = {"parzen": "Parzen", "rete": "RETE"}[k]
        print(f"  {name if k == 'parzen' else '':38} {etichetta:8} {m['ks']:>8.4f} "
              f"{m['ise']:>9.5f} {m['massa']:>9.5f} {m['rapporto_dominio_dati']:>9.2f} "
              f"{m['massa_fuori']:>9.1e}")

print()
print("=" * 108)
print("LETTURA DEI TRE SOSPETTI")
rif = righe[("RIF mistura gaussiana", "rete")]
print(f"\nriferimento gaussiano: KS {rif['ks']:.4f}, ISE {rif['ise']:.5f}, "
      f"dom/dati {rif['rapporto_dominio_dati']:.2f}")

print("\nS1  il dominio esplode sulle code pesanti?")
for name in ("Student-t df=3", "Student-t df=2 (varianza infinita)",
             "Student-t df=1 (Cauchy, media infinita)"):
    m = righe[(name, "rete")]
    print(f"    {name:42} dom/dati {m['rapporto_dominio_dati']:6.2f}   "
          f"massa sul dominio {m['massa']:.4f}   fuori {m['massa_fuori']:.2e}")

print("\nS2/S3  quanto costano code pesanti e bordi netti (ISE rispetto al riferimento)?")
for name in ("uniforme", "esponenziale", "due uniformi separate", "Student-t df=3",
             "lognormale", "gauss + uniforme", "gauss + esponenziale"):
    m = righe[(name, "rete")]
    p = righe[(name, "parzen")]
    print(f"    {name:42} ISE rete {m['ise']:8.5f}  ({m['ise']/rif['ise']:5.1f}x il rif.)"
          f"   Parzen {p['ise']:8.5f}   rete/Parzen {m['ise']/max(p['ise'],1e-12):5.2f}")

print("\nla rete perde rispetto al suo maestro Parzen? (rapporto ISE rete/Parzen > 1 = si')")
peggio = [(n, righe[(n, 'rete')]['ise'] / max(righe[(n, 'parzen')]['ise'], 1e-12))
          for n in CASES]
for n, r in sorted(peggio, key=lambda t: -t[1])[:5]:
    print(f"    {n:42} {r:5.2f}")
