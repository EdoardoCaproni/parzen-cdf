"""I bordi netti sono l'unica debolezza vera: piu' componenti la riducono senza costi altrove?

P-02 ha mostrato che le code pesanti non sono un problema (la rete batte perfino il Parzen),
mentre i bordi netti costano: ISE fino a 13.8 volte il riferimento gaussiano, e sul caso
"due uniformi separate" la rete perde 2.2x contro il suo stesso maestro.

Ipotesi: con J = 12 componenti logistiche la rete non ha abbastanza gradi di liberta' per
approssimare un salto, mentre il Parzen con n = 500 nuclei ne ha. Se cosi' fosse, alzare J
recupererebbe il divario. Ma J e' stato scelto a 12 sulle misture gaussiane (D-08): la
domanda vera e' se alzarlo costi qualcosa LI'.
"""
import numpy as np
import scipy.stats as st
from parzen_cdf import parzen
from parzen_cdf.diagnostics import report_domain
from parzen_cdf.estimate import run_from_samples


class Mix:
    def __init__(self, comps, weights=None):
        self.d = list(comps)
        w = np.ones(len(self.d)) if weights is None else np.asarray(weights, float)
        self.w = w / w.sum()

    def pdf(self, x):
        return sum(wi * di.pdf(x) for wi, di in zip(self.w, self.d))

    def sample(self, n, rng):
        idx = rng.choice(len(self.d), size=n, p=self.w)
        out = np.empty(n)
        for i, di in enumerate(self.d):
            m = idx == i
            out[m] = di.rvs(size=int(m.sum()), random_state=rng)
        return out


N, SEEDS, JS = 500, range(3), (12, 24, 48)
CASES = {
    "BORDI due uniformi separate": Mix([st.uniform(-3, 1), st.uniform(2, 1.5)], [.5, .5]),
    "BORDI uniforme":              Mix([st.uniform(-1, 2)]),
    "BORDI esponenziale":          Mix([st.expon(0, 1)]),
    "GAUSS trimodale":             Mix([st.norm(-2, .5), st.norm(1, 1), st.norm(4, .3)],
                                       [.3, .5, .2]),
    "GAUSS 6 mode scale miste":    Mix([st.norm(-6, 1.2), st.norm(-3, .4), st.norm(0, .25),
                                        st.norm(1.2, .8), st.norm(4, .5), st.norm(7, 1.5)],
                                       [.25, .2, .15, .15, .15, .10]),
}

print(f"ISE della pdf, media su {len(list(SEEDS))} semi, n = {N}")
print(f"{'caso':30} " + "".join(f"{'J=' + str(j):>12}" for j in JS) + f"{'Parzen':>12}")
for name, mix in CASES.items():
    dati = [(mix.sample(N, np.random.default_rng(s)), s) for s in SEEDS]
    prep = []
    for x, s in dati:
        lo, hi = report_domain(x)
        g = np.linspace(lo, hi, 3001)
        prep.append((x, g, mix.pdf(g)))
    row, ises = f"{name:30} ", []
    for J in JS:
        v = [float(np.trapezoid((run_from_samples(x, n_components=J, epochs=6000,
                                                  seed=0).pdf(g) - tp) ** 2, g))
             for x, g, tp in prep]
        ises.append(float(np.mean(v)))
        row += f"{np.mean(v):>12.5f}"
    vp = [float(np.trapezoid((parzen.parzen_pdf(g, x, parzen.lscv_bandwidth(x)) - tp) ** 2, g))
          for x, g, tp in prep]
    print(row + f"{np.mean(vp):>12.5f}" + f"   <- min a J={JS[int(np.argmin(ises))]}")
