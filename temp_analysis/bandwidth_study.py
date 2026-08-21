"""Studio indipendente sulla scelta della finestra h (equivalentemente h1 = h*sqrt(n)).

Metodo, in breve:
  - banco di prova = le 13 densita' di Marron & Wand (1992) verificabili dal sorgente del
    pacchetto R nor1mix, piu' i 3 casi usati dal repo. Sono misture di normali, quindi pdf e
    CDF esatte in forma chiusa. NON sono state scelte da noi: e' il benchmark standard della
    letteratura sui selettori di banda, e include densita' pesantemente multimodali (claw,
    double claw, asymmetric claw) che sono il caso peggiore dichiarato dal docente.
  - ogni selettore usa SOLO i campioni. L'oracolo (h che minimizza l'errore vero) e' un
    limite, non un selettore, e serve a normalizzare.
  - metrica primaria: ISE della pdf (e' il deliverable). Secondaria: KS della CDF.
  - aggregazione per EFFICIENZA relativa = errore(selettore) / errore(oracolo) >= 1, che
    rende confrontabili densita' con scale di errore diversissime.
  - le regole con una costante libera sono calibrate in LEAVE-ONE-DENSITY-OUT: la costante
    usata su una densita' e' calibrata su tutte le altre. Senza questo si ripeterebbe
    esattamente l'errore metodologico che stiamo criticando.

Kernel: logistico (la scelta del progetto), std = pi/sqrt(3) ~ 1.8138. Le regole classiche
sono derivate per kernel gaussiano (std 1) e vanno divise per questo fattore.
"""

import numpy as np
from scipy.special import erf, expit

K_STD = float(np.sqrt(np.pi ** 2 / 3.0))          # deviazione standard del kernel logistico

# ----------------------------------------------------------------------------- misture

class NormMix:
    def __init__(self, name, w, m, s):
        w = np.asarray(w, float); self.w = w / w.sum()
        self.m = np.asarray(m, float); self.s = np.asarray(s, float)
        self.name = name
        assert self.w.shape == self.m.shape == self.s.shape, name
        assert abs(np.asarray(w, float).sum() - 1.0) < 1e-9, f"{name}: pesi non sommano a 1"

    def pdf(self, x):
        z = (np.asarray(x, float)[..., None] - self.m) / self.s
        return (np.exp(-0.5 * z ** 2) / (self.s * np.sqrt(2 * np.pi))) @ self.w

    def cdf(self, x):
        z = (np.asarray(x, float)[..., None] - self.m) / (self.s * np.sqrt(2.0))
        return (0.5 * (1.0 + erf(z))) @ self.w

    def sample(self, n, rng):
        i = rng.choice(self.w.size, size=n, p=self.w)
        return rng.normal(self.m[i], self.s[i])

    def grid(self):
        lo = float((self.m - 6 * self.s).min()); hi = float((self.m + 6 * self.s).max())
        # passo <= min_sd/5, cosi' anche le componenti piu' strette sono risolte
        npts = int(np.clip((hi - lo) / (self.s.min() / 5.0) + 1, 2001, 24001))
        return np.linspace(lo, hi, npts)


def _mw():
    """Marron & Wand (1992), #1-#13. Parametri verificati sul sorgente di nor1mix
    (R/zMarrWand-dens.R). #14 e #15 sono esclusi: non e' stato possibile verificarne i
    parametri da una fonte primaria, e una densita' sbagliata invaliderebbe lo studio."""
    d = []
    d.append(NormMix("MW01 gaussian", [1], [0], [1]))
    d.append(NormMix("MW02 skewed unimodal", [.2, .2, .6], [0, .5, 13 / 12], [1, 2 / 3, 5 / 9]))
    sig = (2 / 3) ** np.arange(8)
    d.append(NormMix("MW03 strongly skewed", np.full(8, 1 / 8), 3 * (sig - 1), sig))
    d.append(NormMix("MW04 kurtotic unimodal", [2 / 3, 1 / 3], [0, 0], [1, .1]))
    d.append(NormMix("MW05 outlier", [.1, .9], [0, 0], [1, .1]))
    d.append(NormMix("MW06 bimodal", [.5, .5], [-1, 1], [2 / 3, 2 / 3]))
    d.append(NormMix("MW07 separated bimodal", [.5, .5], [-1.5, 1.5], [.5, .5]))
    d.append(NormMix("MW08 skewed bimodal", [.75, .25], [0, 1.5], [1, 1 / 3]))
    d.append(NormMix("MW09 trimodal", [.45, .45, .1], [-1.2, 1.2, 0], [3 / 5, 3 / 5, 1 / 4]))
    d.append(NormMix("MW10 claw", [.5] + [.1] * 5, [0] + list(np.arange(-1, 1.01, .5)),
                     [1] + [.1] * 5))
    d.append(NormMix("MW11 double claw", [.49, .49] + [.02 / 7] * 7,
                     [-1, 1] + list(np.arange(-1.5, 1.51, .5)), [2 / 3, 2 / 3] + [.01] * 7))
    d.append(NormMix("MW12 asymmetric claw", [.5] + [2.0 ** (1 - l) / 31 for l in range(-2, 3)],
                     [0] + [l + .5 for l in range(-2, 3)],
                     [1] + [2.0 ** (-l) / 10 for l in range(-2, 3)]))
    d.append(NormMix("MW13 asym double claw", [.46, .46] + [.01 / 3] * 3 + [.07 / 3] * 3,
                     [-1, 1, -1.5, -1, -.5, .5, 1, 1.5],
                     [2 / 3, 2 / 3, .01, .01, .01, .07, .07, .07]))
    return d


def _repo():
    """I tre casi su cui il repo ha calibrato la regola 1.5*sigma, per poter parlare
    direttamente al suo benchmark."""
    return [
        NormMix("REPO trimodale", [.3, .5, .2], [-2, 1, 4], [.5, 1, .3]),
        NormMix("REPO 5 mode strette", [.2] * 5, [-8, -4, 0, 4, 8], [.35] * 5),
        NormMix("REPO 6 mode scale miste", [.25, .2, .15, .15, .15, .10],
                [-6, -3, 0, 1.2, 4, 7], [1.2, .4, .25, .8, .5, 1.5]),
    ]


DENSITIES = _mw() + _repo()

# ----------------------------------------------------------------------------- stimatore

def parzen_pdf(x, s, h):
    u = expit((x[:, None] - s[None, :]) / h)
    return (u * (1 - u)).mean(axis=1) / h


def parzen_cdf(x, s, h):
    return expit((x[:, None] - s[None, :]) / h).mean(axis=1)


def ise(a, b, g):
    return float(np.trapezoid((a - b) ** 2, g))


def ks(a, b):
    return float(np.max(np.abs(a - b)))


# ----------------------------------------------------------------------------- selettori
# Tutti ricevono solo il campione. Restituiscono h per il kernel LOGISTICO.

def _scale(x):
    sd = x.std(ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    return sd, (iqr / 1.349 if iqr > 0 else sd)


def h_repo_start(x):                      # h1 = 1.0  ->  h = 1/sqrt(n)
    return 1.0 / np.sqrt(x.size)


def h_sigma_rule(x):                      # h1 = 1.5*sigma_hat  (regola del repo)
    return 1.5 * x.std(ddof=1) / np.sqrt(x.size)


def h_silverman(x):                       # Silverman, scala robusta, variance-matched
    sd, rob = _scale(x)
    return 0.9 * min(sd, rob) * x.size ** (-0.2) / K_STD


def h_scott(x):                           # Scott, variance-matched
    return 1.06 * x.std(ddof=1) * x.size ** (-0.2) / K_STD


def _loo_density(x, h):
    d = expit((x[:, None] - x[None, :]) / h)
    k = d * (1 - d)
    np.fill_diagonal(k, 0.0)
    return k.sum(axis=1) / ((x.size - 1) * h)


def _cand(x, lo=0.02, hi=2.0, m=30):
    sd, rob = _scale(x)
    return np.geomspace(lo, hi, m) * min(sd, rob) * x.size ** (-0.2) / K_STD


def h_mlcv(x):                            # CV di massima verosimiglianza leave-one-out
    c = _cand(x)
    sc = [np.log(np.maximum(_loo_density(x, h), 1e-300)).mean() for h in c]
    return float(c[int(np.argmax(sc))])


def h_lscv(x):                            # CV dei minimi quadrati (ISE non distorto)
    c = _cand(x)
    sd, _ = _scale(x)
    g = np.linspace(x.min() - 3 * sd, x.max() + 3 * sd, 1500)
    sc = []
    for h in c:
        f = parzen_pdf(g, x, h)
        sc.append(np.trapezoid(f ** 2, g) - 2.0 * _loo_density(x, h).mean())
    return float(c[int(np.argmin(sc))])


def _phi(u):
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def _theta(x, g, r):
    """theta_r = int f^(r) f  stimato con kernel gaussiano di ampiezza g (Sheather-Jones)."""
    u = (x[:, None] - x[None, :]) / g
    if r == 4:
        d = (u ** 4 - 6 * u ** 2 + 3) * _phi(u)
    elif r == 6:
        d = (u ** 6 - 15 * u ** 4 + 45 * u ** 2 - 15) * _phi(u)
    else:
        raise ValueError(r)
    return float(d.sum() / (x.size ** 2 * g ** (r + 1)))


def h_plugin(x):
    """Plug-in a due stadi. Derivazione completa (nessuna formula presa a memoria):

      h_AMISE = [ R(K) / (n * mu2(K)^2 * theta4) ]^(1/5),  theta4 = int (f'')^2 > 0
      con K gaussiano: R(K) = 1/(2 sqrt(pi)), mu2 = 1.

      theta4 va stimato. Stimatore a nucleo:
        theta4_hat(g) = 1/(n^2 g^5) * somma_ij phi4((x_i - x_j)/g),  phi4(u)=(u^4-6u^2+3)phi(u)
      con ampiezza pilota AMSE-ottima  g = [ -2 K4(0) / (mu2 * theta6 * n) ]^(1/7),
      K4(0) = 3/sqrt(2 pi), e theta6 preso dal riferimento normale:
        theta4_N = 3/(8 sqrt(pi) s^5),   theta6_N = -15/(16 sqrt(pi) s^7).
      Sostituendo:  g = (96/(15 sqrt 2))^(1/7) * s * n^(-1/7) ~ 1.2407 * s * n^(-1/7).

    Validazione: su dati gaussiani deve restituire ~1.06 * s * n^(-1/5) (vedi V2).
    Il risultato e' per kernel gaussiano; si divide per K_STD per il logistico.
    """
    n = x.size
    sd, rob = _scale(x)
    s = min(sd, rob)
    g = (96.0 / (15.0 * np.sqrt(2.0))) ** (1 / 7) * s * n ** (-1 / 7)
    t4 = _theta(x, g, 4)
    if not np.isfinite(t4) or t4 <= 0:
        return h_silverman(x)                        # ripiego dichiarato
    h_gauss = (1.0 / (2 * np.sqrt(np.pi)) / (n * t4)) ** 0.2
    if not np.isfinite(h_gauss) or h_gauss <= 0:
        return h_silverman(x)
    return float(h_gauss / K_STD)


def h_knn_factory(c):
    def f(x):
        k = max(2, int(round(np.sqrt(x.size))))
        xs = np.sort(x)
        d = np.abs(xs[:, None] - xs[None, :])
        dk = np.partition(d, k, axis=1)[:, k]        # distanza al k-esimo vicino
        return float(c * np.median(dk))
    return f


def h_const_factory(c, robust=True):
    """Regola a costante libera con l'ESPONENTE CORRETTO: h = c * scala * n^(-1/5)."""
    def f(x):
        sd, rob = _scale(x)
        return float(c * (min(sd, rob) if robust else sd) * x.size ** (-0.2) / K_STD)
    return f


def h_sigma_sqrtn_factory(c):
    """La forma della regola del repo (h = c*sigma/sqrt(n)) ma con c ricalibrata."""
    def f(x):
        return float(c * x.std(ddof=1) / np.sqrt(x.size))
    return f


# =================================================================== driver dello studio

SEL_FIXED = {                       # selettori senza costanti da calibrare
    "repo h1=1.0":      h_repo_start,
    "repo 1.5*sigma":   h_sigma_rule,
    "Silverman robusto": h_silverman,
    "Scott":            h_scott,
    "plug-in 2 stadi":  h_plugin,
    "LSCV":             h_lscv,
    "MLCV (KL)":        h_mlcv,
}

# famiglie con una costante libera: forma -> (fabbrica, griglia di costanti)
SEL_FAMILIES = {
    "c*sigma/sqrt(n)  [forma repo]": (h_sigma_sqrtn_factory, np.geomspace(0.05, 6.0, 40)),
    "c*scala*n^(-1/5) [esponente corretto]": (h_const_factory, np.geomspace(0.05, 4.0, 40)),
    "c*dist k-esimo vicino": (h_knn_factory, np.geomspace(0.05, 6.0, 40)),
}

H_SWEEP = np.geomspace(0.002, 3.0, 70)      # in unita' di scala_robusta * n^(-1/5) / K_STD


def sweep(mix, n, seed):
    """Per un campione: curva ISE(h) e KS(h) su una griglia fitta di h, piu' i campioni."""
    x = mix.sample(n, np.random.default_rng(seed))
    g = mix.grid()
    tp, tc = mix.pdf(g), mix.cdf(g)
    sd, rob = _scale(x)
    unit = min(sd, rob) * n ** (-0.2) / K_STD
    hs = H_SWEEP * unit
    I = np.empty(hs.size); K = np.empty(hs.size)
    for i, h in enumerate(hs):
        I[i] = ise(parzen_pdf(g, x, h), tp, g)
        K[i] = ks(parzen_cdf(g, x, h), tc)
    return x, hs, I, K


def at(hs, vals, h):
    """Valore interpolato (log-lineare in h) della curva, con estremi tagliati."""
    lh = np.log(np.clip(h, hs[0], hs[-1]))
    return float(np.interp(lh, np.log(hs), vals))


def run(n, seeds, densities=DENSITIES):
    """Restituisce res[densita'][selettore] = lista di (eff_ISE, eff_KS) sui semi,
    e le curve grezze per la calibrazione delle famiglie."""
    res = {d.name: {} for d in densities}
    curves = {d.name: [] for d in densities}
    for d in densities:
        for s in seeds:
            x, hs, I, K = sweep(d, n, s)
            oI, oK = I.min(), K.min()
            curves[d.name].append((x, hs, I, K, oI, oK))
            for name, fn in SEL_FIXED.items():
                h = fn(x)
                res[d.name].setdefault(name, []).append((at(hs, I, h) / oI, at(hs, K, h) / oK))
    return res, curves


def calibrate_lodo(curves, family, cgrid, densities):
    """Costante calibrata in leave-one-density-out: per ogni densita' d la costante e'
    quella che minimizza l'efficienza ISE media su TUTTE LE ALTRE densita'."""
    names = [d.name for d in densities]
    # eff[c][densita'] = efficienza ISE media sui semi
    eff = np.empty((cgrid.size, len(names)))
    for ic, c in enumerate(cgrid):
        fn = family(c)
        for j, nm in enumerate(names):
            e = [at(hs, I, fn(x)) / oI for (x, hs, I, K, oI, oK) in curves[nm]]
            eff[ic, j] = float(np.mean(e))
    out, cstar = {}, {}
    for j, nm in enumerate(names):
        others = [k for k in range(len(names)) if k != j]
        ic = int(np.argmin(eff[:, others].mean(axis=1)))
        cstar[nm] = float(cgrid[ic])
        fn = family(cgrid[ic])
        out[nm] = [(at(hs, I, fn(x)) / oI, at(hs, K, fn(x)) / oK)
                   for (x, hs, I, K, oI, oK) in curves[nm]]
    # costante ottima per densita' (in-sample): serve a mostrare quanto varia
    cbest = {nm: float(cgrid[int(np.argmin(eff[:, j]))]) for j, nm in enumerate(names)}
    return out, cstar, cbest
