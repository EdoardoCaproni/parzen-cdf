"""Evidenze numeriche per il re-design della rete (docs/redesign_network.md).

Implementazione da zero in NumPy (torch non e' richiesto) delle architetture confrontate,
con verifica dei gradienti per differenze finite. Ogni numero citato nel documento di
re-design e' prodotto da questo file.

Architetture (una sola cifra nascosta, J unita', input 1-D):

  A  repo, non vincolata   F(x) = s( SUM_j w_j s(a_j x + b_j) + c )          a,w liberi
  B  repo, monotone=True   idem, con a = softplus(alpha), w = softplus(omega)
  C  proposta              F(x) = SUM_j pi_j s(a_j x + b_j),  pi = softmax(u),
                                                              a  = softplus(alpha)

s = sigmoide logistica. A e B replicano CDFNet (models.py); C e' la trasformazione
Deep Sigmoidal Flow (Huang et al., ICML 2018) senza il logit di uscita.

Ogni architettura viene provata sia con l'inizializzazione del repo sia con
l'inizializzazione sui quantili, per separare l'effetto dell'architettura da quello
dell'inizializzazione.

Esperimenti:
  E1 verifica dei gradienti e della monotonia strutturale
  E2 comportamento asintotico (code)
  E3 monotonia: risoluzione dei punti di collocazione della penalita' soft
  E4 accuratezza (KS sulla CDF, ISE sulla pdf) vs Parzen, a parita' di inizializzazione
  E5 massa e coerenza pdf/CDF, effetto del clamp
  E6 running max vs regressione isotona (PAVA)
  E7 invarianza per traslazione e scala
  E8 capacita': quante componenti servono a C
  E9 C addestrata sulla ECDF: serve ancora una finestra di Parzen?
"""

import numpy as np
from scipy.special import erf, expit

# ----------------------------------------------------------------------------- utilita'

def softplus(z):
    return np.logaddexp(0.0, z)


def inv_softplus(y):
    y = np.asarray(y, dtype=float)
    return np.where(y > 30.0, y, np.log(np.expm1(np.minimum(y, 30.0))))


def d_softplus(z):
    return expit(z)


def parzen_cdf(x, samples, h):
    return expit((x[:, None] - samples[None, :]) / h).mean(axis=1)


def parzen_pdf(x, samples, h):
    s = expit((x[:, None] - samples[None, :]) / h)
    return (s * (1 - s)).mean(axis=1) / h


def ks(a, b):
    return float(np.max(np.abs(a - b)))


def ise(a, b, g):
    return float(np.trapezoid((a - b) ** 2, g))


def pava(y):
    """Regressione isotona (pool adjacent violators): proiezione L2 sul cono monotono."""
    y = np.asarray(y, dtype=float)
    lvl, wts = [], []
    for v in y:
        lvl.append(v)
        wts.append(1.0)
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            v2, w2 = lvl.pop(), wts.pop()
            v1, w1 = lvl.pop(), wts.pop()
            lvl.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    out = np.empty(y.size)
    i = 0
    for v, w in zip(lvl, wts):
        k = int(round(w))
        out[i:i + k] = v
        i += k
    return out


def rectify_cdf(cdf_on_grid, grid):
    """Copia fedele di training.py:120-130 (running max + rescale, pdf per differenze finite)."""
    rc = np.maximum.accumulate(np.asarray(cdf_on_grid, dtype=float))
    lo, hi = float(rc[0]), float(rc[-1])
    if hi > lo:
        rc = (rc - lo) / (hi - lo)
    pdf = np.clip(np.gradient(rc, np.asarray(grid, dtype=float)), 0.0, None)
    return rc, pdf


# ----------------------------------------------------------------------------- misture

class Mix:
    def __init__(self, w, m, s):
        self.w = np.asarray(w, float)
        self.w = self.w / self.w.sum()
        self.m = np.asarray(m, float)
        self.s = np.asarray(s, float)

    def pdf(self, x):
        x = np.asarray(x, float)
        z = (x[..., None] - self.m) / self.s
        return (np.exp(-0.5 * z ** 2) / (self.s * np.sqrt(2 * np.pi))) @ self.w

    def cdf(self, x):
        x = np.asarray(x, float)
        z = (x[..., None] - self.m) / (self.s * np.sqrt(2.0))
        return (0.5 * (1.0 + erf(z))) @ self.w

    def sample(self, n, rng):
        idx = rng.choice(self.w.size, size=n, p=self.w)
        return rng.normal(self.m[idx], self.s[idx])

    def grid(self, n=2001):
        return np.linspace((self.m - 5 * self.s).min(), (self.m + 5 * self.s).max(), n)


CASES = {
    "trimodale":          Mix([.3, .5, .2], [-2, 1, 4], [.5, 1.0, .3]),
    "5-mode-strette":     Mix([.2] * 5, [-8, -4, 0, 4, 8], [.35] * 5),
    "6-mode-scale-miste": Mix([.25, .2, .15, .15, .15, .10],
                              [-6, -3, 0, 1.2, 4, 7], [1.2, .4, .25, .8, .5, 1.5]),
}


# ----------------------------------------------------------------------------- architetture

class ArchA:
    """Repo, non vincolata: F = s(SUM w_j s(a_j x + b_j) + c). theta = [a, b, w, c]."""
    monotone_by_design = False

    def __init__(self, J, quantile_init=False):
        self.J = J
        self.qi = quantile_init
        self.name = f"A{'q' if quantile_init else ''}"

    def _first_layer(self, x, rng):
        J = self.J
        if self.qi:
            centres = np.quantile(x, (np.arange(J) + 0.5) / J)
            width = max((x.max() - x.min()) / J, 1e-9)
            a = np.full(J, 1.0 / width)
            return a, -a * centres
        bound = np.sqrt(6.0 / (1.0 + J))          # xavier_uniform + bias 0, come models.py
        return rng.uniform(-bound, bound, J), np.zeros(J)

    def init(self, x, rng):
        a, b = self._first_layer(x, rng)
        bound2 = np.sqrt(6.0 / (self.J + 1.0))
        return np.concatenate([a, b, rng.uniform(-bound2, bound2, self.J), np.zeros(1)])

    def _unpack(self, th):
        J = self.J
        return th[:J], th[J:2 * J], th[2 * J:3 * J], th[3 * J]

    def _eff(self, th):
        return self._unpack(th)

    def forward(self, th, x):
        a, b, w, c = self._eff(th)
        return expit(expit(np.asarray(x, float)[:, None] * a + b) @ w + c)

    def dF_dx(self, th, x):
        a, b, w, c = self._eff(th)
        s = expit(np.asarray(x, float)[:, None] * a + b)
        F = expit(s @ w + c)
        return F * (1 - F) * ((s * (1 - s) * a) @ w)

    pdf = dF_dx

    def grads(self, th, x, y):
        a, b, w, c = self._eff(th)
        x = np.asarray(x, float)
        s = expit(x[:, None] * a + b)
        F = expit(s @ w + c)
        d = 2.0 * (F - y) / x.size * F * (1 - F)
        dz = d[:, None] * w * s * (1 - s)
        return np.concatenate([dz.T @ x, dz.sum(axis=0), s.T @ d, np.array([d.sum()])])


class ArchB(ArchA):
    """Repo monotone=True: a = softplus(alpha), w = softplus(omega)."""
    monotone_by_design = True

    def __init__(self, J, quantile_init=False):
        super().__init__(J, quantile_init)
        self.name = f"B{'q' if quantile_init else ''}"

    def init(self, x, rng):
        J = self.J
        if self.qi:
            a_eff, b = self._first_layer(x, rng)
        else:                                      # come models.py, ramo monotone
            scale = np.sqrt(6.0 / (1.0 + J))
            a_eff, b = np.maximum(rng.random(J) * scale, 1e-6), np.zeros(J)
        w_eff = np.maximum(rng.random(J) * np.sqrt(6.0 / (J + 1.0)), 1e-6)
        return np.concatenate([inv_softplus(a_eff), b, inv_softplus(w_eff), np.zeros(1)])

    def _eff(self, th):
        al, b, om, c = self._unpack(th)
        return softplus(al), b, softplus(om), c

    def grads(self, th, x, y):
        al, b, om, c = self._unpack(th)
        a, w = softplus(al), softplus(om)
        x = np.asarray(x, float)
        s = expit(x[:, None] * a + b)
        F = expit(s @ w + c)
        d = 2.0 * (F - y) / x.size * F * (1 - F)
        dz = d[:, None] * w * s * (1 - s)
        return np.concatenate([(dz.T @ x) * d_softplus(al), dz.sum(axis=0),
                               (s.T @ d) * d_softplus(om), np.array([d.sum()])])


class ArchC:
    """Proposta: F = SUM_j softmax(u)_j s(softplus(alpha_j) x + b_j). theta = [alpha, b, u]."""
    monotone_by_design = True

    def __init__(self, J, quantile_init=True):
        self.J = J
        self.qi = quantile_init
        self.name = f"C{'' if quantile_init else '(init casuale)'}"

    def init(self, x, rng):
        J = self.J
        if self.qi:
            centres = np.quantile(x, (np.arange(J) + 0.5) / J)
            width = max((x.max() - x.min()) / J, 1e-9)
            a = np.full(J, 1.0 / width)
            b = -a * centres
        else:
            a = np.maximum(rng.random(J) * np.sqrt(6.0 / (1.0 + J)), 1e-6)
            b = np.zeros(J)
        return np.concatenate([inv_softplus(a), b, np.zeros(J)])

    def _unpack(self, th):
        J = self.J
        return th[:J], th[J:2 * J], th[2 * J:3 * J]

    def _eff(self, th):
        al, b, u = self._unpack(th)
        e = np.exp(u - u.max())
        return softplus(al), b, e / e.sum()

    def forward(self, th, x):
        a, b, pi = self._eff(th)
        return expit(np.asarray(x, float)[:, None] * a + b) @ pi

    def pdf(self, th, x):
        """Densita' in forma chiusa: nessun autograd, nessuna differenza finita."""
        a, b, pi = self._eff(th)
        s = expit(np.asarray(x, float)[:, None] * a + b)
        return (s * (1 - s) * a) @ pi

    dF_dx = pdf

    def grads(self, th, x, y):
        al, b, u = self._unpack(th)
        a, _, pi = self._eff(th)
        x = np.asarray(x, float)
        s = expit(x[:, None] * a + b)
        d = 2.0 * (s @ pi - y) / x.size
        gpi = s.T @ d
        gu = pi * (gpi - float(gpi @ pi))                # jacobiana della softmax
        dz = d[:, None] * pi * s * (1 - s)
        return np.concatenate([(dz.T @ x) * d_softplus(al), dz.sum(axis=0), gu])


# ----------------------------------------------------------------------------- training

def adam(arch, th, x, y, epochs=6000, lr=0.03):
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        g = arch.grads(th, x, y)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        th = th - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return th


def loo_targets(samples, h):
    """Etichette PNN: F_hat senza il contributo del proprio kernel, K(0) = 1/2."""
    n = samples.size
    return (n * parzen_cdf(samples, samples, h) - 0.5) / (n - 1)


def ecdf_targets(samples):
    """F_n(x_i) = (rango - 0.5)/n: nessuna finestra, nessun iperparametro."""
    return (np.argsort(np.argsort(samples)) + 0.5) / samples.size


def fit(arch, samples, targets, seed, epochs=6000, lr=0.03):
    return adam(arch, arch.init(samples, np.random.default_rng(1000 + seed)),
                samples, targets, epochs=epochs, lr=lr)


class Standardised:
    """Involucro: addestra su z = (x - mu)/s e rimappa. Riparametrizzazione esatta."""

    def __init__(self, arch, x):
        self.arch = arch
        self.name = arch.name + "+std"
        self.mu = float(np.median(x))
        self.sd = float(x.std(ddof=1))

    def _z(self, x):
        return (np.asarray(x, float) - self.mu) / self.sd

    def init(self, x, rng):
        return self.arch.init(self._z(x), rng)

    def forward(self, th, x):
        return self.arch.forward(th, self._z(x))

    def grads(self, th, x, y):
        return self.arch.grads(th, self._z(x), y)

    def pdf(self, th, x):
        return self.arch.pdf(th, self._z(x)) / self.sd


def teacher_h(samples, c=0.5):
    return c * samples.std(ddof=1) / np.sqrt(samples.size - 1)


# ============================================================================= E1
print("=" * 94)
print("E1  CORRETTEZZA DELL'IMPLEMENTAZIONE")
print("    (a) gradienti analitici vs differenze finite centrate, doppia precisione")
rng = np.random.default_rng(0)
x_chk = np.sort(rng.normal(0, 1.5, 40))
y_chk = np.clip(0.5 + 0.3 * x_chk / 3, 0.01, 0.99)
for arch in (ArchA(6), ArchB(6), ArchC(6)):
    th = arch.init(x_chk, np.random.default_rng(3))
    lo = lambda t: float(np.mean((arch.forward(t, x_chk) - y_chk) ** 2))
    ga = arch.grads(th, x_chk, y_chk)
    gn = np.array([(lo(th + e) - lo(th - e)) / 2e-6
                   for e in (np.eye(th.size) * 1e-6)])
    print(f"        {arch.name:20s} errore relativo max = "
          f"{np.max(np.abs(ga - gn) / np.maximum(np.abs(ga) + np.abs(gn), 1e-12)):.2e}")

print("    (b) monotonia strutturale con parametri casuali estremi (1000 perturbazioni)")
gtest = np.linspace(-40, 40, 5000)
for arch in (ArchA(8), ArchB(8), ArchC(8)):
    r = np.random.default_rng(7)
    bad = 0
    for _ in range(1000):
        th = arch.init(x_chk, r) + r.normal(0, 3.0, arch.init(x_chk, r).size)
        if np.any(np.diff(arch.forward(th, gtest)) < -1e-12):
            bad += 1
    print(f"        {arch.name:20s} parametrizzazioni non monotone: {bad}/1000")

# ============================================================================= E2
print()
print("=" * 94)
print("E2  COMPORTAMENTO ASINTOTICO: quanto vale F molto lontano dai dati?")
print("    valutato a mediana +/- 50*sigma dopo il training (trimodale, n=1000, seme 0)")
mix = CASES["trimodale"]
n, seed = 1000, 0
smp = mix.sample(n, np.random.default_rng(seed))
h = teacher_h(smp)
tgt = loo_targets(smp, h)
far = np.array([np.median(smp) - 50 * smp.std(ddof=1),
                np.median(smp) + 50 * smp.std(ddof=1)])
print(f"        {'arch':22s} {'F(-inf)':>11} {'1-F(+inf)':>11} {'massa raggiungibile':>20}")
fitted = {}
for arch in (ArchA(8), ArchA(8, True), ArchB(8), ArchB(8, True), ArchC(8)):
    th = fit(arch, smp, tgt, seed)
    fitted[arch.name] = (arch, th)
    lo_, hi_ = arch.forward(th, far)
    print(f"        {arch.name:22s} {lo_:>11.6f} {1 - hi_:>11.6f} {hi_ - lo_:>20.6f}")
print("    controllo algebrico (l'asintoto e' un valore FINITO, non 0 o 1):")
for nm in ("A", "B"):
    ar, th = fitted[nm]
    a, b, w, c = ar._eff(th)
    u_min = float(c + w[a < 0].sum())
    u_max = float(c + w[a > 0].sum())
    print(f"        {nm}: F(-inf) = sigma({u_min:+.3f}) = {expit(u_min):.6f}   "
          f"F(+inf) = sigma({u_max:+.3f}) = {expit(u_max):.6f}")
print("        C: F(-inf) = 0 e F(+inf) = somma(pi) = 1 esattamente, per ogni parametro")

# ============================================================================= E3
print()
print("=" * 94)
print("E3  RISOLUZIONE DELLA PENALITA' SOFT (256 punti equispaziati, training.py:146-148)")
print("    controesempio costruito: un avvallamento piu' stretto del passo di collocazione")
lo_d, hi_d = -6.0, 8.0
coll = np.linspace(lo_d, hi_d, 256)                      # come training.py:146-148
step = float(coll[1] - coll[0])
mid = 0.5 * (coll[127] + coll[128])                      # esattamente fra due punti
fine = np.linspace(lo_d, hi_d, 400_001)
print(f"    passo fra punti di collocazione = {step:.5f}")
print(f"        {'largh. dip':>12} {'penalita @256':>16} {'viol. su 400k':>15} {'deriv. minima':>15}")
for div in (4, 8, 20, 50):
    w_dip = step / div

    def curve(x, w=w_dip):
        return expit((x - 1.0) / 1.5) - 0.02 * np.exp(-0.5 * ((x - mid) / w) ** 2)

    def deriv(x, w=w_dip, eps=1e-9):
        return (curve(x + eps, w) - curve(x - eps, w)) / (2 * eps)

    print(f"        {w_dip:>12.5f} {float(np.mean(np.maximum(0.0, -deriv(coll)))):>16.3e} "
          f"{np.mean(np.diff(curve(fine)) < 0) * 100:>14.4f}% {deriv(fine).min():>15.3f}")
print("    -> sotto una certa larghezza la penalita' e' ESATTAMENTE 0 mentre la funzione")
print("       non e' monotona: la penalita' certifica solo i punti che guarda.")
print("    ONESTA': un dip cosi' stretto non e' producibile da una rete a 8 unita'")
print("       sigmoidali. Il controesempio dimostra il buco logico, non un fallimento")
print("       osservato: nelle nostre esecuzioni le reti sono uscite monotone senza penalita'.")

# ============================================================================= E4
print()
print("=" * 94)
print("E4  ACCURATEZZA (media su 3 semi, J = 8, ricetta PNN: etichette LOO, h = 0.5*sigma/sqrt(n-1))")
print("    Aq/Bq = stessa architettura del repo ma con l'inizializzazione sui quantili di C,")
print("            per separare l'effetto dell'architettura da quello dell'inizializzazione")
SEEDS = range(3)
LABELS = ["PW maestro", "PW regola-sigma", "A_raw", "A_rect", "Aq_rect",
          "B_rect", "Bq_rect", "C"]
for case_name, mixc in CASES.items():
    g = mixc.grid()
    tc, tp = mixc.cdf(g), mixc.pdf(g)
    print(f"\n    --- {case_name} ---")
    print(f"    {'n':>5}  {'stimatore':18} {'KS':>9} {'ISE pdf':>10}")
    for nn in (500, 1000, 2000):
        acc = {}
        for sd in SEEDS:
            s = mixc.sample(nn, np.random.default_rng(sd))
            hh = teacher_h(s)
            yy = loo_targets(s, hh)
            hr = 1.5 * s.std(ddof=1) / np.sqrt(nn)
            acc.setdefault("PW maestro", []).append(
                (ks(parzen_cdf(g, s, hh), tc), ise(parzen_pdf(g, s, hh), tp, g)))
            acc.setdefault("PW regola-sigma", []).append(
                (ks(parzen_cdf(g, s, hr), tc), ise(parzen_pdf(g, s, hr), tp, g)))
            for arch, lab in ((ArchA(8), "A"), (ArchA(8, True), "Aq"),
                              (ArchB(8), "B"), (ArchB(8, True), "Bq"), (ArchC(8), "C")):
                th = fit(arch, s, yy, sd)
                raw = arch.forward(th, g)
                if lab == "C":
                    acc.setdefault("C", []).append((ks(raw, tc), ise(arch.pdf(th, g), tp, g)))
                    continue
                rc, pd = rectify_cdf(raw, g)
                acc.setdefault(lab + "_rect", []).append((ks(rc, tc), ise(pd, tp, g)))
                if lab == "A":
                    acc.setdefault("A_raw", []).append(
                        (ks(raw, tc), ise(np.gradient(raw, g), tp, g)))
        for lab in LABELS:
            m = np.mean(acc[lab], axis=0)
            print(f"    {nn:>5}  {lab:18} {m[0]:>9.4f} {m[1]:>10.5f}")

# ============================================================================= E5
print()
print("=" * 94)
print("E5  MASSA RAGGIUNGIBILE E COERENZA pdf/CDF (trimodale, n = 1000, seme 0)")
g = CASES["trimodale"].grid()
print(f"        {'arch':22s} {'F(fine)-F(inizio)':>18} {'int f':>12} {'int clamp(f,0)':>16}")
for nm, (arch, th) in fitted.items():
    raw = arch.forward(th, g)
    d = arch.dF_dx(th, g)
    print(f"        {nm:22s} {raw[-1] - raw[0]:>18.6f} "
          f"{np.trapezoid(d, g):>12.6f} {np.trapezoid(np.clip(d, 0, None), g):>16.6f}")
print("    il clamp e' inerte quando la rete e' gia' monotona. Su una curva che non lo e':")
gg = np.linspace(-4, 4, 801)
Fw = expit(1.4 * gg) + 0.06 * np.sin(6 * gg) * np.exp(-0.5 * gg ** 2)
dw = np.gradient(Fw, gg)
print(f"        int f (derivata vera) = {np.trapezoid(dw, gg):.6f} = F(fine)-F(inizio) = "
      f"{Fw[-1] - Fw[0]:.6f}")
print(f"        int clamp(f,0)        = {np.trapezoid(np.clip(dw, 0, None), gg):.6f} "
      f"-> massa inventata {np.trapezoid(np.clip(dw, 0, None), gg) - np.trapezoid(dw, gg):+.2e}")

# ============================================================================= E6
print()
print("=" * 94)
print("E6  RUNNING MAX vs REGRESSIONE ISOTONA (stessa curva non monotona di E5)")
rm = np.maximum.accumulate(Fw)
iso = pava(Fw)
print(f"        violazioni nella curva grezza: {np.mean(np.diff(Fw) < 0) * 100:.2f}%")
print(f"        distanza L2 dalla curva grezza: running max {np.linalg.norm(rm - Fw):.6f}"
      f"   PAVA {np.linalg.norm(iso - Fw):.6f}  ({np.linalg.norm(rm - Fw) / np.linalg.norm(iso - Fw):.2f}x)")
print(f"        entrambe monotone: running max {np.all(np.diff(rm) >= 0)}, "
      f"PAVA {np.all(np.diff(iso) >= -1e-15)}")
print(f"        punti con pdf esattamente 0 (plateau spuri): running max "
      f"{int(np.sum(np.gradient(rm, gg) <= 1e-12))}/{gg.size}, PAVA "
      f"{int(np.sum(np.gradient(iso, gg) <= 1e-12))}/{gg.size}")

# ============================================================================= E7
print()
print("=" * 94)
print("E7  INVARIANZA PER TRASLAZIONE E SCALA (trimodale, n = 1000, seme 0)")
print("    gli stessi dati traslati/riscalati devono dare lo stesso KS")
base = CASES["trimodale"]
s0 = base.sample(1000, np.random.default_rng(0))
g0 = base.grid()
tc0 = base.cdf(g0)
print(f"        {'trasformazione':22s} {'A':>9} {'Aq':>9} {'C':>9} {'C+std':>9}")
for shift, scale, lab in ((0.0, 1.0, "nessuna"), (100.0, 1.0, "traslazione +100"),
                          (0.0, 50.0, "scala x50"), (1000.0, 1.0, "traslazione +1000")):
    s = s0 * scale + shift
    g2 = g0 * scale + shift
    hh = teacher_h(s)
    yy = loo_targets(s, hh)
    row = []
    for arch in (ArchA(8), ArchA(8, True), ArchC(8)):
        th = fit(arch, s, yy, 0)
        raw = arch.forward(th, g2)
        row.append(ks(raw if arch.name == "C" else rectify_cdf(raw, g2)[0], tc0))
    wrap = Standardised(ArchC(8), s)
    thw = fit(wrap, s, yy, 0)
    row.append(ks(wrap.forward(thw, g2), tc0))
    print(f"        {lab:22s} " + " ".join(f"{v:>9.4f}" for v in row))

# ============================================================================= E8
print()
print("=" * 94)
print("E8  CAPACITA': quante componenti servono a C (3 semi, n = 1000)")
print(f"        {'caso':22s} " + " ".join(f"{'J=' + str(j):>16}" for j in (4, 8, 16, 32, 64)))
for case_name, mixc in CASES.items():
    g = mixc.grid()
    tc, tp = mixc.cdf(g), mixc.pdf(g)
    cells = []
    for J in (4, 8, 16, 32, 64):
        r = []
        for sd in SEEDS:
            s = mixc.sample(1000, np.random.default_rng(sd))
            th_ = fit(ArchC(J), s, loo_targets(s, teacher_h(s)), sd)
            arch_ = ArchC(J)
            r.append((ks(arch_.forward(th_, g), tc), ise(arch_.pdf(th_, g), tp, g)))
        m = np.mean(r, axis=0)
        cells.append(f"{m[0]:.4f}/{m[1]:.5f}")
    print(f"        {case_name:22s} " + " ".join(f"{c:>16}" for c in cells))
print("        (celle: KS / ISE pdf)")

# ============================================================================= E9
print()
print("=" * 94)
print("E9  SERVE ANCORA UNA FINESTRA DI PARZEN? C su etichette LOO vs C sulla ECDF")
print("    etichette ECDF = (rango-0.5)/n: nessuna finestra, nessun iperparametro")
print(f"        {'caso':22s} {'n':>6} {'C su LOO(h)':>22} {'C su ECDF':>22}")
for case_name, mixc in CASES.items():
    g = mixc.grid()
    tc, tp = mixc.cdf(g), mixc.pdf(g)
    for nn in (500, 1000, 2000):
        a_, b_ = [], []
        for sd in SEEDS:
            s = mixc.sample(nn, np.random.default_rng(sd))
            arch_ = ArchC(16)
            t1 = fit(arch_, s, loo_targets(s, teacher_h(s)), sd)
            t2 = fit(arch_, s, ecdf_targets(s), sd)
            a_.append((ks(arch_.forward(t1, g), tc), ise(arch_.pdf(t1, g), tp, g)))
            b_.append((ks(arch_.forward(t2, g), tc), ise(arch_.pdf(t2, g), tp, g)))
        m1, m2 = np.mean(a_, axis=0), np.mean(b_, axis=0)
        print(f"        {case_name:22s} {nn:>6} {f'{m1[0]:.4f} / {m1[1]:.5f}':>22} "
              f"{f'{m2[0]:.4f} / {m2[1]:.5f}':>22}")
print("        (celle: KS / ISE pdf, J = 16)")
