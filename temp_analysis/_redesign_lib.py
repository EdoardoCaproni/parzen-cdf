"""Sola libreria del banco di prova (nessun esperimento a livello di modulo)."""
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


