"""Rivalidazione di E4, E8, E9 sullo stack vero (PyTorch + libreria del repo).

Le tre misure che sostengono D-07 (architettura) e D-08 (J=8) erano state fatte sulla
replica NumPy. Nel passaggio replica -> codice reale due affermazioni su cinque sono cadute
(vedi redesign_network.md §4bis), quindi queste vanno rifatte prima di rifattorizzare.

Differenze rispetto alla versione NumPy, tutte volute:
  - ottimizzatore e inizializzazione sono quelli di PyTorch, non i miei;
  - il seme e' fissato PRIMA della costruzione del modello per ogni braccio, quindi il
    confronto e' riproducibile (nel repo non lo e': difetto B8);
  - i bracci A/B usano il percorso consegnato dal repo (rettifica + np.gradient);
    il braccio C usa uscita grezza e pdf in forma chiusa.
"""

import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "temp_analysis"))

from parzen_cdf import data, parzen                                    # noqa: E402
from parzen_cdf.models import CDFNet                                   # noqa: E402
from parzen_cdf.training import TrainConfig, rectify_cdf, train_cdf    # noqa: E402
from study2_common import grid_for, ks, loo_cdf_targets, pnn_window, sigma_hat  # noqa: E402
from mixture_cdf_net import MixtureCDFNet                              # noqa: E402

N = 500
CASES = {
    "trimodale": data.asymmetric_trimodal(),
    "5 mode strette": data.GaussianMixture1D([.2] * 5, [-8, -4, 0, 4, 8], [.35] * 5),
    "6 mode scale miste": data.GaussianMixture1D([.25, .2, .15, .15, .15, .10],
                                                 [-6, -3, 0, 1.2, 4, 7],
                                                 [1.2, .4, .25, .8, .5, 1.5]),
}


def ise(a, b, g):
    return float(np.trapezoid((a - b) ** 2, g))


def quantile_init(net, x):
    """Inizializzazione sui quantili per CDFNet: isola l'effetto architettura da quello init."""
    J = net.weights[0].shape[0]
    centres = np.quantile(x, (np.arange(J) + 0.5) / J)
    width = max((x.max() - x.min()) / J, 1e-9)
    a = np.full(J, 1.0 / width)
    with torch.no_grad():
        raw = torch.as_tensor(a, dtype=torch.float32).unsqueeze(-1)
        if net.monotone:
            raw = torch.log(torch.expm1(raw.clamp_min(1e-6)))
        net.weights[0].copy_(raw)
        net.biases[0].copy_(torch.as_tensor(-a * centres, dtype=torch.float32))
    return net


def fit_repo(x, y, seed, monotone=False, qinit=False, width=8, epochs=6000):
    torch.manual_seed(seed)                       # PRIMA della costruzione: fix di B8
    net = CDFNet(in_dim=1, hidden_sizes=(width,), activation="sigmoid", monotone=monotone)
    if qinit:
        quantile_init(net, x)
    cfg = TrainConfig(epochs=epochs, lr=0.03, optimizer="adam", seed=seed)
    net, _ = train_cdf(net, torch.as_tensor(x, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32), cfg)
    return net


def fit_mix(x, y, seed, J=8, epochs=6000, lr=0.03):
    torch.manual_seed(seed)
    net = MixtureCDFNet(J).init_from_samples(x)   # deterministico dato x
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = mse(net(xt), yt)
        loss.backward()
        opt.step()
    return net


def score_repo(net, g, tc, tp):
    with torch.no_grad():
        raw = net(torch.as_tensor(g, dtype=torch.float32)).numpy().astype(float)
    rc, pdf = rectify_cdf(raw, g)
    return ks(rc, tc), ise(pdf, tp, g)


def score_mix(net, g, tc, tp):
    with torch.no_grad():
        gt = torch.as_tensor(g, dtype=torch.float32)
        return ks(net(gt).numpy().astype(float), tc), ise(net.pdf(gt).numpy().astype(float), tp, g)


def setup(mix, seed, n=N):
    x = mix.sample(n, np.random.default_rng(seed))
    h = pnn_window(x, 0.5 * sigma_hat(x))
    return x, h, loo_cdf_targets(x, h)


# ===================================================================== autotest
print("=" * 96)
print("V0  AUTOTEST DEL PROTOTIPO: le garanzie di T2 valgono davvero in PyTorch?")
_x = CASES["trimodale"].sample(N, np.random.default_rng(0))
_net = MixtureCDFNet(8).init_from_samples(_x)
with torch.no_grad():
    _far = torch.tensor([-1e6, 1e6])
    _lo, _hi = _net(_far).tolist()
    _fine = torch.linspace(float(_x.min()) - 50, float(_x.max()) + 50, 200_001)
    _c = _net(_fine).numpy()
    _p = _net(_fine).numpy()
    _d = _net.pdf(_fine).numpy()
print(f"    F(-1e6) = {_lo:.1f}   F(+1e6) = {_hi:.1f}   (attesi 0 e 1 esatti)")
print(f"    monotona su 200.001 nodi: {bool(np.all(np.diff(_c) >= 0))}"
      f"   dislivello peggiore {float(np.diff(_c).min()):.2e}")
print(f"    pdf sempre >= 0: {bool(np.all(_d >= 0))}   pdf minima {float(_d.min()):.2e}")
print(f"    massa su [min-50, max+50]: {float(np.trapezoid(_d, _fine.numpy())):.6f}   (attesa 1)")
_fd = np.gradient(_c, _fine.numpy())
print(f"    pdf in forma chiusa vs differenza finita: scarto max "
      f"{float(np.max(np.abs(_d - _fd))):.2e}")
_pert = MixtureCDFNet(8).init_from_samples(_x)
_bad = 0
_gt = torch.linspace(-40, 40, 5000)
torch.manual_seed(3)
for _ in range(500):
    with torch.no_grad():
        for _prm in _pert.parameters():
            _prm.add_(torch.randn_like(_prm) * 3.0)
        if np.any(np.diff(_pert(_gt).numpy()) < -1e-6):
            _bad += 1
print(f"    non monotone su 500 perturbazioni casuali estreme: {_bad}/500   (atteso 0)")
print()

# ===================================================================== E4-reale
print("=" * 96)
print(f"E4-reale  ACCURATEZZA sullo stack vero (n = {N}, 5 semi, J/width = 8, Adam 6000 ep.)")
print("          confronto con la colonna corrispondente della replica NumPy")
SEEDS = range(5)
NUMPY_REF = {                       # da redesign_network.md E4, n=500
    "trimodale": {"A_rect": (0.0309, 0.00316), "Aq_rect": (0.0318, 0.00336),
                  "B_rect": (0.0313, 0.00368), "C": (0.0314, 0.00339)},
    "5 mode strette": {"A_rect": (0.1794, 0.08391), "Aq_rect": (0.0342, 0.00700),
                       "B_rect": (0.1015, 0.04084), "C": (0.0338, 0.00473)},
    "6 mode scale miste": {"A_rect": (0.0732, 0.02088), "Aq_rect": (0.0431, 0.00905),
                           "B_rect": (0.0656, 0.02502), "C": (0.0351, 0.00241)},
}
for name, mix in CASES.items():
    g = grid_for(mix)
    tc, tp = mix.cdf(g), mix.pdf(g)
    print(f"\n   --- {name} ---")
    print(f"   {'stimatore':12} {'KS':>9} {'ISE pdf':>10} | {'KS NumPy':>9} {'ISE NumPy':>10}")
    acc = {}
    for s in SEEDS:
        x, h, y = setup(mix, s)
        acc.setdefault("PW maestro", []).append(
            (ks(parzen.parzen_cdf(g, x, h), tc), ise(parzen.parzen_pdf(g, x, h), tp, g)))
        acc.setdefault("A_rect", []).append(score_repo(fit_repo(x, y, s), g, tc, tp))
        acc.setdefault("Aq_rect", []).append(score_repo(fit_repo(x, y, s, qinit=True), g, tc, tp))
        acc.setdefault("B_rect", []).append(score_repo(fit_repo(x, y, s, monotone=True), g, tc, tp))
        acc.setdefault("C", []).append(score_mix(fit_mix(x, y, s), g, tc, tp))
    for lab in ("PW maestro", "A_rect", "Aq_rect", "B_rect", "C"):
        m = np.mean(acc[lab], axis=0)
        ref = NUMPY_REF[name].get(lab)
        tail = f" | {ref[0]:>9.4f} {ref[1]:>10.5f}" if ref else ""
        print(f"   {lab:12} {m[0]:>9.4f} {m[1]:>10.5f}{tail}")

# ===================================================================== E8-reale
print()
print("=" * 96)
print(f"E8-reale  CAPACITA' di MixtureCDFNet (n = {N}, 3 semi). Celle: KS / ISE pdf")
JS = (4, 6, 8, 12, 16, 24)
print(f"   {'caso':20} " + "".join(f"{'J=' + str(j):>17}" for j in JS))
for name, mix in CASES.items():
    g = grid_for(mix)
    tc, tp = mix.cdf(g), mix.pdf(g)
    cells, ises = [], []
    for J in JS:
        r = []
        for s in range(3):
            x, _h, y = setup(mix, s)
            r.append(score_mix(fit_mix(x, y, s, J=J), g, tc, tp))
        m = np.mean(r, axis=0)
        cells.append(f"{m[0]:.4f}/{m[1]:.5f}")
        ises.append(m[1])
    print(f"   {name:20} " + "".join(f"{c:>17}" for c in cells)
          + f"  <- ISE min a J={JS[int(np.argmin(ises))]}")

# ===================================================================== E9-reale
print()
print("=" * 96)
print(f"E9-reale  ETICHETTE: LOO di Parzen contro ECDF (n = {N}, 3 semi, J = 8)")
print(f"   {'caso':20} {'C su LOO(h)':>22} {'C su ECDF':>22} {'rapporto ISE':>14}")
for name, mix in CASES.items():
    g = grid_for(mix)
    tc, tp = mix.cdf(g), mix.pdf(g)
    a_, b_ = [], []
    for s in range(3):
        x, h, y = setup(mix, s)
        y_ecdf = (np.argsort(np.argsort(x)) + 0.5) / x.size
        a_.append(score_mix(fit_mix(x, y, s), g, tc, tp))
        b_.append(score_mix(fit_mix(x, y_ecdf, s), g, tc, tp))
    m1, m2 = np.mean(a_, axis=0), np.mean(b_, axis=0)
    print(f"   {name:20} {f'{m1[0]:.4f} / {m1[1]:.5f}':>22} "
          f"{f'{m2[0]:.4f} / {m2[1]:.5f}':>22} {m2[1] / m1[1]:>13.2f}x")
