"""Confronto end-to-end: percorso NUOVO contro percorso VECCHIO, entrambi dal repo.

Non e' una rivalidazione dell'architettura (fatta in revalidate_torch.py): e' la verifica
che la pipeline **consegnata** -- quella che il docente eseguirebbe -- sia almeno buona
quanto quella che sostituisce, misurata sui suoi stessi termini.

  vecchio : etichette LOO -> CDFNet(width 8) -> rectify_cdf -> pdf per differenze finite
  nuovo   : run_from_samples() -> MixtureCDFNet(J=12) -> pdf in forma chiusa

Differenza sostanziale oltre ai numeri: il percorso nuovo non usa mai la distribuzione vera,
nemmeno per costruire il dominio. Qui la verita' entra solo per calcolare l'errore.
"""

import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parzen_cdf import data, parzen                                   # noqa: E402
from parzen_cdf.diagnostics import report_domain                      # noqa: E402
from parzen_cdf.estimate import run_from_samples                      # noqa: E402
from parzen_cdf.models import CDFNet                                  # noqa: E402
from parzen_cdf.training import (TrainConfig, rectify_cdf, set_seed,  # noqa: E402
                                 train_cdf)
from study2_common import loo_cdf_targets, pnn_window, sigma_hat      # noqa: E402

N, SEEDS = 500, range(5)
CASES = {
    "trimodale": data.asymmetric_trimodal(),
    "bimodale simmetrica": data.symmetric_bimodal(),
    "spike in broad": data.spike_in_broad(),
    "5 mode strette": data.GaussianMixture1D([.2] * 5, [-8, -4, 0, 4, 8], [.35] * 5),
    "6 mode scale miste": data.GaussianMixture1D([.25, .2, .15, .15, .15, .10],
                                                 [-6, -3, 0, 1.2, 4, 7],
                                                 [1.2, .4, .25, .8, .5, 1.5]),
}


def score(cdf_vals, pdf_vals, g, tc, tp):
    return (float(np.max(np.abs(cdf_vals - tc))),
            float(np.trapezoid((pdf_vals - tp) ** 2, g)),
            float(np.trapezoid(pdf_vals, g)),
            int(np.sum(np.diff(cdf_vals) < 0)))


def vecchio(x, g):
    h = pnn_window(x, 0.5 * sigma_hat(x))
    y = loo_cdf_targets(x, h)
    set_seed(0)
    net = CDFNet(in_dim=1, hidden_sizes=(8,), activation="sigmoid")
    net, _ = train_cdf(net, torch.as_tensor(x, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32),
                       TrainConfig(epochs=6000, lr=0.03, optimizer="adam", seed=0))
    with torch.no_grad():
        raw = net(torch.as_tensor(g, dtype=torch.float32)).numpy().astype(float)
    return rectify_cdf(raw, g)


print("=" * 100)
print(f"CONFRONTO END-TO-END  (n = {N}, {len(list(SEEDS))} semi)")
print("  KS e ISE contro la verita'; massa e violazioni sono diagnostiche della validita'.")
print("  Il dominio del percorso nuovo viene dai soli campioni; quello del vecchio anche,")
print("  per non avvantaggiarlo: nel repo verrebbe dalla distribuzione vera.")
print()
print(f"  {'caso':22} {'percorso':10} {'KS':>9} {'ISE pdf':>10} {'massa':>10} {'viol.':>7}")
tot = {"vecchio": [], "nuovo": [], "parzen": []}
for name, mix in CASES.items():
    acc = {k: [] for k in tot}
    for s in SEEDS:
        x = mix.sample(N, np.random.default_rng(s))
        lo, hi = report_domain(x)
        g = np.linspace(lo, hi, 2001)
        tc, tp = mix.cdf(g), mix.pdf(g)

        c_old, p_old = vecchio(x, g)
        acc["vecchio"].append(score(c_old, p_old, g, tc, tp))

        est = run_from_samples(x, epochs=6000, seed=0)
        acc["nuovo"].append(score(est.cdf(g), est.pdf(g), g, tc, tp))

        h_ref = parzen.lscv_bandwidth(x)
        acc["parzen"].append(score(parzen.parzen_cdf(g, x, h_ref),
                                   parzen.parzen_pdf(g, x, h_ref), g, tc, tp))
    for k in ("parzen", "vecchio", "nuovo"):
        m = np.mean(acc[k], axis=0)
        tot[k].append(m)
        etichetta = {"parzen": "Parzen", "vecchio": "vecchio", "nuovo": "NUOVO"}[k]
        print(f"  {name if k == 'parzen' else '':22} {etichetta:10} {m[0]:>9.4f} {m[1]:>10.5f} "
              f"{m[2]:>10.6f} {m[3]:>7.1f}")
    print()

print("=" * 100)
print("SINTESI su tutti i casi")
print(f"  {'percorso':10} {'KS medio':>10} {'KS peggiore':>12} {'ISE media':>11} "
      f"{'ISE peggiore':>13} {'massa media':>12} {'violazioni':>11}")
for k in ("parzen", "vecchio", "nuovo"):
    a = np.array(tot[k])
    etichetta = {"parzen": "Parzen", "vecchio": "vecchio", "nuovo": "NUOVO"}[k]
    print(f"  {etichetta:10} {a[:, 0].mean():>10.4f} {a[:, 0].max():>12.4f} "
          f"{a[:, 1].mean():>11.5f} {a[:, 1].max():>13.5f} {a[:, 2].mean():>12.6f} "
          f"{a[:, 3].sum():>11.1f}")
