"""Fotografia numerica del comportamento attuale, da tenere invariante durante il refactor.

Non sono test di correttezza: sono test di *caratterizzazione*. Fissano cosa il codice fa
oggi, in modo che durante il refactor si distingua un cambiamento **voluto** (l'architettura
nuova produce numeri diversi, e sappiamo quali) da uno **accidentale** (ho rotto la selezione
della finestra mentre spostavo un file).

Coprono deliberatamente solo le parti che il refactor **non deve toccare**:
campionamento, stimatore di Parzen, selettori di finestra, metriche, rettifica, etichette.
Il percorso della rete non e' qui, perche' cambiera' per decisione (D-07).

Se uno di questi test fallisce durante il refactor, la domanda da farsi e' sempre la stessa:
"volevo cambiare questo?". Se la risposta e' no, e' un bug. Se e' si', si aggiorna il valore
in un commit dedicato che dice perche'.

I valori sono stati generati dal codice al commit 5969101 (dopo la correzione di B8).
"""

import pathlib
import sys

import numpy as np
import pytest

from parzen_cdf import data, metrics, parzen
from parzen_cdf.training import rectify_cdf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from study2_common import loo_cdf_targets, pnn_window, sigma_hat  # noqa: E402

TOL = 1e-10

GOLDEN = {
    "sample_first5": [1.921383473246, 3.863314704802, 2.514972982665,
                      -2.623296529143, 1.861723076994],
    "sample_mean": 0.765063812521,
    "parzen_cdf_h0.3": [0.024762532795, 0.285988781259, 0.455659034482,
                        0.690519080702, 0.96035832824],
    "parzen_pdf_h0.3": [0.059461756231, 0.093708036725, 0.152498202767,
                        0.119599351766, 0.095535453194],
    "h_silverman": 0.705143428432,
    "h_variance_matched": 0.388765947511,
    "h_lscv": 0.082223687512,
    "h_likelihood_cv": 0.130363319537,
    "h_sqrt_n": 0.070710678119,
    "pnn_window": 0.080128021371,
    "loo_targets_first5": [0.681745734485, 0.873769633354, 0.741970108147,
                           0.054343139827, 0.673993905302],
    "ks_distance": 0.031630924871,
    "integrates_to_one": 0.999997618577,
    "rectify_cdf_sum": 282.911305508424,
    "rectify_pdf_mass": 1.0,
}

MIX = data.asymmetric_trimodal()
PTS = np.array([-3.0, -1.0, 0.5, 2.0, 4.5])
GRID = np.linspace(-6.0, 8.0, 501)


@pytest.fixture(scope="module")
def x():
    return MIX.sample(200, np.random.default_rng(7))


def test_sampler_is_stable(x):
    """Se questo cambia, ogni altro valore di riferimento e' invalidato."""
    assert np.allclose(x[:5], GOLDEN["sample_first5"], atol=TOL)
    assert abs(float(x.mean()) - GOLDEN["sample_mean"]) < TOL


def test_parzen_estimator_is_stable(x):
    assert np.allclose(parzen.parzen_cdf(PTS, x, 0.3), GOLDEN["parzen_cdf_h0.3"], atol=TOL)
    assert np.allclose(parzen.parzen_pdf(PTS, x, 0.3), GOLDEN["parzen_pdf_h0.3"], atol=TOL)


@pytest.mark.parametrize("key,fn", [
    ("h_silverman", parzen.silverman_bandwidth),
    ("h_variance_matched", parzen.variance_matched_bandwidth),
    ("h_lscv", parzen.lscv_bandwidth),
    ("h_likelihood_cv", parzen.likelihood_cv_bandwidth),
    ("h_sqrt_n", parzen.sqrt_n_bandwidth),
])
def test_bandwidth_selectors_are_stable(x, key, fn):
    assert abs(float(fn(x)) - GOLDEN[key]) < TOL


def test_pnn_recipe_is_stable(x):
    assert abs(pnn_window(x, 0.5 * sigma_hat(x)) - GOLDEN["pnn_window"]) < TOL
    assert np.allclose(loo_cdf_targets(x, 0.3)[:5], GOLDEN["loo_targets_first5"], atol=TOL)


def test_metrics_are_stable(x):
    ks = metrics.ks_distance(MIX.cdf(GRID), parzen.parzen_cdf(GRID, x, 0.3))
    mass = metrics.integrates_to_one(parzen.parzen_pdf(GRID, x, 0.3), GRID)
    assert abs(float(ks) - GOLDEN["ks_distance"]) < TOL
    assert abs(float(mass) - GOLDEN["integrates_to_one"]) < TOL


def test_rectify_is_stable():
    wobbly = 0.5 * (1 + np.tanh(GRID)) + 0.02 * np.sin(5 * GRID)
    rc, pdf = rectify_cdf(wobbly, GRID)
    assert abs(float(rc.sum()) - GOLDEN["rectify_cdf_sum"]) < 1e-8
    assert abs(float(np.trapezoid(pdf, GRID)) - GOLDEN["rectify_pdf_mass"]) < TOL
