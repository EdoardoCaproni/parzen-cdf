"""La diagnostica truth-free: schema, correttezza, e cio' che NON deve contenere."""

import numpy as np
import pytest

from parzen_cdf import data, parzen
from parzen_cdf.diagnostics import diagnose, loo_loglikelihood, lscv_score, report_domain

MIX = data.asymmetric_trimodal()


@pytest.fixture(scope="module")
def x():
    return MIX.sample(400, np.random.default_rng(1))


# ------------------------------------------------------------------ D-12

def test_domain_is_anchored_to_dispersion_not_to_bandwidth(x):
    lo, hi = report_domain(x)
    s = x.std(ddof=1)
    assert lo == pytest.approx(x.min() - 3 * s)
    assert hi == pytest.approx(x.max() + 3 * s)


def test_domain_leaves_negligible_true_mass_outside(x):
    """Il criterio con cui pad = 3 e' stato scelto (E2b): caso peggiore 1.1e-04."""
    lo, hi = report_domain(x)
    outside = float(MIX.cdf(np.array([lo]))[0] + 1.0 - MIX.cdf(np.array([hi]))[0])
    assert outside < 1e-3


def test_domain_needs_no_distribution_object(x):
    """Deve funzionare su un vettore qualunque: e' il punto di tutto il re-design."""
    assert report_domain(np.array([0.0, 1.0, 2.0, 3.0]))


# ------------------------------------------------------------------ D-13

def test_lscv_score_is_minimised_near_the_selected_bandwidth(x):
    """Coerenza: il selettore LSCV deve minimizzare il punteggio LSCV."""
    h = parzen.lscv_bandwidth(x)
    here = lscv_score(x, h)
    assert here < lscv_score(x, h * 4.0)
    assert here < lscv_score(x, h / 4.0)


def test_loo_loglikelihood_penalises_absurd_bandwidths(x):
    h = parzen.lscv_bandwidth(x)
    assert loo_loglikelihood(x, h) > loo_loglikelihood(x, h * 50.0)


# ------------------------------------------------------------------ schema S5

def test_report_has_the_fixed_schema(x):
    h = parzen.lscv_bandwidth(x)
    rep = diagnose(x, h, lambda t: parzen.parzen_cdf(t, x, h), lambda t: parzen.parzen_pdf(t, x, h))
    attese = {"n", "h", "h1", "h_riferimento", "lscv_score", "loo_loglik",
              "massa_sul_dominio", "dominio", "violazioni_monotonia", "avvisi"}
    assert set(rep) == attese
    assert rep["n"] == x.size
    assert rep["h1"] == pytest.approx(h * np.sqrt(x.size))
    assert rep["massa_sul_dominio"] == pytest.approx(1.0, abs=1e-2)
    assert rep["violazioni_monotonia"] == 0


def test_mass_is_reported_not_imposed(x):
    """La massa non viene normalizzata: su un dominio ridotto risulta < 1, e si vede.

    Nota: a ``pad = 0`` la massa fuori vale circa 2/(n+1) per pure statistiche d'ordine,
    cioe' ~0.005 a n = 400. Il valore 0.995 rientra nella banda [0.99, 1.01] e quindi NON
    genera avviso: e' il comportamento voluto, non una svista. L'avviso serve a segnalare
    domini davvero inadeguati, non il taglio fisiologico agli estremi del campione.
    """
    h = parzen.lscv_bandwidth(x)
    rep = diagnose(x, h, lambda t: parzen.parzen_cdf(t, x, h),
                   lambda t: parzen.parzen_pdf(t, x, h), pad=0.0)
    assert rep["massa_sul_dominio"] < 1.0
    assert rep["massa_sul_dominio"] > 0.99


def test_mass_warning_triggers_on_a_truly_inadequate_domain(x):
    """Con un dominio troncato dentro il campione la massa crolla e l'avviso scatta."""
    h = parzen.lscv_bandwidth(x)
    rep = diagnose(x, h, lambda t: parzen.parzen_cdf(t, x, h),
                   lambda t: parzen.parzen_pdf(t, x, h), pad=-1.0)
    assert rep["massa_sul_dominio"] < 0.99
    assert any("massa" in a for a in rep["avvisi"])


def test_warns_when_selectors_disagree(x):
    """Silverman e LSCV differiscono di un fattore ~6 su questa mistura: va segnalato."""
    h = parzen.lscv_bandwidth(x)
    rep = diagnose(x, h, lambda t: parzen.parzen_cdf(t, x, h), lambda t: parzen.parzen_pdf(t, x, h))
    assert rep["h_riferimento"]["rapporto_max"] > 3.0
    assert any("scale miste" in a for a in rep["avvisi"])


# ------------------------------------------------------------------ D-14

def test_ks_against_ecdf_is_not_offered():
    """Divieto esplicito: e' anti-correlato con l'errore vero e premia h -> 0."""
    import parzen_cdf.diagnostics as d
    esposti = [k for k in dir(d) if not k.startswith("_")]
    assert not [k for k in esposti if "ecdf" in k.lower()]
    assert "D-14" in d.__doc__ or "DIVIETO" in d.__doc__


def test_degenerate_domain_is_rejected(x):
    """Un dominio rovesciato produrrebbe una massa negativa: va rifiutato, non restituito."""
    with pytest.raises(ValueError, match="degenere"):
        report_domain(x, pad=-5.0)
