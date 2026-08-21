"""Il percorso completo: dai soli campioni alla stima, senza conoscere la distribuzione.

Il criterio di accettazione del re-design (S7) e' che questo funzioni su un vettore di numeri
qualunque, senza che esista alcun oggetto che descriva una distribuzione.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parzen_cdf import data, parzen                                   # noqa: E402
from parzen_cdf.estimate import Estimate, loo_parzen_cdf_targets, run_from_samples  # noqa: E402

EPOCHS = 400          # basta per i test: la convergenza non e' l'oggetto qui


@pytest.fixture(scope="module")
def est():
    x = data.asymmetric_trimodal().sample(300, np.random.default_rng(0))
    return run_from_samples(x, epochs=EPOCHS)


# ------------------------------------------------------------------ il vincolo centrale

def test_works_on_a_bare_vector_of_numbers():
    """Nessuna Mixture, nessun supporto noto: solo numeri. E' il punto di tutto il re-design."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(-3, 0.4, 120), rng.normal(2, 1.1, 180)])
    e = run_from_samples(x, epochs=EPOCHS)
    assert e.n == 300
    assert np.isfinite(e.h) and e.h > 0


def test_estimation_path_never_imports_distribution_objects():
    """Criterio S7.4: il percorso di stima non deve dipendere da come i dati sono nati.

    Il controllo e' fatto sull'AST e non sul testo: cercare la stringa "import data"
    produce un falso positivo su "from dataclasses import dataclass".
    """
    import ast

    import parzen_cdf.diagnostics
    import parzen_cdf.estimate

    vietati = {"data", "parzen_cdf.data"}
    for modulo in (parzen_cdf.estimate, parzen_cdf.diagnostics):
        albero = ast.parse(open(modulo.__file__, encoding="utf-8").read())
        for nodo in ast.walk(albero):
            if isinstance(nodo, ast.Import):
                nomi = {a.name for a in nodo.names}
            elif isinstance(nodo, ast.ImportFrom):
                base = nodo.module or ""
                nomi = {base} | {f"{base}.{a.name}".lstrip(".") for a in nodo.names}
                nomi |= {a.name for a in nodo.names} if nodo.level else set()
            else:
                continue
            colpevoli = nomi & vietati
            assert not colpevoli, f"{modulo.__name__} importa {colpevoli}"


# ------------------------------------------------------------------ le garanzie sopravvivono

def test_delivered_estimate_is_a_valid_cdf(est):
    lo, hi = est.domain()
    g = np.linspace(lo, hi, 5000)
    c, d = est.cdf(g), est.pdf(g)
    assert np.all(np.diff(c) >= -1e-6)
    assert np.all(d >= 0.0)
    assert abs(float(np.trapezoid(d, g)) - 1.0) < 1e-3
    assert est.cdf(np.array([-1e6]))[0] == 0.0
    assert est.cdf(np.array([1e6]))[0] == 1.0


def test_estimate_is_a_function_not_a_table(est):
    """Valutabile in punti arbitrari, anche fuori dal dominio di riferimento."""
    fuori = np.array([-40.0, 40.0])
    assert est.cdf(fuori).shape == (2,)
    assert est.pdf(fuori).shape == (2,)


# ------------------------------------------------------------------ D-06

def test_h1_is_consistent_with_the_required_schedule(est):
    """h_n = h1/sqrt(n) deve riprodurre esattamente la finestra usata."""
    assert est.h1 / np.sqrt(est.n) == pytest.approx(est.h)


# ------------------------------------------------------------------ etichette

def test_loo_targets_exclude_the_point_itself():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    h = 0.5
    full = parzen.parzen_cdf(x, x, h)
    loo = loo_parzen_cdf_targets(x, h)
    atteso = (x.size * full - 0.5) / (x.size - 1)
    assert np.allclose(loo, atteso)


# ------------------------------------------------------------------ persistenza e diagnostica

def test_round_trip_preserves_the_estimate(est, tmp_path):
    p = tmp_path / "modello.pt"
    est.save(p)
    back = Estimate.load(p)
    g = np.linspace(*est.domain(), 500)
    assert np.allclose(est.cdf(g), back.cdf(g))
    assert np.allclose(est.pdf(g), back.pdf(g))
    assert back.h == pytest.approx(est.h)


def test_diagnostics_reports_no_violations(est):
    rep = est.diagnostics()
    assert rep["violazioni_monotonia"] == 0
    assert rep["massa_sul_dominio"] == pytest.approx(1.0, abs=1e-2)


# ------------------------------------------------------------------ ingressi non validi

@pytest.mark.parametrize("bad,match", [
    (np.arange(5.0), "almeno 10"),
    (np.array([1.0, 2.0, np.nan] + [0.0] * 10), "non finiti"),
])
def test_invalid_input_is_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        run_from_samples(bad, epochs=10)


# ------------------------------------------------------------------ CLI

def test_cli_produces_the_expected_artifacts(tmp_path):
    x = data.asymmetric_trimodal().sample(200, np.random.default_rng(2))
    src = tmp_path / "campioni.csv"
    np.savetxt(src, x, delimiter=",")
    out = tmp_path / "risultati"
    r = subprocess.run([sys.executable, "-m", "parzen_cdf.run", "--samples", str(src),
                        "--out", str(out), "--epochs", "200", "--quiet"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for nome in ("cdf.csv", "pdf.csv", "diagnostics.json", "modello.pt"):
        assert (out / nome).exists(), f"manca {nome}"
    rep = json.loads((out / "diagnostics.json").read_text(encoding="utf-8"))
    assert rep["n"] == 200
    assert rep["h1"] == pytest.approx(rep["h"] * np.sqrt(200))
