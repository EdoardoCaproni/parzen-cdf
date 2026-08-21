"""Ogni modulo del pacchetto deve importarsi.

Questo test esiste per un motivo preciso: fino alla correzione di B1 due moduli
(``parzen`` e ``metrics``) sollevavano AttributeError all'import su NumPy >= 2.4, perche'
``getattr(np, "trapezoid", np.trapz)`` valuta il fallback in modo ansioso. Nessun test
importava tutti i moduli, quindi il difetto passava inosservato: ci si accorgeva del
problema solo eseguendo uno script.
"""

import importlib
import pathlib

import numpy as np
import pytest

MODULES = [
    "parzen_cdf",
    "parzen_cdf.data",
    "parzen_cdf.parzen",
    "parzen_cdf.metrics",
    "parzen_cdf.models",
    "parzen_cdf.training",
]

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("module", ["parzen_cdf.parzen", "parzen_cdf.metrics"])
def test_trapezoid_helper_is_usable(module):
    """Il guard deve essere pigro E il risultato deve essere corretto."""
    fn = importlib.import_module(module)._trapezoid
    x = np.linspace(0.0, 1.0, 1001)
    assert abs(float(fn(x, x)) - 0.5) < 1e-9      # integrale di x su [0,1]


def _code_without_comments(text):
    """Solo il codice: il difetto va cercato nel codice, non nelle note che lo descrivono.

    La prima versione di questo test cercava la stringa nel file intero e si
    autodenunciava, perche' il commento che spiega il bug contiene il bug.
    """
    stripped = [line.split("#", 1)[0] for line in text.splitlines()]
    return "\n".join(stripped)


def test_no_eager_getattr_fallback():
    """Difesa contro la reintroduzione del difetto, per ispezione del sorgente."""
    offenders = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        code = _code_without_comments(path.read_text(encoding="utf-8", errors="ignore"))
        if 'getattr(np, "trapezoid"' in code or "getattr(np, 'trapezoid'" in code:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"guard ansioso reintrodotto in: {offenders}"
