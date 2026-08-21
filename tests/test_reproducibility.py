"""Il seme deve controllare l'intero risultato, inizializzazione compresa.

Difetto B8: in ``fit_cdf_net`` il modello era costruito prima che ``train_cdf`` chiamasse
``set_seed``. Poiche' l'inizializzazione avviene nel costruttore e l'addestramento e'
full-batch deterministico, il seme non aveva alcun effetto: il risultato dipendeva dallo
stato globale dell'RNG di PyTorch al momento della chiamata.

Conseguenza pratica: nessuna delle medie "su N semi" della fase B era riproducibile, e le
etichette dei semi erano fittizie. Questi test fissano le due proprieta' che servono.
"""

import sys
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from study2_common import fit_cdf_net, loo_cdf_targets, pnn_window, sigma_hat  # noqa: E402

from parzen_cdf import data  # noqa: E402

EPOCHS = 200          # basta a distinguere i run: non serve la convergenza
PROBE = torch.linspace(-4.0, 6.0, 41)


def _problem(n=200):
    mix = data.asymmetric_trimodal()
    x = mix.sample(n, np.random.default_rng(0))
    h = pnn_window(x, 0.5 * sigma_hat(x))
    return x, loo_cdf_targets(x, h)


def _curve(seed, ambient=None):
    if ambient is not None:
        torch.manual_seed(ambient)
    x, y = _problem()
    model, _ = fit_cdf_net(x, y, width=8, epochs=EPOCHS, seed=seed)
    with torch.no_grad():
        return model(PROBE).numpy().astype(float)


def test_same_seed_is_reproducible_regardless_of_ambient_rng():
    """Lo stesso seme deve dare lo stesso risultato, qualunque sia lo stato globale."""
    a = _curve(seed=0, ambient=11)
    b = _curve(seed=0, ambient=22)
    c = _curve(seed=0, ambient=33)
    assert np.allclose(a, b, atol=1e-6), "il risultato dipende dallo stato globale dell'RNG"
    assert np.allclose(a, c, atol=1e-6), "il risultato dipende dallo stato globale dell'RNG"


def test_different_seeds_give_different_results():
    """Semi diversi devono dare risultati diversi: altrimenti il seme e' ignorato."""
    a = _curve(seed=0, ambient=99)
    b = _curve(seed=1, ambient=99)
    assert not np.allclose(a, b, atol=1e-6), "il seme non ha alcun effetto sul risultato"


def test_seed_controls_initialisation_not_just_training():
    """Verifica diretta: due modelli con semi diversi partono da pesi diversi."""
    from parzen_cdf.models import CDFNet
    from parzen_cdf.training import set_seed

    def first_layer(seed):
        set_seed(seed)
        return CDFNet(in_dim=1, hidden_sizes=(8,), activation="sigmoid").weights[0].detach().clone()

    assert torch.allclose(first_layer(0), first_layer(0)), "init non riproducibile a parita' di seme"
    assert not torch.allclose(first_layer(0), first_layer(1)), "init identica con semi diversi"
