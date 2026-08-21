"""Le proprieta' che `MixtureCDFNet` deve garantire *per costruzione*.

Ogni test corrisponde a una decisione di progetto documentata in `docs/redesign_network.md`:

    D-07  l'uscita e' una CDF valida per qualunque valore dei parametri (teorema T2)
    D-09  la stima e' equivariante per trasformazioni affini dei dati (teorema T6)
    D-10  la densita' e' la derivata esatta della CDF, senza clamp ne' rettifica
    D-08  il numero di componenti di default e' 12

Sono test di **proprieta'**, non di punto: dove esiste un teorema si verifica sotto parametri
casuali estremi, non su un caso fortunato.

Tolleranze da `redesign_pipeline.md` S6: PyTorch lavora in float32, dove un ULP attorno a 0.5
vale 6.0e-08. La soglia sulla monotonia e' 1e-6, non 1e-9 (una soglia troppo stretta ha gia'
prodotto un falso allarme in fase di analisi).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parzen_cdf import data, parzen                       # noqa: E402
from parzen_cdf.models import MixtureCDFNet               # noqa: E402
from parzen_cdf.training import fit_mixture_cdf           # noqa: E402

MONO_TOL = 1e-6          # S6: float32
GRID = torch.linspace(-12.0, 12.0, 20_001)


@pytest.fixture(scope="module")
def samples():
    return data.asymmetric_trimodal().sample(300, np.random.default_rng(0))


@pytest.fixture(scope="module")
def labels(samples):
    h = parzen.lscv_bandwidth(samples)
    return parzen.parzen_cdf(samples, samples, h)


# ------------------------------------------------------------------ D-07 / T2

def test_default_is_twelve_components():
    assert MixtureCDFNet().n_components == 12


def test_tails_are_exact(samples):
    """F(-inf) = 0 e F(+inf) = 1 ESATTI, non 'molto piccoli'. E' cio' che CDFNet non puo' fare."""
    net = MixtureCDFNet().init_from_samples(samples)
    with torch.no_grad():
        lo, hi = net(torch.tensor([-1e6, 1e6])).tolist()
    assert lo == 0.0
    assert hi == 1.0


def test_properties_hold_under_extreme_random_parameters(samples):
    """Le garanzie sono della parametrizzazione, non dell'addestramento: valgono ovunque."""
    net = MixtureCDFNet().init_from_samples(samples)
    rng = torch.Generator().manual_seed(3)
    bad_mono = bad_range = bad_pdf = 0
    for _ in range(300):
        with torch.no_grad():
            for prm in net.parameters():
                prm.add_(torch.randn(prm.shape, generator=rng) * 3.0)
            c = net(GRID)
            d = net.pdf(GRID)
        if torch.any(torch.diff(c) < -MONO_TOL):
            bad_mono += 1
        if torch.any(c < 0.0) or torch.any(c > 1.0):
            bad_range += 1
        if torch.any(d < 0.0):
            bad_pdf += 1
    assert bad_mono == 0, f"{bad_mono}/300 parametrizzazioni non monotone"
    assert bad_range == 0, f"{bad_range}/300 fuori da [0,1]"
    assert bad_pdf == 0, f"{bad_pdf}/300 con densita' negativa"


def test_mass_is_one_without_normalisation(samples):
    """La massa vale 1 perche' F(+inf)-F(-inf) = 1, non perche' la si sia normalizzata."""
    net = MixtureCDFNet().init_from_samples(samples)
    with torch.no_grad():
        mass = float(np.trapezoid(net.pdf(GRID).numpy(), GRID.numpy()))
    assert abs(mass - 1.0) < 1e-3


# ------------------------------------------------------------------ D-10

def test_pdf_is_the_exact_derivative_of_the_cdf(samples):
    """f = dF/dx come identita' algebrica: e' il test che il percorso vecchio non passava."""
    net = MixtureCDFNet().init_from_samples(samples)
    g = torch.linspace(-8.0, 10.0, 40_001)
    with torch.no_grad():
        cdf = net(g).numpy()
        pdf = net.pdf(g).numpy()
    fd = np.gradient(cdf, g.numpy())
    assert np.max(np.abs(pdf - fd)) < 1e-3


# ------------------------------------------------------------------ riproducibilita'

def test_initialisation_consumes_no_randomness(samples):
    """`init_from_samples` e' deterministica dato x: non dipende dallo stato globale dell'RNG.

    E' cio' che chiude B8 per questa classe *per costruzione* e non per disciplina.
    """
    torch.manual_seed(11)
    a = MixtureCDFNet().init_from_samples(samples)
    torch.manual_seed(22)
    b = MixtureCDFNet().init_from_samples(samples)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_training_is_fully_deterministic(samples, labels):
    """Non c'e' alcuna casualita' da fissare: il seme e' irrilevante, non solo rispettato.

    Conseguenza per gli esperimenti: la variabilita' fra run viene solo dai dati, non
    dall'inizializzazione. Una media "su N semi" diventa una media su N campioni, che e'
    la quantita' statisticamente sensata.
    """
    _, h0 = fit_mixture_cdf(samples, labels, epochs=200, seed=0)
    torch.manual_seed(999)
    _, h1 = fit_mixture_cdf(samples, labels, epochs=200, seed=7)
    assert abs(h0[-1] - h1[-1]) < 1e-12


# ------------------------------------------------------------------ D-09 / T6

@pytest.mark.parametrize("alpha,beta", [(1.0, 100.0), (50.0, 0.0), (1.0, 1000.0)])
def test_estimate_is_equivariant_under_affine_transforms(samples, alpha, beta):
    """Addestrare su x o su alpha*x+beta deve dare la stessa stima, rimappata.

    Senza standardizzazione interna la rete collassa alla costante 0.5 gia' per beta = 100
    (misurato in redesign_network.md, sezione 4bis).
    """
    h = parzen.lscv_bandwidth(samples)
    y = parzen.parzen_cdf(samples, samples, h)
    base, _ = fit_mixture_cdf(samples, y, epochs=400, seed=0)

    moved = samples * alpha + beta
    y_moved = parzen.parzen_cdf(moved, moved, parzen.lscv_bandwidth(moved))
    assert np.allclose(y, y_moved, atol=1e-6), "le etichette di Parzen non sono equivarianti"
    shifted, _ = fit_mixture_cdf(moved, y_moved, epochs=400, seed=0)

    probe = np.linspace(samples.min(), samples.max(), 200)
    with torch.no_grad():
        a = base(torch.as_tensor(probe, dtype=torch.float32)).numpy()
        b = shifted(torch.as_tensor(probe * alpha + beta, dtype=torch.float32)).numpy()
    assert np.max(np.abs(a - b)) < 1e-3


# ------------------------------------------------------------------ persistenza

def test_state_dict_round_trip_preserves_the_estimate(samples, labels):
    """mu e sd sono buffer: un modello ricaricato deve restare valido (S2)."""
    net, _ = fit_mixture_cdf(samples, labels, epochs=200, seed=0)
    clone = MixtureCDFNet(net.n_components)
    clone.load_state_dict(net.state_dict())
    probe = torch.linspace(-6.0, 8.0, 500)
    with torch.no_grad():
        assert torch.allclose(net(probe), clone(probe))
        assert torch.allclose(net.pdf(probe), clone.pdf(probe))
