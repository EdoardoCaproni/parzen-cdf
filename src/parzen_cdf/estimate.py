"""Il punto d'ingresso unico: dai campioni alla stima, senza conoscere la distribuzione.

E' il modulo che rende il progetto eseguibile sui dati di qualcun altro. Non importa
``data`` ne' alcun oggetto che descriva una distribuzione: vede solo un vettore di numeri.

Pipeline (le decisioni sono in ``docs/ROADMAP.md``):

    1. finestra per cross-validation dei minimi quadrati            D-05
    2. consegna nella forma h_n = h1/sqrt(n) richiesta dall'esercizio  D-06
    3. etichette = CDF di Parzen leave-one-out ai soli punti campione D-03
    4. rete a mistura di CDF logistiche, J = 12                     D-07, D-08
    5. nessuna rettifica, nessun clamp: la validita' e' strutturale D-10, D-11
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import diagnostics, parzen
from .models import MixtureCDFNet
from .training import fit_mixture_cdf


def loo_parzen_cdf_targets(x: np.ndarray, h: float, kernel: str = "logistic") -> np.ndarray:
    """Etichette leave-one-out: la CDF di Parzen a x_i senza il contributo di x_i stesso.

    La stima completa include il nucleo del punto, che vale K(0) = 1/2 per un nucleo
    simmetrico; lo si sottrae e si media sugli altri n-1:  y_i = (n*F(x_i) - 1/2)/(n-1).
    E' la ricetta della Parzen Neural Network (Trentin): etichette rumorose ma non distorte,
    che una rete di capacita' limitata media via.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("servono almeno 2 campioni")
    full = parzen.parzen_cdf(x, x, h, kernel)
    return (n * full - 0.5) / (n - 1)


@dataclass
class Estimate:
    """Una stima consegnata: due funzioni, piu' i parametri che le hanno prodotte.

    ``cdf`` e ``pdf`` sono **funzioni valutabili in qualunque punto di R**, non tabelle su
    una griglia. E' la differenza che il re-design introduce: il dominio serve solo a
    disegnare e a integrare, e non fa parte dello stimatore.
    """

    model: MixtureCDFNet
    h: float
    samples: np.ndarray = field(repr=False)

    @property
    def n(self) -> int:
        return int(self.samples.size)

    @property
    def h1(self) -> float:
        """La costante dello schedule h_n = h1/sqrt(n), nella forma richiesta dall'esercizio.

        Non e' una costante universale: e' una **stima** ricavata dal campione, esattamente
        come lo era ``1.5*sigma`` nella regola che sostituisce. Cambia l'estimatore, non la
        forma dello schedule (D-06).
        """
        return float(self.h * np.sqrt(self.n))

    def cdf(self, t) -> np.ndarray:
        with torch.no_grad():
            return self.model(torch.as_tensor(np.asarray(t, dtype=float),
                                              dtype=torch.float32)).numpy().astype(float)

    def pdf(self, t) -> np.ndarray:
        with torch.no_grad():
            return self.model.pdf(torch.as_tensor(np.asarray(t, dtype=float),
                                                  dtype=torch.float32)).numpy().astype(float)

    def domain(self, pad: float = 3.0) -> tuple[float, float]:
        return diagnostics.report_domain(self.samples, pad)

    def diagnostics(self, pad: float = 3.0) -> dict:
        return diagnostics.diagnose(self.samples, self.h, self.cdf, self.pdf, pad=pad)

    def save(self, path) -> None:
        torch.save({"state_dict": self.model.state_dict(),
                    "n_components": self.model.n_components,
                    "h": self.h, "samples": self.samples}, Path(path))

    @classmethod
    def load(cls, path) -> "Estimate":
        blob = torch.load(Path(path), weights_only=False)
        model = MixtureCDFNet(blob["n_components"])
        model.load_state_dict(blob["state_dict"])
        return cls(model=model, h=float(blob["h"]), samples=np.asarray(blob["samples"]))


def run_from_samples(x, *, n_components: int = 12, epochs: int = 6000, lr: float = 0.03,
                     seed: int = 0, h: float | None = None,
                     teacher_scale: float = 0.5) -> Estimate:
    """Stima CDF e densita' a partire dai soli campioni.

    Parametri
    ---------
    x : vettore di osservazioni. Nient'altro: nessuna distribuzione, nessun supporto noto.
    h : finestra. Se ``None`` viene scelta per cross-validation dei minimi quadrati (D-05),
        che sul banco di sedici densita' ha efficienza 1.26 media e 1.59 nel caso peggiore
        rispetto all'oracolo, contro 2.27 e 9.38 della regola ``h1 = 1.5*sigma``.
    teacher_scale : la finestra delle **etichette** e' deliberatamente piu' stretta di quella
        che minimizza l'errore dello stimatore di Parzen: e' la ricetta PNN, in cui la rete
        media via il rumore delle etichette. Con ``1.0`` si usa la stessa finestra.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 10:
        raise ValueError(f"servono almeno 10 campioni, ricevuti {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("i campioni contengono valori non finiti")

    h_sel = float(parzen.lscv_bandwidth(x)) if h is None else float(h)
    if h_sel <= 0:
        raise ValueError(f"finestra non positiva: {h_sel}")

    y = loo_parzen_cdf_targets(x, teacher_scale * h_sel)
    model, _ = fit_mixture_cdf(x, y, n_components=n_components, epochs=epochs, lr=lr, seed=seed)
    return Estimate(model=model, h=h_sel, samples=x)
