"""The neural CDF regressor.

A small MLP ``F_theta : R^d -> (0, 1)`` that approximates the Parzen CDF estimate. In 1-D it maps
a scalar to a scalar; in N-D it maps a point to a single scalar (the joint CDF value). A final
sigmoid squashes the output to (0, 1) to respect the range of a CDF.

Two modes share one code path so the comparison is fair:

- ``monotone=False`` -- a plain MLP. Monotonicity is only *encouraged* via a loss penalty (see
  ``training``); this covers both the unconstrained baseline and the soft-penalty variant.
- ``monotone=True``  -- monotone *by construction*: every weight matrix is passed through
  ``softplus`` (a smooth non-negativity reparameterization), so with monotone activations and a
  final sigmoid the network is non-decreasing in its input and its derivative -- the pdf -- is
  guaranteed non-negative.

  Attribuzione: il vincolo di positivita' dei pesi risale ad Archer & Wang (1993). NON e' la
  costruzione di Sill (*Monotonic Networks*, NIPS 1997), che propone reti **min-max** (max su
  gruppi di iperpiani, poi min dei massimi) proprio perche' i soli pesi positivi sono poco
  espressivi. Il commento precedente attribuiva erroneamente a Sill questa costruzione.

Activations must be **smooth**, because the pdf is obtained by differentiating the output; a ReLU
network would yield a piecewise-constant, discontinuous pdf. For the monotone mode the activation
must additionally be monotone (so ``silu`` is allowed only in the non-monotone mode).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

# Smooth activations only. The flag marks those that are also monotone increasing.
_ACTIVATIONS = {
    "sigmoid": (torch.sigmoid, True),
    "tanh": (torch.tanh, True),
    "softplus": (nn.functional.softplus, True),
    "silu": (nn.functional.silu, False),
}


class CDFNet(nn.Module):
    """MLP mapping an input point to a CDF value in (0, 1).

    NOTA (decisione D-07). Questa architettura e' **superata** da :class:`MixtureCDFNet` per
    il percorso di stima: non puo' raggiungere 0 e 1 con parametri finiti (teorema T1), la
    monotonia va rimediata a valle, e la densita' non e' disponibile in forma chiusa. Resta
    in libreria perche' e' l'architettura usata dagli studi gia' svolti negli script, che
    devono continuare a essere riproducibili. Per il codice nuovo si usa MixtureCDFNet.
    Motivazioni e misure in ``docs/redesign_network.md``.


    Parameters
    ----------
    in_dim       : input dimensionality (1 for Step 1, N for Step 2).
    hidden_sizes : widths of the hidden layers.
    activation   : smooth activation name ("sigmoid", "tanh", "softplus", "silu").
    monotone     : if True, enforce monotonicity by construction (non-negative weights).
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden_sizes: Sequence[int] = (16,),
        activation: str = "sigmoid",
        monotone: bool = False,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(_ACTIVATIONS)}")
        act_fn, is_monotone_act = _ACTIVATIONS[activation]
        if monotone and not is_monotone_act:
            raise ValueError(f"activation {activation!r} is not monotone; cannot use with monotone=True")

        self.in_dim = in_dim
        self.monotone = monotone
        self._activation = act_fn

        dims = [in_dim, *hidden_sizes, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            bias = torch.zeros(out_f)
            if monotone:
                # Initialise so that softplus(raw) starts at a modest, positive Xavier-like scale.
                scale = (6.0 / (in_f + out_f)) ** 0.5
                effective = torch.rand(out_f, in_f) * scale
                weight = torch.log(torch.expm1(effective.clamp_min(1e-6)))  # inverse softplus
            else:
                weight = torch.empty(out_f, in_f)
                nn.init.xavier_uniform_(weight)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))

    def _weight(self, raw: torch.Tensor) -> torch.Tensor:
        """Effective weight matrix: non-negative (via softplus) in monotone mode, else raw."""
        return nn.functional.softplus(raw) if self.monotone else raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the estimated CDF value(s) for ``x`` (shape ``[batch]`` or ``[batch, in_dim]``)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = x
        last = len(self.weights) - 1
        for i, (raw_w, b) in enumerate(zip(self.weights, self.biases)):
            h = nn.functional.linear(h, self._weight(raw_w), b)
            if i < last:
                h = self._activation(h)
        return torch.sigmoid(h).squeeze(-1)


# --------------------------------------------------------------------------------------------------
# MixtureCDFNet -- l'architettura del percorso nuovo (decisione D-07).
#
#     F(x) = somma_j softmax(u)_j * sigmoid( softplus(alpha_j) * z + b_j ),   z = (x - mu)/sd
#
# E' la trasformazione Deep Sigmoidal Flow (Huang et al., ICML 2018) senza il logit di uscita.
# Garantisce per QUALUNQUE valore dei parametri, senza correzioni a valle (teorema T2 in
# docs/redesign_network.md):
#
#     F in (0,1);  F non decrescente;  F(-inf) = 0 e F(+inf) = 1 ESATTI;
#     pdf = F' >= 0 in forma chiusa;  integrale della pdf = 1.
#
# Interpretazione: e' lo stimatore di Parzen a nucleo logistico con i pesi 1/n sostituiti da
# pesi appresi, i centri x_i da centri appresi e la finestra globale h da una finestra per
# componente. Con J = n contiene esattamente lo stimatore di Parzen; con J << n ne e' una
# compressione con finestre locali.
# --------------------------------------------------------------------------------------------------


def _inv_softplus(a):
    """Inverso stabile di softplus: alpha tale che softplus(alpha) = a, per a > 0.

    ``log(expm1(a))`` va in overflow gia' per a ~ 88 in float32. La forma
    ``a + log(1 - exp(-a))`` e' equivalente e stabile per a grande (il secondo termine
    tende a 0), e corretta per a piccolo (tende a log(a)).
    """
    return a + torch.log(-torch.expm1(-a))


class MixtureCDFNet(nn.Module):
    """Mistura convessa di CDF logistiche con standardizzazione interna dell'ingresso.

    Parametri
    ---------
    n_components : numero J di componenti. Il default 12 e' calibrato a n = 500 sullo stack
        PyTorch (docs/redesign_network.md, sezione 4ter): sotto le 8 componenti i bersagli
        multimodali falliscono, sopra le 12 il guadagno e' nullo e l'ISE peggiora appena.
        L'asimmetria del rischio dice di non scendere: sbagliare per difetto costa un fattore
        11, per eccesso il 5 per cento.
    """

    def __init__(self, n_components: int = 12):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_components))   # larghezze, via softplus
        self.b = nn.Parameter(torch.zeros(n_components))       # traslazioni
        self.u = nn.Parameter(torch.zeros(n_components))       # pesi, via softmax
        # statistiche di standardizzazione: buffer, non parametri, cosi' finiscono nel
        # state_dict e un modello ricaricato resta valido (D-09)
        self.register_buffer("mu", torch.zeros(()))
        self.register_buffer("sd", torch.ones(()))

    @property
    def n_components(self) -> int:
        return self.alpha.numel()

    @torch.no_grad()
    def init_from_samples(self, x: torch.Tensor) -> "MixtureCDFNet":
        """Centri sui quantili dei dati standardizzati, larghezze pari al passo fra centri.

        E' l'inizializzazione naturale per una mistura, e non dipende dallo stato globale
        dell'RNG: il modello e' deterministico dato ``x`` (chiude B8 per questa classe).
        """
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        J = self.n_components
        self.mu.fill_(float(x.median()))
        self.sd.fill_(float(x.std(unbiased=True)))
        z = (x - self.mu) / self.sd
        qs = (torch.arange(J, dtype=torch.float32) + 0.5) / J
        centres = torch.quantile(z, qs)
        width = torch.clamp((z.max() - z.min()) / J, min=1e-6)
        a = torch.full((J,), float(1.0 / width))
        self.alpha.copy_(_inv_softplus(a))
        self.b.copy_(-a * centres)
        self.u.zero_()
        return self

    def _parts(self, x: torch.Tensor):
        a = F.softplus(self.alpha)
        z = (torch.as_tensor(x, dtype=torch.float32) - self.mu) / self.sd
        s = torch.sigmoid(z.reshape(-1).unsqueeze(-1) * a + self.b)
        return s, torch.softmax(self.u, dim=0), a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CDF stimata. Valutabile in qualunque punto di R, non su una griglia."""
        s, pi, _ = self._parts(x)
        return s @ pi

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Densita' in forma chiusa: nessun autograd, nessuna differenza finita, nessun clamp."""
        s, pi, a = self._parts(x)
        return ((s * (1.0 - s) * a) @ pi) / self.sd
