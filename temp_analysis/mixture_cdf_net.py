"""Prototipo di validazione di MixtureCDFNet (candidato per src/parzen_cdf/models.py).

Sta in temp_analysis/ e non in src/ di proposito: prima si valida, poi si rifattorizza.
Questa e' la classe proposta in docs/redesign_network.md P1, scritta in PyTorch per poterla
misurare sullo stack vero invece che sulla replica NumPy.

    F(x) = somma_j softmax(u)_j * sigmoid( softplus(alpha_j) * z + b_j ),   z = (x - mu)/sd

Proprieta' garantite per QUALUNQUE valore dei parametri (teorema T2):
    F in (0,1);  F non decrescente;  F(-inf)=0 e F(+inf)=1 ESATTI;
    pdf = F' >= 0 in forma chiusa;  integrale della pdf = 1.
Nessuna rettifica, nessun clamp, nessuna griglia.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _inv_softplus(a):
    """Inverso stabile di softplus: alpha tale che softplus(alpha) = a, per a > 0.

    ``log(expm1(a))`` va in overflow gia' per a ~ 88 in float32. La forma
    ``a + log(1 - exp(-a))`` e' equivalente e stabile per a grande (il secondo termine
    tende a 0), e corretta per a piccolo (tende a log(a)).
    """
    return a + torch.log(-torch.expm1(-a))


class MixtureCDFNet(nn.Module):
    """Mistura convessa di CDF logistiche con standardizzazione interna dell'ingresso."""

    def __init__(self, n_components: int = 8):
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
