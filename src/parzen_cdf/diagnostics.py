"""Diagnostica calcolabile dai soli campioni.

Sui dati con cui il progetto verra' provato la distribuzione generatrice non e' nota: non
esistono ``ks_vs_truth`` ne' ``pdf_mse``. Questo modulo raccoglie le uniche grandezze che
restano disponibili, e riporta **solo quelle che sono state misurate come informative**.

Quali lo siano non e' ovvio, ed e' stato misurato: per ogni valore della finestra si e'
calcolato l'errore vero (ISE) e quattro indicatori truth-free, poi si e' guardato se
ordinano le finestre come le ordina la verita' (16 densita' x 8 semi, n = 500;
``docs/redesign_pipeline.md`` E3):

    punteggio LSCV             rho di Spearman  +0.94     <- il migliore
    log-verosimiglianza LOO    rho di Spearman  +0.86     <- secondo parere
    massa sul dominio          rho di Spearman  +0.44     <- debole, ma diagnostico
    KS contro la ECDF          rho di Spearman  -0.03     <- TRAPPOLA

DIVIETO ESPLICITO (decisione D-14). **Non aggiungere il KS fra la CDF stimata e la CDF
empirica.** E' l'indicatore che verrebbe naturale mettere in un report -- "guarda come la
nostra curva aderisce ai dati" -- ed e' anti-correlato con l'errore vero: premia
sistematicamente le finestre piu' strette, perche' al tendere a zero della finestra lo
stimatore di Parzen converge *per costruzione* alla CDF empirica. Sceglierne la finestra
costa un fattore 13.7 in media e fino a 94.7 nel caso peggiore. Certificherebbe il contrario
di quello che sembra certificare.
"""

from __future__ import annotations

import numpy as np

from . import parzen

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def report_domain(x: np.ndarray, pad: float = 3.0) -> tuple[float, float]:
    """Dominio per grafici, integrali e diagnostica, costruito dai soli campioni.

    ``[min(x) - pad*sigma, max(x) + pad*sigma]``. L'ancoraggio e' la **dispersione**, non la
    finestra: h e' scelta per risolvere la struttura piu' fine, quindi e' piccola proprio
    nelle distribuzioni con componenti strette dentro componenti larghe, e un margine
    proporzionale ad h non coprirebbe le code (``docs/redesign_pipeline.md`` E2b).

    Con ``pad = 3`` la massa vera lasciata fuori e' al piu' 1.1e-04 nel caso peggiore su un
    banco di sedici densita', contro 4.6e-03 usando il solo intervallo dei dati e 1.7e-03 con
    un margine di dieci finestre. Non conviene allargare oltre: i nodi sono in numero finito
    e allargare toglie risoluzione dove serve.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("servono almeno 2 campioni per stimare la dispersione")
    s = float(x.std(ddof=1))
    if s <= 0:
        raise ValueError("i campioni hanno dispersione nulla")
    lo, hi = float(x.min() - pad * s), float(x.max() + pad * s)
    if not lo < hi:
        # con pad molto negativo il dominio si rovescia e ogni integrale calcolato su di esso
        # risulta privo di senso (massa negativa). Meglio rifiutare che restituire numeri.
        raise ValueError(f"dominio degenere [{lo:.4g}, {hi:.4g}]: pad = {pad} e' troppo negativo")
    return lo, hi


def lscv_score(x: np.ndarray, h: float, kernel: str = "logistic",
               grid_points: int = 2000) -> float:
    """Punteggio di cross-validation dei minimi quadrati alla finestra ``h`` (piu' basso = meglio).

    Stima non distorta dell'errore quadratico integrato, a meno di una costante indipendente
    da ``h``. E' il miglior indicatore truth-free disponibile: rho = 0.94 con l'ISE vero.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = report_domain(x)
    g = np.linspace(lo, hi, grid_points)
    f = parzen.parzen_pdf(g, x, h, kernel)
    return float(_trapezoid(f ** 2, g) - 2.0 * _loo_density(x, h, kernel).mean())


def loo_loglikelihood(x: np.ndarray, h: float, kernel: str = "logistic") -> float:
    """Log-verosimiglianza leave-one-out media (piu' alto = meglio). Secondo parere, rho = 0.86."""
    f = _loo_density(np.asarray(x, dtype=float), h, kernel)
    return float(np.log(np.maximum(f, 1e-300)).mean())


def _loo_density(x: np.ndarray, h: float, kernel: str) -> np.ndarray:
    k = parzen.KERNELS[kernel].pdf((x[:, None] - x[None, :]) / h)
    np.fill_diagonal(k, 0.0)
    return k.sum(axis=1) / ((x.size - 1) * h)


def diagnose(x: np.ndarray, h: float, cdf, pdf, *, pad: float = 3.0,
             grid_points: int = 2001, kernel: str = "logistic") -> dict:
    """Blocco diagnostico completo, con lo schema fisso di ``redesign_pipeline.md`` S5.

    ``cdf`` e ``pdf`` sono callable: la stima e' una funzione, non una tabella. Il dominio
    serve solo a integrare e a disegnare.

    La massa e' **riportata, non imposta**: un valore diverso da 1 segnala un dominio stretto
    o una stima difettosa, ed e' un'informazione da mostrare, non da normalizzare via.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    lo, hi = report_domain(x, pad)
    g = np.linspace(lo, hi, grid_points)
    curve = np.asarray(cdf(g), dtype=float)
    dens = np.asarray(pdf(g), dtype=float)

    h_sil = float(parzen.silverman_bandwidth(x))
    h_vm = float(parzen.variance_matched_bandwidth(x, kernel))
    ratio = max(h_sil, h_vm, h) / max(min(h_sil, h_vm, h), 1e-300)
    mass = float(_trapezoid(dens, g))
    viol = int(np.sum(np.diff(curve) < 0))

    avvisi = []
    if ratio > 3.0:
        avvisi.append(f"i selettori differiscono di un fattore {ratio:.1f}: campione "
                      f"probabilmente multimodale a scale miste")
    if not 0.99 <= mass <= 1.01:
        avvisi.append(f"massa sul dominio {mass:.4f} fuori da [0.99, 1.01]: dominio stretto "
                      f"o stima difettosa")
    if viol:
        avvisi.append(f"{viol} violazioni di monotonia: con l'architettura prevista devono "
                      f"essere 0, quindi e' un bug e non un fenomeno da correggere")

    return {
        "n": n,
        "h": float(h),
        "h1": float(h * np.sqrt(n)),
        "h_riferimento": {"silverman": h_sil, "variance_matched": h_vm,
                          "rapporto_max": float(ratio)},
        "lscv_score": lscv_score(x, h, kernel),
        "loo_loglik": loo_loglikelihood(x, h, kernel),
        "massa_sul_dominio": mass,
        "dominio": [lo, hi],
        "violazioni_monotonia": viol,
        "avvisi": avvisi,
    }
