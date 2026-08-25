"""Interfaccia a riga di comando: stima CDF e densita' da un file di campioni.

    python -m parzen_cdf.run --samples campioni.csv --out risultati/

Serve perche' il progetto deve poter essere eseguito da chi non lo ha scritto, su dati di cui
non sappiamo nulla. Accetta un vettore di numeri in ``.csv``, ``.txt`` o ``.npy`` e produce:

    cdf.csv           x, F(x) sul dominio dichiarato
    pdf.csv           x, f(x)
    diagnostics.json  il blocco diagnostico truth-free (schema in redesign_pipeline.md S5)
    stima.png         densita' e CDF stimate, con i campioni

Nessuna delle uscite richiede di conoscere la distribuzione generatrice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import samples
from .estimate import run_from_samples


def _leggi(path: Path, column: int | None, decimal: str) -> np.ndarray:
    """La lettura del file, con gli errori tradotti in un'uscita pulita.

    Il modulo ``samples`` si ferma quando il file si puo' leggere in piu' modi invece di
    sceglierne uno: e' l'unico modo di non consegnare una stima calcolata su numeri che il
    file non conteneva.
    """
    try:
        return samples.load_samples(path, column=column, decimal=decimal)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    except samples.FormatoAmbiguo as e:
        raise SystemExit(
            f"non so come leggere il file: {e}. "
            f"Sulla riga di comando: --column N oppure --decimal comma") from None
    except ValueError as e:
        raise SystemExit(f"file non leggibile: {e}") from None


def _figure(est, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    lo, hi = est.domain()
    g = np.linspace(lo, hi, 2001)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(g, est.pdf(g), lw=1.8)
    ax1.plot(est.samples, np.zeros_like(est.samples), "|", ms=8, alpha=0.35)
    ax1.set_title(f"densita' stimata (n = {est.n}, h = {est.h:.4f})")
    ax2.plot(g, est.cdf(g), lw=1.8)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title(f"CDF stimata (h1 = {est.h1:.3f})")
    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "stima.png", dpi=120)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="parzen_cdf.run",
        description="Stima la CDF di una distribuzione ignota e ne deriva la densita'.")
    ap.add_argument("--samples", required=True, type=Path,
                    help="file con i campioni (.csv, .txt o .npy): un vettore di numeri")
    ap.add_argument("--out", required=True, type=Path, help="cartella di destinazione")
    ap.add_argument("--column", type=int, default=None,
                    help="quale colonna contiene il campione, da 1 (se il file ne ha piu' di una)")
    ap.add_argument("--decimal", choices=("auto", "point", "comma"), default="auto",
                    help="segno decimale; 'comma' per i file scritti come 1,5")
    ap.add_argument("--components", type=int, default=12,
                    help="numero di componenti della mistura (default 12)")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="finestra da usare; se assente e' scelta per cross-validation")
    ap.add_argument("--grid-points", type=int, default=2001)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    x = _leggi(a.samples, a.column, a.decimal)
    if not a.quiet:
        print(f"campioni: n = {x.size}, media {x.mean():.4f}, dev.std {x.std(ddof=1):.4f}")
        print("stima in corso...", flush=True)

    est = run_from_samples(x, n_components=a.components, epochs=a.epochs,
                           seed=a.seed, h=a.bandwidth)

    a.out.mkdir(parents=True, exist_ok=True)
    lo, hi = est.domain()
    g = np.linspace(lo, hi, a.grid_points)
    np.savetxt(a.out / "cdf.csv", np.column_stack([g, est.cdf(g)]),
               delimiter=",", header="x,cdf", comments="", fmt="%.10g")
    np.savetxt(a.out / "pdf.csv", np.column_stack([g, est.pdf(g)]),
               delimiter=",", header="x,pdf", comments="", fmt="%.10g")
    rep = est.diagnostics()
    (a.out / "diagnostics.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    est.save(a.out / "modello.pt")
    figura = _figure(est, a.out)

    if not a.quiet:
        print(f"finestra h = {rep['h']:.6f}   ->   h1 = {rep['h1']:.4f}  "
              f"(schedule h_n = h1/sqrt(n))")
        print(f"dominio [{lo:.4f}, {hi:.4f}]   massa {rep['massa_sul_dominio']:.6f}   "
              f"violazioni {rep['violazioni_monotonia']}")
        for w in rep["avvisi"]:
            print(f"  avviso: {w}")
        prodotti = "cdf.csv, pdf.csv, diagnostics.json, modello.pt"
        print(f"scritti in {a.out}: {prodotti}" + (", stima.png" if figura else ""))
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
