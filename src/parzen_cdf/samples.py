"""Lettura di un campione da un file di numeri scritto da qualcun altro.

Il progetto viene eseguito su dati di cui non sappiamo nulla, compreso il modo in cui sono
stati scritti. Il rischio serio non e' il file che non si riesce a leggere: quello si vede
subito. E' il file che si legge in un modo diverso da come e' stato scritto, perche' allora
esce una stima dall'aspetto perfettamente normale calcolata su numeri inventati.

Due casi lo producono, e sono i due piu' probabili in pratica.

**La virgola decimale.** Un foglio di calcolo con le impostazioni italiane scrive ``1,5``.
Se la virgola viene presa per separatore, quella riga diventa i due numeri 1 e 5: il
campione raddoppia di taglia e i valori sono privi di senso. Non c'e' modo di distinguere
``1,5`` da una riga a due colonne guardando solo quella riga, quindi in quel caso si rifiuta
e si chiede, invece di indovinare.

**Le colonne di troppo.** Un file ``indice,valore`` letto per intero mescola gli indici ai
valori. Anche qui: piu' di una colonna significa fermarsi e chiedere quale.

Il resto (intestazione, righe vuote, spazi, punto e virgola, tabulazioni, notazione
scientifica) si riconosce senza ambiguita' e si legge.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

MIN_N = 10
MAX_N = 200_000

_NUMERO = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_NUMERO_VIRGOLA = re.compile(r"^[+-]?\d+,\d+$")


class FormatoAmbiguo(ValueError):
    """Il file si puo' leggere in piu' modi e la scelta cambia i numeri."""


def _righe_utili(testo: str) -> list[str]:
    fuori = []
    for riga in testo.splitlines():
        riga = riga.strip().lstrip("﻿")          # BOM di un file salvato da Windows
        if riga and not riga.startswith("#"):
            fuori.append(riga)
    return fuori


def _separatore(righe: list[str]) -> str | None:
    """Il carattere che divide le colonne, o ``None`` se c'e' una colonna sola.

    Punto e virgola e tabulazione non sono mai un separatore decimale, quindi quando ci sono
    la virgola e' per forza decimale. La virgola invece e' ambigua e viene decisa dopo.
    """
    for candidato in (";", "\t"):
        if all(candidato in r for r in righe):
            return candidato
    if all("," in r for r in righe):
        return ","
    if any(re.search(r"\s", r) for r in righe):
        return None if all(len(r.split()) == 1 for r in righe) else " "
    return None


def _tabella(righe: list[str], sep: str | None) -> list[list[str]]:
    celle = [r.split() if sep in (None, " ") else [c.strip() for c in r.split(sep)]
             for r in righe]
    larghezze = {len(c) for c in celle}
    if len(larghezze) > 1:
        raise ValueError(
            f"le righe non hanno tutte lo stesso numero di campi (ne ho trovati "
            f"{sorted(larghezze)}); non riesco a capire dove sta il campione")
    return celle


def _togli_intestazione(celle: list[list[str]]) -> list[list[str]]:
    """Via la prima riga se non contiene numeri: e' il nome della colonna."""
    if len(celle) > 1 and not any(_NUMERO.match(c) or _NUMERO_VIRGOLA.match(c)
                                  for c in celle[0]):
        return celle[1:]
    return celle


def parse_samples(testo: str, *, column: int | None = None,
                  decimal: str = "auto") -> np.ndarray:
    """Un vettore di osservazioni dal contenuto di un file.

    ``column`` sceglie la colonna (da 1) quando il file ne ha piu' di una; ``decimal`` vale
    ``"auto"``, ``"point"`` o ``"comma"`` e serve a sciogliere l'ambiguita' della virgola.
    """
    if decimal not in ("auto", "point", "comma"):
        raise ValueError("decimal deve essere 'auto', 'point' o 'comma'")

    righe = _righe_utili(testo)
    if not righe:
        raise ValueError("il file non contiene numeri")

    sep = _separatore(righe)

    # La virgola come separatore e' l'unico caso ambiguo, e lo e' solo se nessun campo usa
    # gia' il punto come segno decimale: con "1.5,2.7" la virgola separa e basta.
    if sep == "," and decimal == "auto":
        senza_virgole = [r.replace(",", " ") for r in righe]
        ha_punti = any("." in c for r in senza_virgole for c in r.split())
        if not ha_punti and all(_NUMERO_VIRGOLA.match(r) for r in righe):
            raise FormatoAmbiguo(
                "ogni riga e' del tipo '1,5' e si puo' leggere in due modi: un numero con la "
                "virgola decimale, oppure due colonne separate da virgola. I due modi danno "
                f"campioni diversi ({len(righe)} valori contro {2 * len(righe)}). Indica quale "
                "vuoi: decimal='comma' per la prima lettura, column=1 o 2 per la seconda")
    if decimal == "comma":
        sep = None if sep == "," else sep
        righe = [r.replace(",", ".") for r in righe]
        sep = _separatore(righe) if sep is None else sep

    celle = _togli_intestazione(_tabella(righe, sep))
    if not celle:
        raise ValueError("il file contiene solo un'intestazione")

    larghezza = len(celle[0])
    # Una riga sola con molti campi e' un vettore scritto in orizzontale, non una tabella:
    # non c'e' niente da scegliere.
    if len(celle) == 1 and larghezza > 1 and column is None:
        celle = [[v] for v in celle[0]]
        larghezza = 1
    if larghezza > 1:
        if column is None:
            anteprima = ", ".join(f"colonna {i + 1}: {celle[0][i]}"
                                  for i in range(min(larghezza, 4)))
            raise FormatoAmbiguo(
                f"il file ha {len(celle)} righe di {larghezza} campi e non so quale sia il "
                f"campione (prima riga -> {anteprima}). Se sono colonne, indica quale con "
                f"column=1..{larghezza}; se e' un vettore mandato a capo, riscrivilo con "
                f"un valore per riga")
        if not 1 <= column <= larghezza:
            raise ValueError(f"colonna {column} fuori intervallo: il file ne ha {larghezza}")
        celle = [[r[column - 1]] for r in celle]

    # La virgola decimale sopravvive quando il separatore era ';' o tab, dove non e' ambigua.
    piatte = [c[0] for c in celle]
    if decimal != "point" and any(_NUMERO_VIRGOLA.match(v) for v in piatte):
        piatte = [v.replace(",", ".") for v in piatte]

    valori = []
    for v in piatte:
        if not _NUMERO.match(v):
            raise ValueError(f"'{v[:24]}' non e' un numero")
        valori.append(float(v))

    x = np.asarray(valori, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("il campione contiene inf o nan")
    if not MIN_N <= x.size <= MAX_N:
        raise ValueError(f"servono fra {MIN_N} e {MAX_N} osservazioni, ne ho trovate {x.size}")
    return x


def load_samples(path: Path | str, *, column: int | None = None,
                 decimal: str = "auto") -> np.ndarray:
    """Come :func:`parse_samples`, da un percorso. ``.npy`` salta l'analisi del testo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file non trovato: {path}")
    if path.suffix.lower() == ".npy":
        x = np.asarray(np.load(path), dtype=float)
        if x.ndim > 1:
            if column is None:
                raise FormatoAmbiguo(
                    f"l'array ha forma {x.shape} e non so quale colonna sia il campione; "
                    f"indica column=1..{x.shape[1]}")
            x = x[:, column - 1]
        x = x.reshape(-1)
        if not MIN_N <= x.size <= MAX_N:
            raise ValueError(f"servono fra {MIN_N} e {MAX_N} osservazioni, ne ho trovate {x.size}")
        return x
    return parse_samples(path.read_text(encoding="utf-8-sig", errors="strict"),
                         column=column, decimal=decimal)
