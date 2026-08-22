"""Cerca nel report le spie meccaniche di una scrittura generata.

Applica le regole di superficie di docs/stile_report.md. Non giudica la voce ne' il ritmo,
che restano affare di una rilettura: trova solo i tic che si possono cercare con una
espressione regolare, e che proprio per questo e' stupido lasciarsi sfuggire.

    python temp_analysis/check_style.py report/report3.tex
"""

import re
import sys
from pathlib import Path

DASH = "-" * 3

REGOLE = [
    ("S1", "trattino lungo", re.compile(re.escape(DASH))),
    ("S2", "apertura con conteggio",
     re.compile(r"^\s*(Two|Three|Four|Five|Several)\s+\w+[,:]", re.M)),
    ("S3", "enfasi di riempimento",
     re.compile(r"\b(It is worth noting|Crucially|Importantly|Notably|Indeed,|"
                r"It should be noted|Of note)\b", re.I)),
    ("S4", "costruzione 'non e' X, e' Y'",
     re.compile(r"\b(is|was) not (a |an |the )?[\w\s]{2,30}[;,] (it|this) (is|was)\b")),
    ("S5", "meta-commento sul documento",
     re.compile(r"\b(this section|this chapter|as we (will )?see|in what follows|"
                r"the reader (will|should)|we now turn)\b", re.I)),
    ("S6", "formula di transizione",
     re.compile(r"\b(That said|Having said that|With that in mind|In other words, then)\b")),
]

# Parole che in un testo tecnico ci stanno, ma che se abbondano segnalano un tono
# uniformemente enfatico. Non sono errori: sono un termometro.
TERMOMETRO = ["deliberate", "precisely", "exactly", "simply", "merely", "genuinely",
              "fundamental", "crucial", "essential", "remarkable"]


def corpo_del_testo(testo):
    """Solo il testo dopo maketitle, senza commenti LaTeX: il preambolo non e' prosa."""
    if "maketitle" in testo:
        testo = testo[testo.index("maketitle"):]
    righe = []
    for riga in testo.splitlines():
        senza = riga.split("%")[0] if not riga.lstrip().startswith("%") else ""
        righe.append(senza)
    return "\n".join(righe)


def controlla(path):
    testo = corpo_del_testo(Path(path).read_text(encoding="utf-8"))
    righe = testo.splitlines()
    trovati = 0

    print(f"--- {path}")
    for codice, nome, pattern in REGOLE:
        colpi = []
        for i, riga in enumerate(righe, 1):
            for m in pattern.finditer(riga):
                colpi.append((i, riga.strip()[:88]))
        if colpi:
            trovati += len(colpi)
            print(f"\n  {codice}  {nome}: {len(colpi)}")
            for i, testo_riga in colpi[:6]:
                print(f"       riga {i:>4}  {testo_riga}")
            if len(colpi) > 6:
                print(f"       ... e altri {len(colpi) - 6}")

    parole = re.findall(r"[a-zA-Z']+", testo.lower())
    n = max(len(parole), 1)
    caldi = [(p, parole.count(p)) for p in TERMOMETRO if parole.count(p)]
    if caldi:
        tot = sum(c for _, c in caldi)
        print(f"\n  termometro dell'enfasi: {tot} occorrenze su {n} parole "
              f"({1000 * tot / n:.1f} ogni mille)")
        print("       " + ", ".join(f"{p} {c}" for p, c in sorted(caldi, key=lambda t: -t[1])))

    print(f"\n  violazioni meccaniche: {trovati}")
    return trovati


if __name__ == "__main__":
    tot = sum(controlla(p) for p in (sys.argv[1:] or ["report/report3.tex"]))
    sys.exit(1 if tot else 0)
