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
     # Allargata dopo la revisione del gruppo: la versione stretta pretendeva la virgola
     # in una posizione fissa e non vedeva "not X but Y".
     re.compile(r"\b(is|was|are|were) not (a |an |the )?[\w\s]{2,34}[;,] "
                r"(it|this|they) (is|was|are|were)\b"
                r"|\bnot (by|from|between) [\w\s]{2,40}, but (by|from|between)\b"
                r"|\brather than merely\b")),
    ("S5", "meta-commento sul documento",
     re.compile(r"\b(this section|this chapter|as we (will )?see|in what follows|"
                r"the reader (will|should)|we now turn)\b", re.I)),
    ("S6", "formula di transizione",
     re.compile(r"\b(That said|Having said that|With that in mind|In other words, then)\b")),
    # Aggiunta dopo averne trovate 13 in 28 pagine: non e' un errore, ma la frequenza
    # tradisce una voce sola che applica una formula. Si segnala se supera le quattro.
    ("S7", "costruzione 'is worth -ing'",
     re.compile(r"\b(is|are|it is|seems) worth\s+\w+ing\b", re.I)),

    # ------------------------------------------------------------------ dalla revisione
    # esterna. Le prime tre sono tic di scrittura, la quarta e' un ritmo.
    ("S8", "rivelazione dopo i due punti",
     re.compile(r"\b(is|are|was|were)\b[^.$:]{0,45}: (it|the|that|this) \w+", re.I)),
    ("S9", "obiezione o domanda fabbricata",
     re.compile(r"\b(it would (therefore )?be tempting"
                r"|the (obvious|standard|natural) (objection|answer)"
                r"|one might (guess|expect)|there is a prior question"
                r"|two questions follow|has an answer that is not obvious)\b", re.I)),
    ("S10", "commento su cosa il lettore deve notare",
     re.compile(r"\b(repays a second look|deserves a (word|comment)"
                r"|we want to draw attention|the column that matters"
                r"|seems better than glossing|instead of hiding it"
                r"|worth (saying|recording))\b", re.I)),
    ("S11", "elenco negativo come chiusura",
     re.compile(r"\bno \w+, no \w+,? and (no|nothing)\b", re.I)),
]

# Alcune regole tollerano poche occorrenze e diventano un problema solo in quantita'.
# Passaggi che una regola intercetta e che restano deliberatamente. Si dichiarano qui
# invece di allentare l'espressione regolare, che altrimenti diventerebbe cieca anche
# sui casi veri.
ESENZIONI = [
    ("S4", "irrelevant rather than merely respected",
     "distinzione tecnica: il seme non conta affatto, contro il seme viene onorato"),
    ("S9", "improving the obvious answer",
     "e' la tesi delle conclusioni, non un'obiezione fabbricata"),
]

# S8 tollera qualche occorrenza: i due punti prima di una formula sono scrittura
# matematica normale, ed e' la frequenza a tradire la formula retorica.
SOGLIE = {"S7": 4, "S8": 6}

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
    esentati = []

    print(f"--- {path}")
    for codice, nome, pattern in REGOLE:
        colpi = []
        for i, riga in enumerate(righe, 1):
            for m in pattern.finditer(riga):
                if any(c == codice and f in riga for c, f, _ in ESENZIONI):
                    esentati.append((i, riga.strip()[:70]))
                    continue
                colpi.append((i, riga.strip()[:88]))
        soglia = SOGLIE.get(codice, 0)
        if len(colpi) > soglia:
            trovati += len(colpi) - soglia
            oltre = f" (soglia {soglia})" if soglia else ""
            print(f"\n  {codice}  {nome}: {len(colpi)}{oltre}")
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

    if esentati:
        print(f"\n  esentate deliberatamente: {len(esentati)}")
        for _, testo_riga in esentati:
            motivo = next(mo for c, f, mo in ESENZIONI if f in testo_riga)
            print(f"       {testo_riga}")
            print(f"         -> {motivo}")
    print(f"\n  violazioni meccaniche: {trovati}")
    return trovati


if __name__ == "__main__":
    tot = sum(controlla(p) for p in (sys.argv[1:] or ["report/report3.tex"]))
    sys.exit(1 if tot else 0)
