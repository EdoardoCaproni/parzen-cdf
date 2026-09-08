"""Cerca ripetizioni fra capitoli scritti in momenti diversi.

Due controlli. Il primo trova gruppi di parole che ricorrono identici in punti lontani del
documento: sono le frasi che si e' scritto due volte senza accorgersene. Il secondo conta le
ricorrenze di alcuni argomenti chiave, per capire se un ragionamento viene rifatto da capo
invece che richiamato.

    python temp_analysis/check_repetition.py report/report3.tex
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

BS = chr(92)
N_PAROLE = 7          # lunghezza dei gruppi confrontati

# Argomenti che ricorrono per natura: se compaiono troppe volte, si sta rifacendo il
# ragionamento invece di richiamarlo.
ARGOMENTI = {
    # Cerca l'ARGOMENTO, non la metrica: "worst case" e' un nome di colonna e ricorre
    # legittimamente in ogni tabella. Con quel termine dentro, il contatore segnalava sedici
    # ricorrenze di cui due sole erano la ridedizione del ragionamento, e un allarme che
    # suona sempre e' un allarme che si impara a ignorare.
    "il caso peggiore conta perche' il bersaglio e' uno solo":
        r"(single unknown distribution|one distribution rather than|one distribution, not"
        r"|one sample from one distribution|not an average over|for the same reason as in"
        r"|scored the same way from here on)",
    "sigma gonfiato dalla separazione fra le mode":
        r"(inflated by the (distance|separation)|global spread)",
    "la mistura e' il Parzen con parametri appresi":
        r"(with (the weights|learned)|Parzen estimator with)",
    "la massa e' riportata e non imposta":
        r"(reported and not imposed|not a result: it is the rescaling|imposed by the rescaling)",
    "l'ISE e' la misura severa, il KS perdona":
        r"(forgiving|integration (already )?(suppresses|averages))",
}


def testo_pulito(path):
    t = Path(path).read_text(encoding="utf-8")
    t = t[t.index("maketitle"):]
    righe = [r for r in t.splitlines() if not r.lstrip().startswith("%")]
    return "\n".join(righe)


def capitoli(t):
    """(nome, testo) per ogni capitolo.

    Nota: in una espressione regolare il backslash va scritto due volte, altrimenti
    ``\\section`` viene letto come ``\\s`` (spazio bianco) seguito da ``ection``, e il
    documento non viene diviso affatto.
    """
    pezzi = re.split(re.escape(BS) + r"section\{([^}]*)\}", t)
    fuori = [("(front matter)", pezzi[0])]
    return fuori + [(pezzi[i], pezzi[i + 1]) for i in range(1, len(pezzi) - 1, 2)]


def main(path):
    t = testo_pulito(path)
    caps = capitoli(t)

    print("=== gruppi di parole ripetuti in capitoli diversi ===")
    print("    (l'abstract ripete il corpo per definizione: si guardano gli altri)")
    visti = defaultdict(list)
    for nome, corpo in caps:
        # via i comandi e gli ambienti LaTeX, altrimenti le tabelle si somigliano tutte
        pulito = re.sub(re.escape(BS) + r"[a-zA-Z]+", " ", corpo)
        pulito = re.sub(r"[{}$&\\]", " ", pulito)
        parole = re.findall(r"[a-z']+", pulito.lower())
        for i in range(len(parole) - N_PAROLE):
            gruppo = " ".join(parole[i:i + N_PAROLE])
            if nome not in visti[gruppo]:
                visti[gruppo].append(nome)
    trovati = {g: c for g, c in visti.items() if len(c) > 1}
    if trovati:
        for g, c in sorted(trovati.items())[:15]:
            print(f"  \"{g}\"")
            print(f"      in: {', '.join(x[:34] for x in c)}")
    else:
        print("  nessuno")

    print()
    print("=== argomenti ricorrenti (quante volte, e dove) ===")
    for etichetta, pattern in ARGOMENTI.items():
        dove = []
        for nome, corpo in caps:
            n = len(re.findall(pattern, corpo, re.I))
            if n:
                dove.append(f"{nome[:30]} ({n})")
        tot = sum(int(d.split("(")[-1].rstrip(")")) for d in dove)
        segno = "  <- da alleggerire" if tot > 3 else ""
        print(f"  {tot:>2}  {etichetta}{segno}")
        if dove:
            print(f"      {'; '.join(dove)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report/report3.tex")
