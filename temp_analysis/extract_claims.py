"""Estrae dal report ogni affermazione che contiene un numero, con il capitolo di
appartenenza, per poterla confrontare a mano con il file di output che l'ha prodotta.

Non verifica nulla da solo: il confronto e' semantico e va fatto leggendo. Serve a
garantire che nessuna affermazione sfugga al controllo.

    python temp_analysis/extract_claims.py report/report3.tex
"""

import re
import sys
from pathlib import Path

BS = chr(92)
# numeri che contano: decimali, notazione scientifica, moltiplicatori, percentuali
NUM = re.compile(r"(?<![a-zA-Z])(\d+\.\d+|\d+\s*(?:times|\\times)|\d+\s*%|10\^\{?-?\d)")
# rumore da ignorare: riferimenti a sezioni, equazioni, numeri di capitolo
IGNORA = re.compile(r"(ref\{|eqref\{|Section~|Chapter~|Table~|Figure~|Theorem~|"
                    r"Appendix~|label\{|cite)")


def frasi(testo):
    """Spezza in frasi, tenendo insieme le righe di uno stesso paragrafo."""
    testo = re.sub(r"\n\s*\n", "\n@@PARA@@\n", testo)
    testo = " ".join(r for r in testo.splitlines() if not r.lstrip().startswith("%"))
    for blocco in testo.split("@@PARA@@"):
        for f in re.split(r"(?<=[.:;])\s+", blocco):
            yield " ".join(f.split())


def main(path):
    testo = Path(path).read_text(encoding="utf-8")
    testo = testo[testo.index("maketitle"):]

    capitolo = "(abstract)"
    n_tot = 0
    for frase in frasi(testo):
        m = re.search(BS + r"section\{([^}]*)\}", frase)
        if m:
            capitolo = m.group(1)
            print(f"\n{'=' * 90}\n{capitolo}\n{'=' * 90}")
            continue
        if BS + "subsection{" in frase or not frase.strip():
            continue
        if IGNORA.search(frase) and not NUM.search(re.sub(IGNORA, "", frase)):
            continue
        if NUM.search(frase):
            pulita = re.sub(BS + r"[a-zA-Z]+\{?|\}|\$", "", frase)
            pulita = " ".join(pulita.split())
            if len(pulita) > 12:
                n_tot += 1
                print(f"  [{n_tot:>3}] {pulita[:250]}")
    print(f"\n\n{n_tot} affermazioni numeriche da verificare.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report/report3.tex")
