"""Controllo di sanità di un file .tex, in assenza di un compilatore.

Non sostituisce la compilazione: verifica solo gli errori che si commettono scrivendo a
mano un documento lungo, cioè ambienti e graffe sbilanciate. Serve perché su questa
macchina non c'è pdflatex e un errore banale si scoprirebbe solo alla consegna.
"""

import sys
from collections import Counter
from pathlib import Path

BS = chr(92)          # backslash, tenuto fuori dalle stringhe per non litigare con gli escape


def struttura(testo):
    for i, riga in enumerate(testo.splitlines(), 1):
        r = riga.strip()
        for marca, rientro in ((BS + "section{", "  "), (BS + "subsection{", "      - ")):
            if r.startswith(marca):
                yield i, rientro + r[len(marca):].rstrip("}")
                break


def controlla(path):
    testo = Path(path).read_text(encoding="utf-8")
    # Le sequenze protette vanno neutralizzate PRIMA di togliere i commenti e di contare:
    # altrimenti "\%" tronca la riga a meta' e "\{" viene contata come graffa vera. Senza
    # questo passaggio il controllo produce falsi allarmi, verificato su report2.tex.
    protetto = (testo.replace(BS + "%", "\x00")
                     .replace(BS + "{", "\x01").replace(BS + "}", "\x02"))
    codice = "\n".join(("" if r.lstrip().startswith("%") else r.split("%")[0])
                       for r in protetto.splitlines())

    aperti, chiusi = [], []
    for pezzo in codice.split(BS + "begin{")[1:]:
        aperti.append(pezzo.split("}")[0])
    for pezzo in codice.split(BS + "end{")[1:]:
        chiusi.append(pezzo.split("}")[0])
    sbil = (Counter(aperti) - Counter(chiusi)) + (Counter(chiusi) - Counter(aperti))

    print(f"--- {path}")
    for n, voce in struttura(testo):
        print(f"{n:>5}  {voce}")
    print()
    print(f"  ambienti: {len(aperti)} aperti, {len(chiusi)} chiusi -> "
          f"{'bilanciati' if not sbil else 'SBILANCIATI: ' + str(dict(sbil))}")
    print(f"  graffe:   {codice.count('{')} aperte, {codice.count('}')} chiuse -> "
          f"{'bilanciate' if codice.count('{') == codice.count('}') else 'SBILANCIATE'}")
    print(f"  sezioni: {codice.count(BS + 'section{')}   "
          f"sottosezioni: {codice.count(BS + 'subsection{')}   "
          f"da riempire: {testo.count('% TODO')}")
    return not sbil and codice.count("{") == codice.count("}")


if __name__ == "__main__":
    ok = all(controlla(p) for p in (sys.argv[1:] or ["report/report3.tex"]))
    sys.exit(0 if ok else 1)
