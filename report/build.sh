#!/bin/sh
# Compila un report e riporta gli avvisi tipografici che vale la pena guardare.
#
#   ./build.sh              compila report3.tex
#   ./build.sh report2      compila report2.tex
#
# MiKTeX si installa in AppData e non finisce nel PATH della shell, quindi lo si aggiunge
# qui invece di chiederlo a chi compila.
set -e
DOC="${1:-report3}"
export PATH="$PATH:$HOME/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
cd "$(dirname "$0")"

# Tre passate: la prima genera l'indice, la seconda lo inserisce cambiando la numerazione
# delle pagine, la terza stabilizza i riferimenti incrociati.
for _ in 1 2 3; do
  if ! pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" > /dev/null 2>&1; then
    echo "compilazione fallita; ultime righe di $DOC.log:"
    tail -25 "$DOC.log"
    exit 1
  fi
done

echo "prodotto $DOC.pdf ($(wc -c < "$DOC.pdf") byte)"

# Gli hbox troppo larghi sono righe che sbordano nel margine e vanno sistemate.
# I vbox troppo alti su un documento ancora corto sono normalmente la pagina dell'indice.
grep -E "^(Overfull|Underfull|LaTeX Warning)" "$DOC.log" | sort -u | head -10 || true
