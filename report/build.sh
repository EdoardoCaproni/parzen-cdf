#!/bin/sh
# Compila un report. Due passate: la prima genera l'indice, la seconda lo inserisce.
#
#   ./build.sh              compila report3.tex
#   ./build.sh report2      compila report2.tex
#
# MiKTeX e' installato in AppData e non sempre e' nel PATH della shell.
set -e
DOC="${1:-report3}"
export PATH="$PATH:$HOME/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" > /dev/null
echo "prodotto $DOC.pdf ($(wc -c < "$DOC.pdf") byte)"
grep -E "^(Overfull|Underfull|LaTeX Warning)" "$DOC.log" | sort -u | head -10 || true
