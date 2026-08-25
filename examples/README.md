# Campioni d'esempio

`campione.csv` sono 500 osservazioni da una mistura di quattro normali a scale miste, il caso
di interesse dell'esercizio. Un valore per riga. La distribuzione che le ha generate non e'
scritta da nessuna parte, ed e' il punto: la stima non deve poterla consultare.

```bash
python -m parzen_cdf.run --samples examples/campione.csv --out risultati/
```

`campione_excel_italiano.csv` sono gli stessi identici numeri scritti come li esporta un
foglio di calcolo con le impostazioni italiane, con la virgola decimale. Serve a vedere cosa
succede quando il file si puo' leggere in due modi:

```bash
python -m parzen_cdf.run --samples examples/campione_excel_italiano.csv --out risultati/
# si ferma e chiede: "1,5" e' un numero o due colonne?

python -m parzen_cdf.run --samples examples/campione_excel_italiano.csv --out risultati/ --decimal comma
# letto come si deve: stessi 500 valori dell'altro file
```

Senza quel controllo la seconda lettura passerebbe in silenzio, con mille valori al posto di
cinquecento e una stima dall'aspetto perfettamente normale calcolata su numeri inventati.
