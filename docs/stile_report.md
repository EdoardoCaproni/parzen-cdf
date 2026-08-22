# Contratto di stile per il report

Il report è il lavoro finale di uno studente per il suo docente. Deve leggersi come tale:
scritto da una persona che ha fatto il lavoro e lo racconta, non come un documento
generato. Queste sono le regole che seguiamo, scelte perché sono **verificabili**: o si
rispettano o non si rispettano, e `temp_analysis/check_style.py` segnala le violazioni
meccaniche.

## Regole di superficie

| # | regola | perché |
|---|---|---|
| S1 | **niente trattini lunghi** (`---`). Si usa la virgola, i due punti, la parentesi, o si spezza la frase | è la spia più immediata, e nessuno scrive così a mano |
| S2 | niente aperture di paragrafo tipo «Two things:», «Three reasons:» | la segnaletica esplicita è un tic da generatore |
| S3 | niente «It is worth noting», «Crucially», «Importantly», «Notably» | se è importante lo si scrive e basta |
| S4 | evitare la struttura «non è X, è Y» ripetuta | va bene una volta, non tre per pagina |
| S5 | niente tricolon perfetti (tre elementi bilanciati) usati per ritmo | la simmetria eccessiva suona artificiale |
| S6 | niente meta-commento sul documento stesso | «questa sezione spiega», «come vedremo», istruzioni per la lettura |
| S7 | le liste puntate solo per elenchi veri, non per spezzare un ragionamento | un ragionamento è un paragrafo |

## Regole di sostanza

| # | regola | perché |
|---|---|---|
| C1 | **non si discute la nostra documentazione interna** | il docente non l'ha letta; correggerla in pubblico è archeologia inutile |
| C2 | **non si argomenta contro tesi che nessuno sostiene** | se un'affermazione sbagliata sta solo nei nostri vecchi file, si cancella e basta |
| C3 | la storia del progetto si racconta come **evoluzione del metodo**, non come cronaca dei nostri errori di scrittura | «prima usavamo X, costava Y, ora usiamo Z» sì; «il documento precedente affermava» no |
| C4 | una sezione «cosa non affermiamo» sola, nel capitolo sui limiti | averne una per capitolo è meccanico |
| C5 | i riquadri riassuntivi non contengono informazione nuova | così si possono togliere tutti senza perdere nulla |
| C6 | i numeri si citano con la loro provenienza, ma senza apparato di marche | «su sedici densità, media su cinque campioni» sì; le sigle tipo `[E4]` no |

## Come si applica

Durante la scrittura: `python temp_analysis/check_style.py report/report3.tex`.

Alla fine, una lettura completa che il controllore non può sostituire. Il criterio è
semplice: **una frase suona artificiale quando è più simmetrica di come parlerebbe una
persona.** Se tutte le frasi di un paragrafo hanno la stessa lunghezza e la stessa forma, il
paragrafo va riscritto anche se ogni singola frase è corretta.

## Cosa questo contratto non risolve

Il controllore trova le spie meccaniche, non la voce. Un testo può passare tutti i controlli
e restare piatto. L'unico rimedio è rileggere ad alta voce e chiedersi se si direbbe così.
