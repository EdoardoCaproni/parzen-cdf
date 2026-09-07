# Contratto di stile per il report

Il report è il lavoro finale di uno studente per il suo docente. Deve leggersi come tale:
scritto da una persona che ha fatto il lavoro e lo racconta, non come un documento
generato. Queste sono le regole che seguiamo, scelte perché sono **verificabili**: o si
rispettano o non si rispettano, e `temp_analysis/check_style.py` segnala le violazioni
meccaniche.

## Regole che il controllore verifica

I codici sono quelli del controllore. Prima erano sfasati rispetto a questo documento, il che
rendeva impossibile passare da una segnalazione alla regola che la motiva.

| # | regola | perché |
|---|---|---|
| S1 | **niente trattini lunghi** (`---`). Si usa la virgola, i due punti, la parentesi, o si spezza la frase | è la spia più immediata, e nessuno scrive così a mano |
| S2 | niente aperture di paragrafo tipo «Two things:», «Three reasons:» | la segnaletica esplicita è un tic da generatore |
| S3 | niente «It is worth noting», «Crucially», «Importantly», «Notably» | se è importante lo si scrive e basta |
| S4 | evitare «non è X, è Y», «not X but Y», «rather than merely» | va bene una volta, non tre per pagina |
| S5 | niente meta-commento sul documento stesso | «questa sezione spiega», «come vedremo», istruzioni per la lettura |
| S6 | niente formule di transizione («That said», «With that in mind») | riempitivi che non spostano l'argomento |
| S7 | niente «is worth doing» in serie. Soglia: quattro | non è un errore, ma la frequenza tradisce una voce sola che applica una formula |
| S8 | niente rivelazioni dopo i due punti («la parte migliore: impara»). Soglia: sei | i due punti prima di una formula sono normali, la posa retorica no |
| S9 | niente obiezioni o domande fabbricate per poi risolverle | «sarebbe tentante pensare che», «l'obiezione standard è», «c'è una domanda preliminare» |
| S10 | niente commenti su cosa il lettore deve notare | «la colonna che conta di più», «merita un commento», «vogliamo richiamare l'attenzione» |
| S11 | niente elenchi negativi usati come chiusura a effetto | «niente griglia, niente derivazione, niente da regolare» |

Le regole da S8 a S11 sono arrivate da una revisione esterna nel settembre 2026, che trovò
una sessantina di occorrenze mentre il controllore ne segnalava zero. Il difetto non erano le
regole ma la loro larghezza: S4 pretendeva la virgola in una posizione fissa.

### Esenzioni

Un passaggio può finire sotto una regola e restare lo stesso. In quel caso si dichiara in
`ESENZIONI` dentro il controllore, con la ragione. **Non si allenta l'espressione regolare**:
diventerebbe cieca anche sui casi veri, e la decisione sparirebbe dalla vista. Oggi ce ne
sono due, entrambe in cui la forma segnalata porta contenuto tecnico.

## Regole che restano a chi rilegge

Il controllore non le vede, e nessuna espressione regolare le vedrà.

| # | regola | perché |
|---|---|---|
| H1 | niente tricolon perfetti (tre elementi bilanciati) usati per ritmo | la simmetria eccessiva suona artificiale |
| H2 | le liste puntate solo per elenchi veri, non per spezzare un ragionamento | un ragionamento è un paragrafo |
| H3 | niente serie di paragrafi che iniziano con la stessa formula | quattro «We do not claim» di fila si leggono come un modulo compilato |

## Regole di sostanza

| # | regola | perché |
|---|---|---|
| C1 | **non si discute la nostra documentazione interna** | il docente non l'ha letta; correggerla in pubblico è archeologia inutile |
| C2 | **non si argomenta contro tesi che nessuno sostiene** | se un'affermazione sbagliata sta solo nei nostri vecchi file, si cancella e basta |
| C3 | la storia del progetto si racconta come **evoluzione del metodo**, non come cronaca dei nostri errori di scrittura | «prima usavamo X, costava Y, ora usiamo Z» sì; «il documento precedente affermava» no |
| C4 | una sezione «cosa non affermiamo» sola, nel capitolo sui limiti | averne una per capitolo è meccanico |
| C5 | ~~i riquadri riassuntivi non contengono informazione nuova~~ **niente riquadri riassuntivi** | erano ammessi a condizione che non aggiungessero nulla; rispettata quella condizione, toglierli non fa perdere niente, ed è quello che il gruppo ha chiesto in revisione |
| C6 | i numeri si citano con la loro provenienza, ma senza apparato di marche | «su sedici densità, media su cinque campioni» sì; le sigle tipo `[E4]` no |
| C7 | **niente promesse scoperte**: se il testo dice che una cosa è stata misurata, deve esistere lo script che l'ha misurata | la revisione ha trovato una sezione che rimandava a un capitolo per una misura che quel capitolo non conteneva |

## Come si applica

Durante la scrittura: `python temp_analysis/check_style.py report/report3.tex`.

Alla fine, una lettura completa che il controllore non può sostituire. Il criterio è
semplice: **una frase suona artificiale quando è più simmetrica di come parlerebbe una
persona.** Se tutte le frasi di un paragrafo hanno la stessa lunghezza e la stessa forma, il
paragrafo va riscritto anche se ogni singola frase è corretta.

## Cosa questo contratto non risolve

Il controllore trova le spie meccaniche, non la voce. Un testo può passare tutti i controlli
e restare piatto. L'unico rimedio è rileggere ad alta voce e chiedersi se si direbbe così.

E non trova i difetti di sostanza. La revisione di settembre ha segnalato una sessantina di
questioni di forma, e il difetto più grave del report in quel momento era che una sezione
prometteva una misura mai eseguita: nessun controllo di superficie poteva vederlo.
