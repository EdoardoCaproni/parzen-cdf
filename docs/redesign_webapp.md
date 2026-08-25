# Riallineamento della webapp — piano

Stato: **da approvare**. Nessuna riga di `app/` è stata toccata.

## 0. Che cosa deve essere l'app

Uno strumento di esplorazione. Non una vetrina del percorso consegnato, ma il banco su cui si
prende la strada sbagliata e si guarda che cosa succede: finestra troppo larga, maestro che
guarda sé stesso, rete senza vincoli, campioni fuori dalle ipotesi. Ogni scelta che il progetto
ha fatto deve essere **una manopola**, con accanto il numero che dice quanto costa spostarla.

Da qui discende tutto il resto, e in particolare la caduta della divisione fra "consegna" e
"studio" che avevo proposto nella prima stesura: non ci sono due modalità, c'è una sola app in
cui tutto è vivo e i nostri valori sono i **valori iniziali**.

## 1. Le due app esistenti, e chi le ha scritte

| | dove | motore | autore |
| --- | --- | --- | --- |
| **FastAPI** | `parzen-cdf/app/` | importa `parzen_cdf` (Python) | ppianigi, commit `8245ef7` |
| **statica** | `github.com/Gianeh/parzen-lab` | riscritto in JS (`math.js`, `worker.js`) | Gianeh |

Sono la **stessa app**. Il confronto dei sorgenti: 29 righe di differenza su 347 in
`index.html`, 20 su 461 in `style.css`, 199 su 1308 in `app.js`. La versione statica è la stessa
interfaccia con il motore riportato in JavaScript, più quattro aggiunte: il campo `h1` per la
strategia `h_n = h1/sqrt(n)`, Sill tolto dal menu, le tessere di confronto MLP-Parzen, e il
salvataggio dei checkpoint come file scaricato.

Entrambe sono di Pietro. La scelta quindi non è *di chi è il codice*, ma quale delle due basi è
più adatta.

## 2. La strada scelta: riallineare l'app FastAPI, ritirare la statica

`parzen-cdf/app/` resta, e importa la libreria che abbiamo rifatto. Dell'app statica si portano
dentro le quattro aggiunte utili (campo `h1`, tessere di confronto MLP-Parzen, aiuto sul
checkpoint, e la strategia `sqrt_n`); **non** si portano le restrizioni temporanee di Pietro
(Sill nascosto, selettori ridotti a due), che vanno contro lo scopo dell'app.

Le ragioni, dopo aver pesato le due strade.

1. **Non esiste una seconda implementazione.** Quello che l'app mostra *e'* quello che
   consegniamo, per costruzione, non entro tolleranza. Cade il banco di riscontro, che sulla
   strada statica non era un extra ma una necessita', e cade il suo rischio residuo: un test di
   riscontro copre i casi che ha, non tutti.
2. **Meno lavoro, di natura piu' sicura.** `MixtureCDFNet`, `lscv_bandwidth`,
   `loo_parzen_cdf_targets`, `diagnose` esistono e sono coperti da 101 test. Sulla strada
   statica ci sarebbe stato il porto del modello con i gradienti scritti a mano.
3. **Niente reimplementazione di scipy.** `math.js` ha dovuto riscrivere in casa `erf`,
   `lgamma`, `ibeta`, il campionamento Gamma e la ppf numerica.
4. **Continuita' degli artefatti.** Il checkpoint e' un `.pt` con lo `state_dict`, lo stesso
   formato di `Estimate.save`, ricaricabile dalla libreria.
5. **Nessun soffitto**: float64, CV su n grandi, CPU vera invece del thread del browser.

Il vantaggio della statica era uno solo, aprirsi da un link senza installare niente. Vale se
qualcuno la aprira' **senza avere Python**; ma il deliverable del progetto e' codice Python, e
chi lo esegue ha gia' l'ambiente. Scenario escluso dal committente.

## 3. I nove scostamenti, come manopole

Ogni difetto trovato diventa una manopola più la lezione che si vede muovendola.

| | manopola | valore iniziale | che cosa si vede spostandola |
| --- | --- | --- | --- |
| **W1** | stimatore: MLP libero / MLP a pesi positivi / mistura | **mistura, J = 12** | si perturbano i pesi: l'MLP libero viola la monotonia in 900 casi su 1000, la mistura in 0. La validità per costruzione si *guarda* invece di leggerla (T2) |
| **W2** | maestro: leave-one-out / autoinclusivo, e la larghezza `teacher_scale` | **LOO, 0.5·h** | il maestro autoinclusivo contiene il punto stesso; con `teacher_scale = 1` sparisce la ricetta PNN e l'etichetta è più liscia del dovuto |
| **W3** | selettore: `sqrt_n` (con `h1` a mano) / `lscv` / `likelihood_cv` / `variance_matched` / `silverman` / `adaptive` / manuale | **`lscv`**, e `h1 = h·sqrt(n)` sempre in vista | si scorre `h1` e si vedono sovralisciamento e sottolisciamento; `silverman` con il nucleo logistico sovraliscia di pi/sqrt(3) = 1.814 |
| **W4** | rettifica a valle | **spenta** | con la mistura non ha niente da correggere; accendendola sull'MLP libero si vede che è una toppa |
| **W5** | penalità di monotonia sui 256 punti di collocazione | **spenta** | l'app mostra una violazione **fra** due punti di collocazione: certifica solo dove guarda |
| **W6** | dominio: dai campioni `[min-3s, max+3s]` / dai quantili veri | **dai campioni** | sui campioni del docente la verità non c'è: il secondo non è disponibile, e va visto |
| **W7** | **origine dei dati: mistura costruita / file caricato** | mistura | è la capacità che manca del tutto. Con un file, i grafici e le tessere che usano la verità si spengono |
| **W8** | blocco diagnostico | punteggio LSCV, log-verosimiglianza LOO, massa | e la tessera **KS contro l'ECDF marcata come trappola**: si scorre `h` verso zero e la si vede migliorare mentre l'errore vero peggiora (rho = -0.03). È la dimostrazione più efficace che abbiamo |
| **W9** | testi | — | `STRATEGY_NOTES`, la nota sulla penalità, l'aiuto sulla rettifica: riscritti sui numeri del report |

Su **W8** una precisazione: D-14 vieta il KS contro l'ECDF *come diagnostica su cui decidere*.
Mostrarlo etichettato come trappola, in uno strumento che serve a far vedere le strade sbagliate,
è l'uso opposto e va bene. Il testo accanto deve dirlo.

## 4. Lavoro, nell'ordine

Un commit per punto, dal piu' sostanzioso al piu' cosmetico.

**C1 — lo stimatore (W1, W4, W5).** Nuovo menu `estimator`: `mixture` (iniziale) oppure `mlp`.
Con `mlp` restano vive le manopole di monotonia (nessuna / pesi positivi) e la rettifica, che
con `mixture` non hanno oggetto e si disattivano. `_weight_snapshot` emette la mistura nella
forma 1 -> J -> 1, cosi' lo schema SVG continua a funzionare senza toccarlo: `a_j = softplus`
sull'arco d'ingresso, `b_j` polarizzazione, `pi_j = softmax` sull'arco d'uscita.

Sulla potatura c'e' un vincolo vero: azzerare un arco d'**uscita** lascia la stima valida
(resta una combinazione convessa di sigmoidi), azzerare un arco d'**ingresso** rende l'unita'
costante a 1/2 e porta `F(-inf)` a `pi_j/2 > 0`, cioe' rompe la CDF. Quindi con la mistura si
potano solo gli archi d'uscita, e sull'arco d'ingresso il clic spiega invece di eseguire.

**C2 — maestro leave-one-out (W2).** Etichette `(n*F(x_i) - 1/2)/(n-1)` con finestra
`teacher_scale * h`, `teacher_scale = 0.5` iniziale. L'autoinclusivo resta come confronto.

**C3 — selettori e dominio (W3, W6).** `sqrt_n` con campo `h1`, `lscv` iniziale,
`h1 = h*sqrt(n)` sempre riportato, dominio dai campioni `[min-3s, max+3s]`.

**C4 — diagnostica (W8).** Il blocco senza verita', e la tessera-trappola KS-contro-ECDF con la
sua etichetta.

**C5 — caricamento di un campione (W7).** `POST /api/samples` piu' un selettore di file nella
UI. Testo: una colonna, o separato da virgole. Senza verita', i grafici che la usano si spengono.

**C6 — confronto e testi (W9).** Tessere MLP-contro-Parzen dalla statica, e la riscrittura di
`STRATEGY_NOTES`, della nota sulla penalita' e dell'aiuto sulla rettifica sui numeri del report.

## 5. Che cosa non tocco

Layout, foglio di stile, grafici, protocollo del worker, registro delle distribuzioni,
costruttore di misture, potatura, checkpoint. Cambia che cosa l'app calcola e che cosa dice,
non come si presenta.

## 6. Verifica

1. `pytest -q` invariato (101 test).
2. Server avviato, i tre stadi percorsi a mano con ogni stimatore.
3. Gli stessi campioni e lo stesso seme in `run_from_samples` e nell'app: stessa `h`, stessa curva.
4. Un file di campioni caricato a mano: la stima esce senza che nessuna verità entri.

## 7. Decisioni

| | decisione | esito |
| --- | --- | --- |
| **D-16** | due modalità (consegna / studio) | **ritirata**: una sola app, tutto vivo, i nostri valori come iniziali |
| **D-17** | quale base | **la FastAPI**: importa la libreria, quindi non puo' divergere dal report. Dalla statica si portano le quattro aggiunte utili |
| **D-18** | formato del campione caricato | testo, una colonna o separato da virgole |
| **D-19** | KS contro l'ECDF nell'app | ammesso **solo** come tessera-trappola etichettata (non contraddice D-14) |
