# Q&A sull'analisi del progetto — registro

Registro delle domande poste e delle risposte verificate, per non perderle. Ogni risposta
riporta solo ciò che è stato **letto nel codice**, **dimostrato** o **misurato**; dove una
mia affermazione è stata poi smentita, la smentita è nel testo e non è stata cancellata.

**Contesto del progetto.** Esame di IA statistica: una Parzen Neural Network deve imparare
la CDF di una distribuzione ignota già campionata, e da lì si deriva la pdf. Il progetto
sarà provato sui campioni del docente, da una distribuzione **fortemente multimodale**
ignota. La parte di scelta e campionamento della distribuzione non è un requisito: serve
solo a generare dati per verificare che il resto funzioni.

**Vincoli fissati durante l'analisi:**
- budget campionario **n ≈ 500**, al più 1000, non oltre (indicazione del docente);
- lo **stimatore di Parzen resta**: è parte dell'esercizio per costruzione;
- il **caso multivariato è fuori programma**;
- il codice "dietro le quinte" (`src/`, `scripts/`) è il riferimento; la **webapp diverge**
  ed è un ulteriore livello di incoerenza, da riallineare **dopo** aver sistemato il codice.

**Nota sul metodo, valida per tutto il registro.** Nessuna affermazione poggia su README,
`DESIGN.md`, `PRODUCT.md`, `docs/study*.md` o `report/*.tex`: quei documenti si contraddicono
a vicenda in almeno un punto accertato (vedi Q1/D2). Le fonti ammesse sono il sorgente, una
dimostrazione, una misura riproducibile o un riferimento bibliografico verificato.

---

## Q1 — Come fa la rete a garantire che l'uscita sia una CDF?

*(monotona crescente, da 0 a 1, derivabile perché deriva da una pdf)*

**Risposta breve.** Tre requisiti, tre meccanismi diversi, e quello che il progetto dichiara
non coincide con quello che il codice fa.

- **Dominio (0,1):** solo la sigmoide finale `[models.py:96]`.
- **Derivabilità:** imposta restringendo le attivazioni a quelle lisce `[models.py:29-34]`.
  È la parte fatta bene: con ReLU la pdf sarebbe costante a tratti.
- **Monotonia:** esistono tre strade nel codice (pesi positivi; penalità soft su 256 punti;
  rettifica a valle), e **quella effettivamente consegnata è la terza**: massimo cumulativo
  + riscalatura sulla griglia `[training.py:120-130]`, con la pdf ottenuta per **differenze
  finite**, non per derivazione della rete.

**Difetti accertati** (dettaglio completo, con prove, in `docs/redesign_network.md` §2):
attribuzione errata a Sill di una costruzione di Archer & Wang; README in contraddizione con
studio e report sullo stesso esperimento; modalità monotona raccomandata ma mai usata; «la
pdf è la derivata della rete» falsa nel percorso consegnato; saturazione strutturale (la rete
non può raggiungere 0 e 1); «massa = 1» artefatto della riscalatura su una griglia costruita
dalla verità; `clamp_min(0)` che rompe la relazione pdf/CDF; massimo cumulativo non ottimo;
penalità che certifica solo i punti che guarda; **nessuna standardizzazione dell'ingresso**,
quindi la stima non è invariante per traslazione (traslando i dati di +1000 la rete collassa
a KS 0.50, cioè restituisce la costante 0.5).

**Proposta.** Sostituire l'architettura con una **mistura convessa di CDF logistiche**
F(x) = Σ_j softmax(u)_j · σ(softplus(α_j)·x + b_j), che è il Deep Sigmoidal Flow senza il
logit di uscita. Garantisce per costruzione, per ogni valore dei parametri: F ∈ (0,1),
monotonia, F(−∞)=0 e F(+∞)=1 esatti, pdf ≥ 0 in forma chiusa, massa 1 — senza rettifica,
senza griglia, senza clamp. Più standardizzazione interna dell'ingresso.

**Esito misurato:** a parità di inizializzazione la proposta migliora l'ISE della densità in
8 casi su 9 (fattori 1.2×–10.8×) e batte il maestro Parzen in 9 su 9. Capacità scelta a
n=500: **J = 8**.

**Stato:** analizzato e progettato, **non ancora implementato**.
**Documento:** `docs/redesign_network.md` (con dimostrazioni T1–T6 ed esperimenti E1–E9).
**Banco di prova:** `temp_analysis/redesign_evidence.py` → `redesign_evidence.txt`.

---

## Q2 — Il numero di campioni è eccessivo? Quanto basta?

**Risposta.** Sì, 20000 è eccessivo, ma non perché peggiori: perché non compra nulla di
qualitativo. Misurato l'errore in variazione totale della pdf (½∫|f̂ − f|, con h scelto da un
oracolo, media su 4 semi), su quattro bersagli:

| n | trimodale | spike_in_broad | 5 mode strette | 6 mode scale miste |
|---|---|---|---|---|
| 50 | 12.4 % | 16.5 % | 16.2 % | 20.1 % |
| 100 | 9.8 % | 12.5 % | 13.2 % | 17.1 % |
| 250 | 8.5 % | 9.5 % | 10.0 % | 12.1 % |
| 500 | 7.0 % | 8.2 % | 8.1 % | 9.9 % |
| 1000 | 5.2 % | 6.2 % | 6.1 % | 7.7 % |
| 2000 | 4.0 % | 4.4 % | 4.6 % | 5.2 % |
| 5000 | 2.8 % | 3.5 % | 3.5 % | 4.0 % |

Sul lato CDF, con h ben scelto il KS è già al floor statistico 0.87/√n a n = 500 su tutte le
forme. Quindi: 500–1000 basta per una CDF indistinguibile dal massimo teorico; a 2000 anche
la pdf è pulita (~5 % di errore TV); sotto 200 si perdono mode. Regola pratica: ~200–500
punti per moda.

**Osservazione che resta valida a prescindere:** n non è una nostra scelta, lo dà il docente.
Ciò che conta è che nulla nella pipeline sia tarato su un n specifico — e qualcosa lo era
(la regola per h, la capacità della rete: entrambe ricalibrate a n = 500).

**Precisazione del docente.** Vuole n ≤ 500 perché ritiene che il Parzen dia già risultati
buoni con pochi punti *se la finestra è scelta bene*. Osservazione da mettere agli atti: è
un'affermazione che si verifica bene **a posteriori**, avendo la distribuzione vera per il
confronto; con in mano solo un campione la verifica è opinabile, ed è esattamente il motivo
per cui serve un selettore di finestra data-driven e non una costante calibrata a occhio.

**Stato:** chiuso. Assunto operativo: **n ≈ 500, al più 1000**.

---

## Q3 — La regola h₁ = 1.5·σ̂: ha senso cercare una regola così?

**Risposta.** No, e per tre motivi distinti, tutti misurati.

1. **La costante è un'interpolazione sul proprio benchmark.** Calibrata su una scaletta di
   3–4 misture più 10 misture generate da `random_mixture` (medie U(−5,5), std U(0.2,1.5)),
   cioè una famiglia dove σ̂ traccia la scala delle feature.
2. **σ̂ è la grandezza sbagliata.** Misura la dispersione **globale**, mentre h deve risolvere
   la feature **più stretta**. Su bersagli con mode separate — il caso del docente — σ̂ è
   gonfiato dalla separazione mentre l'h ottimo scende. Misurato: su "5 mode strette molto
   separate" l'h₁ ottimo è 0.32–0.51·σ̂, cioè 3–5× più piccolo di 1.5, e l'ISE a 1.5σ̂ è
   **7.6× peggiore** dell'oracolo a n=1000.
3. **Validata sulla metrica sbagliata.** Il KS integra e perdona: stesso caso, KS 2× peggiore
   ma ISE 7.6×. Il «siamo al floor statistico» è vero per la CDF e falso per la pdf, che è il
   deliverable.

**Osservazione strutturale.** Anche l'h₁ ottimo non è costante in n: cresce come ~n^0.25–0.30
in tutti i casi provati, che è esattamente il disallineamento fra lo schedule n^(−1/2) e il
tasso ottimo (n^(−1/5) per la pdf, n^(−1/3) per la CDF). A **n fissato** il problema si
dissolve — scegliere h₁ *è* scegliere h — ma la costante non è trasportabile fra budget
diversi.

**Nota sul perimetro.** La regola 1.5 **non è nella libreria**: `sqrt_n_bandwidth` ha
`h1=1.0` di default `[parzen.py:104-115]` e il 1.5 è cablato in due script
`[mlp2_battery.py:23; mlp2_pnn_cdf.py:30]`. La webapp non la espone affatto. È un'ulteriore
divergenza fra codice, studio e app.

**Stato:** **chiusa.** Selettore adottato: **LSCV**, integrato nello schedule
h_n = h₁/√n al quale il docente è affezionato (decisioni D-05 e D-06). Efficienza rispetto
all'oracolo, su 16 densità: 1.26 media e 1.59 nel caso peggiore, contro 2.27 e 9.38 della
regola 1.5·σ̂.
**Documento:** `docs/study_bandwidth.md`.
**Banco di prova:** `temp_analysis/bandwidth_study.py` + `bandwidth_run.py`.

---

## Q4 — Come funziona il campionamento? Cos'è il metodo Ziggurat?

**Perché la domanda è rilevante — correzione a una mia affermazione precedente.** Avevo
scritto che «sui campioni del docente questa sezione sparisce, perché il generatore non fa
parte della pipeline». **È sbagliato.** I dataset su cui sviluppiamo, calibriamo e misuriamo
tutto **li generiamo noi**: un generatore distorto contaminerebbe ogni conclusione a valle
senza dare alcun segnale. Il generatore non è in pipeline ma è a monte di ogni numero che
produciamo, e va quindi verificato. È il punto sollevato dal docente, ed è corretto.

Resta invece valida l'altra osservazione, ma va tenuta separata: **l'argomento
anti-circolarità del report non regge**. Non serve dimostrare che il generatore «non tocca
mai la CDF»: campioni i.i.d. da F non portano informazione su F oltre a quella statistica,
qualunque sia l'algoritmo che li ha prodotti. La non-circolarità si garantisce mostrando che
**lo stimatore vede solo x_1..x_n**, cosa che si verifica leggendo la loss. Le due questioni
— *qualità della generazione* (vera, da verificare) e *circolarità* (non un problema) — sono
state confuse nel report.

**Come funziona — e attenzione a non confondere due livelli diversi.** Campionamento
ancestrale e Ziggurat **non sono alternative**: sono strati che si compongono.

```
livello mistura     campionamento ancestrale       <- codice NOSTRO, data.py:70-73
                    rng.choice(k, p=pesi) -> i
                             |
livello componente  rng.normal(m_i, s_i)           <- chiamata a NumPy
                             |
livello normale     standard_normal() = ZIGGURAT   <- codice di NumPy
                             |
livello bit         PCG64
```

L'ancestrale decide *quale* gaussiana campionare; il Ziggurat è l'algoritmo che genera
materialmente ogni singola normale standard. Quindi **sì, il Ziggurat lo usiamo**, ma sotto e
non al posto dell'ancestrale. Nell'app la struttura è la stessa, con distribuzioni scipy
congelate `[server.py:88-94]`. **Nel repo non c'è alcuna implementazione di Ziggurat**: è
codice di NumPy, noi lo chiamiamo.

**Cos'è il Ziggurat.** Rejection sampling per densità decrescenti (Marsaglia & Tsang, 2000).
Si ricopre la densità con ~128–256 rettangoli di **area uguale** più una regione di coda; si
estrae l'indice di uno strato, poi un punto uniforme al suo interno; nel ~98–99 % dei casi il
punto cade nella parte del rettangolo interamente sotto la curva e si accetta subito, con un
confronto fra interi e **senza valutare exp()**. Solo nella frangia si valuta la densità, e
per lo strato di coda si usa un fallback. Costo ammortizzato ≈ 1 uniforme + 1 confronto,
contro Box–Muller che richiede log, sqrt e sin/cos. Il vecchio `RandomState` di NumPy usava
il metodo polare; `Generator` usa Ziggurat.

**Che NumPy usi davvero il Ziggurat: verificato, non citato.** In una prima stesura avevamo
scritto che la cosa non era verificabile dall'interno di Python. È falso, ci eravamo arresi
troppo presto: i docstring in effetti non nominano l'algoritmo, ma i simboli linkati sì.

| file (NumPy 2.4.6 installato) | occorrenze di "ziggurat" | simboli |
|---|---|---|
| `random/lib/npyrandom.lib` | 6 | `ziggurat_nor_r`, `ziggurat_nor_inv_r`, `ziggurat_nor_r_f`, `ziggurat_nor_inv_r_f`, `ziggurat_exp_r`, `ziggurat_exp_r_f` |
| `random/_generator…pyd` (`Generator`, quello che usiamo) | 1 | `Ziggurat` |
| `random/mtrand…pyd` (`RandomState`, il vecchio) | **0** | — |

`ZIGGURAT_NOR_R` è il raggio di taglio della coda nel ziggurat normale e `ZIGGURAT_NOR_INV_R`
il suo inverso: sono le costanti dell'algoritmo. L'assenza in `mtrand` conferma l'altra metà
del quadro, cioè che il vecchio `RandomState` usa il metodo polare e non il ziggurat.
Verificato inoltre che `rng.normal(loc, scale)` è una pura trasformazione affine di
`standard_normal` — a parità di seme i risultati sono **bit identici** — quindi eredita il
ziggurat.

**Cosa resta da verificare sull'uscita** (ed è ciò che conta di più): che sia distribuita come
la mistura richiesta, che le componenti escano con le probabilità giuste, che non ci sia
struttura seriale e che la scelta dei semi non introduca regolarità.

**Verifica eseguita** (`temp_analysis/sampling_quality.py` → `sampling_quality.txt`,
`sampling_seeds.txt`):

| test | esito |
|---|---|
| T1 aderenza alla CDF vera (KS, 300 repliche) | rifiuti al 5 %: **0.050** (n=500), **0.047** (n=2000) |
| T2 **uniformità dei p-value** (il test corretto, non «quanti rifiuti ho») | superato |
| T3 frequenze delle componenti contro i pesi (chi-quadro) | p = **0.533**; frequenze 0.3003 / 0.5011 / 0.1986 contro pesi 0.3 / 0.5 / 0.2 |
| T4 indipendenza seriale (autocorrelazione lag 1–100 su 200 000 campioni) | tutte entro ±0.0044; test dei run z = **−0.93** |
| T5 i semi 0,1,2,… introducono struttura? | **no**, vedi sotto |
| T6 ancestrale contro inversione numerica della CDF (**strategie di livello mistura**, entrambe con Ziggurat sotto) | indistinguibili (KS a due campioni p = 0.696) |

**T5 in dettaglio, perché ci riguarda direttamente.** Tutti i nostri esperimenti usano
`default_rng(0..k)`. Se semi vicini producessero stream correlati, le «repliche indipendenti»
non lo sarebbero e ogni media su semi sarebbe falsata. Con 300 repliche erano comparsi tre
valori borderline (uniformità p = 0.049, correlazione media/indice +0.092, KS a due campioni
p = 0.066), che potevano essere caso o segnale. Rifatto con **2000 repliche**:

| schema di semi | rifiuti @5 % | mediana p | uniformità | corr(media, indice) |
|---|---|---|---|---|
| 0…1999 (quelli che usiamo) | 0.0475 | 0.4912 | p = 0.834 | −0.0009 |
| casuali a 64 bit | 0.0405 | 0.4946 | p = 0.651 | −0.0066 |
| 10⁶…10⁶+1999 | 0.0505 | 0.5117 | p = 0.624 | −0.0185 |

(banda al 95 % per la correlazione: ±0.0438). Erano fluttuazioni. Controllo aggiuntivo: la
media delle 2000 medie campionarie è **0.69930** contro il valore vero **0.70000**, con
errore standard atteso della media di 2000 repliche pari a 0.0022 — uno scarto di 0.3
deviazioni standard.

**Conclusione.** Il metodo di generazione è adeguato e **va tenuto**, a entrambi i livelli.

- *Livello mistura:* l'ancestrale è **esatto per costruzione**, perché è la definizione stessa
  di mistura — P(X ≤ x) = Σ_i P(I=i)·P(X ≤ x | I=i) = Σ_i w_i F_i(x) — e costa O(n).
  L'inversione numerica della CDF sarebbe esatta solo entro la tolleranza della radice e
  costerebbe O(n × iterazioni di bisezione), senza alcun vantaggio nel nostro caso.
- *Livello normale:* il Ziggurat è **rejection sampling esatto**, non un'approssimazione
  veloce: i punti accettati sono distribuiti esattamente come N(0,1). La velocità non si paga
  in correttezza.

Nessuna conoscenza a priori sulla distribuzione entra nello stimatore.

**Da correggere nel report:** mantenere la sezione sul campionamento ma **ri-motivarla** —
non come garanzia anti-circolarità (argomento che non regge) bensì come garanzia di qualità
dei dati di sviluppo, con i test sopra a supporto. E dichiarare esplicitamente il confine fra
ciò che citiamo dalla documentazione (l'algoritmo) e ciò che verifichiamo noi (le proprietà
dell'uscita).

**Stato:** chiusa e verificata.

---

## Q5 — Come viene usata la CDF di riferimento?

**Risposta.** Usi legittimi e ben separati: come **metrica** (`ks_vs_truth`, `pdf_mse`, ISE) e
come **oracolo** dichiarato nella battery (h che minimizza il KS vero, esplicitamente un
limite superiore non raggiungibile). Verificato che la selezione di h e le etichette sono
**truth-free**: tutti i selettori usano solo i campioni, le etichette sono `parzen_cdf` o LOO,
e la verità non entra mai nella loss.

**Ma la verità entra in due punti che non sono valutazione:**

1. **La griglia.** `grid_for(mix)` = `means ± 5·stds` `[study2_common.py:21-24]`; nell'app
   `mix.grid()` usa `ppf(0.001/0.999)` delle componenti vere `[server.py:96-100]`. La griglia
   è il dominio su cui la CDF viene rettificata e la pdf differenziata: **il supporto stimato
   viene dalla verità**.
2. **`rectify_cdf`** normalizza in [0,1] usando i valori agli estremi di quella griglia: la
   massa unitaria è garantita rispetto a un dominio scelto conoscendo la distribuzione.

Non è leakage nel senso forte, ma è un'ipotesi che sui dati del docente non esiste.

**Il buco operativo.** Nessuno script accetta campioni esterni: in tutto `scripts/` e `src/`
non c'è un `argparse`, un `np.load`, una lettura CSV; `/api/parzen` pretende `components` per
costruire una `Mixture`; l'UI non ha upload. **Oggi la pipeline non è eseguibile sulla
batteria del docente senza scrivere codice nuovo**, e il pezzo mancante non è banale perché
griglia, rettifica e diagnostica sono tutte agganciate all'oggetto `Mixture`.

Da notare: con l'architettura proposta in Q1 il problema si riduce, perché lo stimatore torna
a essere una **funzione** valutabile ovunque e la griglia torna a essere solo uno strumento di
disegno e di misura.

**Stato:** **chiusa** (progettata, non implementata). Tre risultati hanno cambiato il
progetto rispetto alla prima ipotesi:

1. il dominio va ancorato alla **dispersione** (`[min − 3σ̂, max + 3σ̂]`), non alla finestra:
   massa vera lasciata fuori 1.1e-04 nel caso peggiore contro 1.7e-03 di un margine
   proporzionale ad h;
2. il **punteggio LSCV** è anche la migliore diagnostica truth-free (rho = 0.94 con l'errore
   vero), non solo un selettore;
3. il **KS contro l'ECDF è una trappola** e va bandito: rho = −0.03, premia h → 0. Nella
   riga qui sopra lo avevo proposto come diagnostica — era sbagliato, ed è la correzione più
   importante di Q5.

**Documento:** `docs/redesign_pipeline.md`.
**Banchi di prova:** `temp_analysis/pipeline_evidence.py`, `temp_analysis/grid_rule.py`.

---

## Q6 — Difetti trasversali

Difetti che non appartengono a nessuna singola area: attraversano libreria, script, app e
documentazione. Catalogati durante Q1–Q5 e **istruiti qui**, con verifica diretta.

### B1 — Il repo non si avvia su NumPy >= 2.4 `CORRETTO`

`getattr(np, "trapezoid", np.trapz)` valuta `np.trapz` **sempre**, perché Python valuta gli
argomenti prima della chiamata. Su NumPy >= 2.4, dove `trapz` è stato rimosso, l'import
solleva `AttributeError`. Verificato sull'ambiente corrente (NumPy 2.4.6):

| modulo | esito dell'import |
|---|---|
| `parzen_cdf` | ok |
| `parzen_cdf.data` | ok |
| **`parzen_cdf.parzen`** | **AttributeError** `[codice parzen.py:30]` |
| **`parzen_cdf.metrics`** | **AttributeError** `[codice metrics.py:12]` |
| `parzen_cdf.models` | ModuleNotFoundError: torch (vedi B6) |

Più due usi diretti che falliranno all'esecuzione: `[codice study2_common.py:37]` e
`[codice mlp2_monotonicity.py:57]`.

**Precisazione rispetto alla prima stesura:** i punti da correggere sono **quattro**, non tre.
E `[codice app/server.py:277]` — che avevamo contato fra i difetti — è invece **corretto**:
usa un ternario, quindi la valutazione è pigra.

Correzione applicata: `np.trapezoid if hasattr(np, "trapezoid") else np.trapz` in
`parzen.py:30` e `metrics.py:12`; gli usi diretti in `study2_common.py:37` e
`mlp2_monotonicity.py:57` passano ora dallo stesso guard pigro. Verificato: i quattro moduli
si importano, la suite e' verde, e la pipeline gira end-to-end.

**Test di regressione aggiunto** (`tests/test_imports.py`): importa ogni modulo, verifica che
il guard sia usabile e ispeziona il sorgente per impedire la reintroduzione del pattern
ansioso. Ne e' stata verificata la *capacita' di fallire*: reintroducendo il difetto di
proposito, tre test falliscono.

### B2 — Tre documenti si contraddicono sullo stesso esperimento

README dice che «la costruzione di Sill costa accuratezza» `[codice README.md:45]`;
`docs/study2.md` dice che «Sill è gratis in questo regime» `[codice docs/study2.md:149-153]`;
`report/report2.tex` dice che «il vincolo di Sill eguaglia l'accuratezza non vincolata»
`[codice report/report2.tex:586]`. Due su tre affermano il contrario del terzo. Va risolto
scegliendo quale è vero (lo studio corretto è `study2.md`) e allineando gli altri due — più
la correzione dell'attribuzione a Sill, vedi Q1/D1.

### B3 — Libreria, app e script espongono tre insiemi diversi di selettori

| dove | selettori disponibili |
|---|---|
| libreria `[parzen.py]` | `sqrt_n_bandwidth`, `silverman`, `variance_matched`, `likelihood_cv`, `lscv`, `adaptive` |
| app `[server.py:107-114]` | `manual`, `silverman`, `variance_matched`, `adaptive`, `likelihood_cv`, `lscv` |
| script della fase B | la regola σ (1.5) **cablata**, assente da entrambi gli altri |

Tre osservazioni: l'app **non espone `sqrt_n_bandwidth`**, cioè proprio lo schedule
h_n = h₁/√n a cui il docente è affezionato; la regola σ non è in libreria; e nessuno dei tre
insiemi contiene tutti gli altri. Con D-05 il problema si semplifica — il selettore diventa
uno solo — ma l'unificazione va fatta.

### B4 — La regola σ è cablata e non parametrizzata

`C_SIGMA_RULE = 1.5` in `[mlp2_battery.py:23]` e `[mlp2_pnn_cdf.py:30]`, mentre
`sqrt_n_bandwidth` in libreria ha `h1=1.0` di default `[parzen.py:104-115]`. Superato da D-05,
ma va rimosso invece che lasciato lì a contraddire la scelta nuova.

### B5 — Nessuna ingestione di campioni esterni

**Assorbito da Q5:** progettato in `redesign_pipeline.md` (P1 `run_from_samples`, P4 CLI).
Non è più una voce autonoma.

### B6 — La suite di test non era eseguibile `RISOLTO`

Mancavano **sia `pytest` sia `torch`**, quindi la suite non era mai stata eseguita e
l'affermazione implicita «i test passano» era **non verificata**.

Creato un ambiente isolato `.venv/` (già in `.gitignore`) con le dipendenze dichiarate in
`requirements.txt` — fra cui torch 2.13.0, pytest 9.1.1, numpy 2.4.6 — e il pacchetto
installato in modalità editable.

**Esito della prima esecuzione: 55 test superati, 0 falliti.** La suite preesistente era
quindi corretta; semplicemente nessuno l'aveva mai vista girare, e non copriva l'import dei
moduli (che era rotto).

**Prima esecuzione end-to-end della pipeline originale** (non delle nostre repliche NumPy),
mistura trimodale, n = 500: Parzen KS 0.0457, rete KS 0.0515, massa 1.0000, violazioni di
monotonia 0.00 %. Nota collaterale: su questo campione Silverman dà h = 0.5814 e LSCV
h = 0.0995, un fattore **5.8** — cioè il segnale «multimodale a scale miste» previsto come
diagnostica in `redesign_pipeline.md` P3, che si presenta già sulla mistura di default del
repo.

Resta valido che tre test verificano proprietà che con D-07 cambieranno (monotonia della
modalità `monotone`, `rectify_cdf`): la suite va comunque riscritta, vedi
`redesign_network.md` §6.4. Ma ora esiste una **baseline verde** da cui partire.

### B7 — Materiale superato ancora nel repo `nuovo`

`old/` contiene **31 file** (notebook, script, risultati e un report della prima passata);
`temp_analysis/` contiene **16 script** `wf*`/`test_*` di indagini precedenti; `docs/study.md`
sono **447 righe** di uno studio dichiarato superato da `study2.md`. Non è un difetto di
correttezza, ma è un problema di leggibilità per chi ispeziona il progetto, e rende ambiguo
quale sia il risultato valido. Da affrontare con P-04 (pulizia finale), non prima.

---

## Domande aperte — risposte ricevute

**1. Il deliverable ufficiale è la CDF o la pdf?**
→ **La pdf.** La rotta via CDF è lo scopo dell'esercizio, non una scorciatoia: si stima la
CDF con la Parzen Neural Network e da lì si deriva la densità. Conseguenza operativa: ogni
calibrazione va fatta sull'**ISE della pdf**, non sul KS della CDF (che integra e perdona).
Resta interessante un confronto fra la rotta-CDF e una PNN che stima direttamente la pdf, con
la riserva che sarebbero mele e pere: la pdf richiede una forma d'uscita diversa e quindi
un'architettura diversa. Parcheggiato come **P-01**, da riparlarne prima di impostarlo.

**2. Che tolleranza c'è sul supporto e su famiglie non gaussiane?**
→ Non è noto; è ragionevole aspettarsi misture di gaussiane, visto che il docente ha parlato
di multimodalità. Indicazione: se emergono **easy win** che rendono il progetto più
generalista (uniforme, esponenziale, Student-t, anche combinate) **senza perdere precisione**
sulle multimodali gaussiane, si prendono. Altrimenti ci si limita a uno studio finale di
resilienza, senza feature creep. Parcheggiato come **P-02**.

**3. Come verrà consegnato il modello?**
→ Il docente **deve poter usare tutto da solo**; se ci manda i campioni, tanto meglio. Ma la
priorità non è il pacchetto: è un **report umano e ben spiegato**, in cui ogni scelta
progettuale è discussa e motivata. Se il report convince, un'ispezione ravvicinata del codice
potrebbe non arrivare mai. Serve inoltre un pass finale di pulizia da artefatti e segni di
assistenza AI, a bassa priorità. Parcheggiato come **P-04**.

**4. Si può cambiare architettura senza uscire dalla traccia?**
→ **Nessun vincolo sull'architettura.** Si può progettare liberamente. Però prima si chiude
il pass sull'esistente: idee alternative di MLP vanno in uno studio a lato, da fare prima di
stilare il report. Parcheggiato come **P-03**.

**5. Il docente si aspetta lo schedule h_n = h₁/√n?**
→ **Sì, ed è un punto sensibile: ha criticato esplicitamente il passaggio h₁ = h·√n.** Non è
quindi una questione di presentazione ma di sostanza. La risoluzione adottata (decisione
**D-06**) è mantenere lo schedule nella forma che vuole lui e usare LSCV per **stimare** h₁,
esattamente come `1.5·σ̂` era già una stima dipendente dai dati: cambia la qualità
dell'estimatore, non la forma dello schedule. Vedi `study_bandwidth.md` §5.

---

## Domande ancora aperte

1. **Il docente accetta un h₁ *stimato dai dati*, o pretende una costante decisa a priori?**
   È il nodo della decisione D-06. Da notare che anche `h₁ = 1.5·σ̂` è una funzione del
   campione, quindi la richiesta di una costante universale sarebbe già violata dalla regola
   attuale: la nostra proposta cambia solo l'estimatore, in meglio e con le misure a supporto.
2. **Consegnerà un solo insieme di campioni o più d'uno?** Con uno solo non si può riportare
   variabilità e ogni risultato è un singolo tiro di dado; conviene saperlo prima di
   impostare la sezione dei risultati del report.
3. **n sarà esattamente 500?** L'indicazione è "≤ 500, circa 500". Se fosse molto più piccolo
   (100–200) alcune calibrazioni (J della rete, griglia dei candidati di LSCV) andrebbero
   rifatte a quel budget.
4. **Il report va scritto in italiano?** I sorgenti esistenti (`report.tex`, `report2.tex`)
   sono in italiano; si assume di sì salvo smentita.
5. **C'è una scadenza?** Determina quanto dei parcheggiati P-01…P-05 rientra nel perimetro.
