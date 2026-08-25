# Roadmap del progetto e registro delle decisioni

> **Nota di pulizia.** Questo documento cita `docs/study.md`, `docs/study2.md`,
> `report/report.tex` o `report/report2.tex`, rimossi perche' descrivevano un design
> sostituito. Restano recuperabili da `git log`. Dove questo documento e' in disaccordo
> con `report/report3.pdf`, vale il report.


Documento di coordinamento: dove siamo, cosa manca, cosa è già stato deciso e su quali prove.
Serve a non perdere il filo fra le fasi e a non ridiscutere due volte la stessa cosa.

**Obiettivo.** Una Parzen Neural Network che impara la CDF di una distribuzione ignota già
campionata, dalla quale si ricava la **pdf** (deliverable finale). Verrà provata sui campioni
del docente, da una distribuzione fortemente multimodale.

**Criterio di successo, in ordine.** (1) Un report umano, self-contained, in cui ogni scelta
progettuale è discussa e motivata con prove. (2) Un progetto che regge un'ispezione del
codice. Se (1) convince, (2) potrebbe non essere nemmeno esaminato da vicino.

---

## Fasi

### F1 — Revisione severa `in corso`

Passare al setaccio il progetto esistente, senza fidarsi di README, `docs/study*.md` o
`report/*.tex` (si contraddicono a vicenda in almeno un punto accertato). Ogni difetto
trovato va annotato in un documento di re-design **con la sua prova**.

| # | area | stato | documento |
|---|---|---|---|
| Q1 | architettura della rete e forma della CDF | **chiusa** (progettata, non implementata) | `redesign_network.md` |
| Q2 | numerosità campionaria | **chiusa** | `qa_analisi.md` |
| Q3 | scelta della finestra h | **chiusa** | `study_bandwidth.md` |
| Q4 | campionamento e Ziggurat | **chiusa** | `qa_analisi.md` |
| Q5 | uso della CDF di riferimento, ingestione campioni esterni | **chiusa** (progettata) | `redesign_pipeline.md` |
| Q6 | difetti trasversali (B1–B7) | **chiusa** (istruita) | `qa_analisi.md` |

**Non tutto ciò che va cambiato è la rete.** Finora sono emersi almeno quattro blocchi
distinti: architettura, selezione della finestra, uso della griglia/verità, ingestione dati.

### F2 — Refactor `in corso`

Si applica solo dopo che F1 ha prodotto la motivazione scritta.

| passo | stato |
|---|---|
| **B1** — guard pigro su `np.trapezoid` (4 punti) + test di regressione | **fatto** |
| **B6** — ambiente `.venv/`, dipendenze, baseline dei test | **fatto: 55 test verdi** |
| prototipo `MixtureCDFNet` + rivalidazione E4/E8/E9 su PyTorch | **fatto** |
| **B8** — il seme controlla l'inizializzazione, run riproducibili | **fatto** |
| test di caratterizzazione (rete di sicurezza per il refactor) | **fatto** |
| specifiche di dettaglio S1–S7 | **fatto** |
| `MixtureCDFNet` in `src/` + `fit_mixture_cdf` (D-07…D-11) | **fatto** |
| `report_domain` + `diagnostics.py` (D-12…D-14) | **fatto** |
| `run_from_samples`, `Estimate`, CLI (P-D3) | **fatto** |
| verifica end-to-end nuovo contro vecchio | **fatto** |
| B2 — allineamento di README, `study2.md`, `report2.tex` | con la stesura del report (F3) |
| B3, B4 — unificazione dei selettori e dell'app | **fatto** (`redesign_webapp.md`, C1–C6) |
| B7 — pulizia del materiale superato | con P-04, prima della consegna |
| D-15 — isolare in `evaluation.py` ciò che usa la verità | non necessario: il percorso nuovo già non importa `data`, verificato da test sull'AST |

**Baseline registrata:** prima esecuzione end-to-end del codice originale (trimodale,
n = 500): Parzen KS 0.0457, rete KS 0.0515, massa 1.0000, violazioni 0.00 %.

**Esito del refactor** (5 distribuzioni, 5 semi, n = 500; `redesign_pipeline.md` §9):

| percorso | KS medio | KS peggiore | ISE media | ISE peggiore |
|---|---|---|---|---|
| Parzen (LSCV) | 0.0370 | 0.0393 | 0.00514 | 0.00771 |
| vecchio | 0.0418 | 0.0592 | 0.00594 | 0.01026 |
| **nuovo** | **0.0361** | **0.0390** | **0.00402** | **0.00649** |

**101 test verdi.** Il percorso nuovo vince su tutte le aggregate, con il margine maggiore
nel caso peggiore e sul bersaglio più multimodale (KS 1.5×, ISE 3.5×).

### F3 — Report `non iniziata`

Documento umano, self-contained, che cita concretamente gli studi fatti. È il deliverable
che conta di più. Ogni studio di F1 ha già una sezione "frasi pronte per il report".

### F4 — Extra `non iniziata`

Deliverable di codice eseguibile dal docente, riallineamento della webapp, pulizia finale.

---

## Registro delle decisioni

Decisioni prese, con la prova che le sostiene. Non si riaprono senza una prova nuova.

| # | decisione | motivo / prova | stato |
|---|---|---|---|
| D-01 | **Deliverable = pdf**, ottenuta derivando la CDF stimata | è lo scopo dell'esercizio | fissata dall'utente |
| D-02 | **n ≈ 500**, al più 1000 | indicazione del docente | fissata |
| D-03 | **Il Parzen resta** la sorgente delle etichette | parte dell'esercizio per costruzione | fissata |
| D-04 | **Multivariato fuori programma** | deciso con l'utente | fissata |
| D-05 | **Finestra: LSCV**, integrata nello schedule h_n = h₁/√n | efficienza ISE 1.26 media / 1.59 peggiore contro 2.27 / 9.38 della regola 1.5σ̂, su 16 densità | `study_bandwidth.md` §4–5 |
| D-06 | **Lo schedule h_n = h₁/√n si mantiene** nella forma richiesta dal docente | vincolo esplicito; LSCV fornisce ĥ₁ = h_LSCV·√n, che è una stima come lo era 1.5σ̂ | `study_bandwidth.md` §5 |
| D-07 | **Architettura: mistura convessa di CDF logistiche** (DSF senza logit) | garanzie strutturali esatte anche in float32; migliore fra le architetture su 3 casi su 3, ma con margine 1.05×–1.9× (non 10× come suggeriva la replica NumPy) | `redesign_network.md` §3, T2, §4ter |
| D-08 | ~~J = 8~~ → **J = 12 componenti** | **rivista dopo la rivalidazione in PyTorch**: a J=8 il caso a 6 mode ha ISE doppia (0.00489 contro 0.00246); J=12 ha il miglior caso peggiore | `redesign_network.md` §4ter |
| D-09 | **Standardizzazione interna dell'ingresso** | senza, la stima non è invariante: traslando di +1000 la rete collassa a KS 0.50 | `redesign_network.md` T6, E7 |
| D-10 | **Niente rettifica, niente clamp** nel percorso principale | con D-07 non servono; il clamp rompe f = dF/dx | `redesign_network.md` T5, P4–P5 |
| D-11 | **Niente penalità di monotonia** | con D-07 non ha oggetto; certifica solo i punti che guarda | `redesign_network.md` D9, E3 |
| D-12 | **Dominio di valutazione = `[min(x) − 3σ̂, max(x) + 3σ̂]`** | massa vera fuori 1.1e-04 nel caso peggiore su 16 densità, contro 4.6e-03 del solo min/max | `redesign_pipeline.md` E2b |
| D-13 | **Diagnostica truth-free = punteggio LSCV + log-verosimiglianza LOO + massa** | rho 0.94 e 0.86 con l'errore vero | `redesign_pipeline.md` E3 |
| D-14 | **Vietato il KS contro l'ECDF come diagnostica** | anti-correlato con l'errore vero (rho −0.03); premia h → 0 | `redesign_pipeline.md` P-D5, E3 |
| D-15 | **Ciò che usa la verità va isolato in `evaluation.py`**, non importato dalla pipeline | separazione strutturale invece che disciplinare | `redesign_pipeline.md` P5 |
| D-16 | ~~Due modalità nell'app, consegna e studio~~ → **una sola app, tutto vivo** | lo scopo dell'app è esplorare le scelte sbagliate: metterle in una modalità secondaria le nasconde. I nostri valori sono i valori iniziali | `redesign_webapp.md` §0 |
| D-17 | **Si tiene l'app FastAPI**, la statica si ritira | importa la libreria, quindi non può divergere dal report; dalla statica si portano campo `h1`, `sqrt_n`, tessere di confronto | `redesign_webapp.md` §2 |
| D-18 | **Campione esterno = testo**, una colonna o separato da virgole | è il formato che si esporta davvero; fra 10 e 50000 osservazioni | `redesign_webapp.md` §4 C5 |
| D-19 | **Il KS contro l'ECDF è ammesso nell'app solo come tessera-trappola** | D-14 vieta di *decidere* su quel numero; qui lo si guarda fallire, ed è la dimostrazione più efficace | `redesign_webapp.md` §3 W8 |

---

## Studi collaterali parcheggiati

Da fare **prima del report**, ma non ora: chiudere prima il pass corrente, altrimenti si
finisce in feature creep.

| # | studio | priorità | nota |
|---|---|---|---|
| P-01 | Confronto rotta-CDF contro PNN diretta sulla pdf | **annullata** | la rotta via CDF è il progetto per costruzione: il docente chiede la stima della CDF tramite PNN e da lì la densità. Non c'è alcuna alternativa da confrontare |
| P-02 | Resilienza a componenti non gaussiane | **fatta** | `study_resilienza.md`. Nessun *easy win*: due sospetti su tre smentiti (le code pesanti sono più facili, il dominio si autocorregge), e l'unico limite vero — i bordi netti — non è riparabile alzando J senza pagare il 31–39 % sulle multimodali gaussiane. J = 12 confermato, limite dichiarato |
| P-03 | Architetture MLP alternative | **annullata** | feature creep: la mistura funziona, ha le garanzie strutturali e vince il confronto end-to-end. Cercarne altre è curiosità |
| P-04 | Pulizia finale: artefatti e segni di assistenza AI, stile del codice | bassa | pass immediatamente prima della consegna |
| P-05 | Riallineamento della webapp al codice | **da fare dopo il report** | confermato: il report è il deliverable principale e viene prima |

---

## Regole di lavoro

Valgono per tutti i documenti prodotti.

1. **Non fidarsi della documentazione esistente.** Fonti ammesse: il sorgente (con
   file:riga), una dimostrazione, una misura riproducibile, un riferimento verificato.
2. **Ogni affermazione porta la sua marca** (`[codice]`, `[T#]`, `[E#]`, `[lett.]`).
3. **Le previsioni smentite restano scritte**, con la smentita accanto. È già capitato cinque
   volte (PAVA e i plateau, il controesempio sulla penalità, la sovrastima del vantaggio del
   maestro a n=500, «le violazioni non si presentano», «il clamp non si attiva mai») e ogni
   volta la conclusione corretta era più solida, non più debole.
7. **Ciò che è misurato su una replica va rimisurato sul codice vero** prima di entrare in un
   documento definitivo. Due affermazioni su cinque non sono sopravvissute al passaggio
   (§4bis di `redesign_network.md`).
4. **I confronti vanno spogliati dei confondenti** prima di essere creduti (esempio:
   l'inizializzazione, che da sola spiegava metà del divario fra architetture).
5. **Ogni documento ha una sezione "cosa NON afferma".**
6. **I banchi di prova esterni battono quelli fatti in casa.** Un benchmark scelto da noi
   misura quanto bene ci siamo adattati a noi stessi.

---

## Tracciabilità delle prove

I risultati non vivono nella chat: ogni numero citato nei documenti è riprodotto da uno
script versionato.

| studio | script | output |
|---|---|---|
| architettura della rete | `temp_analysis/redesign_evidence.py` | `temp_analysis/redesign_evidence.txt` |
| libreria del banco rete (riusabile) | `temp_analysis/_redesign_lib.py` | — |
| selezione della finestra | `temp_analysis/bandwidth_study.py` + `bandwidth_run.py` | `bandwidth_n500.txt`, `bandwidth_n1000.txt` |
| esponente dello schedule | `temp_analysis/schedule_exponent.py` | `schedule_exponent.txt` |
| qualità del campionamento | `temp_analysis/sampling_quality.py` | `sampling_quality.txt`, `sampling_seeds.txt` |
| dominio e diagnostica truth-free | `temp_analysis/pipeline_evidence.py`, `grid_rule.py` | `pipeline_evidence.txt`, `grid_rule.txt` |
| confronto end-to-end nuovo/vecchio | `temp_analysis/endtoend_compare.py` | `endtoend_compare.txt` |
| resilienza a famiglie non gaussiane | `temp_analysis/resilience_non_gaussian.py`, `edges_capacity.py` | `resilience_non_gaussian.txt`, `edges_capacity.txt` |
| equivalenza Parzen replica/repo | — | `parzen_equivalence.txt` |
| rivalidazione della proposta su PyTorch | `temp_analysis/revalidate_torch.py`, `mixture_cdf_net.py` | `revalidate_torch.txt` |
| **verifica incrociata sul codice reale** | `temp_analysis/crosscheck_real_code.py` | `crosscheck_real_code.txt`, `clamp_on_violator.txt`, `seed_reproducibility.txt` |
