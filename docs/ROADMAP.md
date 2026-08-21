# Roadmap del progetto e registro delle decisioni

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
| `MixtureCDFNet` + standardizzazione + pdf in forma chiusa (D-07…D-11) | prossimo |
| `report_domain`, `run_from_samples`, diagnostica truth-free (D-12…D-15) | dopo |
| B2, B3, B4 — allineamento documenti e unificazione selettori | contestuale |

**Baseline registrata:** prima esecuzione end-to-end del codice originale (trimodale,
n = 500): Parzen KS 0.0457, rete KS 0.0515, massa 1.0000, violazioni 0.00 %.

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
| D-07 | **Architettura: mistura convessa di CDF logistiche** (DSF senza logit) | garantisce per costruzione dominio, monotonia, code, massa e pdf ≥ 0; ISE migliore in 8 casi su 9 | `redesign_network.md` §3, T2 |
| D-08 | **J = 8 componenti** | calibrato a n = 500: J < 8 fallisce sulle multimodali, J > 8 sovradatta | `redesign_network.md` E8 |
| D-09 | **Standardizzazione interna dell'ingresso** | senza, la stima non è invariante: traslando di +1000 la rete collassa a KS 0.50 | `redesign_network.md` T6, E7 |
| D-10 | **Niente rettifica, niente clamp** nel percorso principale | con D-07 non servono; il clamp rompe f = dF/dx | `redesign_network.md` T5, P4–P5 |
| D-11 | **Niente penalità di monotonia** | con D-07 non ha oggetto; certifica solo i punti che guarda | `redesign_network.md` D9, E3 |
| D-12 | **Dominio di valutazione = `[min(x) − 3σ̂, max(x) + 3σ̂]`** | massa vera fuori 1.1e-04 nel caso peggiore su 16 densità, contro 4.6e-03 del solo min/max | `redesign_pipeline.md` E2b |
| D-13 | **Diagnostica truth-free = punteggio LSCV + log-verosimiglianza LOO + massa** | rho 0.94 e 0.86 con l'errore vero | `redesign_pipeline.md` E3 |
| D-14 | **Vietato il KS contro l'ECDF come diagnostica** | anti-correlato con l'errore vero (rho −0.03); premia h → 0 | `redesign_pipeline.md` P-D5, E3 |
| D-15 | **Ciò che usa la verità va isolato in `evaluation.py`**, non importato dalla pipeline | separazione strutturale invece che disciplinare | `redesign_pipeline.md` P5 |

---

## Studi collaterali parcheggiati

Da fare **prima del report**, ma non ora: chiudere prima il pass corrente, altrimenti si
finisce in feature creep.

| # | studio | priorità | nota |
|---|---|---|---|
| P-01 | Confronto rotta-CDF contro PNN diretta sulla pdf | media | rischio mele/pere: la pdf richiede una forma d'uscita diversa, quindi un'architettura diversa. Da riparlarne prima di impostarlo |
| P-02 | Resilienza a componenti non gaussiane (uniforme, esponenziale, Student-t, miste) | media | se emergono *easy win* che generalizzano senza costare precisione sulle multimodali gaussiane, si prendono; altrimenti diventa solo uno studio finale di resilienza |
| P-03 | Architetture MLP alternative | bassa | nessun vincolo d'esame sull'architettura; studio a lato prima del report |
| P-04 | Pulizia finale: artefatti e segni di assistenza AI, stile del codice | bassa | pass immediatamente prima della consegna |
| P-05 | Riallineamento della webapp al codice | bassa | la webapp diverge dal codice "dietro le quinte"; si allinea quando il progetto è solido |

---

## Regole di lavoro

Valgono per tutti i documenti prodotti.

1. **Non fidarsi della documentazione esistente.** Fonti ammesse: il sorgente (con
   file:riga), una dimostrazione, una misura riproducibile, un riferimento verificato.
2. **Ogni affermazione porta la sua marca** (`[codice]`, `[T#]`, `[E#]`, `[lett.]`).
3. **Le previsioni smentite restano scritte**, con la smentita accanto. Sono già capitate tre
   volte (PAVA e i plateau, il controesempio sulla penalità, la sovrastima del vantaggio del
   maestro a n=500) e ogni volta la conclusione corretta era più solida, non più debole.
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
