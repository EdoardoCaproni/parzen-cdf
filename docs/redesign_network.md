# Re-design della rete neurale: dalla CDF "corretta a valle" alla CDF corretta per costruzione

**Ambito.** Solo il blocco *rete*: come l'uscita della rete viene resa una funzione di
ripartizione (dominio, monotonia, derivabilità, code, massa) e come da essa si ricava la
densità. Restano fuori — e verranno trattate in documenti separati — la scelta della
finestra di Parzen, la numerosità campionaria, il campionamento, e l'uso della CDF di
riferimento. Dove il confine è poroso lo segnalo esplicitamente.

**Vincoli fissati.** (a) Il budget di riferimento è **n ≈ 500**, indicato dal docente: tutte
le scelte tarabili sono calibrate lì `[E8]` `[E9]`. (b) Lo stimatore di Parzen **resta** la
sorgente delle etichette: è parte dell'esercizio per costruzione, e nessuna proposta di
questo documento lo tocca. (c) Il caso multivariato è fuori programma.

**Perché esiste questo documento.** Il codice attuale funziona, ma quello che *garantisce*
non è quello che la documentazione *dichiara*, e le garanzie che offre sono ottenute a
valle, su una griglia, invece che dentro il modello. Questo documento (a) mappa cosa fa
davvero il codice, riga per riga; (b) elenca i difetti accertati, ciascuno con la sua prova;
(c) propone le modifiche; (d) le dimostra, matematicamente dove è possibile e
numericamente altrimenti; (e) dice cosa NON stiamo affermando.

**Regola di ingaggio.** Nessuna affermazione di questo documento poggia su README, DESIGN,
PRODUCT, `docs/study*.md` o `report/*.tex`. Ogni riga è marcata con la sua fonte:

| marca | significato |
|---|---|
| `[codice f:r]` | letto direttamente nel sorgente indicato |
| `[T#]` | dimostrato analiticamente in §5 |
| `[E#]` | misurato da `temp_analysis/redesign_evidence.py`, output in `redesign_evidence.txt` |
| `[lett.]` | riferimento bibliografico verificato, §9 |

Il banco di prova è implementato **da zero in NumPy**, con verifica dei gradienti per
differenze finite (errore relativo max ≈ 1e-8, `[E1a]`): non dipende da PyTorch e non
riusa nulla del repo, così un errore del repo non può propagarsi nelle misure che lo
giudicano.

---

## 1. Cosa fa davvero il codice oggi

L'architettura è `CDFNet` `[codice src/parzen_cdf/models.py:37-96]`. Con una cifra nascosta
di J unità e ingresso scalare, la funzione realizzata è

$$F_\theta(x)=\sigma\!\Big(\textstyle\sum_{j=1}^{J} w_j\,\sigma(a_j x + b_j) + c\Big)$$

con σ la sigmoide logistica. Due modalità:

- **non vincolata** (`monotone=False`): `a`, `w` liberi, inizializzazione Xavier uniforme e
  **bias a zero** `[codice models.py:76-80]`;
- **"Sill"** (`monotone=True`): peso efficace `softplus(raw)`, quindi positivo
  `[codice models.py:82-84]`; inizializzazione via softplus inversa `[codice models.py:71-75]`.

In entrambe, l'ultima operazione è una sigmoide `[codice models.py:96]`. Le attivazioni sono
ristrette a quelle lisce `[codice models.py:29-34]`, e in modalità monotona anche a quelle
non decrescenti `[codice models.py:59-60]`.

Il resto della pipeline:

| passo | dove | cosa fa |
|---|---|---|
| etichette | `[codice study2_common.py:89-97]` | CDF di Parzen leave-one-out ai soli punti campione |
| training | `[codice study2_common.py:105-117]` | Adam, lr 0.03, 6000 epoche, full batch |
| monotonia (penalità) | `[codice training.py:85-94, 146-148]` | `mean(relu(-dF/dx))` su 256 punti equispaziati |
| monotonia (a valle) | `[codice training.py:120-130]` | massimo cumulativo + riscalatura in [0,1] |
| densità (percorso A) | `[codice training.py:112-117]` | autograd, con `clamp_min(0)` **di default** |
| densità (percorso B) | `[codice training.py:129]` | `np.gradient` della curva rettificata |
| app | `[codice app/server.py:261-269, 325]` | stesso schema, penalità su 256 punti della griglia |

**Quale percorso è quello consegnato.** Tutti gli script della fase corrente
(`checkpoint1.py`, `mlp2_battery.py`, `study2_common.eval_cdf_net`) e l'app usano
`rectify_cdf`, cioè il **percorso B**: la densità consegnata è una **differenza finita
sulla griglia** di una curva post-processata, non la derivata della rete
`[codice study2_common.py:120-128; app/server.py:267]`.

---

## 2. Difetti accertati

Ogni voce: cosa afferma il progetto → cosa fa il codice → prova.

### D1 — Attribuzione errata della costruzione monotona `[lett.]`

Il codice chiama "Sill, 1998" la modalità a pesi non negativi
`[codice models.py:11-14]`. Il lavoro di Sill (*Monotonic Networks*, NIPS 1997) propone una
**rete min-max**: primo strato lineare a pesi vincolati positivi, `max` su gruppi di
iperpiani, poi `min` dei massimi; è un approssimatore universale delle funzioni monotone.
La costruzione "pesi positivi + attivazione monotona", che è quella implementata, è di
Archer & Wang (1993) ed è precisamente il punto di *partenza* che Sill critica per la sua
scarsa espressività. Il repo non implementa nulla di Sill.

**Perché conta:** è un errore verificabile in trenta secondi da chiunque conosca la
letteratura, e sta in un commento del sorgente, in `docs/study2.md`, nel README e nel
report. Va corretto ovunque.

### D2 — Il README contraddice lo studio e il report sullo stesso esperimento

- README: «the soft penalty fails and **the Sill construction costs accuracy**; downstream
  rectification wins» `[codice README.md:45]`
- `docs/study2.md`: «**Sill is free in this regime** … Practical recipe: rectification
  alone, **or Sill + rectification**» `[codice docs/study2.md:149-153]`
- `report/report2.tex`: «Sill's constraint **matches the unconstrained accuracy**»
  `[codice report/report2.tex:586]`

Due dei tre documenti dicono il contrario del terzo sullo stesso risultato.

### D3 — La modalità monotona è raccomandata ma mai usata

`docs/study2.md:185` la indica nella ricetta di riferimento, ma la funzione che *tutti* gli
script della fase B usano per costruire il modello non la attiva:
`CDFNet(in_dim=1, hidden_sizes=(width,), activation="sigmoid")`, con `monotone` al suo
default `False` `[codice study2_common.py:113]`. L'unico uso di `monotone=True` è
nell'ablazione `mlp2_monotonicity.py:73`.

### D4 — «la pdf è la derivata della rete» è falso nel percorso consegnato

Il README lo afferma `[codice README.md:8-9]`. Nel percorso consegnato la densità è
`np.clip(np.gradient(curva_rettificata, griglia), 0, None)` `[codice training.py:129]`:
differenza finita di una curva che non è l'uscita della rete. Le due cose coincidono solo
dove la rete è già monotona e la griglia è abbastanza fitta.

### D5 — Saturazione: la rete non può raggiungere 0 e 1 `[T1]` `[E2]`

Struttura, non addestramento: la pre-attivazione dell'ultimo strato è una somma finita di
termini limitati, quindi è limitata, quindi la sigmoide finale la mappa in un
**sottointervallo chiuso e stretto** di (0,1). Le code non sono raggiungibili con parametri
finiti `[T1]`.

Misurato dopo il training (trimodale, n=1000, seme 0), come massa fra i due asintoti: rete
non vincolata 0.9921, monotona 0.9898, e con l'inizializzazione sui quantili la non
vincolata scende a **0.9064** `[E2]`. Il valore algebrico previsto da `[T1]` e quello
misurato concordano: rete monotona, previsto σ(−4.579) = 0.010161 e misurato 0.010161
(sei cifre decimali); rete non vincolata, previsto σ(−5.626) = 0.003589 e misurato 0.003537
(scarto 5e-05, dovuto al fatto che mediana − 50σ̂ è un punto finito e non il limite).

**Precisazione.** Per la rete monotona i valori misurati a mediana ± 50σ̂ *sono* gli
asintoti. Per la rete non vincolata non è garantito che lo siano su tutto ℝ, perché la
funzione può non essere monotona fuori dalla regione dei dati; il limite all'infinito esiste
comunque ed è quello calcolato da `[T1]`. La grandezza operativamente rilevante è la massa
sul dominio di valutazione, riportata in `[E5]`: 0.9941 / 0.9925 / 0.9878 / 0.9885 per
A / Aq / B / Bq contro **0.999927** per la proposta.

### D6 — La «massa = 1 garantita» è un artefatto della riscalatura

`rectify_cdf` normalizza con i valori agli **estremi della griglia**
`[codice training.py:125-128]`: impone F=0 al primo nodo e F=1 all'ultimo. Ne segue
∫f = F(fine) − F(inizio) = 1 per costruzione, qualunque sia la qualità della stima. Il
difetto D5 non viene corretto: viene **redistribuito all'interno del dominio** dallo
stiramento. E la griglia, nel repo, è costruita dalla distribuzione vera
(`means ± 5·stds`) `[codice study2_common.py:21-24]`, quindi la garanzia poggia su
un'informazione che sui dati del professore non esiste (questo aspetto appartiene al
documento sulla CDF di riferimento; qui conta solo che la massa unitaria non è una
proprietà del modello).

### D7 — Il `clamp_min(0)` rompe la relazione pdf/CDF `[T5]`

`density_from_cdf(..., clamp=True)` è il default `[codice training.py:112-117]`. Azzerare
i tratti negativi di f aggiunge area: ∫clamp(f,0) = ∫f + ∫max(0,−f) ≥ F(b) − F(a), con
uguaglianza solo se f ≥ 0 quasi ovunque `[T5]`. Le due uscite consegnate (CDF e pdf)
smettono di essere l'una la primitiva dell'altra.

**Aggiornamento dopo la verifica sul codice reale (§4bis).** In prima stesura, sulla base
delle repliche NumPy, questo era classificato come *rischio latente mai osservato*. **È stato
osservato:** con il codice del repo e un'inizializzazione che produce violazioni (3 casi su 11
provati) il clamp si attiva su 32–132 nodi e gonfia la massa di +4.7e-06 … +1.03e-04 `[V6]`.

Testo originale, conservato: nelle nostre esecuzioni NumPy il clamp non si è mai attivato, perché le reti sono
uscite già monotone `[E5]`; su una curva non monotona costruita apposta l'errore appare
(0.99259 → 0.99339) `[E5]`. È quindi un **rischio latente**, non un errore osservato. Il
punto resta: una garanzia deve essere strutturale, non condizionata al fatto che il difetto
non si presenti.

### D8 — Il massimo cumulativo non è la correzione monotona ottima `[T4]` `[E6]`

`np.maximum.accumulate` `[codice training.py:125]` produce *una* curva monotona, non la più
vicina. La proiezione L2 sul cono delle funzioni non decrescenti è la regressione isotona
(PAVA). Misurato su una curva con il 2% di punti in violazione: distanza L2 dalla curva
grezza **1.70× maggiore** con il massimo cumulativo che con PAVA (0.001274 contro 0.000750)
`[E6]`. Inoltre il massimo cumulativo corregge **solo verso l'alto**, quindi è una
correzione sistematicamente sbilanciata, mentre la proiezione si muove in entrambi i versi
`[T4]`.

**Correzione a una nostra ipotesi errata.** Avevamo previsto che PAVA evitasse i plateau a
densità nulla. È **falso**: PAVA appiattisce i blocchi accorpati, quindi genera plateau
esattamente come il massimo cumulativo, e nella misura ne genera anche qualcuno in più (22
punti su 801 contro 20) `[E6]`. Un plateau nella CDF significa "probabilità nulla qui",
affermazione quasi sempre falsa. La conclusione corretta è quindi più forte, non più
debole: **nessuna correzione a valle risolve il problema**, perché entrambe producono
regioni a densità nulla. L'unica via che le evita è una densità strettamente positiva per
costruzione `[T2]`.

### D9 — La penalità soft certifica solo i punti che guarda `[E3]`

256 punti equispaziati, numero mai calibrato `[codice training.py:35, 146-148]`. Un
avvallamento più stretto del passo di collocazione dà penalità **esattamente 0** con la
funzione non monotona: misurato, con passo 0.0549 e avvallamento di larghezza 0.0069, la
penalità è 0.000e+00 mentre lo 0.134% dei punti di una griglia da 400.001 nodi è in
violazione e la derivata minima è −1.60 `[E3]`.

**Onestà:** un avvallamento così stretto non è producibile da una rete a 8 unità
sigmoidali, che è troppo liscia. Il controesempio dimostra il **buco logico** (la penalità
non è un vincolo), non un fallimento osservato. Il difetto pratico è un altro: essendo un
metodo di penalizzazione, non dà garanzie e introduce un compromesso su λ, che nella
tabella dello studio stesso costa 2–3× di ISE a λ=100 `[codice docs/study2.md:138-145]`.

**Aggiornamento dopo §4bis.** Avevamo aggiunto che «nelle nostre esecuzioni le reti sono uscite
monotone, quindi non abbiamo prova che la penalità serva». **Era un artefatto della replica
NumPy.** Con il codice del repo le violazioni compaiono in circa un quarto delle
inizializzazioni, fino al 6.15 % dei nodi `[V4]` `[V6]`. Questo **rafforza** P1: il problema
esiste, e la risposta giusta è una garanzia strutturale, non una penalità che lo insegue.

### D10 — Nessuna standardizzazione dell'ingresso: la stima non è invariante `[T6]` `[E7]`

I campioni grezzi entrano nella rete `[codice training.py:55; study2_common.py:114-116;
app/server.py:318]`. Con bias inizializzati a zero `[codice models.py:70]` **tutti i centri
delle sigmoidi partono da x=0**, indipendentemente da dove stiano i dati; il passo di Adam
è fisso in unità assolute di parametro. Conseguenza: la stessa distribuzione traslata dà un
risultato diverso.

**Perché conta più di tutto il resto:** la parte Parzen è invariante per costruzione
(h ∝ σ̂); la parte rete no. Il progetto verrà provato su campioni di cui non conosciamo né
posizione né scala. Misurato: traslando i dati di +1000, il KS della rete del repo passa da
0.028 a un valore molto peggiore, mentre la stessa architettura con standardizzazione
interna resta **numericamente identica** al caso non traslato `[E7]`.

### D11 — (fuori ambito ma bloccante) `np.trapz` rimosso

`_trapezoid = getattr(np, "trapezoid", np.trapz)` `[codice parzen.py:30; metrics.py:12]`
valuta `np.trapz` **sempre**, perché Python valuta gli argomenti prima della chiamata: su
NumPy ≥ 2.4, dove `trapz` non esiste più, l'import solleva `AttributeError`. Stesso
problema con l'uso diretto in `[codice study2_common.py:37]`. Il repo non è eseguibile su
un ambiente aggiornato. Riscontrato al primo import con NumPy 2.4.6.

---

## 3. La proposta

### P1 — Architettura: mistura convessa di CDF logistiche

Sostituire `CDFNet` con

$$F_\theta(x)=\sum_{j=1}^{J}\pi_j\,\sigma\!\big(a_j x+b_j\big),\qquad
\pi=\mathrm{softmax}(u),\qquad a_j=\mathrm{softplus}(\alpha_j)$$

Parametri liberi: α ∈ ℝ^J, b ∈ ℝ^J, u ∈ ℝ^J (3J numeri; l'attuale ne ha 3J+1).

Non è un'invenzione: è la trasformazione **Deep Sigmoidal Flow** (Huang et al., ICML 2018)
privata del logit di uscita `[lett.]`. Il DSF è dimostrato approssimatore universale; qui
serve la stessa famiglia con l'uscita lasciata in (0,1) invece che rimappata su ℝ.

Cosa si ottiene, **per qualunque valore dei parametri** e senza addestramento `[T2]`:

| requisito | come è garantito |
|---|---|
| F ∈ (0,1) | media pesata di sigmoidi, pesi in simplesso |
| F non decrescente | F′ = Σ π_j a_j σ′ ≥ 0, somma di termini non negativi |
| F(−∞) = 0, F(+∞) = 1 | **esatti**: σ→0 e σ→1, Σπ_j = 1 |
| ∫f = 1 | = F(+∞) − F(−∞) = 1, senza griglia e senza riscalatura |
| f ≥ 0 | idem sopra |
| f in forma chiusa | f(x) = Σ π_j a_j σ(z_j)(1−σ(z_j)) |
| F ∈ C^∞ | composizione di funzioni lisce |

Verifica numerica della monotonia strutturale: su 1000 parametrizzazioni casuali estreme,
l'architettura attuale non vincolata è non monotona **955 volte su 1000**; la proposta
**0 volte su 1000** `[E1b]`.

**Bonus concettuale.** La formula è lo stimatore di Parzen a nucleo logistico con i pesi
1/n sostituiti da π_j appresi, i centri x_i da −b_j/a_j appresi e la finestra unica h da
1/a_j per componente. La rete diventa letteralmente **un Parzen compresso da n a J
componenti con finestre locali**: è la tesi onesta del progetto resa architettura invece
che retorica, e risponde in anticipo a «cosa aggiunge la rete rispetto al Parzen».

### P2 — Standardizzazione interna dell'ingresso

Il modello lavora su z = (x − μ)/s con μ = mediana e s = deviazione standard campionarie,
memorizzate nel modello; l'inferenza rimappa. Con questo, tutta la pipeline diventa
equivariante per x → αx + β `[T6]`, esattamente come già lo è la parte Parzen.

### P3 — Densità in forma chiusa

`pdf(x)` restituisce Σ π_j a_j σ(z_j)(1−σ(z_j)) (diviso s se si standardizza). Niente
autograd, niente `np.gradient`, niente dipendenza dalla griglia, nessun clamp: la relazione
f = dF/dx è un'identità algebrica, non una speranza numerica.

### P4 — Rettifica: rimossa dal percorso principale

Con P1 non serve. Resta utile come **diagnostica**: se su una griglia fine `np.diff(F) < 0`
da qualche parte, c'è un bug, non un fenomeno da correggere. Se per qualche ragione si
vuole mantenere una correzione a valle (per esempio per confrontare la vecchia
architettura), usare **PAVA** e non il massimo cumulativo `[T4]` `[E6]`.

### P5 — `clamp_min(0)` rimosso; la massa si riporta, non si impone

Con P1 la densità è non negativa per costruzione. La massa sul dominio di valutazione va
**riportata come diagnostica** (`∫f` sulla griglia, che è < 1 esattamente di quanto è la
massa nelle code fuori griglia): è un'informazione onesta sulla qualità della stima, mentre
un 1.000000 ottenuto per normalizzazione non dice nulla.

### P6 — Penalità di monotonia: ritirata

Con P1 non ha oggetto. Se si mantiene la sezione di ablazione nel report (ha valore
didattico: mostra perché il vincolo strutturale è preferibile), i punti di collocazione
vanno **ricampionati a ogni epoca** invece che fissati, così la penalità stima l'integrale
su tutto l'intervallo e non un insieme finito `[D9]`.

### P7 — Lo stimatore torna a essere una funzione

Conseguenza di P1+P3, e la ragione per cui questo blocco è quello da sistemare per primo:
oggi il prodotto finale è una **tabella su griglia** (la curva rettificata) e non esiste
fuori da [griglia[0], griglia[-1]]. Con la proposta il prodotto finale è una funzione
valutabile in qualunque punto di ℝ, con derivata analitica. La griglia torna a essere
soltanto uno strumento di disegno e di misura, come dev'essere.

---

## 4. Prove numeriche

Banco: `temp_analysis/redesign_evidence.py`, libreria riusabile in
`temp_analysis/_redesign_lib.py`, output integrale in `temp_analysis/redesign_evidence.txt`.
Sigle: **A** = architettura del repo non vincolata, **B** = repo con pesi softplus,
**q** = stessa architettura ma con l'inizializzazione sui quantili dei dati (per separare
l'effetto dell'architettura da quello dell'inizializzazione), **C** = proposta,
**_rect** = con `rectify_cdf` applicata, **_raw** = uscita grezza.

### E1 — Il banco è corretto

(a) Gradienti analitici contro differenze finite centrate: errore relativo massimo
**5.19e-09** (A), **7.69e-09** (B), **2.77e-08** (C).

(b) Monotonia strutturale, 1000 parametrizzazioni casuali estreme, verificata su 5000 punti:

| architettura | parametrizzazioni non monotone |
|---|---|
| A (repo, non vincolata) | **955 / 1000** |
| B (repo, pesi softplus) | 0 / 1000 |
| C (proposta) | 0 / 1000 |

**Caveat di precisione, emerso sul codice reale `[V1]`.** I teoremi T1/T2 valgono in aritmetica
esatta; PyTorch lavora in float32. Ripetendo la prova con `CDFNet(monotone=True)` si osserva
1 caso su 1000 con un dislivello di −1.5e-08, contro un ULP di float32 attorno a 0.5 pari a
6.0e-08: il dislivello è **più piccolo di un ULP**, quindi è rumore di rappresentazione e non
una violazione del teorema. Nei test la soglia va posta a ~1e-6, non a 1e-9.

### E2 — Le code

Trimodale, n = 1000, seme 0; valore di F a mediana ± 50 σ̂ dopo il training:

| architettura | F(−∞) | 1 − F(+∞) | massa fra gli asintoti |
|---|---|---|---|
| A | 0.003537 | 0.004410 | 0.992053 |
| Aq | 0.093086 | 0.000469 | **0.906445** |
| B | 0.010161 | 0.000005 | 0.989834 |
| Bq | 0.010945 | 0.000034 | 0.989021 |
| **C** | **0.000000** | **0.000000** | **1.000000** |

Per C i due valori sono esatti per costruzione `[T2]`, non "molto piccoli".

### E3 — La penalità soft certifica solo i punti che guarda

256 punti di collocazione equispaziati su un intervallo di ampiezza 14, passo 0.0549.
Avvallamento gaussiano di ampiezza 0.02 centrato esattamente fra due punti:

| larghezza avvallamento | penalità sui 256 punti | punti in violazione su 400.001 | derivata minima |
|---|---|---|---|
| 0.01373 (= passo/4) | 8.897e-04 | 0.2320 % | −0.717 |
| **0.00686 (= passo/8)** | **0.000e+00** | **0.1340 %** | **−1.601** |
| 0.00275 (= passo/20) | 0.000e+00 | 0.0610 % | −4.252 |
| 0.00110 (= passo/50) | 0.000e+00 | 0.0270 % | −10.879 |

**Onestà.** Un avvallamento così stretto non è producibile da una rete a 8 unità sigmoidali,
che è troppo liscia: il controesempio dimostra il buco logico (la penalità non è un vincolo),
non un fallimento osservato.

**Correzione dopo §4bis.** Qui avevamo scritto che «nelle nostre esecuzioni tutte le reti sono
uscite monotone, quindi non abbiamo prova che la penalità serva». Vero sulla replica NumPy,
**falso sul codice reale**: con la ricetta del repo le violazioni compaiono in circa un quarto
delle inizializzazioni, fino al 6.15 % dei nodi `[V4]`. Il problema è reale; resta che la
penalità è il rimedio sbagliato, perché non è un vincolo e introduce un λ che a 100 costa
2–3× di ISE `[codice docs/study2.md:138-145]`.

### E4 — Accuratezza

Media su 3 semi, J = 8, ricetta PNN del repo (etichette LOO, h = 0.5 σ̂/√(n−1)),
6000 epoche Adam lr 0.03. Celle: **KS / ISE della pdf**.

**Trimodale** (il caso facile, quello su cui il repo ha calibrato tutto):

| stimatore | n=500 | n=1000 | n=2000 |
|---|---|---|---|
| PW maestro (0.5σ̂) | 0.0336 / 0.00681 | 0.0284 / 0.00473 | 0.0139 / 0.00318 |
| PW regola-σ (1.5σ̂) | 0.0341 / 0.00462 | 0.0286 / 0.00222 | 0.0129 / 0.00134 |
| A_raw | 0.0307 / 0.00309 | 0.0280 / 0.00246 | 0.0131 / 0.00116 |
| A_rect | 0.0309 / 0.00316 | 0.0268 / 0.00250 | 0.0129 / 0.00117 |
| Aq_rect | 0.0318 / 0.00336 | 0.0292 / 0.00233 | 0.0166 / 0.00131 |
| B_rect | 0.0313 / 0.00368 | 0.0290 / 0.00268 | 0.0157 / 0.00173 |
| Bq_rect | 0.0327 / 0.00336 | 0.0289 / 0.00248 | 0.0149 / 0.00136 |
| **C** | 0.0314 / 0.00339 | **0.0273 / 0.00190** | **0.0126 / 0.00098** |

**5 mode strette molto separate** (il caso "del professore"):

| stimatore | n=500 | n=1000 | n=2000 |
|---|---|---|---|
| PW maestro (0.5σ̂) | 0.0333 / 0.00543 | 0.0220 / 0.00325 | 0.0130 / 0.00146 |
| PW regola-σ (1.5σ̂) | 0.0573 / 0.04099 | 0.0382 / 0.02288 | 0.0236 / 0.01029 |
| A_raw | 0.1769 / 0.14367 | 0.0450 / 0.02412 | 0.1237 / 0.06611 |
| A_rect | 0.1794 / 0.08391 | 0.0444 / 0.02314 | 0.0323 / 0.01183 |
| Aq_rect | 0.0342 / 0.00700 | 0.0236 / 0.00543 | 0.0241 / 0.00902 |
| B_rect | 0.1015 / 0.04084 | 0.0988 / 0.03803 | 0.1024 / 0.03989 |
| Bq_rect | 0.0344 / 0.00846 | 0.0280 / 0.00544 | 0.0246 / 0.00347 |
| **C** | **0.0338 / 0.00473** | **0.0231 / 0.00264** | **0.0139 / 0.00113** |

**6 mode a scale miste**:

| stimatore | n=500 | n=1000 | n=2000 |
|---|---|---|---|
| PW maestro (0.5σ̂) | 0.0336 / 0.00340 | 0.0238 / 0.00279 | 0.0100 / 0.00138 |
| PW regola-σ (1.5σ̂) | 0.0488 / 0.01196 | 0.0308 / 0.00648 | 0.0141 / 0.00281 |
| A_rect | 0.0732 / 0.02088 | 0.0625 / 0.01896 | 0.0487 / 0.01942 |
| Aq_rect | 0.0431 / 0.00905 | 0.0332 / 0.00575 | 0.0185 / 0.00585 |
| B_rect | 0.0656 / 0.02502 | 0.0602 / 0.02293 | 0.0428 / 0.01783 |
| Bq_rect | 0.0430 / 0.00981 | 0.0369 / 0.00809 | 0.0245 / 0.00825 |
| **C** | **0.0351 / 0.00241** | **0.0247 / 0.00158** | **0.0083 / 0.00054** |

**Come vanno lette queste tabelle — onestamente.**

1. **Gran parte del disastro di A non è l'architettura, è l'inizializzazione.** Sul caso a
   5 mode, n=500, A_rect fa 0.1794 di KS e Aq_rect 0.0342: un fattore 5, ottenuto solo
   spostando i centri delle sigmoidi sui quantili dei dati. È il difetto `[D10]` in azione
   (bias a zero ⇒ tutti i centri in x=0, mentre i dati stanno su [−9, 9]). Chiunque
   confrontasse C con A senza questo controllo starebbe barando.
2. **A parità di inizializzazione, l'architettura conta ancora, e in una direzione sola.**
   Sull'ISE della pdf, C batte il migliore fra Aq e Bq in 8 casi su 9: ×1.5 (5 mode, n=500),
   ×2.1 (5 mode, n=1000), ×3.1 (5 mode, n=2000), ×3.8 / ×3.6 / ×10.8 (6 mode), ×1.2 / ×1.3
   sulla trimodale a n=1000 e 2000; l'unico pareggio è la trimodale a n=500 (0.00339 contro
   0.00336, differenza non significativa su 3 semi).
3. **Sul caso facile non cambia quasi nulla.** Sulla trimodale tutte le varianti stanno in
   un fazzoletto. È coerente con `[D10]`: la trimodale è centrata vicino a zero e ha una sola
   scala, quindi l'inizializzazione sbagliata non fa danno e l'architettura non è sotto
   sforzo. **Il repo ha calibrato tutto su questo caso**, ed è per questo che i difetti non
   sono emersi.
4. **La rettifica guadagna valore quando la rete fallisce.** Caso a 5 mode, n=2000: A_raw
   0.1237 → A_rect 0.0323. La rettifica non sta migliorando una buona stima, sta rattoppando
   una stima rotta. Con C non c'è niente da rattoppare.
5. **C batte il maestro Parzen da cui impara** sull'ISE in tutti e 9 i casi (es. 6 mode
   n=2000: 0.00054 contro 0.00138), e batte il Parzen con la regola-σ di un fattore fra
   **1.2× e 9.1×** (1.2–1.4× sulla trimodale, 4.1–5.2× sulle 6 mode, 8.7–9.1× sulle 5 mode).
   Il meccanismo è quello della PNN di Trentin — la capacità limitata rimuove varianza — qui
   però con la validità garantita per costruzione anziché per rettifica.

### E5 — Massa e coerenza pdf/CDF

Trimodale, n = 1000, seme 0, sul dominio di valutazione:

| architettura | F(fine) − F(inizio) | ∫f (derivata vera) | ∫clamp(f,0) |
|---|---|---|---|
| A | 0.994128 | 0.994128 | 0.994128 |
| Aq | 0.992494 | 0.992494 | 0.992494 |
| B | 0.987774 | 0.987774 | 0.987774 |
| Bq | 0.988486 | 0.988486 | 0.988486 |
| **C** | **0.999927** | **0.999927** | **0.999927** |

**Sulla replica NumPy il clamp non si è mai attivato**, perché tutte le reti sono uscite
monotone. **Sul codice reale si attiva**: vedi §4bis `[V6]`. Su una curva costruita non monotona il difetto si
manifesta: ∫f = 0.992595 = F(fine) − F(inizio), mentre ∫clamp(f,0) = 0.993393, cioè
**+7.98e-04 di massa inventata** `[T5]`.

### E6 — Massimo cumulativo contro PAVA

Curva con il 2.00 % di punti in violazione:

| | distanza L2 dalla curva grezza | monotona | punti a densità nulla |
|---|---|---|---|
| massimo cumulativo | 0.001274 | sì | 20 / 801 |
| PAVA (isotona) | **0.000750** (1.70× più vicina) | sì | 22 / 801 |

PAVA è più vicina, come garantito da `[T4]`, ma **non elimina i plateau**: li produce anche
lei, e in questa misura leggermente di più. Vedi la correzione in `[D8]`.

### E7 — Invarianza per traslazione e scala

Trimodale, n = 1000, seme 0. Valori di KS; per una stima corretta devono essere tutti
uguali `[T6]`:

| trasformazione | A | Aq | C | **C + standardizzazione** |
|---|---|---|---|---|
| nessuna | 0.0283 | 0.0316 | 0.0258 | **0.0259** |
| traslazione +100 | 0.1165 | 0.0564 | 0.0784 | **0.0265** |
| scala ×50 | 0.0325 | 0.0265 | 0.0261 | **0.0259** |
| traslazione +1000 | **0.5012** | **0.5000** | **0.5000** | **0.0259** |

A +1000 tutte e tre le varianti non standardizzate collassano a KS ≈ 0.5, cioè la rete
restituisce la costante 0.5: **non ha imparato nulla**. Con la standardizzazione il KS resta
0.0259–0.0265 ovunque.

**Il residuo (0.0259 contro 0.0265) è rumore, e l'abbiamo verificato.** In aritmetica esatta
l'equivarianza è esatta `[T6]`. In virgola mobile, z = (x−μ)/s calcolato su dati traslati
differisce di ~5e-15 per cancellazione; misurando lo scarto sulla curva finale si ottiene
max|F − F_rif| = 6.7e-04 (+100), 2.0e-06 (×50), 6.1e-05 (+1000). Che sia amplificazione
caotica e non un difetto sistematico è confermato dal fatto che **due implementazioni
matematicamente identiche della stessa pipeline amplificano la stessa perturbazione in
modo diverso** (8.3e-07 contro 6.7e-04): un errore sistematico sarebbe riproducibile, il
caos no. Conseguenza pratica: riportare sempre medie su più semi, come fa `[E4]`.

### E8 — Quante componenti servono

C, n = 1000, 3 semi. Celle: KS / ISE.

| caso | J=4 | J=8 | J=16 | J=32 | J=64 |
|---|---|---|---|---|---|
| trimodale | **0.0260 / 0.00172** | 0.0273 / 0.00190 | 0.0274 / 0.00220 | 0.0276 / 0.00277 | 0.0282 / 0.00357 |
| 5 mode strette | 0.0752 / 0.04925 | **0.0231 / 0.00264** | 0.0229 / 0.00276 | 0.0223 / 0.00286 | 0.0221 / 0.00287 |
| 6 mode scale miste | 0.0561 / 0.02291 | **0.0247 / 0.00158** | 0.0245 / 0.00166 | 0.0236 / 0.00187 | 0.0233 / 0.00213 |

**Al budget operativo n = 500** (5 semi, perché è il budget indicato dal docente e merita più
precisione):

| caso | J=4 | J=6 | J=8 | J=12 | J=16 | J=24 |
|---|---|---|---|---|---|---|
| trimodale | **0.0329 / 0.00260** | 0.0322 / 0.00299 | 0.0336 / 0.00344 | 0.0347 / 0.00420 | 0.0346 / 0.00449 | 0.0345 / 0.00497 |
| 5 mode strette | 0.0995 / 0.05093 | 0.0376 / 0.00485 | **0.0380 / 0.00482** | 0.0378 / 0.00482 | 0.0378 / 0.00499 | 0.0379 / 0.00503 |
| 6 mode scale miste | 0.0624 / 0.02350 | 0.0542 / 0.01403 | **0.0389 / 0.00285** | 0.0387 / 0.00290 | 0.0387 / 0.00302 | 0.0385 / 0.00315 |

Lettura: **J deve essere almeno pari al numero di mode** — J=4 fallisce su 5 e 6 mode con un
ISE 10–20× peggiore, ed è un fallimento vistoso (KS 0.0995 contro 0.0376). Oltre quella
soglia la curva è piatta, con un lieve peggioramento al crescere di J dovuto al
sovradattamento del rumore delle etichette, **più marcato a n = 500 che a n = 1000**: sulla
trimodale l'ISE passa da 0.00260 (J=4) a 0.00497 (J=24).

**Scelta: J = 8.** Il numero di mode non è noto a priori, quindi non si può scendere sotto la
soglia; e a n = 500 il costo di esagerare non è più trascurabile, quindi non conviene salire.
J = 8 è il primo valore che soddisfa tutti e tre i casi ed è il minimo dell'ISE in due su tre.
*(Questa raccomandazione corregge la precedente "J = 8–16", che era tarata su n = 1000: a
n = 500 il ramo destro della curva è più ripido e J = 16 costa il 5–30 % di ISE in più.)*

### E9 — Serve ancora una finestra di Parzen?

C con J = 16, 3 semi. Confronto fra etichette LOO-Parzen (h = 0.5 σ̂/√(n−1)) ed etichette
ECDF, F_n(x_i) = (rango − ½)/n, che **non hanno alcun iperparametro**. Celle: KS / ISE.

| caso | n | C su etichette LOO(h) | C su ECDF |
|---|---|---|---|
| trimodale | 500 | 0.0327 / 0.00462 | 0.0331 / 0.00608 |
| trimodale | 1000 | 0.0274 / 0.00220 | 0.0273 / 0.00246 |
| trimodale | 2000 | 0.0128 / 0.00139 | 0.0128 / 0.00148 |
| 5 mode strette | 500 | 0.0331 / 0.00480 | 0.0341 / 0.00510 |
| 5 mode strette | 1000 | 0.0229 / 0.00276 | 0.0207 / 0.00275 |
| 5 mode strette | 2000 | 0.0142 / 0.00119 | 0.0137 / 0.00155 |
| 6 mode scale miste | 500 | 0.0342 / 0.00256 | 0.0355 / 0.00453 |
| 6 mode scale miste | 1000 | 0.0245 / 0.00166 | 0.0254 / 0.00225 |
| 6 mode scale miste | 2000 | 0.0092 / 0.00064 | 0.0090 / 0.00077 |

**Misura dedicata al budget operativo** (n = 500, 5 semi, J = 8, cioè le impostazioni
effettivamente scelte — la tabella sopra usava J = 16 e 3 semi):

| caso | PW maestro | C su etichette LOO(h) | C su ECDF | rapporto ISE ECDF/LOO |
|---|---|---|---|---|
| trimodale | 0.0352 / 0.00677 | **0.0336 / 0.00344** | 0.0343 / 0.00407 | 1.18× |
| 5 mode strette | 0.0376 / 0.00560 | 0.0380 / 0.00482 | **0.0370 / 0.00464** | 0.96× |
| 6 mode scale miste | 0.0385 / 0.00399 | **0.0389 / 0.00285** | 0.0396 / 0.00355 | 1.25× |

**Lettura, e correzione di una nostra sovrastima.** Con 3 semi e J = 16 avevamo misurato un
vantaggio del maestro Parzen del +31 % e +77 % a n = 500. Con 5 semi e J = 8 il vantaggio si
ridimensiona a **+18 %, −4 %, +25 %**: reale ma modesto, e su un caso la ECDF è addirittura
migliore. La stima precedente era rumore su 3 semi in una configurazione che non è quella
scelta.

**Conclusione operativa.** Il Parzen **resta**: è parte dell'esercizio per costruzione, è la
sorgente delle etichette, ed è comunque la variante migliore su 2 casi su 3. La ECDF non è
un sostituto proposto ed è mantenuta **solo come ablazione**. Il valore di questo esperimento
è un altro, ed è a nostro favore: mostra che, con un'architettura che è già una mistura di
nuclei lisci, **la regolarizzazione la fa la struttura e non la finestra**, quindi la scelta
di h non è più un punto critico sul lato rete. È un argomento di robustezza da citare nel
report, non un invito a togliere il Parzen.

Nota di lettura per il resto della sezione: C addestrata sulle etichette LOO batte il PW
maestro sull'ISE in tutti e tre i casi anche a questo budget (0.00344 contro 0.00677;
0.00482 contro 0.00560; 0.00285 contro 0.00399).

---

## 4bis. Verifica incrociata sul codice reale

Tutta la sezione 4 è stata misurata su **repliche indipendenti in NumPy**, scelta deliberata:
un errore del repo non doveva poter contaminare le misure che lo giudicano. Dopo la
correzione di B1 e la creazione dell'ambiente (B6) il repo gira, quindi ogni affermazione che
riguarda il **comportamento del codice** è stata rimisurata con la libreria del repo.

Banco: `temp_analysis/crosscheck_real_code.py` → `crosscheck_real_code.txt`, più
`clamp_on_violator.txt` e `seed_reproducibility.txt`.

| # | affermazione | replica NumPy | codice reale | esito |
|---|---|---|---|---|
| V1 | monotonia strutturale, 1000 parametrizzazioni casuali | A: 955/1000 non monotone; B: 0/1000 | A: **956/1000**; B: **1/1000** | **confermata** (l'1 è rumore float32, vedi sotto) |
| V2 | saturazione delle code dopo il training | massa fra asintoti 0.9921 / 0.9898 | **0.9955 / 0.9903** | **confermata** |
| V3 | invarianza per traslazione | KS 0.028 → 0.50 a +1000 | KS 0.031 → **0.50 già a +100**, fino a **0.70** | **confermata e peggiore** |
| V4 | violazioni di monotonia nella ricetta del repo | 0 % (mai osservate) | **fino al 6.15 %**, in circa 1 inizializzazione su 4 | **SMENTITA** |
| V6 | il clamp si attiva? | mai | **sì**: 32–132 nodi, massa gonfiata di +4.7e-06 … +1.03e-04 | **SMENTITA** |

### Cosa è cambiato, in concreto

**1. Le violazioni di monotonia esistono davvero.** Era l'affermazione più fragile della
sezione 4, e va nella direzione che rafforza la proposta: il difetto che la rettifica e la
penalità cercano di rimediare **si presenta**, in circa un quarto delle inizializzazioni e
fino al 6.15 % dei nodi della griglia. La replica NumPy non lo mostrava perché il mio
ottimizzatore e la mia inizializzazione differiscono da quelli di PyTorch. Conclusione
invariata, argomento più forte: serve una garanzia strutturale (P1), non un rimedio a valle.

**2. Il clamp è un difetto osservato, non latente.** Su una rete che viola davvero, `clamp_min(0)`
si attiva e rompe l'identità ∫f = F(b) − F(a) di +4.7e-06 … +1.03e-04 `[T5]`. Nella stessa
misura si vede anche che **la rettifica peggiora il KS** in tutti e tre i casi provati
(0.0255→0.0267, 0.0279→0.0310, 0.0254→0.0289): non è la correzione gratuita che il repo
descrive.

**3. La fragilità alla traslazione è peggiore del previsto.** La replica NumPy collassava a
+1000; il codice reale collassa **già a +100**, con KS 0.50 su due semi su tre e 0.70 sul
terzo — cioè peggio della costante 0.5. La decisione D-09 (standardizzazione interna) ne esce
rafforzata.

**4. Il teorema regge, ma in float32 va misurato con la tolleranza giusta.** L'unico caso
non monotono su 1000 con `monotone=True` ha un dislivello di −1.5e-08 contro un ULP di float32
attorno a 0.5 pari a 6.0e-08: è **più piccolo di un ULP**, quindi rumore di rappresentazione.
Nei test la soglia va posta a ~1e-6.

### Il difetto che ha reso la verifica difficile

Durante V4 le violazioni non si riproducevano fra un'esecuzione e l'altra sugli stessi semi.
La causa è un difetto del repo, ora accertato (**B8** in `qa_analisi.md`):

> In `fit_cdf_net` `[codice study2_common.py:113]` il modello è costruito **prima** che
> `train_cdf` chiami `set_seed(config.seed)` `[codice training.py:145]`. L'inizializzazione
> dipende quindi dallo stato globale dell'RNG di PyTorch al momento della chiamata, non dal
> seme. Verificato: semi **diversi** con lo stesso stato globale danno risultati **identici**;
> lo **stesso** seme con stato globale diverso dà risultati **diversi**.

Conseguenza per la sezione 4: i valori assoluti misurati sul codice reale non sono
riproducibili fra esecuzioni, né i nostri né quelli di `docs/study2.md`. Le medie su più semi
del repo restano medie su inizializzazioni diverse — perché lo stato avanza a ogni iterazione
— ma le etichette «seme *k*» sono fittizie e nessun risultato è riproducibile. Va corretto
insieme al resto (spostare la costruzione del modello dopo `set_seed`, o passare un
`torch.Generator` esplicito).

---

## 5. Dimostrazioni

Notazione: σ(t) = 1/(1+e^{−t}), σ′(t) = σ(t)(1−σ(t)) > 0. Tutti i parametri sono finiti.

### T1 — Saturazione dell'architettura attuale

**Enunciato.** Sia F(x) = σ(u(x)) con u(x) = Σ_j w_j σ(a_j x + b_j) + c e attivazione
nascosta **limitata** (sigmoid o tanh). Allora per ogni x

  σ(m) ≤ F(x) ≤ σ(M),  con m = c + Σ_j min(0, w_j),  M = c + Σ_j max(0, w_j),

e σ(m) > 0, σ(M) < 1 strettamente. Inoltre i limiti esistono e valgono

  F(+∞) = σ( c + Σ_{a_j>0} w_j + Σ_{a_j=0} w_j σ(b_j) ),
  F(−∞) = σ( c + Σ_{a_j<0} w_j + Σ_{a_j=0} w_j σ(b_j) ).

**Dimostrazione.** σ(a_j x + b_j) ∈ (0,1), quindi w_j σ(·) è compreso fra min(0,w_j) e
max(0,w_j); sommando su j e aggiungendo c si ottiene m ≤ u(x) ≤ M. σ è crescente, da cui la
prima parte; σ(m) > 0 e σ(M) < 1 perché m, M sono finiti. Per i limiti: se a_j > 0 allora
σ(a_j x + b_j) → 1 per x → +∞ e → 0 per x → −∞; se a_j < 0 il contrario; se a_j = 0 resta
σ(b_j). ∎

**Corollario.** 0 e 1 non sono né raggiunti né avvicinati con parametri finiti: per
F(+∞) → 1 serve M → +∞, cioè norma dei pesi divergente. La rete sottostima
sistematicamente la massa nelle code.

**Ipotesi che serve davvero.** La limitatezza dell'attivazione nascosta. Con `softplus`
(illimitata superiormente, presente nel registro `[codice models.py:32]`) la coda destra può
essere raggiunta, ma non la sinistra: softplus → 0 per argomento → −∞, quindi u → c e
F(−∞) = σ(c) > 0. La ricetta congelata usa `sigmoid` `[codice study2_common.py:113]`, quindi
il caso limitato è quello che conta. Verifica numerica in `[E2]`: valore algebrico e valore
misurato coincidono fino alla sesta cifra decimale.

### T2 — L'architettura proposta è una CDF, per costruzione

**Enunciato.** Sia F(x) = Σ_j π_j σ(a_j x + b_j) con π_j > 0, Σ_j π_j = 1, a_j > 0. Allora
F è la funzione di ripartizione di una distribuzione assolutamente continua su ℝ:
(i) F ∈ C^∞; (ii) F′(x) > 0 per ogni x, quindi F è strettamente crescente;
(iii) F(−∞) = 0 e F(+∞) = 1; (iv) ∫_ℝ F′ = 1.

**Dimostrazione.** (i) composizione e combinazione lineare finita di funzioni C^∞.
(ii) F′(x) = Σ_j π_j a_j σ′(a_j x + b_j), somma di termini strettamente positivi.
(iii) per x → −∞ ogni σ(a_j x + b_j) → 0 (perché a_j > 0), quindi F → 0; per x → +∞ ogni
σ → 1, quindi F → Σ_j π_j = 1. (iv) ∫F′ = F(+∞) − F(−∞) = 1 per il teorema fondamentale. ∎

**Nota sulla parametrizzazione.** π = softmax(u) garantisce π_j > 0 e Σπ_j = 1 per ogni
u ∈ ℝ^J; a = softplus(α) garantisce a_j > 0 per ogni α ∈ ℝ^J. Le ipotesi del teorema sono
quindi soddisfatte **su tutto lo spazio dei parametri**, non su una regione ammissibile da
sorvegliare: non esistono parametri "illegali", e nessun passo di ottimizzazione può uscire
dall'insieme delle CDF valide. Verifica numerica: 0 violazioni su 1000 parametrizzazioni
casuali estreme, contro 955/1000 dell'architettura attuale non vincolata `[E1b]`.

**Limite da dichiarare.** F′ > 0 ovunque significa densità strettamente positiva su tutto
ℝ: la famiglia non può rappresentare esattamente una distribuzione a supporto compatto (può
solo approssimarla). È lo stesso limite dello stimatore di Parzen a nucleo logistico o
gaussiano, quindi non è una regressione rispetto al maestro.

### T3 — Nessuna perdita di espressività rispetto al maestro

**(a) Contenimento esatto.** Lo stimatore di Parzen a nucleo logistico
F̂_h(x) = (1/n) Σ_i σ((x − x_i)/h) è **un membro della famiglia**, con J = n, π_j = 1/n,
a_j = 1/h, b_j = −x_i/h. Quindi l'errore di approssimazione del maestro è esattamente zero
a capacità piena: tutto ciò che si perde con J < n è deliberato (è la compressione), non
strutturale.

**(b) Densità.** Sia G una CDF continua e J ≥ 1. Poniamo μ_j = G^{−1}((j−½)/J), π_j = 1/J,
a_j = 1/s. Sia H la CDF a gradini con atomi 1/J nei μ_j. Per x ∈ [μ_j, μ_{j+1}) si ha
H(x) = j/J e G(x) ∈ [(j−½)/J, (j+½)/J], quindi sup_x |H − G| ≤ 1/(2J). Inoltre, per s → 0,
F_θ → H puntualmente e lo scarto massimo (raggiunto negli atomi, dove F_θ vale
(j−1)/J + 1/(2J) contro j/J) tende a 1/(2J). Per la disuguaglianza triangolare

  limsup_{s→0} sup_x |F_θ(x) − G(x)| ≤ 1/J.

Dunque per ogni ε > 0 basta J ≥ 1/ε: la famiglia è densa in sup-norma nelle CDF continue. ∎

**(c) Letteratura.** La stessa famiglia, con un logit applicato all'uscita, è la
trasformazione Deep Sigmoidal Flow, dimostrata approssimatore universale `[lett.]`.

### T4 — Il massimo cumulativo non è la correzione monotona ottima

**Enunciato.** Dati y ∈ ℝ^N, il problema min { Σ_i (m_i − y_i)² : m_1 ≤ … ≤ m_N } ha
soluzione unica m\*, calcolata esattamente da PAVA in O(N). La successione del massimo
cumulativo r_i = max_{k ≤ i} y_k è ammissibile ma in generale diversa da m\*, e soddisfa
r ≥ y puntualmente.

**Dimostrazione.** L'insieme ammissibile è un cono convesso chiuso; l'obiettivo è
strettamente convesso: minimo unico, che è la proiezione euclidea di y sul cono. Che PAVA la
calcoli è il risultato classico sulla regressione isotona. r è ammissibile (non decrescente
per costruzione) e r_i ≥ y_i per definizione di massimo. Essendo m\* la proiezione,
‖r − y‖ ≥ ‖m\* − y‖, con uguaglianza solo se r = m\*. ∎

**Conseguenza qualitativa.** r corregge **solo verso l'alto**: è una correzione
sistematicamente sbilanciata, mentre la proiezione può muoversi in entrambi i versi. Misura
in `[E6]`.

### T5 — Il clamp aggiunge massa

**Enunciato.** Sia f integrabile su [a,b] con F(t) = F(a) + ∫_a^t f. Allora
∫_a^b max(f,0) = (F(b) − F(a)) + ∫_a^b max(−f, 0) ≥ F(b) − F(a), con uguaglianza se e solo
se f ≥ 0 quasi ovunque.

**Dimostrazione.** f = f⁺ − f⁻ con f⁺ = max(f,0), f⁻ = max(−f,0) ≥ 0. Quindi
∫f⁺ = ∫f + ∫f⁻ = (F(b) − F(a)) + ∫f⁻. Il termine ∫f⁻ è ≥ 0 ed è nullo se e solo se f⁻ = 0
q.o. ∎

**Lettura.** La densità consegnata cessa di essere la derivata della CDF consegnata:
integrando la pdf non si ritrova la CDF. Con l'architettura proposta f⁻ ≡ 0 per T2, quindi
il clamp è identicamente inerte e va tolto perché non serve, non perché faccia danno.

### T6 — Equivarianza per traslazione e scala

Sia S la mappa (x_1,…,x_n) ↦ F̂. Chiediamo: per α > 0, β ∈ ℝ,
S(αx + β)(αt + β) = S(x)(t) per ogni t.

**Parte Parzen: equivariante.** Con h = c·σ̂/√n si ha σ̂(αx+β) = α σ̂(x), quindi h ↦ αh, e
((αt+β) − (αx_i+β))/(αh) = (t − x_i)/h: gli argomenti dei nuclei sono invarianti, dunque
F̂ e le etichette LOO lo sono. ∎

**Parte rete, così com'è: non equivariante.** L'inizializzazione non dipende affatto dai
dati (Xavier su a, bias a zero `[codice models.py:70,76-78]`): il modello parte dalla stessa
funzione qualunque sia la posizione dei dati, mentre il bersaglio si sposta di β. Inoltre il
passo di Adam è fisso in unità assolute di parametro, mentre la scala dei parametri
necessari cambia con α. ∎

**Parte rete, con standardizzazione: equivariante.** Se μ(·) e s(·) sono statistiche
equivarianti (mediana e deviazione standard: μ(αx+β) = αμ(x)+β, s(αx+β) = α s(x)), allora
z = (x − μ)/s è **invariante**, quindi lo sono l'inizializzazione, ogni gradiente, ogni
passo dell'ottimizzatore e i parametri finali; e F̂(αt+β) = F̂_z((αt+β − μ')/s') =
F̂_z((t − μ)/s) = F̂(t). ∎

Misura in `[E7]`: con la standardizzazione il risultato è stabile su traslazioni fino a
+1000 e scale fino a ×50, mentre senza si degrada di un fattore 4.


---

## 6. Piano di implementazione

### 6.1 Nuovo modello — `src/parzen_cdf/models.py`

Aggiungere `MixtureCDFNet`; **conservare** `CDFNet` invariata, perché serve come termine di
paragone nella sezione di ablazione del report (non si può dimostrare che la nuova
architettura è migliore se si cancella la vecchia).

```python
class MixtureCDFNet(nn.Module):
    """F(x) = sum_j softmax(u)_j * sigmoid(a_j * z + b_j),  z = (x - mu) / sd,
    a_j = softplus(alpha_j).  CDF valida per ogni valore dei parametri (vedi T2)."""

    def __init__(self, n_components: int = 8):    # J = 8, calibrato a n = 500 (E8)
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_components))
        self.b     = nn.Parameter(torch.zeros(n_components))
        self.u     = nn.Parameter(torch.zeros(n_components))
        self.register_buffer("mu", torch.zeros(()))   # stat. di standardizzazione: buffer,
        self.register_buffer("sd", torch.ones(()))    # non parametri (salvate nel checkpoint)

    @torch.no_grad()
    def init_from_samples(self, x):                   # P2 + inizializzazione sui quantili
        J = self.alpha.numel()
        self.mu.fill_(x.median()); self.sd.fill_(x.std(unbiased=True))
        z = (x - self.mu) / self.sd
        q = torch.quantile(z, (torch.arange(J) + 0.5) / J)
        a = torch.full((J,), 1.0 / ((z.max() - z.min()) / J))
        self.alpha.copy_(torch.log(torch.expm1(a))); self.b.copy_(-a * q); self.u.zero_()

    def _parts(self, x):
        a = F.softplus(self.alpha)
        z = (x - self.mu) / self.sd
        return torch.sigmoid(z.unsqueeze(-1) * a + self.b), torch.softmax(self.u, 0), a

    def forward(self, x):                              # CDF
        s, pi, _ = self._parts(x); return s @ pi

    def pdf(self, x):                                  # densita' in forma chiusa (P3)
        s, pi, a = self._parts(x); return ((s * (1 - s) * a) @ pi) / self.sd
```

Note di implementazione: `mu`/`sd` come **buffer** e non parametri, così finiscono nel
`state_dict` e un modello ricaricato resta valido; i quantili vanno calcolati su z, non su x.

### 6.2 Training — `src/parzen_cdf/training.py`

| intervento | motivo |
|---|---|
| `density_from_cdf(..., clamp=True)` → `clamp=False` come default, e deprecare la funzione per il nuovo modello (che ha `pdf` esatta) | `[D7]` `[T5]` |
| spostare `rectify_cdf` in un modulo `diagnostics.py`, fuori dal percorso principale | `[D8]` `[P4]` |
| aggiungere `pava(y)` accanto a `rectify_cdf`, con docstring che dica quale delle due è la proiezione L2 | `[T4]` |
| `monotonicity_penalty`: se si mantiene, ricampionare i punti di collocazione a ogni epoca | `[D9]` `[P6]` |
| aggiungere `mass_on_grid(model, grid)` come diagnostica riportata, non imposta | `[P5]` |

### 6.3 Script e app

- `scripts/study2_common.fit_cdf_net`: costruire `MixtureCDFNet(J)` e chiamare
  `init_from_samples`; `eval_cdf_net`: usare `model.pdf(grid)` e **non** `rectify_cdf`.
- `app/server.py`: `_build_model` con la nuova classe; `_eval_snapshot` senza rettifica
  (mantenerla come interruttore per il confronto didattico, che è il senso dell'app); la
  voce "violazioni" resta come diagnostica che deve valere 0.
- **Bug bloccante `[D11]`**: `getattr(np, "trapezoid", np.trapz)` →
  `np.trapezoid if hasattr(np, "trapezoid") else np.trapz` in `parzen.py:30` e
  `metrics.py:12`; `np.trapz` diretto in `study2_common.py:37`. Senza questo il repo non
  parte su NumPy >= 2.4.

### 6.4 Test da aggiungere — `tests/test_models.py`

I test attuali verificano che l'uscita stia in (0,1) e che la modalità monotona sia monotona.
Per la nuova architettura vanno verificate le proprietà di `[T2]`, che sono più forti e
tutte controllabili:

1. monotonia su 1000 parametrizzazioni casuali estreme (replica `[E1b]`);
2. `F(-1e6) == 0.0` e `F(+1e6) == 1.0` esattamente;
3. `model.pdf(x)` contro la differenza finita di `model(x)`: coincidenza entro 1e-6. È il
   test che oggi manca e che avrebbe intercettato `[D4]`;
4. `pdf >= 0` ovunque, senza clamp;
5. **equivarianza**: addestrare su `x` e su `alpha*x + beta`, verificare che le curve
   coincidano entro 1e-3 (replica `[E7]`; la tolleranza non può essere più stretta, per il
   motivo spiegato in `[E7]`);
6. massa: `trapezoid(pdf, griglia_larga)` ~ 1 entro 1e-3.

### 6.5 Caso multivariato — non più in programma

Lo Step 2 multivariato è stato tolto dagli obiettivi del progetto, quindi non c'è nulla da
implementare. Si registra solo, per memoria, che la costruzione proposta lo risolverebbe
gratis: F(x_1,…,x_N) = somma_j pi_j prodotto_d sigma(a_jd x_d + b_jd) è N-crescente per
costruzione, perché ogni prodotto è la CDF di una distribuzione a coordinate indipendenti e
una combinazione convessa di CDF è una CDF. Mai verificato numericamente; fuori programma.

---

## 7. Cosa questo documento NON afferma

Elenco esplicito, perché un documento che rivendica troppo è attaccabile quanto uno sbagliato.

1. **Non affermiamo che la proposta batta il Parzen in generale.** Sul KS della CDF è
   sostanzialmente alla pari con il maestro `[E4]`. Il vantaggio misurato è sull'ISE della
   densità ed è concentrato sui bersagli multimodali; sulla trimodale a n=500 è un pareggio.
2. **Non affermiamo che i numeri del repo siano sbagliati.** Non abbiamo riprodotto la loro
   pipeline in PyTorch: A e B sono reimplementazioni fedeli della *funzione*, non
   dell'implementazione (Adam scritto da noi, inizializzazione con un altro RNG). I valori
   assoluti non sono confrontabili con quelli di `docs/study2.md`; i confronti **interni**
   alle nostre tabelle sì, perché tutti i bracci condividono lo stesso codice.
3. **Non affermiamo che la penalità soft causi fallimenti osservati.** Nelle nostre
   esecuzioni non è mai servita, perché le reti sono uscite monotone da sole `[E5]`. La
   critica è logica (`[E3]`: non è un vincolo) e di costo (l'iperparametro lambda), non
   empirica.
4. **Non affermiamo che PAVA elimini i plateau a densità nulla.** Lo avevamo previsto ed è
   **falso** `[E6]`; vedi la correzione in `[D8]`.
5. **Non affermiamo che la proposta sia originale.** È il Deep Sigmoidal Flow senza il logit
   di uscita `[lett.]`. Il contributo è la scelta e la giustificazione, non l'invenzione.
6. **Significatività statistica limitata.** Tre semi per cella: differenze sotto il ~10% non
   vanno interpretate. I confronti su cui poggiamo sono fattori 1.5-20x.
7. **Copertura limitata.** Solo 1-D, solo misture di gaussiane come bersaglio, nessuna coda
   pesante e nessun supporto compatto. Su supporto compatto la proposta ha un limite noto e
   dichiarato: densità strettamente positiva su tutto R `[T2]`, quindi non può rappresentare
   esattamente una uniforme. È un limite condiviso con qualunque KDE a nucleo gaussiano o
   logistico, quindi non è una regressione rispetto al maestro.
8. **Fuori ambito.** La scelta di h, la numerosità campionaria, il campionamento e l'uso
   della CDF di riferimento non sono trattati qui, se non dove `[D6]` ed `[E9]` li toccano
   di striscio.

---

## 8. Formulazioni corrette da usare nel report

Frasi pronte, che sostituiscono quelle attualmente false o imprecise.

**Sulla monotonia strutturale** (sostituisce ogni occorrenza di "Sill" riferita ai pesi
positivi):

> La monotonia per costruzione tramite pesi non negativi e attivazioni non decrescenti
> risale ad Archer & Wang (1993). Sill (1997) mostra che questa famiglia è poco espressiva e
> propone in alternativa le reti min-max, che sono approssimatori universali delle funzioni
> monotone; la nostra implementazione non usa le reti min-max.

**Sulla densità** (sostituisce «the pdf is recovered as the derivative of the trained
network» ovunque si descriva il percorso con rettifica):

> La densità è ottenuta in forma chiusa derivando analiticamente la mistura:
> f(x) = somma_j pi_j a_j sigma(z_j)(1 - sigma(z_j)). Non è una differenza finita su griglia
> né una derivata per autodifferenziazione, e l'identità f = dF/dx vale esattamente, non
> numericamente.

**Sulle garanzie:**

> L'architettura garantisce, per ogni valore dei parametri e senza alcuna correzione a valle:
> F in (0,1), F non decrescente, F(-inf) = 0, F(+inf) = 1, f >= 0 e integrale di f pari a 1.
> Sono proprietà della parametrizzazione (Teorema T2), non esiti dell'addestramento né
> effetti di una post-elaborazione su griglia.

**Sulla relazione con il Parzen:**

> L'architettura coincide con lo stimatore di Parzen a nucleo logistico in cui i pesi 1/n
> sono sostituiti da pesi appresi, i centri x_i da centri appresi e la finestra globale h da
> una finestra per componente. Con J = n essa contiene esattamente lo stimatore di Parzen
> (Teorema T3a); con J << n realizza una compressione con finestre locali.

**Sull'invarianza:**

> Lo stimatore è equivariante per trasformazioni affini dei dati: standardizzando l'ingresso
> con statistiche equivarianti, la stima prodotta su x e quella prodotta su alpha*x + beta
> coincidono (Teorema T6). Senza standardizzazione la stima dipende dalla posizione assoluta
> dei dati e degenera per traslazioni dell'ordine di 10^3 deviazioni standard.

---

## 9. Riferimenti verificati

- **Archer, N. P., Wang, S. (1993).** *Learning bias in neural networks and an approach to
  controlling its effect in monotonic classification.* IEEE TPAMI 15(9), 962-966. — Vincolo
  di positività dei pesi per la monotonia. È la costruzione che il repo attribuisce a Sill.
- **Sill, J. (1997).** *Monotonic Networks.* NIPS (atti pubblicati nel 1998). — Reti min-max:
  pesi positivi al primo strato, `max` su gruppi di iperpiani, `min` dei massimi;
  approssimatori universali delle funzioni monotone. Non è ciò che il repo implementa.
  <https://papers.neurips.cc/paper/1358-monotonic-networks.pdf>
- **Huang, C.-W., Krueger, D., Lacoste, A., Courville, A. (2018).** *Neural Autoregressive
  Flows.* ICML. — Deep Sigmoidal Flow: logit inverso di somma_j softmax(w)_j sigma(a_j x +
  b_j) con a_j > 0. La nostra architettura è la stessa trasformazione senza il logit di
  uscita. <https://arxiv.org/pdf/1804.00779>
- **Trentin, E. (2018).** Parzen Neural Network: etichette leave-one-out, finestra
  deliberatamente stretta, rete a capacità limitata. — È la ricetta di addestramento che
  conserviamo: cambiamo l'architettura, non la ricetta.

---

## 10. Ordine di esecuzione consigliato

1. `[D11]` bug `np.trapz`: senza questo non gira nulla su un ambiente aggiornato.
2. `MixtureCDFNet` + `init_from_samples` + `pdf` (§6.1) e i test (§6.4). Da soli chiudono
   `[D4]` `[D5]` `[D6]` `[D7]` `[D8]` `[D9]` `[D10]`.
3. Aggiornare `study2_common` e rilanciare la battery: attesi i numeri della colonna **C**
   di `[E4]`, entro il rumore dovuto alla diversa implementazione di Adam (§7.2).
4. App: nuova classe, rettifica declassata a interruttore didattico.
5. Correzioni testuali `[D1]` `[D2]` `[D3]` in README, `docs/`, `report/`, usando §8.
6. Solo dopo, e solo come ablazione per il report: la variante ECDF `[E9]`. Il Parzen resta
   il maestro di riferimento. Il caso multivariato è fuori programma (§6.5).
