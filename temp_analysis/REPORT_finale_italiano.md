# Ha senso addestrare un MLP su una Parzen? — Report dettagliato

**Domanda di partenza.** La pipeline del progetto addestra un MLP sigmoidale a regredire la CDF
stimata con finestre di Parzen logistiche (label = `Parzen(xᵢ)` solo nei punti dato), e ricava la pdf
per derivazione. In 1D abbiamo gia' visto che la rete *eguaglia* il suo target Parzen ma non lo batte,
e che Parzen (e perfino la CDF empirica) stimano bene una CDF 1D. Quindi: **questo approccio ha un
vantaggio reale, o no?** Questa indagine lo mette alla prova con 4 test manuali + 11 linee di indagine
parallele (un workflow), ognuna con verifica avversariale, piu' ricerca bibliografica.

**Risposta in una riga.** Si', ha senso, ma **non come stimatore piu' accurato** (li' non ha vantaggi):
ha senso come **surrogato compatto, valido per costruzione, differenziabile e a costo di interrogazione
costante di una KDE che gia' ti fidi** (distillazione/serving), e come **mattone per il caso
multivariato** (route CDF -> derivate / copula). Se invece puoi addestrare un modello a verosimiglianza
diretta (GMM/flow) avendo i dati, quello e' piu' accurato e mantiene gli stessi vantaggi strutturali.

---

## 1. Metodo dell'indagine

Tutti i test rispettano il vincolo del progetto (la rete si addestra solo sui punti dato, label =
Parzen(xᵢ)). Ogni esperimento e' uno script in `temp_analysis/` (prefissi `test_*` per i 4 manuali,
`wf0x_*` per le 11 angolazioni del workflow), con numeri reali misurati e, dove utile, una figura.
Ogni claim del workflow e' stato **verificato in modo avversariale** da un secondo agente scettico
(che ha ri-eseguito o ri-derivato il risultato). La sintesi finale verificata e' in `99_verdict.md`.

Metrica principale: **scarto CDF (KS)** rispetto alla verita' nota; per le densita' anche MSE, massa
(deve fare 1), violazioni di monotonia.

---

## 2. Dove NON c'e' vantaggio (accuratezza)

Questi sono i risultati onesti, tutti confermati dalla verifica avversariale.

### 2.1 La rete non batte la migliore Parzen (denoising) — `wf02`, `test_beats_best_parzen`
Sweep della window size dal nitido al liscio, su 3 distribuzioni, 3-4 seed, rete ben convergente
(controllo di convergenza: a 800 epoche la rete era sotto-addestrata, KS-vs-target 0.107; a 2500
epoche 0.008, regressore fedele). La **migliore** rete (su tutte le window) resta sempre **peggio**
della migliore Parzen (oracolo di banda):
- KS Parzen 0.0142 / 0.0148 / 0.0177 vs KS rete 0.0183 / 0.0196 / 0.0622 (Gauss / bimodale / trimodale).
- Manuale (bimodale simmetrica): miglior Parzen 0.0120 vs miglior rete 0.0132 (−10%).

Conclusione: **nessun vantaggio di accuratezza in 1D.** Il "la rete vince" del primo run era un
artefatto di sotto-addestramento.

### 2.2 Nessuna maggiore efficienza in campioni a n piccolo — `wf06`
n in {25,50,100,200}, >=5 seed: pari sulla Gaussiana (±1% KS), **strettamente peggio** sulla bimodale
(KS da −1.4% a −7.2%, MSE +20-37%). Anzi, la rozza CDF empirica a gradini batte Parzen sulla bimodale
(Silverman sovra-liscia). Nessun vantaggio.

### 2.3 Code peggiori (lieve passivo) — `wf05`
Parzen vince sull'accuratezza della CDF nelle code: ~2x (coda moderata) fino a 5-20x (coda estrema), e
raggiunge 0/1 a precisione macchina; la rete si ferma a un offset residuo (~2e-4..1e-3) e ha una **coda
di pdf spuriamente grassa** (5.5e-6 a x=16 contro la vera 1e-56). Per quantili estremi e' un passivo.

### 2.4 Non puo' battere una KDE, per costruzione — `wf11` (scettico)
Regredire label KDE fisse rende la KDE il **tetto** di accuratezza. Un **GMM addestrato a
verosimiglianza diretta** batte la rete del **47-81% sullo scarto CDF e 6-45x sull'MSE della pdf**,
mantenendo tutti i vantaggi strutturali. La rete non puo' battere il maestro che imita; un obiettivo a
verosimiglianza si'. Questo e' il limite concettuale piu' importante da dichiarare.

---

## 3. Dove il vantaggio e' GENUINO (strutturale), ordinato

### 3.1 Validita' gratuita via la route CDF (moderato) — `wf04` — il vantaggio piu' netto
La vittoria decisiva e senza tuning. Confronto a parita' di tutto (route CDF vs rete che produce la
pdf direttamente), su 3 distribuzioni:
- **Route CDF**: dopo `rectify_cdf`, massa = **1.00000 esatta** ovunque, **zero negativita'**;
  pdf-MSE medio **0.00242**.
- **Route pdf diretta + penalita' di normalizzazione**: pdf-MSE **0.01754** (~7x peggio), alta varianza.
- **Route pdf diretta + rinormalizzazione a posteriori**: pdf-MSE 0.00376 (~1.6x peggio), e **massa
  nativa = 1.28** prima della pezza (densita' non valida senza "barare").
- La massa nativa della route-pdf **deriva sempre piu' da 1** con la difficolta': 1.11 (Gauss) ->
  1.33 (bimodale) -> 1.40 (trimodale).

Conclusione: imporre i vincoli di una **CDF** (monotona, in [0,1]) e' molto piu' facile e *valido* che
imporre quelli di una **pdf** (≥0, integra a 1) — esattamente la tesi di Magdon-Ismail & Atiya (2002).
**Si allarga in alta dimensione**, dove il normalizzatore della pdf e' un integrale d-dimensionale
intrattabile.

### 3.2 Ammortamento + compressione (moderato) — `wf01`, `test_query_cost`
Costo di interrogazione della rete **costante in n** (0.097 ms, width-32) vs Parzen O(n) (8.6 ms a 1k
-> 1173 ms a 100k): **~12.000x** a n=100k *contro KDE naive*, e **~1031x** di compressione di memoria
(97 float vs n). I parametri crescono solo **+32 per dimensione** (D=1->97, D=20->705).
- **Limiti onesti**: e' un vantaggio di *serving*, non di accuratezza; non e' stato confrontato con le
  KDE veloci (KD-tree, FFT/binned-KDE) che esistono proprio per uccidere il costo O(n) — quindi la
  magnitudine 12.000x e' gonfiata; per una singola interrogazione one-shot Parzen e' piu' economica
  (zero training). La curva della maledizione e' stata testata solo fino a D=3 (Gaussiana separabile):
  MISE relativa migliore 0.0004 (D=1) -> 0.0041 (D=2) -> 0.0232 (D=3), ~10x peggio per dimensione.

### 3.3 Economia di modello via copula (Sklar) (moderato, confidenza media) — `wf09`
In 2D, decomporre il congiunto in **CDF marginali 1D + una copula parametrica** batte la rete
congiunta monolitica in tutti e 4 i casi:
- gauss ρ=0.6: rete congiunta sup 0.289 vs marginali+copula-gaussiana sup 0.056 (~12x meno MSE).
- pavimento copula su marginali VERE: sup **0.0003** (quasi zero) quando la famiglia di copula
  combacia con la verita' -> la stima diventa "due buoni fit 1D + un parametro interpretabile".
- **Condizionale**: se la dipendenza e' davvero non-gaussiana il pavimento e' irriducibile (sup 0.091
  / 0.027) e la rete congiunta e' poco dietro; il cattivo risultato del congiunto e' in parte
  sotto-addestramento.

---

## 4. Capacita' reali ma non uniche (capability_only)

### 4.1 Densita' congiunta differenziabile in 2D — `test_2d` (manuale)
La rete impara la CDF congiunta 2D (kernel logistico prodotto) e ne ricava la densita' come **derivata
mista ∂²F/∂x∂y** via autograd: MSE 0.00006 ≈ KDE 0.00004 (pari accuratezza), ma **query 3.8 ms vs KDE
597 ms (157x)** sulla griglia 80×80, e la CDF **empirica** congiunta non e' derivabile (la sua mista e'
una somma di spuntoni). La rete *abilita* una densita' congiunta liscia e compatta.

### 4.2 In 3D la derivata mista si rompe — `wf03`
Esteso a 3D (derivata terza ∂³F/∂x∂y∂z): la rete e' **4.3x peggio** della KDE in MSE e **non integra a
1** (Riemann 1.70 -> **errore di massa del 70%**). Causa: la **monotonia N-increasing** (validita' di
una CDF congiunta) **non e' garantita** dalla rettifica 1D. (La rivendicazione iniziale "rete 6.3x piu'
lenta" si e' ribaltata al rerun, rete ~10x piu' veloce, ma una route piu' veloce a un output rotto non
e' un vantaggio.) **Questo e' il vero nodo aperto del multivariato.**

### 4.3 Campionamento via CDF inversa — `wf08`, `test_sampling`
Generatore compatto, monotono, differenziabile: inverti `x=F⁻¹(u)` senza scorrere i campioni (97 float,
compressione 20.6x). Ma la qualita' dei campioni *pareggia* Parzen (KS verso il vero 0.033 vs 0.028) e
un GMM a 9 parametri ha gli stessi pregi. Capacita' reale, non unica.

### 4.4 Estrapolazione limitata — `wf05`
La sigmoide finale garantisce una CDF valida in [0,1] e monotona **arbitrariamente lontano** (0
non-monotonie), una proprieta' di robustezza; pagata pero' con l'offset di coda e la pdf di coda
grassa di 2.3.

---

## 5. Teoria e scettico — `wf10`, `wf11`

- **Teoria** (`wf10`): la CDF converge a ~n^(−1/2) **in ogni dimensione** (la maledizione colpisce la
  *densita'*, non la CDF). Ma derivare d volte una CDF stimata **riamplifica il rumore**, quindi la
  maledizione **rientra** sulla densita': il vantaggio "CDF facile" non sopravvive intatto alla
  derivazione. I metodi che davvero vincono la maledizione (normalizing flows) si addestrano a
  verosimiglianza diretta, non regredendo una KDE.
- **Scettico** (`wf11`): se hai i dati e liberta', addestra un modello a verosimiglianza (GMM/flow):
  domina su accuratezza e tiene ogni pregio strutturale. MLP-su-Parzen e' giustificato solo come
  teacher-student/serving di una KDE fidata.

Riferimenti: Magdon-Ismail & Atiya (2002, IEEE T-NN); "From CDF to PDF" (arXiv:1804.05316); "Neural
Likelihoods via CDFs" (arXiv:1811.00974). Dettagli in `01_research.md` e `wf10_theory.md`.

---

## 6. Verdetto: quando ha senso, quando no

**Ha senso quando valgono TUTTE:**
1. interroghi la stima **molte volte** (ammortizzi il training) o devi **spedirla/salvarla** compatta;
2. vuoi un **unico surrogato liscio, differenziabile e monotono** di una KDE/CDF-empirica **di cui gia'
   ti fidi**;
3. ti serve **validita' garantita** (massa=1, ≥0) senza penalita' di normalizzazione, specie in alta
   dimensione dove il normalizzatore e' intrattabile;
4. **non** puoi/non vuoi fittare un modello a verosimiglianza diretto.

**E' inutile quando vale ANCHE solo UNA:**
- vuoi una stima *migliore* (non puo' battere il maestro);
- e' un problema one-shot / poche query (Parzen costa meno, zero training);
- puoi fittare un GMM/flow a verosimiglianza (domina su accuratezza e tiene i pregi);
- ti servono code/quantili estremi accurati (la coda della rete e' un passivo);
- una KDE veloce (FFT/binned/KD-tree) gia' risolve il costo di query.

---

## 7. Raccomandazione per il progetto

Inquadrare onestamente il contributo come **distillazione/serving, non come stima**. La tesi dovrebbe
dire: *"un MLP sigmoidale regredisce fedelmente una CDF di Parzen in un surrogato compatto, a costo
costante, valido per costruzione e differenziabile"*, dichiarando esplicitamente che **non** batte in
accuratezza il suo maestro. Guidare con i due vantaggi strutturali robusti (route-CDF a validita'
gratuita; ammortamento/compressione), presentare l'economia della copula come risultato 2D
condizionale, e tenere le capacita' (campionamento, estrapolazione) chiaramente etichettate come non
uniche. **Non** rivendicare che batta la maledizione della dimensionalita' per le *densita'*. E
prioritizzare, per il multivariato, la **monotonia N-increasing** (il difetto del 3D): senza quella, la
densita' congiunta via derivate miste non e' valida.

Posizionamento piu' forte e onesto: **teacher-student**. E' cio' che usi quando devi servire una KDE
fidata in modo economico, valido e differenziabile su larga scala — e nulla di piu'. Se la meta' e'
accuratezza di stima, il percorso giusto e' un modello a verosimiglianza diretta.

---

*Riproducibilita': script in `temp_analysis/` (`test_query_cost.py`, `test_beats_best_parzen.py`,
`test_2d.py`, `test_sampling.py`, `wf01`–`wf11`). Verdetto verificato sintetico in `99_verdict.md`;
piano e ricerca in `00_plan_and_reflection.md`, `01_research.md`.*
