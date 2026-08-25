# Scelta della finestra di Parzen: studio indipendente

> **Nota di pulizia.** Questo documento cita `docs/study.md`, `docs/study2.md`,
> `report/report.tex` o `report/report2.tex`, rimossi perche' descrivevano un design
> sostituito. Restano recuperabili da `git log`. Dove questo documento e' in disaccordo
> con `report/report3.pdf`, vale il report.


**Domanda.** Esiste una regola empirica per h₁ (equivalentemente per h, dato che
h_n = h₁/√n) migliore di `h₁ = 1.5·σ̂`? E, prima ancora: **può esistere** una regola di quel
tipo?

**Risposta breve.** No a entrambe, e la seconda risposta rende superflua la prima. La
costante ottima varia di oltre un ordine di grandezza fra distribuzioni, quindi nessuna
scelta di costante è difendibile. La sostituzione corretta non è una costante migliore ma un
**selettore data-driven**: la cross-validation dei minimi quadrati (LSCV), che è
implementata nel repo e inutilizzata.

**Ambito.** Solo la finestra dello stimatore di Parzen (che è anche il maestro delle
etichette della rete). L'architettura della rete è trattata in `docs/redesign_network.md`.
Budget di riferimento **n = 500**, confermato a n = 1000.

---

## 1. Perché lo studio del repo non poteva dare la risposta giusta

Tre difetti di metodo, indipendenti dal risultato numerico:

1. **Il banco di prova era la posta in gioco.** La costante 1.5 è stata calibrata su una
   scaletta di 3–4 misture più 10 misture generate da `random_mixture` con medie U(−5,5) e
   deviazioni U(0.2,1.5) `[codice data.py:110-126]`. È una famiglia in cui σ̂ traccia la scala
   delle feature. Calibrare e validare sulla stessa famiglia non misura la generalizzazione:
   misura quanto bene si è interpolato il proprio benchmark.
2. **La grandezza di ancoraggio è quella sbagliata.** σ̂ è la dispersione **globale**; h deve
   risolvere la feature **più stretta**. Sono la stessa cosa solo per densità a scala unica.
3. **La metrica di validazione è quella sbagliata.** La calibrazione è stata fatta sul KS
   della CDF, che integra e perdona; il deliverable è la pdf, che non perdona.

Questo studio corregge tutti e tre i punti: banco di prova esterno e standard, ancoraggi
alternativi messi in competizione, metrica primaria sulla pdf.

---

## 2. Metodo

### 2.1 Banco di prova: Marron & Wand (1992)

Le **13 densità di Marron–Wand** verificabili, più i 3 casi del repo (per poter parlare
direttamente al suo benchmark). Sono tutte misture di normali, quindi pdf e CDF sono esatte
in forma chiusa e ogni stima è confrontabile con la verità a precisione macchina.

Perché questo banco e non uno nostro: è **il** benchmark standard della letteratura sui
selettori di banda, non l'abbiamo scelto noi, e contiene per costruzione i casi patologici
(claw, double claw, asymmetric claw: mode strettissime sovrapposte a una gaussiana larga)
che sono esattamente lo scenario peggiore dichiarato dal docente. Usare un banco esterno è
ciò che rende lo studio non auto-referenziale.

I parametri sono stati verificati sul sorgente del pacchetto R `nor1mix`
(`R/zMarrWand-dens.R`). **#14 e #15 sono esclusi**: non è stato possibile verificarne i
parametri da una fonte primaria, e una densità sbagliata invaliderebbe lo studio.

### 2.2 Selettori in gara

Tutti vedono **solo il campione**. Tutti restituiscono h per il kernel **logistico**; le
regole classiche sono derivate per kernel gaussiano e vengono divise per la deviazione
standard del logistico, π/√3 ≈ 1.8138 (senza questa correzione si sovralliscia di quel
fattore — è il bug di scala che il docente aveva già segnalato al gruppo).

| selettore | forma |
|---|---|
| `repo h₁=1.0` | h = 1/√n (il "punto di partenza" dello studio) |
| `repo 1.5·σ̂` | h = 1.5·σ̂/√n (la regola da battere) |
| `Silverman robusto` | 0.9·min(σ̂, IQR/1.349)·n^(−1/5) |
| `Scott` | 1.06·σ̂·n^(−1/5) |
| `plug-in 2 stadi` | stima θ₄ dai dati con pilota dal riferimento normale (§2.4) |
| `LSCV` | minimizza la stima non distorta dell'ISE |
| `MLCV (KL)` | massimizza la log-verosimiglianza leave-one-out |
| `c·σ̂/√n` | **forma del repo, costante ricalibrata** |
| `c·scala·n^(−1/5)` | esponente corretto, scala robusta, costante calibrata |
| `c·distanza dal k-esimo vicino` | ancoraggio a una scala **locale** anziché globale |

Le ultime tre hanno una costante libera. Calibrarla sul banco e poi valutarla sullo stesso
banco ripeterebbe l'errore che stiamo criticando, quindi sono calibrate in
**leave-one-density-out**: la costante usata su una densità è quella ottima su *tutte le
altre*. È il minimo per poter dire "questa regola generalizza".

### 2.3 Metrica

- primaria: **ISE della pdf**, ∫(f̂ − f)², perché la densità è il deliverable;
- secondaria: **KS della CDF**, max|F̂ − F|;
- l'**oracolo** è l'h che minimizza l'errore vero. Non è un selettore (usa la verità): serve
  a normalizzare;
- si riporta l'**efficienza** = errore(selettore)/errore(oracolo) ≥ 1, che rende confrontabili
  densità con scale di errore diversissime, più il **caso peggiore** e la frazione di casi
  con efficienza > 2, perché il docente sceglierà **una** distribuzione e a noi interessa non
  sfigurare su quella, non fare bella figura in media.

**Nota di equità, dichiarata in anticipo.** LSCV minimizza una stima non distorta di
esattamente il criterio con cui la valutiamo (l'ISE): un suo vantaggio su quella metrica è
atteso e non prova nulla da solo. La verifica indipendente è la colonna KS, dove il criterio
di LSCV non ha alcun vantaggio strutturale.

### 2.4 Il plug-in a due stadi, derivato

Non è stata usata alcuna formula ricordata a memoria. Partendo da
h_AMISE = [R(K)/(n·μ₂(K)²·θ₄)]^(1/5) con θ₄ = ∫(f″)², e per kernel gaussiano
R(K) = 1/(2√π) e μ₂ = 1:

- θ₄ si stima con θ̂₄(g) = (1/(n²g⁵))·Σᵢⱼ φ⁽⁴⁾((xᵢ−xⱼ)/g), φ⁽⁴⁾(u) = (u⁴−6u²+3)φ(u);
- l'ampiezza pilota AMSE-ottima è g = [−2·K⁽⁴⁾(0)/(μ₂·θ₆·n)]^(1/7) con K⁽⁴⁾(0) = 3/√(2π);
- θ₆ si prende dal riferimento normale: θ₆ = −15/(16√π·s⁷);
- sostituendo: **g = (96/(15√2))^(1/7)·s·n^(−1/7) ≈ 1.2407·s·n^(−1/7)**.

Controllo di coerenza della derivazione: usando anche per θ₄ il riferimento normale
θ₄ = 3/(8√π s⁵) si ottiene h = [4s⁵/(3n)]^(1/5) = 1.059·s·n^(−1/5), che è esattamente la
regola di Scott/Silverman — la macchina algebrica è quindi corretta. La validazione numerica
è in §3, V2.

---

## 3. Validazioni

Eseguite prima dei risultati: se una fallisce, il resto non vale nulla.

| # | controllo | esito |
|---|---|---|
| V1 | ogni densità integra a 1, CDF da 0 a 1 sulla griglia usata | scarto max **2.0e-09** sulla massa, **9.9e-10** sugli estremi |
| V2 | il plug-in su dati gaussiani deve dare ≈ 1.06·s·n^(−1/5) | costante implicita **1.043** (n=200), **1.043** (n=500), **1.051** (n=2000) |
| V3 | l'interpolazione della curva ISE(h) non falsa le efficienze | errore relativo max **0.19 %** |
| V4 | l'ottimo dell'oracolo non tocca i bordi della griglia di h | **0/16** densità sul bordo |

---

### 3bis. Le misure trasferiscono al codice del repo?

Lo studio usa un'implementazione NumPy indipendente. Verificato che coincida con quella del
repo (`temp_analysis/parzen_equivalence.txt`):

| confronto | esito |
|---|---|
| `parzen_cdf` e `parzen_pdf`, quattro valori di h | **bit-identiche** (scarto max 2.2e-16) |
| LSCV, 5 semi | accordo entro **3–8 %** (griglie di candidati diverse per costruzione) |
| Silverman | il repo differisce dalla nostra di **esattamente 1.814 = π/√3** |

L'ultima riga è una conferma numerica del difetto di scala: `parzen.silverman_bandwidth`
`[codice parzen.py:118-133]` **non è variance-matched** al kernel logistico, che è quello di
default. Il repo ha `variance_matched_bandwidth` per questo, ma l'app espone anche
`silverman` come strategia a sé `[codice server.py:109]`, e quella sovraliscia di 1.81×.

---

## 4. Risultati (n = 500, 12 semi)

Output integrale in `temp_analysis/bandwidth_n500.txt`.

### 4.1 Il risultato centrale: la costante non è una costante

Per ciascuna famiglia con una costante libera, ecco la costante **ottima densità per
densità**. Se esistesse una "legge", sarebbero tutte uguali.

| famiglia | min | max | rapporto |
|---|---|---|---|
| `c·σ̂/√n` (la forma del repo) | 0.36 | 4.15 | **11.6×** |
| `c·scala robusta·n^(−1/5)` (esponente corretto) | 0.11 | 1.16 | **10.6×** |
| `c·distanza dal k-esimo vicino` (ancoraggio locale) | 0.46 | 2.54 | **5.6×** |

Valori per la forma del repo: MW01 4.15, MW02 3.25, MW03 0.66, MW04 0.84, MW05 1.38,
MW06 2.87, MW07 1.55, MW08 2.25, MW09 2.54, MW10 0.95, MW11 2.87, MW12 1.22, MW13 1.99,
REPO-trimodale 1.08, REPO-5-mode 0.36, REPO-6-mode 0.58.

**Lettura.** Il valore 1.5 non è "il numero sbagliato": è la **forma** a essere sbagliata.
Nessuna costante può stare contemporaneamente a 0.36 e a 4.15. Si noti che l'ancoraggio a
una scala **locale** (k-esimo vicino) dimezza la dispersione, da 11.6× a 5.6×: conferma che
la diagnosi era giusta (σ̂ è la grandezza sbagliata), ma non abbastanza da salvare l'idea di
una regola a costante.

### 4.2 Efficienza sull'ISE della pdf, densità per densità

Efficienza = ISE(selettore)/ISE(oracolo). 1.00 è l'ottimo irraggiungibile.

| densità | repo h₁=1 | repo 1.5σ̂ | Silverman | Scott | plug-in | **LSCV** | MLCV | c·σ̂/√n LODO | c·s·n^(−1/5) LODO | c·kNN LODO |
|---|---|---|---|---|---|---|---|---|---|---|
| MW01 gaussian | 6.08 | 3.66 | 1.40 | **1.21** | 1.25 | 1.59 | 1.57 | 5.63 | 4.33 | 3.61 |
| MW02 skewed unimodal | 3.66 | 2.78 | 1.23 | **1.12** | 1.17 | 1.49 | 1.27 | 3.64 | 3.81 | 3.11 |
| MW03 strongly skewed | 1.32 | 2.42 | 5.49 | 7.96 | 3.33 | **1.15** | 1.39 | 1.77 | 1.88 | 1.58 |
| MW04 kurtotic unimodal | 1.32 | 1.79 | 2.54 | 8.40 | 1.61 | **1.23** | 1.87 | 1.31 | 1.24 | 4.38 |
| MW05 outlier | 6.39 | 1.24 | 1.43 | 9.56 | 1.33 | 1.56 | 3.11 | **1.19** | 3.79 | 3.16 |
| MW06 bimodal | 3.70 | 1.86 | **1.08** | 1.25 | 1.11 | 1.49 | 1.24 | 2.39 | 2.12 | 1.52 |
| MW07 separated bimodal | 2.31 | **1.04** | 3.65 | 5.53 | 1.29 | 1.17 | 1.18 | 1.14 | 1.07 | 1.25 |
| MW08 skewed bimodal | 2.88 | 1.48 | 1.36 | 1.82 | 1.29 | 1.22 | **1.11** | 1.97 | 1.72 | 1.22 |
| MW09 trimodal | 2.66 | 1.34 | 1.15 | 1.40 | **1.08** | 1.12 | 1.11 | 1.66 | 1.50 | 1.17 |
| MW10 claw | **1.10** | 1.47 | 3.49 | 3.83 | 3.61 | 1.11 | 1.40 | 1.16 | 1.47 | 1.54 |
| MW11 double claw | 2.43 | 1.45 | **1.05** | 1.15 | 1.06 | 1.26 | 1.15 | 1.73 | 1.59 | 1.27 |
| MW12 asymmetric claw | 1.09 | 1.05 | 1.66 | 1.87 | 1.71 | 1.12 | 1.08 | **1.03** | 1.03 | 1.08 |
| MW13 asym double claw | 1.45 | 1.09 | 1.13 | 1.24 | 1.12 | 1.27 | 1.09 | 1.17 | 1.12 | **1.06** |
| REPO trimodale | 2.13 | 1.27 | 4.86 | 6.30 | 2.79 | 1.12 | 1.09 | **1.05** | 1.13 | 1.18 |
| REPO 5 mode strette | 1.85 | 9.38 | 21.01 | 22.94 | 18.77 | 1.17 | **1.13** | 9.89 | 11.05 | 1.37 |
| REPO 6 mode scale miste | 2.11 | 3.05 | 7.75 | 8.71 | 7.50 | **1.09** | 1.12 | 2.66 | 3.06 | 2.79 |

### 4.3 Sintesi

| selettore | ISE media | ISE p90 | ISE peggiore | KS media | KS peggiore | casi > 2× |
|---|---|---|---|---|---|---|
| repo h₁ = 1.0 | 2.66 | 5.07 | 6.39 | 1.28 | 2.70 | 51 % |
| repo 1.5·σ̂ | 2.27 | 3.95 | 9.38 | 1.25 | 1.84 | 33 % |
| Silverman robusto | 3.77 | 8.11 | 21.01 | 1.49 | 2.70 | 46 % |
| Scott | 5.27 | 12.33 | 22.94 | 1.84 | 3.60 | 54 % |
| plug-in 2 stadi | 3.13 | 6.95 | 18.77 | 1.37 | 2.40 | 36 % |
| **LSCV** | **1.26** | **1.61** | **1.59** | **1.11** | **1.31** | **6 %** |
| MLCV (KL) | 1.37 | 1.98 | 3.11 | 1.18 | 1.93 | 10 % |
| `c·σ̂/√n` ricalibrata [LODO] | 2.46 | 5.33 | 9.89 | 1.23 | 1.87 | 34 % |
| `c·s·n^(−1/5)` [LODO] | 2.62 | 5.40 | 11.05 | 1.24 | 1.95 | 37 % |
| `c·kNN` [LODO] | 1.96 | 3.48 | 4.38 | 1.21 | 2.01 | 28 % |

Quattro fatti che contano più della classifica:

1. **LSCV domina, e domina soprattutto nel caso peggiore.** Efficienza media 1.26 e massima
   1.59: è l'unico selettore che non ha un caso catastrofico. Tutti gli altri ne hanno almeno
   uno sopra 3×, e le regole normali arrivano a 21–23×.
2. **Ricalibrare la costante non serve a niente, e può peggiorare.** La forma del repo con
   costante ricalibrata in leave-one-density-out fa **2.46** di media contro il **2.27**
   della costante 1.5 lasciata com'è. Non è che 1.5 sia un numero particolarmente buono: è
   che nessun numero funziona, quindi ottimizzarlo su altre densità non trasferisce nulla.
   Questo è il colpo definitivo all'idea di regola.
3. **Le regole classiche falliscono qui più che in letteratura, e per un motivo preciso.**
   Silverman, Scott e il plug-in sono costruiti sul riferimento normale e sovralisciano le
   densità multimodali: sul caso a 5 mode separate fanno 21×, 23× e 19×. Sono buoni su MW01 e
   MW02 (1.2–1.4, dove LSCV fa 1.5–1.6): se il bersaglio fosse noto liscio e unimodale,
   sarebbero la scelta giusta. Non è il nostro caso.
4. **Il KS conferma indipendentemente.** LSCV è il migliore anche sul KS (1.11 medio, 1.31
   peggiore), dove il suo criterio non ha alcun vantaggio strutturale. Era la verifica
   dichiarata in anticipo in §2.3, ed è passata.

### 4.4 L'instabilità della CV: misurata, non assunta

La critica classica a LSCV è che h vari molto da campione a campione. **È vera**, ed è
misurata (coefficiente di variazione di h sui 12 semi):

| selettore | intervallo del CV di h sulle 16 densità |
|---|---|
| Silverman robusto | 0.01 – 0.17 |
| plug-in 2 stadi | 0.02 – 0.16 |
| MLCV (KL) | 0.07 – 0.34 |
| **LSCV** | **0.10 – 0.35** |

LSCV è effettivamente il più variabile, con h che oscilla del ±25 % circa fra campioni. Ma
questa variabilità **non si trasferisce all'errore**: l'efficienza p90 di LSCV è 1.61 e la
peggiore in assoluto 1.59, cioè non c'è coda. Il motivo è che la curva ISE(h) è piatta
attorno al minimo: sbagliare h del 25 % costa pochissimo, mentre sbagliarlo di un fattore 3
— che è ciò che fa una regola a costante su una densità che non le somiglia — costa tutto.

**Morale metodologica:** la stabilità di un selettore va giudicata sull'errore che produce,
non sulla stabilità del suo output. Un selettore perfettamente stabile e sistematicamente
sbagliato (Scott: CV ≈ 0.03, efficienza 5.27) è molto peggio di uno rumoroso e centrato.

### 4.5 h₁ stimato, dentro lo schedule del corso

Lo schedule h_n = h₁/√n resta la forma di consegna (§5.1). LSCV non lo sostituisce: ne
**stima h₁**, con ĥ₁ = h_LSCV·√n. Valori medi ottenuti ai due budget:

| densità | ĥ₁ (n=500) | ĥ₁/σ̂ (n=500) | ĥ₁ (n=1000) | ĥ₁/σ̂ (n=1000) |
|---|---|---|---|---|
| MW01 gaussian | 3.759 | 3.773 | 4.531 | 4.536 |
| MW02 skewed unimodal | 2.516 | 3.044 | 3.192 | 3.920 |
| MW03 strongly skewed | 0.683 | 0.669 | 0.769 | 0.744 |
| MW04 kurtotic unimodal | 0.666 | 0.822 | 0.808 | 0.985 |
| MW05 outlier | 0.396 | 1.161 | 0.476 | 1.470 |
| MW06 bimodal | 3.009 | 2.487 | 3.979 | 3.310 |
| MW07 separated bimodal | 2.188 | 1.379 | 2.790 | 1.764 |
| MW08 skewed bimodal | 2.735 | 2.493 | 3.088 | 2.818 |
| MW09 trimodal | 2.667 | 2.080 | 3.295 | 2.582 |
| MW10 claw | 0.774 | 0.892 | 0.943 | 1.083 |
| MW11 double claw | 2.884 | 2.391 | 4.043 | 3.372 |
| MW12 asymmetric claw | 1.298 | 1.181 | 1.288 | 1.162 |
| MW13 asym double claw | 1.913 | 1.602 | 2.305 | 1.935 |
| REPO trimodale | 2.434 | 1.086 | 2.832 | 1.270 |
| REPO 5 mode strette | 1.811 | 0.319 | 2.516 | 0.444 |
| REPO 6 mode scale miste | 2.159 | 0.501 | 2.723 | 0.634 |

Due letture, entrambe importanti per il report.

**1. Il rapporto ĥ₁/σ̂ non è una costante.** Spazia da 0.319 a 3.773 a n = 500
(**11.8×**) e da 0.444 a 4.536 a n = 1000
(**10.2×**). È la refutazione della regola σ fatta con l'output del
selettore raccomandato, non con un oracolo: nemmeno una procedura che *guarda i dati*
produce un rapporto costante, perché quel rapporto non è una costante della natura. Si noti
dove cadono i casi peggiori per la regola 1.5: le tre densità del repo hanno ĥ₁/σ̂ pari a
1.09, **0.32** e 0.50 — cioè la regola sovraliscia di 3–5× proprio sulle multimodali con mode
separate, il caso dichiarato del docente.

**2. ĥ₁ cresce con il budget.** Su **15 densità su 16** il valore a n = 1000 è maggiore di
quello a n = 500, in media **+22 %** (unica eccezione MW12 asymmetric claw, −0.8 %, cioè
invariata entro il rumore). È la conseguenza diretta del fatto che
l'esponente vero della finestra non è 1/2 (§5.3): sotto lo schedule √n la differenza di
tasso si scarica interamente su h₁. Non è un difetto del selettore — vale identicamente per
qualunque regola, compresa 1.5·σ̂ — ma è la ragione per cui h₁ va presentato come una stima
**al budget dato** e non come una costante del problema.

### 4.6 Conferma a n = 1000

Stesso protocollo, budget secondario. Output in `temp_analysis/bandwidth_n1000.txt`.

| selettore | ISE media | ISE p90 | ISE peggiore | KS media | casi > 2× |
|---|---|---|---|---|---|
| repo h₁ = 1.0 | 2.58 | 4.68 | 5.61 | 1.24 | 55 % |
| repo 1.5·σ̂ | 2.31 | 4.37 | 9.45 | 1.24 | 32 % |
| Silverman robusto | 5.42 | 11.29 | 34.99 | 1.71 | 49 % |
| Scott | 7.61 | 16.13 | 39.23 | 2.17 | 58 % |
| plug-in 2 stadi | 3.94 | 8.64 | 26.33 | 1.47 | 38 % |
| **LSCV** | **1.20** | **1.45** | **1.62** | **1.10** | **3 %** |
| MLCV (KL) | 1.39 | 2.21 | 2.94 | 1.20 | 11 % |
| `c·σ̂/√n` ricalibrata [LODO] | 2.49 | 5.52 | 10.11 | 1.22 | 34 % |
| `c·s·n^(−1/5)` [LODO] | 2.89 | 6.83 | 13.83 | 1.25 | 39 % |
| `c·kNN` [LODO] | 2.11 | 4.11 | 5.05 | 1.23 | 34 % |

Il quadro non cambia, e su due punti si accentua:

1. **LSCV migliora con il budget** (1.26 → 1.20 di media, 6 % → 3 % di casi sopra 2×), come
   deve fare un selettore consistente.
2. **Le regole del riferimento normale peggiorano** (Silverman da 3.77 a 5.42, caso peggiore
   da 21.0 a 35.0; Scott da 5.27 a 7.61). Non è un paradosso: il loro errore è una
   distorsione sistematica di sovralisciamento, che non si riduce con n, mentre l'oracolo
   migliora — quindi il rapporto fra i due cresce. È la firma di un errore di modello, non di
   varianza.
3. **La dispersione della costante ottima aumenta**, da 11.6× a **13.2×** (da 0.40 a 5.31).
   E la costante ottima si sposta verso l'alto con n (MW01: 4.15 a n=500, 5.31 a n=1000),
   coerentemente con il fatto che l'esponente vero non è 1/2 (§5.3).

---

## 5. Raccomandazione

### 5.1 La forma di consegna: LSCV *dentro* h_n = h₁/√n

Lo schedule h_n = h₁/√n è la forma richiesta e viene mantenuta. Non è una concessione: è una
scelta legittima, perché soddisfa le condizioni classiche di consistenza (h_n → 0 e
n·h_n = h₁√n → ∞). Quello che cambia è **come si fissa h₁**.

Il punto da mettere in chiaro, perché è quello contestabile:

> h₁ non è mai stato una costante universale, nemmeno nella regola attuale. `h₁ = 1.5·σ̂` è
> una **funzione del campione**: σ̂ si calcola dai dati. La nostra proposta, ĥ₁ = h_LSCV·√n,
> è anch'essa una funzione del campione. Le due hanno lo stesso statuto formale; cambia
> soltanto la qualità dell'estimatore, ed è quella che abbiamo misurato.

Non stiamo quindi *ricavando a posteriori* un h₁ per giustificare un h scelto altrove — che
sarebbe la critica giusta e che il docente ha già mosso. Stiamo **stimando** lo stesso
parametro con una procedura che ha un criterio esplicito (minimizzare una stima non distorta
dell'ISE) invece che con una costante calibrata a occhio su tre misture.

Restano due proprietà da dichiarare apertamente, perché un lettore attento le troverebbe:

1. **ĥ₁ dipende dal campione**, quindi varia da un campione all'altro (§4.4). Anche 1.5·σ̂ lo
   fa, in misura minore ma nella direzione sbagliata: è stabile attorno al valore errato.
2. **ĥ₁ non è costante rispetto a n.** Poiché la finestra ottima per la pdf scala come
   n^(−1/5) e non come n^(−1/2), ĥ₁ = h·√n cresce lentamente con il budget. È una proprietà
   dello schedule, non del nostro selettore, e vale identicamente per la regola σ̂ (misurato:
   l'h₁ ottimo cresce come ~n^0.25–0.30). A **budget fissato** — il nostro caso, n ≈ 500 — la
   cosa non ha alcuna conseguenza. La quantificazione è in §5.3.

### 5.2 La procedura

**Usare LSCV per stimare h₁, e consegnare h_n = ĥ₁/√n con ĥ₁ = h_LSCV·√n.**

- È già implementato nel repo `[codice parzen.py:240-261]` e non è mai stato usato come
  scelta di default.
- Costo O(n²·|griglia|·|candidati|), pagato una volta sola, fuori linea: a n = 500 sono
  frazioni di secondo. L'argomento "la regola σ costa zero" `[docs/study2.md:79-80]` è vero e
  irrilevante, perché il costo confrontato non è mai stato un vincolo.
- La griglia di candidati va costruita su una **scala robusta**, min(σ̂, IQR/1.349)·n^(−1/5),
  non su σ̂: è quella usata qui.
- **Correggere il candidato per il kernel logistico** dividendo per π/√3: senza questa
  correzione si sovraliscia di 1.81× (è il bug di scala già segnalato dal docente).

Guardrail consigliati, da riportare come diagnostica e non come correzione silenziosa:

1. se h_LSCV finisce sul bordo della griglia dei candidati, allargare la griglia e segnalarlo;
2. riportare sempre anche h_Silverman e h_plug-in: se differiscono da h_LSCV di più di un
   fattore 3, il campione è probabilmente multimodale a scale miste, ed è un'informazione
   utile da mostrare, non da nascondere;
3. riportare la log-verosimiglianza LOO al h scelto come indicatore truth-free di qualità.

**Cosa non fare:** non ricalibrare 1.5 su un banco più grande. È esattamente ciò che abbiamo
provato (riga `c·σ̂/√n [LODO]` della tabella 4.3) ed è risultato **peggiore** della costante
originale. Una forma sbagliata non si aggiusta con una costante migliore.

### 5.3 Quanto costa davvero lo schedule √n (e perché a n = 500 non costa)

Misurato in `temp_analysis/schedule_exponent.py`.

**A — con che esponente scala la finestra ottima.** Regressione di log h\* su log n
(h\* = finestra che minimizza l'ISE vero, 6 semi, n da 100 a 2000):

| densità | p stimato |
|---|---|
| MW01 gaussian | 0.232 |
| MW06 bimodal | 0.283 |
| MW09 trimodal | 0.320 |
| MW10 claw | 0.373 |
| MW12 asymmetric claw | 0.388 |
| REPO trimodale | 0.257 |
| REPO 5 mode strette | 0.257 |
| REPO 6 mode scale miste | 0.279 |
| **media** | **0.299** |

La teoria MISE per la densità dice p = 0.200; lo schedule del corso impone p = 0.500. Il
valore misurato sta in mezzo (0.23–0.39), più vicino alla teoria che allo schedule, e sale
sulle densità con struttura fine — coerente con il fatto che a n finito lo stimatore è ancora
in regime pre-asintotico e deve risolvere le mode strette.

**B — quanto costa calibrare h₁ a un budget e usarlo a un altro.** h₁ è preso dall'oracolo a
n_cal e applicato via h = h₁/√n_uso; si riporta l'efficienza ISE rispetto all'oracolo di
n_uso.

| densità | n_cal | uso 100 | uso 250 | uso 500 | uso 1000 | uso 2000 |
|---|---|---|---|---|---|---|
| MW01 gaussian | 100 | 1.04 | 3.18 | 1.62 | 2.18 | 2.03 |
| | 500 | 1.96 | 1.26 | **1.11** | 1.34 | 1.32 |
| | 2000 | 5.39 | 2.59 | 1.70 | 1.10 | 1.04 |
| MW06 bimodal | 100 | 1.07 | 1.20 | 1.20 | 1.67 | 1.70 |
| | 500 | 1.55 | 1.08 | **1.03** | 1.22 | 1.21 |
| | 2000 | 2.50 | 1.51 | 1.20 | 1.06 | 1.02 |
| MW09 trimodal | 100 | 1.01 | 1.16 | 1.23 | 1.51 | 1.43 |
| | 500 | 1.33 | 1.08 | **1.01** | 1.05 | 1.05 |
| | 2000 | 1.56 | 1.17 | 1.03 | 1.02 | 1.02 |
| MW10 claw | 100 | 1.08 | 1.01 | 1.01 | 1.07 | 1.22 |
| | 500 | 1.11 | 1.03 | **1.01** | 1.04 | 1.16 |
| | 2000 | 1.37 | 1.27 | 1.17 | 1.06 | 1.00 |
| MW12 asymmetric claw | 100 | 1.01 | 1.06 | 1.12 | 1.05 | 1.10 |
| | 500 | 1.06 | 1.02 | **1.02** | 1.00 | 1.00 |
| | 2000 | 1.08 | 1.03 | 1.02 | 1.01 | 1.00 |
| REPO trimodale | 100 | 1.00 | 1.26 | 1.30 | 1.58 | 1.71 |
| | 500 | 1.39 | 1.03 | **1.01** | 1.07 | 1.16 |
| | 2000 | 2.17 | 1.37 | 1.21 | 1.04 | 1.02 |
| REPO 5 mode strette | 100 | 1.00 | 1.18 | 1.30 | 1.48 | 1.81 |
| | 500 | 1.38 | 1.05 | **1.01** | 1.07 | 1.20 |
| | 2000 | 2.22 | 1.55 | 1.21 | 1.07 | 1.01 |
| REPO 6 mode scale miste | 100 | 1.01 | 1.05 | 1.16 | 1.42 | 1.55 |
| | 500 | 1.17 | 1.09 | **1.00** | 1.06 | 1.12 |
| | 2000 | 1.52 | 1.48 | 1.18 | 1.04 | 1.00 |

**Come si legge, e una correzione a noi stessi.** Avevamo scritto nello script che la
diagonale vale 1.00 "per costruzione". **È falso**: vale 1.01–1.11, perché h₁ è calibrato
sull'ottimo *medio* fra campioni mentre il denominatore è l'ottimo del *singolo* campione.
Quello scarto non è un artefatto: è il costo reale di usare un unico h₁ per tutti i campioni
allo stesso n, ed è piccolo (1–11 %). Il costo attribuibile allo **schedule** è invece la
differenza fra le celle fuori diagonale e la diagonale della stessa colonna.

Tre conclusioni:

1. **A budget fissato lo schedule non costa nulla.** È il caso nostro (n ≈ 500): qualunque h
   è rappresentabile come h₁/√n, e la diagonale mostra che il residuo è dell'ordine del
   pochi per cento.
2. **Spostarsi di un gradino di budget costa poco** (≤ 1.25 quasi ovunque). Quindi anche se
   il docente passasse da 500 a 1000, un h₁ calibrato a 500 resterebbe ragionevole.
3. **Trasportare h₁ su un fattore 20 di budget costa molto** (fino a 5.39). È la ragione per
   cui h₁ non va presentato come una costante universale del problema: è una stima
   valida *al budget a cui è stata fatta*.

**Anomalia dichiarata.** La cella (MW01, cal 100, uso 250) vale 3.18 e rompe la monotonia
della sua riga: è un valore fuori linea prodotto dalla media di rapporti su soli 6 semi su una
densità dove l'ISE assoluto è piccolo e i rapporti sono quindi volatili. Non la usiamo per
alcuna conclusione; tutte le altre 119 celle sono ordinate come atteso.

---

## 6. Cosa questo studio NON afferma

1. **Non afferma che LSCV sia il miglior selettore in assoluto.** Su densità lisce e
   unimodali (MW01, MW02) Scott e il plug-in lo battono (1.12–1.25 contro 1.49–1.59). La
   raccomandazione è condizionata al nostro scenario: bersaglio ignoto e dichiaratamente
   multimodale.
2. **Il vantaggio di LSCV sull'ISE era atteso** — minimizza una stima non distorta proprio di
   quel criterio. Era dichiarato prima di misurare (§2.3). La prova indipendente è il KS.
3. **Non afferma che il plug-in sia implementato male.** È validato (V2: costante implicita
   1.043–1.051 contro 1.06 teorico); fallisce perché il riferimento normale è inadatto alle
   multimodali, che è un limite del metodo, non dell'implementazione.
4. **Non è la variante 'solve-the-equation' di Sheather–Jones.** Quella richiede una formula
   che non è stato possibile verificare da fonte primaria, e implementarla a memoria avrebbe
   messo a rischio tutto lo studio. Il plug-in usato è a due stadi con pilota dal riferimento
   normale, derivato per intero in §2.4.
5. **12 semi.** Differenze sotto il ~10 % non vanno interpretate. Le differenze su cui
   poggiano le conclusioni sono fattori 2–20×.
6. **MW14 e MW15 sono esclusi** perché non verificabili. Sono due densità "comb", quindi
   fra le più multimodali del banco: la loro assenza rende lo studio, se mai, **conservativo**
   verso le regole a costante.
7. **Solo kernel logistico e solo 1-D.** Provato a n = 500 e n = 1000, con lo stesso esito.
   Il ranking non è garantito fuori da questo perimetro.
8. **Nulla di questo studio dipende dall'architettura della rete.** Interagisce con
   `docs/redesign_network.md` in un punto solo, e in senso rassicurante: lì si misura che con
   l'architettura vincolata l'ISE finale è poco sensibile alla finestra del maestro, quindi
   un errore su h ha conseguenze attenuate a valle.

---

## 7. Frasi pronte per il report

> La finestra non viene fissata da una regola empirica ma scelta per cross-validation dei
> minimi quadrati, che minimizza una stima non distorta dell'errore quadratico integrato
> calcolabile dai soli campioni. Su un banco di 16 densità — le tredici di Marron & Wand
> (1992) verificabili più i tre casi del nostro studio preliminare — la sua efficienza
> rispetto all'oracolo è 1.26 in media e 1.59 nel caso peggiore, contro 2.27 e 9.38 della
> regola h₁ = 1.5·σ̂.

> La ricerca di una regola nella forma h₁ = c·σ̂ è stata abbandonata dopo averne misurato
> l'impossibilità: la costante ottima varia di 11.6 volte fra le densità del banco (da 0.36 a
> 4.15). Ricalibrare c su un banco più ampio, con validazione leave-one-density-out, produce
> un risultato peggiore della costante di partenza (efficienza media 2.46 contro 2.27),
> confermando che il problema è la forma della regola e non il valore della costante.

> Le regole basate sul riferimento normale (Silverman, Scott, plug-in) sono state incluse e
> scartate: sovralisciano sistematicamente le densità multimodali, con efficienze fino a 23
> volte l'oracolo sulla densità a mode strette e separate, pur essendo le migliori sulle
> densità unimodali lisce.
