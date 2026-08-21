# Re-design della pipeline: dalla verità come stampella ai campioni come unico ingresso

**Ambito.** Il blocco che sta *attorno* allo stimatore: da dove viene il dominio di
valutazione, come si consegna un risultato senza avere la distribuzione vera, e come si fa
girare la pipeline sui campioni di qualcun altro. L'architettura della rete è in
`redesign_network.md`, la scelta della finestra in `study_bandwidth.md`.

**Il vincolo che genera tutto.** Il progetto verrà provato sui campioni del docente. Là non
esiste una `Mixture`, non esistono `means` e `stds`, non esiste una CDF di riferimento. Tutto
ciò che oggi dipende da quegli oggetti va sostituito o eliminato.

**Marche delle fonti:** `[codice f:r]` letto nel sorgente; `[E#]` misurato da
`temp_analysis/pipeline_evidence.py` e `temp_analysis/grid_rule.py`.

---

## 1. Cosa fa il codice oggi

| passo | dove | dipende dalla verità? |
|---|---|---|
| dominio di valutazione | `grid_for(mix)` = `means ± 5·stds` `[codice study2_common.py:21-24]` | **sì** |
| dominio nell'app | `mix.grid()` = `ppf(0.001)`/`ppf(0.999)` delle componenti vere `[codice server.py:96-100]` | **sì** |
| dominio in alcuni script | cablato: `np.linspace(-6, 6, 2000)` `[codice mlp_gaussian.py:41; mlp_empirical_cdf.py:39]` | **sì**, peggio |
| rettifica e massa unitaria | riscala usando gli estremi della griglia `[codice training.py:125-128]` | **sì**, per conseguenza |
| scelta di h | `silverman / variance_matched / lscv / likelihood_cv / adaptive` | no |
| griglia interna di LSCV | `samples.min() − 5·h0 … samples.max() + 5·h0` `[codice parzen.py:252]` | **no**, già corretta |
| etichette della rete | `parzen_cdf(samples, samples, h)` oppure LOO | no |
| metriche | `ks_vs_truth`, `pdf_mse`, ISE | sì, ma è il loro mestiere |

---

## 2. Difetti accertati

### P-D1 — Il dominio viene dalla distribuzione vera

Tre varianti dello stesso difetto (righe sopra). Il dominio non è una comodità grafica: è
l'insieme su cui la CDF viene rettificata, la pdf differenziata e la massa calcolata.
Sceglierlo conoscendo la risposta è un'informazione che sui dati del docente non avremo.

### P-D2 — La massa unitaria è relativa a quel dominio

`rectify_cdf` impone F=0 al primo nodo e F=1 all'ultimo `[codice training.py:125-128]`,
quindi ∫f = 1 per costruzione qualunque sia la qualità della stima. La massa vera che cade
fuori dal dominio non viene persa: viene **ridistribuita all'interno**, cioè trasformata in un
errore sistematico invisibile. Quantificata in `[E2b]`.

### P-D3 — Nessuna ingestione di campioni esterni, in tutto il repo

Verificato con ricerca su `src/`, `scripts/`, `app/`: **nessun** `argparse`, `sys.argv`,
`np.load`, `loadtxt`, `genfromtxt`, `read_csv`, `.npy`, `.csv`. L'endpoint `/api/parzen`
pretende `components` per costruire una `Mixture` `[codice server.py:128-138]`; l'interfaccia
non ha caricamento file. **Oggi la pipeline non è eseguibile sui campioni del docente senza
scrivere codice nuovo**, e il pezzo mancante non è isolato: dominio, rettifica e diagnostica
sono tutti agganciati all'oggetto `Mixture`.

### P-D4 — Nessuna diagnostica utilizzabile senza la verità

Tutto ciò che il codice riporta come "qualità" (`ks_vs_truth`, `pdf_mse`, `ks_target`)
richiede la distribuzione vera o il target di Parzen. Sui dati del docente resterebbero solo
`mass` e `violations`, e `[E3]` mostra che la massa è un indicatore debole (rho = 0.44).

### P-D5 — L'unica diagnostica truth-free "ovvia" è una trappola

Il candidato naturale sarebbe misurare l'accordo fra la CDF stimata e la ECDF dei campioni.
`[E3]` mostra che è **anti-correlata** con l'errore vero (rho = −0.03 in media, mediana
−0.26): premia h → 0, perché al tendere a zero della finestra la stima di Parzen converge
esattamente alla ECDF. Usarla per scegliere h costa un fattore **13.7** in media e fino a
**94.7** nel caso peggiore. È il tipo di numero che finirebbe in un report come "validazione"
mentre sta certificando il contrario.

---

## 3. Prove numeriche

### E1 — Quanto è sensibile la rettifica al dominio

Test isolante: si applica `rectify_cdf` a una CDF di Parzen **già valida** (monotona, in
[0,1]), dove dovrebbe essere un'operazione neutra, e si misura quanto non lo è. Dominio
`[min − k·h, max + k·h]`, media su 8 semi, n = 500.

| densità | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| MW01 gaussian | 0.0018 | 0.0008 | 0.0003 | 0.0000 | 0.0000 |
| MW02 skewed unimodal | 0.0019 | 0.0008 | 0.0003 | 0.0000 | 0.0000 |
| MW03 strongly skewed | 0.0013 | 0.0005 | 0.0002 | 0.0000 | 0.0000 |
| MW04 kurtotic unimodal | 0.0007 | 0.0003 | 0.0001 | 0.0000 | 0.0000 |
| MW05 outlier | 0.0005 | 0.0002 | 0.0001 | 0.0000 | 0.0000 |

Da confrontare con il KS tipico (~0.03): la distorsione diretta della rettifica è piccola e
sparisce con un dominio ampio. **Non è questo il problema del dominio** — è la massa tagliata,
misurata sotto.

### E2b — Come si costruisce il dominio dai soli campioni

Massa **vera** lasciata fuori dal dominio (meno è meglio; conta il caso peggiore). 16 densità,
8 semi, n = 500.

| regola | media | **caso peggiore** |
|---|---|---|
| A — solo min/max dei dati | 3.18e-03 | 4.60e-03 |
| B — padding 3·h (h da LSCV) | 1.16e-03 | 3.02e-03 |
| B — padding 10·h | 3.92e-04 | 1.70e-03 |
| C — padding 1·σ̂ | 1.18e-04 | 8.06e-04 |
| **C — padding 3·σ̂** | **7.04e-06** | **1.12e-04** |
| D — padding 3·scala robusta | 5.40e-05 | 7.85e-04 |
| E — quantili della CDF stimata (eps = 1e-4) | 8.63e-04 | 2.84e-03 |

**Il pavimento di riferimento:** senza padding la massa fuori vale circa 2/(n+1) = 4.0e-03 per
pure statistiche d'ordine, indipendentemente dalla densità. È esattamente la colonna A.

**Perché B ed E falliscono, ed è la stessa ragione.** Entrambe sono ancorate ad h. Ma h è
scelto per **risolvere la feature più stretta**, quindi è piccolo proprio nelle densità con
componenti strette dentro componenti larghe (MW04, MW05, MW10). Il padding k·h è allora
irrilevante rispetto a code che si estendono su una scala pari a σ. Per la regola E vale
identicamente: la coda della CDF stimata decade con rate 1/h, quindi i suoi quantili restano
appiccicati ai dati.

**Perché D è peggiore di C.** La scala robusta `min(σ̂, IQR/1.349)` è costruita per
*sottostimare* la dispersione in presenza di code pesanti. È la cosa giusta per scegliere la
risoluzione e quella sbagliata per scegliere l'estensione.

> **Principio che ne esce:** σ̂ è l'ancoraggio dell'**estensione**; h e la scala robusta sono
> gli ancoraggi della **risoluzione**. Sono due mestieri diversi e scambiarli si paga in modo
> misurabile.

### E3 — Quali diagnostiche truth-free funzionano

Per ogni valore di h si calcola l'errore vero (ISE) e quattro indicatori calcolabili dai soli
campioni, poi si misura se ordinano gli h come li ordina la verità. 16 densità x 8 semi,
sweep di 25 valori di h.

| indicatore | rho di Spearman con l'ISE vero | efficienza se lo si usa per scegliere h |
|---|---|---|
| **punteggio LSCV** | **+0.937** (mediana +0.97) | **1.21** (peggiore 3.3) |
| log-verosimiglianza LOO | +0.862 (mediana +0.96) | 1.52 (peggiore 15.0) |
| massa sulla griglia | +0.442 (mediana +0.60) | 5.95 (peggiore 55.2) |
| **KS contro la ECDF** | **−0.033** (mediana −0.26) | **13.66** (peggiore 94.7) |

Tre letture:

1. **Il punteggio LSCV non è solo un selettore, è la nostra migliore diagnostica.** rho = 0.94
   significa che ordina gli h quasi come farebbe la verità. Sui dati del docente possiamo
   riportarlo sapendo cosa vuol dire.
2. **La log-verosimiglianza LOO è buona in mediana ma ha una coda cattiva** (peggiore 15.0):
   utile come secondo parere, non come criterio unico.
3. **Il KS contro la ECDF va bandito.** È l'indicatore che verrebbe naturale mettere in un
   report — «guarda come la nostra CDF aderisce ai dati» — ed è quello che inganna di più.

---

## 4. La proposta

### P1 — Un ingresso unico: `run_from_samples(x)`

```python
def run_from_samples(x, *, n_components=8, epochs=6000, seed=0):
    """Unico punto d'ingresso. Non conosce alcuna distribuzione: vede solo x."""
    h     = parzen.lscv_bandwidth(x)                   # study_bandwidth.md, D-05
    h1    = h * np.sqrt(x.size)                        # forma richiesta: h_n = h1/sqrt(n)
    y     = loo_parzen_cdf_targets(x, h)               # etichette ai soli punti campione
    model = MixtureCDFNet(n_components)                # redesign_network.md, D-07
    model.init_from_samples(torch.as_tensor(x))
    train(model, x, y, epochs=epochs, seed=seed)
    return Estimate(model=model, h=h, h1=h1, samples=x)
```

`Estimate` espone `cdf(t)` e `pdf(t)` **valutabili in qualunque punto di R**, non tabelle. È
la conseguenza diretta dell'architettura scelta in `redesign_network.md`: la griglia esce
dallo stimatore e resta solo uno strumento di disegno e di misura.

### P2 — Il dominio, quando serve, si costruisce così

```python
def report_domain(x, pad=3.0):
    """Dominio per grafici, integrali e diagnostica. Ancorato alla DISPERSIONE."""
    s = x.std(ddof=1)
    return x.min() - pad * s, x.max() + pad * s
```

Con `pad = 3` la massa vera lasciata fuori è al più **1.1e-04** nel caso peggiore su 16
densità `[E2b]`, due ordini di grandezza sotto il KS tipico. Non conviene allargare oltre: il
numero di nodi è finito e allargare toglie risoluzione dove serve, cioè nelle densità a
componenti strette.

### P3 — Un blocco di diagnostica truth-free, sempre riportato

| quantità | perché |
|---|---|
| **punteggio LSCV al h scelto** | migliore indicatore disponibile, rho = 0.94 con l'errore vero `[E3]` |
| **log-verosimiglianza LOO** | secondo parere indipendente, rho = 0.86 |
| **∫f̂ sul dominio** | deve valere circa 1 **senza essere imposto**; se no, il dominio è stretto |
| **h scelto, e h₁ = h·√n** | la forma richiesta dall'esercizio |
| **h di Silverman e del plug-in, per confronto** | se differiscono da h_LSCV oltre un fattore 3, il campione è multimodale a scale miste: informazione da mostrare |
| **violazioni di monotonia** | con l'architettura scelta deve essere 0; se non lo è, è un bug |

**Da non riportare mai:** il KS contro la ECDF `[P-D5]`.

### P4 — Una CLI, perché il docente deve poter eseguire da solo

```
python -m parzen_cdf.run --samples campioni.csv --out risultati/
```

Produce `pdf.csv`, `cdf.csv` (su un dominio dichiarato), `diagnostics.json` e due figure.
Accetta `.csv`, `.txt` o `.npy` contenenti un vettore di numeri.

### P5 — Ciò che dipende dalla verità va isolato, non cancellato

Le funzioni che usano la distribuzione vera (metriche, oracolo, generazione dei banchi)
restano, ma in un modulo separato — `evaluation.py` — che **la pipeline non importa**. Così la
separazione è strutturale e verificabile a colpo d'occhio, invece che affidata alla disciplina.

---

## 5. Cosa questo documento NON afferma

1. **Non afferma che il dominio sia un problema grave di accuratezza.** `[E1]` mostra che la
   distorsione diretta della rettifica è al più 0.002 e sparisce con un dominio adeguato. Il
   problema è di *metodo* (usiamo un'informazione che non avremo) e di *massa tagliata*
   `[E2b]`, non di errore grossolano.
2. **Non afferma che `pad = 3σ̂` sia ottimo.** È sufficiente: caso peggiore 1.1e-04 su questo
   banco. Su una densità a code molto più pesanti di una mistura gaussiana potrebbe non
   bastare, e non l'abbiamo provato.
3. **La regola E non è stata scartata perché sbagliata in principio**, ma perché ancorata alla
   grandezza sbagliata. Con un eps molto più piccolo, o applicata a una stima con code più
   larghe, si comporterebbe diversamente. Non l'abbiamo esplorato: la regola C è più semplice
   e funziona.
4. **Il rho di Spearman misura l'ordinamento, non la calibrazione.** Un indicatore con rho
   alto dice quale h è meglio, non *quanto* è buono in assoluto. Sui dati del docente sapremo
   scegliere, non sapremo dichiarare un errore.
5. **Nessuna di queste misure coinvolge la rete.** Sono tutte fatte sullo stimatore di Parzen,
   per isolare l'effetto del dominio da quello dell'architettura.
6. **Non abbiamo un formato concordato** con il docente per i campioni: la CLI assume un
   vettore di numeri, che è l'ipotesi ragionevole ma non confermata (domanda aperta 2 in
   `qa_analisi.md`).

---

## 6. Piano di implementazione

1. `evaluation.py`: spostare lì metriche e oracolo; la pipeline non lo importa `[P5]`.
2. `report_domain(x, pad=3.0)` in libreria, e sostituzione di ogni `grid_for(mix)` e
   `np.linspace(-6, 6, ...)` negli script `[P2]`.
3. `run_from_samples(x)` `[P1]`, che dipende dall'implementazione di `MixtureCDFNet`
   (`redesign_network.md` §6.1): quello va per primo.
4. `diagnostics(x, estimate)` `[P3]`, con il divieto esplicito sul KS-vs-ECDF scritto nella
   docstring, così nessuno lo reintroduce.
5. CLI `[P4]`.
6. Ingestione campioni nell'app: resta parcheggiata come P-05 nella roadmap.

---

## 7. Frasi pronte per il report

> Il dominio di valutazione è costruito dai soli campioni, come `[min(x) − 3σ̂, max(x) + 3σ̂]`.
> La scelta dell'ancoraggio non è arbitraria: su un banco di sedici densità la massa lasciata
> fuori è al più 1.1·10⁻⁴, contro 4.6·10⁻³ usando il solo intervallo dei dati e 1.7·10⁻³
> usando un margine proporzionale alla finestra. L'ampiezza del dominio va ancorata alla
> dispersione campionaria, non alla finestra: quest'ultima è scelta per risolvere la struttura
> più fine e risulta quindi troppo piccola proprio nelle distribuzioni con componenti strette.

> Poiché sui dati di prova la distribuzione generatrice non è nota, la qualità della stima è
> riportata attraverso indicatori calcolabili dai soli campioni. Fra i candidati, il punteggio
> di cross-validation dei minimi quadrati mostra una correlazione di rango di 0.94 con
> l'errore quadratico integrato vero, e la log-verosimiglianza leave-one-out di 0.86. È stato
> invece escluso il confronto fra CDF stimata e CDF empirica, che risulta anti-correlato con
> l'errore vero (−0.03): premia sistematicamente le finestre più strette, perché al tendere a
> zero della finestra lo stimatore converge alla CDF empirica per costruzione.
