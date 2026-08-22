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

---

## 8. Specifiche di dettaglio

Le decisioni D-xx dicono *cosa* fare; qui si fissa *come*, al livello di dettaglio necessario
perché qualcun altro possa implementare senza chiedere nulla. Criterio adottato: una specifica
è completa quando si riesce a scriverne il test **prima** del corpo della funzione.

### S1 — Perimetro del refactor: il vecchio non si tocca

**Il percorso nuovo si affianca a quello esistente, non lo sostituisce.** `rectify_cdf` ha
sette chiamanti fra script e app; sono tutti script che documentano studi già fatti
(`checkpoint1.py`, `mlp_*.py`, `mlp2_*.py`) e vanno lasciati funzionanti così come sono. Il
loro valore è di essere la traccia riproducibile di ciò che è stato misurato.

Conseguenze:
- `rectify_cdf`, `density_from_cdf`, `monotonicity_penalty`, `CDFNet` **restano in libreria**,
  con una nota nella docstring che rimanda alla decisione che li ha superati;
- il percorso nuovo (`MixtureCDFNet` → `run_from_samples`) è **additivo**;
- l'app si allinea dopo (P-05), non ora.

Questo limita il raggio d'azione: nessuna riga esistente cambia comportamento.

### S2 — `MixtureCDFNet` in `src/parzen_cdf/models.py`

Firma, invarianti e responsabilità:

| elemento | specifica |
|---|---|
| costruttore | `MixtureCDFNet(n_components: int = 12)`; J = 12 da D-08 rivista |
| parametri | `alpha`, `b`, `u`, ciascuno di forma `(J,)` |
| buffer | `mu`, `sd` scalari: statistiche di standardizzazione, nel `state_dict` |
| `init_from_samples(x)` | centri sui quantili di z, larghezze pari al passo fra centri, pesi uniformi; **deterministica dato x**, non consuma RNG |
| `forward(x)` | CDF, accetta qualunque forma, restituisce `(n,)` |
| `pdf(x)` | densità in forma chiusa, divisa per `sd` |
| invarianti | F(±∞) = 0/1 esatti; F non decrescente; pdf ≥ 0; ∫pdf = 1 — **per ogni valore dei parametri** |

`init_from_samples` non deve consumare numeri casuali: è ciò che rende il modello
riproducibile senza dipendere dal seme globale, e chiude B8 per questa classe per costruzione
e non per disciplina.

### S3 — Addestramento: `fit_mixture_cdf` in `training.py`

`train_cdf` resta per `CDFNet`. Per la classe nuova serve una funzione separata, perché le
penalità di monotonia e curvatura non hanno più oggetto (D-10, D-11) e lasciarle disponibili
inviterebbe a riusarle.

```python
def fit_mixture_cdf(x, y, *, n_components=12, epochs=6000, lr=0.03, seed=0):
    """Costruisce e addestra. Il seme e' applicato PRIMA della costruzione."""
```

Restituisce `(model, history)`. Nessuna penalità, nessun clamp, nessuna rettifica: la
validità è dell'architettura.

### S4 — `Estimate`: l'oggetto consegnato

```python
@dataclass
class Estimate:
    model: MixtureCDFNet
    h: float                  # finestra scelta da LSCV
    h1: float                 # h * sqrt(n): la forma h_n = h1/sqrt(n) richiesta
    samples: np.ndarray

    def cdf(self, t) -> np.ndarray      # valutabile in qualunque punto di R
    def pdf(self, t) -> np.ndarray
    def domain(self, pad: float = 3.0) -> tuple[float, float]
    def diagnostics(self) -> dict
    def save(self, path) / load(path)   # state_dict + h + h1 + samples
```

Punto centrale: `cdf` e `pdf` sono **funzioni**, non tabelle. Il dominio serve solo a
disegnare e a integrare, ed è un metodo dell'oggetto, non un suo attributo: non fa parte
dello stimatore.

### S5 — Diagnostica truth-free: schema fisso

`diagnostics()` restituisce un dizionario con queste chiavi, sempre le stesse, così la CLI e
il report leggono la stessa struttura:

```json
{
  "n": 500,
  "h": 0.0995,
  "h1": 2.2249,
  "h_riferimento": {"silverman": 0.5814, "variance_matched": 0.3205, "rapporto_max": 5.84},
  "lscv_score": -0.2341,
  "loo_loglik": -1.8703,
  "massa_sul_dominio": 0.99982,
  "dominio": [-4.12, 6.31],
  "violazioni_monotonia": 0,
  "avvisi": []
}
```

Regole:
- `lscv_score` e `loo_loglik` sono i due indicatori con rho 0.94 e 0.86 rispetto all'errore
  vero (D-13). Vanno riportati sempre;
- `massa_sul_dominio` è **riportata, non imposta** (D-14 sulla massa): un valore diverso da 1
  segnala un dominio stretto, ed è un'informazione;
- `violazioni_monotonia` deve valere 0: se non lo è, è un bug dell'architettura, non un
  fenomeno da correggere;
- **il KS contro la ECDF non compare e non deve comparire** (D-14). Il divieto va scritto
  nella docstring del modulo, con il motivo, per impedirne la reintroduzione;
- `avvisi` raccoglie le condizioni da segnalare: h al bordo della griglia dei candidati,
  rapporto fra selettori superiore a 3, massa fuori da [0.99, 1.01].

### S6 — Tolleranze nei test

PyTorch lavora in float32: un ULP attorno a 0.5 vale 6.0e-08. Le soglie vanno fissate di
conseguenza, altrimenti si producono falsi allarmi (`[V1]` ne ha prodotto uno).

| proprietà | tolleranza |
|---|---|
| monotonia (`diff >= -tol`) | **1e-6** |
| pdf in forma chiusa contro differenza finita | 1e-3 (domina l'errore della differenza finita) |
| F(±10⁶) contro 0 e 1 | esatta (`== 0.0`, `== 1.0`) |
| massa su dominio largo | 1e-3 |
| equivarianza per traslazione/scala | 1e-3 (rumore amplificato dall'ottimizzatore, vedi `[E7]`) |

### S7 — Criteri di accettazione del refactor

Il refactor è concluso quando:

1. i test di caratterizzazione passano **invariati** (nessun cambiamento accidentale);
2. ogni decisione da D-07 a D-15 ha un test che è stato **visto fallire** prima di passare;
3. `run_from_samples` gira su un vettore di numeri senza che esista alcun oggetto `Mixture`;
4. `grep -r "Mixture" src/parzen_cdf/` non compare nel percorso di stima, solo in `data.py` e
   `evaluation.py`;
5. gli script esistenti continuano a funzionare come prima.

---

## 9. Verifica finale: il percorso consegnato contro quello che sostituisce

Le sezioni precedenti giustificano le singole decisioni. Questa misura la **pipeline
completa**, cioè ciò che il docente eseguirebbe davvero, contro quella che sostituisce.

Banco: `temp_analysis/endtoend_compare.py` → `endtoend_compare.txt`. n = 500, 5 semi,
5 distribuzioni. Per non avvantaggiare il percorso nuovo, **anche quello vecchio riceve il
dominio costruito dai campioni**: nel repo lo prenderebbe dalla distribuzione vera.

| caso | percorso | KS | ISE pdf | massa | violazioni |
|---|---|---|---|---|---|
| trimodale | Parzen (LSCV) | 0.0346 | 0.00398 | 1.000000 | 0 |
| | vecchio | 0.0365 | **0.00340** | 1.000000 | 0 |
| | **nuovo** | 0.0348 | 0.00419 | 1.000000 | 0 |
| bimodale simmetrica | Parzen | 0.0393 | 0.00304 | 1.000000 | 0 |
| | vecchio | 0.0397 | 0.00318 | 1.000000 | 0 |
| | **nuovo** | **0.0390** | **0.00304** | 1.000000 | 0 |
| spike in broad | Parzen | 0.0331 | 0.00771 | 1.000000 | 0 |
| | vecchio | 0.0339 | **0.00620** | 1.000000 | 0 |
| | **nuovo** | **0.0321** | 0.00649 | 0.999990 | 0 |
| 5 mode strette | Parzen | 0.0393 | 0.00702 | 1.000000 | 0 |
| | vecchio | 0.0396 | 0.00667 | 1.000000 | 0 |
| | **nuovo** | **0.0364** | **0.00346** | 1.000000 | 0 |
| 6 mode scale miste | Parzen | 0.0389 | 0.00396 | 1.000000 | 0 |
| | vecchio | 0.0592 | 0.01026 | 1.000000 | 0 |
| | **nuovo** | **0.0385** | **0.00289** | 0.999998 | 0 |

**Sintesi su tutti i casi:**

| percorso | KS medio | KS peggiore | ISE media | ISE peggiore | massa media |
|---|---|---|---|---|---|
| Parzen (LSCV) | 0.0370 | 0.0393 | 0.00514 | 0.00771 | 1.000000 |
| vecchio | 0.0418 | 0.0592 | 0.00594 | 0.01026 | 1.000000 |
| **nuovo** | **0.0361** | **0.0390** | **0.00402** | **0.00649** | 0.999997 |

**Quattro letture, l'ultima delle quali è la più importante.**

1. **Il percorso nuovo vince su tutte le aggregate**, e vince soprattutto nel caso peggiore:
   KS 0.0390 contro 0.0592, ISE 0.00649 contro 0.01026. È la metrica che conta, perché il
   docente sceglierà **una** distribuzione e non una media.
2. **Il guadagno si concentra dove serve.** Sul caso a 6 mode a scale miste — il più vicino a
   «fortemente multimodale» — il KS migliora di 1.5× e l'ISE di 3.5×. Sui casi facili la
   differenza è nel rumore, e sulla trimodale il percorso vecchio ha un ISE leggermente
   migliore (0.00340 contro 0.00419): va detto.
3. **Il percorso nuovo batte anche il Parzen da cui impara**, su entrambe le metriche
   aggregate. Non era scontato: nella rivalidazione a J = 8 perdeva su un caso su tre.
4. **La colonna della massa dice la cosa più interessante.** Il percorso vecchio segna
   1.000000 su ogni riga perché la massa gli viene **imposta** dalla riscalatura sulla
   griglia; il nuovo segna 0.999990 e 0.999998 su due casi perché quella è la massa che
   **misura** sul dominio finito. Lo scarto di 1e-05 non è un difetto del nuovo: è la
   quantità che il vecchio nascondeva ridistribuendola all'interno del dominio (P-D2). Un
   1.000000 ottenuto per costruzione non è un risultato, è una normalizzazione.

**Zero violazioni di monotonia in tutte le 75 esecuzioni**, per entrambi i percorsi — ma per
ragioni diverse: il vecchio perché la rettifica le rimuove a posteriori, il nuovo perché non
possono esistere (T2).
