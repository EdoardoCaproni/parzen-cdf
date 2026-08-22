# Resilienza a distribuzioni non gaussiane

**Domanda.** Tutte le calibrazioni del progetto — finestra, capacità, dominio — sono state
fatte su misture di normali. Il docente ha parlato di multimodalità, quindi è l'ipotesi più
probabile, ma non è una garanzia. Dove si rompe la pipeline se i dati vengono da un'altra
famiglia?

**Risposta breve.** Regge molto meglio del previsto. Due dei tre sospetti che avevamo
formulato erano sbagliati: le code pesanti non solo non sono un problema, sono più facili
delle misture gaussiane. L'unica debolezza vera sono i **bordi netti** — supporto compatto o
semiretta — ed è strutturale, non riparabile alzando la capacità senza pagarla altrove.

Banco: `temp_analysis/resilience_non_gaussian.py` e `edges_capacity.py`, n = 500, 5 semi
(3 per lo studio sulla capacità). Le distribuzioni sono oggetti `scipy.stats`, quindi pdf e
CDF esatte per il confronto.

---

## 1. I tre sospetti, formulati prima di misurare

| # | sospetto | esito |
|---|---|---|
| S1 | `report_domain` usa σ̂, che su code pesanti è instabile e per varianza infinita non esiste: il dominio esplode | **smentito** |
| S2 | il nucleo logistico ha code esponenziali, quindi non rappresenta code polinomiali | **smentito** |
| S3 | la mistura di CDF logistiche ha densità strettamente positiva ovunque (T2), quindi non può rappresentare un bordo netto | **confermato** |

---

## 2. Risultati

ISE della densità, media su 5 semi. Il riferimento è la mistura gaussiana su cui è stato
calibrato tutto (ISE 0.00377).

| distribuzione | ISE rete | rispetto al riferimento | ISE Parzen | rete / Parzen |
|---|---|---|---|---|
| **mistura gaussiana (riferimento)** | 0.00377 | 1.0× | 0.00435 | 0.87 |
| Student-t df = 3 | **0.00158** | **0.4×** | 0.00226 | **0.70** |
| Student-t df = 2 (varianza infinita) | 0.00216 | 0.6× | 0.00279 | 0.77 |
| Student-t df = 1 (Cauchy, media infinita) | 0.00138 | 0.4× | 0.00217 | 0.64 |
| lognormale | 0.00570 | 1.5× | 0.00485 | 1.17 |
| gauss + Student-t df = 3 | 0.00414 | 1.1× | 0.00554 | 0.75 |
| laplace + uniforme + gauss | 0.00859 | 2.3× | 0.00675 | 1.27 |
| gauss + uniforme | 0.01168 | 3.1× | 0.00958 | 1.22 |
| **uniforme** | 0.02201 | **5.8×** | 0.01753 | 1.26 |
| **gauss + esponenziale** | 0.02086 | 5.5× | 0.01651 | 1.26 |
| **esponenziale** | 0.03025 | **8.0×** | 0.02431 | 1.24 |
| **due uniformi separate** | 0.05212 | **13.8×** | 0.02368 | **2.20** |

### S1 — Il dominio non esplode, e il motivo è istruttivo

Il sospetto era che su code pesanti σ̂ si gonfiasse e il dominio `[min − 3σ̂, max + 3σ̂]` si
dilatasse fino a perdere risoluzione dove stanno i dati. Misurando il rapporto fra ampiezza
del dominio e ampiezza dei dati:

| distribuzione | dominio / intervallo dei dati | massa vera lasciata fuori |
|---|---|---|
| mistura gaussiana | 2.64 | 0.0e+00 |
| Student-t df = 3 | 1.60 | 1.1e-03 |
| Student-t df = 2 | 1.34 | 1.2e-03 |
| Student-t df = 1 (Cauchy) | **1.27** | 3.3e-03 |

Il rapporto **scende** invece di salire. La ragione: σ̂ effettivamente si gonfia sulle code
pesanti, ma l'intervallo dei dati `max − min` si gonfia **più in fretta**, perché è governato
dalle statistiche d'ordine estreme che su una Cauchy sono enormi. Il margine di 3σ̂ diventa
quindi trascurabile rispetto a un intervallo già larghissimo, e la regola si autocorregge.

La massa lasciata fuori sale a 3.3e-03 sulla Cauchy — trenta volte il riferimento gaussiano —
ma resta due ordini di grandezza sotto il KS tipico, quindi non è un problema operativo.

**Conclusione:** la regola del dominio è più robusta di quanto pensassimo, per una ragione che
non avevamo previsto. Nessuna modifica necessaria.

### S2 — Le code pesanti sono più facili, non più difficili

Il sospetto sembrava solido: un nucleo con code esponenziali non può riprodurre code
polinomiali. La misura dice il contrario — la Student-t df = 3 ha ISE **0.4 volte** il
riferimento gaussiano, e la rete batte il Parzen di un fattore 0.70.

Il motivo è che l'argomento confondeva due cose diverse. È vero che le code non vengono
riprodotte bene, ma **le code contengono pochissima massa**: l'errore quadratico integrato è
dominato dalla regione centrale, e una Student-t è un singolo picco liscio, cioè un bersaglio
molto più semplice di una mistura trimodale. La difficoltà di una densità, per questa misura,
sta nella sua struttura fine, non nel peso delle code.

**Conclusione:** nessuna modifica necessaria. Vale la pena dirlo nel report, perché è
controintuitivo.

### S3 — I bordi netti sono il limite vero, ed è strutturale

Qui il sospetto era giusto. La proposta ha densità strettamente positiva su tutto ℝ
(teorema T2): un salto di densità — il bordo di un'uniforme, l'origine di un'esponenziale —
può solo essere arrotondato.

I numeri lo confermano: uniforme 5.8× il riferimento, esponenziale 8.0×, due uniformi
separate **13.8×**. E su quest'ultimo caso la rete perde **2.2 volte** contro il suo stesso
maestro Parzen — l'unico caso in tutto lo studio in cui accade in modo marcato.

L'interpretazione è che il Parzen con n = 500 nuclei ha molti più gradi di libertà per
avvicinare un salto di quanti ne abbia una mistura con J = 12 componenti. La capacità
limitata, che è una virtù sui bersagli lisci perché rimuove varianza, diventa un limite
davanti a una discontinuità.

---

## 3. Esiste un rimedio a costo zero? No

Se la diagnosi è "servono più gradi di libertà", alzare J dovrebbe ridurre il divario. La
domanda decisiva è però un'altra: **quanto costa alzarlo sulle misture gaussiane**, che
restano il bersaglio atteso.

ISE della densità, media su 3 semi:

| caso | J = 12 | J = 24 | J = 48 | Parzen |
|---|---|---|---|---|
| due uniformi separate | 0.05378 | 0.03279 | **0.02692** | 0.02352 |
| uniforme | 0.02257 | 0.02105 | **0.02022** | 0.01897 |
| esponenziale | 0.03217 | 0.02834 | **0.02809** | 0.02454 |
| trimodale gaussiana | **0.00436** | 0.00530 | 0.00606 | 0.00477 |
| 6 mode gaussiane a scale miste | **0.00284** | 0.00325 | 0.00372 | 0.00475 |

Il compromesso è netto e va in due direzioni opposte:

- sui bordi, passare da J = 12 a J = 48 dimezza l'errore (0.0538 → 0.0269);
- sulle misture gaussiane lo peggiora del **39 %** sulla trimodale e del **31 %** sulle sei
  mode.

**Non esiste quindi un guadagno gratuito.** Ed è anche interessante che nemmeno J = 48 basti
a raggiungere il Parzen sui bordi (0.02692 contro 0.02352): la capacità non è tutta la
storia, la mistura di logistiche arrotonda comunque il salto.

**Decisione: si resta a J = 12** (D-08), e il limite si dichiara invece di nasconderlo. Il
progetto è calibrato su bersagli multimodali lisci, che sono quelli annunciati; se il campione
avesse un supporto compatto la stima resterebbe valida — è pur sempre una CDF corretta — ma
arrotonderebbe i bordi, e l'ISE peggiorerebbe di un fattore fra 3 e 14.

---

## 4. Cosa questo studio NON afferma

1. **Non afferma che la pipeline sia indifferente alla famiglia.** Sui bordi netti degrada in
   modo misurabile, e su un caso perde contro il suo stesso maestro.
2. **Non abbiamo provato supporti compatti multimodali complessi**, che sarebbero il caso
   peggiore in assoluto (molti bordi da arrotondare con poche componenti).
3. **Le code pesanti sono state valutate sull'ISE.** Se la metrica di interesse fosse la
   qualità della coda in sé — per esempio per stimare quantili estremi — la conclusione
   sarebbe diversa: lì il nucleo logistico è effettivamente inadeguato.
4. **n = 500.** Il compromesso su J dipende dal budget: con più dati la capacità maggiore
   costerebbe meno.
5. **Nessuna delle distribuzioni provate è stata scelta dal docente.** Sono la nostra ipotesi
   di cosa potrebbe capitare.

---

## 5. Frasi pronte per il report

> La procedura è stata calibrata su misture di normali. Per verificarne la portata l'abbiamo
> applicata a dodici distribuzioni di altre famiglie, fra cui uniformi, esponenziali,
> lognormali e Student-t fino a un grado di libertà, dove né la varianza né la media esistono.
> Il comportamento è risultato stabile in tutti i casi tranne uno.

> Le code pesanti, che ci aspettavamo problematiche perché il nucleo logistico decade in modo
> esponenziale, si sono rivelate più semplici delle misture gaussiane: l'errore quadratico
> integrato è dominato dalla regione centrale, e una Student-t è un singolo picco liscio. La
> regola per il dominio di valutazione si è mostrata robusta per una ragione che non avevamo
> previsto: su code pesanti la deviazione standard campionaria cresce, ma l'ampiezza del
> campione cresce più in fretta, e il margine proporzionale a σ̂ diventa trascurabile.

> Il limite reale è un altro, ed è strutturale: una mistura di funzioni di ripartizione
> logistiche ha densità strettamente positiva su tutta la retta, e non può quindi riprodurre
> un salto di densità. Su distribuzioni a supporto compatto l'errore cresce di un fattore fra
> 3 e 14. Abbiamo verificato se aumentare il numero di componenti risolvesse: dimezza l'errore
> sui bordi ma lo peggiora di circa un terzo sulle misture multimodali lisce, che sono il caso
> di interesse. Abbiamo quindi mantenuto dodici componenti e dichiarato il limite.
