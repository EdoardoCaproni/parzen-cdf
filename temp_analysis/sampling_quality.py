"""Qualita' del generatore usato per costruire i dataset di sviluppo.

Perche' serve, nonostante il generatore non sia "in pipeline": i campioni su cui sviluppiamo
e misuriamo tutto li produciamo noi. Un generatore distorto contaminerebbe ogni conclusione
a valle senza dare segnali. La verifica non riguarda quindi la circolarita' (che non e' un
problema: campioni i.i.d. da F non portano informazione su F oltre a quella statistica), ma
la CORRETTEZZA DISTRIBUTIVA e l'INDIPENDENZA di cio' che generiamo.

Cosa si puo' e cosa non si puo' verificare, dichiarato in partenza:
  - NON verificabile dall'interno di Python: che NumPy usi davvero l'algoritmo Ziggurat per
    `Generator.standard_normal`. E' un fatto di implementazione, citabile dalla
    documentazione, non dimostrabile da noi.
  - VERIFICABILE, ed e' cio' che conta: che l'uscita sia distribuita come la mistura
    richiesta, che le componenti siano estratte con le probabilita' giuste, che non ci sia
    struttura seriale, e che la scelta dei semi non introduca regolarita'.

Test:
  T1  aderenza alla CDF vera (Kolmogorov-Smirnov) su molte repliche
  T2  uniformita' dei p-value di T1: e' il test corretto, non "quanti rifiuti ho"
  T3  frequenze delle componenti contro i pesi (chi-quadro)
  T4  indipendenza seriale (autocorrelazione a vari lag + test dei run)
  T5  i semi piccoli e consecutivi non introducono struttura (confronto con semi a 64 bit)
  T6  campionamento ancestrale vs inversione numerica della CDF
"""

import numpy as np
from scipy import stats
from scipy.special import erf

# ------------------------------------------------------------------ la mistura di prova

W = np.array([.3, .5, .2])
M = np.array([-2.0, 1.0, 4.0])
S = np.array([.5, 1.0, .3])


def mix_cdf(x):
    z = (np.asarray(x, float)[..., None] - M) / (S * np.sqrt(2.0))
    return (0.5 * (1.0 + erf(z))) @ W


def sample_ancestral(n, rng):
    """Il metodo del repo: scelta categorica della componente + normale dalla componente."""
    i = rng.choice(3, size=n, p=W)
    return rng.normal(M[i], S[i]), i


def sample_inverse(n, rng, tol=1e-12):
    """Alternativa: inversione numerica della CDF della mistura (bisezione)."""
    u = rng.random(n)
    lo = np.full(n, -20.0)
    hi = np.full(n, 20.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        go = mix_cdf(mid) < u
        lo = np.where(go, mid, lo)
        hi = np.where(go, hi, mid)
    return 0.5 * (lo + hi)


print("=" * 92)
print("T1/T2  ADERENZA ALLA DISTRIBUZIONE VERA")
print("   Un generatore corretto non deve solo 'passare' il test KS: i suoi p-value devono")
print("   essere UNIFORMI in [0,1]. Un generatore troppo 'bravo' (p-value tutti alti) sarebbe")
print("   sospetto quanto uno che rifiuta troppo.")
for n in (500, 2000):
    p = np.array([stats.kstest(sample_ancestral(n, np.random.default_rng(s))[0], mix_cdf).pvalue
                  for s in range(300)])
    ks_u = stats.kstest(p, "uniform")
    print(f"   n={n:>5}: rifiuti al 5% = {np.mean(p < 0.05):.3f} (atteso 0.050) | "
          f"p-value mediano = {np.median(p):.3f} (atteso 0.500) | "
          f"uniformita' dei p-value: p = {ks_u.pvalue:.3f}")

print()
print("=" * 92)
print("T3  FREQUENZE DELLE COMPONENTI contro i pesi (chi-quadro)")
tot = np.zeros(3)
for s in range(200):
    _, idx = sample_ancestral(500, np.random.default_rng(s))
    tot += np.bincount(idx, minlength=3)
exp = tot.sum() * W
chi = stats.chisquare(tot, exp)
print(f"   osservate {tot.astype(int)}  attese {exp.astype(int)}  ->  chi2 p = {chi.pvalue:.3f}")
print(f"   frequenze relative {np.round(tot / tot.sum(), 4)}  contro pesi {W}")

print()
print("=" * 92)
print("T4  INDIPENDENZA SERIALE")
print("   Se ci fosse struttura nello stream, i campioni non sarebbero i.i.d. e ogni misura")
print("   di errore basata su repliche indipendenti sarebbe falsata.")
big = sample_ancestral(200_000, np.random.default_rng(12345))[0]
z = (big - big.mean()) / big.std()
for lag in (1, 2, 5, 10, 100):
    r = float(np.corrcoef(z[:-lag], z[lag:])[0, 1])
    lim = 1.96 / np.sqrt(z.size - lag)          # banda al 95% sotto ipotesi di indipendenza
    flag = "ok" if abs(r) < lim else "SOSPETTO"
    print(f"   autocorrelazione lag {lag:>3}: {r:+.5f}   banda 95% +-{lim:.5f}   {flag}")
med = np.median(big)
sgn = big > med
runs = 1 + int(np.sum(sgn[1:] != sgn[:-1]))
n1, n2 = int(sgn.sum()), int((~sgn).sum())
mu = 2 * n1 * n2 / (n1 + n2) + 1
var = (mu - 1) * (mu - 2) / (n1 + n2 - 1)
print(f"   test dei run: osservati {runs}, attesi {mu:.0f}, z = {(runs - mu) / np.sqrt(var):+.2f}"
      f"   (|z| < 1.96 = ok)")

print()
print("=" * 92)
print("T5  I SEMI PICCOLI E CONSECUTIVI (0,1,2,...) INTRODUCONO STRUTTURA?")
print("   Tutti i nostri esperimenti usano default_rng(0..k). Se semi vicini producessero")
print("   stream correlati, le 'repliche indipendenti' non lo sarebbero.")
p_small = np.array([stats.kstest(sample_ancestral(500, np.random.default_rng(s))[0],
                                mix_cdf).pvalue for s in range(300)])
rng_big = np.random.default_rng(999)
seeds_big = rng_big.integers(0, 2**63, size=300)
p_big = np.array([stats.kstest(sample_ancestral(500, np.random.default_rng(int(s)))[0],
                              mix_cdf).pvalue for s in seeds_big])
print(f"   semi 0..299   : rifiuti {np.mean(p_small < .05):.3f}  mediana {np.median(p_small):.3f}")
print(f"   semi a 64 bit : rifiuti {np.mean(p_big < .05):.3f}  mediana {np.median(p_big):.3f}")
print(f"   le due distribuzioni di p-value sono distinguibili?  KS a due campioni: "
      f"p = {stats.ks_2samp(p_small, p_big).pvalue:.3f}  (alto = indistinguibili = ok)")
m1 = np.array([sample_ancestral(500, np.random.default_rng(s))[0].mean() for s in range(300)])
print(f"   correlazione fra media campionaria e indice del seme: "
      f"{np.corrcoef(np.arange(300), m1)[0, 1]:+.4f}  (atteso ~0)")

print()
print("=" * 92)
print("T6  ANCESTRALE vs INVERSIONE NUMERICA DELLA CDF")
print("   L'ancestrale e' ESATTO: sceglie la componente e campiona da una normale esatta.")
print("   L'inversione richiede una radice numerica, quindi e' esatta solo entro tolleranza.")
a = sample_ancestral(50_000, np.random.default_rng(7))[0]
b = sample_inverse(50_000, np.random.default_rng(7))
print(f"   KS ancestrale vs CDF vera : {stats.kstest(a, mix_cdf).statistic:.5f} "
      f"(p = {stats.kstest(a, mix_cdf).pvalue:.3f})")
print(f"   KS inversione  vs CDF vera : {stats.kstest(b, mix_cdf).statistic:.5f} "
      f"(p = {stats.kstest(b, mix_cdf).pvalue:.3f})")
print(f"   i due campioni sono distinguibili fra loro? KS a due campioni: "
      f"p = {stats.ks_2samp(a, b).pvalue:.3f}")
print("   -> entrambi corretti; l'ancestrale e' preferibile perche' esatto per costruzione")
print("      e O(n) invece che O(n * iterazioni di bisezione).")
