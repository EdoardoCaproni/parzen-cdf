"""I numeri del report che non provenivano da nessuno script salvato.

Erano stati calcolati al volo durante la stesura. L'appendice B promette che ogni numero
citato e' riproducibile, quindi devono avere anche loro un file di output.

    python temp_analysis/report_numbers.py > temp_analysis/report_numbers.txt
"""

import numpy as np
import torch

from parzen_cdf import data, parzen
from parzen_cdf.models import CDFNet, MixtureCDFNet

print("=" * 88)
print("CAP. 3  Il nucleo logistico e la sua primitiva")
u = np.linspace(-8, 8, 200_001)
pdf = parzen.KERNELS["logistic"].pdf(u)
cdf = parzen.KERNELS["logistic"].cdf(u)
print(f"  max|d(sigmoide)/du - densita' del nucleo| = "
      f"{np.max(np.abs(np.gradient(cdf, u) - pdf)):.2e}")

print()
print("  deviazione standard dei nuclei disponibili")
for nome, k in parzen.KERNELS.items():
    print(f"    {nome:14} {k.std:.4f}")
print(f"  fattore di correzione logistico/gaussiano: {parzen.LOGISTIC_KERNEL_STD:.4f}")

print()
print("  pavimento statistico E[KS] della CDF empirica")
print(f"    sqrt(pi/2)*ln2 = {float(np.sqrt(np.pi / 2) * np.log(2)):.6f}")
print(f"    a n = 500      = {float(np.sqrt(np.pi / 2) * np.log(2) / np.sqrt(500)):.4f}")

print()
print("=" * 88)
print("CAP. 3  Il costo di non correggere Silverman per il nucleo logistico")
print("        (trimodale, n = 500, media su 10 semi)")
MIX = data.asymmetric_trimodal()
g = np.linspace(-5, 7, 4001)
tc, tp = MIX.cdf(g), MIX.pdf(g)
acc = {"silverman": [], "corretto": []}
for s in range(10):
    x = MIX.sample(500, np.random.default_rng(s))
    for nome, h in (("silverman", parzen.silverman_bandwidth(x)),
                    ("corretto", parzen.variance_matched_bandwidth(x))):
        acc[nome].append((float(np.max(np.abs(parzen.parzen_cdf(g, x, h) - tc))),
                          float(np.trapezoid((parzen.parzen_pdf(g, x, h) - tp) ** 2, g))))
for nome in ("silverman", "corretto"):
    m = np.mean(acc[nome], axis=0)
    print(f"  {nome:12} KS {m[0]:.4f}   ISE {m[1]:.4f}")

print()
print("=" * 88)
print("CAP. 5  Monotonia sotto parametri casuali estremi")
print("        1000 estrazioni, 5000 punti, tolleranza 1e-6 (float32)")
xs = MIX.sample(300, np.random.default_rng(0))
gt = torch.linspace(-40, 40, 5000)
for nome, fabbrica in (
        ("MLP, pesi liberi", lambda: CDFNet(1, (8,), "sigmoid")),
        ("MLP, pesi positivi", lambda: CDFNet(1, (8,), "sigmoid", monotone=True)),
        ("mistura, J = 12", lambda: MixtureCDFNet(12).init_from_samples(xs))):
    torch.manual_seed(7)
    cattivi = 0
    for _ in range(1000):
        rete = fabbrica()
        with torch.no_grad():
            for p in rete.parameters():
                p.add_(torch.randn_like(p) * 3.0)
            if np.any(np.diff(rete(gt).numpy()) < -1e-6):
                cattivi += 1
    print(f"  {nome:22} non monotone: {cattivi}/1000")

print()
print("=" * 88)
print("CAP. 5  Equivarianza: il limite e' la precisione della standardizzazione")
print("        La sottrazione (x - mu) in float32 perde precisione relativa al crescere")
print("        dell'offset, perche' i due termini diventano di grandezza simile.")
xr = np.random.default_rng(0).normal(0, 2, 500)
for shift in (0, 100, 1000, 10000):
    xs = xr + shift
    ts = torch.as_tensor(xs, dtype=torch.float32)
    mu, sd = ts.median(), ts.std(unbiased=True)
    z32 = ((ts - mu) / sd).numpy().astype(np.float64)
    z64 = (xs - float(mu)) / float(sd)          # stessa mu e sd, sottrazione in float64
    print(f"  offset {shift:>6}: errore su z = {np.max(np.abs(z32 - z64)):.1e}")

print()
print("  Effetto sulla stima consegnata (KS contro la verita', n = 500)")
from parzen_cdf.estimate import run_from_samples          # noqa: E402

g0 = np.linspace(-5, 7, 2001)
tc0 = MIX.cdf(g0)
print(f"  {'trasformazione':>22} {'seme 0':>9} {'seme 1':>9} {'seme 2':>9}")
for a, b, lab in ((1, 0, "nessuna"), (1, 100, "traslazione +100"),
                  (1, 1000, "traslazione +1000"), (50, 0, "scala x50"),
                  (1, 10000, "traslazione +10000")):
    riga = f"  {lab:>22} "
    for s in range(3):
        x = MIX.sample(500, np.random.default_rng(s)) * a + b
        est = run_from_samples(x, epochs=6000, seed=0)
        riga += f"{float(np.max(np.abs(est.cdf(g0 * a + b) - tc0))):>9.4f}"
    print(riga)
