"""Figure del report, prodotte dalla pipeline consegnata.

Le figure devono mostrare cio' che il metodo produce davvero, non risultati di studi
precedenti: sono generate chiamando ``run_from_samples`` come lo chiamerebbe chiunque.

    python scripts/report_figures.py

Scrive in results/: report3_estimate.png, report3_window.png,
report3_diagnostics.png
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parzen_cdf import data, parzen
from parzen_cdf.diagnostics import lscv_score, report_domain
from parzen_cdf.estimate import run_from_samples

N, SEED = 500, 0
MIX = data.asymmetric_trimodal()

# --------------------------------------------------------------- figura 1: cosa produce
x = MIX.sample(N, np.random.default_rng(SEED))
est = run_from_samples(x, epochs=6000, seed=0)
lo, hi = est.domain()
g = np.linspace(lo, hi, 2001)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

ax1.plot(g, MIX.pdf(g), color="0.55", lw=2.4, label="true density")
ax1.plot(g, est.pdf(g), color="C0", lw=1.7, label="estimate")
ax1.plot(x, np.full_like(x, -0.006), "|", color="0.35", ms=7, alpha=0.5)
ax1.set_ylim(bottom=-0.015)
ax1.set_title(f"density  (n = {N}, J = {est.model.n_components})")
ax1.set_xlabel("x")
ax1.legend(frameon=False)

emp = np.searchsorted(np.sort(x), g, side="right") / x.size
ax2.step(g, emp, where="post", color="0.8", lw=1.0, label="empirical CDF")
ax2.plot(g, MIX.cdf(g), color="0.55", lw=2.4, label="true CDF")
ax2.plot(g, est.cdf(g), color="C0", lw=1.7, label="estimate")
ax2.set_ylim(-0.03, 1.03)
ax2.set_title("distribution function")
ax2.set_xlabel("x")
ax2.legend(frameon=False, loc="lower right")

for ax in (ax1, ax2):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/report3_estimate.png", dpi=130)
plt.close(fig)
print("results/report3_estimate.png")

# --------------------------------------------------------------- figura 2: la finestra
# La curva dell'errore contro la finestra, con le scelte dei vari selettori sopra.
# Serve a mostrare due cose: che la curva e' asimmetrica, e dove cadono le regole.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
casi = [("trimodal", MIX),
        ("five narrow separated modes",
         data.GaussianMixture1D([.2] * 5, [-8, -4, 0, 4, 8], [.35] * 5))]

for ax, (nome, mix) in zip(axes, casi):
    xs = mix.sample(N, np.random.default_rng(SEED))
    lo_, hi_ = report_domain(xs)
    gg = np.linspace(lo_, hi_, 2001)
    tp = mix.pdf(gg)
    sd = xs.std(ddof=1)
    hs = np.geomspace(0.01 * sd, 1.5 * sd, 60)
    ise = [float(np.trapezoid((parzen.parzen_pdf(gg, xs, h) - tp) ** 2, gg)) for h in hs]
    ax.loglog(hs, ise, color="0.3", lw=1.6)

    scelte = [("LSCV", parzen.lscv_bandwidth(xs), "C0"),
              ("Silverman", parzen.silverman_bandwidth(xs), "C1"),
              (r"$1.5\,\hat{\sigma}/\sqrt{n}$", 1.5 * sd / np.sqrt(N), "C3")]
    for etichetta, h, colore in scelte:
        ax.axvline(h, color=colore, ls="--", lw=1.3)
        ax.annotate(etichetta, xy=(h, max(ise)), xytext=(0, -4),
                    textcoords="offset points", rotation=90, ha="right", va="top",
                    color=colore, fontsize=8)
    ax.axvline(hs[int(np.argmin(ise))], color="0.1", lw=1.0)
    ax.annotate("oracle", xy=(hs[int(np.argmin(ise))], min(ise)), xytext=(4, 6),
                textcoords="offset points", fontsize=8)
    ax.set_title(nome)
    ax.set_xlabel("window $h$")
    ax.set_ylabel("integrated squared error")
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("results/report3_window.png", dpi=130)
plt.close(fig)
print("results/report3_window.png")


# --------------------------------------------------------------- figura 3: la diagnostica
# Il capitolo 8 dice che il confronto con la CDF empirica e' anticorrelato con l'errore vero
# e premia la finestra piu' stretta possibile. Detto in una tabella di correlazioni di rango
# resta un numero; visto come curve e' immediato, perche' una scende dove l'altra sale.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
casi3 = [("trimodal", MIX),
         ("five narrow separated modes",
          data.GaussianMixture1D([.2] * 5, [-8, -4, 0, 4, 8], [.35] * 5))]

for ax, (nome, mix) in zip(axes, casi3):
    xs = mix.sample(N, np.random.default_rng(SEED))
    lo_, hi_ = report_domain(xs)
    gg = np.linspace(lo_, hi_, 2001)
    tp = mix.pdf(gg)
    ecdf = np.searchsorted(np.sort(xs), gg, side="right") / xs.size
    sd = xs.std(ddof=1)
    hs = np.geomspace(0.006 * sd, 1.2 * sd, 55)

    # L'errore vero e' quello sulla densita': sulla CDF, con h che tende a zero, la stima
    # tende alla CDF empirica e l'errore si ferma sul pavimento statistico invece di
    # risalire, quindi il costo del sottolisciamento non si vedrebbe.
    vero, contro_ecdf, punteggio = [], [], []
    for h in hs:
        vero.append(float(np.trapezoid((parzen.parzen_pdf(gg, xs, h) - tp) ** 2, gg)))
        contro_ecdf.append(float(np.max(np.abs(parzen.parzen_cdf(gg, xs, h) - ecdf))))
        punteggio.append(float(lscv_score(xs, float(h))))

    # ognuna sulla propria scala: conta dove cade il minimo, non il valore
    def riscala(v):
        v = np.asarray(v, dtype=float)
        return (v - v.min()) / (v.max() - v.min())

    ax.semilogx(hs, riscala(vero), color="0.15", lw=1.8, label="density error against the truth (ISE)")
    ax.semilogx(hs, riscala(punteggio), color="C0", lw=1.5, label="LSCV score")
    ax.semilogx(hs, riscala(contro_ecdf), color="C3", lw=1.5, ls="--",
                label="KS against the empirical CDF")
    ax.axvline(hs[int(np.argmin(vero))], color="0.15", lw=1.0, alpha=0.6)
    ax.axvline(hs[int(np.argmin(punteggio))], color="C0", lw=1.0, alpha=0.6)
    ax.set_title(nome)
    ax.set_xlabel("window $h$")
    ax.set_ylabel("each curve rescaled to [0, 1]")
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(fontsize=8, loc="upper center", frameon=False)
fig.tight_layout()
fig.savefig("results/report3_diagnostics.png", dpi=130)
plt.close(fig)
print("results/report3_diagnostics.png")
