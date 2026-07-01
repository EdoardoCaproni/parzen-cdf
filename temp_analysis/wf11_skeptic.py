"""wf11_skeptic -- the adversarial case AGAINST any advantage of MLP-on-Parzen.

The claimed advantages (see 00_plan_and_reflection.md) are: A/F amortization, B denoising,
C multivariate, D constraints, E sampling. This script attacks the load-bearing ones with the
strongest honest alternative: fit the data DIRECTLY by maximum likelihood (a GMM by EM), which --
unlike regressing the KDE -- can actually beat the KDE. We measure, on the SAME samples:

  (1) ACCURACY CEILING. MLP-on-Parzen is, by the project's own 1D finding, a faithful regressor of
      its Parzen target: its accuracy ceiling IS the Parzen estimate. A direct-MLE GMM has no such
      ceiling. Compare true-CDF KS and true-pdf MSE: net-on-Parzen vs best-window Parzen vs GMM-MLE.

  (2) THE AMORTIZATION CLAIM IS NOT UNIQUE TO THE NET. A GMM-MLE is also a compact (O(k) params),
      n-independent, closed-form, differentiable, analytically-sampleable, exactly-normalized CDF/pdf
      -- every structural property A/D/E claim for the net, but it also beats the KDE on accuracy and
      it is fit directly, not bootstrapped off a KDE we then throw away. So the net's structural perks
      are real but generic; they are perks of "use a smooth parametric model", not of "regress a KDE".

  (3) THE PIPELINE IS STRICTLY DOMINATED in its own logic: you must build the Parzen estimate to make
      the labels, so you ALREADY HAVE a deployable density estimate before the net is trained. The net
      can at best reproduce it. So the marginal value of the net over just-deploy-the-Parzen is only
      the structural perks of (2), all of which a direct MLE delivers WITHOUT the KDE detour and WITH a
      better fit. We quantify "net never beats its own target".

Lightweight: n<=2000, one mixture family, 3 seeds.
"""
import time
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from parzen_cdf import data, metrics, parzen
from parzen_cdf.models import CDFNet
from parzen_cdf.training import TrainConfig, make_sample_training_set, train_cdf, set_seed, density_from_cdf

GRID = np.linspace(-7, 9, 1500)
GRID_T = torch.as_tensor(GRID, dtype=torch.float32)
N = 1500
SEEDS = range(2)
WINDOW_SCALES = [0.3, 0.5, 0.7, 1.0, 1.4]   # x Silverman, to find the BEST-window Parzen (strong baseline)


def _erf(z):
    # vectorized erf via torch (numpy has no erf without scipy.special, keep it dependency-light)
    return torch.special.erf(torch.as_tensor(z, dtype=torch.float64)).numpy()


def gmm_cdf_pdf(gm, grid):
    """Closed-form CDF and pdf of an sklearn 1D GaussianMixture on a grid."""
    w = gm.weights_
    mu = gm.means_.ravel()
    sd = np.sqrt(gm.covariances_.ravel())
    cdf = np.zeros_like(grid)
    pdf = np.zeros_like(grid)
    for wk, mk, sk in zip(w, mu, sd):
        cdf += wk * 0.5 * (1.0 + _erf((grid - mk) / (sk * np.sqrt(2.0))))
        pdf += wk * np.exp(-0.5 * ((grid - mk) / sk) ** 2) / (sk * np.sqrt(2.0 * np.pi))
    return cdf, pdf


def run_one(mix, label):
    TRUE_CDF = mix.cdf(GRID)
    TRUE_PDF = mix.pdf(GRID)

    rows = []
    for seed in SEEDS:
        s = mix.sample(N, np.random.default_rng(seed))
        h0 = parzen.silverman_bandwidth(s)

        # --- best-window Parzen (a STRONG KDE baseline: we let it pick its best window vs truth) ---
        best_pk_cdf, best_h = np.inf, h0
        for sc in WINDOW_SCALES:
            h = sc * h0
            pk = parzen.parzen_cdf(GRID, s, h)
            ks = metrics.ks_distance(TRUE_CDF, pk)
            if ks < best_pk_cdf:
                best_pk_cdf, best_h = ks, h
        parzen_cdf_ks = best_pk_cdf
        parzen_pdf_mse = metrics.mse(TRUE_PDF, parzen.parzen_pdf(GRID, s, best_h))
        # also the Silverman-default Parzen (what the pipeline actually uses for labels)
        silv_pk = parzen.parzen_cdf(GRID, s, h0)
        silv_cdf_ks = metrics.ks_distance(TRUE_CDF, silv_pk)

        # --- the project's net: regress the Silverman-Parzen CDF on the data points only -----------
        set_seed(seed)
        inp, tgt = make_sample_training_set(s, h0)
        net = CDFNet(1, (32,), activation="sigmoid", monotone=False)
        net, _ = train_cdf(net, inp, tgt, TrainConfig(optimizer="adam", lr=0.03, epochs=2500, seed=seed))
        net_cdf = net(GRID_T).detach().numpy()
        net_cdf_ks_truth = metrics.ks_distance(TRUE_CDF, net_cdf)
        net_cdf_ks_target = metrics.ks_distance(silv_pk, net_cdf)  # gap to its OWN target
        net_pdf = density_from_cdf(net, GRID_T).detach().numpy()
        net_pdf_mse = metrics.mse(TRUE_PDF, net_pdf)

        # --- direct MLE: GaussianMixture by EM on the SAME samples (model selection by BIC, k<=5) ---
        best_gm, best_bic = None, np.inf
        for k in (1, 2, 3, 4, 5):
            gm = GaussianMixture(n_components=k, covariance_type="full",
                                 random_state=seed, n_init=2).fit(s.reshape(-1, 1))
            b = gm.bic(s.reshape(-1, 1))
            if b < best_bic:
                best_bic, best_gm = b, gm
        gmm_cdf, gmm_pdf = gmm_cdf_pdf(best_gm, GRID)
        gmm_cdf_ks = metrics.ks_distance(TRUE_CDF, gmm_cdf)
        gmm_pdf_mse = metrics.mse(TRUE_PDF, gmm_pdf)
        gmm_k = best_gm.n_components

        rows.append(dict(seed=seed,
                         silv_cdf_ks=silv_cdf_ks,
                         parzen_cdf_ks=parzen_cdf_ks, parzen_pdf_mse=parzen_pdf_mse,
                         net_cdf_ks_truth=net_cdf_ks_truth, net_cdf_ks_target=net_cdf_ks_target,
                         net_pdf_mse=net_pdf_mse,
                         gmm_cdf_ks=gmm_cdf_ks, gmm_pdf_mse=gmm_pdf_mse, gmm_k=gmm_k))

    def m(key):
        return float(np.mean([r[key] for r in rows]))

    print(f"\n===== {label} (n={N}, {len(list(SEEDS))} seeds) =====")
    print("  CDF accuracy vs TRUTH (KS, lower=better):")
    print(f"    Silverman-Parzen (the net's label) : {m('silv_cdf_ks'):.4f}")
    print(f"    BEST-window Parzen (oracle window)  : {m('parzen_cdf_ks'):.4f}")
    print(f"    MLP-on-Parzen (net vs truth)        : {m('net_cdf_ks_truth'):.4f}")
    print(f"    direct-MLE GMM (EM+BIC)             : {m('gmm_cdf_ks'):.4f}   (mean k={m('gmm_k'):.1f})")
    print("  pdf accuracy vs TRUTH (MSE, lower=better):")
    print(f"    BEST-window Parzen pdf              : {m('parzen_pdf_mse'):.5f}")
    print(f"    MLP-on-Parzen pdf (derivative)      : {m('net_pdf_mse'):.5f}")
    print(f"    direct-MLE GMM pdf                  : {m('gmm_pdf_mse'):.5f}")
    print("  the net vs its OWN target (KS, ~0 => faithful copy, no statistical value added):")
    print(f"    net-CDF vs its Silverman-Parzen target : {m('net_cdf_ks_target'):.4f}")
    return dict(label=label,
                silv=m('silv_cdf_ks'), bestparzen=m('parzen_cdf_ks'),
                net=m('net_cdf_ks_truth'), gmm=m('gmm_cdf_ks'),
                netpdf=m('net_pdf_mse'), gmmpdf=m('gmm_pdf_mse'),
                parzenpdf=m('parzen_pdf_mse'),
                net_vs_target=m('net_cdf_ks_target'))


def amortization_rebuttal():
    """The amortization perk (A) is NOT unique to the net: a GMM is also compact, O(k), n-independent,
    closed-form, differentiable, exactly normalized, analytically sampleable. Show its query cost."""
    print("\n===== AMORTIZATION REBUTTAL: GMM is also compact & n-independent =====")
    mix = data.asymmetric_trimodal()
    s = mix.sample(2000, np.random.default_rng(0))
    gm = GaussianMixture(n_components=3, covariance_type="full", random_state=0, n_init=2).fit(s.reshape(-1, 1))
    n_params_gmm = gm.n_components * 3  # weight, mean, var per component
    q = np.linspace(-7, 9, 1000)

    def gmm_query():
        gmm_cdf_pdf(gm, q)
    gmm_query()
    t0 = time.perf_counter()
    for _ in range(50):
        gmm_query()
    t_gmm = (time.perf_counter() - t0) / 50
    print(f"  GMM: {n_params_gmm} params, query (1000 pts) CDF+pdf = {1000*t_gmm:.3f} ms,"
          f" n-independent, EXACT normalization, analytic inverse-CDF sampling.")
    print("  => every structural perk claimed for the net (compactness, O(1)-in-n query, smoothness,")
    print("     differentiability, sampling, exact constraints) a direct GMM-MLE ALSO has, plus it")
    print("     beats the KDE and needs no KDE detour. The net's perks are perks of 'parametric smooth")
    print("     model', not of 'regress a Parzen estimate'.")


if __name__ == "__main__":
    results = []
    for mk, lbl in [(data.symmetric_bimodal, "symmetric_bimodal"),
                    (data.asymmetric_trimodal, "asymmetric_trimodal"),
                    (data.single_gaussian, "single_gaussian")]:
        results.append(run_one(mk(), lbl))
    amortization_rebuttal()

    print("\n\n========== VERDICT NUMBERS ==========")
    for r in results:
        net_beats_target = r['net_vs_target']
        gmm_vs_net = (r['net'] - r['gmm']) / r['net'] * 100
        print(f"{r['label']}: net-CDF-KS={r['net']:.4f}  GMM-CDF-KS={r['gmm']:.4f}"
              f"  (GMM {'beats' if r['gmm']<r['net'] else 'loses to'} net by {gmm_vs_net:+.0f}%);"
              f"  net-vs-own-target KS={net_beats_target:.4f} (faithful copy);"
              f"  net-pdf-MSE={r['netpdf']:.5f} vs GMM-pdf-MSE={r['gmmpdf']:.5f}")
