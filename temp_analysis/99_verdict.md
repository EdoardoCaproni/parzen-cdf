# Verdict: the advantage of training an MLP on a Parzen-window estimate

## 1. Bottom line

Training a sigmoidal MLP on a logistic-Parzen CDF buys **no statistical-accuracy advantage** in 1D: across every test (oracle-bandwidth denoising, small-n sample efficiency, tails, sampling) the net is a faithful regressor of its Parzen target and never beats it, and a direct-likelihood model (GMM/flow) beats both. The genuine, defensible wins are **structural, not accuracy**: a compact, constant-cost, n-independent query surface (amortization + compression), a free-validity density via the CDF route (mass = 1 and non-negativity for one cheap rectify), and the Sklar/copula modeling economy when the dependence is simple. Use it as a *distilled, deployable surrogate of a KDE you already trust*, not as a better estimator.

## 2. Where the advantage is genuine (ranked, with numbers)

**1. Free-validity via the CDF route (moderate).** The decisive, tuning-free win. `rectify_cdf` forces mass = exactly 1.00000 and zero negativity for every distribution; the direct-pdf route's global "integrate to 1" constraint is the hard one, with native mass drifting 1.11 -> 1.33 -> 1.40 as the density gets harder, and every repair is lossy (soft penalty ~7x worse pdf-MSE and high-variance; post-hoc renorm ~1.6x worse, silently invalid pre-division). Structural, would widen in higher d where the pdf normalizer is an intractable d-dim integral. (wf04)

**2. Amortization + compression (moderate).** Net query cost is constant in n (0.097 ms, width-32, uncontended) vs Parzen's O(n) (8.6 ms at 1k -> 1173 ms at 100k); ~12,000x at n=100k *against naive KDE*, and ~1031x storage compression (97 floats vs n). Param count grows only +32/dim. HONEST LIMITS: it is a serving-cost win, not accuracy; it never pits the net against fast KDE (KD-trees, FGT, FFT/binned-KDE) that exist to kill the O(n) cost, so the headline magnitude is inflated; for a one-shot single query Parzen is strictly cheaper (no training). The dimension argument is directionally solid but only tested to D=3 on a separable Gaussian. (wf01)

**3. Sklar/copula modeling economy (moderate, medium confidence).** Decomposing a 2D joint into 1D marginal nets + a parametric copula beat the monolithic 2D net in all four cases. Decisive evidence is the copula-on-TRUE-marginals floor: sup 0.0003 (Gaussian) / 0.005 (independence) -- near machine-zero when the copula family matches the truth, so the estimate is two strong 1D fits + one interpretable parameter. CONDITIONAL: when the dependence is genuinely non-Gaussian the floor is irreducible (0.091 mix_diag, 0.027 mix_cross) and the monolithic net is close behind; the joint net's poor showing is partly an underfitting artifact. (wf09)

**4. Structural capabilities, real but non-unique and dominated (capability_only).** Compact differentiable monotone generator for inverse-CDF sampling (97 floats, 20.6x compression, O(depth) inversion, no sample scan) -- but sampling KS to truth ties Parzen (0.0332 vs 0.0279) and a 9-param GMM owns the same perks. Bounded sigmoid output guarantees a valid [0,1] monotone CDF arbitrarily far out (0.00e+00 non-monotonicity) -- a robustness property, paid for with a residual tail offset and a fat fake pdf tail. Smoothing makes a CDF differentiable at all (the empirical CDF differentiates to Dirac spikes, 613/6859 nonzero cells) -- but that capability is the Parzen/KDE's, not the net's. (wf08, wf05, wf03, wf10)

## 3. Where there is NO advantage

- **1D point accuracy / denoising (none).** Converged net's best-over-windows CDF is strictly worse than oracle-bandwidth Parzen on all three distributions, every scale: KS 0.0142/0.0148/0.0177 (Parzen) vs 0.0183/0.0196/0.0622 (net). The 800-epoch "net wins" result was an underfitting artifact (net-vs-target KS 0.107 -> 0.008 by 2500 ep). (wf02)
- **Small-n sample efficiency (none).** Tie on the single Gaussian (+/-1% KS), strictly worse on the bimodal at every n (KS -1.4% to -7.2%, MSE +20-37%). The dumb empirical step CDF even beats Parzen on the bimodal (Silverman over-smooths). (wf06)
- **Tails (none; mild liability).** Parzen wins tail CDF accuracy ~2x (moderate tail) to 5-20x (far tail), reaches 0/1 to machine precision; the net plateaus at a ~2e-4..1e-3 offset and has a spuriously fat pdf tail (5.5e-6 at x=16 vs true 1e-56). (wf05)
- **3D joint density (none + a defect).** Net's third mixed partial is 4.3x worse MSE than product-KDE and does not integrate to 1 (Riemann 1.70 -> 70% mass error, the multivariate monotonicity defect). KDE delivers the smooth density more accurately, normalized. (NB: the original "net 6.3x slower" cost claim was reversed on rerun -- net is ~10x faster -- but a faster route to a broken output is no advantage.) (wf03)
- **Beating a KDE at all (none, by construction).** Regressing fixed KDE labels makes the KDE the accuracy ceiling. A direct-MLE GMM beats the net 47-81% on CDF KS and 6-45x on pdf MSE while owning every structural perk. The net cannot beat the target it imitates; a likelihood objective can. (wf11)

## 4. When MLP-on-Parzen is justified vs pointless

**Justified when ALL hold:**
- You will query the estimate **many times** (amortize the training cost) or must **ship/store** the model compactly, AND
- You are committed to producing a single smooth, differentiable, monotone surrogate of an **existing KDE/empirical-CDF target you already trust**, AND
- You need **guaranteed validity** (mass=1, non-negative) without a normalization penalty -- especially in higher d where the pdf normalizer is intractable, AND
- You are **not** free to fit a likelihood model directly (no access to the raw data, or a hard requirement to match a specific KDE).

**Pointless when ANY hold:**
- You want a *better* density/CDF estimate (it cannot beat its teacher), OR
- It is a one-shot / few-query problem (Parzen is cheaper, no training), OR
- You can fit a direct-MLE model (GMM/flow) -- it dominates on accuracy AND keeps every structural perk, OR
- You need accurate tails or extreme quantiles (the net's tail is a liability), OR
- A fast KDE approximation (FFT/binned/KD-tree) already kills the query-cost problem.

## 5. Recommendation for this project

Frame the contribution **honestly as distillation/serving**, not estimation. The thesis claim should be: "a sigmoidal MLP faithfully regresses a Parzen CDF into a compact, constant-cost, validity-guaranteed, differentiable surrogate" -- and explicitly disclaim any point-accuracy beat. Lead with the two robust structural wins (free-validity CDF route; amortization/compression), present the copula economy as a conditional 2D result, and keep the capability claims (sampling, bounded extrapolation) clearly labeled as non-unique. Do not claim it beats the curse of dimensionality for *densities* (the CDF's n^(-1/2) rate does not survive differentiation). The strongest honest positioning is teacher-student: it is what you reach for when you must serve a trusted KDE cheaply and validly at scale, and nothing more.
