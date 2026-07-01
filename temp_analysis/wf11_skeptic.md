# wf11_skeptic — the adversarial case AGAINST any advantage of MLP-on-Parzen

**Brief:** argue the strongest case that MLP-on-Parzen has no real advantage, and pin down exactly
where the rationale is weakest. Measured on the same samples (n=1500, 2 seeds) against the strongest
honest alternative — fit the data **directly by maximum likelihood** (a Gaussian mixture by EM, model
order chosen by BIC), which *can* beat the KDE, unlike regressing it.

## The three-line indictment

1. **The net cannot beat its own target — by the project's own finding it is a faithful regressor.**
   Measured gap of the net's CDF to its Silverman-Parzen *label*: KS 0.019 / 0.051 / 0.019. So its
   accuracy ceiling is the KDE you had to build *before* training it. You already own a deployable
   estimate at label-construction time; the net adds no statistical information, only resampling noise.
2. **A direct MLE crushes it.** A GMM fit by EM+BIC on the *same* samples beats the net's CDF KS by
   **+47% to +81%**, and its pdf MSE by **10×–45×**. Regressing a KDE inherits the KDE's bias/variance;
   MLE does not — that is the whole point the brief flagged.
3. **Every structural perk the net claims is generic to "use a smooth parametric model", not to
   "regress a Parzen estimate".** A 9-parameter GMM is compact, queryable in O(k) independent of n
   (0.20 ms / 1000 pts), differentiable, exactly normalized, and analytically invertible for sampling —
   and it is fit *directly*, with no KDE detour to throw away.

## Numbers (mean over seeds)

CDF accuracy vs TRUTH — KS distance, lower is better:

| distribution | Silverman-Parzen (the net's label) | best-window Parzen (oracle) | **MLP-on-Parzen (net)** | **direct-MLE GMM** |
|---|---|---|---|---|
| symmetric_bimodal | 0.0539 | 0.0197 | 0.0534 | **0.0166** |
| asymmetric_trimodal | 0.0471 | 0.0205 | 0.0830 | **0.0161** |
| single_gaussian | 0.0284 | 0.0234 | 0.0339 | **0.0179** |

pdf accuracy vs TRUTH — MSE, lower is better:

| distribution | best-window Parzen | **MLP-on-Parzen (deriv.)** | **direct-MLE GMM** |
|---|---|---|---|
| symmetric_bimodal | 0.00010 | 0.00112 | **0.00005** |
| asymmetric_trimodal | 0.00017 | 0.00314 | **0.00007** |
| single_gaussian | 0.00009 | 0.00013 | **0.00002** |

The net's CDF vs **its own** Silverman-Parzen target (faithfulness): KS 0.0242 / 0.0505 / 0.0188.
It tracks the target it was given and that target is mediocre (Silverman over-smooths; even the
*oracle-window* Parzen at 0.020 KS loses to the GMM). Differentiating a CDF amplifies its wiggle, so
the net's **pdf** is the worst of the three on every case — the derivative route makes the density
*noisier*, not smoother, the opposite of the "denoising" claim (B).

Amortization rebuttal: GMM = 9 params, CDF+pdf query of 1000 pts in **0.20 ms**, n-independent.

## Where the rationale is weakest (the load-bearing failure)

- **The pipeline is self-dominated.** Step 3 builds the Parzen estimate to manufacture the labels.
  That estimate is *already* a complete, deployable CDF/pdf. Step 4's net, being a faithful regressor,
  can at best match it and in practice (trimodal, single Gaussian) does slightly worse than its own
  target after differentiation. The net is therefore a lossy compressor of an estimate you already
  hold — never a refiner of it.
- **Regress-the-KDE forecloses the only honest way to beat the KDE.** The literature's "neural density
  estimators beat KDE" results come from **maximum-likelihood** training on the data (normalizing
  flows, autoregressive nets, mixture density nets) — an objective that can move *off* the KDE. Fixing
  the labels to KDE values throws that away by construction. The approach adopts the neural-density
  machinery while discarding the one mechanism that makes it worth more than the KDE.
- **The cited perks aren't differentiators.** Compact/amortized/differentiable/normalized/sampleable
  are properties of any smooth parametric fit. A GMM (or a flow) has them too, fits directly, and wins
  on accuracy. So "MLP-on-Parzen" wins none of A/B/C/D/E *uniquely*.

## When is MLP-on-Parzen genuinely justified vs pointless?

**Pointless** (the common case): 1D, or any setting where (i) you only need point accuracy — KDE or a
GMM-MLE dominates; (ii) you want a compact/fast/sampleable estimate — a GMM-MLE gives all of it, better
and cheaper; (iii) you can afford MLE at all — then do MLE, not KDE-regression.

**Genuinely justified** (narrow): the construction's only non-redundant niche is the **multivariate
joint via a learned CDF differentiated to a density** (claim C), *and only when you specifically refuse
to do direct MLE* — e.g. you have an existing Parzen/empirical-CDF target object, want a single smooth
differentiable surrogate of it with O(params) query and built-in monotone constraints, and a likelihood
model is unavailable or undesired. Even there a normalizing flow trained by MLE is the stronger play; the
Parzen-regression route is the fallback you pick when you are *committed* to imitating a given KDE rather
than fitting the data. As a 1D density estimator, the advantage is **none**.

## Honest caveat to my own attack

GMM-MLE wins *here* because every test distribution is itself a Gaussian mixture — the parametric model
is correctly specified, an unfair home advantage. The fair reading is not "GMM is universally best" but
the weaker, sufficient claim: **a direct-likelihood smooth model can beat a KDE, and MLP-on-Parzen by
construction cannot beat the KDE it regresses.** That asymmetry is what sinks the advantage, and it does
not depend on the GMM being correctly specified — any direct-MLE family (a flow on a non-mixture target)
inherits the same freedom the KDE-regressor lacks.
