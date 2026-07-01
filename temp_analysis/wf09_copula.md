# wf09 — Copula (Sklar) decomposition in 2D

## The question

The 1D result is settled: the MLP is a *faithful regressor* of its logistic-Parzen
target, and Parzen/empirical CDFs already estimate a 1D CDF well. So in 1D there is
no accuracy advantage. The natural place to look for an advantage is **structure**:
in higher dimensions, does decomposing the joint into 1D marginals (where the method
is strong) plus a copula buy us anything over fitting one monolithic joint CDF?

**Sklar's theorem** says any joint CDF factors as
`F(x,y) = C(F_X(x), F_Y(y))`, where `C` is a copula (a CDF on `[0,1]^2` with uniform
marginals) carrying *all* the dependence and the `F_X, F_Y` carry *all* the marginal
shape. This is exactly a separation into "the part the method is good at" (1D
marginals) and "the dependence part."

We compare, against the analytic truth on a grid:

- **(a) single joint neural CDF** — one `CDFNet(in_dim=2)` trained in the strict
  regime on the 2D *product-logistic* Parzen CDF, `F̂(x,y)=1/n Σ σ((x−Xᵢ)/h)σ((y−Yᵢ)/h)`.
- **(b) marginals + copula** — two 1D `CDFNet`s on the marginals (strict regime),
  glued with either the **independence** copula `C(u,v)=u·v` or a **Gaussian** copula
  with `ρ` from the sample's Spearman correlation.

We also evaluate the Gaussian copula on the *true* marginals to separate "copula model
error" from "marginal estimation error."

## Test distributions

1. `gauss_corr0.6` — single bivariate normal, `ρ=0.6`. The Gaussian copula is the
   *true* copula here; this is the best case for route (b).
2. `gauss_indep` — single bivariate normal, `ρ=0`. The independence copula is exact.
3. `mix_diag` — two equal blobs at `(−2,−2)` and `(2,2)`, each isotropic. Marginals
   are bimodal; dependence is strongly **non-Gaussian** (a Gaussian copula cannot
   represent two separated lumps).
4. `mix_cross` — two equal blobs at the origin with opposite-sign correlation
   (`+1.6` and `−1.6` off-diagonal). Net linear correlation ≈ 0 but there is strong
   **tail/cross dependence** — the adversarial case for *any* simple parametric copula.

All: n = 1200, h = mean per-dim Silverman, MLP width 40 (joint) / 24 (marginals),
800 Adam epochs, single seed. Lightweight by design.

## Results

Sup-distance (max |F̂ − F_true|) on the grid. Lower is better. `marg-sup` is the
worst sup error of the two 1D marginal nets (the "method is strong here" baseline).

| case | marg-sup (x,y) | (a) JOINT net | (b) INDEP cop | (b) GAUSS cop | GAUSS on TRUE marg |
|------|----------------|---------------|---------------|---------------|---------------------|
| `gauss_corr0.6` (true cop = Gaussian) | 0.042 / 0.035 | **0.289** | 0.093 | **0.056** | 0.0003 |
| `gauss_indep` (true cop = indep)      | 0.036 / 0.034 | **0.113** | **0.065** | 0.066 | 0.005 |
| `mix_diag` (bimodal marg, non-Gauss dep) | 0.096 / 0.089 | 0.358 | 0.254 | **0.127** | 0.091 |
| `mix_cross` (zero corr, cross dep)    | 0.032 / 0.035 | **0.084** | 0.063 | **0.063** | 0.027 |

MSE tells the same story (e.g. `gauss_corr0.6`: joint 7.3e-3 vs gauss-copula 6.2e-4,
a ~12x reduction; gauss-copula on true marginals 4.1e-9, i.e. machine-zero).

## Reading

1. **The Sklar decomposition beats the monolithic joint net in every case.** The
   single 2D `CDFNet` on the product-logistic Parzen target is the *worst* estimator
   throughout (sup 0.084 – 0.358). It is hard to fit: with only `n` training points
   scattered in 2D and a product-of-logistics target, the joint net underfits at the
   budget where the 1D nets are already good. The marginal nets reach sup ≈ 0.03–0.10
   on the *same* budget. Sklar lets you spend the modelling effort where the method is
   strong (1D marginals) and attach a cheap dependence model on top.

2. **When the true copula is in the glued family, the decomposition is dramatically
   better and the residual error is *entirely* the marginals.** On `gauss_corr0.6`
   the Gaussian copula on the *estimated* marginals hits sup 0.056 (joint net: 0.289),
   and on the *true* marginals it hits 0.0003 — i.e. the copula model contributes ~0
   error; all 0.056 is marginal-net error. Same logic on `gauss_indep` with the
   independence copula (0.005 on true marginals). This is the clean win: 1 interpretable
   dependence parameter + two strong 1D fits.

3. **When the true copula is NOT in the family, the advantage shrinks and is honestly
   capped.** On `mix_diag` (two diagonal blobs) the Gaussian copula still beats the
   joint net (0.127 vs 0.358) but can no longer reach zero even on true marginals
   (0.091) — that 0.091 is irreducible *copula-model* error. On `mix_cross` (cross /
   tail dependence, zero linear correlation) the Gaussian and independence copulas are
   indistinguishable (both 0.063, because ρ≈0), the joint net is close behind (0.084),
   and even the copula-on-true-marginals floor (0.027) is not dramatically better. Here
   a simple parametric copula has little to offer and a richer dependence model would be
   needed.

## Honest limits

- This is a *small* test, single seed, one bandwidth rule. It establishes direction,
  not a benchmark.
- The "advantage" of the Sklar route is **conditional and structural**, not a free
  accuracy win:
  - It only helps when the *true* copula is in (or near) the parametric family you
    glue with. When it is (`gauss_corr`, `gauss_indep`), you exploit the strong 1D
    marginal estimator and a 1-parameter dependence model — far fewer effective
    degrees of freedom than a 2D MLP, and the dependence parameter is interpretable.
  - When the true copula is *not* in the family (`mix_diag`, `mix_cross`), the
    decomposition is **actively wrong** — it cannot represent the dependence no
    matter how good the marginals are. The monolithic joint net, fed a real 2D
    Parzen target, is not constrained to any copula family and can in principle
    follow the truth.
- So the genuine advantage is **not** "Parzen-MLP gives a better joint." It is that
  the *marginal* sub-problems are where this method is demonstrably strong, and Sklar
  lets you isolate them, attach an interpretable low-dimensional dependence model, and
  get a joint estimate with far fewer parameters — *if* the dependence is simple.
  That is a modelling-economy / interpretability advantage, contingent on the copula
  assumption, not a universal accuracy advantage.
