# wf08 — Inverse-CDF sampling quality: is the net a faithful, compact generator?

## Question

A monotone CDF can be inverted to generate random variates: draw `u ~ U(0,1)`, return
`x = F^{-1}(u)`. This is the generative use of a learned CDF highlighted by Magdon-Ismail &
Atiya. The honest question for this project: when we invert the **learned net CDF**, do we
get a sampler that is (a) *faithful* — draws that match the true distribution — and (b)
*better than* simply inverting the Parzen CDF the net was trained on?

## Method

- Distribution: `asymmetric_bimodal` (modes at 0 and 3, weights 0.65/0.35, unequal widths — a
  shape with a real tail and a valley, so a sampler can actually go wrong).
- Train n = 2000, draw n = 20000, seeds {0,1,2}. Silverman bandwidth.
- Net: `CDFNet(1, (32,), sigmoid, monotone=True)`, Adam lr 0.03, **8000 epochs** (see note
  below on why budget matters). 97 parameters.
- Inversion: monotone-clamp the CDF on a 4000-point grid, normalise to [0,1], invert by
  `searchsorted` + linear interpolation. **The same uniforms drive net and Parzen draws**, so
  the comparison is free of Monte-Carlo noise between the two methods.
- Verdict metrics: KS gap of the drawn samples' empirical CDF vs the TRUE CDF; absolute error
  of seven quantiles vs the true quantiles (true CDF inverted on the grid the same way). A
  fresh true sample of n = 20000 gives the irreducible-noise floor.

## Results (mean over 3 seeds)

| sampler | KS gap vs TRUE CDF | mean \|quantile error\| |
|---|---|---|
| **net** inverse-CDF | **0.0332 ± 0.0059** | **0.1054** |
| **parzen** inverse-CDF | **0.0279 ± 0.0027** | **0.1392** |
| fresh true sample (floor) | 0.0061 ± 0.0006 | — |

Per-quantile \|error\| at [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:

- true quantiles: [-1.43, -1.02, -0.29, 0.74, 2.67, 3.34, 3.64]
- net:    [0.048, 0.062, 0.022, 0.173, 0.057, 0.134, 0.243]
- parzen: [0.216, 0.156, 0.030, 0.133, 0.076, 0.126, 0.238]

Generator footprint:

- net: **97 float params**
- parzen: 2000 stored samples + 1 bandwidth = 2001 floats → **20.6x more storage**

## The training-budget caveat (why this is honest, not cherry-picked)

At the lightweight 2000-epoch budget the net's sampling looked clearly *worse* (KS 0.058 vs
Parzen 0.028; quantile err 0.258 vs 0.139). A convergence check showed why:

| epochs | net-vs-TRUE KS | net-vs-Parzen KS | parzen-vs-TRUE KS |
|---|---|---|---|
| 2000 | 0.0532 | 0.0436 | 0.0270 |
| 8000 | 0.0266 | 0.0170 | 0.0270 |

At 2000 epochs the net has **not yet matched its Parzen target** (net-vs-Parzen KS 0.044), so
its draws inherit the *fit error on top of* Parzen's estimation error. At 8000 epochs the net
converges onto its target (net-vs-Parzen KS 0.017) and its accuracy vs the truth (0.0266)
**equals Parzen's** (0.0270). The faithful-regressor result holds for sampling: the net
reproduces its Parzen target's quality and nothing more.

## Verdict — advantage: capability_only

- **Sampling accuracy: no advantage.** KS gaps tie within error bars (0.033 vs 0.028, both
  ~5x the irreducible floor). The slight edge in mean quantile error (0.105 vs 0.139, helped
  by the tails) is small, not robust across seeds, and disappears under the obvious read that
  both estimators carry the same n = 2000 of information. Inverting the net is, at best,
  inverting Parzen.
- **The genuine, defensible win is compactness.** The trained net is a 97-float, closed-form,
  differentiable, guaranteed-monotone `F`. Inverting it is a clean root-find on a smooth
  function and costs O(network depth) per variate with **no scan over the 2000 stored
  samples** that Parzen requires. The sampler ships as ~20x less data and decouples generation
  cost from training-set size.

So: yes, a *faithful* generator (matches Parzen / the truth, given enough training), and yes a
*compact* one — but it does not produce *better* samples than the Parzen window it learned
from. The advantage is a deployment/representation property, not statistical quality.

Figure: `temp_analysis/wf08_sampling.png` (net inverse-CDF draws over the true pdf).
