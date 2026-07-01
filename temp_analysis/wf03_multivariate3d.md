# wf03 — 3D joint density via the third mixed partial vs product-KDE

## The angle

Extend the CDF → mixed-partial idea to 3D. A net `F : R^3 → (0,1)` is trained **only on the n data
points**, with label = the joint product-logistic Parzen CDF there. The joint density is recovered as
the **third mixed partial** `d3F/dx dy dz` via three nested autograd passes, and compared to a 3D
product-logistic KDE and the analytic truth on a coarse 20^3 grid (n=1000, epochs=2500).

## Setup

- True density: a 3-component 3D Gaussian mixture with diagonal covariances (closed-form
  product-of-normal-CDFs joint CDF used as a reference for the faithfulness check).
- Bandwidths: per-axis Silverman, `h = [0.348, 0.316, 0.306]`.
- Net: `CDFNet(in_dim=3, hidden_sizes=(64,64), sigmoid)`, 4481 params, trained on the 1000 sample
  points only. Final train MSE on the Parzen-CDF labels = `2.01e-04`.
- Density operators:
  - product-KDE: `(1/n) Σ_i Π_d k_d`, `k_d = σ(z)(1−σ(z))/h_d` — the **analytic** mixed partial of the
    Parzen CDF.
  - net: `d3F/dx dy dz` via `autograd.grad` chained three times (clamped at 0).

## Results (measured)

| quantity | KDE (product-logistic) | net (d3F/dxdydz) |
|---|---|---|
| joint density MSE vs truth | **4.70e-06** | 2.01e-05 |
| Riemann integral of the density (should be 1) | 0.994 | **1.695** |
| query cost, 1000 pts | **145 ms** | 919 ms |
| storage | n·d = 3000 floats, O(n)/query | 4481 floats, O(params)/query |

- CDF faithfulness: net-surrogate vs its Parzen target CDF MSE = `6.66e-04`; Parzen target vs true CDF
  = `3.28e-04`. The net tracks its target but, in 3D with only 1000 points fixed to the data, the fit
  is looser than in 1D.

### The structural point (the one real win)

The **joint empirical CDF** is the multivariate step function
`F_n(x) = (1/n) Σ_i 1[x ≥ x_i componentwise]`. Its third mixed partial is a sum of Dirac point masses
on the n data points — **not a usable density**. Demonstrated numerically: the discrete third mixed
difference of `F_n` on the grid is nonzero in only **613 of 6859 cells** (spikes), with max spike
0.0060 against a true density that peaks at 0.0566. So in any dimension, differentiating a CDF to a
density *requires a smooth CDF*; the empirical CDF will not do. The Parzen smoothing (and the net
surrogate) is what supplies smoothness.

## Honest verdict — advantage: capability_only (no quantitative edge)

In 3D the MLP-on-Parzen approach has **no accuracy or cost advantage** over the plain product-KDE,
and it inherits a new defect:

1. **Accuracy: loses.** Net density MSE is 4.3× worse than the KDE (2.01e-05 vs 4.70e-06). Consistent
   with the 1D finding that the net is a faithful regressor of its Parzen target, not a beater of it —
   and the analytic KDE *is* that target's exact derivative, so the net can only add error.
2. **Normalisation: loses badly.** The unconstrained net's third mixed partial integrates to **1.70**,
   not 1. Differentiating a CDF three times amplifies fit wiggle; the product-KDE integrates to ~1 by
   construction. A valid joint density needs the *N-increasing* property, which a per-coordinate-monotone
   (or unconstrained) net does not guarantee — this is the multivariate version of the monotonicity
   problem and it shows up directly as a 70% mass error.
3. **Query cost: loses at this scale.** Triple-nested autograd is ~6× slower than the KDE at n=1000
   (919 ms vs 145 ms / 1000 pts). The O(params) vs O(n) crossover sits far above n=1000; the net only
   wins on per-query cost for n much larger than its parameter count, which is not where 3D KDE is used.

The **only** thing that survives as a genuine point is structural, not numerical: a joint density via
differentiating the CDF is *possible at all* only because the CDF is smoothed — the empirical CDF gives
spikes. But that capability belongs to the Parzen/KDE smoothing, not to the neural net: the analytic
product-KDE already delivers the smooth mixed partial, more accurately, normalised, and cheaper. The
net surrogate adds error and a normalisation bug without buying anything back in 3D.
