# wf02 — Denoising: does the net beat the BEST fixed-window Parzen?

## Question

The MLP-on-Parzen pipeline trains a sigmoidal net to regress a logistic-Parzen
CDF at the n sample points. A possible *statistical* advantage: the net's
limited capacity could act as a **denoiser** — fed a sharp, low-bias /
high-variance Parzen target (small bandwidth), the net's smoothness might
average out the wiggle and land closer to the truth than *any* single
fixed-bandwidth Parzen could.

Test: fix `n = 2000`; sweep the target window over scales of the Silverman
bandwidth. For each scale, measure the Parzen CDF gap vs truth and the gap of
the net trained on that Parzen target. Then compare **net min-over-windows**
against **Parzen min-over-windows**. Note the Parzen min is the *oracle*
bandwidth (chosen by looking at the truth), so this is a deliberately hard bar.
3 seeds, KS (sup gap) and MSE (L2 gap). Distributions: `single_gaussian`,
`symmetric_bimodal`, `asymmetric_trimodal`.

## The convergence trap (why epochs matter)

A first pass at 800 epochs showed the net losing by 65–660%. That was an
**artifact of underfitting**, not a real answer. A side check trains at a fixed
representative scale (symmetric_bimodal, 0.35×Silverman) and measures how far
the net is from *its own Parzen target*:

| epochs | net-vs-truth KS | net-vs-its-target KS |
|-------:|----------------:|---------------------:|
|    800 |          0.1120 |               0.1073 |
|   1500 |          0.0253 |               0.0236 |
|   2500 |          0.0162 |               0.0084 |

At 800 epochs the net is nowhere near its target (KS 0.107 to its own label);
by 2500 it has essentially matched it (0.008). The 32-unit net needs ~2500
full-batch Adam epochs to fit a multimodal target. **All results below use 2500
epochs**, where the net is a faithful regressor of its target — so the
comparison measures the *target's* quality, not training slack.

## Results (n=2000, 3 seeds, 2500 epochs)

### single_gaussian
| scale | Parzen KS | net KS | Parzen MSE | net MSE |
|------:|----------:|-------:|-----------:|--------:|
| 0.35  | 0.0152 | 0.0183 | 0.000018 | 0.000064 |
| 0.55  | 0.0142 | 0.0206 | 0.000020 | 0.000082 |
| 0.80  | 0.0158 | 0.0224 | 0.000034 | 0.000103 |
| 1.20  | 0.0233 | 0.0290 | 0.000097 | 0.000200 |

best Parzen KS=0.0142, best net KS=0.0183 → **net loses by 29.4%** (MSE: loses).

### symmetric_bimodal
| scale | Parzen KS | net KS | Parzen MSE | net MSE |
|------:|----------:|-------:|-----------:|--------:|
| 0.35  | 0.0148 | 0.0196 | 0.000029 | 0.000134 |
| 0.55  | 0.0229 | 0.0261 | 0.000102 | 0.000241 |
| 0.80  | 0.0359 | 0.0394 | 0.000321 | 0.000551 |
| 1.20  | 0.0558 | 0.0600 | 0.000969 | 0.001408 |

best Parzen KS=0.0148, best net KS=0.0196 → **net loses by 31.9%** (MSE: loses).

### asymmetric_trimodal
| scale | Parzen KS | net KS | Parzen MSE | net MSE |
|------:|----------:|-------:|-----------:|--------:|
| 0.35  | 0.0177 | 0.0622 | 0.000029 | 0.000628 |
| 0.55  | 0.0271 | 0.0655 | 0.000078 | 0.000731 |
| 0.80  | 0.0390 | 0.0719 | 0.000185 | 0.000877 |
| 1.20  | 0.0543 | 0.0785 | 0.000420 | 0.001105 |

best Parzen KS=0.0177, best net KS=0.0622 → **net loses by 252%** (MSE: loses).

## Verdict: no advantage

On all three distributions, the net's best-over-windows CDF is **worse** than
the best fixed-window Parzen, on both KS and MSE. The denoising hypothesis is
**not supported**.

Two regimes:

- **single_gaussian, symmetric_bimodal**: the net tracks Parzen closely (KS
  within ~30%, MSE a few×). It is a faithful regressor that pays a small
  smoothing/approximation tax — it does not recover a free accuracy gain.
- **asymmetric_trimodal**: the net loses badly (KS 0.062 vs 0.018). A single
  32-unit hidden layer cannot represent three well-separated modes' worth of
  curvature; even at 2500 epochs it does not fully reach its own sharp target.
  This is a *capacity* shortfall, the opposite of a denoising win.

Notably the net never even ties Parzen at any individual scale — at every
bandwidth the smooth Parzen CDF is already closer to truth than the net's copy
of it. There is no window where the net's smoothing turns a noisy target into
something better than the truth-tuned Parzen.

## Honest caveat

This measures **point accuracy of the CDF against an oracle-tuned Parzen**, and
on that axis the answer is unambiguous: none. It does *not* test other possible
advantages (a closed-form differentiable density, monotone-by-construction
output, query cost independent of n, behaviour in higher dimensions). The
narrow claim "the net statistically denoises a fixed-window Parzen into a
better CDF estimate" is false here.
