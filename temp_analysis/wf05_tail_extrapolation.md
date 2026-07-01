# wf05 — Tail / out-of-range behaviour: net vs Parzen

**Question.** Outside the data range the logistic-Parzen CDF saturates (each kernel
is a sigmoid) and its pdf decays toward 0. Does the trained sigmoidal MLP extrapolate
the CDF *sensibly* to 0 and 1, and how do tail CDF errors compare to Parzen?

**Setup.** Single Gaussian and symmetric bimodal, n=2000, variance-matched bandwidth,
strict pipeline (net trains only on `(x_i, F_hat(x_i))`), 32-wide sigmoid MLP, 1500
epochs. Headline table averaged over 3 seeds (`wf05_tail_extrapolation.py`); the
far-tail / pdf probe is 1 seed (`wf05b_tail_probe.py`).

## Result 1 — Parzen wins the tail CDF, the net does not

KS / max|err| of the CDF vs truth, by region (3-seed mean):

| region | metric | single Gaussian | sym. bimodal |
|---|---|---|---|
| in-range | max\|err\| | parzen **0.0142** / net 0.0224 | parzen **0.0230** / net 0.0367 |
| far-tail | max\|err\| | parzen **0.0004** / net 0.0092 | parzen **0.0017** / net 0.0322 |

Moderate tail where accuracy actually bites (true F in `1e-3..0.1` and `0.9..1-1e-3`),
max|CDF err| (1 seed):

| | single Gaussian | sym. bimodal |
|---|---|---|
| parzen | **0.0091** | **0.0159** |
| net | 0.0242 | 0.0330 |

The net is ~2x worse than Parzen in the tails, just as it is in-range. Consistent with
the established result that the net is a *faithful regressor* of its Parzen target, not a
tail-improver. **No tail accuracy advantage.**

## Result 2 — The net never reaches 0/1; Parzen does

Asymptotic CDF limit far outside the data (3-seed mean):

| | left limit (want 0) | right limit (want 1) |
|---|---|---|
| parzen | 0.0000 | 1.0000 |
| net | 0.0006 | 0.9993 |

The net plateaus at a residual offset (≈2e-4..1e-3) and stays there out to 3× the plot
span — it does **not** asymptote to exactly 0/1. Parzen hits 0/1 to machine precision
because the extreme-sample sigmoids saturate. So on the specific "extrapolate to 0 and 1"
test, **Parzen extrapolates better.**

## Result 3 — The net invents a fat fake pdf tail

Far-tail pdf at increasing distance (single Gaussian, seed 0), `true / parzen / net`:

| x | true f | parzen f | net f |
|---|---|---|---|
| 6.3 | 1.3e-09 | 4.8e-15 | **6.7e-04** |
| 9.3 | 9.8e-20 | 0 | **1.1e-04** |
| 16 | 1.0e-56 | 0 | **5.5e-06** |
| 24 | 3.3e-126 | 0 | **2.1e-07** |

(bimodal is the same story: net f ≈ 1e-3 at x=10 where truth is 1e-31.) The net's pdf is
the derivative of a sigmoidal MLP, whose hidden units have a slowly-decaying logistic
shoulder; far out it leaves a spurious, polynomially-thin-but-heavy tail orders of
magnitude above truth. Parzen's pdf decays much faster (the logistic kernel's
exponential tail). For tail-mass / extreme-quantile work this net pdf is actively
misleading.

## Result 4 — The one thing the net does get right: it stays valid far out

Over `[-3·span, 3·span]` the net CDF range is `[0.00019, 0.99924]` (Gaussian) and
`[0.00007, 0.99843]` (bimodal) — **inside [0,1], no overshoot**, and the max decrease
(non-monotonicity) is `0.00e+00`. A bounded sigmoidal output cannot leave [0,1] or run
away, so the extrapolation degrades gracefully (flat plateau) rather than diverging.
That is a robustness property, not an accuracy one — and it costs the residual offset of
Result 2 and the fat pdf tail of Result 3.

## Verdict

**No genuine advantage in the tails.** Parzen beats the net on tail CDF accuracy
(~2x lower error), reaches 0/1 exactly, and has a far lighter pdf tail. The net's only
redeeming tail property is structural — its CDF stays in [0,1] and monotone arbitrarily
far out — but it buys that with a CDF that never closes the last ~1e-3 to 0/1 and a pdf
tail that is spuriously heavy by many orders of magnitude. If anything, the tails are a
mild liability of the MLP-on-Parzen approach, not a selling point.

Plot: `wf05_tail_extrapolation.png`.
