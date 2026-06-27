# Didactic study — estimating a CDF, one step at a time

A deliberately **didactic** rebuild: we start from the simplest possible case and add one ingredient
at a time, each motivated by the previous step's shortcoming. (The earlier, broader exploration lives
in `old/`.)

**Terminology (fixed).** We use a **Parzen Window** estimator; its smoothing parameter is the
**window size**. The CDF estimate is the **integral of the estimated pdf** — with the logistic window
this is available in closed form as a mean of sigmoids. We judge every estimate by the **CDF gap**
(Kolmogorov–Smirnov distance, the largest vertical gap) against the *known* truth.

**Plan.**

- **Phase A — Parzen Window estimation (no network yet).** Get a good Parzen estimate of the true CDF.
  1. single Gaussian: fixed trivial window size → adaptive window size → Silverman → (if needed) grid
     search over the number of samples;
  2. two harder Gaussian mixtures with the best window-size strategy so far (grid search if needed);
  3. ~10 more complex distributions, all tested with the consolidated estimator; harden as needed.
- **Phase B — the MLP.** Train the simplest possible network (one hidden layer, fixed learning rate,
  sigmoidal activations) to learn the consolidated Parzen CDF, then improve it step by step, and
  finally recover the pdf as the network's derivative → **checkpoint 1**.

**Gating.** One step at a time; the call on "good enough" is the owner's.

---

## Phase A · single Gaussian `N(0,1)`

Setup: 2000 samples, evaluation against the exact `N(0,1)`. Script:
`scripts/01_parzen_gaussian.py`.

| step | window-size strategy | CDF gap (KS) | verdict |
|---|---|---|---|
| 1 | fixed = 1.0 (trivial symbolic value) | 0.157 | over-smoothed |
| 2 | adaptive, per-point (pilot = 1.0) | 0.151 | barely helps |
| 3 | **Silverman (data-driven, 0.196)** | **0.0196** | matches the truth |
| 4 | more samples (Silverman, n 500→20k) | **0.008 @ n=20k** | **best; fixes the pdf peak** |

![window-size strategies](../results/parzen_gaussian_window_strategies.png)
![Silverman visual check](../results/parzen_gaussian_silverman_check.png)
![sample-size sweep](../results/parzen_gaussian_sample_size.png)
![sample-size progression](../results/parzen_gaussian_sample_size_progression.png)

**What each step taught us.**

1. **Fixed window size = 1.0 over-smooths.** The estimated pdf is far too wide and low (peak 0.21 vs
   0.40); the logistic window of "size 1" is itself wide, so the estimate behaves like a much broader
   Gaussian. CDF gap 0.157.
2. **Adaptivity around a too-large pilot does not rescue it.** The per-point windows came out at mean
   1.005 (range 0.93–1.91): they barely shrank in the dense bulk and only widened in the tails, so the
   core over-smoothing remained (gap 0.151). Lesson: an adaptive window size needs a *sensible global
   window* to adapt around.
3. **A data-driven global window size (Silverman) solves it.** Window size 0.196 brings the gap to
   0.0196 — the estimated pdf and CDF sit on top of the truth. The only residual imperfection is the
   pdf peak, estimated a touch low (0.368 vs 0.399).
4. **More samples improve it further and fix the pdf peak.** Sweeping `n` with Silverman (5 seeds):
   the window shrinks (∝ n^(−1/5)), the CDF gap drops from 0.034 (n=500) to 0.008 (n=20000), and the
   pdf peak climbs to 0.390 — closing the only visible gap. Convergence is slow (the expected
   Silverman rate), so the practical lever is "as much data as available."

**State of the art (single Gaussian):** Parzen Window with **Silverman window size**; quality improves
smoothly with the sample count. Next: carry it to two Gaussian mixtures (Phase A, step 2).

---

## Phase A · two Gaussian mixtures

We carry **two operating points** forward from here on (owner's request): **best overall** = abundant
data, `n = 20000`; **best under-2k** = a scarce-data budget, `n = 1000` (`n = 500` reported too). We
compare the consolidated **Silverman** window size against an **adaptive** (per-point) window with a
Silverman pilot, since Silverman tends to over-smooth the valley between modes. Script:
`scripts/parzen_mixtures.py`.

**CDF gap (KS) vs truth** (mean over 3 seeds; lower is better):

| distribution | strategy | n=500 | **n=1000 (under-2k)** | n=2000 | **n=20000 (overall)** |
|---|---|---|---|---|---|
| symmetric `0.5 N(±2,0.7)` | Silverman | 0.079 | 0.065 | 0.048 | 0.024 |
| | **adaptive** | 0.076 | **0.060** | 0.041 | **0.0155** |
| asymmetric `0.65 N(0,1)+0.35 N(3,0.6)` | Silverman | 0.048 | 0.043 | 0.027 | 0.014 |
| | **adaptive** | 0.042 | **0.039** | 0.024 | **0.0098** |

![symmetric — window strategies](../results/parzen_symmetric_bimodal_window_strategies.png)
![symmetric — sample size](../results/parzen_symmetric_bimodal_sample_size.png)
![symmetric — progression](../results/parzen_symmetric_bimodal_progression.png)
![asymmetric — window strategies](../results/parzen_asymmetric_bimodal_window_strategies.png)
![asymmetric — sample size](../results/parzen_asymmetric_bimodal_sample_size.png)
![asymmetric — progression](../results/parzen_asymmetric_bimodal_progression.png)

**Findings.**

1. **Adaptive (pilot Silverman) beats plain Silverman everywhere**, modestly at low `n` and more at
   high `n` (symmetric at 20k: 0.024 → 0.0155; asymmetric: 0.014 → 0.0098). So the best strategy for
   mixtures is the adaptive window size.
2. **The two budgets.** *Best overall* (`n=20000`): CDF gap ≈ **0.010–0.016** — good. *Under-2k*
   (`n=1000`): ≈ **0.04–0.06** — moderate. The **symmetric** mixture (well-separated modes, deep
   valley) is the harder one at low `n`; the asymmetric (overlapping modes) is easier.
3. **What needs samples is the multimodal structure.** At low `n` the pdf valley between modes is
   filled in (over-smoothed) and the modes are blunt; the valley deepens and the modes sharpen toward
   the truth as `n` grows (progression figures). The CDF itself stays usable even at `n=500`.
4. **An untapped lever (noted, not yet used).** The adaptive pilot here is the full Silverman window;
   a *sharper* pilot (variance-matched) or a cross-validation window would sharpen further at fixed
   `n`. Held in reserve in case the under-2k accuracy needs to improve.

**State of the art (mixtures):** Parzen Window with an **adaptive window size** (Silverman pilot),
carried at the two budgets. Next: ~10 more complex distributions tested with this estimator (Phase A,
step 3).
