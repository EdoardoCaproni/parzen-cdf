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
4. **Sample-size sweep, three methods (fixed 1.0 / Silverman / adaptive with a Silverman pilot;
   5 seeds).** Two clear lessons:
   - **A fixed window does not benefit from more samples** — it never shrinks, so its CDF gap stays
     flat at ~0.16 and its pdf peak is stuck near 0.21 at every `n`. Only data-driven windows improve.
   - **Adaptive (pilot Silverman) is the best at every `n`** and reaches the true pdf peak fastest:

     | n | fixed 1.0 | Silverman | adaptive |
     |---|---|---|---|
     | 50 | 0.165 | 0.066 | 0.056 |
     | 500 | 0.162 | 0.034 | 0.026 |
     | 2000 | 0.158 | 0.019 | **0.012** |
     | 20000 | 0.157 | 0.008 | **0.005** |

   Note the pilot: the adaptive in step 2 used a too-large pilot (1.0) and failed; with a Silverman
   pilot it instead beats Silverman. So adaptivity is good *given a sensible pilot*.

**State of the art (single Gaussian):** Parzen Window with an **adaptive window size (Silverman
pilot)** is best at every budget (Silverman alone is already excellent on this unimodal case; adaptive
adds a further ~35% on the CDF gap). Quality improves smoothly with the sample count; a fixed window
does not. This unifies with the mixtures, where adaptive is also the winner. Next: Phase A, step 2.

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

**State of the art (mixtures, provisional):** adaptive window size (Silverman pilot). **Revised by
step 3 below** once tested on a broader battery. Next: ~10 more complex distributions.

---

## Phase A · step 3: a battery of 10 random complex mixtures

We generate 10 random Gaussian mixtures (3 to 6 modes, varied means and widths via
`data.random_mixture`) and test every truth-free window-size selector on all of them, at the two
budgets. Cross-validation is `O(n^2 * candidates)`, so it is run only at the under-2k budget
(impractical at `n=20000`). One sample set per (mixture, budget), seeded by mixture index. Script:
`scripts/parzen_random.py`.

**Mean CDF gap across the 10 mixtures** (lower is better):

| selector | under-2k (n=1000) | overall (n=20000) |
|---|---|---|
| Silverman | 0.069 | 0.032 |
| variance-matched | 0.044 | **0.015** |
| adaptive (pilot Silverman) | 0.061 | 0.021 |
| likelihood-CV | 0.029 | impractical (`O(n²·cand)`) |
| **LSCV** | **0.028** | impractical |

![selectors](../results/parzen_random_selectors.png)
![the 10 mixtures](../results/parzen_random_gallery.png)

**Findings (the battery overturns the bimodal conclusion).**

1. **Cross-validation (LSCV ≈ likelihood-CV) is the most accurate truth-free selector** (~0.028 at
   under-2k, about 2.4× better than Silverman and clearly better than adaptive). It searches for the
   window that fits the data, which pays off on complex multimodal shapes.
2. **Adaptive does NOT generalize as the winner.** It won on the two hand-picked bimodals, but across
   10 complex mixtures it is only middling (0.061 under-2k, 0.021 overall), worse than variance-matched
   at *both* budgets. Lesson: a conclusion from 2 distributions did not survive 10.
3. **Variance-matched (a cheap `×0.55` shrink of Silverman) is the best practical selector at high
   `n`** (0.015 overall) and second at low `n`. Being `O(n)`, it scales to any `n`, unlike CV.
4. **Silverman is consistently the weakest** (over-smooths), except on near-unimodal/broad mixtures.
5. The gallery shows the adaptive estimate still blunting sharp peaks; CV / variance-matched sharpen
   them better.

**State of the art (truth-free Parzen window), revised:**
- **low-data (under-2k): cross-validation (LSCV)**;
- **high-data (overall): variance-matched** (use CV if its cost is affordable).
Adaptive is demoted to a special-case method (good on simple, well-separated bimodals). This is the
estimator we carry into Phase B (the MLP).
