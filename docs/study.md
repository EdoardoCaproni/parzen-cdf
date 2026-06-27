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
