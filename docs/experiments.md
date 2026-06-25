# Experiments — neural CDF/pdf from Parzen targets

Track record for the `experiments/cdf-extension` branch. Every idea is logged here, including the
ones that don't pan out. A progressive LaTeX write-up of the same material lives in
[`../report/report.tex`](../report/report.tex).

## The pipeline (constraint-correct)

1. Pick a known CDF/pdf — a 1-D Gaussian mixture (closed-form truth).
2. Draw a fixed number of samples `X = {x₁..xₙ}` from it.
3. Fit a logistic-window Parzen estimator on the samples (unsupervised; window initially fixed
   size, Silverman): `F̂(x) = (1/n) Σ σ((x − xᵢ)/h)`.
4. Train a sigmoidal MLP to estimate the **true** CDF, using **label `F̂(xᵢ)` at each sample point
   `xᵢ` only**.
5. Recover the pdf from the learned CDF, via 5.1) the autograd/empirical derivative `dF/dx`, or
   5.2) the copula (Sklar; multivariate / later).

### Hard constraints (inviolable)

- **Train on the data points only.** Inputs are exactly the `n` drawn samples `xᵢ`; labels are
  `F̂(xᵢ)`. We do **not** sample the Parzen CDF at synthetic `x`, build collocation grids, produce
  extra representations, or augment the data.
- **Monotone non-decreasing CDF**, enforced by a loss penalty, a downstream rectification, or a
  monotone-by-construction architecture.
- **Sigmoidal network**; **Gaussian-mixture** targets (the four ladder distributions in
  `parzen_cdf.data`).

### Rectification of the inherited code

`training.make_training_set` trains on ~1024 **uniform collocation** points labelled with `F̂` — it
samples the Parzen CDF at synthetic `x`, violating constraint 1. It is kept as an *out-of-constraint
reference* only. The constraint-correct builder is `training.make_sample_training_set(samples, h)`,
which returns `(xᵢ, F̂(xᵢ))` for the `n` samples. All experiments here use the latter.

## Conventions

- **Referee:** every estimate is scored against the *known* analytic truth on a dense evaluation
  grid (the grid and truth are evaluation only — never training data).
- **Metrics:** `KS(CDF) = max|F_true − F̂|`; `MSE(pdf)`; monotonicity violation fraction (share of
  decreasing CDF steps on the grid); recovered pdf mass `∫ f̂ dx` (should be ≈ 1).
- **Parzen row:** the Parzen estimate is the target the MLP fits — it is the *ceiling* for a given
  `h`. The MLP cannot beat it on the CDF unless it incidentally undoes the bandwidth's over-smoothing.
- **Reproducibility:** `seed=0`; seed set before model construction and inside `train_cdf`.

## Roadmap — distribution ladder (outer) × ablation axes (inner)

We explore step 1 **deeply**: distribution complexity is the *outer* progression, the ablation axes
are the *inner* deep-dive. We exhaust the inner axes on a simple distribution before adding
complexity. Univariate only — the multivariate case is parked until step 1 is understood. Progression
is gated.

**Outer ladder (distributions):**

| Stage | id | pdf | status |
|---|---|---|---|
| A | `G1` | `N(0,1)` | ✅ deep dive done |
| B | `M1` | `0.5·N(−2,0.7) + 0.5·N(2,0.7)` — symmetric bimodal | ⬜ next |
| B | `M2` | `0.65·N(0,1) + 0.35·N(3,0.6)` — asymmetric bimodal | ⬜ next |
| C | `T1` | `0.3·N(−2,0.5) + 0.5·N(1,1) + 0.2·N(4,0.3)` — trimodal | ⬜ later |
| C | `S1` | `0.6·N(0,1.5) + 0.4·N(0.5,0.2)` — spike-in-broad | ⬜ later |
| C | `R*` | random k-mode mixtures — arbitrary complex 1D | ⬜ later |

**Inner axes (run per stage):** E1 capacity (width, depth) · E2 optimization (lr, epochs) · E3
bandwidth / target quality (Silverman → var-matched → CV → adaptive) · E4 sample count · E5
monotonicity strategy (soft-λ vs Sill vs downstream rectification; pdf-mass renormalisation). Then
E6 (combine per-axis winners) and E7 (pdf recovery: mass loss, boundary handling).

*Note on E0:* the first naive baseline (below) ran all four of Edo's distributions at a single
config — a premature cross-distribution snapshot. It is superseded by the staged dives but kept for
the multimodal preview it gave.

### Hypothesis ledger

- **H1 (capacity):** *G1:* width helps with diminishing returns (KS 0.031→0.026, w4→128); depth-2
  (32,32) reaches the Parzen ceiling, depth-3 hurts at a fixed training budget. Multimodal open.
- **H2 (optimization):** *G1 — confirmed, the dominant lever.* The naive baseline (lr 1e-2, 2000
  epochs) is under-trained; lr 0.1 **or** 10k epochs each bring KS to the Parzen ceiling (~0.020).
- **H3 (target quality):** *G1:* a minor lever for a unimodal target — slightly finer / adaptive `h`
  is marginally best, over-smoothing (1.5× Silverman) is worst; selectors cluster. Expected to
  *dominate* on multimodal targets (Silverman over-smooths) — to be tested in Stage B/C.
- **H4 (samples):** *G1:* modest, noisy improvement with `n` (single seed; multi-seed needed).
- **H5 (monotonicity):** *G1 — confirmed.* The unconstrained net has 0 violations → the soft penalty
  is inactive at every λ (`baseline ≡ soft`); Sill is pure cost (KS 0.077, mass 0.88). The
  constraint is expected to bite only on multimodal / high-capacity / fine-`h` regimes; downstream
  rectification (cumulative max + renormalise) is the deferred alternative for when it does.
- **H6 (pdf mass):** *G1:* mass loss is small once trained (≈0.99); post-hoc mass-normalisation
  gives only a slight MSE gain. Larger effect expected where mass loss is larger (multimodal — E0
  showed 0.82–0.91 there).

---

## E0 — Early cross-distribution naive snapshot (superseded by Stage A)

**Hypothesis.** A small sigmoidal MLP trained on `(xᵢ, F̂(xᵢ))` recovers the Parzen target
reasonably; monotonicity may not yet be binding; by-construction monotonicity (Sill) costs accuracy.

**Setup.** All four mixtures. `n = 2000`; fixed **Silverman** bandwidth; one hidden layer of
**width 16**, sigmoid activation + final sigmoid; **Adam lr 1e-2**, **2000 epochs**; `seed=0`.
Training set = data points only. Three monotonicity arms (Edo's variants, reused):
`baseline` (unconstrained), `soft` (penalty λ=5 on a uniform grid over the observed data range),
`sill` (monotone by construction). pdf = autograd `dF/dx`, negatives clamped.

**Results** (`parzen` row = the target / ceiling; lower KS & MSE better; mass → 1 ideal):

| dist | variant | KS(CDF) | MSE(pdf) | viol% | pdf mass |
|---|---|---|---|---|---|
| single_gaussian (h=0.196) | parzen | 0.0196 | 0.00019 | 0.00% | 0.9999 |
| | baseline | 0.0319 | 0.00024 | 0.00% | 0.9793 |
| | soft | 0.0319 | 0.00024 | 0.00% | 0.9793 |
| | sill | 0.0774 | 0.00094 | 0.00% | 0.8786 |
| symmetric_bimodal (h=0.413) | parzen | 0.0477 | 0.00154 | 0.00% | 0.9991 |
| | baseline | 0.1144 | 0.00828 | 0.00% | 0.9092 |
| | soft | 0.1144 | 0.00828 | 0.00% | 0.9092 |
| | sill | 0.1599 | 0.00916 | 0.00% | 0.8266 |
| asymmetric_trimodal (h=0.443) | parzen | 0.0470 | 0.00232 | 0.00% | 0.9957 |
| | baseline | 0.0883 | 0.00474 | 0.00% | 0.9096 |
| | soft | 0.0883 | 0.00474 | 0.00% | 0.9096 |
| | sill | 0.1268 | 0.00545 | 0.00% | 0.8164 |
| spike_in_broad (h=0.158) | parzen | 0.0542 | 0.00336 | 0.00% | 0.9999 |
| | baseline | 0.0590 | 0.00361 | 0.00% | 0.9729 |
| | soft | 0.0590 | 0.00361 | 0.00% | 0.9729 |
| | sill | 0.0934 | 0.00717 | 0.00% | 0.8783 |

![E0 trimodal](../results/e0_naive_asymmetric_trimodal.png)

**Observations.**

1. **`baseline ≡ soft` exactly.** The unconstrained net already has 0% monotonicity violations, so
   `mean(relu(−dF/dx)) = 0` and the penalty contributes no gradient. Monotonicity is **not binding**
   at this scale → motivates H5 (revisit at higher capacity / finer `h`).
2. **Under-fitting dominates.** The recovered pdf (figure) is a single smooth bump that misses the
   trimodal structure — even though the net cleanly fits its training labels (`train_MSE ~3e-4`).
   The width-16 net has too little capacity / training to express the shape → H1, H2.
3. **Two stacked headrooms.** The Parzen target itself (gray) is over-smoothed by Silverman and
   already misses the true modes; so improving the neural fit alone cannot reach the truth — the
   bandwidth/target must improve too → H3.
4. **Sill underperforms** on every distribution (higher KS/MSE) and loses the most mass
   (0.82–0.88): non-negative weights cost expressiveness at width 16.
5. **pdf mass < 1 for all neural arms** (0.82–0.98) vs Parzen ≈ 1.0 → H6.

**Verdict.** Baseline established. The naive net is in a clear under-fitting regime with an
over-smoothed target — exactly the headroom Phase 1 (capacity, training, bandwidth) will probe.
Reproduce with `python scripts/experiments/e0_naive_baseline.py` (writes `results/e0_naive_*.png`,
`results/e0_naive_baseline.json`).

---

## Stage A — single Gaussian `N(0,1)`: inner ablation (deep dive)

**Hypothesis.** On the simplest target — one mode, no mixture — isolate how each knob moves the
neural CDF fit, with no multimodality confound, and learn which axes matter before adding mixtures.

**Setup.** `N(0,1)`; referee grid `[−6, 6]`; one knob varied off the naive baseline (width 16,
lr 1e-2, 2000 epochs, Silverman `h`, `n=2000`, unconstrained); data points only; `seed=0`. **Parzen
ceiling: KS = 0.0196, MSE(pdf) = 0.00012.** `MSEn` below = pdf MSE after mass-normalisation
(dividing the recovered pdf by its integral).

**E1 — capacity** (width at depth 1; then depth at width 32)

| config | KS | MSE(pdf) | MSEn | mass |
|---|---|---|---|---|
| width=4 | 0.0306 | 0.00012 | 0.00008 | 0.955 |
| width=8 | 0.0329 | 0.00012 | 0.00007 | 0.971 |
| width=16 | 0.0319 | 0.00016 | 0.00014 | 0.993 |
| width=32 | 0.0302 | 0.00014 | 0.00013 | 0.995 |
| width=64 | 0.0273 | 0.00014 | 0.00014 | 0.998 |
| width=128 | 0.0263 | 0.00012 | 0.00012 | 0.998 |
| **depth=2 (32,32)** | **0.0208** | 0.00010 | 0.00010 | 0.995 |
| depth=3 (32,32,32) | 0.0421 | 0.00020 | 0.00018 | 0.988 |

**E2 — optimization** (lr at 2000 epochs; epochs at lr 1e-2)

| lr | KS | mass | | epochs | KS | mass |
|---|---|---|---|---|---|---|
| 1e-3 | 0.0372 | 0.978 | | 1000 | 0.0360 | 0.984 |
| 3e-3 | 0.0369 | 0.983 | | 2000 | 0.0319 | 0.993 |
| 1e-2 | 0.0319 | 0.993 | | 5000 | 0.0210 | 0.998 |
| 3e-2 | 0.0290 | 0.997 | | 10000 | **0.0197** | 0.999 |
| **1e-1** | **0.0193** | 0.997 | | | | |

**E3 — bandwidth / target quality** (each retrains on the new Parzen target)

| h | KS | mass | | selector | h | KS |
|---|---|---|---|---|---|---|
| 0.3× (0.059) | 0.0261 | 0.994 | | var_matched | 0.108 | 0.0274 |
| 0.5× (0.098) | 0.0270 | 0.993 | | likelihood_cv | 0.110 | 0.0280 |
| 0.7× (0.137) | 0.0285 | 0.993 | | lscv | 0.114 | 0.0284 |
| 1.0× (0.196) | 0.0319 | 0.993 | | adaptive | 0.165* | **0.0256** |
| 1.5× (0.295) | 0.0409 | 0.991 | | | | |

**E4 — sample count** | **E5 — monotonicity**

| n | KS (neural) | KS (parzen) | | strategy | KS | mass |
|---|---|---|---|---|---|---|
| 250 | 0.0391 | 0.0461 | | baseline | 0.0319 | 0.993 |
| 500 | 0.0317 | 0.0331 | | soft λ=1 | 0.0319 | 0.993 |
| 1000 | 0.0391 | 0.0292 | | soft λ=5 | 0.0319 | 0.993 |
| 2000 | 0.0319 | 0.0196 | | soft λ=20 | 0.0319 | 0.993 |
| 4000 | 0.0288 | 0.0214 | | sill | 0.0774 | 0.884 |
| 8000 | 0.0249 | 0.0175 | | | | |

![Stage A ablation](../results/stage_a_single_gaussian.png)

**Findings.**

1. **The naive baseline was under-trained — optimization is the dominant lever.** lr 1e-2 / 2000
   epochs gives KS 0.032; lr→0.1 **or** epochs→10k each reach the Parzen ceiling (~0.0196). [E2]
2. **Capacity:** width has diminishing returns; **depth-2 (32,32) is best** (KS 0.021), but depth-3
   *hurts* (0.042) — deeper nets are harder to train at the same budget (train_MSE 3.5e-4). [E1]
3. **Bandwidth is a minor lever for a unimodal target:** slightly finer / adaptive `h` is marginally
   best (~0.026); over-smoothing (1.5× Silverman) is worst (0.041); the truth-free selectors
   cluster. The big bandwidth effect is expected only once the target is multimodal. [E3]
4. **Samples:** modest, noisy gain with `n` (single seed — 1000 is an outlier; multi-seed needed to
   read the trend cleanly). Neural tracks the Parzen target but stays slightly above it. [E4]
5. **Monotonicity is not binding:** `baseline ≡ soft` at every λ (0 violations → inactive penalty);
   Sill is pure cost (KS 0.077, mass 0.88). [E5]
6. **Mass / normalisation:** mass ≈ 0.99 once trained; post-hoc normalisation barely helps here
   (it mattered more at tiny width, where mass dipped to 0.955). [H6]

**Best `G1` config:** lr 0.1 (or ≥5k epochs) with width ≥32 / depth-2 → neural KS ≈ the Parzen
ceiling. **The unconstrained sigmoidal MLP can match its Parzen target on the single Gaussian.** The
open question for Stage B: does that still hold when the target is multimodal, and does bandwidth
then become the dominant lever (as H3 predicts)?

**Verdict.** Stage A understood. Optimization dominates; monotonicity is free; Sill is a net cost;
bandwidth is secondary for a unimodal target. Carry **lr 0.1 + adequate width/epochs** into Stage B
as the new baseline. Reproduce: `python scripts/experiments/stage_a_single_gaussian.py` (writes
`results/stage_a_single_gaussian.{png,json}`).
