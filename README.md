# Parzen CDF

Neural estimation of a cumulative distribution function (CDF) and its density (pdf), using
Parzen-window estimates as training targets. University project.

The pipeline: a **Parzen window with logistic kernels** gives a smooth, closed-form CDF
estimate `F̂(x) = (1/n) Σ σ((x − xᵢ)/h)`. An MLP is trained to regress that CDF **on the data
points only** (inputs `xᵢ`, labels `F̂(xᵢ)` — no collocation, no augmentation), the pdf is
recovered as the derivative of the trained network, and a **downstream rectification**
(cumulative max + rescale) guarantees a monotone CDF and a unit-mass density.

## Status

- **Phase A (Parzen window): redone and consolidated** after the professor's review exposed a
  window-size scale error in the first pass. The corrected study, under the deterministic
  schedule **h_n = h₁/√n** and budgets capped at n = 2000, is in
  [`docs/study2.md`](docs/study2.md) (the original run is kept in
  [`docs/study.md`](docs/study.md) as the historical record).
- **Phase B (the MLP): redone with the Parzen-Neural-Network recipe** (leave-one-out targets,
  sharp teacher window, small-capacity net): trained on the sample points only, the MLP
  **generalizes a better density estimate than the Parzen Window it learns from** (~2× lower
  pdf ISE across the battery, 30/33 cases), and at the baseline n = 500 it beats the best
  truth-free Parzen on both metrics. See Phase B in [`docs/study2.md`](docs/study2.md).
- **Step 2 (multivariate): pending** a discussion with the professor. The known blocker is
  *N-increasing* monotonicity: 1-D rectification does not make a joint CDF valid, and the
  3-D mixed-partial density shows a ~70% mass error without it.

## Phase A results, in brief (corrected study)

Logistic-window Parzen with the schedule **h_n = 1.5·σ̂/√n** (the "σ-rule", calibrated by
stressing h₁ across three orders of magnitude on a ladder of shapes) sits at the statistical
floor E[KS] ≈ 0.87/√n on every distribution tested, already at **n = 500**:

| selector (battery of 10 random mixtures) | n=500 | n=1000 | n=2000 |
|---|---|---|---|
| σ-rule h₁ = 1.5·σ̂ (zero cost) | 0.031 | 0.025 | 0.015 |
| LSCV (O(n²)) | 0.030 | 0.023 | 0.015 |
| Silverman (deprecated: kernel-scale mismatch) | 0.078 | 0.068 | 0.057 |
| oracle fixed-h (truth-peeking bound) | 0.028 | 0.022 | 0.014 |
| empirical CDF | 0.036 | 0.028 | 0.018 |

**Phase B — the MLP.** With Adam and capacity/training scaled to the target's sharpness, the
network is a **faithful regressor of its Parzen target** on every distribution tested
(single Gaussian, bimodals, 10 random mixtures, and the sharp trimodal after a capacity
push). For monotonicity, the soft penalty fails and the Sill construction costs accuracy;
**downstream rectification wins** (0% violations, mass exactly 1, negligible KS cost).
The consolidated end-to-end run is [`scripts/checkpoint1.py`](scripts/checkpoint1.py):

![Checkpoint 1](results/checkpoint1.png)

**Does MLP-on-Parzen have an advantage?** A dedicated investigation
([`temp_analysis/`](temp_analysis/), verdict in
[`temp_analysis/99_verdict.md`](temp_analysis/99_verdict.md), Italian report in
[`report/analisi_vantaggio.tex`](report/analisi_vantaggio.tex)) found **no accuracy
advantage in 1-D**: the net cannot beat the Parzen target it regresses, and a
direct-likelihood model (GMM/flow) beats both. The genuine advantages are structural:
guaranteed validity via the CDF route (mass = 1, non-negative pdf, for one cheap rectify),
constant-cost n-independent querying with ~1000× compression, and the Sklar/copula modeling
economy. Honest framing: **distillation/serving of a trusted KDE**, not better estimation.

## Interactive lab

An educational web app to build mixture densities, watch the Parzen estimate assemble
live, and train the neural CDF regressor with live charts and a network weight view:

```bash
pip install -r requirements.txt
python app/server.py            # then open http://localhost:8000
```

See [`app/`](app/) for details. A fully static, dependency-free port of the app is published
separately as [parzen-lab](https://github.com/Gianeh/parzen-lab), live at
[gianeh.github.io/parzen-lab](https://gianeh.github.io/parzen-lab/).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest          # sanity checks
```

## Layout

```
src/parzen_cdf/   library (data, parzen, models, training, metrics)
scripts/          reproducible experiment runs (Phase A, Phase B, checkpoint 1)
tests/            sanity checks
docs/             study log and references
report/           LaTeX reports (main report + advantage analysis)
temp_analysis/    the "is there an advantage?" investigation (scripts + findings)
results/          generated figures (regenerable via the scripts)
app/              interactive educational lab (FastAPI + browser UI)
old/              earlier exploratory pass (kept for history)
```

See [docs/references.md](docs/references.md) for the bibliography.
