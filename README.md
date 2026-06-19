# Parzen CDF

Neural estimation of a cumulative distribution function (CDF) and its density (pdf), using
Parzen-window estimates as training targets.

Master's project. The core idea: a Parzen-window estimator gives a smooth, closed-form CDF
estimate; we train an MLP to regress that CDF under a monotonicity constraint, then obtain the
pdf as the derivative of the trained network. We work first in one dimension, then extend to N.

## Roadmap

**Step 1 — Univariate.**
1. Sample synthetic data from a *known* pdf (a non-trivial mixture of Gaussians).
2. Estimate the CDF with **Parzen windows using logistic kernels**, whose integral is the
   logistic sigmoid — giving a closed-form CDF estimate `F̂(x) = (1/n) Σ σ((x − xᵢ)/h)`.
3. Build a training set of `(x, F̂(x))` pairs.
4. Train an MLP to regress the CDF, with a **monotonicity constraint in the loss**.
5. Recover the pdf as the derivative of the network output (clamping negatives to zero if needed).

**Step 2 — Multivariate.**
1. Synthetic data on an N-dimensional domain (N grows as experiments succeed).
2. N-D (product) logistic Parzen windows for the joint CDF.
3. Training set and MLP as before, re-imposing monotonicity.
4. Recover the joint pdf via mixed partial derivatives, or the **copula** (Sklar's theorem).

See [docs/references.md](docs/references.md) for the bibliography.

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
src/parzen_cdf/   library code (data, parzen, models, training, metrics)
notebooks/        exploratory experiments
scripts/          reproducible runs
tests/            sanity checks
docs/             references and notes
```
