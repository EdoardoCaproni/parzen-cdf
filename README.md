# parzen-cdf

Neural estimation of a distribution function and its density, from samples alone.

A Parzen window with logistic kernels gives a closed-form estimate of the CDF at the sample
points. Those values become the labels for a small network, which is a convex mixture of
logistic CDFs: monotone, bounded in `[0,1]`, and unit-mass at every parameter value, so
nothing downstream has to repair it. The density is the derivative of that mixture, in closed
form. Nothing in the estimation path knows the true distribution.

University project. **The report is the source of truth**: everything else in this repository
either implements it or is working material that led to it.

---

## Start here

**1. Read the report.** `report/report3.pdf`, 30 pages. One chapter per stage of the pipeline,
in the order the data flows through them. Table 1, on page 4, is the delivered configuration at
a glance; Appendix C gives the evidence behind each choice.

To rebuild it (needs a LaTeX distribution):

```bash
cd report && bash build.sh
```

**2. Open the lab.** An interactive version of the whole pipeline, with every choice in Table 1
exposed as a control, so you can take the rejected road and watch it fail.

```bash
pip install -r requirements.txt
python app/server.py            # then http://localhost:8000
```

Three stages, top to bottom: build a distribution (or load a file of numbers), estimate it with
a Parzen window, train the network on those labels. See `app/README.md`.

**3. Run it on data.** The entry point takes a file of numbers and returns an estimate.

```bash
python -m parzen_cdf.run --samples examples/campione.csv --out risultati/
```

`examples/campione.csv` is 500 observations from a four-mode mixture, one number per line,
with nothing recording what generated them. Your own file works the same way.

A file that can be read in more than one way is refused rather than guessed at, because both
readings produce numbers and only one of them is your data. `examples/campione_excel_italiano.csv`
is the same 500 values as a spreadsheet with Italian settings writes them, and shows what that
looks like:

```bash
python -m parzen_cdf.run --samples examples/campione_excel_italiano.csv --out risultati/
# stops and asks: is "1,5" one number, or two columns?

python -m parzen_cdf.run --samples examples/campione_excel_italiano.csv --out risultati/ --decimal comma
# same 500 values as the other file
```

In Python:

```python
from parzen_cdf.estimate import run_from_samples

est = run_from_samples(x)          # x: a 1-D array of observations
est.cdf(t), est.pdf(t)             # functions, evaluable anywhere
est.h, est.h1                      # the window, and h1 = h*sqrt(n)
est.diagnostics()                  # what can be checked without knowing the answer
```

**4. Check it still works.**

```bash
pytest -q                          # library and app, about three minutes
```

---

## What is in here

| | |
| --- | --- |
| `report/` | The report. `report3.tex` builds `report3.pdf`. |
| `src/parzen_cdf/` | The library. `estimate.py` is the entry point; `parzen.py` the window selectors; `models.py` the estimator; `diagnostics.py` what is measurable without a truth. |
| `app/` | The interactive lab: FastAPI server plus a browser front end. |
| `tests/` | The property and characterisation tests the report refers to. |
| `docs/` | Working record: the studies, the redesign documents, the decision register. |
| `temp_analysis/` | The scripts behind the measurements. Appendix B of the report maps each number to the script that produced it. |
| `scripts/` | Figure generation for the report. |
| `examples/` | Two sample files to run the pipeline on, and what they are for. |

### On `docs/` and `temp_analysis/`

These are working material, kept because the report compresses them and because the code
refers to them, not because they are documentation to read first. They record the project as it
happened, including things that were measured, believed, and then withdrawn. Where they
disagree with the report, **the report is right**.

Two files that used to live in `docs/`, `study.md` and `study2.md`, described a design that has
since been replaced, along with the reports that went with them. They were removed rather than
left to mislead; `git log` still has them.

---

## The short version of the findings

- **No rule of the form `h1 = c·σ̂` can work.** Across a benchmark of sixteen densities the
  optimal constant spans a factor of twelve, and recalibrating it carefully makes the rule
  worse. Least-squares cross-validation is used instead: efficiency 1.26 on average and 1.59 in
  the worst case against the oracle, against 2.27 and 9.38 for the rule it replaces. The
  schedule `h_n = h1/√n` is kept, with `h1 = ĥ·√n` reported.
- **Validity belongs in the architecture, not downstream.** The previous design produced a
  monotone curve by taking a running maximum on a grid and rescaling to `[0,1]`, which needs a
  domain read off the true distribution. The mixture needs none of that: perturb its parameters
  and it stays a CDF, 0 times out of 1000 against 900 for a free MLP.
- **The most natural sample-based check is a trap.** Comparing the estimate to the empirical
  CDF is anticorrelated with the true error and rewards the narrowest possible window. The
  diagnostics that do work are the LSCV score and the leave-one-out log-likelihood.
