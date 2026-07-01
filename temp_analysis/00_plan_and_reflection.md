# Why train an MLP on a Parzen estimate? Plan + reflection

**The question (owner's):** we are fairly sure we understood what the Professor wants (train a
sigmoidal MLP to regress a logistic-Parzen CDF, recover the pdf by differentiation), but does it make
*sense*? In our 1D study the network at best *matches* its Parzen target, never beats it, and Parzen
(even the empirical CDF) already estimates a 1D CDF well. So: demonstrate a real advantage, or fail
honestly and we change course.

## What the literature says (web research, see 01_research.md)

- **Magdon-Ismail & Atiya (2002), the project's precedent.** Train a net to approximate the CDF, get
  the pdf by differentiation. Their stated rationale: **a CDF's constraints (valued in [0,1],
  monotone non-decreasing) are far easier to impose than a pdf's (non-negative, integrates to 1)**,
  and CDFs are sigmoid-shaped, matching the activations. They use the *empirical* CDF as the target.
- **Amortization:** a neural estimator's prediction cost **does not depend on n**, unlike KDE which is
  `O(m*n)` to evaluate at m points (expensive at large n / high dimension). Train once, query cheaply.
- **Smoothing / denoising:** reported that NN pdfs are "closer to the true pdf and less fluctuating
  than KDE".
- **Differentiable, composable form:** the estimate is a differentiable computational graph, usable
  inside gradient-based pipelines, sampling, etc.
- **"From CDF to PDF" (arXiv:1804.05316):** estimate the CDF and recover the density from it, aimed at
  high-dimensional data.

## Candidate advantages, and whether they are testable here

| # | Candidate advantage | Test |
|---|---|---|
| A | **Amortization / compact, n-independent query** (net `O(params)` vs Parzen `O(n)` per query; memory) | `test_query_cost.py` |
| B | **Denoising: the net can beat the *best* fixed-window Parzen** (its smoothness cleans a sharp, low-bias target) | `test_beats_best_parzen.py` |
| C | **Multivariate: smooth differentiable joint density via mixed partials**, where the empirical CDF gives only spikes and KDE is cursed/expensive | `test_2d.py` |
| D | **Constraint handling**: a valid (monotone, unit-mass) CDF/pdf is cheap to guarantee (rectification) | already shown in the main study; recap |
| E | **Generative use: inverse-CDF sampling** from the learned monotone CDF (a capability Parzen does not give directly) | `test_sampling.py` |
| F | **Dimension scaling of query cost**: KDE eval cost grows with n (and the n needed grows with D); the net stays flat | `test_query_cost.py` (D sweep) |

## Honest prior (from our 1D study)

- In 1D the net is a *faithful regressor* of the Parzen target; the accuracy ceiling is the Parzen
  window, not the net. So the advantage is **not** 1D point accuracy.
- The genuine advantages are expected to be: amortization (A/F), the multivariate density route (C),
  cheap valid-CDF constraints (D), sampling (E), and possibly a small denoising win (B).

**Goal of this folder:** run every test above, then judge in `99_verdict.md` whether MLP-on-Parzen
has a demonstrable advantage and where.
