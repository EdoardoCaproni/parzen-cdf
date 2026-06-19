# References

> **Note.** This is a working bibliography seeded at project start. **Verify each entry**
> (authors, venue, year, exact title) against the original source before citing it in the report.

## Parzen-window / kernel density estimation

- **Parzen, E. (1962).** *On Estimation of a Probability Density Function and Mode.* Annals of
  Mathematical Statistics. — The foundational reference for Parzen windows.
- **Rosenblatt, M. (1956).** *Remarks on Some Nonparametric Estimates of a Density Function.* —
  Origins of kernel density estimation.
- **Silverman, B. W. (1986).** *Density Estimation for Statistics and Data Analysis.* — Bandwidth
  selection (Silverman's rule of thumb).

## Neural CDF / density estimation (core precedent for Step 1)

- **Magdon-Ismail, M. & Atiya, A.** Neural-network approach to estimating a CDF and differentiating
  it to obtain the density. *Closest precedent to this project.* Reported in NIPS 1998 ("Neural
  Networks for Density Estimation") and IEEE Transactions on Neural Networks 2002 ("Density
  Estimation and Random Variate Generation Using Multilayer Networks"). — **Confirm exact titles/venues.**

## Monotonic neural networks (monotonicity direction)

- **Sill, J. (1998).** *Monotonic Networks.* NIPS. — Architectural monotonicity: non-negative
  weights + monotone activations ⇒ a monotone network (and a universal approximator of monotone
  functions); non-negativity enforced by reparameterizing weights (Sill used `exp`; we use the
  smoother `softplus`). **This is the basis of our `monotone=True` `CDFNet`.**
- **Daniels, H. & Velikova, M. (2010).** Extension of Sill's construction to *partial*
  monotonicity (monotone in a subset of inputs) — relevant for Step 2.
- *Smooth Monotonic Networks* (researchgate.net/publication/371290194). — Smoothness of the
  monotone map matters here because we differentiate the CDF to obtain the pdf.
- **Wehenkel, A. & Louppe, G. (2019).** *Unconstrained Monotonic Neural Networks (UMNN).* NeurIPS.
  — Models the *derivative* as a positive network and integrates it; the modern alternative if the
  soft penalty / Sill construction proves limiting.

## Function approximation: where to place training inputs

- **Collocation points (PINN literature).** Fitting a known deterministic function over a domain is
  an experimental-design choice; for smooth targets, uniform/equispaced collocation is sufficient.
  *Provably Accurate Adaptive Sampling for Collocation Points in PINNs* (arXiv:2504.00910) reviews
  uniform vs. adaptive sampling. **Justifies our uniform-collocation training inputs.**

## Neural density estimation: the integral / normalization term (out of scope, recorded)

- A network that outputs a **pdf** directly does not integrate to 1 automatically; a normalization
  penalty `λ·(∫f − 1)²` (or `λ|ln ∫f|`) is added, with the integral done by Monte-Carlo in high
  dimensions to dodge the curse of dimensionality. **Not used here** — our network outputs the
  *CDF*, so normalization is automatic and the pdf comes from differentiation.
- *From CDF to PDF: A Density Estimation Method for High-Dimensional Data* (arXiv:1804.05316). —
  On-topic for Step 2; estimate the CDF and obtain the density from it.

## Multivariate / copulas (Step 2 direction)

- **Sklar, A. (1959).** Sklar's theorem — joint CDF decomposed into marginals and a copula.
- **Nelsen, R. B.** *An Introduction to Copulas.* Springer. — Standard reference for copula theory.
