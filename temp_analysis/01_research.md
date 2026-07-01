# Web research notes: MLP-on-(K)DE / CDF density estimation

Key sources and what they contribute to "why train a net on a Parzen estimate".

- **Magdon-Ismail, M. & Atiya, A. (2002), "Density Estimation and Random Variate Generation Using
  Multilayer Networks", IEEE T-NN 13(3):497-520.** The seminal precedent. Net approximates the CDF;
  pdf by differentiation. Rationale: **CDF constraints (in [0,1], monotone) are easier to impose than
  pdf constraints (>=0, integrates to 1)**; CDFs are sigmoid-shaped. Targets = the *empirical* CDF.
  Also covers random-variate generation (sampling via the learned CDF).
  - https://www.researchgate.net/publication/3303098_Density_estimation_and_random_variate_generation_using_multilayer_networks

- **"From CDF to PDF: A Density Estimation Method for High-Dimensional Data" (arXiv:1804.05316).**
  Estimate the CDF, then differentiate to the density; motivated for high dimensions.
  - https://arxiv.org/pdf/1804.05316

- **"Neural Likelihoods via Cumulative Distribution Functions" (arXiv:1811.00974).** Parameterise a
  multivariate CDF with a net; benefits: tail probabilities and low-dimensional marginals read off
  directly without fitting the full joint; density via differentiation.
  - https://arxiv.org/pdf/1811.00974

- **Amortization / cost independent of n.** Neural density estimators give a prediction whose
  complexity does not depend on the number of training samples; KDE is `O(m*n)` to evaluate at m
  points, expensive for large n and higher dimensions (motivating fast-KDE approximations).
  - https://www.emergentmind.com/topics/neural-density-estimators
  - https://arxiv.org/pdf/2208.01206 (fast KDE, on the O(mn) cost)

- **Parzen neural networks (ScienceDirect, 2017).** A line of work explicitly bridging Parzen-window
  estimation and neural networks.
  - https://www.sciencedirect.com/science/article/abs/pii/S0893608017302332

**Takeaways for our case:** the recognised advantages are (i) easier constraint handling on a CDF than
a pdf; (ii) amortised, n-independent, differentiable evaluation vs KDE's O(mn); (iii) a smoother pdf;
(iv) the CDF route scales to high dimensions and yields marginals/tails directly; (v) sampling via the
learned CDF. None of these is a *1D point-accuracy* win, consistent with our experiments. We now test
each concretely.
