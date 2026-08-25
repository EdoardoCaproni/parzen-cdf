# References

The sources the report cites, and the ones behind techniques it uses without naming them.
Every entry below has been checked against a primary source: authors, venue, year, exact title.

Entries that were in the working bibliography but could not be verified, or that turned out to
be wrong, are recorded at the bottom rather than silently dropped.

## Kernel density estimation

- **Rosenblatt, M. (1956).** *Remarks on Some Nonparametric Estimates of a Density Function.*
  Annals of Mathematical Statistics 27(3), 832–837.
- **Parzen, E. (1962).** *On Estimation of a Probability Density Function and Mode.* Annals of
  Mathematical Statistics 33(3), 1065–1076. The estimator the project is built on.
- **Silverman, B. W. (1986).** *Density Estimation for Statistics and Data Analysis.* Chapman
  and Hall. Source of the rule of thumb `0.9 min(σ̂, IQR/1.349) n^(-1/5)`.
- **Scott, D. W. (1992).** *Multivariate Density Estimation: Theory, Practice, and
  Visualization.* Wiley. Source of `1.059 σ̂ n^(-1/5)`, which the two-stage plug-in reproduces
  when the normal reference is fed into both stages (report, Appendix A).

## Window selection

- **Rudemo, M. (1982).** *Empirical Choice of Histograms and Kernel Density Estimators.*
  Scandinavian Journal of Statistics 9(2), 65–78. Least-squares cross-validation.
- **Bowman, A. W. (1984).** *An Alternative Method of Cross-Validation for the Smoothing of
  Density Estimates.* Biometrika 71(2), 353–360. Least-squares cross-validation, independently.
  This is the selector the project delivers.
- **Habbema, J. D. F., Hermans, J., and van den Broek, K. (1974).** *A Stepwise Discriminant
  Analysis Program Using Density Estimation.* COMPSTAT 1974. Likelihood cross-validation.
- **Abramson, I. S. (1982).** *On Bandwidth Variation in Kernel Estimates: A Square Root Law.*
  Annals of Statistics 10(4), 1217–1223. The adaptive per-sample window, narrower where the
  data is dense. Available in the app, not delivered.
- **Sheather, S. J. and Jones, M. C. (1991).** *A Reliable Data-Based Bandwidth Selection Method
  for Kernel Density Estimation.* Journal of the Royal Statistical Society B 53(3), 683–690.
  The plug-in family; the two-stage version derived in Appendix A of the report is of this kind.
- **Wand, M. P. and Jones, M. C. (1995).** *Kernel Smoothing.* Chapman and Hall. Source of the
  pilot-width formula `g = [-2 K⁽⁴⁾(0) / (μ₂ θ₆ n)]^(1/7)` used in that derivation.

The *variance-matched* window is not a named method. It is Silverman's constant rescaled by
`√3/π` so that the logistic kernel, whose variance is `π²/3`, smooths like the unit-variance
kernel the classical rules assume.

## The benchmark

- **Marron, J. S. and Wand, M. P. (1992).** *Exact Mean Integrated Squared Error.* Annals of
  Statistics 20(2), 712–736. The fifteen normal mixtures used to stress window selectors. The
  project uses thirteen of them, verified against the R package `nor1mix`; densities 14 and 15,
  the two comb shapes, were left out because their parameters could not be confirmed.

## Neural estimation of a distribution function

- **Magdon-Ismail, M. and Atiya, A. F. (1998).** *Neural Networks for Density Estimation.*
  Advances in Neural Information Processing Systems 11. And **(2002)**, *Density Estimation and
  Random Variate Generation Using Multilayer Networks*, IEEE Transactions on Neural Networks
  13(3), 497–520. The closest precedent: estimate the CDF with a network, differentiate it for
  the density.
- **Trentin, E., Lusnig, L., and Cavalli, F. (2018).** *Parzen Neural Networks: Fundamentals,
  Properties, and an Application to Forensic Anthropology.* Neural Networks 97, 137–151. The
  recipe the training stage follows: a network regressing Parzen values at the sample points,
  with its limited capacity doing the smoothing.

## Monotone architectures

- **Archer, N. P. and Wang, S. (1993).** *Application of the Back Propagation Neural Network
  Algorithm with Monotonicity Constraints for Two-Group Classification Problems.* Decision
  Sciences 24(1), 60–75. Non-negative weights plus monotone activations give a monotone
  network. **This, not Sill, is the construction behind the positive-weight option in the app.**
- **Sill, J. (1998).** *Monotonic Networks.* Advances in Neural Information Processing Systems
  10. A *different* construction: groups of linear hyperplanes with a maximum inside each group
  and a minimum across groups. The project does not use it.
- **Daniels, H. and Velikova, M. (2010).** *Monotone and Partially Monotone Neural Networks.*
  IEEE Transactions on Neural Networks 21(6), 906–917.
- **Wehenkel, A. and Louppe, G. (2019).** *Unconstrained Monotonic Neural Networks.* Advances in
  Neural Information Processing Systems 32. Models the derivative as a positive network and
  integrates it.
- **Huang, C.-W., Krueger, D., Lacoste, A., and Courville, A. (2018).** *Neural Autoregressive
  Flows.* ICML 2018. The deep sigmoidal flow is `σ⁻¹(Σ π_j σ(a_j x + b_j))` with `π` on the
  simplex and `a_j > 0`. The delivered estimator is that transformation **without the output
  logit**, which is what keeps it inside `[0,1]` instead of mapping back to the whole line.

## Sampling

- **Marsaglia, G. and Tsang, W. W. (2000).** *The Ziggurat Method for Generating Random
  Variables.* Journal of Statistical Software 5(8). The algorithm NumPy's `Generator` uses for
  a standard normal: `npyrandom` exports `ziggurat_nor_r` and `ziggurat_nor_inv_r`, while the
  legacy `RandomState` exports no such symbol.
- **Marsaglia, G. and Bray, T. A. (1964).** *A Convenient Method for Generating Normal
  Variables.* SIAM Review 6(3), 260–264. The polar method.
- **Box, G. E. P. and Muller, M. E. (1958).** *A Note on the Generation of Random Normal
  Deviates.* Annals of Mathematical Statistics 29(2), 610–611.

## Removed from the working bibliography

- An entry attributing *non-negative weights plus monotone activations* to Sill (1998). That is
  Archer and Wang (1993); Sill's monotonic networks are the min-max construction. The error had
  propagated into a source comment and into the app's wording, and both were corrected.
- Two arXiv preprints on collocation-point placement and on high-dimensional CDF-to-PDF
  estimation, kept while the project still trained on collocation grids. The delivered pipeline
  trains at the sample points only, so neither is relevant any more.
- **Sklar, A. (1959)** and **Nelsen, R. B.**, *An Introduction to Copulas*. Both real and both
  correctly cited, but they belong to the multivariate extension, which the project does not
  attempt. Recorded here so the direction is not lost.
