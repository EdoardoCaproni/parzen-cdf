# wf10 — The theoretical case for MLP-on-Parzen (synthesis, no experiment)

This is a theory/literature synthesis. No code, no measured numbers. It collects the
*a priori* arguments for why training a sigmoidal MLP to regress a logistic-Parzen CDF
(and differentiating it for the pdf) could be worth doing, and — being honest, per the
project rule — separates the arguments that survive scrutiny from the ones that do not.

The 1D empirical finding we are reasoning *against* is settled: the network is a **faithful
regressor** of its Parzen target. It matches the teacher, it does not beat it on point
accuracy, and the accuracy ceiling is the bandwidth, not the network. So none of the
arguments below may claim a 1D point-accuracy win. The genuine case, if there is one, is
structural: constraints, amortization, smoothness, and dimension scaling.

---

## 1. Bias–variance: is the net a denoiser of a noisy estimate?

### The setup
The Parzen CDF target is itself a random object. With kernel `K_h` of bandwidth `h`,
`F_hat(x) = (1/n) Σ_i G_h(x − X_i)` where `G_h` is the integrated kernel (here the logistic
sigmoid). Standard KDE asymptotics (Parzen 1962; Silverman 1986) give, for the *density*
`f_hat`,

- bias  ≈ `(h²/2) f''(x) ∫u²K(u)du`  → grows with `h`,
- variance ≈ `f(x) R(K) / (n h)`     → shrinks with `h`,

so MISE is minimized at `h* ~ n^(−1/5)` with MISE `~ n^(−4/5)`. The bandwidth is a single
scalar knob trading one against the other globally.

### The denoiser hypothesis
The appealing intuition is that the MLP, being a smooth low-complexity function class,
acts as a *second* smoother: it averages out the sampling fluctuation in `F_hat` and so
lands closer to the true `F` than the teacher it was trained on. If true, this is a real
bias–variance win — the net would do variance reduction the bandwidth cannot, because it
smooths in *function space* rather than by widening a kernel.

### Why this argument is weak here, and the honest version
The hypothesis is structurally suspect for the canonical pipeline, for two reasons:

1. **The net only sees the n training points, with label `F_hat(X_i)`.** It is fit to
   *interpolate* those n teacher values, not to denoise them. There is no held-out signal
   telling it which part of `F_hat` is sampling noise and which is true `F`. A sufficiently
   flexible regressor reproduces the teacher exactly at the training points; a regularized
   one reproduces a slightly smoothed version. Either way its target *is* the noisy
   estimate. It cannot systematically know the true `F` better than the statistic it was
   handed, because no information about `F` enters except through `F_hat`. This is exactly
   the data-processing intuition: post-processing a statistic cannot add information about
   the parameter.

2. **CDF noise is already tiny.** The empirical CDF converges at `O_p(n^(−1/2))`
   *uniformly* (Dvoretzky–Kiefer–Wolfowitz: `P(sup_x |F_n − F| > ε) ≤ 2 e^(−2nε²)`), with no
   bandwidth and no dimension penalty. The Parzen CDF is a smoothed `F_n` and is at least as
   good. So the thing the net would be "denoising" is already a low-variance object. There
   is little noise to remove, which is consistent with the observed result: the net matches,
   it does not beat.

**Where a denoising win could still hide:** the net smooths in `x`, so it can clean
*localized* roughness that a single global bandwidth leaves behind — e.g. a Parzen estimate
made with a slightly-too-small `h` is jagged, and the net's smoothness could reduce its
*density* MSE below that of the same jagged teacher (the pdf is far noisier than the CDF;
its MISE is `n^(−4/5)`, not `n^(−1)`). This is the one bias–variance angle worth a number,
and it is what the sibling experiment `test_beats_best_parzen.py` targets. The *a priori*
expectation: at worst the net ties the best-bandwidth Parzen, and any win is small and
confined to under-smoothed teachers — not a general improvement over a well-tuned KDE.

**Verdict on this axis: weak.** The denoiser story is real only against a *badly tuned*
teacher; against a well-chosen bandwidth the net inherits, it does not beat, the teacher.

---

## 2. The CDF-vs-pdf constraint argument (the strongest classical reason)

This is the original Magdon-Ismail & Atiya (2002) rationale, and it is the most defensible
theoretical advantage of the *CDF route* (independent of whether the teacher is empirical or
Parzen).

A valid **pdf** must satisfy two constraints that are awkward for a function approximator:
`f(x) ≥ 0` everywhere, and `∫ f = 1`. A net that outputs a density directly satisfies
neither for free; the literature handles them with a non-negative output layer plus a
normalization penalty `λ(∫f − 1)²`, and in high dimensions the integral itself must be
Monte-Carlo'd (the references file records this as the route we deliberately avoid).

A valid **CDF** must satisfy: monotone non-decreasing, `F(−∞)=0`, `F(+∞)=1`, range `[0,1]`.
These map onto neural machinery *for free or cheaply*:

- **Range `[0,1]` and the limits** come from a sigmoid output. No penalty.
- **Monotonicity** is a *shape* constraint, enforceable architecturally with non-negative
  weights + monotone activations (Sill 1998; UMNN, Wehenkel & Louppe 2019), or cheaply
  post-hoc by rectification (the project's `rectify_cdf`). The main study already shows
  rectification gives a valid monotone CDF at negligible cost.
- **Unit mass is automatic.** Because the pdf is obtained as `f = F'`, and `∫ F' = F(∞) −
  F(−∞) = 1` whenever `F` is a proper CDF, normalization is *free*. There is no integral to
  estimate, no `λ` to tune, no Monte-Carlo. This is the crux: **the CDF route converts the
  pdf's hard global constraint (`∫f=1`) into a trivial boundary condition on `F`.**

This is a genuine, dimension-independent structural advantage of estimating `F` and
differentiating — and it is *capability*, not point-accuracy. It is exactly why the project
outputs a CDF rather than a density. It does not by itself argue for the *Parzen teacher*
over the empirical CDF; it argues for the CDF parameterization. (Parzen vs `F_n` as the
teacher is a separate, smaller question: the Parzen CDF is differentiable and gives the net
a smooth target, whereas `F_n` is a step function and would hand the net a non-smooth
target to fit — relevant because we then differentiate.)

**Verdict on this axis: strong (as a capability).** The "no normalization term, ever"
property is real and is the cleanest theoretical reason to prefer the CDF route. It buys a
guaranteed-valid density, not a more accurate one.

---

## 3. Amortized inference and compact representation

KDE / Parzen is a **memory-based** (lazy) estimator: to evaluate at a query point you sum
over all `n` training samples, `O(n)` per query, `O(mn)` for `m` queries, and you must keep
all `n` samples around (the references cite the `O(mn)` cost and the fast-KDE literature
that exists to dodge it). The neural estimate is **amortized**: training cost is paid once,
after which a query is `O(#params)`, *independent of `n`*, and the model stores a fixed
parameter vector instead of the dataset.

This is a real, uncontroversial systems advantage with two consequences:

- **Query/throughput:** for large `n` and many queries, a fixed-size net is cheaper to
  evaluate than re-summing the kernel over the sample. This is `test_query_cost.py`.
- **Compression / privacy:** the net is a lossy compression of the sample into weights; the
  raw points need not be retained or shipped.

It is, however, an advantage *of any* amortized estimator over a lazy one — a fitted
parametric model or a normalizing flow has it too. It is not unique to MLP-on-Parzen, and it
is orthogonal to accuracy. It also has a crossover: for small `n` and few queries, KDE's
zero training cost wins. So the claim must be quantified, not asserted.

**Verdict on this axis: moderate, and not unique.** Genuine `O(params)` vs `O(n)` query
scaling, but shared by every amortized method and only winning past a sample/query crossover.

---

## 4. Position relative to normalizing flows

This is the most important honesty check, because flows are the modern neural density
estimator and they expose what MLP-on-Parzen is and is not.

A normalizing flow learns an invertible map `T` with tractable Jacobian and maximizes the
**exact data likelihood** `Σ_i log p(X_i)` via the change-of-variables formula. Crucially:

- **Flows train on the data directly, by MLE.** Their objective is the likelihood of the
  *samples*. There is no KDE teacher. The estimator's statistical efficiency is governed by
  the likelihood and the model class, not by a pre-smoothed target.
- **MLP-on-Parzen trains on a KDE teacher, by regression.** Its objective is to match
  `F_hat(X_i)`. This caps it: it can be at best as good as its teacher (Section 1), because
  the teacher is the only channel through which the data speaks to it. A flow has no such
  cap — it can in principle exploit structure the Parzen smoother washes out.

So in the space of neural density estimators, **MLP-on-Parzen is the distillation /
teacher-student variant, and flows are the direct-MLE variant.** That framing makes the
trade-offs precise:

- *Against flows, MLP-on-Parzen gives up* the ability to surpass the smoother and the clean
  generative model with exact density (flows give exact `p(x)` *and* one-pass sampling via
  `T^(−1)`; the CDF route gives sampling only via inverse-CDF, easy in 1D, awkward in `d>1`).
- *Against flows, MLP-on-Parzen keeps* simplicity (a plain MLP regression, no coupling
  layers, no Jacobian bookkeeping, no MLE optimization pathologies), a teacher that is itself
  a consistent estimator (so the student is a consistent estimator of `F` as `h→0, nh→∞`),
  and the trivial constraint story of Section 2. It is the lazy, robust option; flows are
  the powerful, fiddly option.

The "Neural Likelihoods via CDFs" line (arXiv:1811.00974) sits between the two: it
parameterizes a multivariate CDF with a net and trains by likelihood (density = mixed
partial of the CDF), keeping the CDF's marginal/tail conveniences while not being limited to
a KDE teacher. It shows the CDF *parameterization* (Section 2) is separable from the *KDE
teacher* (Section 1) — and that the teacher is the part with the accuracy ceiling.

**Verdict on this axis: clarifying, not a win.** Flows dominate MLP-on-Parzen on raw density
accuracy and on generative use; MLP-on-Parzen wins only on simplicity and on the
constraint/teacher-consistency story. State this plainly in the report.

---

## 5. The curse of dimensionality — and why the CDF route partly sidesteps it

This is, theoretically, the most consequential argument, and it is the one that motivates
"From CDF to PDF" (arXiv:1804.05316) and the multivariate direction.

### Density estimation is cursed
A `d`-dimensional KDE has MISE `~ n^(−4/(4+d))` (Silverman 1986): the optimal rate degrades
sharply with `d`, and the sample size needed to hold accuracy fixed grows exponentially. The
density is a *local* object; in high `d` every neighborhood is empty (data sparsity), so
local averaging — which is all a kernel does — fails.

### The CDF is (almost) not cursed
The CDF is a *cumulative*, hence global, object, and its sample estimate has a rate that is
much kinder to dimension:

- The **empirical CDF** satisfies the multivariate DKW / VC bound: `sup_x |F_n(x) − F(x)|`
  is `O_p(n^(−1/2))` in *every* fixed dimension. The class of "lower-left orthants"
  `{(−∞, t] : t ∈ R^d}` has finite VC dimension (`d`), so uniform convergence is
  `n^(−1/2)` with at most logarithmic/`√d` constants — **no exponential `d` penalty in the
  rate.** This is the precise sense in which "the CDF converges at `~1/√n` in any dimension
  while the density does not."

So the CDF route's premise is sound: estimate the well-behaved cumulative object first, and
obtain the density by differentiation, rather than estimating the cursed local object
directly. The smooth differentiable CDF (which a net gives, and the empirical CDF does not —
`F_n` is a sum of indicators, its mixed partials are spikes, not a usable density) is the
enabling piece.

### The honest caveat (this is where rigor matters)
Differentiation does **not** preserve the `n^(−1/2)` rate. The density is the mixed `d`-th
partial `∂^d F / ∂x_1…∂x_d`, and differentiation is an unbounded, noise-amplifying operator:
recovering `d` derivatives from a function estimated at rate `n^(−1/2)` reintroduces a
dimension-dependent penalty. You cannot get a non-cursed *density* by differentiating a
non-cursed CDF for free — the curse moves from the estimation step into the differentiation
step. What the CDF route genuinely buys is:

1. a **well-posed, low-rate target** (`F`) to fit, instead of a cursed one (`f`);
2. **marginals and tail/orthant probabilities for free and accurately** — any marginal CDF
   is read off by sending the other coordinates to `+∞`, and a marginal of a `d`-dim CDF is
   still a 1D CDF estimated at `n^(−1/2)` (this is the arXiv:1811.00974 selling point: "tail
   probabilities and low-dimensional marginals without fitting the full joint");
3. a **smooth, valid, differentiable joint** where the empirical CDF gives only spikes —
   enabling a usable (if accuracy-limited) joint density via mixed partials, which is the
   `test_2d.py` direction.

So the correct claim is not "MLP-on-Parzen beats the curse of dimensionality for densities."
It is: **the CDF is the right intermediate object because it estimates at `n^(−1/2)` in any
dimension and carries marginals/tails for free; the net makes that CDF smooth and
differentiable so a joint density can be extracted at all.** The density extraction still
pays a dimension cost. State the caveat; do not overclaim the rate.

**Verdict on this axis: strong premise, with a stated caveat.** The CDF's dimension-robust
rate and free marginals are the most distinctive theoretical advantage; but the density
recovered by differentiation does not inherit the `n^(−1/2)` rate.

---

## Synthesis: what survives

Ranking the theoretical arguments by how well they survive an honest reading:

| Argument | Status | One-line reason |
|---|---|---|
| **CDF parameterization → free unit mass + cheap validity (§2)** | **Strong (capability)** | `∫F'=1` automatically; no normalization penalty, ever. |
| **CDF rate `n^(−1/2)` in any `d` + free marginals/tails (§5)** | **Strong (premise), with caveat** | Right intermediate object; but the *density* loses the rate under differentiation. |
| **Amortized `O(params)` query vs KDE `O(n)` (§3)** | **Moderate, not unique** | Real systems win past a crossover; any amortized model has it. |
| **Denoising the teacher (§1)** | **Weak** | Net's target *is* the noisy estimate; wins only vs an under-smoothed teacher. |
| **Beating a well-tuned KDE on 1D point accuracy** | **None** | Settled empirically: faithful regressor, matches but does not beat. |
| **Vs normalizing flows (§4)** | **No accuracy win** | Flows train on data by MLE with no teacher ceiling and exact density; MLP-on-Parzen wins only on simplicity/robustness. |

**Bottom line for the report.** The defensible theoretical case for MLP-on-Parzen is
*capability and structure, not point accuracy*: (i) it produces a guaranteed-valid density
with no normalization term because the pdf is the derivative of a sigmoid-bounded CDF; (ii)
it builds on the CDF, which is the statistically well-behaved object in any dimension
(`n^(−1/2)`, free marginals and tails), turning it into a smooth differentiable surface from
which a joint density can be extracted at all — with the honest caveat that differentiation
reintroduces a dimension cost; (iii) it is amortized and compact relative to lazy KDE. It is
explicitly *not* a way to beat a well-tuned Parzen window in 1D, and *not* competitive with
normalizing flows on raw density accuracy or generative quality. If the project needs a
headline advantage, it is the **constraint-free valid density via the CDF derivative**,
backed by the CDF's dimension-robust convergence — everything else is either shared with
other methods or limited by the teacher.

---

## References (as named in the task; verify venues against originals before citing)

- **Magdon-Ismail, M. & Atiya, A. (2002).** *Density Estimation and Random Variate Generation
  Using Multilayer Networks.* IEEE Transactions on Neural Networks 13(3):497–520. — The
  precedent: approximate the CDF, differentiate for the pdf; CDF constraints are easier than
  pdf constraints. (Earlier: "Neural Networks for Density Estimation," NIPS 1998.)
- **"From CDF to PDF: A Density Estimation Method for High-Dimensional Data" (arXiv:1804.05316).**
  — Estimate the CDF, recover the density from it; motivated by high dimension. Basis for §5.
- **"Neural Likelihoods via Cumulative Distribution Functions" (arXiv:1811.00974).** —
  Parameterize a multivariate CDF with a net; marginals and tail probabilities read off
  directly, density via differentiation. Shows CDF parameterization is separable from a KDE
  teacher (trains by likelihood). Basis for §4 and §5.
- **Parzen, E. (1962); Silverman, B. W. (1986).** KDE bias–variance and the `n^(−4/(4+d))`
  density MISE rate. Basis for §1, §5.
- **Dvoretzky, Kiefer & Wolfowitz (1956) / Massart (1990); VC theory.** Uniform `n^(−1/2)`
  convergence of the (multivariate) empirical CDF, finite VC dimension of orthants. Basis
  for §5.
- **Sill, J. (1998), Monotonic Networks; Wehenkel & Louppe (2019), UMNN.** Architectural
  monotonicity for valid CDFs. Basis for §2.
