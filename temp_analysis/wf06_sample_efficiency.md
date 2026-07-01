# wf06 — Sample efficiency at very small n

**Question.** At very small sample sizes (n ∈ {25, 50, 100, 200}), does the smooth
MLP-on-Parzen generalize better than the raw Parzen CDF (lower CDF gap to truth),
averaged over seeds, on a single Gaussian and a symmetric bimodal? Is the net more
sample-efficient, equal, or worse?

**Setup.** Everything is fit *only* from the n samples:
- Silverman bandwidth `h` from the n samples,
- Parzen CDF target built from those samples at `h`,
- net (`CDFNet(1,(32,))`, sigmoid, Adam, lr 0.03, 2000 epochs) trains *only* on
  `(xᵢ, F̂(xᵢ))` at the n sample points — no collocation.

Evaluated against the analytic truth CDF on a 1200-pt grid over [-6,6], 8 seeds per cell.
We also report the empirical (step) CDF as the dumbest baseline.
Script: `temp_analysis/wf06_sample_efficiency.py`. Runtime ≈ 35 min wall (CPU-starved by concurrent jobs).

## Results — mean over 8 seeds

### single_gaussian

| n   | emp KS | par KS | net KS | emp MSE | par MSE | net MSE |
|-----|--------|--------|--------|---------|---------|---------|
| 25  | 0.1679 | 0.0911 | 0.0915 | 0.00207 | 0.00147 | 0.00159 |
| 50  | 0.1323 | 0.0594 | 0.0591 | 0.00093 | 0.00054 | 0.00069 |
| 100 | 0.0890 | 0.0747 | 0.0739 | 0.00058 | 0.00089 | 0.00112 |
| 200 | 0.0658 | 0.0502 | 0.0504 | 0.00025 | 0.00042 | 0.00058 |

Net vs Parzen on KS: n=25 −0.5% (worse), n=50 +0.5%, n=100 +1.0%, n=200 −0.4%.
All within ±1% — a statistical tie. On MSE the net is uniformly *slightly worse*
than Parzen (8–38% higher), because the smooth fit trades a touch of point-wise
fidelity for smoothness.

### symmetric_bimodal

| n   | emp KS | par KS | net KS | emp MSE | par MSE | net MSE |
|-----|--------|--------|--------|---------|---------|---------|
| 25  | 0.1767 | 0.1516 | 0.1617 | 0.00478 | 0.00636 | 0.00850 |
| 50  | 0.1647 | 0.1489 | 0.1510 | 0.00376 | 0.00562 | 0.00710 |
| 100 | 0.0946 | 0.1101 | 0.1181 | 0.00131 | 0.00331 | 0.00453 |
| 200 | 0.0737 | 0.1038 | 0.1056 | 0.00082 | 0.00245 | 0.00312 |

Net vs Parzen on KS: −6.7%, −1.4%, −7.2%, −1.7% — net **worse at every n**.
On MSE the net is 20–37% worse than Parzen.

## Verdict

**No sample-efficiency advantage.** At small n the net is, at best, a statistical
tie with its Parzen target (single Gaussian, ±1% KS) and clearly *worse* on the
bimodal (−1.4% to −7.2% KS, +20–37% MSE). The net never beats the Parzen CDF it was
trained to copy. This is exactly the "faithful regressor" finding extended to the
low-n regime: small n does not unlock a generalization edge — if anything the
extra smoothing of the net hurts slightly when the target itself is biased
(over-smoothed Silverman window on a multimodal shape).

**Secondary finding (about Parzen, not the net):** the dumb empirical step CDF
*beats Parzen* on the bimodal — on KS at n≥100 (0.0946 vs 0.1101, 0.0737 vs 0.1038)
and on MSE at every n. Silverman over-smooths a bimodal, so Parzen's smoothing is
itself a liability there; the net inherits and amplifies that bias. Smoothing buys
sample efficiency only for the genuinely smooth single Gaussian, where Parzen
(and the net tied with it) beat the empirical CDF by ~2× on KS.

**Honest answer to the angle: the net is equal-or-worse, never more sample-efficient.**
The advantage of the MLP-on-Parzen approach, if any, is not low-n generalization
accuracy in 1D.
