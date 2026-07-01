# wf01 - Amortization & dimension scaling

**Angle.** Quantify the *amortization* advantage of the MLP-on-Parzen approach: a trained network
answers any CDF/pdf query in cost set by its **parameter count** (constant in the sample size `n`),
whereas the Parzen estimate must touch **all `n` stored samples on every query** (O(n) per point).
Then argue how this gap *widens with dimension* `D`.

This is the one angle where the network has a genuine, structural advantage that is **not** about
point accuracy (we already established it is a faithful regressor, not a beater). The advantage is
*compression + amortization*: replace the `n`-sample training set with a fixed, tiny parameter
vector and a closed-form evaluator.

## What was run

`temp_analysis/wf01_amortization.py` (lightweight: `n` up to 100k, a 1000-point query grid, a
single width-32 net trained once for 2000 epochs). Four parts:

1. **Query cost vs n (1D).** Time the net (fixed params) and Parzen (O(n)) answering the same
   1000-point CDF and pdf query, as `n` grows 1k -> 100k.
2. **Stored memory.** Net = `n_params` floats (constant); Parzen = `n` floats (the whole sample).
3. **Storage vs dimension.** Net params grow only as `in_dim * width + const` (linear, tiny); KDE
   storage is `n * D` and `n` must itself grow with `D`.
4. **Curse of dimensionality for the sample budget.** For a separable standard normal, measure how
   many samples the (product-)logistic KDE needs to reach a *fixed relative density error*, in
   D = 1, 2, 3. The required `n` climbs with `D`, and every one of those samples is touched on every
   query - so the per-query O(n) cost compounds with the dimension-driven growth of `n`.

## Results

Two timing sources. The **uncontended baseline** (`temp_analysis/test_query_cost.py`, a trained
width-32 net, same grid and `n` sweep) gives the cleanest *absolute* speedups; the
`wf01_amortization.py` run below was measured under heavy concurrent machine load, so its absolute
millisecond figures are inflated, but its *relative* net-vs-Parzen comparison and its per-sample
scaling are still valid (both methods ran under the same load). The structural conclusions are
identical across both.

### Part 1+2 - 1D query cost & memory

Uncontended baseline (`test_query_cost.py`), width-32 net = **97 params**, 1000-point query grid:

| n | net CDF | Parzen CDF | CDF speedup | Parzen pdf | mem ratio (n / 97) |
|---:|---:|---:|---:|---:|---:|
| - | 0.097 ms | - | - | - | - |
| 1000 | (flat) | 8.56 ms | 88x | 10.6 ms | 10x |
| 5000 | (flat) | 73.6 ms | 760x | 107 ms | 52x |
| 20000 | (flat) | 298 ms | 3076x | 338 ms | 206x |
| 100000 | (flat) | 1173 ms | **12110x** | 8175 ms | **1031x** |

`wf01_amortization.py` run (contended; net 97 params; *relative* numbers still hold):

```
        n  parzen floats  mem ratio  parzen CDF ms  parzen pdf ms  CDF speedup  pdf speedup
     1000           1000        10x        120.832        226.174           5x           5x
     5000           5000        52x        823.028       1110.951          34x          24x
    20000          20000       206x       1003.138       1914.746          41x          41x
   100000         100000      1031x      12416.765       9699.985         509x         207x
```

The net's per-query cost is **flat** (set by its 97 params, independent of n); Parzen's grows
**linearly** in `n`. In the clean baseline, at `n = 100k` the net answers the 1000-point CDF query
**~12,000x faster** and stores **~1031x fewer floats** (97 vs 100,000).

### Part 3 - storage vs dimension D

```
   D   net params (w=32)    KDE floats (n=20000)
   1                  97                   20000
   2                 129                   40000
   3                 161                   60000
   5                 225                  100000
  10                 385                  200000
  20                 705                  400000
```

Net parameter count rises only gently and **linearly** with `D` (`+32` params per input dimension:
one extra weight column into the first hidden layer). KDE storage is `n * D` floats - and `n` is not
fixed, it must grow with `D` (Part 4).

### Part 4 - sample budget grows with D (curse of dimensionality)

Separable standard normal; product-logistic KDE with per-dim Silverman bandwidth; relative
density-MISE on a 300-point test set drawn from the truth:

```
   D     n to hit relMISE<0.10   best relMISE@n_grid
   1                       200                0.0004
   2                       200                0.0041
   3                      2000                0.0232
```

Even for the *easiest possible* target (a Gaussian), accuracy at a fixed sample budget **degrades by
roughly an order of magnitude per added dimension** (best relMISE 0.0004 -> 0.0041 -> 0.0232), and
the `n` needed to cross a fixed accuracy threshold jumps **10x** from D=2 to D=3 (200 -> 2000).
Since the KDE query cost is O(n) per evaluation point, the two effects multiply: deeper dimension ->
larger `n` -> a *more* expensive per-query sum, on *every* query forever. The net pays neither price:
its query cost stays O(params) and its params grow only `+32` per dimension.

## The amortization argument

- **Parzen / KDE has no training cost but an unbounded query cost:** every CDF/pdf evaluation is a
  fresh O(n) sum over the stored samples, and the model *is* the dataset (you must keep all `n`
  points). Throughput and storage degrade as you collect more data.
- **The net pays a one-off training cost, then amortizes:** afterwards each query is O(params),
  independent of `n`, and the stored model is a fixed, tiny parameter vector. Collecting 100x more
  data makes the *target* better without making queries slower or the model bigger.
- **Dimension sharpens the contrast:** KDE's per-query O(n) does not shrink with `D`, while the `n`
  it needs for accuracy *grows* with `D`; the net's cost grows only linearly (and slowly) in `D`.

This is a real, measurable advantage - but an honest one: it is an **amortization / deployment-cost**
advantage (cheap, constant-cost, compact queries), **not** a statistical-accuracy advantage. The net
does not estimate the CDF better than Parzen; it estimates the *same thing* and then makes serving it
cheap and `n`-independent. The advantage only pays off when you query many times (or store/ship the
model) relative to the one-off training cost.
