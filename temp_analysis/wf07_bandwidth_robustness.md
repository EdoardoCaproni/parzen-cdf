# wf07 — Robustness to the window-size choice

**Question.** Is the MLP-on-Parzen net *less sensitive* to the target window size `h`
than the Parzen estimate is to its own `h`? If so, that would be a genuine advantage:
you'd pay less for getting the window wrong.

**Setup.** Symmetric bimodal, `n = 2000`, 3 seeds. For each seed we draw samples, compute
its Silverman `h`, and sweep `h` over factors `{0.1 … 7.0}` × Silverman. At each window
we measure the CDF gap = KS distance to the *true* CDF, for (a) the Parzen estimate and
(b) a sigmoidal net trained only on `(x_i, Parzen(x_i))` (hard monotonicity via downstream
rectification). Main sweep uses a width-32 net, 800 epochs. Grid 1000 pts on [-6, 6].
`torch.set_num_threads(1)` because the box was load-averaged ~80 on 14 cores.

## Result 1 — the sweep looks like the net is far more window-robust

Mean CDF gap (KS) across seeds:

| factor ×Silverman | Parzen KS | Net KS |
|---|---|---|
| 0.1  | 0.0132 | 0.0970 |
| 0.2  | 0.0125 | 0.0972 |
| 0.35 | 0.0148 | 0.0986 |
| 0.5  | 0.0206 | 0.1059 |
| 0.75 | 0.0332 | 0.1220 |
| 1.0  | 0.0463 | 0.1224 |
| 1.5  | 0.0698 | 0.1205 |
| 2.5  | 0.1142 | 0.1138 |
| 4.0  | 0.1687 | 0.1160 |
| 7.0  | 0.2519 | 0.1492 |

Spread of the gap across the window sweep (mean over seeds):

| | std | range | best | worst |
|---|---|---|---|---|
| Parzen | 0.0806 | 0.2394 | 0.0125 | 0.2519 |
| Net    | 0.0194 | 0.0589 | 0.0903 | 0.1492 |

- **std ratio (net/parzen) = 0.24**, **range ratio = 0.25**, **worst-case ratio = 0.59**.

So the net's KS-vs-window curve is ~4× flatter than Parzen's and its worst case is ~40%
better. Taken alone, that reads as a strong window-robustness advantage.

**But the catch is in the same table.** The net is flat at a *bad* level (~0.12). Where
Parzen is excellent (small-to-moderate `h`, KS 0.01–0.05) the net is ~0.10–0.12, i.e.
**~5× worse** (small-window subset factor ≤ 0.5: mean worst Parzen 0.021 vs net 0.106,
ratio 5.16). The net only "wins" at factors ≥ 2.5, i.e. windows so over-smoothed nobody
would pick them. Flatness here is "it can't fit anything sharp," not "it degrades
gracefully from a good estimate."

## Result 2 — the flat floor is mostly an under-capacity artifact, not smoothing

If the flat ~0.12 floor were a genuine implicit-regularization advantage, a bigger/longer-
trained net would stay flat. It does not. Re-running the small-window end (seed 0) with
more capacity, also reporting how well the net tracks its *own* Parzen target:

| f | h | net | net KS (vs true) | net-vs-Parzen KS (tracking) |
|---|---|---|---|---|
| 0.1 | 0.041 | w32 / 800ep  | 0.1231 | 0.1207 |
| 0.1 | 0.041 | w64 / 2000ep | 0.1260 | 0.1239 |
| 0.1 | 0.041 | **w128 / 2500ep** | **0.0493** | **0.0476** |
| 0.5 | 0.207 | w32 / 800ep  | 0.1225 | 0.1044 |
| 0.5 | 0.207 | w64 / 2000ep | 0.1270 | 0.1075 |
| 0.5 | 0.207 | **w128 / 2500ep** | **0.0366** | **0.0245** |

(Parzen KS at these windows: 0.018 and 0.023.)

With enough capacity the net **tracks Parzen back down** toward Parzen's good small-`h`
accuracy (tracking KS 0.048 / 0.025), recovering the established "faithful regressor"
behavior. The small net's flatness was under-fitting of the near-step small-`h` Parzen
CDF, not graceful smoothing. A net that *can* fit the target inherits the target's
window-sensitivity.

## Honest verdict

**No genuine advantage — capability only.** The eye-catching "4× flatter, 0.24 spread
ratio" is real but it is the signature of an under-capacity net that cannot reproduce a
sharp small-`h` target, not of a net that beats Parzen's window-sensitivity. In the window
regime you'd actually operate in (small-to-moderate `h`, where Parzen is accurate) the net
is several times *worse*, and when you give it enough capacity to be accurate there it
re-acquires Parzen's sensitivity. The only regime where the net's gap is below Parzen's is
gross over-smoothing (factor ≥ 2.5), which is not a window anyone would choose.

The one defensible reading: *if you are forced to over-smooth* (e.g. a deliberately large
safety window), the net's gap saturates (~0.15 at 7× Silverman) while Parzen's keeps
climbing (0.25). That is a narrow, low-value form of robustness, not the advantage the
project is looking for.

Numbers: `temp_analysis/wf07_bandwidth_robustness.py`, plot
`temp_analysis/wf07_bandwidth_robustness.png`.
