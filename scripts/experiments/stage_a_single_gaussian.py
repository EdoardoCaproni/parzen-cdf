"""Stage A -- deep inner ablation on the single Gaussian N(0,1).

Thin caller of the shared engine (``_ablation.run``) at the NAIVE baseline (width 16, lr 1e-2, 2000
epochs, Silverman h, n=2000, unconstrained) -- deliberately under-tuned, so the ablation reveals
which knobs matter. See docs/experiments.md for the findings.

    python scripts/experiments/stage_a_single_gaussian.py
"""

from _ablation import run

from parzen_cdf import data

NAIVE = dict(n=2000, hidden=(16,), lr=1e-2, epochs=2000, mono=False, soft=0.0)

if __name__ == "__main__":
    run("single_gaussian N(0,1)", data.single_gaussian(), NAIVE,
        out_name="stage_a_single_gaussian", grid_lo=-6.0, grid_hi=6.0, e4_seeds=(0,))
