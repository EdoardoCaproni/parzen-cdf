"""Stage B -- inner ablation on the two simple mixtures (M1 symmetric, M2 asymmetric bimodal).

Thin caller of the shared engine at the IMPROVED baseline carried over from Stage A (lr 0.1, width
32, 5000 epochs, unconstrained) -- on N(0,1) that reached the Parzen ceiling. The sample-count axis
is averaged over 3 seeds (Stage A's single-seed n-trend was too noisy to read). The open questions:
does the net still match its target once multimodal, does bandwidth become the dominant lever
(Silverman over-smooths mixtures), and does monotonicity finally bite between the modes?

    python scripts/experiments/stage_b_simple_mixtures.py
"""

from _ablation import run

from parzen_cdf import data

IMPROVED = dict(n=2000, hidden=(32,), lr=1e-1, epochs=5000, mono=False, soft=0.0)

if __name__ == "__main__":
    run("M1 symmetric_bimodal", data.symmetric_bimodal(), IMPROVED,
        out_name="stage_b_symmetric_bimodal", e4_seeds=(0, 1, 2))
    run("M2 asymmetric_bimodal", data.asymmetric_bimodal(), IMPROVED,
        out_name="stage_b_asymmetric_bimodal", e4_seeds=(0, 1, 2))
