# Product

## Register

product

## Users

Pietro (the project owner) and his professor, plus fellow students. Context: a university
office or lecture hall, laptop or projector, daylight. The job: *show* how the project's
pipeline works — from a known density, through Parzen-window estimation, to a neural CDF
regressor — step by step, live, with every knob exposed.

## Product Purpose

An educational lab for the parzen-cdf university project (not a thesis). It lets anyone
build an arbitrary 1-D mixture density, watch a Parzen-window estimate assemble itself
sample by sample, then configure and train an MLP CDF regressor with live loss/metric
charts, a live fit plot, and a weight-graph view of the network. Success: a viewer who has
never read the report understands the pipeline after one session, and the owner can probe
any parameter combination without touching code.

## Brand Personality

Precise, calm, didactic. A well-kept lab notebook with live instruments: cobalt ink on
white paper, unhurried, nothing decorative. The math is the show; the interface recedes.

## Anti-references

- SaaS dashboard clichés: hero metrics with gradient accents, identical card grids, glassmorphism.
- Dark "AI product" neon aesthetics; glowing charts.
- Jupyter-widget sprawl: dozens of unlabeled sliders with no narrative order.

## Design Principles

1. **The pipeline is the layout.** Three stages in the exact order of the method
   (distribution → Parzen → network); each stage's output is the next stage's input.
2. **Show the construction, not just the result.** Estimation and training are animated
   processes, not final images.
3. **Every number is honest.** Metrics are computed against the known truth and labeled
   with what they measure; caps and subsampling in views are stated, never silent.
4. **Knobs teach.** Every control uses the project's fixed terminology (Parzen Window,
   window size) and explains its effect in one line.
5. **Extensible by registry.** Distribution types, window shapes, and training techniques
   are pluggable lists, not hardcoded branches.

## Accessibility & Inclusion

WCAG AA: body text ≥ 4.5:1, chart series validated colorblind-safe with legends (never
color alone), full keyboard reachability for controls, `prefers-reduced-motion`
alternatives for the construction/training animations (instant final state).
