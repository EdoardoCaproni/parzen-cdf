# The interactive lab

The project's pipeline, with every choice exposed as a control. It is not a demonstration that
the delivered configuration is good: it is a bench for taking the rejected road and watching
what happens. The values it starts with are the delivered ones, and everything else is live.

## Run

```bash
pip install -r ../requirements.txt   # adds fastapi + uvicorn to the project deps
python server.py                     # then http://localhost:8000
```

The server binds both loopbacks, so `localhost` works whichever one the browser resolves first.
Nothing is exposed to the network.

## The three stages

**1. Where the data comes from.** Build a mixture from a registry of five families, and every
error downstream is measured against that exact truth. Or load a file of numbers, which is the
situation the estimate is built for: then there is no truth, and every chart and tile that
needs one goes dark rather than showing something that looks like an answer.

**2. Estimate it with a Parzen window.** Watch the estimate assemble itself window by window.
The window-size strategy is the choice that matters more than any other on the page: seven of
them are available, including the classical schedule `h = h1/√n` with `h1` in your hands, and
`h1 = h·√n` is reported whichever one you pick.

**3. Train the estimator.** The mixture of logistic CDFs, or the free MLP it replaced, on the
labels from stage 2. Loss and metric charts, a live fit plot, and a diagram of the network
redrawn at every snapshot. Pause, stop, save, reset, and **Continue** for unbounded training
from the current epoch.

## Things worth doing

- **Slide the window towards zero and watch the trap.** The `KS vs ECDF` tile keeps improving
  while the error against the truth gets worse. The LSCV score, next to it, does not.
- **Switch the estimator to the MLP.** The monotonicity-violation tile stops reading zero.
  Turn on downstream rectification: the drawn curve becomes monotone and the tile does not
  change, because it measures the raw curve. The patch hides the symptom.
- **Prune a mixing weight** by clicking the outgoing edge of a component in the network
  diagram. The component switches off and the estimate is still a valid CDF, which is the
  theorem made visible. The incoming edge refuses to be cut, and says why.
- **Compare the teachers.** The leave-one-out labels against the self-inclusive ones, which
  contain the point they are labelling.
- **Ask for a domain taken from the true quantiles**, then load a file and ask again.

## Notes

Cross-validated selectors are O(n²) and capped at n = 2000. Checkpoints are `.pt` files with
the same layout the library writes, so `Estimate.load` can read them.

The visual system is documented in `../DESIGN.md`; the brief it was built to is `../PRODUCT.md`.
