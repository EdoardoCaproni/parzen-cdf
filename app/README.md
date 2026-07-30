# parzen-lab

Educational web app for the project pipeline: build a mixture density with known truth, watch a
Parzen-window estimate assemble itself sample by sample, then train the neural CDF regressor
live (loss/metric charts, fit plot, weight-graph view; pause / stop / save / reset, plus
**Continue** for unbounded training from the current epoch and a snapshot-cadence override).
Every figure expands into a modal with wheel-zoom and drag-pan; in the network view, clicking a
connection prunes it (kept at zero through further training, click again to restore) with the
effect visible live on the CDF plot.

## Run

```bash
pip install -r ../requirements.txt   # adds fastapi + uvicorn to the project deps
python server.py                     # http://localhost:8000
```

## Architecture

- `server.py` — FastAPI backend. Reuses the project library directly:
  `parzen_cdf.parzen` (estimator + window-size selectors), `parzen_cdf.models.CDFNet`,
  `parzen_cdf.training` (penalties, `rectify_cdf`). REST for stages 1–2; a websocket
  (`/ws/train`) streams training snapshots and accepts `pause/resume/stop/save` commands.
  Checkpoints land in `app/checkpoints/` (gitignored).
- `static/` — dependency-free frontend (vanilla JS + SVG/canvas). `app.js` holds a small
  reusable `Chart` class, the construction animation, and the network diagram. Design tokens
  in `style.css`; see `../DESIGN.md` and `../PRODUCT.md`.

## Extending (registries, no UI changes needed)

- **New distribution family** → add an entry to `DISTRIBUTIONS` in `server.py`
  (param defaults + a factory returning a frozen scipy distribution).
- **New window shape** → add a `Kernel(cdf, pdf, std)` to `KERNELS` in
  `src/parzen_cdf/parzen.py`, and its pdf to `KERNELS_JS` in `static/app.js`
  (used only to animate the construction).
- **New window-size strategy** → add to `STRATEGIES` in `server.py`.
- **New training technique** → extend the config handling in `_train_loop` and add the
  control to the stage-3 panel in `static/index.html`.
