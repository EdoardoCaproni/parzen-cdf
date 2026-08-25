"""Interactive educational lab for the parzen-cdf pipeline.

Run with:  python app/server.py   →  http://localhost:8000

Three registries make the app extensible without touching the UI logic:
- ``DISTRIBUTIONS``: component families for building mixtures (add a family here);
- ``parzen_cdf.parzen.KERNELS``: window shapes (add a shape in the library);
- ``STRATEGIES``: window-size selectors.

Training runs as an asyncio task per websocket, streaming snapshots (loss, metrics,
curves, weights) and honouring pause / resume / stop / save / reset commands.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from parzen_cdf import parzen, training
from parzen_cdf.estimate import loo_parzen_cdf_targets
from parzen_cdf.models import CDFNet, MixtureCDFNet

APP_DIR = Path(__file__).parent
CHECKPOINT_DIR = APP_DIR / "checkpoints"
GRID_POINTS = 301
CV_MAX_N = 2000  # CV selectors are O(n^2 * candidates); guard the UI from freezing the server

# --------------------------------------------------------------------------------------------------
# Distribution components. Each entry: parameter spec (name -> default) and a factory returning a
# frozen scipy distribution. The mixture below combines any of these with weights.
# --------------------------------------------------------------------------------------------------

DISTRIBUTIONS = {
    "gaussian":    {"params": {"mean": 0.0, "std": 1.0},
                    "make": lambda p: st.norm(p["mean"], _pos(p["std"], "std"))},
    "uniform":     {"params": {"low": -1.0, "high": 1.0},
                    "make": lambda p: st.uniform(p["low"], _pos(p["high"] - p["low"], "high - low"))},
    "exponential": {"params": {"shift": 0.0, "rate": 1.0},
                    "make": lambda p: st.expon(p["shift"], 1.0 / _pos(p["rate"], "rate"))},
    "laplace":     {"params": {"mean": 0.0, "scale": 1.0},
                    "make": lambda p: st.laplace(p["mean"], _pos(p["scale"], "scale"))},
    "student_t":   {"params": {"df": 3.0, "loc": 0.0, "scale": 1.0},
                    "make": lambda p: st.t(_pos(p["df"], "df"), p["loc"], _pos(p["scale"], "scale"))},
}


def _pos(v: float, name: str) -> float:
    if v <= 0:
        raise ValueError(f"{name} must be positive")
    return v


class Mixture:
    """A weighted mixture of frozen scipy components with pdf, cdf, and exact sampling."""

    def __init__(self, components: list[dict]):
        if not components:
            raise ValueError("add at least one component")
        self.dists, weights = [], []
        for c in components:
            family = DISTRIBUTIONS.get(c["type"])
            if family is None:
                raise ValueError(f"unknown component type {c['type']!r}")
            params = {**family["params"], **{k: float(v) for k, v in c.get("params", {}).items()}}
            self.dists.append(family["make"](params))
            weights.append(float(c.get("weight", 1.0)))
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        self.weights = w / w.sum()

    def pdf(self, x):
        return sum(w * d.pdf(x) for w, d in zip(self.weights, self.dists))

    def cdf(self, x):
        return sum(w * d.cdf(x) for w, d in zip(self.weights, self.dists))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(self.dists), size=n, p=self.weights)
        out = np.empty(n)
        for i, d in enumerate(self.dists):
            mask = idx == i
            out[mask] = d.rvs(size=int(mask.sum()), random_state=rng)
        return out

    def grid(self, n: int = GRID_POINTS) -> np.ndarray:
        lo = min(d.ppf(0.001) for d in self.dists)
        hi = max(d.ppf(0.999) for d in self.dists)
        pad = 0.08 * (hi - lo)
        return np.linspace(lo - pad, hi + pad, n)


# --------------------------------------------------------------------------------------------------
# Window-size strategies (names shown in the UI; CV ones guarded by CV_MAX_N)
# --------------------------------------------------------------------------------------------------

STRATEGIES = {
    "manual":           lambda s, k, h: float(h),
    "silverman":        lambda s, k, h: parzen.silverman_bandwidth(s),
    "variance_matched": lambda s, k, h: parzen.variance_matched_bandwidth(s, kernel=k),
    "adaptive":         lambda s, k, h: parzen.adaptive_bandwidths(s, kernel=k),
    "likelihood_cv":    lambda s, k, h: parzen.likelihood_cv_bandwidth(s, kernel=k),
    "lscv":             lambda s, k, h: parzen.lscv_bandwidth(s, kernel=k),
}


def _pick_window_size(samples: np.ndarray, kernel: str, strategy: str, h_manual: float | None):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy in ("likelihood_cv", "lscv") and samples.size > CV_MAX_N:
        raise ValueError(f"cross-validation is O(n²) and capped at n = {CV_MAX_N}; "
                         f"use variance-matched (or fewer samples)")
    if strategy == "manual" and (h_manual is None or h_manual <= 0):
        raise ValueError("manual strategy needs a positive window size")
    return STRATEGIES[strategy](samples, kernel, h_manual)


def _parzen_payload(req: dict) -> dict:
    mix = Mixture(req["components"])
    n = int(req.get("n", 1000))
    if not 10 <= n <= 50000:
        raise ValueError("sample count must be between 10 and 50000")
    seed = int(req.get("seed", 0))
    kernel = req.get("kernel", "logistic")
    if kernel not in parzen.KERNELS:
        raise ValueError(f"unknown window shape {kernel!r}")
    samples = mix.sample(n, np.random.default_rng(seed))
    h = _pick_window_size(samples, kernel, req.get("strategy", "silverman"), req.get("h_manual"))
    grid = mix.grid()
    p_cdf = parzen.parzen_cdf(grid, samples, h, kernel)
    p_pdf = parzen.parzen_pdf(grid, samples, h, kernel)
    t_cdf, t_pdf = mix.cdf(grid), mix.pdf(grid)
    h_arr = np.asarray(h, dtype=float)
    return {
        "samples": samples.tolist(),
        "h": h_arr.tolist() if h_arr.ndim else float(h_arr),
        "h_summary": {"mean": float(np.mean(h_arr)), "min": float(np.min(h_arr)),
                      "max": float(np.max(h_arr)), "per_sample": bool(h_arr.ndim)},
        "grid": grid.tolist(),
        "parzen_pdf": p_pdf.tolist(), "parzen_cdf": p_cdf.tolist(),
        "truth_pdf": t_pdf.tolist(), "truth_cdf": t_cdf.tolist(),
        "ks_vs_truth": float(np.max(np.abs(p_cdf - t_cdf))),
        "empirical_cdf": _empirical_on_grid(samples, grid).tolist(),
    }


def _empirical_on_grid(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(samples), grid, side="right") / samples.size


# --------------------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------------------

app = FastAPI(title="parzen-cdf lab")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/registries")
def registries():
    return {
        "distributions": {k: v["params"] for k, v in DISTRIBUTIONS.items()},
        "kernels": list(parzen.KERNELS),
        "strategies": list(STRATEGIES),
        "activations": ["sigmoid", "tanh", "softplus", "silu"],
        "cv_max_n": CV_MAX_N,
    }


@app.post("/api/distribution")
async def distribution(req: dict):
    try:
        mix = Mixture(req["components"])
        grid = mix.grid()
        return {"grid": grid.tolist(), "pdf": mix.pdf(grid).tolist(), "cdf": mix.cdf(grid).tolist()}
    except (ValueError, KeyError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/parzen")
async def parzen_endpoint(req: dict):
    try:
        return await asyncio.to_thread(_parzen_payload, req)
    except (ValueError, KeyError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# --------------------------------------------------------------------------------------------------
# Training over a websocket
# --------------------------------------------------------------------------------------------------


class TrainSession:
    """State shared between the websocket command loop and the training task.

    The training context (``ctx``) survives between runs so that ``continue`` resumes from the
    current epoch, and ``mask`` holds manually pruned connections (kept at zero through any
    further training; the original value is stored so unpruning restores it).
    """

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.resume = asyncio.Event()
        self.resume.set()
        self.stopped = False
        self.model: "Model | None" = None
        self.config: dict = {}
        self.ctx: dict | None = None
        self.epoch = 0
        self.mask: dict[tuple[int, int, int], float] = {}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def apply_mask(self) -> None:
        if not self.mask or self.model is None:
            return
        with torch.no_grad():
            if isinstance(self.model, MixtureCDFNet):
                # Pruning a mixing weight means driving it out of the softmax.
                for (_l, _o, i) in self.mask:
                    self.model.u.data[i] = -20.0
                return
            # With positive weights the effective value is softplus(raw), which cannot be 0;
            # a very negative raw value makes it numerically zero instead.
            zero = -20.0 if self.model.monotone else 0.0
            for (l, o, i) in self.mask:
                self.model.weights[l].data[o, i] = zero

    def pruned_list(self) -> list[list[int]]:
        return [list(k) for k in self.mask]


Model = CDFNet | MixtureCDFNet


def _build_model(cfg: dict, samples: np.ndarray) -> Model:
    """The mixture is the delivered estimator; the free MLP stays available to be compared.

    The mixture is initialised from the samples (quantile centres), which is deterministic and
    consumes no randomness: two runs with the same data start from the same place whatever the
    seed does. The MLP is initialised at random, which is what makes its seed matter.
    """
    if cfg.get("estimator", "mixture") == "mixture":
        j = int(cfg.get("n_components", 12))
        if not 1 <= j <= 256:
            raise ValueError("components: 1 to 256")
        return MixtureCDFNet(j).init_from_samples(samples)
    hidden = [int(w) for w in cfg.get("hidden", [32])]
    if not hidden or any(not 1 <= w <= 256 for w in hidden) or len(hidden) > 6:
        raise ValueError("hidden layers: 1 to 6 layers, widths 1 to 256")
    return CDFNet(in_dim=1, hidden_sizes=hidden,
                  activation=cfg.get("activation", "sigmoid"),
                  monotone=cfg.get("monotonicity") == "positive")


def _layer_sizes(model: Model, cfg: dict) -> list[int]:
    if isinstance(model, MixtureCDFNet):
        return [1, model.n_components, 1]
    return [1, *[int(w) for w in cfg.get("hidden", [32])], 1]


def _n_weights(model: Model) -> int:
    if isinstance(model, MixtureCDFNet):
        return 3 * model.n_components
    return sum(w.numel() for w in model.weights)


def _weight_snapshot(model: Model) -> dict:
    """Effective weights, in the layered shape the diagram draws.

    The mixture has no layers of its own, but it is one: J sigmoid units read the same input
    and a convex combination reads them. So it is drawn as 1 -> J -> 1, with the width
    ``a_j = softplus(alpha_j)`` on the incoming edge, ``b_j`` as the unit's bias, and the
    mixing weight ``pi_j = softmax(u)_j`` on the outgoing edge. Everything shown is the
    effective value, never the raw parameter.
    """
    with torch.no_grad():
        if isinstance(model, MixtureCDFNet):
            a = torch.nn.functional.softplus(model.alpha).cpu().numpy()
            pi = torch.softmax(model.u, dim=0).cpu().numpy()
            b = model.b.cpu().numpy()
            return {
                "weights": [[[round(float(v), 4)] for v in a],
                            [[round(float(v), 4) for v in pi]]],
                "biases": [[round(float(v), 4) for v in b], [0.0]],
            }
        return {
            "weights": [model._weight(w).cpu().numpy().round(4).tolist() for w in model.weights],
            "biases": [b.cpu().numpy().round(4).tolist() for b in model.biases],
        }


def _eval_snapshot(model: Model, grid: np.ndarray, truth_cdf, truth_pdf, target_cdf,
                   rectify: bool) -> dict:
    gt = torch.as_tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        raw_cdf = model(gt).numpy().astype(float)
    viol = float(np.mean(np.diff(raw_cdf) < 0))
    if isinstance(model, MixtureCDFNet):
        # Closed-form density, and nothing to repair: no rectification, no clamp (D-10, D-11).
        with torch.no_grad():
            net_cdf, net_pdf = raw_cdf, model.pdf(gt).numpy().astype(float)
    elif rectify:
        net_cdf, net_pdf = training.rectify_cdf(raw_cdf, grid)
    else:
        net_cdf, net_pdf = raw_cdf, np.clip(np.gradient(raw_cdf, grid), 0.0, None)
    return {
        "net_cdf": np.round(net_cdf, 6).tolist(),
        "net_pdf": np.round(net_pdf, 6).tolist(),
        "ks_truth": float(np.max(np.abs(net_cdf - truth_cdf))),
        "ks_target": float(np.max(np.abs(net_cdf - target_cdf))),
        "pdf_mse": float(np.mean((net_pdf - truth_pdf) ** 2)),
        "mass": float(np.trapezoid(net_pdf, grid) if hasattr(np, "trapezoid")
                      else np.trapz(net_pdf, grid)),
        "violations": viol,
    }


def _cadence(cfg: dict, epochs: int | None) -> int:
    """Snapshot every N epochs: explicit override, else ~150 snapshots per bounded run."""
    override = int(cfg.get("snapshot_every") or 0)
    if override > 0:
        return override
    return max(1, (epochs or 5000) // 150)


async def _setup_training(ws: WebSocket, sess: TrainSession, cfg: dict):
    """Build data, targets, model, and optimizer; store the context; send the hello frame."""
    payload = await asyncio.to_thread(_parzen_payload, cfg)
    samples = np.asarray(payload["samples"])
    grid = np.asarray(payload["grid"])
    truth_cdf, truth_pdf = np.asarray(payload["truth_cdf"]), np.asarray(payload["truth_pdf"])
    h = payload["h"]

    # The teacher: what the network is asked to reproduce at the sample points. The delivered
    # recipe leaves each point out of its own label and uses a window deliberately narrower
    # than the one that minimises the Parzen estimator's own error, letting the network's
    # limited capacity average the noise away instead of copying it.
    kernel = cfg.get("kernel", "logistic")
    target = cfg.get("target", "parzen_loo")
    scale = float(cfg.get("teacher_scale", 0.5))
    if not 0.05 <= scale <= 4.0:
        raise ValueError("teacher window scale must be between 0.05 and 4")
    if target == "empirical":
        order = np.argsort(np.argsort(samples))
        targets_np = (order + 0.5) / samples.size          # F_n(x_i) = (rank − 0.5)/n
        target_cdf = np.asarray(payload["empirical_cdf"])
    elif target == "parzen":
        # Self-inclusive: every label contains the point it labels, worth K(0) = 1/2 of it.
        targets_np = parzen.parzen_cdf(samples, samples, h, kernel)
        target_cdf = np.asarray(payload["parzen_cdf"])
    else:
        h_teacher = np.asarray(h, dtype=float) * scale
        targets_np = loo_parzen_cdf_targets(samples, h_teacher, kernel)
        target_cdf = parzen.parzen_cdf(grid, samples, h_teacher, kernel)

    epochs = int(cfg.get("epochs", 5000))
    if not 1 <= epochs <= 100000:
        raise ValueError("epochs must be between 1 and 100000")
    training.set_seed(int(cfg.get("net_seed", 0)))   # before construction: that is where
    model = _build_model(cfg, samples)              # initialisation happens (defect B8)
    opt_cls = torch.optim.SGD if cfg.get("optimizer") == "sgd" else torch.optim.Adam
    n_weights = _n_weights(model)
    is_mixture = isinstance(model, MixtureCDFNet)

    sess.model, sess.config = model, cfg
    sess.epoch = 0
    sess.mask = {}
    sess.ctx = {
        "inputs": torch.as_tensor(samples, dtype=torch.float32),
        "targets": torch.as_tensor(targets_np, dtype=torch.float32),
        "optimizer": opt_cls(model.parameters(), lr=float(cfg.get("lr", 0.03))),
        "mse": torch.nn.MSELoss(),
        # Both penalties and the rectification exist to repair an MLP. With the mixture there
        # is nothing to repair, and running them anyway would hide that (D-10, D-11).
        "mono_w": 0.0 if is_mixture else (
            float(cfg.get("mono_weight", 0.0)) if cfg.get("monotonicity") == "penalty" else 0.0),
        "curv_w": 0.0 if is_mixture else float(cfg.get("curv_weight", 0.0)),
        "rectify": False if is_mixture else bool(cfg.get("rectify", False)),
        "penalty_pts": torch.linspace(float(grid[0]), float(grid[-1]), 256),
        "grid": grid, "truth_cdf": truth_cdf, "truth_pdf": truth_pdf, "target_cdf": target_cdf,
        "epochs": epochs,
        "every": _cadence(cfg, epochs),
        # weights ride along on every snapshot unless the net is huge (then every 5th)
        "weights_stride": 5 if n_weights > 20000 else 1,
    }

    show = np.linspace(0, samples.size - 1, min(samples.size, 250)).astype(int)
    order_idx = np.argsort(samples)[show]
    await ws.send_json({
        "type": "hello",
        "grid": grid.tolist(),
        "truth_cdf": truth_cdf.tolist(), "truth_pdf": truth_pdf.tolist(),
        "parzen_cdf": payload["parzen_cdf"], "parzen_pdf": payload["parzen_pdf"],
        "target_points": {"x": samples[order_idx].tolist(), "y": targets_np[order_idx].tolist(),
                          "shown": int(show.size), "total": int(samples.size)},
        "layer_sizes": _layer_sizes(model, cfg),
        "estimator": "mixture" if is_mixture else "mlp",
        "n_weights": int(n_weights),
        "pruned": sess.pruned_list(),
        **_weight_snapshot(model),
    })


def _full_snapshot(sess: TrainSession, loss: float | None, elapsed: float,
                   target_epoch: int | None) -> dict:
    c = sess.ctx
    return {"type": "snapshot", "epoch": sess.epoch, "epochs": target_epoch,
            "loss": loss, "elapsed": elapsed, "pruned": sess.pruned_list(),
            **_eval_snapshot(sess.model, c["grid"], c["truth_cdf"], c["truth_pdf"],
                             c["target_cdf"], c["rectify"]),
            **_weight_snapshot(sess.model)}


async def _run_epochs(ws: WebSocket, sess: TrainSession, n_epochs: int | None):
    """Train for ``n_epochs`` more (None = unbounded, until stop), from the current epoch."""
    c = sess.ctx
    model, optimizer, mse = sess.model, c["optimizer"], c["mse"]
    target_epoch = None if n_epochs is None else sess.epoch + n_epochs
    t0 = time.monotonic()
    snaps = 0
    loss_val = None
    sess.apply_mask()
    while target_epoch is None or sess.epoch < target_epoch:
        await sess.resume.wait()
        if sess.stopped:
            break
        sess.epoch += 1
        optimizer.zero_grad()
        loss = mse(model(c["inputs"]), c["targets"])
        if c["mono_w"] > 0:
            loss = loss + c["mono_w"] * training.monotonicity_penalty(model, c["penalty_pts"])
        if c["curv_w"] > 0:
            loss = loss + c["curv_w"] * training.curvature_penalty(model, c["penalty_pts"])
        loss.backward()
        optimizer.step()
        sess.apply_mask()   # pruned connections stay pruned through training
        loss_val = float(loss.item())

        if sess.epoch % c["every"] == 0 or sess.epoch == target_epoch:
            snaps += 1
            elapsed = time.monotonic() - t0
            if snaps % c["weights_stride"] == 0 or sess.epoch == target_epoch:
                snap = _full_snapshot(sess, loss_val, elapsed, target_epoch)
            else:
                snap = {"type": "snapshot", "epoch": sess.epoch, "epochs": target_epoch,
                        "loss": loss_val, "elapsed": elapsed,
                        **_eval_snapshot(model, c["grid"], c["truth_cdf"], c["truth_pdf"],
                                         c["target_cdf"], c["rectify"])}
            await ws.send_json(snap)
        if sess.epoch % 10 == 0:
            await asyncio.sleep(0)  # let pause/stop/save/prune commands in

    # closing frame: full state (weights + pruned) so the UI is exact after stop/finish
    await ws.send_json(_full_snapshot(sess, loss_val, time.monotonic() - t0, target_epoch))
    await ws.send_json({"type": "done", "stopped": sess.stopped,
                        "epoch": sess.epoch, "elapsed": time.monotonic() - t0})


def _save_checkpoint(sess: TrainSession, name: str) -> str:
    if sess.model is None:
        raise ValueError("nothing to save yet; start a training run first")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", name or "run").strip("_") or "run"
    path = CHECKPOINT_DIR / f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({"state_dict": sess.model.state_dict(), "config": sess.config,
                "epoch": sess.epoch, "pruned": sess.pruned_list()}, path)
    return str(path.relative_to(APP_DIR.parent))


def _toggle_prune(sess: TrainSession, cmd: str, msg: dict) -> None:
    if sess.model is None:
        raise ValueError("no model yet; start a training run first")
    l, o, i = int(msg["l"]), int(msg["o"]), int(msg["i"])
    if isinstance(sess.model, MixtureCDFNet):
        # Switching off a component (its outgoing edge) leaves a convex combination of
        # logistic CDFs, so the estimate stays a CDF: that is the point worth showing.
        # An incoming edge is different. With a_j = 0 the unit is the constant 1/2, so
        # F(-inf) = pi_j/2 > 0 and F is no longer a CDF at all. It is refused rather than
        # executed, because the guarantee this estimator offers is exactly that it holds
        # at every parameter value.
        if l != 1:
            raise ValueError("with the mixture only the outgoing edges can be pruned: zeroing "
                             "an incoming edge makes the unit constant at 1/2, which lifts "
                             "F(-inf) off zero and stops it being a CDF")
        if not 0 <= i < sess.model.n_components:
            raise ValueError("component index out of range")
        with torch.no_grad():
            if cmd == "prune" and (l, o, i) not in sess.mask:
                sess.mask[(l, o, i)] = float(sess.model.u.data[i])
                sess.apply_mask()
            elif cmd == "unprune":
                orig = sess.mask.pop((l, o, i), None)
                if orig is not None:
                    sess.model.u.data[i] = orig
        return
    w = sess.model.weights[l]
    if not (0 <= o < w.shape[0] and 0 <= i < w.shape[1]):
        raise ValueError("connection index out of range")
    with torch.no_grad():
        if cmd == "prune" and (l, o, i) not in sess.mask:
            sess.mask[(l, o, i)] = float(w.data[o, i])
            sess.apply_mask()
        elif cmd == "unprune":
            orig = sess.mask.pop((l, o, i), None)
            if orig is not None:
                w.data[o, i] = orig


@app.websocket("/ws/train")
async def train_ws(ws: WebSocket):
    await ws.accept()
    sess = TrainSession()

    async def run(cfg, more: int | None = None, fresh: bool = True):
        try:
            if fresh:
                await _setup_training(ws, sess, cfg)
                more = sess.ctx["epochs"]
            await _run_epochs(ws, sess, more)
        except (ValueError, KeyError) as e:
            await ws.send_json({"type": "error", "message": str(e)})
        except WebSocketDisconnect:
            pass

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                if sess.running:
                    await ws.send_json({"type": "error", "message": "already training; stop first"})
                    continue
                sess.stopped = False
                sess.resume.set()
                sess.task = asyncio.create_task(run(msg.get("config", {})))
            elif cmd == "continue":
                if sess.running:
                    await ws.send_json({"type": "error", "message": "already training"})
                    continue
                if sess.ctx is None:
                    await ws.send_json({"type": "error", "message": "nothing to continue; train first"})
                    continue
                if msg.get("snapshot_every"):
                    sess.ctx["every"] = max(1, int(msg["snapshot_every"]))
                more = msg.get("epochs")          # None → unbounded, until stop
                sess.stopped = False
                sess.resume.set()
                sess.task = asyncio.create_task(run(sess.config, int(more) if more else None, fresh=False))
            elif cmd in ("prune", "unprune"):
                try:
                    _toggle_prune(sess, cmd, msg)
                    if not sess.running:   # idle: push the effect on the curves immediately
                        await ws.send_json(_full_snapshot(sess, None, 0.0, None))
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
            elif cmd == "pause":
                sess.resume.clear()
                await ws.send_json({"type": "status", "state": "paused"})
            elif cmd == "resume":
                sess.resume.set()
                await ws.send_json({"type": "status", "state": "running"})
            elif cmd == "stop":
                sess.stopped = True
                sess.resume.set()
            elif cmd == "save":
                try:
                    await ws.send_json({"type": "saved", "path": _save_checkpoint(sess, msg.get("name", ""))})
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        sess.stopped = True
        sess.resume.set()
        if sess.task:
            sess.task.cancel()


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

async def _serve_loopback(port: int = 8000):
    """Serve on BOTH loopbacks (127.0.0.1 and ::1): some browsers resolve ``localhost`` to the
    IPv6 loopback first and do not fall back, so an IPv4-only bind looks dead from localhost.
    Still local-only; nothing is exposed to the network."""
    servers = [uvicorn.Server(uvicorn.Config(app, host=h, port=port)) for h in ("127.0.0.1", "::1")]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(_serve_loopback())
