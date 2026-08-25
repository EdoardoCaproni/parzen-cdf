/* parzen-lab — front-end. Three stages wired to the FastAPI backend:
   1) mixture builder → /api/distribution
   2) Parzen construction (animated client-side from the samples) → /api/parzen
   3) live MLP training over /ws/train (snapshots: loss, metrics, curves, weights),
      with continue/unbounded runs, manual pruning of connections, and a modal
      pan-and-zoom view for every figure. */

"use strict";

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const REDUCED_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

const C = {
  truth: () => css("--c-truth"),
  parzen: () => css("--c-parzen"),
  net: () => css("--c-net"),
  emp: () => css("--c-emp"),
  kernel: () => css("--c-kernel"),
  wpos: () => css("--w-pos"),
  wneg: () => css("--w-neg"),
  hairline: () => css("--hairline"),
  baseline: () => css("--baseline"),
  ink2: () => css("--ink-2"),
};

function fmt(v, sig = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return Number(v.toPrecision(sig)).toString();
}
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

function nearestIndex(arr, v) {
  let lo = 0, hi = arr.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid; else hi = mid; }
  return v - arr[lo] <= arr[hi] - v ? lo : hi;
}

/* ---------------------------------------------------------------- kernels (mirror of the
   server registry; used only to animate the construction — the final curve is the server's) */

const KERNELS_JS = {
  logistic: (u) => { const s = 1 / (1 + Math.exp(-u)); return s * (1 - s); },
  gaussian: (u) => Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI),
  box: (u) => (Math.abs(u) <= 1 ? 0.5 : 0),
  epanechnikov: (u) => (Math.abs(u) <= 1 ? 0.75 * (1 - u * u) : 0),
  triangular: (u) => Math.max(1 - Math.abs(u), 0),
};
const KERNEL_NOTES = {
  logistic: "The project's window: its integral is the sigmoid, so the CDF estimate is closed-form.",
  gaussian: "The classic smooth bell; infinite support.",
  box: "The original “square” Parzen window; the density estimate becomes a staircase.",
  epanechnikov: "Compact support; minimizes asymptotic MSE among all windows.",
  triangular: "Compact support, piecewise linear.",
};
const STRATEGY_NOTES = {
  manual: "You pick h yourself. Slide it to feel over- and under-smoothing.",
  silverman: "Rule of thumb derived for a Gaussian window; tends to over-smooth multimodal shapes.",
  variance_matched: "Silverman rescaled by the window's own spread; the study's best selector at large n.",
  adaptive: "Abramson: a per-sample size, smaller where data is dense. Best on disparate-scale densities.",
  likelihood_cv: "Leave-one-out likelihood search. Strong at small n; O(n²), capped at n ≤ 2000.",
  lscv: "Least-squares CV, minimizes integrated squared error; the study's best under 2k samples. O(n²), capped at n ≤ 2000.",
};

/* ---------------------------------------------------------------- a small SVG line chart */

function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const span = hi - lo;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = span / n / step0;
  const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return ticks;
}

let _chartUid = 0;

class Chart {
  /** card: element containing .chart-head .legend and .chart-body */
  constructor(card, opts = {}) {
    this.body = card.querySelector(".chart-body");
    this.legendEl = card.querySelector(".legend");
    this.opts = { height: 240, logY: false, zeroBase: false, yMaxPad: 1.06, zoomable: false, ...opts };
    this.uid = ++_chartUid;
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("class", "chart-svg");
    this.svg.setAttribute("role", "img");
    this.body.style.position = "relative";
    this.body.appendChild(this.svg);
    this.tooltip = document.createElement("div");
    this.tooltip.className = "tooltip";
    this.body.appendChild(this.tooltip);
    this.series = [];
    this._raf = 0;
    this._zoom = null;      // [x0, x1] domain override when zoomed
    this._mirror = null;    // a modal Chart kept in sync with this one
    this._ro = new ResizeObserver(() => this._schedule());
    this._ro.observe(this.body);
  }

  set(series) {
    this.series = series;
    this._schedule();
    if (this._mirror) this._mirror.set(series);
  }

  destroy() {
    this._ro.disconnect();
    this.svg.remove();
    this.tooltip.remove();
  }

  _schedule() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._render(); });
  }

  _yT(v) { return this.opts.logY ? Math.log10(Math.max(v, 1e-300)) : v; }

  _render() {
    const S = this.series.filter((s) => s.x.length);
    const W = Math.max(this.body.clientWidth, 320);
    const H = this.opts.height;
    const m = { l: 48, r: 14, t: 10, b: 28 };
    if (!S.length) { this.svg.replaceChildren(); this._legend(); return; }

    let fx0 = Infinity, fx1 = -Infinity;
    for (const s of S) { fx0 = Math.min(fx0, s.x[0]); fx1 = Math.max(fx1, s.x[s.x.length - 1]); }
    if (!isFinite(fx0)) { this.svg.replaceChildren(); return; }
    if (fx1 === fx0) fx1 = fx0 + 1;
    this._full = [fx0, fx1];
    let x0 = fx0, x1 = fx1;
    if (this._zoom) {
      x0 = Math.max(fx0, this._zoom[0]);
      x1 = Math.min(fx1, this._zoom[1]);
      if (!(x1 > x0)) { this._zoom = null; x0 = fx0; x1 = fx1; }
    }

    // visible index range per series, then the y extent over just the visible points
    const ranges = S.map((s) => {
      if (!this._zoom) return [0, s.x.length - 1];
      return [Math.max(0, nearestIndex(s.x, x0) - 1), Math.min(s.x.length - 1, nearestIndex(s.x, x1) + 1)];
    });
    let y0 = Infinity, y1 = -Infinity;
    S.forEach((s, si) => {
      for (let i = ranges[si][0]; i <= ranges[si][1]; i++) {
        const y = s.y[i];
        if (this.opts.logY && !(y > 0)) continue;
        const ty = this._yT(y);
        if (ty < y0) y0 = ty;
        if (ty > y1) y1 = ty;
      }
    });
    if (!isFinite(y0)) { y0 = 0; y1 = 1; }
    if (this.opts.zeroBase && !this.opts.logY) y0 = Math.min(y0, 0);
    if (y1 === y0) y1 = y0 + 1;
    y1 *= y1 > 0 ? this.opts.yMaxPad : 1;

    const xs = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const ys = (v) => H - m.b - ((this._yT(v) - y0) / (y1 - y0)) * (H - m.t - m.b);
    this._map = { xs, ys, x0, x1, m, W, H };

    let yTicks;
    if (this.opts.logY) {
      const decades = Array.from({ length: Math.floor(y1) - Math.ceil(y0) + 1 }, (_, i) => Math.ceil(y0) + i);
      yTicks = (decades.length >= 2 ? decades : niceTicks(y0, y1, 3)).map((d) => Math.pow(10, d));
    } else {
      yTicks = niceTicks(y0, y1, 4);
    }
    const xTicks = niceTicks(x0, x1, 6);

    const clipId = `clip${this.uid}`;
    let g = `<clipPath id="${clipId}"><rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H - m.b}"></rect></clipPath>`;
    g += `<g class="grid">`;
    for (const t of yTicks) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(t)}" y2="${ys(t)}"></line>`;
    g += `</g>`;
    g += `<line class="axisline" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"></line>`;
    for (const t of yTicks) g += `<text x="${m.l - 6}" y="${ys(t) + 3.5}" text-anchor="end">${this.opts.logY ? t.toExponential(1).replace(".0", "") : fmt(t)}</text>`;
    for (const t of xTicks) g += `<text x="${xs(t)}" y="${H - m.b + 16}" text-anchor="middle">${fmt(t)}</text>`;

    g += `<g clip-path="url(#${clipId})">`;
    S.forEach((s, si) => {
      const [i0, i1] = ranges[si];
      if (s.type === "dots") {
        let dots = "";
        for (let i = i0; i <= i1; i++) dots += `<circle cx="${xs(s.x[i])}" cy="${ys(s.y[i])}" r="3" fill="${s.color}" stroke="var(--bg)" stroke-width="1.5"></circle>`;
        g += `<g opacity="0.85">${dots}</g>`;
        return;
      }
      let d = "";
      let started = false;
      for (let i = i0; i <= i1; i++) {
        const y = s.y[i];
        if (this.opts.logY && !(y > 0)) { started = false; continue; }
        const px = xs(s.x[i]).toFixed(1), py = ys(y).toFixed(1);
        if (!started) { d += `M${px},${py}`; started = true; }
        else if (s.type === "step") d += `H${px}V${py}`;
        else d += `L${px},${py}`;
      }
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round" opacity="${s.opacity ?? 1}"></path>`;
    });
    g += `</g>`;

    g += `<line class="xhair" id="xh" y1="${m.t}" y2="${H - m.b}" visibility="hidden"></line>`;
    g += `<rect id="ov" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" tabindex="0" style="outline:none${this.opts.zoomable ? ";cursor:crosshair" : ""}"></rect>`;
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svg.setAttribute("height", H);
    this.svg.innerHTML = g;
    this.tooltip.classList.remove("show");  // a re-render invalidates a hover in progress
    this._legend();
    this._wireHover();
  }

  _legend() {
    if (!this.legendEl) return;
    const items = this.series.filter((s) => s.label && s.inLegend !== false);
    if (items.length < 2) { this.legendEl.replaceChildren(); return; }
    this.legendEl.innerHTML = items.map((s) =>
      `<span class="key"><span class="swatch ${s.dash ? "dashed" : ""} ${s.type === "dots" ? "dot" : ""}" style="${s.type === "dots" ? "background" : "border-top-color"}:${s.color}"></span>${s.label}</span>`).join("");
  }

  _domainX(clientX) {
    const r = this.svg.getBoundingClientRect();
    const vx = ((clientX - r.left) / r.width) * this._map.W;
    return this._map.x0 + ((vx - this._map.m.l) / (this._map.W - this._map.m.l - this._map.m.r)) * (this._map.x1 - this._map.x0);
  }

  _wireHover() {
    const ov = this.svg.querySelector("#ov");
    const xh = this.svg.querySelector("#xh");
    if (!ov) return;
    const ref = this.series.find((s) => s.type !== "dots" && s.x.length);
    if (!ref) return;
    const show = (idx) => {
      const x = ref.x[idx];
      xh.setAttribute("x1", this._map.xs(x));
      xh.setAttribute("x2", this._map.xs(x));
      xh.setAttribute("visibility", "visible");
      let rows = `<div class="tt-x">x = ${fmt(x, 4)}</div>`;
      for (const s of this.series) {
        if (s.type === "dots" || s.inTooltip === false || !s.x.length) continue;
        const j = s.x === ref.x ? idx : nearestIndex(s.x, x);
        rows += `<div class="tt-row"><span class="k" style="border-top-color:${s.color};${s.dash ? "border-top-style:dashed" : ""}"></span><span class="v">${fmt(s.y[j], 4)}</span><span class="l">${s.label || ""}</span></div>`;
      }
      this.tooltip.innerHTML = rows;
      this.tooltip.classList.add("show");
      const bw = this.body.clientWidth;
      const px = (this._map.xs(x) / this._map.W) * bw;
      const flip = px > bw - 170;
      this.tooltip.style.left = flip ? "" : `${px + 14}px`;
      this.tooltip.style.right = flip ? `${bw - px + 14}px` : "";
      this.tooltip.style.top = "12px";
      this._idx = idx;
    };
    const hide = () => { xh.setAttribute("visibility", "hidden"); this.tooltip.classList.remove("show"); };
    ov.addEventListener("pointermove", (e) => {
      if (this._panning) {
        const dx = this._domainX(e.clientX) - this._panStartDomain;
        const [z0, z1] = this._panStartZoom;
        const [f0, f1] = this._full;
        const shift = clamp(-dx, f0 - z0, f1 - z1);
        this._zoom = [z0 + shift, z1 + shift];
        this._schedule();
        return;
      }
      show(nearestIndex(ref.x, this._domainX(e.clientX)));
    });
    ov.addEventListener("pointerleave", hide);
    ov.addEventListener("keydown", (e) => {
      const step = Math.max(1, Math.round(ref.x.length / 60));
      if (e.key === "ArrowRight") show(Math.min((this._idx ?? 0) + step, ref.x.length - 1));
      else if (e.key === "ArrowLeft") show(Math.max((this._idx ?? 0) - step, 0));
      else if (e.key === "Escape") hide();
      else return;
      e.preventDefault();
    });
    ov.addEventListener("focus", () => show(this._idx ?? Math.floor(ref.x.length / 2)));
    ov.addEventListener("blur", hide);

    if (!this.opts.zoomable) return;
    ov.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [f0, f1] = this._full;
      const cur = this._zoom || [f0, f1];
      const span = cur[1] - cur[0];
      const factor = e.deltaY > 0 ? 1.25 : 0.8;
      const ns = clamp(span * factor, (f1 - f0) / 500, f1 - f0);
      const cx = clamp(this._domainX(e.clientX), f0, f1);
      let n0 = cx - ((cx - cur[0]) / span) * ns;
      n0 = clamp(n0, f0, f1 - ns);
      this._zoom = ns >= (f1 - f0) * 0.999 ? null : [n0, n0 + ns];
      this._schedule();
    }, { passive: false });
    ov.addEventListener("pointerdown", (e) => {
      if (!this._zoom) return;
      this._panning = true;
      this._panStartDomain = this._domainX(e.clientX);
      this._panStartZoom = [...this._zoom];
      ov.setPointerCapture(e.pointerId);
    });
    ov.addEventListener("pointerup", () => { this._panning = false; });
    ov.addEventListener("dblclick", () => { this._zoom = null; this._schedule(); });
  }
}

/* ---------------------------------------------------------------- global state */

const state = {
  reg: null,
  components: [],
  dist: null,          // {grid,pdf,cdf}
  parzen: null,        // last /api/parzen payload
  parzenCfg: null,     // config snapshot used for that payload (training reuses it verbatim)
  train: {
    ws: null, running: false, paused: false, started: false, unbounded: false,
    hist: null, hello: null,
    net: null,         // {sizes, weights, biases, pruned:Set("l:o:i")}
  },
};

/* ================================================================ stage 1: distribution */

const PRESETS = {
  gauss: [{ type: "gaussian", params: { mean: 0, std: 1 }, weight: 1 }],
  sym: [{ type: "gaussian", params: { mean: -2, std: 0.7 }, weight: 0.5 },
        { type: "gaussian", params: { mean: 2, std: 0.7 }, weight: 0.5 }],
  asym3: [{ type: "gaussian", params: { mean: -2, std: 0.5 }, weight: 0.3 },
          { type: "gaussian", params: { mean: 1, std: 1 }, weight: 0.5 },
          { type: "gaussian", params: { mean: 4, std: 0.3 }, weight: 0.2 }],
  spike: [{ type: "gaussian", params: { mean: 0, std: 1.5 }, weight: 0.6 },
          { type: "gaussian", params: { mean: 0.5, std: 0.2 }, weight: 0.4 }],
  zoo: [{ type: "gaussian", params: { mean: -2.5, std: 0.8 }, weight: 0.4 },
        { type: "uniform", params: { low: 0, high: 2 }, weight: 0.35 },
        { type: "exponential", params: { shift: 2.5, rate: 1.2 }, weight: 0.25 }],
};

const truePdfChart = new Chart($("#chart-true-pdf"), { height: 220, zeroBase: true });
const trueCdfChart = new Chart($("#chart-true-cdf"), { height: 200, zeroBase: true });

function renderComponents() {
  const host = $("#components");
  host.replaceChildren();
  state.components.forEach((comp, i) => {
    const el = document.createElement("div");
    el.className = "component";
    const opts = Object.keys(state.reg.distributions)
      .map((k) => `<option value="${k}" ${k === comp.type ? "selected" : ""}>${k.replace("_", " ")}</option>`).join("");
    const params = Object.entries(comp.params)
      .map(([k, v]) => `<div><label>${k}</label><input type="number" step="0.1" data-param="${k}" value="${v}"></div>`).join("");
    el.innerHTML = `
      <div class="comp-head">
        <select aria-label="component ${i + 1} type">${opts}</select>
        <button class="remove" title="remove component" aria-label="remove component ${i + 1}">×</button>
      </div>
      <div class="params">${params}</div>
      <div class="weight-row">
        <label for="w${i}">weight</label>
        <input type="range" id="w${i}" min="1" max="100" value="${Math.round(comp.weight * 100)}">
        <output>${comp.weight.toFixed(2)}</output>
      </div>`;
    el.querySelector("select").addEventListener("change", (e) => {
      const t = e.target.value;
      state.components[i] = { type: t, params: { ...state.reg.distributions[t] }, weight: comp.weight };
      $("#preset").value = "";
      renderComponents();
      distChanged();
    });
    el.querySelector(".remove").addEventListener("click", () => {
      state.components.splice(i, 1);
      $("#preset").value = "";
      renderComponents();
      distChanged();
    });
    el.querySelectorAll("[data-param]").forEach((inp) =>
      inp.addEventListener("input", () => {
        comp.params[inp.dataset.param] = parseFloat(inp.value);
        $("#preset").value = "";
        distChanged();
      }));
    const wr = el.querySelector(".weight-row input");
    wr.addEventListener("input", () => {
      comp.weight = wr.value / 100;
      el.querySelector(".weight-row output").textContent = comp.weight.toFixed(2);
      $("#preset").value = "";
      distChanged();
    });
    host.appendChild(el);
  });
}

const distChanged = debounce(async () => {
  const err = $("#dist-error");
  err.textContent = "";
  if (!state.components.length) { err.textContent = "Add at least one component."; return; }
  const res = await fetch("/api/distribution", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ components: state.components }),
  });
  const data = await res.json();
  if (!res.ok) { err.textContent = data.error; return; }
  state.dist = data;
  truePdfChart.set([{ label: "true pdf", color: C.truth(), x: data.grid, y: data.pdf }]);
  trueCdfChart.set([{ label: "true CDF", color: C.truth(), x: data.grid, y: data.cdf }]);
  $("#nav-dist").classList.add("is-done");
  if (state.parzen) $("#parzen-stale").classList.add("show");
}, 250);

$("#add-component").addEventListener("click", () => {
  state.components.push({ type: "gaussian", params: { ...state.reg.distributions.gaussian }, weight: 0.5 });
  $("#preset").value = "";
  renderComponents();
  distChanged();
});
$("#preset").addEventListener("change", (e) => {
  if (!PRESETS[e.target.value]) return;
  state.components = PRESETS[e.target.value].map((c) => ({ ...c, params: { ...c.params } }));
  renderComponents();
  distChanged();
});

/* ================================================================ stage 2: parzen window */

const N_MIN = 50, N_MAX = 20000;
const sliderToN = (t) => Math.round(N_MIN * Math.pow(N_MAX / N_MIN, t / 100));
const H_MIN = 0.005, H_MAX = 3;
const sliderToH = (t) => H_MIN * Math.pow(H_MAX / H_MIN, t / 100);

const parzenCdfChart = new Chart($("#chart-parzen-cdf"), { height: 220, zeroBase: true });

function drawKernelPreview() {
  const k = KERNELS_JS[$("#kernel").value];
  const svg = $("#kernel-preview");
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const u = -3 + (6 * i) / 60;
    pts.push(`${(i / 60) * 116 + 2},${30 - k(u) / 0.78 * 26}`);
  }
  svg.innerHTML = `<line x1="2" x2="118" y1="31" y2="31" stroke="${C.baseline()}"/><polyline points="${pts.join(" ")}" fill="none" stroke="${C.parzen()}" stroke-width="2" stroke-linejoin="round"/>`;
  $("#kernel-note").textContent = KERNEL_NOTES[$("#kernel").value] || "";
}

function updateStrategyUI() {
  const s = $("#strategy").value;
  const n = sliderToN(+$("#n-samples").value);
  $("#manual-h-field").hidden = s !== "manual";
  $("#h1-field").hidden = s !== "sqrt_n";
  if (s === "sqrt_n") {
    $("#h1-note").textContent =
      `At n = ${n.toLocaleString("en-US")} this gives h = ${fmt(+$("#h1").value / Math.sqrt(n))}.`;
  }
  let note = STRATEGY_NOTES[s] || "";
  if ((s === "lscv" || s === "likelihood_cv") && n > state.reg.cv_max_n) {
    note += ` Current n = ${n.toLocaleString("en-US")} exceeds the cap.`;
  }
  $("#strategy-note").textContent = note;
}

$("#n-samples").addEventListener("input", () => {
  $("#n-samples-out").textContent = sliderToN(+$("#n-samples").value).toLocaleString("en-US");
  updateStrategyUI();
});
$("#manual-h").addEventListener("input", () => {
  $("#manual-h-out").textContent = fmt(sliderToH(+$("#manual-h").value));
});
$("#kernel").addEventListener("change", drawKernelPreview);
$("#strategy").addEventListener("change", updateStrategyUI);
$("#h1").addEventListener("input", updateStrategyUI);

/* seed fields: "random" draws a fresh seed per run and shows it, so runs stay reproducible */
function wireSeed(randSel, inputSel) {
  $(randSel).addEventListener("change", () => { $(inputSel).disabled = $(randSel).checked; });
}
function seedValue(randSel, inputSel) {
  if ($(randSel).checked) $(inputSel).value = crypto.getRandomValues(new Uint32Array(1))[0] % 2147483647;
  return +$(inputSel).value;
}
wireSeed("#sample-seed-rand", "#sample-seed");
wireSeed("#net-seed-rand", "#net-seed");

/* --- the construction animation (canvas) --- */

const construct = {
  canvas: $("#construct-canvas"),
  raf: 0, i: 0, carry: 0, sum: null, payload: null, kernel: null, speed: 150,

  start(payload, kernelName, speed) {
    cancelAnimationFrame(this.raf);
    this.payload = payload;
    this.kernel = KERNELS_JS[kernelName];
    this.speed = REDUCED_MOTION ? 0 : speed;
    this.i = 0;
    this.carry = 0;
    this.sum = new Float64Array(payload.grid.length);
    this.maxY = Math.max(...payload.truth_pdf, ...payload.parzen_pdf) * 1.18;
    $("#skip-anim").hidden = this.speed === 0;
    if (this.speed === 0) this.finish();
    else this.raf = requestAnimationFrame(() => this.tick());
  },

  hOf(j) { return this.payload.h_summary.per_sample ? this.payload.h[j] : this.payload.h; },

  addSample(j) {
    const { grid, samples } = this.payload;
    const h = this.hOf(j), x = samples[j];
    for (let g = 0; g < grid.length; g++) this.sum[g] += this.kernel((grid[g] - x) / h) / h;
  },

  tick() {
    const n = this.payload.samples.length;
    this.carry += this.speed / 60;
    let steps = Math.floor(this.carry);
    this.carry -= steps;
    while (steps-- > 0 && this.i < n) this.addSample(this.i++);
    this.draw(this.i < n);
    if (this.i < n) this.raf = requestAnimationFrame(() => this.tick());
    else this.finish();
  },

  finish() {
    cancelAnimationFrame(this.raf);
    this.i = this.payload.samples.length;
    $("#skip-anim").hidden = true;
    this.draw(false);
    onConstructionDone();
  },

  draw(animating) {
    const { grid, samples, truth_pdf, parzen_pdf, h_summary } = this.payload;
    const cv = this.canvas, dpr = devicePixelRatio || 1;
    const W = cv.clientWidth || 640, H = 260;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);
    const m = { l: 44, r: 12, t: 8, b: 26 };
    const x0 = grid[0], x1 = grid[grid.length - 1];
    const xs = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const ys = (v) => H - m.b - Math.min(v / this.maxY, 1.04) * (H - m.t - m.b);

    ctx.clearRect(0, 0, W, H);
    ctx.font = "11px system-ui"; ctx.fillStyle = C.ink2(); ctx.strokeStyle = C.hairline();
    for (const t of niceTicks(0, this.maxY, 3)) {
      ctx.beginPath(); ctx.moveTo(m.l, ys(t)); ctx.lineTo(W - m.r, ys(t)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(fmt(t), m.l - 5, ys(t) + 3.5);
    }
    ctx.strokeStyle = C.baseline();
    ctx.beginPath(); ctx.moveTo(m.l, H - m.b); ctx.lineTo(W - m.r, H - m.b); ctx.stroke();
    ctx.textAlign = "center";
    for (const t of niceTicks(x0, x1, 6)) ctx.fillText(fmt(t), xs(t), H - m.b + 15);

    const line = (ys_, color, dash = [], width = 2) => {
      ctx.save(); ctx.beginPath(); ctx.rect(m.l, 0, W - m.l - m.r, H - m.b); ctx.clip();
      ctx.beginPath();
      for (let g = 0; g < grid.length; g++) { const px = xs(grid[g]), py = ys(ys_(g)); g ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.lineJoin = "round"; ctx.stroke();
      ctx.restore();
    };

    // sample rug (cap what we draw; the caption states the cap)
    const rugMax = 3000;
    const rugStep = Math.max(1, Math.ceil(this.i / rugMax));
    ctx.strokeStyle = "rgba(42,120,214,0.30)"; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.beginPath();
    for (let j = 0; j < this.i; j += rugStep) {
      const px = xs(samples[j]);
      ctx.moveTo(px, H - m.b); ctx.lineTo(px, H - m.b - 7);
    }
    ctx.stroke();

    line((g) => truth_pdf[g], C.truth(), [6, 4]);
    if (this.i > 0 && animating) line((g) => this.sum[g] / this.i, C.parzen());
    if (!animating) line((g) => parzen_pdf[g], C.parzen());

    if (animating && this.i > 0) {
      const j = this.i - 1, h = this.hOf(j), x = samples[j];
      ctx.save(); ctx.beginPath(); ctx.rect(m.l, 0, W - m.l - m.r, H - m.b); ctx.clip();
      ctx.beginPath();
      for (let g = 0; g < grid.length; g++) {
        const px = xs(grid[g]), py = ys(this.kernel((grid[g] - x) / h) / h);
        g ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.lineTo(xs(x1), H - m.b); ctx.lineTo(xs(x0), H - m.b); ctx.closePath();
      ctx.fillStyle = C.kernel(); ctx.fill();
      ctx.restore();
    }

    const n = samples.length;
    const hTxt = h_summary.per_sample
      ? `h = ${fmt(h_summary.mean)} avg (adaptive, ${fmt(h_summary.min)}–${fmt(h_summary.max)})`
      : `h = ${fmt(h_summary.mean)}`;
    $("#construct-caption").textContent = animating
      ? `window ${this.i.toLocaleString("en-US")} of ${n.toLocaleString("en-US")} · each carries mass 1/n · the estimate is their running average`
      : `${n.toLocaleString("en-US")} windows averaged · ${hTxt}` + (n > rugMax ? ` · rug shows ${rugMax.toLocaleString("en-US")} of ${n.toLocaleString("en-US")} ticks` : "");

    if (modalConstructOpen && modalChart) modalChart.set(constructSeries());
  },
};

$("#skip-anim").addEventListener("click", () => construct.finish());

$("#run-parzen").addEventListener("click", async () => {
  const err = $("#parzen-error");
  err.textContent = "";
  if (!state.dist) { err.textContent = "Build a distribution in stage 1 first."; return; }
  const cfg = {
    components: JSON.parse(JSON.stringify(state.components)),
    n: sliderToN(+$("#n-samples").value),
    seed: seedValue("#sample-seed-rand", "#sample-seed"),
    kernel: $("#kernel").value,
    strategy: $("#strategy").value,
    h_manual: sliderToH(+$("#manual-h").value),
    h1: +$("#h1").value,
    domain: $("#domain").value,
  };
  const btn = $("#run-parzen");
  btn.disabled = true; btn.textContent = "Sampling…";
  try {
    const res = await fetch("/api/parzen", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg),
    });
    const data = await res.json();
    if (!res.ok) { err.textContent = data.error; return; }
    state.parzen = data;
    state.parzenCfg = cfg;
    $("#parzen-empty").hidden = true;
    $("#parzen-results").hidden = false;
    $("#parzen-stale").classList.remove("show");
    $("#construct-legend").innerHTML =
      `<span class="key"><span class="swatch dashed" style="border-top-color:${C.truth()}"></span>true pdf</span>` +
      `<span class="key"><span class="swatch" style="border-top-color:${C.parzen()}"></span>running estimate</span>` +
      `<span class="key"><span class="swatch" style="border-top-color:${C.kernel()};border-top-width:6px"></span>current window ÷ n</span>`;
    construct.start(data, cfg.kernel, +$("#anim-speed").value);
  } finally {
    btn.disabled = false; btn.textContent = "Run estimation";
  }
});

function onConstructionDone() {
  const d = state.parzen;
  parzenCdfChart.set([
    { label: "truth", color: C.truth(), dash: "6 4", x: d.grid, y: d.truth_cdf },
    { label: "Parzen estimate", color: C.parzen(), x: d.grid, y: d.parzen_cdf },
    { label: "empirical CDF", color: C.emp(), type: "step", width: 1.5, x: d.grid, y: d.empirical_cdf },
  ]);
  const h = d.h_summary;
  $("#parzen-tiles").innerHTML = [
    ["Samples", state.parzenCfg.n.toLocaleString("en-US")],
    ["Window size h", h.per_sample ? `${fmt(h.mean)} <span class="unit">avg</span>` : fmt(h.mean)],
    ["h₁ = h·√n", h.per_sample ? `${fmt(h.h1)} <span class="unit">avg</span>` : fmt(h.h1)],
    ["CDF gap (KS) vs truth", fmt(d.ks_vs_truth)],
    ["Window shape", state.parzenCfg.kernel],
    ["Strategy", state.parzenCfg.strategy.replace("_", " ")],
  ].map(([l, v]) => `<div class="tile"><div class="t-label">${l}</div><div class="t-value">${v}</div></div>`).join("");
  $("#nav-parzen").classList.add("is-done");
  $("#net-empty").textContent = "Ready: the training set is the stage-2 samples with their Parzen CDF labels. Configure the network and press Train.";
  setTrainUI();
}

/* ================================================================ stage 3: the network */

const fitChart = new Chart($("#chart-fit"), { height: 250, zeroBase: true });
const pdfFitChart = new Chart($("#chart-pdf-fit"), { height: 220, zeroBase: true });
const lossChart = new Chart($("#chart-loss"), { height: 170, logY: true });
const ksChart = new Chart($("#chart-ks"), { height: 170, zeroBase: true });

/* hidden layers editor */
let layers = [32];
function renderLayers() {
  const host = $("#layers");
  host.replaceChildren();
  layers.forEach((w, i) => {
    const chip = document.createElement("span");
    chip.className = "layer-chip";
    chip.innerHTML = `<input type="number" min="1" max="256" value="${w}" aria-label="layer ${i + 1} width">` +
      (layers.length > 1 ? `<button class="remove" aria-label="remove layer ${i + 1}">×</button>` : "");
    chip.querySelector("input").addEventListener("change", (e) => {
      layers[i] = Math.max(1, Math.min(256, Math.round(+e.target.value || 1)));
      e.target.value = layers[i];
    });
    chip.querySelector(".remove")?.addEventListener("click", () => { layers.splice(i, 1); renderLayers(); });
    host.appendChild(chip);
  });
  if (layers.length < 6) {
    const add = document.createElement("button");
    add.className = "btn add-layer";
    add.textContent = "+ layer";
    add.addEventListener("click", () => { layers.push(layers[layers.length - 1] || 32); renderLayers(); });
    host.appendChild(add);
  }
}

/* The estimator decides which controls mean anything. Penalties and rectification exist to
   repair an unconstrained network; the mixture has nothing for them to repair, so they are
   hidden rather than left switched off, which would read as a choice. */
const ESTIMATOR_NOTES = {
  mixture: "A convex combination of logistic CDFs. Monotone, bounded in [0,1] and unit-mass at every parameter value, so the density is a formula rather than a numerical derivative.",
  mlp: "Weights are free, so nothing keeps F monotone: perturb them and it breaks. Kept to be compared with.",
};

function syncEstimatorUI() {
  const mixture = $("#estimator").value === "mixture";
  for (const el of document.querySelectorAll("[data-mlp-only]")) el.hidden = mixture;
  $("#components-field").hidden = !mixture;
  $("#estimator-note").textContent = ESTIMATOR_NOTES[$("#estimator").value];
  if (!mixture) $("#monotonicity").dispatchEvent(new Event("change"));
}
$("#estimator").addEventListener("change", syncEstimatorUI);

$("#monotonicity").addEventListener("change", () => {
  const v = $("#monotonicity").value;
  $("#mono-weight-field").hidden = v !== "penalty";
  const notes = {
    none: "Nothing constrains F. Whether it comes out monotone is luck, and the diagram shows how much.",
    penalty: "Adds λ·mean(relu(−dF/dx)) at 256 collocation points: it certifies those 256 points and says nothing about what happens between them.",
    positive: "Non-negative weights and monotone activations make F monotone by construction (Archer & Wang), at some accuracy cost. It still does not pin F(−∞) to 0 or F(+∞) to 1.",
  };
  $("#mono-note").textContent = notes[v];
  if (v === "positive" && $("#activation").value === "silu") {
    $("#activation").value = "sigmoid";
    $("#mono-note").textContent += " (silu is not monotone; switched to sigmoid.)";
  }
});
$("#activation").addEventListener("change", () => {
  if ($("#activation").value === "silu" && $("#monotonicity").value === "positive") {
    $("#monotonicity").value = "none";
    $("#monotonicity").dispatchEvent(new Event("change"));
  }
});
const TARGET_NOTES = {
  parzen_loo: "yᵢ = (n·F̂(xᵢ) − ½)/(n−1): the Parzen estimate at xᵢ with xᵢ itself taken out. A symmetric kernel contributes K(0) = ½ to its own point, and that is what gets subtracted.",
  parzen: "Labels are the plain Parzen estimate at the sample points, so each label contains the point it is labelling. The network is being asked to reproduce that.",
  empirical: "Labels are Fₙ(xᵢ) = (rank − ½)/n: unbiased but noisy, and the network's smoothness has to do all the denoising.",
};
function syncTargetUI() {
  const v = $("#target").value;
  $("#target-note").textContent = TARGET_NOTES[v];
  $("#teacher-scale-field").hidden = v !== "parzen_loo";
}
$("#target").addEventListener("change", syncTargetUI);

/* ---------------------------------------------------------------- network diagram */

class NetDiagram {
  constructor(svg, tip, opts = {}) {
    this.svg = svg;
    this.tip = tip;
    this.o = { W: 800, H: 340, padX: 64, padY: 26, maxEdges: 1200,
               panzoom: false, noteEl: null, scaleEls: null, onToggle: null, ...opts };
    this.sizes = null;
    this.estimator = "mlp";             // decides what an edge is called, and what a click does
    this.vb = null;                     // pan/zoom viewBox override
    this._downAt = null;
    this._wire();
  }

  layout(sizes) { this.sizes = sizes; this.vb = null; }

  update(weights, biases, pruned) {
    if (!this.sizes) return;
    this.weights = weights; this.biases = biases; this.pruned = pruned;
    this._render();
  }

  _pos() {
    const { W, H, padX, padY } = this.o;
    const L = this.sizes.length;
    const lx = (l) => padX + (l * (W - 2 * padX)) / (L - 1);
    const ny = (l, k) => {
      const m = this.sizes[l];
      const sp = Math.min(30, (H - 2 * padY) / Math.max(m - 1, 1));
      const span = sp * (m - 1);
      return H / 2 - span / 2 + k * sp;
    };
    const nr = (l) => {
      if (l === 0 || l === this.sizes.length - 1) return 9;
      const sp = Math.min(30, (H - 2 * padY) / Math.max(this.sizes[l] - 1, 1));
      return clamp(sp * 0.38, 1.2, 6.5);
    };
    return { lx, ny, nr, L };
  }

  _render() {
    const { lx, ny, nr, L } = this._pos();
    const { W, H } = this.o;
    let maxAbs = 1e-9;
    for (const layer of this.weights) for (const row of layer) for (const w of row) {
      const a = Math.abs(w);
      if (a > maxAbs) maxAbs = a;
    }
    if (this.o.scaleEls) {
      this.o.scaleEls[0].textContent = "−" + fmt(maxAbs);
      this.o.scaleEls[1].textContent = "+" + fmt(maxAbs);
    }

    const live = [], cut = [];
    for (let l = 0; l < this.weights.length; l++) {
      for (let o = 0; o < this.sizes[l + 1]; o++) {
        for (let i = 0; i < this.sizes[l]; i++) {
          (this.pruned.has(`${l}:${o}:${i}`) ? cut : live).push({ l, i, o, w: this.weights[l][o][i] });
        }
      }
    }
    let drawn = live;
    if (live.length > this.o.maxEdges) {
      drawn = [...live].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, this.o.maxEdges);
    }

    // each edge = a visible line (no pointer events) + a fat transparent hit line on top,
    // so hovering and clicking a hair-thin weight is actually possible with a real mouse
    let g = "", hits = "";
    const edgeLines = (e, paint, data) => {
      const coords = `x1="${lx(e.l)}" y1="${ny(e.l, e.i)}" x2="${lx(e.l + 1)}" y2="${ny(e.l + 1, e.o)}"`;
      g += `<line ${coords} ${paint} pointer-events="none"></line>`;
      hits += `<line ${coords} stroke="transparent" stroke-width="9" style="cursor:pointer" ${data}></line>`;
    };
    // The mixture is drawn as a 1 → J → 1 network because that is what it is, but its edges
    // have their own names, and the incoming one cannot be cut: with a = 0 the unit is the
    // constant ½, so F(−∞) lifts off zero and F stops being a CDF.
    const edgeTip = (e) => {
      if (this.estimator !== "mixture") {
        return `w = ${fmt(e.w, 4)} · layer ${e.l + 1}, unit ${e.i + 1} → unit ${e.o + 1} · click to prune`;
      }
      return e.l === 0
        ? `width a = ${fmt(e.w, 4)} · component ${e.o + 1} · cannot be cut: a = 0 makes the unit constant at ½, and F(−∞) leaves 0`
        : `mixing weight π = ${fmt(e.w, 4)} · component ${e.i + 1} · click to switch the component off`;
    };
    for (const e of drawn) {
      const r = Math.abs(e.w) / maxAbs;
      edgeLines(e,
        `stroke="${e.w >= 0 ? C.wpos() : C.wneg()}" stroke-width="${(0.6 + 3.2 * r).toFixed(2)}" opacity="${(0.18 + 0.72 * r).toFixed(2)}"`,
        `data-l="${e.l}" data-i="${e.i}" data-o="${e.o}" data-tip="${edgeTip(e)}"`);
    }
    for (const e of cut) {
      const what = this.estimator === "mixture"
        ? `component ${e.i + 1} switched off · click to restore`
        : `pruned · layer ${e.l + 1}, unit ${e.i + 1} → unit ${e.o + 1} · click to restore`;
      edgeLines(e,
        `stroke="${C.truth()}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.75"`,
        `data-l="${e.l}" data-i="${e.i}" data-o="${e.o}" data-pruned="1" data-tip="${what}"`);
    }
    g += hits;
    for (let l = 0; l < L; l++) {
      const r = nr(l);
      for (let k = 0; k < this.sizes[l]; k++) {
        const b = l > 0 ? this.biases[l - 1][k] : null;
        const tip = l === 0 ? "input x" : l === L - 1 ? `output F̂(x) · bias = ${fmt(b, 4)}` : `layer ${l}, unit ${k + 1} · bias = ${fmt(b, 4)}`;
        g += `<circle class="node" cx="${lx(l)}" cy="${ny(l, k)}" r="${r}" data-tip="${tip}"></circle>`;
      }
    }
    g += `<text x="${lx(0) - 16}" y="${ny(0, 0) + 4}" text-anchor="end">x</text>`;
    g += `<text x="${lx(L - 1) + 16}" y="${ny(L - 1, 0) + 4}">F̂(x)</text>`;
    this.svg.setAttribute("viewBox", (this.vb || [0, 0, W, H]).join(" "));
    this.svg.innerHTML = g;

    if (this.o.noteEl) {
      const parts = ["Edge thickness is |weight|; blue positive, red negative. Hover for values, click a connection to prune it."];
      if (drawn.length < live.length) {
        parts.push(`Showing the strongest ${drawn.length.toLocaleString("en-US")} of ${live.length.toLocaleString("en-US")} connections (pruned ones always shown); expand ⤢ to see more.`);
      }
      this.o.noteEl.textContent = parts.join(" ");
    }
  }

  _applyVB() { this.svg.setAttribute("viewBox", (this.vb || [0, 0, this.o.W, this.o.H]).join(" ")); }

  _wire() {
    const svg = this.svg, tip = this.tip;
    svg.addEventListener("pointerover", (e) => {
      const t = e.target.getAttribute?.("data-tip");
      if (!t) { tip.classList.remove("show"); return; }
      tip.textContent = t;
      tip.classList.add("show");
    });
    svg.addEventListener("pointermove", (e) => {
      const r = svg.parentElement.getBoundingClientRect();
      const x = e.clientX - r.left, flip = x > r.width - 240;
      tip.style.left = flip ? "" : `${x + 14}px`;
      tip.style.right = flip ? `${r.width - x + 14}px` : "";
      tip.style.top = `${e.clientY - r.top + 14}px`;
      if (this._panning) {
        const rect = svg.getBoundingClientRect();
        const vb = this.vb;
        if (vb) {
          vb[0] -= (e.clientX - this._downAt[0]) * vb[2] / rect.width;
          vb[1] -= (e.clientY - this._downAt[1]) * vb[3] / rect.height;
          this._downAt = [e.clientX, e.clientY, this._downAt[2], this._downAt[3]];
          this._applyVB();
        }
      }
    });
    svg.addEventListener("pointerleave", () => tip.classList.remove("show"));
    svg.addEventListener("click", (e) => {
      if (this._moved) return;   // a pan, not a click
      const el = e.target;
      if (el.tagName !== "line") return;
      const l = +el.dataset.l, i = +el.dataset.i, o = +el.dataset.o;
      this.o.onToggle?.(l, i, o, el.dataset.pruned === "1");
    });
    if (!this.o.panzoom) return;
    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const { W, H } = this.o;
      const vb = this.vb || [0, 0, W, H];
      const r = svg.getBoundingClientRect();
      const px = vb[0] + ((e.clientX - r.left) / r.width) * vb[2];
      const py = vb[1] + ((e.clientY - r.top) / r.height) * vb[3];
      const f = e.deltaY > 0 ? 1.2 : 1 / 1.2;
      const nw = clamp(vb[2] * f, W / 20, W);
      const nh = nw * H / W;
      this.vb = nw >= W * 0.999 ? null
        : [px - ((px - vb[0]) / vb[2]) * nw, py - ((py - vb[1]) / vb[3]) * nh, nw, nh];
      this._applyVB();
    }, { passive: false });
    svg.addEventListener("pointerdown", (e) => {
      this._panning = !!this.vb;
      this._moved = false;
      this._downAt = [e.clientX, e.clientY, e.clientX, e.clientY];
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener("pointermove", (e) => {
      if (this._downAt && Math.hypot(e.clientX - this._downAt[2], e.clientY - this._downAt[3]) > 5) this._moved = true;
    });
    svg.addEventListener("pointerup", () => { this._panning = false; });
    svg.addEventListener("dblclick", () => { this.vb = null; this._applyVB(); });
  }
}

function togglePrune(l, i, o, isPruned) {
  if (!state.train.started) return;
  if (netDiagram.estimator === "mixture" && l === 0) {
    log("a mixture's width cannot be cut: with a = 0 the unit is the constant ½, so F(−∞) " +
        "leaves 0 and F is no longer a CDF. Cut the mixing weight instead.", "em");
    return;
  }
  // optimistic: flip the edge locally right away; the server's snapshot confirms it
  const key = `${l}:${o}:${i}`;
  if (isPruned) state.train.net.pruned.delete(key);
  else state.train.net.pruned.add(key);
  updateNetViews({});
  wsSend({ cmd: isPruned ? "unprune" : "prune", l, i, o });
  log(`${isPruned ? "restored" : "pruned"} connection · layer ${l + 1}, unit ${i + 1} → unit ${o + 1}`, "em");
}

const netDiagram = new NetDiagram($("#net-svg"), $("#net-tooltip"), {
  noteEl: $("#net-diagram-note"),
  scaleEls: [$("#w-min"), $("#w-max")],
  onToggle: togglePrune,
});
const modalNet = new NetDiagram($("#modal-net-svg"), $("#modal-net-tooltip"), {
  W: 1400, H: 640, padX: 90, padY: 34, maxEdges: 4000, panzoom: true,
  noteEl: $("#modal-net-note"),
  onToggle: togglePrune,
});

/* ---------------------------------------------------------------- expand-to-modal */

const modal = $("#viz-modal");
let modalChart = null, modalSrcChart = null, modalNetOpen = false, modalConstructOpen = false;

const VIZ = {
  "chart-true-pdf": { title: "True density f(x)", chart: () => truePdfChart },
  "chart-true-cdf": { title: "True CDF F(x)", chart: () => trueCdfChart },
  "chart-parzen-cdf": { title: "CDF: estimate vs truth", chart: () => parzenCdfChart },
  "chart-fit": { title: "What the network is learning · CDF", chart: () => fitChart },
  "chart-pdf-fit": { title: "Recovered density · derivative of the network", chart: () => pdfFitChart },
  "chart-loss": { title: "Training loss (MSE, log scale)", chart: () => lossChart },
  "chart-ks": { title: "CDF gap (KS) over training", chart: () => ksChart },
};

function cleanupModal() {
  if (modalSrcChart) { modalSrcChart._mirror = null; modalSrcChart = null; }
  if (modalChart) { modalChart.destroy(); modalChart = null; }
  modalNetOpen = false;
  modalConstructOpen = false;
  $("#modal-chart-card").hidden = true;
  $("#modal-net-card").hidden = true;
}

/* the construction figure is a canvas animation; its modal twin is a regular zoomable Chart
   rebuilt from the same data, refreshed live while the animation runs */
function constructSeries() {
  const p = construct.payload;
  if (!p) return [];
  const n = p.samples.length;
  const animating = construct.i < n;
  const s = [{ label: "true pdf", color: C.truth(), dash: "6 4", x: p.grid, y: p.truth_pdf }];
  if (!animating) {
    s.push({ label: "Parzen estimate", color: C.parzen(), x: p.grid, y: p.parzen_pdf });
  } else if (construct.i > 0) {
    s.push({ label: `running estimate · ${construct.i.toLocaleString("en-US")} of ${n.toLocaleString("en-US")}`,
             color: C.parzen(), x: p.grid, y: Array.from(construct.sum, (v) => v / construct.i) });
    const j = construct.i - 1, h = construct.hOf(j), x = p.samples[j];
    s.push({ label: "current window ÷ n", color: C.kernel(), width: 1.5,
             x: p.grid, y: p.grid.map((gx) => construct.kernel((gx - x) / h) / h) });
  }
  return s;
}

function openConstructModal() {
  if (!construct.payload) return;
  cleanupModal();
  $("#modal-title").textContent = "Density construction";
  $("#modal-chart-card").hidden = false;
  modalChart = new Chart($("#modal-chart-card"), {
    height: Math.max(420, Math.floor(innerHeight * 0.62)), zeroBase: true, zoomable: true,
  });
  modalConstructOpen = true;
  modalChart.set(constructSeries());
  modal.showModal();
}

function openChartModal(id) {
  const meta = VIZ[id];
  if (!meta) return;
  cleanupModal();
  const src = meta.chart();
  $("#modal-title").textContent = meta.title;
  $("#modal-chart-card").hidden = false;
  modalChart = new Chart($("#modal-chart-card"), {
    ...src.opts, height: Math.max(420, Math.floor(innerHeight * 0.62)), zoomable: true,
  });
  src._mirror = modalChart;
  modalSrcChart = src;
  modalChart.set(src.series);
  modal.showModal();
}

function openNetModal() {
  const net = state.train.net;
  if (!net) return;
  cleanupModal();
  $("#modal-title").textContent = "The network";
  $("#modal-net-card").hidden = false;
  modalNetOpen = true;
  modalNet.layout(net.sizes);
  modalNet.update(net.weights, net.biases, net.pruned);
  modal.showModal();
}

// the dialog "close" event fires as a task, possibly after an immediate re-open: only clean
// up if the modal is really closed by then (open-time cleanup covers the re-open path)
modal.addEventListener("close", () => { if (!modal.open) cleanupModal(); });
$("#modal-close").addEventListener("click", () => modal.close());
document.querySelectorAll(".expand").forEach((b) =>
  b.addEventListener("click", () => {
    if (b.dataset.expand === "network") openNetModal();
    else if (b.dataset.expand === "chart-construct") openConstructModal();
    else openChartModal(b.dataset.expand);
  }));

/* ---------------------------------------------------------------- training tiles + log */

const TILES = [
  ["epoch", "Epoch"], ["loss", "Train loss (MSE)"], ["kst", "KS vs target"],
  ["ksT", "KS vs truth"], ["mass", "pdf mass"], ["viol", "Monotonicity viol."],
];
function resetTiles() {
  $("#net-tiles").innerHTML = TILES.map(([id, l]) =>
    `<div class="tile" id="tile-${id}"><div class="t-label">${l}</div><div class="t-value">–</div></div>`).join("");
}
function setTile(id, html, cls = "") {
  const el = $(`#tile-${id}`);
  el.className = `tile ${cls}`;
  el.querySelector(".t-value").innerHTML = html;
}
function log(text, cls = "") {
  const el = $("#trainlog");
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  el.appendChild(line);
  while (el.childElementCount > 400) el.firstChild.remove();
  el.scrollTop = el.scrollHeight;
}

/* ---------------------------------------------------------------- websocket + controls */

function setStatus(stateName, text) {
  $("#status-dot").className = `status-dot ${stateName}`;
  $("#status-text").textContent = text;
}
function setTrainUI() {
  const t = state.train;
  $("#btn-train").disabled = t.running || !state.parzen;
  $("#btn-continue").disabled = t.running || !t.started;
  $("#btn-pause").disabled = !t.running;
  $("#btn-pause").textContent = t.paused ? "Resume" : "Pause";
  $("#btn-stop").disabled = !t.running;
  $("#btn-save").disabled = !t.started;
  $("#btn-reset").disabled = t.running || !t.started;
  document.querySelectorAll("#stage-net .panel select, #stage-net .panel input:not(#ckpt-name)")
    .forEach((el) => { el.disabled = t.running; });
  document.querySelectorAll("#stage-net .layers button, #stage-net .layers input")
    .forEach((el) => { el.disabled = t.running; });
  $("#net-seed").disabled = t.running || $("#net-seed-rand").checked;
}

function snapEvery() {
  const v = Math.round(+$("#snap-every").value);
  return v > 0 ? v : null;
}

function wsSend(msg) { state.train.ws?.readyState === 1 && state.train.ws.send(JSON.stringify(msg)); }

function connectWS() {
  return new Promise((resolve, reject) => {
    if (state.train.ws?.readyState === 1) return resolve();
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/train`);
    state.train.ws = ws;
    ws.onopen = () => resolve();
    ws.onerror = () => reject(new Error("websocket connection failed"));
    ws.onclose = () => {
      if (state.train.running) {
        state.train.running = false;
        setStatus("", "connection lost");
        setTrainUI();
      }
    };
    ws.onmessage = (e) => onTrainMessage(JSON.parse(e.data));
  });
}

function updateNetViews(msg) {
  const t = state.train;
  if (!t.net) return;
  if (msg.weights) { t.net.weights = msg.weights; t.net.biases = msg.biases; }
  if (msg.pruned) t.net.pruned = new Set(msg.pruned.map(([l, o, i]) => `${l}:${o}:${i}`));
  netDiagram.update(t.net.weights, t.net.biases, t.net.pruned);
  if (modalNetOpen) modalNet.update(t.net.weights, t.net.biases, t.net.pruned);
}

function onTrainMessage(msg) {
  const t = state.train;
  if (msg.type === "hello") {
    t.hello = msg;
    t.hist = { epoch: [], loss: [], kst: [], ksT: [] };
    t.net = {
      sizes: msg.layer_sizes,
      weights: msg.weights,
      biases: msg.biases,
      pruned: new Set((msg.pruned || []).map(([l, o, i]) => `${l}:${o}:${i}`)),
    };
    netDiagram.estimator = modalNet.estimator = msg.estimator || "mlp";
    netDiagram.layout(msg.layer_sizes);
    updateNetViews({});
    if (modalNetOpen) modalNet.layout(msg.layer_sizes);
    $("#fit-note").textContent =
      `Dots: the ${msg.target_points.shown} of ${msg.target_points.total.toLocaleString("en-US")} training labels shown (xᵢ, target CDF).`;
    fitChart.set([
      { label: "truth", color: C.truth(), dash: "6 4", x: msg.grid, y: msg.truth_cdf },
      { label: "training labels", color: C.parzen(), type: "dots", x: msg.target_points.x, y: msg.target_points.y },
      { label: "network", color: C.net(), x: [], y: [] },
    ]);
    pdfFitChart.set([
      { label: "truth", color: C.truth(), dash: "6 4", x: msg.grid, y: msg.truth_pdf },
      { label: "Parzen", color: C.parzen(), width: 1.5, x: msg.grid, y: msg.parzen_pdf },
      { label: "network derivative", color: C.net(), x: [], y: [] },
    ]);
    log(`training started · layers [${msg.layer_sizes.join(", ")}] · ${msg.n_weights.toLocaleString("en-US")} weights`, "em");
    return;
  }
  if (msg.type === "snapshot") {
    const h = t.hello;
    // curves + metric tiles always (also for the instant re-eval after a prune/restore)
    setTile("kst", fmt(msg.ks_target));
    setTile("ksT", fmt(msg.ks_truth));
    setTile("mass", fmt(msg.mass, 4), Math.abs(msg.mass - 1) < 0.01 ? "good" : "");
    setTile("viol", `${fmt(msg.violations * 100)}<span class="unit">%</span>`, msg.violations === 0 ? "good" : "bad");
    fitChart.set([
      { label: "truth", color: C.truth(), dash: "6 4", x: h.grid, y: h.truth_cdf },
      { label: "training labels", color: C.parzen(), type: "dots", x: h.target_points.x, y: h.target_points.y },
      { label: "network", color: C.net(), x: h.grid, y: msg.net_cdf },
    ]);
    pdfFitChart.set([
      { label: "truth", color: C.truth(), dash: "6 4", x: h.grid, y: h.truth_pdf },
      { label: "Parzen", color: C.parzen(), width: 1.5, x: h.grid, y: h.parzen_pdf },
      { label: "network derivative", color: C.net(), x: h.grid, y: msg.net_pdf },
    ]);
    updateNetViews(msg);
    if (msg.loss === null || msg.loss === undefined) return;   // prune re-eval: no epoch advanced

    t.hist.epoch.push(msg.epoch); t.hist.loss.push(msg.loss);
    t.hist.kst.push(msg.ks_target); t.hist.ksT.push(msg.ks_truth);
    const total = msg.epochs !== null && msg.epochs !== undefined
      ? msg.epochs.toLocaleString("en-US") : "∞";
    setTile("epoch", `${msg.epoch.toLocaleString("en-US")} <span class="unit">/ ${total}</span>`);
    setTile("loss", fmt(msg.loss));
    lossChart.set([{ label: "train MSE", color: C.net(), x: t.hist.epoch, y: t.hist.loss }]);
    ksChart.set([
      { label: "KS vs target", color: C.parzen(), x: t.hist.epoch, y: t.hist.kst },
      { label: "KS vs truth", color: C.truth(), x: t.hist.epoch, y: t.hist.ksT },
    ]);
    if (t.running && !t.paused) {
      setStatus("running", `training · epoch ${msg.epoch.toLocaleString("en-US")} of ${total} · ${msg.elapsed.toFixed(1)}s`);
    }
    log(`epoch ${String(msg.epoch).padStart(6)} · loss ${fmt(msg.loss).padStart(9)} · KS→target ${fmt(msg.ks_target)} · KS→truth ${fmt(msg.ks_truth)} · mass ${fmt(msg.mass, 4)}`);
    return;
  }
  if (msg.type === "done") {
    t.running = false; t.paused = false; t.unbounded = false;
    setStatus("", msg.stopped ? `stopped at epoch ${msg.epoch.toLocaleString("en-US")}` : `done · epoch ${msg.epoch.toLocaleString("en-US")} · ${msg.elapsed.toFixed(1)}s`);
    log(msg.stopped ? `stopped at epoch ${msg.epoch}` : `run complete at epoch ${msg.epoch} in ${msg.elapsed.toFixed(1)}s`, "em");
    setTrainUI();
    return;
  }
  if (msg.type === "status") {
    t.paused = msg.state === "paused";
    setStatus(t.paused ? "paused" : "running", t.paused ? "paused" : "training");
    setTrainUI();
    return;
  }
  if (msg.type === "saved") { log(`checkpoint saved → ${msg.path}`, "em"); return; }
  if (msg.type === "error") {
    $("#train-error").textContent = msg.message;
    log(`error: ${msg.message}`, "err");
    t.running = false; t.paused = false; t.unbounded = false;
    setStatus("", "error");
    setTrainUI();
  }
}

$("#btn-train").addEventListener("click", async () => {
  const t = state.train;
  $("#train-error").textContent = "";
  if (!state.parzenCfg) return;
  const config = {
    ...state.parzenCfg,
    estimator: $("#estimator").value,
    n_components: Math.round(+$("#n-components").value),
    hidden: [...layers],
    activation: $("#activation").value,
    optimizer: $("#optimizer").value,
    lr: +$("#lr").value,
    epochs: Math.round(+$("#epochs").value),
    snapshot_every: snapEvery(),
    target: $("#target").value,
    teacher_scale: +$("#teacher-scale").value,
    monotonicity: $("#monotonicity").value,
    mono_weight: +$("#mono-weight").value,
    curv_weight: +$("#curv-weight").value,
    rectify: $("#rectify").checked,
    net_seed: seedValue("#net-seed-rand", "#net-seed"),
  };
  try {
    await connectWS();
  } catch (e) {
    $("#train-error").textContent = e.message;
    return;
  }
  $("#net-empty").hidden = true;
  $("#net-results").hidden = false;
  resetTiles();
  t.running = true; t.paused = false; t.started = true; t.unbounded = false;
  t.hist = { epoch: [], loss: [], kst: [], ksT: [] };
  lossChart.set([]); ksChart.set([]);
  setStatus("running", "starting…");
  setTrainUI();
  wsSend({ cmd: "start", config });
  $("#nav-net").classList.add("is-done");
});

$("#btn-continue").addEventListener("click", () => {
  const t = state.train;
  $("#train-error").textContent = "";
  t.running = true; t.paused = false; t.unbounded = true;
  setStatus("running", "continuing…");
  setTrainUI();
  wsSend({ cmd: "continue", snapshot_every: snapEvery() ?? undefined });
  log("continuing training · unbounded, stop whenever you like", "em");
});

$("#btn-pause").addEventListener("click", () => {
  wsSend({ cmd: state.train.paused ? "resume" : "pause" });
});
$("#btn-stop").addEventListener("click", () => wsSend({ cmd: "stop" }));
$("#btn-save").addEventListener("click", () => wsSend({ cmd: "save", name: $("#ckpt-name").value }));
$("#btn-reset").addEventListener("click", () => {
  const t = state.train;
  t.started = false; t.hist = null; t.net = null; t.unbounded = false;
  $("#net-results").hidden = true;
  $("#net-empty").hidden = false;
  $("#trainlog").replaceChildren();
  setStatus("", "idle");
  setTrainUI();
});

/* ================================================================ nav highlighting + boot */

const sections = [["stage-dist", "nav-dist"], ["stage-parzen", "nav-parzen"], ["stage-net", "nav-net"]];
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    for (const [sec, nav] of sections) $("#" + nav).classList.toggle("is-current", sec === e.target.id);
  }
}, { rootMargin: "-30% 0px -60% 0px" });
sections.forEach(([sec]) => io.observe($("#" + sec)));

async function boot() {
  const res = await fetch("/api/registries");
  state.reg = await res.json();
  $("#kernel").innerHTML = state.reg.kernels.map((k) => `<option value="${k}">${k}</option>`).join("");
  $("#strategy").innerHTML = state.reg.strategies
    .map((s) => `<option value="${s}" ${s === "lscv" ? "selected" : ""}>${s.replace(/_/g, " ")}</option>`).join("");
  $("#activation").innerHTML = state.reg.activations
    .map((a) => `<option value="${a}" ${a === "sigmoid" ? "selected" : ""}>${a}</option>`).join("");
  state.components = PRESETS.gauss.map((c) => ({ ...c, params: { ...c.params } }));
  renderComponents();
  renderLayers();
  drawKernelPreview();
  updateStrategyUI();
  syncEstimatorUI();
  syncTargetUI();
  $("#manual-h-out").textContent = fmt(sliderToH(+$("#manual-h").value));
  setTrainUI();
  distChanged();
}
boot();
