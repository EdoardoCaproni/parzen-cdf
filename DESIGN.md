# Design

Visual system of the interactive lab (`app/static/`). Register: **product** (see PRODUCT.md).
Theme: light "lab notebook" — cobalt ink on white paper; the interface recedes, the math is the show.

## Color (OKLCH; tokens in `app/static/style.css` `:root`)

| Role | Value | Use |
|---|---|---|
| `--bg` | `oklch(100% 0 0)` | page (pure white, no hidden warmth) |
| `--surface` / `--surface-2` | `oklch(97.8% 0.003 244)` / `oklch(95.8% 0.005 244)` | panels / wells |
| `--ink` / `--ink-2` / `--muted` | L 24% / 43% / 55%, hue 250 | body / secondary / minor text |
| `--hairline` / `--baseline` | L 91% / 82% | gridlines, borders / axes, control borders |
| `--primary` | `oklch(42% 0.125 244)` | cobalt: primary buttons, current stage, focus ring (white text on fills) |
| `--danger-ink` / `--good-ink` | hue 25 / 150 | error text, bad metrics / good metrics |

Strategy: **restrained** — tinted neutrals plus one cobalt accent; color for state, never decoration.

### Chart series (validated with the dataviz palette validator, CVD ΔE ≥ 25)

| Entity | Token | Value | Line style |
|---|---|---|---|
| truth | `--c-truth` | `#6f6e69` | 2px dashed `6 4` |
| Parzen estimate / target | `--c-parzen` | `#2a78d6` | 2px solid |
| network | `--c-net` | `#eb6834` | 2px solid |
| empirical CDF | `--c-emp` | `#1baf7a` | 1.5px step (sub-3:1 contrast: always legended + tooltip) |
| weights diverging | `--w-pos` / `--w-neg` / `--w-mid` | `#2a78d6` / `#e34948` / `#f0efec` | network edges only |

Color follows the entity across every chart (truth is always gray-dashed, etc.). Identity is
never color-alone: legend + line style + tooltip.

## Typography

One family: `system-ui` stack. Base 14px/1.5; labels 12px w550; panel/chart titles 13px w650;
stage headings 20px w650, `text-wrap: balance`. Monospace (`ui-monospace`) only in the training
log. `tabular-nums` only where digits align (log columns, slider outputs); tiles use
proportional figures.

## Layout & components

- Stage sections in pipeline order (1 Distribution → 2 Parzen → 3 Network), each a
  `344px | 1fr` grid (controls panel left, viz column right), stacking under 960px.
- Sticky top bar: brand + stage pills (current = cobalt wash; done = filled numeral).
- Panels: `--surface`, 1px hairline, radius 10px. Charts: white cards, radius 10px.
- Buttons: primary = filled cobalt/white; others ghost with `--baseline` border. All controls
  have hover, focus-visible (2px cobalt outline), and disabled states.
- Stat tiles: hairline-separated row; label 11.5px `--ink-2`, value 17px w600.
- Charts (`Chart` class in app.js): hairline solid y-gridlines, `--baseline` x-axis, 2px round
  lines, legend above-right (hidden for single series), crosshair + tooltip (values lead,
  labels follow, line-keys) with keyboard access (arrow keys on focused plot).
- Caps are stated in captions, never silent (target dots shown, network edges shown, rug ticks).

## Motion

150–250ms `cubic-bezier(0.22,1,0.36,1)` on hover/state only. Content animations (Parzen
construction, live training) are data, driven by rAF. `prefers-reduced-motion`: transitions
collapse and the construction animation jumps to the final state.
