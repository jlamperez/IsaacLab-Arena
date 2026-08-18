# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Build a scrubbable HTML viewer of predicted wrist target vs. actual wrist, over a chunk range.

Companion to ``project_wrist_target.py`` (reuses its ``wrist_target_pixels()``/``annotate_frame()``)
-- that script annotates one ``chunk_NNN.npz``/``chunk_NNN_first_person.png`` pair at a time; this
one does the same over an entire range and packages the result as a single self-contained HTML
file (images and per-chunk pixel data embedded inline, no external files) with a slider, a
play/pause button, a numeric readout, and an x-pixel trend chart with the visible-frame band
([0, 224]) marked -- built 2026-08-18 to scrub through a failed ``g1_dex1_ikea_lightwheel`` episode
chunk by chunk instead of eyeballing annotated PNGs one at a time.

The script has zero simulation dependency and only requires ``numpy``/``Pillow`` -- run it against
a debug-dump directory copied out of the container (e.g. via ``docker cp``), same as
``project_wrist_target.py``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/build_wrist_target_viewer.py \\
        /tmp/gr00t_dex1_wbc_debug_host --start 1 --end 40 -o /tmp/wrist_target_viewer.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from project_wrist_target import annotate_frame, wrist_target_pixels

_FRAME_SIZE = 224
_DISPLAY_SCALE = 3  # upsample the embedded 224x224 frames for on-screen sharpness

_HTML_TEMPLATE = r"""<title>__TITLE__</title>
<style>
  :root {
    --bg: #0d1015;
    --panel: #151a22;
    --panel-2: #1b212b;
    --border: #262e3a;
    --text: #e7eaef;
    --text-dim: #8891a3;
    --text-faint: #565f70;
    --left: #ff6b5b;
    --left-dim: #7a3a34;
    --right: #33d6e6;
    --right-dim: #235e66;
    --frame-band: #ffffff14;
    --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }

  * { box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    margin: 0;
    padding: 28px 32px 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 22px;
  }

  .app { width: 100%; max-width: 980px; display: flex; flex-direction: column; gap: 18px; }

  header { display: flex; flex-direction: column; gap: 4px; }

  h1 {
    font-size: 1.25rem;
    font-weight: 650;
    margin: 0;
    letter-spacing: -0.01em;
    text-wrap: balance;
  }

  .context { margin: 0; color: var(--text-dim); font-size: 0.85rem; }
  .context b { color: var(--text); font-weight: 600; }

  main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 260px;
    gap: 18px;
    align-items: start;
  }

  .viewer {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .viewer img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    border-radius: 6px;
    background: #000;
    display: block;
  }

  .legend { display: flex; gap: 18px; font-size: 0.78rem; color: var(--text-dim); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .swatch.left { background: var(--left); }
  .swatch.right { background: var(--right); }

  aside.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .chunk-readout {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.95rem;
    color: var(--text);
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }

  .chunk-readout .status {
    font-family: var(--sans);
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 20px;
    background: var(--panel-2);
    color: var(--text-dim);
  }

  .chunk-readout .status.warn { color: #ffb347; background: #4a370f; }

  table.data-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.76rem;
  }

  table.data-table th {
    text-align: right;
    font-family: var(--sans);
    font-weight: 500;
    color: var(--text-faint);
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0 0 6px;
  }

  table.data-table th:first-child, table.data-table td:first-child { text-align: left; }

  table.data-table td {
    text-align: right;
    padding: 5px 0;
    border-top: 1px solid var(--border);
    color: var(--text);
  }

  table.data-table tr.left td:first-child { color: var(--left); font-weight: 600; }
  table.data-table tr.right td:first-child { color: var(--right); font-weight: 600; }

  .delta-hi { color: #ffb347; font-weight: 600; }

  .chart-block { display: flex; flex-direction: column; gap: 6px; }

  .chart-title {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-faint);
  }

  canvas#trend { width: 100%; height: 128px; display: block; }

  .controls {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 14px;
    width: 100%;
  }

  button#play {
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--text);
    width: 34px;
    height: 34px;
    border-radius: 8px;
    font-size: 0.85rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  button#play:hover { background: #222a36; }
  button#play:focus-visible, input#slider:focus-visible { outline: 2px solid var(--right); outline-offset: 2px; }

  input#slider { flex: 1; accent-color: var(--right); height: 4px; }

  .step-label {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.78rem;
    color: var(--text-dim);
    width: 76px;
    text-align: right;
    flex-shrink: 0;
  }

  @media (max-width: 720px) {
    main { grid-template-columns: 1fr; }
  }
</style>

<div class="app">
  <header>
    <h1>Target de muñeca predicho vs. muñeca real</h1>
    <p class="context">__CONTEXT__</p>
  </header>

  <main>
    <div class="viewer">
      <img id="frame" src="" alt="Frame de cámara con muñeca y target anotados">
      <div class="legend">
        <span><span class="swatch left"></span>izquierda: muñeca (●) / target (+)</span>
        <span><span class="swatch right"></span>derecha: muñeca (●) / target (+)</span>
      </div>
    </div>

    <aside class="panel">
      <div class="chunk-readout">
        <span>chunk <span id="chunkNum">__START__</span></span>
        <span class="status" id="statusBadge">dentro de cuadro</span>
      </div>

      <table class="data-table">
        <thead>
          <tr><th></th><th>muñeca (px)</th><th>target (px)</th><th>Δ</th></tr>
        </thead>
        <tbody>
          <tr class="left"><td>L</td><td id="lw">—</td><td id="lt">—</td><td id="ld">—</td></tr>
          <tr class="right"><td>R</td><td id="rw">—</td><td id="rt">—</td><td id="rd">—</td></tr>
        </tbody>
      </table>

      <div class="chart-block">
        <div class="chart-title">coordenada x del target a lo largo del rango</div>
        <canvas id="trend" width="520" height="256"></canvas>
      </div>
    </aside>
  </main>

  <div class="controls">
    <button id="play" aria-label="Reproducir secuencia" aria-pressed="false">▶</button>
    <input type="range" id="slider" min="__START__" max="__END__" value="__START__" step="1" aria-label="Chunk">
    <span class="step-label" id="stepLabel"></span>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const IMAGES = __IMAGES_JSON__;
const FRAME_W = __FRAME_SIZE__, FRAME_H = __FRAME_SIZE__;
const START = __START__, END = __END__;

const img = document.getElementById('frame');
const slider = document.getElementById('slider');
const chunkNum = document.getElementById('chunkNum');
const stepLabel = document.getElementById('stepLabel');
const statusBadge = document.getElementById('statusBadge');
const lw = document.getElementById('lw'), lt = document.getElementById('lt'), ld = document.getElementById('ld');
const rw = document.getElementById('rw'), rt = document.getElementById('rt'), rd = document.getElementById('rd');
const playBtn = document.getElementById('play');
const canvas = document.getElementById('trend');
const ctx = canvas.getContext('2d');

function dist(a, b) { return (!a || !b) ? null : Math.hypot(a[0] - b[0], a[1] - b[1]); }
function fmtPx(p) { return p ? `${p[0].toFixed(0)},${p[1].toFixed(0)}` : '—'; }
function offFrame(p) { return p && (p[0] < 0 || p[0] > FRAME_W || p[1] < 0 || p[1] > FRAME_H); }

function render(i) {
  const row = DATA[i - START];
  img.src = 'data:image/png;base64,' + IMAGES[i - START];
  chunkNum.textContent = String(i).padStart(3, '0');
  stepLabel.textContent = `${i} / ${END}`;
  slider.value = i;

  lw.textContent = fmtPx(row.left_wrist_px);
  lt.textContent = fmtPx(row.left_target_px);
  rw.textContent = fmtPx(row.right_wrist_px);
  rt.textContent = fmtPx(row.right_target_px);

  const dl = dist(row.left_wrist_px, row.left_target_px);
  const dr = dist(row.right_wrist_px, row.right_target_px);
  ld.textContent = dl == null ? '—' : dl.toFixed(0) + 'px';
  rd.textContent = dr == null ? '—' : dr.toFixed(0) + 'px';
  ld.className = dl != null && dl > 60 ? 'delta-hi' : '';
  rd.className = dr != null && dr > 60 ? 'delta-hi' : '';

  const anyOff = offFrame(row.left_wrist_px) || offFrame(row.left_target_px) ||
                 offFrame(row.right_wrist_px) || offFrame(row.right_target_px);
  statusBadge.textContent = anyOff ? 'fuera de cuadro' : 'dentro de cuadro';
  statusBadge.className = 'status' + (anyOff ? ' warn' : '');

  drawChart(i);
}

function drawChart(activeChunk) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 34, padR = 10, padT = 10, padB = 20;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  const minX = START, maxX = END;
  const allVals = [];
  DATA.forEach(r => {
    [r.left_wrist_px, r.left_target_px, r.right_wrist_px, r.right_target_px].forEach(p => {
      if (p) allVals.push(p[0]);
    });
  });
  const minV = Math.min(0, ...allVals) - 10;
  const maxV = Math.max(FRAME_W, ...allVals) + 10;

  const xPix = c => padL + ((c - minX) / (maxX - minX || 1)) * plotW;
  const yPix = v => padT + (1 - (v - minV) / (maxV - minV)) * plotH;

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--frame-band').trim();
  ctx.fillRect(padL, yPix(FRAME_W), plotW, yPix(0) - yPix(FRAME_W));

  ctx.strokeStyle = '#333c4a';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = '#565f70';
  [0, FRAME_W].forEach(v => {
    const y = yPix(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillText(String(v), 2, y + 3);
  });

  function line(key, color, dashed) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.setLineDash(dashed ? [4, 3] : []);
    let started = false;
    DATA.forEach(r => {
      const p = r[key];
      if (!p) { started = false; return; }
      const x = xPix(r.chunk), y = yPix(p[0]);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  line('left_wrist_px', getComputedStyle(document.documentElement).getPropertyValue('--left-dim').trim(), false);
  line('left_target_px', getComputedStyle(document.documentElement).getPropertyValue('--left').trim(), true);
  line('right_wrist_px', getComputedStyle(document.documentElement).getPropertyValue('--right-dim').trim(), false);
  line('right_target_px', getComputedStyle(document.documentElement).getPropertyValue('--right').trim(), true);

  const px = xPix(activeChunk);
  ctx.strokeStyle = '#e7eaef55';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, h - padB); ctx.stroke();
}

let playing = false, timer = null;
playBtn.addEventListener('click', () => {
  playing = !playing;
  playBtn.textContent = playing ? '❚❚' : '▶';
  playBtn.setAttribute('aria-pressed', String(playing));
  if (playing) {
    timer = setInterval(() => {
      let next = parseInt(slider.value, 10) + 1;
      if (next > END) next = START;
      render(next);
    }, 260);
  } else {
    clearInterval(timer);
  }
});

slider.addEventListener('input', () => {
  if (playing) { playing = false; playBtn.textContent = '▶'; playBtn.setAttribute('aria-pressed', 'false'); clearInterval(timer); }
  render(parseInt(slider.value, 10));
});

render(START);
</script>
"""


def build_viewer_html(debug_dir: Path, start: int, end: int, title: str) -> str:
    """Render the self-contained HTML viewer for ``chunk_{start:03d}..{end:03d}`` in ``debug_dir``."""
    rows = []
    images_b64 = []
    for i in range(start, end + 1):
        npz_path = debug_dir / f"chunk_{i:03d}.npz"
        png_path = debug_dir / f"chunk_{i:03d}_first_person.png"
        d = np.load(npz_path)
        image = Image.open(png_path)
        pixels = wrist_target_pixels(d, *image.size)

        row = {"chunk": i}
        for side in ("left", "right"):
            row[f"{side}_wrist_px"] = pixels[f"{side}_wrist_px"]
            row[f"{side}_target_px"] = pixels[f"{side}_target_px"]
        rows.append(row)

        annotated = annotate_frame(image, pixels)
        annotated = annotated.resize(
            (annotated.width * _DISPLAY_SCALE, annotated.height * _DISPLAY_SCALE), Image.NEAREST
        )
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    context = (
        f"<b>{debug_dir.name}</b> · chunks {start:03d}&ndash;{end:03d} · posiciones en píxel sobre "
        f"<code>dataset_first_person_cam</code> ({_FRAME_SIZE}×{_FRAME_SIZE})"
    )

    html = _HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__CONTEXT__", context)
    html = html.replace("__FRAME_SIZE__", str(_FRAME_SIZE))
    html = html.replace("__START__", str(start))
    html = html.replace("__END__", str(end))
    html = html.replace("__DATA_JSON__", json.dumps(rows))
    html = html.replace("__IMAGES_JSON__", json.dumps(images_b64))
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("debug_dir", help="Directory of chunk_NNN.npz / chunk_NNN_first_person.png pairs.")
    parser.add_argument("--start", type=int, default=1, help="First chunk number, inclusive (default: 1).")
    parser.add_argument("--end", type=int, required=True, help="Last chunk number, inclusive.")
    parser.add_argument("-o", "--output", required=True, help="Where to write the HTML viewer.")
    parser.add_argument(
        "--title", default="Wrist target vs. real", help="Page title (default: 'Wrist target vs. real')."
    )
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    html = build_viewer_html(debug_dir, args.start, args.end, args.title)
    Path(args.output).write_text(html)
    print(f"Wrote viewer for chunks {args.start:03d}-{args.end:03d} to {args.output}")


if __name__ == "__main__":
    main()
