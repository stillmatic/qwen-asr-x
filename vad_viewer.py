"""Tiny web server to browse VAD debug output in examples/vad/.

Start:  uv run python vad_viewer.py
        uv run python vad_viewer.py --port 8888
"""

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

VAD_DIR = Path("examples/vad")

INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VAD Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 24px; }
  h1 { margin-bottom: 16px; font-size: 20px; color: #58a6ff; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  .card h2 { font-size: 15px; margin-bottom: 8px; color: #e6edf3; }
  .stats { font-size: 12px; color: #8b949e; margin-bottom: 12px; }
  .stats span { margin-right: 16px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 24px; margin-bottom: 12px; }
  .control { display: flex; align-items: center; gap: 8px; font-size: 12px; }
  .control label { color: #8b949e; min-width: 90px; }
  .control input[type=range] { width: 140px; accent-color: #58a6ff; }
  .control .val { color: #e6edf3; min-width: 40px; font-variant-numeric: tabular-nums; }
  canvas { width: 100%; height: 120px; border-radius: 4px; background: #0d1117; cursor: crosshair; }
  .legend { display: flex; gap: 20px; margin-top: 6px; font-size: 12px; color: #8b949e; }
  .legend .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
  .zoom-hint { font-size: 11px; color: #484f58; margin-left: auto; }
  .empty { color: #8b949e; font-style: italic; }
  .tooltip { position: fixed; background: #30363d; color: #e6edf3; font-size: 11px; padding: 4px 8px;
             border-radius: 4px; pointer-events: none; display: none; z-index: 10; }
</style>
</head>
<body>
<h1>VAD Viewer</h1>
<div id="root"></div>
<div class="tooltip" id="tooltip"></div>
<script>

function computeSegments(probs, winSec, params) {
  const { threshold, minSpeechMs, minSilenceMs, padMs, mergeGapS } = params;
  const negThresh = threshold - 0.15;
  const minSpeechSamples = Math.round(minSpeechMs / 1000 / winSec);
  const minSilenceSamples = Math.round(minSilenceMs / 1000 / winSec);
  let raw = [];
  let inSpeech = false, start = 0, silCount = 0;
  for (let i = 0; i < probs.length; i++) {
    if (!inSpeech) {
      if (probs[i] >= threshold) { inSpeech = true; start = i; silCount = 0; }
    } else {
      if (probs[i] < negThresh) {
        silCount++;
        if (silCount >= minSilenceSamples) {
          const end = i - silCount + 1;
          if (end - start >= minSpeechSamples) raw.push({ start, end });
          inSpeech = false;
        }
      } else { silCount = 0; }
    }
  }
  if (inSpeech) {
    const end = probs.length;
    if (end - start >= minSpeechSamples) raw.push({ start, end });
  }
  const padSec = padMs / 1000;
  const dur = probs.length * winSec;
  let segs = raw.map(r => ({
    start: Math.max(0, r.start * winSec - padSec),
    end: Math.min(dur, r.end * winSec + padSec),
  }));
  if (mergeGapS > 0 && segs.length > 1) {
    let merged = [segs[0]];
    for (let i = 1; i < segs.length; i++) {
      const prev = merged[merged.length - 1];
      if (segs[i].start - prev.end <= mergeGapS) prev.end = segs[i].end;
      else merged.push(segs[i]);
    }
    segs = merged;
  }
  return segs;
}

function draw(canvas, d, params, view) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const probs = d.probs.values;
  const winSec = d.probs.window_sec;
  const dur = d.duration;

  const vStart = view.start, vEnd = view.end, vDur = vEnd - vStart;

  const segs = computeSegments(probs, winSec, params);

  // Map time to x
  const tx = t => ((t - vStart) / vDur) * W;

  // Speech segments
  ctx.fillStyle = 'rgba(247, 129, 0, 0.15)';
  for (const s of segs) {
    const x0 = tx(s.start), x1 = tx(s.end);
    if (x1 < 0 || x0 > W) continue;
    ctx.fillRect(x0, 0, x1 - x0, H);
  }
  ctx.strokeStyle = 'rgba(247, 129, 0, 0.5)';
  ctx.lineWidth = 1;
  for (const s of segs) {
    const x0 = tx(s.start), x1 = tx(s.end);
    if (x0 >= 0 && x0 <= W) { ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, H); ctx.stroke(); }
    if (x1 >= 0 && x1 <= W) { ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, H); ctx.stroke(); }
  }

  // Threshold line
  ctx.strokeStyle = '#f85149';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  const threshY = H - params.threshold * H;
  ctx.beginPath(); ctx.moveTo(0, threshY); ctx.lineTo(W, threshY); ctx.stroke();
  ctx.setLineDash([]);

  // Probability curve
  if (probs.length > 0) {
    ctx.strokeStyle = '#1f6feb';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    const iStart = Math.max(0, Math.floor(vStart / winSec) - 1);
    const iEnd = Math.min(probs.length, Math.ceil(vEnd / winSec) + 1);
    for (let i = iStart; i < iEnd; i++) {
      const x = tx(i * winSec);
      const y = H - probs[i] * H;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Time labels
  ctx.fillStyle = '#484f58';
  ctx.font = '10px system-ui';
  let step;
  if (vDur <= 2) step = 0.1;
  else if (vDur <= 5) step = 0.5;
  else if (vDur <= 20) step = 1;
  else if (vDur <= 60) step = 5;
  else if (vDur <= 120) step = 10;
  else step = 30;
  const labelStart = Math.ceil(vStart / step) * step;
  for (let t = labelStart; t <= vEnd; t += step) {
    const x = tx(t);
    const label = step < 1 ? t.toFixed(1) + 's' : Math.round(t) + 's';
    ctx.fillText(label, x + 2, H - 2);
  }

  return segs;
}

function makeSlider(label, min, max, step, value, unit, onChange) {
  const div = document.createElement('div');
  div.className = 'control';
  const valSpan = document.createElement('span');
  valSpan.className = 'val';
  valSpan.textContent = value + unit;
  const input = document.createElement('input');
  input.type = 'range'; input.min = min; input.max = max;
  input.step = step; input.value = value;
  input.addEventListener('input', () => {
    valSpan.textContent = parseFloat(input.value).toFixed(step < 1 ? 2 : 0) + unit;
    onChange(parseFloat(input.value));
  });
  const lbl = document.createElement('label');
  lbl.textContent = label;
  div.append(lbl, input, valSpan);
  return div;
}

async function load() {
  const root = document.getElementById('root');
  const tooltip = document.getElementById('tooltip');
  const res = await fetch('/api/list');
  const items = await res.json();
  if (!items.length) {
    root.innerHTML = '<p class="empty">No VAD data found. Run the server with --visualize-vad and transcribe some files.</p>';
    return;
  }
  for (const name of items) {
    const r = await fetch('/api/data/' + encodeURIComponent(name));
    const d = await r.json();

    const card = document.createElement('div');
    card.className = 'card';

    const h2 = document.createElement('h2');
    h2.textContent = name;
    card.appendChild(h2);

    const stats = document.createElement('div');
    stats.className = 'stats';
    card.appendChild(stats);

    const controls = document.createElement('div');
    controls.className = 'controls';
    card.appendChild(controls);

    const canvas = document.createElement('canvas');
    canvas.height = 120;
    canvas.style.cssText = 'width:100%;height:120px;border-radius:4px;background:#0d1117;cursor:crosshair;';
    card.appendChild(canvas);

    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = `
      <div><span class="swatch" style="background:#1f6feb"></span>Speech prob</div>
      <div><span class="swatch" style="background:#f7810040"></span>Detected speech</div>
      <div><span class="swatch" style="background:#f85149"></span>Threshold</div>
      <span class="zoom-hint">Scroll to zoom, drag to pan, double-click to reset</span>
    `;
    card.appendChild(legend);
    root.appendChild(card);

    const p = {
      threshold: d.vad_params.threshold,
      minSpeechMs: d.vad_params.min_speech_duration_ms,
      minSilenceMs: d.vad_params.min_silence_duration_ms,
      padMs: d.vad_params.speech_pad_ms,
      mergeGapS: d.vad_params.merge_gap_s,
    };
    const view = { start: 0, end: d.duration };

    function update() {
      const segs = draw(canvas, d, p, view);
      const speechSec = segs.reduce((a, s) => a + (s.end - s.start), 0).toFixed(1);
      stats.innerHTML = `<span>Duration: ${d.duration.toFixed(1)}s</span>`
        + `<span>Segments: ${segs.length}</span>`
        + `<span>Speech: ${speechSec}s</span>`
        + `<span>View: ${view.start.toFixed(1)}s \u2013 ${view.end.toFixed(1)}s</span>`;
    }

    controls.appendChild(makeSlider('Threshold', 0.05, 0.99, 0.01, p.threshold, '', v => { p.threshold = v; update(); }));
    controls.appendChild(makeSlider('Min speech', 0, 2000, 50, p.minSpeechMs, 'ms', v => { p.minSpeechMs = v; update(); }));
    controls.appendChild(makeSlider('Min silence', 0, 2000, 50, p.minSilenceMs, 'ms', v => { p.minSilenceMs = v; update(); }));
    controls.appendChild(makeSlider('Pad', 0, 500, 10, p.padMs, 'ms', v => { p.padMs = v; update(); }));
    controls.appendChild(makeSlider('Merge gap', 0, 2, 0.05, p.mergeGapS, 's', v => { p.mergeGapS = v; update(); }));

    // Zoom (scroll wheel)
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const xFrac = (e.clientX - rect.left) / rect.width;
      const tAtCursor = view.start + xFrac * (view.end - view.start);
      const factor = e.deltaY > 0 ? 1.3 : 1 / 1.3;
      const newDur = Math.min(d.duration, Math.max(0.5, (view.end - view.start) * factor));
      view.start = Math.max(0, tAtCursor - xFrac * newDur);
      view.end = Math.min(d.duration, view.start + newDur);
      view.start = view.end - Math.min(view.end - view.start, newDur);
      update();
    }, { passive: false });

    // Pan (drag)
    let dragging = false, dragStartX = 0, dragStartView = 0;
    canvas.addEventListener('mousedown', e => {
      dragging = true; dragStartX = e.clientX; dragStartView = view.start;
      canvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      const dx = e.clientX - dragStartX;
      const dt = -(dx / rect.width) * (view.end - view.start);
      const vDur = view.end - view.start;
      view.start = Math.max(0, Math.min(d.duration - vDur, dragStartView + dt));
      view.end = view.start + vDur;
      update();
    });
    window.addEventListener('mouseup', () => {
      if (dragging) { dragging = false; canvas.style.cursor = 'crosshair'; }
    });

    // Double-click to reset
    canvas.addEventListener('dblclick', () => {
      view.start = 0; view.end = d.duration; update();
    });

    // Tooltip
    canvas.addEventListener('mousemove', e => {
      if (dragging) { tooltip.style.display = 'none'; return; }
      const rect = canvas.getBoundingClientRect();
      const xFrac = (e.clientX - rect.left) / rect.width;
      const t = view.start + xFrac * (view.end - view.start);
      const idx = Math.floor(t / d.probs.window_sec);
      const prob = idx >= 0 && idx < d.probs.values.length ? d.probs.values[idx] : 0;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 28) + 'px';
      tooltip.textContent = `${t.toFixed(2)}s  prob: ${prob.toFixed(3)}`;
    });
    canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

    update();
  }
}

load();
</script>
</body>
</html>
"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._respond(200, "text/html", INDEX_HTML)
        elif self.path == "/api/list":
            names = sorted(
                d.name for d in VAD_DIR.iterdir()
                if d.is_dir() and (d / "vad.json").exists()
            ) if VAD_DIR.exists() else []
            self._respond(200, "application/json", json.dumps(names))
        elif self.path.startswith("/api/data/"):
            from urllib.parse import unquote
            name = unquote(self.path[len("/api/data/"):])
            fpath = VAD_DIR / name / "vad.json"
            if fpath.exists():
                self._respond(200, "application/json", fpath.read_text())
            else:
                self._respond(404, "text/plain", "not found")
        else:
            self._respond(404, "text/plain", "not found")

    def _respond(self, code, content_type, body):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # silence request logs


def main():
    parser = argparse.ArgumentParser(description="VAD debug viewer")
    parser.add_argument("--port", type=int, default=8899, help="Port (default: 8899)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"VAD viewer running at http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
