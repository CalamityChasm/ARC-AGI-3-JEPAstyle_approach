"""Render an ARC-AGI-3 recording (*.recording.jsonl) as a self-contained,
scrubbable HTML replay -- no server needed, just open the output file in a
browser. Built to answer "what is the agent actually doing on a game it
struggles with," which nothing else in this repo shows visually; every
other tool here reports numbers (scorecards, changed-patches percentages),
not frames.

Usage:
    python scripts/visualize_recording.py --game s5i5 --out viz.html
    python scripts/visualize_recording.py --game s5i5 --agent hypothesis --out viz.html
    python scripts/visualize_recording.py --recording path/to/x.recording.jsonl --out viz.html

--game finds the most recently modified matching recording under
ARC-AGI-3-Agents/recordings/ and (if present) the E: drive overflow
archive this project uses when local disk gets tight -- see CLAUDE.md's
Gotchas section. --agent narrows to recordings from that agent specifically
(filename convention: <game>-<hash>.<agent>.<guid>.recording.jsonl).
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDING_DIRS = [
    REPO_ROOT / "ARC-AGI-3-Agents" / "recordings",
    Path("E:/jepa_overflow/recordings"),
]

# Extends the standard 10-color ARC-1/2 palette (0-9) with 6 more distinct
# colors for ARC-3's 16-color range (jepa/grid.py: NUM_COLORS=16) -- purely
# a visualization choice, the model itself only ever sees integer ids.
PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
    "#FFFFFF", "#B10DC9", "#39CCCC", "#01FF70", "#8B4513", "#FFB6C1",
]

ACTION_NAMES = {
    0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4",
    5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}


def find_recording(game: str, agent: str | None) -> Path:
    candidates = []
    for d in RECORDING_DIRS:
        if not d.exists():
            continue
        for f in d.glob(f"{game}-*.recording.jsonl"):
            if agent and f".{agent}." not in f.name:
                continue
            candidates.append(f)
    if not candidates:
        raise FileNotFoundError(
            f"No recording found for game={game!r} agent={agent!r} under {RECORDING_DIRS}"
        )
    return max(candidates, key=lambda f: f.stat().st_mtime)


def load_steps(path: Path) -> list[dict]:
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            d = rec["data"]
            frame = d["frame"][0]  # (H, W) -- recordings wrap in a 1-layer list
            action = d.get("action_input") or {}
            action_id = action.get("id", 0)
            action_data = action.get("data") or {}
            steps.append({
                "frame": frame,
                "state": d.get("state"),
                "levels_completed": d.get("levels_completed", 0),
                "win_levels": d.get("win_levels", 0),
                "guid": d.get("guid"),
                "action_id": action_id,
                "action_name": ACTION_NAMES.get(action_id, f"?{action_id}"),
                "x": action_data.get("x"),
                "y": action_data.get("y"),
                "available_actions": d.get("available_actions", []),
            })
    return steps


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; background: #1e1e1e; color: #ddd; margin: 0; padding: 16px; }}
  h1 {{ font-size: 15px; color: #999; font-weight: normal; margin: 0 0 12px; }}
  .layout {{ display: flex; gap: 20px; align-items: flex-start; }}
  #gridCanvas {{ border: 1px solid #444; image-rendering: pixelated; cursor: crosshair; }}
  .panel {{ min-width: 260px; }}
  .row {{ margin-bottom: 8px; font-size: 13px; }}
  .row b {{ color: #7FDBFF; }}
  .controls {{ display: flex; gap: 6px; align-items: center; margin: 10px 0; }}
  button {{ background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 5px 10px; cursor: pointer; font-size: 13px; }}
  button:hover {{ background: #444; }}
  button.active {{ background: #0074D9; border-color: #0074D9; }}
  input[type=range] {{ flex: 1; }}
  #stepLabel {{ font-variant-numeric: tabular-nums; min-width: 90px; text-align: center; }}
  .win {{ color: #2ECC40; font-weight: bold; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; background: #333; }}
  #cellInfo {{ font-size: 12px; color: #888; margin-top: 6px; min-height: 16px; }}
</style>
</head>
<body>
<h1>{subtitle}</h1>
<div class="layout">
  <canvas id="gridCanvas" width="512" height="512"></canvas>
  <div class="panel">
    <div class="row">Step <b id="stepNum">0</b> / <span id="stepMax">0</span></div>
    <div class="row">Action: <b id="actionLabel">-</b></div>
    <div class="row">State: <b id="stateLabel">-</b></div>
    <div class="row">Levels: <b id="levelsLabel">-</b></div>
    <div class="row">Episode: <span id="episodeLabel" class="badge">-</span></div>
    <div class="controls">
      <button id="prevBtn">&larr; prev</button>
      <button id="playBtn">&#9654; play</button>
      <button id="nextBtn">next &rarr;</button>
    </div>
    <input type="range" id="slider" min="0" value="0">
    <div class="controls">
      <button id="diffBtn">highlight changes</button>
      <label style="font-size:12px;"><input type="checkbox" id="skipReset" checked> skip to next reset on play</label>
    </div>
    <div id="cellInfo"></div>
  </div>
</div>
<script>
const STEPS = {steps_json};
const PALETTE = {palette_json};
const CELL = 8; // 64 * 8 = 512px canvas
const canvas = document.getElementById('gridCanvas');
const ctx = canvas.getContext('2d');
let idx = 0, playing = false, playTimer = null, showDiff = false;

function draw() {{
  const step = STEPS[idx];
  const prev = idx > 0 ? STEPS[idx - 1] : null;
  const frame = step.frame;
  for (let r = 0; r < 64; r++) {{
    for (let c = 0; c < 64; c++) {{
      const v = frame[r][c];
      ctx.fillStyle = PALETTE[v] || '#FF00FF';
      ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
      if (showDiff && prev && prev.frame[r][c] !== v) {{
        ctx.strokeStyle = '#FF4136';
        ctx.lineWidth = 1;
        ctx.strokeRect(c * CELL + 0.5, r * CELL + 0.5, CELL - 1, CELL - 1);
      }}
    }}
  }}
  // ACTION6 click marker: shown on the frame the click produced
  if (step.action_id === 6 && step.x !== null && step.y !== null) {{
    ctx.strokeStyle = '#FFDC00';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(step.x * CELL + CELL / 2, step.y * CELL + CELL / 2, CELL, 0, 2 * Math.PI);
    ctx.stroke();
  }}

  document.getElementById('stepNum').textContent = idx;
  document.getElementById('stepMax').textContent = STEPS.length - 1;
  let actionLabel = step.action_name;
  if (step.action_id === 6) actionLabel += ` @ (${{step.x}}, ${{step.y}})`;
  document.getElementById('actionLabel').textContent = actionLabel;
  const stateEl = document.getElementById('stateLabel');
  stateEl.textContent = step.state;
  stateEl.className = step.state === 'WIN' ? 'win' : '';
  document.getElementById('levelsLabel').textContent = `${{step.levels_completed}} / ${{step.win_levels}}`;
  document.getElementById('episodeLabel').textContent = (step.guid || '-').slice(0, 8);
  document.getElementById('slider').value = idx;
}}

function step(delta) {{
  idx = Math.max(0, Math.min(STEPS.length - 1, idx + delta));
  draw();
}}

document.getElementById('prevBtn').onclick = () => step(-1);
document.getElementById('nextBtn').onclick = () => step(1);
document.getElementById('slider').max = STEPS.length - 1;
document.getElementById('slider').oninput = (e) => {{ idx = parseInt(e.target.value); draw(); }};
document.getElementById('diffBtn').onclick = (e) => {{
  showDiff = !showDiff;
  e.target.classList.toggle('active', showDiff);
  draw();
}};
document.getElementById('playBtn').onclick = (e) => {{
  playing = !playing;
  e.target.textContent = playing ? '&#10074;&#10074; pause' : '&#9654; play';
  e.target.innerHTML = playing ? '&#10074;&#10074; pause' : '&#9654; play';
  if (playing) {{
    playTimer = setInterval(() => {{
      const skipReset = document.getElementById('skipReset').checked;
      if (idx >= STEPS.length - 1) {{ idx = 0; }}
      else {{
        idx++;
        if (skipReset) {{
          while (idx < STEPS.length - 1 && STEPS[idx].guid !== STEPS[idx - 1].guid) idx++;
        }}
      }}
      draw();
    }}, 120);
  }} else {{
    clearInterval(playTimer);
  }}
}};
canvas.onmousemove = (e) => {{
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor((e.clientX - rect.left) / CELL);
  const r = Math.floor((e.clientY - rect.top) / CELL);
  if (r >= 0 && r < 64 && c >= 0 && c < 64) {{
    document.getElementById('cellInfo').textContent = `(x=${{c}}, y=${{r}}) color=${{STEPS[idx].frame[r][c]}}`;
  }}
}};
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowRight') step(1);
  if (e.key === 'ArrowLeft') step(-1);
}});
draw();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an ARC-AGI-3 recording as a scrubbable HTML replay.")
    parser.add_argument("--recording", type=Path, default=None, help="Explicit path to a .recording.jsonl file.")
    parser.add_argument("--game", default=None, help="Game id prefix; finds the most recent matching recording.")
    parser.add_argument("--agent", default=None, help="Narrow --game search to this agent's recordings.")
    parser.add_argument("--out", type=Path, required=True, help="Output HTML file path.")
    args = parser.parse_args()

    if args.recording:
        path = args.recording
    elif args.game:
        path = find_recording(args.game, args.agent)
    else:
        parser.error("Provide either --recording or --game")

    steps = load_steps(path)
    n_episodes = len({s["guid"] for s in steps})
    wins = sum(1 for s in steps if s["state"] == "WIN")

    html = HTML_TEMPLATE.format(
        title=f"Replay: {path.name}",
        subtitle=f"{path.name} -- {len(steps)} steps, {n_episodes} episode(s){' -- reached WIN' if wins else ''}",
        steps_json=json.dumps(steps),
        palette_json=json.dumps(PALETTE),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Loaded: {path}")
    print(f"Steps: {len(steps)}  Episodes: {n_episodes}  Reached WIN: {bool(wins)}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
