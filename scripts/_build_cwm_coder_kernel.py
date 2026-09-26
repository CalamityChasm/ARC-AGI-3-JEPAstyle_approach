"""Build the coder-model arm of the CodeWorldModel backtest.

The question: does a **coder-specialised** model write better world models
than the general model our appliance serves?

Measured so far on Qwen3.8-Flash-Next-NVFP4, 12 segments per arm:
  think-16k   1/12 passed, but 27/27 replies truncated -- budget-bound
  nothink-8k  0/12 passed, 4 loadable-and-ran, one 9/40 prefix

This arm swaps only the model: **Qwen3-Coder-30B-A3B-Instruct**, mounted
from Kaggle's own model registry, on the same 12 segments with the same
prompt and the same calibration (null floor 0/61, proven ceiling 58/61).

Why this kernel is far simpler than the prototype's
---------------------------------------------------
**The backtest plays no games.** It needs no ARC runtime, no competition
wheelhouse, no anim solver bundle, and no vLLM -- only the segments, the
package and the model. Every one of those is a component that has already
failed a run in this arm, so dropping them is the point.

`transformers` is used deliberately rather than vLLM. Throughput is the
reason `transformers` is wrong for the *live agent* (it cannot multiplex
games), but the backtest makes only ~36 calls total, so the objection does
not apply -- and it avoids serving a non-NVFP4 model through a runtime
built for one.

Usage:
    python scripts/_build_cwm_coder_kernel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

KERNEL_SLUG = "arc3-cwm-backtest-coder"
OWNER = "calamitychasm"
DATASET = "cwm-backtest"
#: Already mounted by the llm-world-engine kernels, so it is known to
#: resolve on this account.
CODER_MODEL = "qwen-lm/qwen3-coder/Transformers/30b-a3b-instruct/1"

HEADER = """# arc3-cwm-backtest-coder - does a CODER model write better world models?

Free run, **no submission quota**.

Swaps only the model. Same 12 segments, same prompt, same calibration as the
`Qwen3.8-Flash-Next-NVFP4` arms (null floor **0/61**, proven ceiling
**58/61** via a generated lookup-table oracle).

| arm | passed |
|---|---|
| Flash-Next, thinking, 16k | 1/12 (but 27/27 replies truncated) |
| Flash-Next, no thinking, 8k | 0/12 (4 loaded and ran) |
| **Qwen3-Coder-30B-A3B** | this run |

No games are played, so this kernel needs no ARC runtime, no solver bundle
and no vLLM -- only the segments, the package and the model.
"""

SETUP_CELL = r'''
# ============================================================
# Environment + input validation. Cheap, and FIRST.
# A previous run in this arm spent a 7.5h queue wait plus 14 minutes of
# GPU to discover a missing filename; everything checkable without the
# model is checked here.
# ============================================================
import glob, json, os, subprocess, sys, time
from pathlib import Path

T0 = time.time()
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip() or "nvidia-smi unavailable", flush=True)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "no GPU"
_p = torch.cuda.get_device_properties(0)
_vram = _p.total_memory / 1e9
print(f"gpu {_p.name} sm_{_p.major}{_p.minor} {_vram:.1f} GB "
      f"bf16={torch.cuda.is_bf16_supported()}", flush=True)

# Qwen3-Coder-30B-A3B is ~60 GB in bf16. v1 of this kernel was handed a
# Tesla T4 (15.6 GB) despite requesting NvidiaRtxPro6000, so device_map=
# "auto" silently offloaded to disk and died ~10 minutes later inside
# from_pretrained with a confusing missing-offload_folder ValueError.
#
# The cause was almost certainly this kernel being the only one of 17 in
# the repo WITHOUT `competition_sources` -- that attachment appears to gate
# the competition's premium hardware pool. Re-added; this assert is the
# backstop so a wrong card costs seconds, not a queue slot.
MIN_VRAM_GB = float(os.environ.get("CWM_MIN_VRAM_GB", "40"))
assert _vram >= MIN_VRAM_GB, (
    f"got {_p.name} with {_vram:.1f} GB, need >= {MIN_VRAM_GB} GB for a 30B model. "
    "Check that kernel-metadata.json still has competition_sources AND "
    "machine_shape=NvidiaRtxPro6000 -- dropping the former silently downgrades "
    "the card even when the latter is set."
)


def find_input_dir(name):
    base = Path("/kaggle/input")
    for path in base.rglob("*"):
        if path.is_dir() and path.name == name:
            return path
    raise RuntimeError(f"{name!r} not found under /kaggle/input")


DATA_DIR = find_input_dir("__DATASET__")
sys.path.insert(0, str(DATA_DIR))
os.environ["ARC3_CWM_ENGINE_DIR"] = str(DATA_DIR)
print("backtest data:", DATA_DIR, flush=True)

# Kaggle decompresses an uploaded .gz, so accept either name.
_cands = ["cwm_segments.json", "cwm_segments.json.gz"]
SEGMENTS_PATH = next((DATA_DIR / c for c in _cands if (DATA_DIR / c).is_file()), None)
if SEGMENTS_PATH is None:
    raise FileNotFoundError(
        f"no segments file in {DATA_DIR}; tried {_cands}; dir holds "
        f"{sorted(p.name for p in DATA_DIR.iterdir())}"
    )
print("segments file:", SEGMENTS_PATH.name, flush=True)

# Locate the coder model by its config.json, not a guessed path.
def find_model_dir(keyword):
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
            if keyword in p.lower()]
    return sorted(hits, key=len)[0] if hits else None


CODER_MODEL_DIR = find_model_dir("qwen3-coder") or find_model_dir("coder") or find_model_dir("qwen")
print("CODER_MODEL_DIR:", CODER_MODEL_DIR, flush=True)
assert CODER_MODEL_DIR, "no coder model mounted -- check kernel-metadata model_sources"

os.environ["LLM_BACKEND"] = "transformers"
os.environ["CODER_MODEL_DIR"] = CODER_MODEL_DIR

from arc3_cwm.determinism import census
from arc3_cwm.oracle import verify_oracle
from arc3_cwm.serialize import load_segments

_segs = load_segments(SEGMENTS_PATH)
print(f"loaded {len(_segs)} segments from {len({s.game_id for s in _segs})} games", flush=True)
_det = census(_segs)
print(_det.summary(), flush=True)
_oracle_ok = sum(1 for s in _segs if verify_oracle(s)[0])
print(f"positive control: oracle replays {_oracle_ok}/{len(_segs)} segments", flush=True)
assert _oracle_ok > 0, (
    "the oracle cannot pass a single segment -- the harness could not report a "
    "pass even if the model produced one, so any result would be meaningless"
)
print(f"INPUT VALIDATION PASSED in {time.time()-T0:.1f}s", flush=True)
'''

BACKTEST_CELL = r'''
# ============================================================
# The measurement. Only the MODEL differs from the Flash-Next arms.
# ============================================================
from arc3_cwm.harness import BacktestConfig, run_segment
from arc3_cwm.report import build_report, per_game_table

MAX_SEGMENTS = int(os.environ.get("CWM_MAX_SEGMENTS", "12"))
MAX_ATTEMPTS = int(os.environ.get("CWM_MAX_ATTEMPTS", "3"))
MAX_TOKENS = int(os.environ.get("CWM_MAX_TOKENS", "8192"))
SOFT_DEADLINE_S = float(os.environ.get("CWM_SOFT_DEADLINE_S", str(6.0 * 3600)))

RESULTS_PATH = Path("/kaggle/working/cwm_coder_results.json")
SOURCES_DIR = Path("/kaggle/working/passing_models")

segments = load_segments(SEGMENTS_PATH)

# Round-robin across games, matching the Flash-Next arms exactly so the
# comparison is on identical segments -- not the alphabetical file order.
by_game = {}
for seg in segments:
    by_game.setdefault(seg.game_id, []).append(seg)
for group in by_game.values():
    group.sort(key=lambda s: s.level)
ordered, depth = [], 0
while len(ordered) < len(segments):
    added = False
    for game in sorted(by_game):
        if depth < len(by_game[game]):
            ordered.append(by_game[game][depth]); added = True
    if not added:
        break
    depth += 1
segments = ordered[:MAX_SEGMENTS]
print(f"pilot: {len(segments)} segments across "
      f"{len({s.game_id for s in segments})} games "
      f"(levels {sorted({s.level for s in segments})})", flush=True)

# If anything still spills, give it somewhere to go rather than raising.
os.environ.setdefault("CWM_OFFLOAD_DIR", "/kaggle/working/offload")
Path(os.environ["CWM_OFFLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

print("loading the coder model (this is the slow part)...", flush=True)
_t = time.time()
from llm_engine.llm_client import make_client

client = make_client("coder")
print(f"model loaded in {time.time()-_t:.1f}s", flush=True)

# Preflight that exercises the REAL shape of the task. The Flash-Next run's
# preflight used a short prompt, passed, and told us nothing -- every real
# prompt then behaved differently.
_probe = client.complete(
    "You write Python. Reply with only a code fence.",
    "Write a class named WorldModel with a predict(self, state, action_name, "
    "x=None, y=None) method returning (state, 0, False) and a goal_hint(self, "
    "state) method returning 0.0.",
    max_tokens=512,
)
print(f"preflight reply ({len(_probe)} chars):\n{_probe[:600]}", flush=True)
assert _probe.strip(), "model returned nothing -- do NOT trust any result from this run"
print("preflight contains a WorldModel class:", "class WorldModel" in _probe, flush=True)

config = BacktestConfig(max_attempts=MAX_ATTEMPTS, max_tokens=MAX_TOKENS)
results = []
started = time.time()


def _persist(partial):
    payload = build_report(results).as_dict()
    payload["determinism"] = _det.as_dict()
    payload["oracle_passing_segments"] = _oracle_ok
    payload["model_dir"] = CODER_MODEL_DIR
    payload["backend"] = "transformers"
    payload["max_tokens"] = MAX_TOKENS
    payload["partial"] = partial
    payload["segments_attempted"] = len(results)
    payload["segments_total"] = len(segments)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


for index, segment in enumerate(segments, start=1):
    if time.time() - started > SOFT_DEADLINE_S:
        print(f"soft deadline -- stopping with {len(results)}/{len(segments)}", flush=True)
        break
    result = run_segment(client, segment, config)
    results.append(result)
    print(f"[{index}/{len(segments)}] {segment.key:22s} n={len(segment):3d} "
          f"{result.outcome:14s} prefix={result.best_prefix:3d}/{len(segment):<3d} "
          f"att={result.attempts} {result.elapsed_s:7.1f}s", flush=True)
    _persist(partial=True)
    if result.source:
        SOURCES_DIR.mkdir(exist_ok=True)
        (SOURCES_DIR / f"coder_{result.segment_key.replace('/', '_')}.py").write_text(
            result.source, encoding="utf-8")

report = build_report(results)
print()
print(report.summary(), flush=True)
print()
print(per_game_table(results), flush=True)
_persist(partial=len(results) < len(segments))
print(f"\nwrote {RESULTS_PATH}", flush=True)
print(f"total wall clock: {time.time()-started:.1f}s", flush=True)
'''


def build() -> dict:
    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": src.splitlines(True)}

    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": HEADER.splitlines(True)},
            code(SETUP_CELL.replace("__DATASET__", DATASET)),
            code(BACKTEST_CELL),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build()
    out_dir = REPO_ROOT / "kaggle_submission_cwm_coder" / "notebook"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{KERNEL_SLUG}.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(notebook['cells'])} cells)")

    body = "".join("".join(c["source"]) for c in notebook["cells"])
    required = [
        ("transformers backend", '"LLM_BACKEND"] = "transformers"'),
        ("coder model resolved", "find_model_dir(\"qwen3-coder\")"),
        ("oracle control", "verify_oracle"),
        ("round-robin selection", "by_game"),
        ("incremental persist", "_persist(partial=True)"),
        ("real-shape preflight", "class WorldModel"),
        ("VRAM gate", "MIN_VRAM_GB"),
    ]
    missing = [n for n, probe in required if probe not in body]
    if missing:
        print("REFUSING TO SHIP -- missing: " + ", ".join(missing))
        return 1
    for n, _ in required:
        print(f"  OK  {n}")

    from _check_notebook_cell import check_notebook

    problems = check_notebook(path)
    if problems:
        print(f"REFUSING TO SHIP -- {len(problems)} use-before-definition problem(s):")
        for p in problems[:10]:
            print("  " + p)
        return 1
    print("use-before-definition check PASSED")

    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [f"{OWNER}/{DATASET}"],
        "kernel_sources": [],
        # KEPT even though the backtest needs no competition data: dropping
        # it downgraded the kernel from RTX PRO 6000 to a 15.6 GB T4, which
        # cannot hold a 30B model. See the VRAM assert above.
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [CODER_MODEL],
        "machine_shape": "NvidiaRtxPro6000",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'kernel-metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
