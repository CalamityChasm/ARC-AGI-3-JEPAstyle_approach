"""Build the Kaggle dataset + notebook that run the CodeWorldModel backtest.

The measurement needs the served model, which only exists on the sealed
Kaggle appliance (the local box is an RTX 2070 with 8 GB). So the segments
travel as a dataset and the backtest runs in a kernel that boots the same
NVFP4 vLLM stack our champion configuration uses.

The notebook reuses the anim/HUD notebook's setup cells **verbatim** up to
and including the solver setup commands -- that is what starts vLLM on
127.0.0.1:1234 -- and then, instead of playing games, runs the backtest.
Reusing those cells rather than reimplementing them is deliberate: the
serving stack is the part most likely to break and the part we least want
to vary.

Usage:
    python scripts/_build_cwm_backtest_kernel.py --artifacts <run>/artifacts \
        --stage <dir> [--max-segments N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc3_cwm.extract import extract_run  # noqa: E402
from arc3_cwm.render import MAX_STEPS, build_user_prompt, fit_to_budget  # noqa: E402
from arc3_cwm.serialize import dump_segments, load_segments  # noqa: E402

#: The champion configuration, and on `master` -- deliberately not the HUD
#: fork, whose notebook lives on an unmerged branch. This kernel must not
#: depend on a branch that may never land.
SOURCE_NOTEBOOK = (
    REPO_ROOT / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "arc3-duck-nvfp4-anim.ipynb"
)
#: Cells 0..SETUP_THROUGH are the serving stack, reused unchanged. Cell 9 is
#: the one that runs `setup_commands.json` and therefore starts vLLM.
SETUP_THROUGH = 9

DATASET_SLUG = "cwm-backtest"
DATASET_OWNER = "calamitychasm"
KERNEL_SLUG = "arc3-cwm-backtest"

#: 32768-token context shared between prompt and response. At ~3 chars per
#: token for digit-heavy text, 60k chars leaves comfortable room for a
#: 4096-token reply. Segments over budget are shrunk, not dropped.
PROMPT_CHAR_BUDGET = 60_000


BACKTEST_CELL = r'''
# ============================================================
# CodeWorldModel backtest -- replaces the game-playing benchmark.
# vLLM is already serving on 127.0.0.1:1234 (started by the setup
# commands in the cell above). Nothing below plays a game.
# ============================================================
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

def _find_input_dir(name):
    for root in ("/kaggle/input",):
        base = Path(root)
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_dir() and path.name == name:
                return path
    raise RuntimeError(f"dataset {name!r} not found under /kaggle/input")

DATA_DIR = _find_input_dir("__DATASET_SLUG__")
print("backtest data:", DATA_DIR, flush=True)
sys.path.insert(0, str(DATA_DIR))
os.environ["ARC3_CWM_ENGINE_DIR"] = str(DATA_DIR)

from arc3_cwm.determinism import census
from arc3_cwm.harness import BacktestConfig, run_segment
from arc3_cwm.oracle import verify_oracle
from arc3_cwm.report import build_report, per_game_table
from arc3_cwm.serialize import load_segments

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_ID = "Qwen/Qwen3.8-Flash-Next-NVFP4"
MAX_ATTEMPTS = int(os.environ.get("CWM_MAX_ATTEMPTS", "3"))
# Bounded pilot. A kernel that is still RUNNING cannot have its output
# pulled, so an unbounded run that overruns a deadline yields NOTHING
# however carefully it persists. 25 segments is ample to separate the
# pre-registered <10% / 10-40% / >40% bands.
MAX_SEGMENTS = int(os.environ.get("CWM_MAX_SEGMENTS", "25")) or None


class VLLMClient:
    """Minimal OpenAI-compatible client over stdlib urllib.

    Deliberately not the `openai` package: this kernel is offline and the
    duck bundle does not guarantee that dependency. One POST per call.
    """

    def __init__(self):
        self.field_counts = {}
        self.errors = 0

    def complete(self, system, user, max_tokens=4096):
        body = json.dumps({
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")
        request = urllib.request.Request(
            BASE_URL + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read().decode("utf-8"))

        message = payload["choices"][0]["message"]
        # THE GOTCHA THIS PROJECT ALREADY PAID FOR: this build's Qwen3
        # reasoning parser puts generated tokens in `reasoning`, not
        # `content`. A harness that reads only `content` reports a healthy
        # server as silent and produces a complete, plausible, entirely
        # empty results table. Three free GPU runs were burned on that.
        # Read content first, then fall back, and COUNT which field won so
        # the log says which one actually carried the answer.
        for field in ("content", "reasoning", "reasoning_content"):
            text = message.get(field) or ""
            if text.strip():
                self.field_counts[field] = self.field_counts.get(field, 0) + 1
                return text
        self.field_counts["empty"] = self.field_counts.get("empty", 0) + 1
        return ""


client = VLLMClient()

# ---- preflight: the server answers, and we know which field it uses ----
print("preflight: probing the server...", flush=True)
_probe = client.complete(
    "You are a terse assistant.",
    "Reply with exactly: READY",
    max_tokens=2048,
)
print(f"preflight reply ({len(_probe)} chars): {_probe.strip()[:200]!r}", flush=True)
print(f"preflight field counts: {client.field_counts}", flush=True)
assert _probe.strip(), (
    "server returned nothing in content/reasoning/reasoning_content -- "
    "do NOT trust any result from this run"
)
client.field_counts.clear()

# ---- the measurement ----
segments = load_segments(DATA_DIR / "cwm_segments.json.gz")
print(f"loaded {len(segments)} segments from "
      f"{len({s.game_id for s in segments})} games", flush=True)

det = census(segments)
print(det.summary(), flush=True)

oracle_ok = sum(1 for s in segments if verify_oracle(s)[0])
print(f"positive control: oracle replays {oracle_ok}/{len(segments)} segments", flush=True)

if MAX_SEGMENTS:
    # Round-robin across games rather than taking the file order, which is
    # alphabetical and would measure the first few games only. One segment
    # per game comes first, so a 25-cap covers all 25 games.
    #
    # This selects mostly LEVEL 1 segments, which are the easiest. That is
    # a deliberate upper bound: if the model cannot model level 1, it
    # certainly cannot model level 4, so a failure here is decisive while a
    # pass is optimistic. Stated in the write-up, not buried.
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

config = BacktestConfig(max_attempts=MAX_ATTEMPTS)
results = []
started = time.time()

RESULTS_PATH = Path("/kaggle/working/cwm_backtest_results.json")
SOURCES_DIR = Path("/kaggle/working/passing_models")

# Kaggle kills a GPU kernel at its wall-clock cap. Writing results only at
# the end would mean a timeout yields NOTHING -- hours of GPU for no
# number. So the file is rewritten after every segment, and the run stops
# itself cleanly with time to spare rather than being killed mid-write.
SOFT_DEADLINE_S = float(os.environ.get("CWM_SOFT_DEADLINE_S", str(2.5 * 3600)))


def _persist(partial):
    payload = build_report(results).as_dict()
    payload["determinism"] = det.as_dict()
    payload["oracle_passing_segments"] = oracle_ok
    payload["response_field_counts"] = client.field_counts
    payload["model_id"] = MODEL_ID
    payload["partial"] = partial
    payload["segments_attempted"] = len(results)
    payload["segments_total"] = len(segments)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


for index, segment in enumerate(segments, start=1):
    elapsed = time.time() - started
    if elapsed > SOFT_DEADLINE_S:
        print(f"soft deadline reached after {elapsed/3600:.2f}h -- stopping with "
              f"{len(results)}/{len(segments)} segments measured", flush=True)
        break

    result = run_segment(client, segment, config)
    results.append(result)
    print(f"[{index}/{len(segments)}] {segment.key:24s} n={len(segment):3d} "
          f"{result.outcome:16s} prefix={result.best_prefix:3d}/{len(segment):<3d} "
          f"attempts={result.attempts} {result.elapsed_s:7.1f}s", flush=True)

    # Persist after every segment, and save any passing source immediately.
    _persist(partial=True)
    if result.source:
        SOURCES_DIR.mkdir(exist_ok=True)
        (SOURCES_DIR / f"{result.segment_key.replace('/', '_')}.py").write_text(
            result.source, encoding="utf-8"
        )

report = build_report(results)
print()
print(report.summary(), flush=True)
print()
print(per_game_table(results), flush=True)
print()
print(f"response field counts: {client.field_counts}", flush=True)
print(f"total wall clock: {time.time() - started:.1f}s", flush=True)

_persist(partial=len(results) < len(segments))
print(f"wrote {RESULTS_PATH} "
      f"({len(results)}/{len(segments)} segments measured)", flush=True)
'''


def build_notebook(dataset_slug: str) -> dict:
    source = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = source["cells"][: SETUP_THROUGH + 1]

    header = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# arc3-cwm-backtest - can this model write a world model that replays?\n",
            "\n",
            "Setup cells below are reused **verbatim** from `arc3-duck-nvfp4-hud` ",
            "(itself a reproduction of other people's work -- see that notebook's ",
            "`THIRD_PARTY_NOTICE.md`). They exist here only to boot the same NVFP4 ",
            "vLLM serving stack our champion configuration uses.\n",
            "\n",
            "**No games are played.** The final cell replays recorded play from the ",
            "2026-09-21 anim run against world models this model writes, and reports ",
            "a counted pass rate against a null floor and a proven ceiling.\n",
        ],
    }

    backtest = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": BACKTEST_CELL.replace("__DATASET_SLUG__", dataset_slug).splitlines(True),
    }

    source["cells"] = [header] + cells[1:] + [backtest]
    return source


def stage(artifacts: Path, stage_dir: Path, max_segments: int | None) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)

    segments, stats = extract_run(artifacts, min_transitions=4)
    windowed = [s.window(MAX_STEPS) for s in segments]
    windowed = [s for s in windowed if len(s) >= 4]
    fitted = [fit_to_budget(s, PROMPT_CHAR_BUDGET) for s in windowed]
    if max_segments:
        fitted = fitted[:max_segments]

    shrunk = sum(1 for a, b in zip(windowed, fitted) if len(b) < len(a))
    largest = max(len(build_user_prompt(s)) for s in fitted)
    print(f"extraction: {json.dumps(stats.as_dict())}")
    print(f"segments: {len(fitted)} ({shrunk} shrunk to fit the prompt budget)")
    print(f"largest prompt: {largest:,} chars (~{largest // 3:,} tokens)")

    out = dump_segments(fitted, stage_dir / "cwm_segments.json.gz", source=str(artifacts))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    # Round-trip check before shipping: a lossy export would move the very
    # boards the model is asked about, invisibly.
    restored = load_segments(out)
    assert len(restored) == len(fitted)
    for a, b in zip(fitted, restored):
        assert len(a) == len(b) and a.game_id == b.game_id
        for ta, tb in zip(a.transitions, b.transitions):
            assert ta.frame_before == tb.frame_before
            assert ta.frame_after == tb.frame_after
            assert (ta.action.name, ta.action.x, ta.action.y) == (
                tb.action.name, tb.action.x, tb.action.y
            )
    print("round-trip verified exact")

    shutil.copytree(REPO_ROOT / "arc3_cwm", stage_dir / "arc3_cwm", dirs_exist_ok=True)
    shutil.copytree(
        REPO_ROOT / "kaggle_submission_llm_world_engine" / "dataset_stage" / "llm_engine",
        stage_dir / "llm_engine",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.rmtree(stage_dir / "arc3_cwm" / "__pycache__", ignore_errors=True)

    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "cwm-backtest",
                "id": f"{DATASET_OWNER}/{DATASET_SLUG}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"staged dataset at {stage_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--notebook-out", type=Path, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    args = parser.parse_args(argv)

    stage(args.artifacts, args.stage, args.max_segments)

    notebook = build_notebook(DATASET_SLUG)
    out_dir = args.notebook_out or (REPO_ROOT / "kaggle_submission_cwm_backtest" / "notebook")
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = out_dir / f"{KERNEL_SLUG}.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {notebook_path} ({len(notebook['cells'])} cells)")

    source_meta = json.loads(
        (SOURCE_NOTEBOOK.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    source_meta["id"] = f"{DATASET_OWNER}/{KERNEL_SLUG}"
    source_meta["title"] = KERNEL_SLUG
    source_meta["code_file"] = f"{KERNEL_SLUG}.ipynb"
    sources = list(source_meta.get("dataset_sources", []))
    if f"{DATASET_OWNER}/{DATASET_SLUG}" not in sources:
        sources.append(f"{DATASET_OWNER}/{DATASET_SLUG}")
    source_meta["dataset_sources"] = sources
    # `competition_sources` is KEPT even though this kernel never submits:
    # the setup cells install `arc-agi` from the competition's offline
    # wheelhouse, so dropping the mount breaks the run before vLLM starts.
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(source_meta, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'kernel-metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
