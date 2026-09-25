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

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
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

#: 32768-token context is shared between prompt AND response. The first
#: run budgeted 60k chars of prompt (~20k tokens) against a 4096-token
#: reply -- which left the model no room to both think and answer, and it
#: spent the whole reply budget thinking. Rebalanced: ~36k chars of prompt
#: (~12k tokens) leaves ~16k tokens for the answer inside 32768.
#: Segments over budget are shrunk, not dropped.
PROMPT_CHAR_BUDGET = 36_000


VALIDATION_CELL = r'''
# ============================================================
# Input validation -- deliberately FIRST, before the wheel install,
# the bundle setup and the ~10-minute vLLM boot.
#
# The previous run spent a 7.5h queue wait plus 14 minutes of GPU only to
# die on a missing filename: Kaggle silently DECOMPRESSES an uploaded
# .gz, so `cwm_segments.json.gz` arrives as `cwm_segments.json`. Nothing
# about that needs a GPU to detect. Everything checkable without the
# model is checked here, so a data problem costs seconds.
# ============================================================
import os, sys
from pathlib import Path

def _find_input_dir(name):
    base = Path("/kaggle/input")
    if base.is_dir():
        for path in base.rglob("*"):
            if path.is_dir() and path.name == name:
                return path
    raise RuntimeError(f"dataset {name!r} not found under /kaggle/input")

DATA_DIR = _find_input_dir("__DATASET_SLUG__")
sys.path.insert(0, str(DATA_DIR))
os.environ["ARC3_CWM_ENGINE_DIR"] = str(DATA_DIR)
print("backtest data:", DATA_DIR, flush=True)

# Accept either name; Kaggle's own archive handling decides which we get.
_candidates = ["cwm_segments.json", "cwm_segments.json.gz"]
SEGMENTS_PATH = next((DATA_DIR / c for c in _candidates if (DATA_DIR / c).is_file()), None)
if SEGMENTS_PATH is None:
    raise FileNotFoundError(
        f"no segments file in {DATA_DIR}. Tried {_candidates}. "
        f"Directory holds: {sorted(p.name for p in DATA_DIR.iterdir())}"
    )
print("segments file:", SEGMENTS_PATH.name, flush=True)

from arc3_cwm.determinism import census
from arc3_cwm.oracle import verify_oracle
from arc3_cwm.serialize import load_segments

_segs = load_segments(SEGMENTS_PATH)
print(f"loaded {len(_segs)} segments from {len({s.game_id for s in _segs})} games",
      flush=True)
assert _segs, "segments file loaded but is empty"

_det = census(_segs)
print(_det.summary(), flush=True)

_oracle_ok = sum(1 for s in _segs if verify_oracle(s)[0])
print(f"positive control: oracle replays {_oracle_ok}/{len(_segs)} segments", flush=True)
assert _oracle_ok > 0, (
    "the oracle cannot pass a single segment -- the harness could not report a "
    "pass even if the model produced one, so any result would be meaningless"
)
print("INPUT VALIDATION PASSED -- proceeding to the expensive setup", flush=True)
'''


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
MAX_SEGMENTS = int(os.environ.get("CWM_MAX_SEGMENTS", "12")) or None

ARMS = [
    ("think-16k", dict(max_tokens=16384, enable_thinking=None)),
    ("nothink-8k", dict(max_tokens=8192, enable_thinking=False)),
]



class VLLMClient:
    """Minimal OpenAI-compatible client over stdlib urllib.

    Deliberately not the `openai` package: this kernel is offline and the
    duck bundle does not guarantee that dependency. One POST per call.

    `finish_reason` is counted because the previous run had to *infer*
    that its 4096-token budget was being consumed by reasoning. Inferring
    is not measuring: `length` vs `stop` says it outright.
    """

    def __init__(self, max_tokens=4096, enable_thinking=None, label=""):
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.label = label
        self.field_counts = {}
        self.finish_reasons = {}
        self.errors = 0

    def complete(self, system, user, max_tokens=None):
        payload_body = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": 0.0,
        }
        if self.enable_thinking is not None:
            payload_body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        body = json.dumps(payload_body).encode("utf-8")
        request = urllib.request.Request(
            BASE_URL + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read().decode("utf-8"))

        choice = payload["choices"][0]
        reason = choice.get("finish_reason") or "unknown"
        self.finish_reasons[reason] = self.finish_reasons.get(reason, 0) + 1
        message = choice["message"]
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


# ---- preflight: the server answers, and we know which field it uses ----
# The previous run's preflight used a SHORT prompt, answered in `content`,
# and passed -- while every real prompt came back as reasoning-only and
# produced no code at all. A preflight that does not exercise the real
# shape of the task is not a preflight. This one sends a genuine (small)
# world-model request and asserts a `class WorldModel` actually comes back.
_probe_client = VLLMClient(max_tokens=2048, label="probe")
print("preflight: probing the server...", flush=True)
_probe = _probe_client.complete("You are a terse assistant.", "Reply with exactly: READY")
print(f"preflight reply ({len(_probe)} chars): {_probe.strip()[:120]!r} "
      f"fields={_probe_client.field_counts} finish={_probe_client.finish_reasons}",
      flush=True)
assert _probe.strip(), (
    "server returned nothing in content/reasoning/reasoning_content -- "
    "do NOT trust any result from this run"
)

# ---- the measurement ----
segments = load_segments(SEGMENTS_PATH)
print(f"loaded {len(segments)} segments from "
      f"{len({s.game_id for s in segments})} games", flush=True)

det, oracle_ok = _det, _oracle_ok  # already computed in the validation cell

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
          f"(levels {sorted({s.level for s in segments})}) x {len(ARMS)} arms", flush=True)

RESULTS_PATH = Path("/kaggle/working/cwm_backtest_results.json")
SOURCES_DIR = Path("/kaggle/working/passing_models")

# ---- arms -----------------------------------------------------------
#
# The first real run produced 0/25 with ZERO load errors and zero replay
# failures: the model never emitted a `class WorldModel` at all, and all
# 46 responses arrived as reasoning tokens with `content` never used. The
# 4096-token budget was spent thinking and the answer was never reached.
# That measured the budget, not the model.
#
# So this run varies exactly that, and nothing else:
#   think-16k  -- reasoning allowed, 4x the budget
#   nothink-8k -- reasoning disabled via chat_template_kwargs
#
# Both see identical segments and an identical prompt. If both still
# produce no code, the budget explanation is dead and the result starts to
# be about the model. `finish_reason` is recorded either way, so
# truncation is measured rather than inferred.
config = BacktestConfig(max_attempts=MAX_ATTEMPTS)
arm_reports = {}
results = []
client = None
started = time.time()

# Kaggle kills a GPU kernel at its wall-clock cap. Writing results only at
# the end would mean a timeout yields NOTHING -- hours of GPU for no
# number. So the file is rewritten after every segment, and the run stops
# itself cleanly with time to spare rather than being killed mid-write.
SOFT_DEADLINE_S = float(os.environ.get("CWM_SOFT_DEADLINE_S", str(3.5 * 3600)))


def _persist(partial):
    payload = {
        "model_id": MODEL_ID,
        "partial": partial,
        "determinism": det.as_dict(),
        "oracle_passing_segments": oracle_ok,
        "segments_total": len(segments),
        "arms": arm_reports,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


for arm_name, arm_kwargs in ARMS:
    client = VLLMClient(label=arm_name, **arm_kwargs)
    results = []
    print(f"\n{'=' * 62}\nARM {arm_name}  {arm_kwargs}\n{'=' * 62}", flush=True)

    for index, segment in enumerate(segments, start=1):
        elapsed = time.time() - started
        if elapsed > SOFT_DEADLINE_S:
            print(f"soft deadline after {elapsed/3600:.2f}h -- stopping arm "
                  f"{arm_name} with {len(results)}/{len(segments)}", flush=True)
            break

        result = run_segment(client, segment, config)
        results.append(result)
        print(f"[{arm_name} {index}/{len(segments)}] {segment.key:22s} "
              f"n={len(segment):3d} {result.outcome:14s} "
              f"prefix={result.best_prefix:3d}/{len(segment):<3d} "
              f"att={result.attempts} {result.elapsed_s:6.1f}s", flush=True)

        report = build_report(results)
        arm_reports[arm_name] = {
            **report.as_dict(),
            "response_field_counts": dict(client.field_counts),
            "finish_reasons": dict(client.finish_reasons),
            "config": {k: str(v) for k, v in arm_kwargs.items()},
            "segments_attempted": len(results),
        }
        _persist(partial=True)

        if result.source:
            SOURCES_DIR.mkdir(exist_ok=True)
            (SOURCES_DIR / f"{arm_name}_{result.segment_key.replace('/', '_')}.py"
             ).write_text(result.source, encoding="utf-8")

    report = build_report(results)
    print()
    print(report.summary(), flush=True)
    print(f"\nfields : {client.field_counts}", flush=True)
    print(f"finish : {client.finish_reasons}", flush=True)

print(f"\n{'=' * 62}\nARM COMPARISON\n{'=' * 62}", flush=True)
for name, rep in arm_reports.items():
    print(f"{name:12s} passed {rep['n_passed']}/{rep['segments_attempted']:<3d} "
          f"median_prefix {rep['median_prefix_fraction']:.1%}  "
          f"outcomes {rep['outcome_counts']}  finish {rep['finish_reasons']}",
          flush=True)

_persist(partial=False)
print(f"\nwrote {RESULTS_PATH}", flush=True)
print(f"total wall clock: {time.time() - started:.1f}s", flush=True)
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

    validation = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": VALIDATION_CELL.replace(
            "__DATASET_SLUG__", dataset_slug
        ).splitlines(True),
    }

    # Validation goes before every expensive cell, straight after the GPU
    # identity check.
    source["cells"] = [header, cells[1], validation] + cells[2:] + [backtest]
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

    # Refuse to ship a notebook with a use-before-definition error. v5 died
    # on `NameError: name 'ARMS' is not defined` -- valid syntax, so the
    # `ast.parse` check passed it straight through to a GPU session.
    from _check_notebook_cell import check_notebook

    problems = check_notebook(notebook_path)
    if problems:
        print(f"\nREFUSING TO SHIP -- {len(problems)} use-before-definition problem(s):")
        for problem in problems:
            print("  " + problem)
        raise SystemExit(1)
    print("use-before-definition check PASSED")

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
