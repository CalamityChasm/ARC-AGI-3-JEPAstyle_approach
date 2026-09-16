"""Build the ``LOCAL_ANALYZER_YIELD_SECONDS`` 60 -> 180 variant of the NVFP4 baseline.

The one-variable test
---------------------
``experiments/stage7_sota_research.md`` §1.3 measured that **46% of every LLM
call the nvfp4 baseline pays for takes no game action**, and that every single
one of those turns stops with the same reason::

    step_executed: False
    message: Yielded control to solver: turn_time_budget.

The cause is a constant, verified in the bundle we actually run
(``keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1``)::

    serving_setup.py:2940                       "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    ARC3-Inference/inference/agent/tool_agent.py:143
        _LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
    ARC3-Inference/inference/agent/tool_agent.py:934
        self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(...)
    ARC3-Inference/inference/agent/tool_agent.py:1777
        if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:
            return "turn_time_budget"

against a median ~153 s LLM round-trip. The analyzer's tool loop
(``tool_agent.py:1782-1978``) checks that predicate before each model call and
again after every non-acting tool dispatch, so at 60 s it gets **exactly one**
call per turn: if that call was an inspect-only ``python`` call, the turn dies
having taken no action. (``LOCAL_ANALYZER_TOOL_STEPS`` is persisted as ``"0"``
-> ``self._tool_steps is None`` -> the step count is *unbounded*; time is the
only gate.) At 180 s a second call fits (153 < 180) and a third does not
(306 > 180), so the model can inspect **and** act inside one turn with the tool
result already in context.

This script changes that constant and **nothing else**. It asserts every other
cell of the baseline notebook is byte-identical, so the free run it produces is
a genuine single-variable test against the baseline's 10.69 / 3,633 actions.

Where the patch goes, and why there
-----------------------------------
At the **tail of the setup cell** (index 10) -- after the bundle's
``serving_setup.py`` has persisted the analyzer environment, and *before* the
first ``inference`` import. That matters: ``tool_agent.py:143`` reads the env
at **module import time**, and the first import happens one cell later, when
cell 12 unpickles ``benchmark_initial.pkl`` (whose ``inference.framework.solver``
does ``from inference.agent.tool_agent import ToolAgent`` at line 37). Patching
the env before that point makes the module global bind 180.0 naturally, rather
than relying on a post-hoc reassignment.

``ToolAgent.__init__`` re-reads the module global per construction
(``tool_agent.py:934``) and the solver builds one agent per game in-process on a
``ThreadPoolExecutor``, so a single patch here covers all 25 games.

The guard pattern (assert the persisted value is still ``"60"``, assert nothing
is imported yet, assert the bound float is 180.0 afterwards) is taken from
``sahasawatt/thui-rs-v0`` cell 9, as ``stage7_sota_research.md`` §6.2 recommends
-- minus their ``LOCAL_ANALYZER_SEED`` override, which would be a second
variable.

Usage
-----
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_yield.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kaggle_submission_duck_nvfp4" / "notebook" / "duck-qwen3-8-anim-base.ipynb"
OUT_DIR = REPO / "kaggle_submission_duck_nvfp4_yield" / "notebook"
OUT_NB = OUT_DIR / "duck-qwen3-8-yield180.ipynb"

# md5 of the vendored baseline chassis. Asserted so this build cannot silently
# measure a different notebook than the one that scored 10.69 / 3,633.
EXPECTED_SRC_MD5 = "57ffcd516bd26d10c6314df80d384eeb"

KERNEL_ID = "calamitychasm/arc3-duck-nvfp4-yield180"
KERNEL_TITLE = "arc3-duck-nvfp4-yield180"

SETUP_CELL_INDEX = 10
# Last line of the baseline setup cell -- asserted, so an upstream edit to that
# cell fails the build instead of appending the patch to something else.
SETUP_CELL_TAIL = "        sys.path.insert(0, entry)"

YIELD_PATCH = '''

# ---------------------------------------------------------------------------
# [calamitychasm] THE ONE VARIABLE UNDER TEST: LOCAL_ANALYZER_YIELD_SECONDS 60 -> 180.
#
# Measured on this exact chassis (experiments/stage7_sota_research.md SS1.3):
# 46.0% of the baseline's 1,287 analyzer turns executed no game action, and every
# one of them stopped with "Yielded control to solver: turn_time_budget".
#
# Why: tool_agent.py:1777 breaks the tool loop once (now - turn_started_at) >= 60 s,
# against a median ~153 s round-trip, so the loop gets exactly one model call per
# turn -- an inspect-only call wastes the whole turn. LOCAL_ANALYZER_TOOL_STEPS is
# persisted as "0", so self._tool_steps is None and the step count is unbounded:
# time is the only gate. At 180 s a second call fits (153 < 180) and a third does
# not (306 > 180), so inspect->act can complete inside one turn.
#
# Placed here, at the tail of the setup cell, because tool_agent.py:143 reads the
# env at IMPORT time and the first import is the next cell's unpickle of
# benchmark_initial.pkl (inference.framework.solver:37 imports ToolAgent).
# Guard pattern from sahasawatt/thui-rs-v0 cell 9, minus their seed override.
_YIELD_KNOB = {"LOCAL_ANALYZER_YIELD_SECONDS": "180"}
_persisted = json.loads(SETUP_ENV_PATH.read_text())

# Fail loudly if the bundle we are overriding is not the one this was derived from.
assert _persisted.get("LOCAL_ANALYZER_YIELD_SECONDS") == "60", (
    "serving_setup.py no longer persists yield 60 -- re-derive the override",
    _persisted.get("LOCAL_ANALYZER_YIELD_SECONDS"),
)
# Everything else the analyzer reads must be untouched, or this is not a
# single-variable test any more.
_EXPECTED_UNCHANGED = {
    "LOCAL_ANALYZER_MODEL_ID": "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}
for _k, _v in _EXPECTED_UNCHANGED.items():
    assert _persisted.get(_k) == _v, (_k, _persisted.get(_k), _v)
# No LOCAL_ANALYZER_SEED override: the baseline does not set one (tool_agent
# default -1), and adding one would be a second variable.
assert "LOCAL_ANALYZER_SEED" not in _persisted, _persisted.get("LOCAL_ANALYZER_SEED")

_persisted.update(_YIELD_KNOB)
SETUP_ENV_PATH.write_text(json.dumps(_persisted, indent=2, sort_keys=True) + "\\n")
os.environ.update(_YIELD_KNOB)

# The knob must land before the solver binds it at import time.
assert "inference" not in sys.modules and "taaf" not in sys.modules, (
    "solver imported before the yield override",
    sorted(m for m in sys.modules if m.split(".")[0] in {"inference", "taaf"}),
)
import inference.agent.tool_agent as _tool_agent

assert float(_tool_agent._LOCAL_ANALYZER_YIELD_SECONDS) == 180.0, _tool_agent._LOCAL_ANALYZER_YIELD_SECONDS
# Unbounded tool steps: confirms time really is the only gate on the loop.
assert int(_tool_agent._LOCAL_ANALYZER_TOOL_STEPS) == 0, _tool_agent._LOCAL_ANALYZER_TOOL_STEPS
print(
    f"YIELD_PATCH ok yield_seconds={_tool_agent._LOCAL_ANALYZER_YIELD_SECONDS} "
    f"tool_steps={_tool_agent._LOCAL_ANALYZER_TOOL_STEPS} "
    f"seed={_tool_agent._LOCAL_ANALYZER_SEED} "
    f"thinking={_tool_agent._LOCAL_ANALYZER_ENABLE_THINKING} "
    f"solver={Path(_tool_agent.__file__).parent}",
    flush=True,
)
'''

HEADER_MD = """\
# arc3-duck-nvfp4-yield180 - one knob changed against our NVFP4 baseline

This is `calamitychasm/arc3-duck-nvfp4-baseline` with **exactly one** difference:
`LOCAL_ANALYZER_YIELD_SECONDS` is raised from the bundle's persisted `60` to
`180`, at the tail of the setup cell (before the solver import binds it). Every
other code cell is byte-identical to that baseline; the build script asserts it.

**Why.** On the baseline run, 46.0% of the 1,287 analyzer turns executed no game
action, and every one stopped with `Yielded control to solver:
turn_time_budget`. `tool_agent.py:1777` ends the tool loop once the turn has run
60 s, against a median ~153 s LLM round-trip -- so the loop gets exactly one
model call per turn, and an inspect-only call wastes the turn. At 180 s a second
call fits, so inspect->act can complete inside one turn.

**What to read off the log.** `YIELD_PATCH ok yield_seconds=180.0`, then the
public-25 mean and total actions against the baseline's **10.69 / 3,633**, and
the share of `[ANALYZER STATUS]` blocks carrying `step_executed: False`
(baseline 46.0%).

**Attribution.** Unchanged from the baseline: the Duck solver is Tufa Labs'
(Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar,
Stefano Viel); the NVFP4 serving stack is Keith Tyser's; the weights are
RadixArk's NVFP4 quantisation of Qwen/Qwen3.8-Flash-Next. The guard pattern
around the override follows `sahasawatt/thui-rs-v0` (Thuitanium / Knowless
Crew). No score quoted by any of them is ours. Full licence text in
`THIRD_PARTY_NOTICE.md`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    raw = SRC.read_bytes()
    got = hashlib.md5(raw).hexdigest()
    if got != EXPECTED_SRC_MD5:
        raise SystemExit(f"baseline notebook md5 {got} != expected {EXPECTED_SRC_MD5}")

    nb = json.loads(raw.decode("utf-8"))
    cells = nb["cells"]

    setup = cells[SETUP_CELL_INDEX]
    if setup["cell_type"] != "code":
        raise SystemExit(f"cell {SETUP_CELL_INDEX} is {setup['cell_type']}, expected code")
    src = "".join(setup["source"])
    if not src.endswith(SETUP_CELL_TAIL):
        raise SystemExit(
            f"cell {SETUP_CELL_INDEX} does not end with the expected line; "
            f"tail is {src[-120:]!r}"
        )
    setup["source"] = (src + YIELD_PATCH).splitlines(keepends=True)

    # Replace only the fork-header markdown cell (index 1), leaving Tufa's own
    # header (index 2) as-is, exactly as the baseline does.
    cells[1]["source"] = HEADER_MD.splitlines(keepends=True)

    # Every other cell must be byte-identical to the baseline.
    base = json.loads(raw.decode("utf-8"))["cells"]
    changed = [
        i
        for i, (a, b) in enumerate(zip(base, cells))
        if "".join(a["source"]) != "".join(b["source"])
    ]
    if changed != [1, SETUP_CELL_INDEX]:
        raise SystemExit(f"expected cells [1, {SETUP_CELL_INDEX}] to differ, got {changed}")
    if len(base) != len(cells):
        raise SystemExit("cell count changed")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / OUT_NB.name
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = json.loads(
        (SRC.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_TITLE
    meta["code_file"] = out_nb.name
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    notice = (SRC.parent / "THIRD_PARTY_NOTICE.md").read_text(encoding="utf-8")
    (out_dir / "THIRD_PARTY_NOTICE.md").write_text(notice, encoding="utf-8")

    print(f"wrote {out_nb} ({len(nb['cells'])} cells; changed {changed})")
    print(f"      {out_dir / 'kernel-metadata.json'} id={KERNEL_ID}")


if __name__ == "__main__":
    main()
