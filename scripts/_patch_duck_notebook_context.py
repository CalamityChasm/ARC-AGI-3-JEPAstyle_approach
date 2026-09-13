"""Patch the NVFP4 Duck notebook to shrink the HARNESS's context budget.

One variable, one place. Everything else in the notebook is byte-identical to
the 10.69 public-25 baseline (`calamitychasm/arc3-duck-nvfp4-baseline` v2), and
this script asserts that before it touches anything.

WHAT IS BEING CHANGED, AND WHY IT IS NOT `--max-model-len`
----------------------------------------------------------
`serving_setup.py:129` sets `ANALYZER_CONTEXT = 32_768` and uses it for two
unrelated things:

  * `serving_setup.py:2323`  -> vLLM's `--max-model-len 32768`
  * `serving_setup.py:2935`  -> the env var `LOCAL_ANALYZER_CONTEXT_WINDOW`,
                                which the harness reads at
                                `tool_agent.py:138` and turns into
                                `self._context_budget_tokens` at
                                `tool_agent.py:945-948`
                                (= window - 512 reply reserve - 512 safety).

Only the *second* one controls how many tokens the harness packs into a
request. `tool_agent.py:1686`'s trim loop drops the oldest message block until
its own estimate fits that budget, so the prompt is always pushed right up
against it (measured: 29,404 of 31,744 estimated tokens = 92.6% fill, over the
25 final-request snapshots of the baseline run).

Lowering vLLM's `--max-model-len` is explicitly NOT the lever: vLLM allocates
KV on demand, so the cap is not what makes sequences long, and dropping it
below what the harness sends gets requests rejected rather than truncated.
This patch therefore leaves `--max-model-len` at 32768 and moves only the
harness-side budget, by overriding the module constant in the notebook's own
designated customization hook (cell 13/14) AFTER `serving_setup.py` has run and
written its value into the process environment.

Usage:
    python scripts/_patch_duck_notebook_context.py --window 16384 \
        --kernel-id calamitychasm/arc3-duck-nvfp4-ctx16k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _duck_notebook_patch import build

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kaggle_submission_duck_nvfp4" / "notebook_baseline" / "duck-qwen3-8-anim-base.ipynb"
NB = REPO / "kaggle_submission_duck_nvfp4" / "notebook" / "duck-qwen3-8-anim-base.ipynb"
META = REPO / "kaggle_submission_duck_nvfp4" / "notebook" / "kernel-metadata.json"

# md5 of the verified 10.69 public-25 baseline notebook.
BASELINE_MD5 = "57ffcd516bd26d10c6314df80d384eeb"

# The customization-hook cell the notebook itself designates ("## 6.
# Customization hook ... the safe place for one-off experiments").
HOOK_CELL = 14
HOOK_ANCHOR = "# Exact public-25 and competition settings."

MARKER = "CONTEXT_BUDGET_OVERRIDE"

PATCH_TEMPLATE = '''

# [calamitychasm] stage7-context-budget -- THE ONE CHANGE IN THIS NOTEBOOK.
# Shrink the harness's own context budget so each analyzer request holds fewer
# KV tokens. This is NOT vLLM's --max-model-len (serving_setup.py:2323 still
# passes 32768); vLLM allocates KV on demand, so the model-length cap is not
# what makes a sequence long, and lowering it below what the harness sends gets
# requests rejected instead of truncated.
#
# The real budget is tool_agent.py:945-948
#     _context_budget_tokens = LOCAL_ANALYZER_CONTEXT_WINDOW - 512 - 512
# and tool_agent.py:1686's trim loop refills history right up to it every turn.
# serving_setup.py:2935 writes the window into the environment during cell 10,
# and tool_agent.py:138 reads it at import time, so the override has to happen
# here -- after setup, before the first ToolAgent is built (solver.py:1189
# constructs one per game, at run time).
CONTEXT_WINDOW_OVERRIDE = {window}

import sys as _cb_sys

import inference.agent.tool_agent as _cb_tool_agent
import inference.framework.solver as _cb_solver

# Guard: the module the solver will actually build ToolAgents from must be the
# same module object being patched here.
if _cb_sys.modules[_cb_solver.ToolAgent.__module__] is not _cb_tool_agent:
    raise RuntimeError("solver.ToolAgent came from a different module object")

_cb_prev_window = _cb_tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW
if _cb_prev_window != 32768:
    raise RuntimeError(
        f"expected the 32768 baseline context window, got {{_cb_prev_window!r}}"
    )

# solver.py:1186 short-circuits to `self.analyzer_factory` if one is set, which
# would bypass the ToolAgent path entirely. Verify it is not.
if getattr(bm.solver, "analyzer_factory", None) is not None:
    raise RuntimeError("solver has an analyzer_factory; the ToolAgent path is bypassed")

os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = str(CONTEXT_WINDOW_OVERRIDE)
_cb_tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW = CONTEXT_WINDOW_OVERRIDE

# Direct proof the override reaches a real agent, not just a module global:
# build a throwaway ToolAgent and read its derived budget. Construction is pure
# (attributes + string building, no I/O), so this is free.
_cb_expected = max(1024, CONTEXT_WINDOW_OVERRIDE - 512 - 512)
_cb_probe = _cb_tool_agent.ToolAgent(
    model=os.environ.get("INFERENCE_ANALYZER_MODEL")
    or os.environ.get("LOCAL_ANALYZER_MODEL_ID")
    or "local",
)
if _cb_probe._context_budget_tokens != _cb_expected:
    raise RuntimeError(
        f"context budget did not take: {{_cb_probe._context_budget_tokens}} "
        f"!= {{_cb_expected}}"
    )
del _cb_probe

print(
    f"{marker} window={{_cb_prev_window}}->{{CONTEXT_WINDOW_OVERRIDE}} "
    f"budget_tokens={{_cb_expected}} vllm_max_model_len=32768 (unchanged) "
    f"multimodal={{os.environ.get('MULTIMODAL_CONTEXT')!r}}"
    f"@{{os.environ.get('MULTIMODAL_UPSCALE')!r}}",
    flush=True,
)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--kernel-id", required=True)
    args = ap.parse_args()
    build(
        base_nb=BASE_NB,
        out_nb=NB,
        meta=META,
        baseline_md5=BASELINE_MD5,
        hook_cell=HOOK_CELL,
        hook_anchor=HOOK_ANCHOR,
        patch=PATCH_TEMPLATE.format(window=args.window, marker=MARKER),
        kernel_id=args.kernel_id,
    )
    print(f"patched cell {HOOK_CELL}: harness context window -> {args.window}")


if __name__ == "__main__":
    main()
