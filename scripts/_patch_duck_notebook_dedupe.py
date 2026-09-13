"""Patch the NVFP4 Duck notebook to stop re-sending duplicated history content.

CANDIDATE B. Independent of the context-window cut in
`_patch_duck_notebook_context.py`; both are generated from the same pristine
baseline notebook by `_duck_notebook_patch.build`, so only ever one of them is
present in a given build.

WHAT IT REMOVES, AND WHY THAT IS INFORMATION-PRESERVING
--------------------------------------------------------
Measured on the 25 final-request snapshots of the 10.69 baseline run
(`scripts/analyze_context_budget.py`):

  * `tool_agent.py:_build_user_prompt` rebuilds the same ~24-line instruction
    block on every turn. It costs **621 real tokens per user message** and the
    baseline retains **8.6** of them, so **4,721 real tokens (23% of the whole
    prompt)** are byte-identical copies of text the model is also reading in
    the current turn's user message.
  * `tool_agent.py:1148 _build_user_message` attaches a base64 PNG of the grid
    to every user message, and they are all retained. The stale ones cost
    **502 real vision tokens** -- but, because `_estimate_tokens` (chars/3 over
    `json.dumps`) sees the whole `data:image/png;base64,...` string, they cost
    **3,998 tokens of the trim budget**, an 8.0x overcount.

The rule applied to every user message except the newest is deliberately
conservative and self-verifying: **drop a line only if it appears verbatim in
the newest user message**, i.e. only if the model is still reading it. Lines
unique to that historical turn -- the executed-actions line, the step/level
line, that turn's world-model snapshot -- are all kept. Stale images are
dropped outright; the grids remain reachable through the python tool
(`previous_frame`, `history[*].frame.ascii/.segmentation`), which the system
prompt already directs the model to use for diffs.

WHERE IT HOOKS
--------------
`ToolAgent._chat_completion` (defined at `tool_agent.py:1282`; its single call
site is line 1822) -- the send path. Deliberately NOT
`_trim_messages_for_context` and NOT `_persistent_history_messages`: leaving
the stored history fat means the trim loop keeps making the same retention
decisions on the same estimates, so the number of retained turns is unchanged
and the only measured difference is the size of the wire payload. That keeps
this a clean single-variable throughput test rather than a simultaneous change
to how much history the model sees.

Projected: real prompt 20,126 -> ~14,904, tokens/request 21,559 -> ~16,337,
KV residency 4.88 -> 6.44 (**1.32x**), retained turns unchanged at ~8.6.

Usage:
    python scripts/_patch_duck_notebook_dedupe.py \
        --kernel-id calamitychasm/arc3-duck-nvfp4-dedupe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _duck_notebook_patch import build

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "kaggle_submission_duck_nvfp4" / "notebook_baseline" / "duck-qwen3-8-anim-base.ipynb"
NB = REPO / "kaggle_submission_duck_nvfp4" / "notebook" / "duck-qwen3-8-anim-base.ipynb"
META = REPO / "kaggle_submission_duck_nvfp4" / "notebook" / "kernel-metadata.json"

BASELINE_MD5 = "57ffcd516bd26d10c6314df80d384eeb"
HOOK_CELL = 14
HOOK_ANCHOR = "# Exact public-25 and competition settings."
MARKER = "HISTORY_DEDUPE"

PATCH = r'''

# [calamitychasm] stage7-context-budget -- THE ONE CHANGE IN THIS NOTEBOOK.
# Stop re-sending content the model is already reading in the current turn.
#
# Measured on the baseline run (scripts/analyze_context_budget.py, 25 games):
#   - the per-turn instruction block is rebuilt verbatim every turn and costs
#     621 real tokens x 8.6 retained messages; 4,721 of those are duplicates.
#   - every retained user message carries a base64 PNG of its grid. The stale
#     ones cost 502 real vision tokens, but 3,998 tokens of the harness's own
#     chars/3 trim budget, because _estimate_tokens sees the base64 string.
#
# The hook is ToolAgent._chat_completion -- the send path -- and NOT
# _trim_messages_for_context / _persistent_history_messages, so the stored
# history and every retention decision stay exactly as in the baseline. The
# only thing that changes is the size of the payload on the wire, which keeps
# this a single-variable throughput test.
#
# Drop rule for a historical user message: remove a line only if it appears
# verbatim in the NEWEST user message (so the model still reads it this turn).
# Lines unique to that turn -- executed actions, step/level, that turn's world
# model -- survive untouched.
import inference.agent.tool_agent as _dd_tool_agent

_DD_STATS = {"calls": 0, "chars_before": 0, "chars_after": 0}


def _dd_text_of(message):
    content = message.get("content")
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


def _dd_lean(messages):
    user_idx = [i for i, m in enumerate(messages) if str(m.get("role", "")) == "user"]
    if len(user_idx) < 2:
        return messages
    newest = set(_dd_text_of(messages[user_idx[-1]]).split("\n"))
    out = []
    for i, message in enumerate(messages):
        if str(message.get("role", "")) != "user" or i == user_idx[-1]:
            out.append(message)
            continue
        text = _dd_text_of(message)
        kept = [ln for ln in text.split("\n") if ln not in newest]
        # Never empty a message entirely; if everything matched, keep the head.
        if not any(ln.strip() for ln in kept):
            kept = text.split("\n")[:4]
        lean = dict(message)
        # Plain-string content also drops the image part of a multimodal message.
        lean["content"] = "\n".join(kept).strip()
        out.append(lean)
    return out


_dd_original_chat_completion = _dd_tool_agent.ToolAgent._chat_completion


def _dd_chat_completion(self, messages, **kwargs):
    try:
        lean = _dd_lean(messages)
        _DD_STATS["calls"] += 1
        _DD_STATS["chars_before"] += len(json.dumps(messages, default=str))
        _DD_STATS["chars_after"] += len(json.dumps(lean, default=str))
    except Exception as exc:  # never let the optimisation kill a game
        print("__MARKER__ WARNING: falling back to full messages: %r" % (exc,), flush=True)
        lean = messages
    return _dd_original_chat_completion(self, lean, **kwargs)


_dd_tool_agent.ToolAgent._chat_completion = _dd_chat_completion

# Prove the transform on a synthetic two-turn history before the run starts.
_dd_boiler = "Only tool: `python`. It receives current_frame."
_dd_probe = [
    {"role": "system", "content": "sys"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Current state: step 1, level 1.\n" + _dd_boiler + "\n\nCurrent grid image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 12000}},
        ],
    },
    {"role": "assistant", "content": "a"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Current state: step 9, level 1.\n" + _dd_boiler + "\n\nCurrent grid image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "B" * 12000}},
        ],
    },
]
_dd_out = _dd_lean(_dd_probe)
assert "step 1" in _dd_out[1]["content"], "unique historical line was dropped"
assert _dd_boiler not in _dd_out[1]["content"], "duplicated line was not dropped"
assert "base64" not in json.dumps(_dd_out[1]), "stale image was not dropped"
assert _dd_out[3] == _dd_probe[3], "newest user message must be untouched"
assert _dd_out[0] == _dd_probe[0] and _dd_out[2] == _dd_probe[2], "non-user message touched"
print(
    "__MARKER__ active: hook=ToolAgent._chat_completion probe_chars=%d->%d"
    % (len(json.dumps(_dd_probe)), len(json.dumps(_dd_out))),
    flush=True,
)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-id", required=True)
    args = ap.parse_args()
    build(
        base_nb=BASE_NB,
        out_nb=NB,
        meta=META,
        baseline_md5=BASELINE_MD5,
        hook_cell=HOOK_CELL,
        hook_anchor=HOOK_ANCHOR,
        patch=PATCH.replace("__MARKER__", MARKER),
        kernel_id=args.kernel_id,
    )
    print(f"patched cell {HOOK_CELL}: history dedupe on the send path")


if __name__ == "__main__":
    main()
