"""The exact source edits the HUD arm makes to the anim solver bundle.

Four files change, and nothing else. Every edit is an **anchored** string
replacement: the anchor must occur exactly once in the file or the patch
refuses, so a bundle that has drifted underneath us fails loudly instead of
being half-patched. ``scripts/_build_hud_bundle.py`` applies these to a
pristine bundle and sha256-checks that the other 71 files are untouched;
``tests/test_hud_arm.py`` applies them in memory and executes the result.

The arm's whole design constraint, restated because it is the thing that makes
this safe: **the annotation is additive and advisory. No pixel, node, edge or
field is removed, masked, hidden or altered.** With
``ARC3_HUD_ANNOTATION`` unset the patched ``segment_layer`` returns a dict that
is equal to the unpatched one, key for key.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent

SPLICE_BEGIN = "# _HUD_SPLICE_BEGIN"
SPLICE_END = "# _HUD_SPLICE_END"

#: Files this arm touches, relative to ``<bundle>/src/ARC3-Inference``.
PATCHED_FILES = (
    "inference/utils/segmentation.py",
    "inference/agent/python_tool_sandbox.py",
    "inference/agent/prompts.py",
    "inference/agent/tool_agent.py",
)

ENV_FLAG = "ARC3_HUD_ANNOTATION"


def detector_source() -> str:
    """The body of ``arc3_hud/hud_detect.py`` between the splice markers."""
    text = (_HERE / "hud_detect.py").read_text(encoding="utf-8")
    try:
        start = text.index(SPLICE_BEGIN)
        end = text.index(SPLICE_END)
    except ValueError as exc:  # pragma: no cover - a corrupted checkout
        raise RuntimeError("hud_detect.py lost its splice markers") from exc
    # Skip the remainder of the marker's own line (it carries a trailing comment).
    start = text.index("\n", start) + 1
    body = text[start:end]
    return body.strip("\n") + "\n"


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{where}: anchor occurs {count} times, expected exactly 1 -- "
            f"the bundle drifted. Anchor: {old[:90]!r}"
        )
    return text.replace(old, new)


# --------------------------------------------------------------- segmentation

_SEG_IMPORT_ANCHOR = "import hashlib\n"
_SEG_IMPORT_NEW = """\
import hashlib
import os
"""

_SEG_GATE = '''\

# ---------------------------------------------------------------------------
# [calamitychasm] HUD annotation -- ADDITIVE AND ADVISORY ONLY.
#
# Ported from Rudakov, Shock & Cowley, arXiv:2512.24156 (MIT); see
# THIRD_PARTY_NOTICE.md. Gated OFF by default: with ARC3_HUD_ANNOTATION unset,
# segment_layer's return value is identical to the unpatched bundle's, key for
# key, and no node carries a "hud" field.
#
# The gate is read per call rather than at import because the only caller is the
# Python-tool sandbox, which is a fresh subprocess per tool call.
# ---------------------------------------------------------------------------


def _hud_annotation_enabled():
    return os.environ.get("ARC3_HUD_ANNOTATION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


__HUD_DETECTOR__

'''

_SEG_DOC_ANCHOR = """\
    Returns a dict with:
      - ``nodes``: list of the node dicts above, in id order.
      - ``adjacency_list``: sorted list of ``[i, j]`` id pairs for components that share
        a 4-connected edge (includes parent/child pairs, since they physically touch).
    \"\"\"
"""

_SEG_DOC_NEW = """\
    Returns a dict with:
      - ``nodes``: list of the node dicts above, in id order.
      - ``adjacency_list``: sorted list of ``[i, j]`` id pairs for components that share
        a 4-connected edge (includes parent/child pairs, since they physically touch).

    When ``ARC3_HUD_ANNOTATION`` is set, and only then, two advisory fields are
    **added** (nothing is removed or altered): each node gains ``hud`` (bool), and the
    returned dict gains ``hud_node_ids`` (the sorted ids where ``hud`` is True). These
    are a shape-and-position heuristic for timer/score/status chrome -- see
    ``detect_hud_nodes`` -- and may be both wrong and incomplete.
    \"\"\"
"""

_SEG_RETURN_ANCHOR = '''\
    return {"nodes": nodes, "adjacency_list": adjacency_list}
'''

_SEG_RETURN_NEW = '''\
    # [calamitychasm] advisory HUD annotation; strictly additive, gated off by default.
    if _hud_annotation_enabled():
        hud_ids = detect_hud_nodes(nodes, height, width)
        flagged = set(hud_ids)
        for node in nodes:
            node["hud"] = node["id"] in flagged
        return {"nodes": nodes, "adjacency_list": adjacency_list, "hud_node_ids": hud_ids}

    return {"nodes": nodes, "adjacency_list": adjacency_list}
'''


def patch_segmentation(text: str) -> str:
    out = _replace_once(text, _SEG_IMPORT_ANCHOR, _SEG_IMPORT_NEW, "segmentation.py import")
    gate = _SEG_GATE.replace("__HUD_DETECTOR__\n", detector_source())
    out = _replace_once(
        out, "\n\ndef segment_layer(", gate + "\ndef segment_layer(", "segmentation.py helpers"
    )
    out = _replace_once(out, _SEG_DOC_ANCHOR, _SEG_DOC_NEW, "segmentation.py docstring")
    out = _replace_once(out, _SEG_RETURN_ANCHOR, _SEG_RETURN_NEW, "segmentation.py return")
    return out


# ------------------------------------------------------------- sandbox env

_SANDBOX_ANCHOR = '''\
def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }
'''

_SANDBOX_NEW = '''\
def _sandbox_env() -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }
    # [calamitychasm] The sandbox is a subprocess with an explicit allowlist env, and
    # segment_layer -- the only caller of the HUD detector -- runs *inside* it. Without
    # this one passthrough the gate could never reach the detector and the arm would
    # silently ship as a no-op. Nothing else is added: the allowlist stays an allowlist.
    _hud_flag = os.environ.get("ARC3_HUD_ANNOTATION")
    if _hud_flag is not None:
        env["ARC3_HUD_ANNOTATION"] = _hud_flag
    return env
'''


def patch_sandbox(text: str) -> str:
    return _replace_once(text, _SANDBOX_ANCHOR, _SANDBOX_NEW, "python_tool_sandbox.py env")


# ----------------------------------------------------------------- prompts

_PROMPTS_IMPORT_ANCHOR = '''\
"""Prompt templates for the analyzer agent."""

from inference.utils.grid_utils import ARC_COLOR_LEGEND
'''

_PROMPTS_IMPORT_NEW = '''\
"""Prompt templates for the analyzer agent."""

import os

from inference.utils.grid_utils import ARC_COLOR_LEGEND

# [calamitychasm] HUD-annotation arm. Read once at import, in the host process, because
# the prompt constants below are built at import time. Default OFF: with the flag unset
# every string in this module is byte-identical to the unpatched bundle's.
_HUD_ANNOTATION = os.environ.get("ARC3_HUD_ANNOTATION", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Deliberately honest about what the flag is and is not. The detector is a rule over
# shape and position with no knowledge of the game; it misses HUD blocks set away from
# the edges, and it can flag a genuine playable object that happens to be a long
# edge-hugging bar. Overstating it would buy a confident wrong belief, which is worse
# than the status quo of re-deriving the geometry every turn.
_HUD_SEGMENTATION_BULLETS = (
    "- Each node also carries `hud` (bool), and `segmentation['hud_node_ids']` lists the ids"
    " flagged True. The flag marks a node that sits flush against a frame edge and is either"
    " a long thin bar (at least 5:1) or one of at least three identical shapes along that"
    " edge -- the shape of a timer / score / status strip rather than a playable object.\\n"
    "- HEURISTIC. It is a rule over shape and position only and knows nothing about this"
    " game. It MISSES HUD drawn as a block away from the edges, and it can wrongly flag a"
    " real object that happens to be a long edge-hugging bar. Nothing is hidden because of"
    " it: every node is present exactly as it would be otherwise, flagged or not. Verify"
    " before relying on it -- if a flagged object responds to your actions in a way that"
    " matters, it is not HUD, and if the interior stops making sense look for HUD the flag"
    " missed.\\n"
) if _HUD_ANNOTATION else ""

_HUD_PYTHON_BULLET = (
    "- `segmentation['hud_node_ids']` is a cheap first pass at which objects are timer /"
    " status chrome. Use it to skip re-deriving HUD geometry by hand, then confirm it from"
    " behaviour rather than trusting it.\\n"
) if _HUD_ANNOTATION else ""

# Leading AND trailing space: this is spliced between two adjacent literals in a
# concatenated run, so both sides need the separator.
_HUD_TOOL_SENTENCE = (
    " Nodes may carry an advisory, sometimes-wrong `hud` flag (also `segmentation['hud_node_ids']`)"
    " marking probable timer/status chrome; nothing is hidden because of it. "
) if _HUD_ANNOTATION else ""

_HUD_TURN_LINE = (
    "`segmentation['hud_node_ids']` is a heuristic, possibly wrong, list of probable"
    " timer/status-chrome nodes -- use it instead of re-deriving HUD geometry, but confirm it."
) if _HUD_ANNOTATION else ""
'''

_PROMPTS_ADJ_ANCHOR = (
    '    "- `segmentation[\'adjacency_list\']` is a list of `[i, j]` node-id pairs '
    'whose objects share an edge.\\n"\n'
)

# Explicit `+`, not implicit concatenation: a name cannot be juxtaposed with a
# string literal, and these constants sit inside paren-joined literal runs.
_PROMPTS_ADJ_NEW = _PROMPTS_ADJ_ANCHOR + "    + _HUD_SEGMENTATION_BULLETS +\n"

_PROMPTS_PY_ANCHOR = (
    '    "- Use `current_frame.ascii` only to read a small, specific region of the board '
    'when `segmentation` is not enough; never use it to scan or summarize the whole board.\\n"\n'
)

_PROMPTS_PY_NEW = _PROMPTS_PY_ANCHOR + "    + _HUD_PYTHON_BULLET +\n"


def patch_prompts(text: str) -> str:
    out = _replace_once(text, _PROMPTS_IMPORT_ANCHOR, _PROMPTS_IMPORT_NEW, "prompts.py header")
    out = _replace_once(out, _PROMPTS_ADJ_ANCHOR, _PROMPTS_ADJ_NEW, "prompts.py segmentation")
    out = _replace_once(out, _PROMPTS_PY_ANCHOR, _PROMPTS_PY_NEW, "prompts.py python addendum")
    return out


# --------------------------------------------------------------- tool_agent

_TA_IMPORT_ANCHOR = (
    '    "The raw numeric grid is not available. Use `.segmentation` as the primary view; '
    'use `.ascii` only to read a small, specific region. "\n'
)

_TA_IMPORT_NEW = (
    '    "The raw numeric grid is not available. Use `.segmentation` as the primary view; '
    'use `.ascii` only to read a small, specific region. "\n'
    "    + _HUD_TOOL_SENTENCE +\n"
)

_TA_TURN_ANCHOR = (
    '                "Keep tool output compact: use `current_frame.segmentation` as the '
    "primary view, and `current_frame.ascii` only for a small specific region; never print "
    'full boards.",\n'
)

_TA_TURN_NEW = _TA_TURN_ANCHOR + "                *( [_HUD_TURN_LINE] if _HUD_TURN_LINE else [] ),\n"

_TA_RETRY_ANCHOR = (
    '                        "Call the `python` tool with code that inspects `current_frame`, '
    "`previous_frame`, `last_transition`, `history`, or `valid_actions` -- use "
    '`current_frame.segmentation` as the primary view, and `.ascii` only for a small specific region -- "\n'
)

_TA_RETRY_NEW = (
    _TA_RETRY_ANCHOR
    + '                        + ("(nodes may carry an advisory, sometimes-wrong `hud` flag) " if _HUD_TURN_LINE else "") +\n'
)

_TA_PROMPT_IMPORT_ANCHOR = "from inference.agent.prompts import (\n"
_TA_PROMPT_IMPORT_NEW = (
    "from inference.agent.prompts import (\n"
    "    _HUD_TOOL_SENTENCE,   # [calamitychasm] HUD arm; empty string when the flag is off\n"
    "    _HUD_TURN_LINE,\n"
)


def patch_tool_agent(text: str) -> str:
    out = _replace_once(
        text, _TA_PROMPT_IMPORT_ANCHOR, _TA_PROMPT_IMPORT_NEW, "tool_agent.py prompt import"
    )
    out = _replace_once(out, _TA_IMPORT_ANCHOR, _TA_IMPORT_NEW, "tool_agent.py tool description")
    out = _replace_once(out, _TA_TURN_ANCHOR, _TA_TURN_NEW, "tool_agent.py turn lines")
    out = _replace_once(out, _TA_RETRY_ANCHOR, _TA_RETRY_NEW, "tool_agent.py retry prompt")
    return out


PATCHERS = {
    "inference/utils/segmentation.py": patch_segmentation,
    "inference/agent/python_tool_sandbox.py": patch_sandbox,
    "inference/agent/prompts.py": patch_prompts,
    "inference/agent/tool_agent.py": patch_tool_agent,
}


def patch_text(rel_path: str, text: str) -> str:
    return PATCHERS[rel_path](text)
