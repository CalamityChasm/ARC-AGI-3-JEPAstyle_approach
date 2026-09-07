"""Sparse LLM fallback for planner stalls. Called only when
`planner.plan()` reports `stalled=True` -- the beam search found no action
that predicts a level win or even meaningfully distinguishes itself via
goal_hint. That's exactly the situation a pure code-simulator search
handles badly (it can only rank actions by what the current WorldModel
already believes; if the model has no opinion, search has nothing to act
on) and a model with broad gameplay/visual judgment might reasonably do
better by eyeballing the board.

This is deliberately a *sparse* fallback, not a per-step policy: the
moment-to-moment loop stays the cheap, LLM-free beam search in planner.py.
See architecture.md's "Two LLM roles" section for the full rationale,
including why this is a good fit for a weaker-precision model (Gemma) even
though the same imprecision ruled images out for the primary state
representation -- picking *one* promising action tolerates being
approximately right in a way that predicting exact cell values does not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .diff import format_grid
from .llm_client import LLMClient
from .types import ALL_ACTIONS, Action, GameTranscript

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are choosing the next move in an unfamiliar grid-based game. A \
search over the current (possibly incomplete or wrong) rule model found no promising \
action -- every option looks equally uninformative to it. Use your own judgment about \
what a game like this probably wants: look for anything that stands out (an odd-colored \
tile, an object that looks like a door/key/switch, an edge or corner pattern) and guess \
an action aimed at it.

Grids are lists of layers; each layer is a 64x64 list of integers 0-15 (colors). \
Coordinates are (x, y), 0-indexed, origin top-left.

Actions: ACTION1-ACTION5 and ACTION7 take no arguments. ACTION6 takes (x, y).

Respond with EXACTLY ONE line in this format, nothing else:
ACTIONn
or, for the coordinate action:
ACTION6 x y
"""

_ACTION_LINE_RE = re.compile(r"\bACTION([1-7])\b(?:\s+(\d+)\s+(\d+))?")


def _render_recent(transcript: GameTranscript, n: int = 5) -> str:
    if not transcript.transitions:
        return "(no history yet)"
    lines = []
    for t in transcript.transitions[-n:]:
        lines.append(f"action {t.action} -> levels_completed {t.levels_completed_after}, state {t.state_after}")
    return "\n".join(lines)


def parse_action(response_text: str) -> Optional[Action]:
    match = _ACTION_LINE_RE.search(response_text)
    if not match:
        return None
    num = match.group(1)
    name = "ACTION6" if num == "6" else f"ACTION{num}"
    if name not in ALL_ACTIONS:
        return None
    if name == "ACTION6":
        if match.group(2) is None or match.group(3) is None:
            return None
        x, y = int(match.group(2)), int(match.group(3))
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return None
        return Action(name=name, x=x, y=y)
    return Action(name=name)


@dataclass
class ActionHeadOutcome:
    ok: bool
    action: Optional[Action] = None
    raw_response: Optional[str] = None


def suggest_action(client: LLMClient, transcript: GameTranscript, current_grid) -> ActionHeadOutcome:  # type: ignore[no-untyped-def]
    """One call, no retry loop -- this is a sparse fallback, not something
    worth spending a repair-style attempt budget on. If it fails to
    produce a parseable action, the caller falls back further (e.g. a
    random legal action)."""
    user_prompt = (
        f"Recent history:\n{_render_recent(transcript)}\n\n"
        f"Current board (layer 0):\n{format_grid(current_grid)}\n\n"
        "Which action?"
    )
    try:
        response = client.complete(_SYSTEM_PROMPT, user_prompt, max_tokens=64)
    except Exception as e:  # noqa: BLE001 -- a dead/unreachable server must not crash the agent
        logger.warning("action_head call failed: %s: %s", type(e).__name__, e)
        return ActionHeadOutcome(ok=False)

    action = parse_action(response)
    if action is None:
        logger.info("action_head: could not parse an action from response: %r", response[:200])
        return ActionHeadOutcome(ok=False, raw_response=response)
    return ActionHeadOutcome(ok=True, action=action, raw_response=response)
