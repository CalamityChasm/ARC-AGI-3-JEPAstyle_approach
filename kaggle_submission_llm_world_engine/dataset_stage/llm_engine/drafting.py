"""Draft and repair WorldModel source via the LLM, gated on replay -- see
architecture.md steps 2-3 (draft, validate-by-replay) and step 5 (surprise
triggers repair). Every accepted revision is one that has actually passed
replay on the transcript seen so far; nothing is accepted on the model's
say-so alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .diff import format_diff, format_grid
from .llm_client import LLMClient, extract_code
from .replay import ReplayResult, describe_failure, replay
from .types import GameTranscript
from .world_model import WORLD_MODEL_SKELETON, LoadResult, WorldModelProtocol, load_world_model

logger = logging.getLogger(__name__)


def _safe_complete(client: LLMClient, system: str, user: str, max_tokens: int) -> Optional[str]:
    """Wrap client.complete so a dead/unreachable LLM server (connection
    refused, timeout, HTTP error) degrades to 'no response this attempt'
    instead of crashing the agent thread outright. A real failure here is
    exactly the kind of thing that must not take down a whole Swarm game."""
    try:
        return client.complete(system, user, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001 -- any client/network failure
        logger.warning("LLM call failed: %s: %s", type(e).__name__, e)
        return None

_SYSTEM_PROMPT = """You are writing a Python simulator for one level of an unfamiliar \
grid-based game. You will be shown a transcript of actions taken and what happened. \
Infer the general RULE behind each observed change, not the specific numbers in the \
examples -- your code will be tested against the same transcript and must reproduce \
it exactly, but it should express a rule that plausibly generalizes.

Grids are lists of layers; each layer is a 64x64 list of integers 0-15 (colors). Most \
games use exactly one layer. Coordinates are (x, y), 0-indexed, origin top-left.

Actions: ACTION1-ACTION5 and ACTION7 take no arguments. ACTION6 takes (x, y). What \
each action actually does is game-specific and only knowable from the transcript.

Your code runs with NO imports available (no `import` statements at all -- write \
plain Python using only builtins like list/dict/range/len/enumerate/min/max/abs). \
Do not use numpy, copy, math, or any other module.

Respond with ONLY a single Python code fence containing a `WorldModel` class with \
this exact interface, nothing else:

```python
class WorldModel:
    def __init__(self):
        ...  # hidden state, e.g. self.counters = {}, if you infer the game tracks one

    def predict(self, state, action_name, x=None, y=None):
        # Return (next_state, levels_delta, done)
        ...

    def goal_hint(self, state):
        # float: higher = closer to a win condition
        ...
```
"""


def _render_transcript(transcript: GameTranscript, max_transitions: int = 40) -> str:
    lines = []
    shown = transcript.transitions[-max_transitions:]
    if len(transcript.transitions) > max_transitions:
        lines.append(f"[... {len(transcript.transitions) - max_transitions} earlier transitions omitted ...]")
    for i, t in enumerate(shown):
        idx = len(transcript.transitions) - len(shown) + i
        lines.append(
            f"--- transition #{idx} ---\n"
            f"action: {t.action}\n"
            f"before (layer 0):\n{format_grid(t.frame_before)}\n"
            f"diff after action: {format_diff(t.frame_before, t.frame_after)}\n"
            f"levels_completed: {t.levels_completed_before} -> {t.levels_completed_after}\n"
            f"state_after: {t.state_after}"
        )
    return "\n\n".join(lines)


@dataclass
class DraftOutcome:
    ok: bool
    source: Optional[str] = None
    world_model: Optional[WorldModelProtocol] = None
    attempts: int = 0
    replay_result: Optional[ReplayResult] = None


def draft_world_model(
    client: LLMClient,
    transcript: GameTranscript,
    max_attempts: int = 5,
) -> DraftOutcome:
    """First-draft loop: prompt for a WorldModel, replay-check it against
    the transcript, and if it fails, tell the LLM exactly which transition
    broke and why, up to max_attempts. Falls back to the (honest, useless)
    skeleton model on total failure so callers always get *something*
    loadable rather than needing to special-case None."""
    user_prompt = (
        f"Transcript for game {transcript.game_id}:\n\n"
        f"{_render_transcript(transcript)}\n\n"
        "Write the WorldModel now."
    )

    last_load: Optional[LoadResult] = None
    last_replay: Optional[ReplayResult] = None

    for attempt in range(1, max_attempts + 1):
        response = _safe_complete(client, _SYSTEM_PROMPT, user_prompt, max_tokens=2048)
        if response is None:
            logger.warning("draft attempt %d: LLM unreachable, aborting to fallback", attempt)
            break
        source = extract_code(response)
        load = load_world_model(source)
        last_load = load

        if not load.ok:
            logger.info("draft attempt %d: load failed: %s", attempt, load.error)
            user_prompt = (
                f"Your previous code failed to load: {load.error}\n\n"
                f"Here it was:\n```python\n{source}\n```\n\n"
                "Fix it and respond with only the corrected code fence."
            )
            continue

        result = replay(transcript, load.world_model)  # type: ignore[arg-type]
        last_replay = result
        if result.passed:
            logger.info("draft attempt %d: replay passed (%d/%d)", attempt, result.pass_count, result.total)
            return DraftOutcome(ok=True, source=source, world_model=load.world_model, attempts=attempt, replay_result=result)

        logger.info(
            "draft attempt %d: replay failed (%d/%d), first failure at #%d",
            attempt, result.pass_count, result.total, result.first_failure.index,  # type: ignore[union-attr]
        )
        user_prompt = (
            f"Your code loaded but doesn't reproduce the transcript.\n\n"
            f"{describe_failure(transcript, result.first_failure)}\n\n"  # type: ignore[arg-type]
            f"Here was your code:\n```python\n{source}\n```\n\n"
            "Fix it (you may rewrite the whole thing) and respond with only the "
            "corrected code fence."
        )

    logger.warning(
        "draft failed to pass replay after %d attempts for game %s; falling back to skeleton",
        max_attempts, transcript.game_id,
    )
    fallback = load_world_model(WORLD_MODEL_SKELETON)
    return DraftOutcome(
        ok=False,
        source=WORLD_MODEL_SKELETON,
        world_model=fallback.world_model,
        attempts=max_attempts,
        replay_result=last_replay,
    )


def repair_world_model(
    client: LLMClient,
    transcript: GameTranscript,
    current_source: str,
    max_attempts: int = 3,
) -> DraftOutcome:
    """Online repair loop (architecture.md step 5): called when a
    known-good model's prediction diverged from a *new* real observation
    (already appended to `transcript`). Requires the patch to re-pass
    replay on the FULL transcript, old transitions included -- the guard
    against the model 'fixing' the new case by breaking old ones."""
    result = replay(transcript, load_world_model(current_source).world_model)  # type: ignore[union-attr]
    if result.passed:
        # Shouldn't normally happen (caller only repairs on an observed
        # divergence), but if it does there's nothing to repair.
        return DraftOutcome(ok=True, source=current_source, world_model=load_world_model(current_source).world_model, attempts=0, replay_result=result)

    user_prompt = (
        f"This WorldModel worked until now, but just failed to predict a new observation:\n\n"
        f"{describe_failure(transcript, result.first_failure)}\n\n"  # type: ignore[arg-type]
        f"Current code:\n```python\n{current_source}\n```\n\n"
        "Patch it so it handles this new case WITHOUT breaking any earlier transitions "
        "(your patch will be replayed against the full history). Respond with only the "
        "corrected code fence."
    )

    for attempt in range(1, max_attempts + 1):
        response = _safe_complete(client, _SYSTEM_PROMPT, user_prompt, max_tokens=2048)
        if response is None:
            logger.warning("repair attempt %d: LLM unreachable, aborting", attempt)
            break
        source = extract_code(response)
        load = load_world_model(source)

        if not load.ok:
            user_prompt = f"That failed to load: {load.error}\n\nFix it and respond with only the corrected code fence."
            continue

        replay_result = replay(transcript, load.world_model)  # type: ignore[arg-type]
        if replay_result.passed:
            return DraftOutcome(ok=True, source=source, world_model=load.world_model, attempts=attempt, replay_result=replay_result)

        user_prompt = (
            f"Still failing: {describe_failure(transcript, replay_result.first_failure)}\n\n"  # type: ignore[arg-type]
            f"Current code:\n```python\n{source}\n```\n\n"
            "Try again -- respond with only the corrected code fence."
        )

    logger.warning("repair failed after %d attempts for game %s; keeping last known-good model", max_attempts, transcript.game_id)
    return DraftOutcome(ok=False, source=current_source, world_model=load_world_model(current_source).world_model, attempts=max_attempts, replay_result=result)
