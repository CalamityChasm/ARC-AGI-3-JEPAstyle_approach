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
from .world_model import LoadResult, WorldModelProtocol, load_world_model

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


# A complete WorldModel for a 64x64 game is not a short function. The
# original 2048 left no headroom for a model that writes any preamble at
# all, and a response cut off mid-class is not a partial answer -- it is a
# guaranteed compile failure that costs a whole attempt. See
# experiments/stage7_codeworld_fixes.md.
DRAFT_MAX_TOKENS = 4096


@dataclass
class DraftOutcome:
    ok: bool
    source: Optional[str] = None
    world_model: Optional[WorldModelProtocol] = None
    attempts: int = 0
    replay_result: Optional[ReplayResult] = None
    # The last candidate the LLM produced when `ok` is False. Kept for the
    # on-disk diagnostic trail only -- it has NOT passed replay and must
    # never be installed as the agent's world model.
    last_candidate_source: Optional[str] = None


def _retry_prompt(transcript: GameTranscript, source: str, problem: str) -> str:
    """Build the next attempt's prompt.

    Crucially this **re-includes the transcript**. The original retry
    prompts replaced the user message wholesale with just the error and
    the broken code, so from attempt 2 onward the model was asked to
    infer a rule for data it could no longer see -- and every attempt
    after the first was made blind. That alone makes the multi-attempt
    loop close to worthless.
    """
    return (
        f"Transcript for game {transcript.game_id}:\n\n"
        f"{_render_transcript(transcript)}\n\n"
        f"You already tried this, and it did not work:\n\n"
        f"```python\n{source}\n```\n\n"
        f"The problem: {problem}\n\n"
        "Rewrite the WorldModel so it reproduces the transcript above exactly. "
        "Respond with only a single Python code fence."
    )


def draft_world_model(
    client: LLMClient,
    transcript: GameTranscript,
    max_attempts: int = 5,
) -> DraftOutcome:
    """First-draft loop: prompt for a WorldModel, replay-check it against
    the transcript, and if it fails, tell the LLM exactly which transition
    broke and why (alongside the transcript itself), up to max_attempts.

    On total failure this returns `ok=False` with **no world model at
    all** -- deliberately. It used to hand back a loaded copy of
    WORLD_MODEL_SKELETON so callers "always get something loadable", but
    the skeleton is an 'assume nothing ever changes' model: installing it
    gives the beam search a flat, zero-information objective *and*
    convinces the agent it has a model, which suppresses any further
    drafting and redirects the whole LLM budget into repairing a stub that
    can never pass replay. A caller with no model can fall back to a
    random legal action for free; a caller with a stub cannot. See
    experiments/stage7_codeworld_fixes.md.
    """
    user_prompt = (
        f"Transcript for game {transcript.game_id}:\n\n"
        f"{_render_transcript(transcript)}\n\n"
        "Write the WorldModel now."
    )

    last_load: Optional[LoadResult] = None
    last_replay: Optional[ReplayResult] = None
    last_source: Optional[str] = None
    attempts_made = 0

    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        response = _safe_complete(client, _SYSTEM_PROMPT, user_prompt, max_tokens=DRAFT_MAX_TOKENS)
        if response is None:
            logger.warning("draft attempt %d: LLM unreachable, aborting", attempt)
            break
        source = extract_code(response)
        last_source = source
        load = load_world_model(source)
        last_load = load

        if not load.ok:
            logger.info("draft attempt %d: load failed: %s", attempt, load.error)
            user_prompt = _retry_prompt(
                transcript, source, f"it failed to load: {load.error}"
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
        user_prompt = _retry_prompt(
            transcript,
            source,
            "it loaded but does not reproduce the transcript. "
            + describe_failure(transcript, result.first_failure),  # type: ignore[arg-type]
        )

    logger.warning(
        "draft failed to produce a replay-passing model after %d attempt(s) for game %s "
        "(last load error: %s); continuing WITHOUT a world model rather than installing "
        "the do-nothing skeleton",
        attempts_made, transcript.game_id, last_load.error if last_load else "n/a",
    )
    return DraftOutcome(
        ok=False,
        source=None,
        world_model=None,
        attempts=attempts_made,
        replay_result=last_replay,
        last_candidate_source=last_source,
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
    current_load = load_world_model(current_source)
    if not current_load.ok:
        # The caller should never hand us un-loadable source (it only ever
        # passes a revision that previously passed replay), but if it does,
        # this is a re-draft, not a repair.
        logger.warning("repair called with un-loadable source: %s", current_load.error)
        return draft_world_model(client, transcript, max_attempts=max_attempts)

    result = replay(transcript, current_load.world_model)  # type: ignore[arg-type]
    if result.passed:
        # Shouldn't normally happen (caller only repairs on an observed
        # divergence), but if it does there's nothing to repair.
        return DraftOutcome(ok=True, source=current_source, world_model=current_load.world_model, attempts=0, replay_result=result)

    user_prompt = _retry_prompt(
        transcript,
        current_source,
        "it worked until now, but just failed to predict a new observation. "
        + describe_failure(transcript, result.first_failure)  # type: ignore[arg-type]
        + " Patch it so it handles this new case WITHOUT breaking any earlier "
        "transitions -- your patch will be replayed against the full history above.",
    )

    for attempt in range(1, max_attempts + 1):
        response = _safe_complete(client, _SYSTEM_PROMPT, user_prompt, max_tokens=DRAFT_MAX_TOKENS)
        if response is None:
            logger.warning("repair attempt %d: LLM unreachable, aborting", attempt)
            break
        source = extract_code(response)
        load = load_world_model(source)

        if not load.ok:
            user_prompt = _retry_prompt(
                transcript, source, f"it failed to load: {load.error}"
            )
            continue

        replay_result = replay(transcript, load.world_model)  # type: ignore[arg-type]
        if replay_result.passed:
            return DraftOutcome(ok=True, source=source, world_model=load.world_model, attempts=attempt, replay_result=replay_result)

        user_prompt = _retry_prompt(
            transcript,
            source,
            "it still does not reproduce the transcript. "
            + describe_failure(transcript, replay_result.first_failure),  # type: ignore[arg-type]
        )

    logger.warning("repair failed after %d attempts for game %s; keeping last known-good model", max_attempts, transcript.game_id)
    return DraftOutcome(ok=False, source=current_source, world_model=current_load.world_model, attempts=max_attempts, replay_result=result)
