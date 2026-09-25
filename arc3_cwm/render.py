"""Compact prompt rendering for a level segment.

Why this exists instead of reusing `llm_engine.drafting._render_transcript`:
that renderer emits `format_grid(t.frame_before)` for **every** transition
it shows, up to 40 of them. A 64x64 grid renders as 64 lines of 64
characters, so a 40-transition transcript is ~170k characters of grid
before a single diff is added -- far past any context this model is served
with, and overwhelmingly redundant, because consecutive `frame_before`
grids differ only by the previous step's diff, which is already printed.

Running the backtest through that renderer would measure context overflow
and call it a capability result. So this module renders the equivalent
information in the form the data actually has:

    full opening grid, once
    then per step: action, cell diff, levels, terminal state

which is lossless with respect to the transcript (the reader can
reconstruct every intermediate grid by applying the diffs in order) at a
small fraction of the size. `measure_sizes()` reports both so the
write-up can quote the real ratio rather than an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._engine import format_diff, format_grid
from .extract import LevelSegment

#: Cap on cells listed per diff. A step that rewrites more of the board
#: than this is noted as truncated rather than silently shortened -- a
#: model told "37 cells changed" when 900 did is being actively misled.
MAX_DIFF_CELLS = 240

#: Default step window, matching `llm_engine.drafting._render_transcript`'s
#: own `max_transitions=40` so size comparisons between the two renderers
#: are like-for-like. Real segments run to 209 steps, which no renderer
#: makes fit. Windowing is done by `LevelSegment.window()` **before**
#: rendering, not here, so that the steps shown and the steps replayed
#: against are the same list by construction -- this module renders
#: exactly what it is given.
MAX_STEPS = 40


def render_segment(segment: LevelSegment, max_diff_cells: int = MAX_DIFF_CELLS) -> str:
    """Render one level segment as the transcript body of a draft prompt."""
    if not segment.transitions:
        return "(no transitions)"

    shown = segment.transitions
    first = shown[0]
    lines = [
        f"Game {segment.game_id}, level {segment.level}. {len(shown)} observed steps.",
        "",
        "Opening grid (layer 0), one row per line, one hex digit per cell:",
        format_grid(first.frame_before),
        "",
        "Then each step below gives the action taken and the cells it changed, "
        "as (layer,x,y): old->new. Apply them in order to follow the board.",
        "",
    ]

    for i, t in enumerate(shown):
        diff = format_diff(t.frame_before, t.frame_after, max_cells=max_diff_cells)
        changed = sum(
            1
            for lb, la in zip(t.frame_before, t.frame_after)
            for rb, ra in zip(lb, la)
            for a, b in zip(rb, ra)
            if a != b
        )
        if changed > max_diff_cells:
            diff += f"  [... {changed - max_diff_cells} further changed cells omitted ...]"

        parts = [f"step {i}: {t.action} -> {diff}"]
        if t.levels_delta:
            parts.append(f"levels_delta={t.levels_delta}")
        if t.state_after != "NOT_FINISHED":
            parts.append(f"state={t.state_after}")
        lines.append("  ".join(parts))

    return "\n".join(lines)


def build_user_prompt(segment: LevelSegment, max_diff_cells: int = MAX_DIFF_CELLS) -> str:
    """The full user message for a first-draft attempt."""
    return (
        f"{render_segment(segment, max_diff_cells=max_diff_cells)}\n\n"
        "Write the WorldModel now. It must reproduce every step above exactly "
        "when replayed from the opening grid."
    )


def build_repair_prompt(
    segment: LevelSegment,
    source: str,
    problem: str,
    max_diff_cells: int = MAX_DIFF_CELLS,
) -> str:
    """Next-attempt prompt: transcript re-included, as the engine's own
    drafting loop learned it must be (a retry that drops the transcript
    asks the model to infer a rule for data it can no longer see)."""
    return (
        f"{render_segment(segment, max_diff_cells=max_diff_cells)}\n\n"
        f"You already tried this, and it did not work:\n\n"
        f"```python\n{source}\n```\n\n"
        f"The problem: {problem}\n\n"
        "Rewrite the WorldModel so it reproduces every step above exactly. "
        "Respond with only a single Python code fence."
    )


@dataclass
class SizeComparison:
    """Rendered character counts, compact vs. the engine's own renderer."""

    segment_key: str
    transitions: int
    compact_chars: int
    engine_chars: int

    @property
    def ratio(self) -> float:
        return self.engine_chars / self.compact_chars if self.compact_chars else 0.0


def measure_sizes(segment: LevelSegment) -> SizeComparison:
    """Measure this renderer against `llm_engine.drafting`'s, on real data.

    Imported lazily so that merely rendering a prompt does not pull in the
    drafting module's LLM-client dependencies.
    """
    from llm_engine.drafting import _render_transcript
    from llm_engine.types import GameTranscript

    transcript = GameTranscript(game_id=segment.game_id)
    for t in segment.transitions:
        transcript.append(t)

    return SizeComparison(
        segment_key=segment.key,
        transitions=len(segment.transitions),
        compact_chars=len(render_segment(segment)),
        engine_chars=len(_render_transcript(transcript)),
    )


def fit_to_budget(
    segment: LevelSegment, max_prompt_chars: int, min_steps: int = 4
) -> LevelSegment:
    """Shrink the step window until the rendered prompt fits the budget.

    The served model has `max_model_len: 32768`, shared between prompt and
    response. A segment whose prompt overflows that does not produce a bad
    result -- it produces a request error or a silent truncation, which
    would be scored as a model failure. Dropping steps until it fits keeps
    the measurement honest, and the number of steps actually used is
    recorded per segment so a short window is visible in the output rather
    than hidden.

    Returns a segment of at least `min_steps` even if that still overflows;
    the caller decides whether to skip it, because silently returning
    something unusable is worse than an explicit oversize segment.
    """
    if max_prompt_chars <= 0:
        return segment

    current = segment
    while len(current) > min_steps and len(build_user_prompt(current)) > max_prompt_chars:
        # Halve the overshoot rather than stepping down one at a time:
        # rendering is the expensive part and segments run to 209 steps.
        overshoot = len(build_user_prompt(current)) / max_prompt_chars
        target = max(min_steps, int(len(current) / max(overshoot, 1.05)))
        if target >= len(current):
            target = len(current) - 1
        current = segment.window(target)
    return current
