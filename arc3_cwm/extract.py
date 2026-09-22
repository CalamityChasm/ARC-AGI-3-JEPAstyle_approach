"""Turn a real Duck/anim free-run's event log into replayable per-level
transition segments.

Input is `artifacts/<game>_p0_events.jsonl` as written by the anim solver
bundle: one JSON object per line, with `type` in
{initial, action, analysis, experiment}. Only `initial` and `action`
carry a board; the other two are the model's own deliberation and are
ignored here.

Three properties of that format drive everything in this module, each
verified directly against a real run (ar25-0c556536, 2026-09-21) rather
than assumed:

1. **`board` is a bare 64x64 grid**, not the engine's layered
   `Grid = list[list[list[int]]]`. It is wrapped as `[board]` on the way
   in, because every downstream consumer (replay, diff, the LLM prompt)
   speaks the layered form.

2. **`score` is levels_completed**, and `level` is the 1-based current
   level. On the event that clears a level, both have already advanced
   and `level_completed` is True.

3. **The board on a level-clearing event is the NEXT level's opening
   layout**, not the cleared level's final frame. Measured: boundary
   events rewrite 693-1054 of 4096 cells, against a median of 109 for an
   ordinary move. A per-level world model cannot predict a fresh layout,
   and the drafting prompt explicitly forbids hardcoding grids, so the
   boundary step is **excluded** from its segment's transitions and
   recorded separately via `cleared_level`.

Point 3 is the one that determines what the backtest measures. See
`experiments/stage7_codeworld_backtest.md`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from ._engine import Action, Transition

# 'MOUSE(row=23, col=52)' -- the only place an ACTION6's coordinates
# survive in the event log; `action_name` is just "ACTION6".
_MOUSE_RE = re.compile(r"MOUSE\(\s*row\s*=\s*(\d+)\s*,\s*col\s*=\s*(\d+)\s*\)")

BOARD_EVENT_TYPES = ("initial", "action")


class ExtractionError(ValueError):
    """Raised only for input this module cannot honestly interpret.

    Deliberately *not* raised for ordinary gaps (a missing board, an
    unparseable line): those are counted and skipped, because a single
    truncated line at the end of a killed run must not cost the whole
    game's data -- this repo has hit exactly that before (see CLAUDE.md's
    note on `extract_level_up_transitions.py` and the disk-full crash).
    """


@dataclass
class LevelSegment:
    """One level's worth of within-level transitions from one game."""

    game_id: str
    level: int
    transitions: list[Transition] = field(default_factory=list)
    cleared_level: bool = False
    #: Cells changed by the level-clearing step, when there was one. Kept
    #: only so the write-up can show *why* the boundary is excluded.
    boundary_cells_changed: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.game_id}/L{self.level}"

    def __len__(self) -> int:
        return len(self.transitions)

    def window(self, max_steps: int) -> "LevelSegment":
        """A copy holding only the first `max_steps` transitions.

        The backtest windows a segment **once**, before anything else, so
        that the steps rendered into the prompt and the steps replayed
        against are the same list by construction. A model must never be
        scored on a step it was not shown; keeping one cap in one place is
        what guarantees that.
        """
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        return LevelSegment(
            game_id=self.game_id,
            level=self.level,
            transitions=list(self.transitions[:max_steps]),
            # A truncated segment no longer runs up to the level change,
            # so it must not claim to.
            cleared_level=self.cleared_level and max_steps >= len(self.transitions),
            boundary_cells_changed=self.boundary_cells_changed,
        )

    @property
    def changed_fraction(self) -> float:
        """Share of transitions that actually altered the board.

        The complement is the free win rate for a do-nothing model, which
        is why the backtest always reports an identity baseline next to
        any pass rate.
        """
        if not self.transitions:
            return 0.0
        changed = sum(1 for t in self.transitions if t.frame_before != t.frame_after)
        return changed / len(self.transitions)


@dataclass
class ExtractionStats:
    """Everything skipped, and why. Counted, never silently dropped."""

    lines_read: int = 0
    lines_unparseable: int = 0
    events_without_board: int = 0
    transitions_built: int = 0
    transitions_dropped_boundary: int = 0
    #: RESET is not in the engine's ALL_ACTIONS and is not a modelable
    #: game action -- it restarts the level rather than transforming the
    #: board by a rule. Dropping the RESET *step* is safe for the pairing:
    #: the post-reset board is still the next step's `frame_before`.
    transitions_dropped_reset: int = 0
    transitions_dropped_malformed: int = 0
    action6_without_coords: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


def _parse_mouse_xy(display: Any) -> Optional[tuple[int, int]]:
    """(x, y) from an `action_display` string, or None.

    The log records row/col; the engine's Action takes (x, y) with origin
    top-left, so **x is col and y is row**. Getting this backwards would
    transpose every click and is invisible in an aggregate pass rate, so
    it is asserted in the unit tests against a real recorded event.
    """
    if not isinstance(display, str):
        return None
    m = _MOUSE_RE.search(display)
    if m is None:
        return None
    row, col = int(m.group(1)), int(m.group(2))
    return col, row


def _is_grid(board: Any) -> bool:
    """True for a non-empty rectangular list-of-lists of ints."""
    if not isinstance(board, list) or not board:
        return False
    width: Optional[int] = None
    for row in board:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool):
                return False
    return True


def _cells_changed(before: list[list[int]], after: list[list[int]]) -> int:
    return sum(1 for rb, ra in zip(before, after) for a, b in zip(rb, ra) if a != b)


def iter_events(path: Path, stats: ExtractionStats) -> Iterator[dict]:
    """Yield parsed events, tolerating a truncated or corrupt line."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats.lines_read += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                stats.lines_unparseable += 1
                continue
            if isinstance(event, dict):
                yield event


def segments_from_events(
    game_id: str, events: Iterable[dict], stats: ExtractionStats
) -> list[LevelSegment]:
    """Build per-level segments from an ordered event stream."""
    board_events = [e for e in events if e.get("type") in BOARD_EVENT_TYPES]

    usable: list[dict] = []
    for event in board_events:
        if not _is_grid(event.get("board")):
            stats.events_without_board += 1
            continue
        usable.append(event)

    if len(usable) < 2:
        return []

    segments: list[LevelSegment] = []
    current = LevelSegment(game_id=game_id, level=int(usable[0].get("level") or 1))

    for before_event, after_event in zip(usable, usable[1:]):
        if after_event.get("type") != "action":
            continue

        action = _build_action(after_event, stats)
        if action is None:
            if after_event.get("action_name") == "RESET":
                stats.transitions_dropped_reset += 1
            else:
                stats.transitions_dropped_malformed += 1
            continue

        before_board = before_event["board"]
        after_board = after_event["board"]

        if bool(after_event.get("level_completed")):
            # Boundary step: frame_after is the *next* level's opening
            # layout. Close the segment; do not ask a per-level model to
            # have predicted a fresh board.
            stats.transitions_dropped_boundary += 1
            current.cleared_level = True
            current.boundary_cells_changed = _cells_changed(before_board, after_board)
            segments.append(current)
            current = LevelSegment(
                game_id=game_id,
                level=int(after_event.get("level") or current.level + 1),
            )
            continue

        state_after = after_event.get("state") or "NOT_FINISHED"
        if after_event.get("game_over"):
            state_after = "GAME_OVER"

        current.transitions.append(
            Transition(
                frame_before=[before_board],
                action=action,
                frame_after=[after_board],
                levels_completed_before=int(before_event.get("score") or 0),
                levels_completed_after=int(after_event.get("score") or 0),
                state_after=str(state_after),
            )
        )
        stats.transitions_built += 1

    segments.append(current)
    return [s for s in segments if s.transitions]


def _build_action(event: dict, stats: ExtractionStats) -> Optional[Action]:
    name = event.get("action_name")
    if not isinstance(name, str):
        return None
    if name == "ACTION6":
        xy = _parse_mouse_xy(event.get("action_display"))
        if xy is None:
            # An ACTION6 whose coordinates we cannot recover is not a
            # transition anyone can replay -- count it and move on.
            stats.action6_without_coords += 1
            return None
        return Action(name="ACTION6", x=xy[0], y=xy[1])
    try:
        return Action(name=name)
    except ValueError:
        return None


def game_id_from_path(path: Path) -> str:
    """'ar25-0c556536_p0_events.jsonl' -> 'ar25-0c556536'."""
    stem = Path(path).name
    for suffix in ("_events.jsonl", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return re.sub(r"_p\d+$", "", stem)


def extract_run(
    artifacts_dir: Path, min_transitions: int = 1
) -> tuple[list[LevelSegment], ExtractionStats]:
    """Extract every per-level segment from a whole run's artifacts dir."""
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.is_dir():
        raise ExtractionError(f"not a directory: {artifacts_dir}")

    paths = sorted(artifacts_dir.glob("*_events.jsonl"))
    if not paths:
        raise ExtractionError(
            f"no *_events.jsonl under {artifacts_dir} -- point this at a run's "
            "`artifacts/` directory, not the run root"
        )

    stats = ExtractionStats()
    segments: list[LevelSegment] = []
    for path in paths:
        game_id = game_id_from_path(path)
        segments.extend(segments_from_events(game_id, iter_events(path, stats), stats))

    return [s for s in segments if len(s) >= min_transitions], stats
