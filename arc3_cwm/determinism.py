"""Is the visible board a sufficient state descriptor?

The whole CodeWorldAgent premise is that `predict(state, action)` can be
a function -- that the same board plus the same action always yields the
same result. If a segment contains two steps with a byte-identical
`frame_before` and an identical action but different outcomes, then no
such function exists for that segment, and **no model of any capability
can pass replay on it**. The engine's contract does allow small hidden
counters on `self`, so such a segment is not strictly unmodelable, but it
does require inferring the existence and dynamics of invisible state from
the visible board alone -- a categorically harder task than inferring a
rule.

This matters for reading the backtest: it is a hard ceiling on the
achievable pass rate that has nothing to do with the model. Measuring it
costs no GPU time and is exact, so it is measured up front and reported
alongside every result rather than discovered afterwards as a surprise.

Measured on the 2026-09-21 anim run: 3 of 64 windowed segments (5%),
across 2 games. The affected games are animation-driven ones, which is
consistent with the anim solver bundle -- the graft worth +20% -- being
specifically animation-aware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ._engine import Transition
from .extract import LevelSegment


def _state_key(t: Transition) -> tuple:
    """Exact, hashable (board, action) key. No checksum, no approximation."""
    return (
        tuple(tuple(row) for row in t.frame_before[0]),
        t.action.name,
        t.action.x,
        t.action.y,
    )


def _outcome_key(t: Transition) -> tuple:
    return (
        tuple(tuple(row) for row in t.frame_after[0]),
        t.levels_delta,
        t.done,
    )


@dataclass
class Contradiction:
    """Two steps that agree on (board, action) and disagree on the result."""

    segment_key: str
    first_index: int
    second_index: int
    action: str
    cells_changed_first: int
    cells_changed_second: int


@dataclass
class DeterminismCensus:
    n_segments: int = 0
    contradictory_segments: list[str] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)

    @property
    def n_contradictory(self) -> int:
        return len(self.contradictory_segments)

    @property
    def affected_games(self) -> set[str]:
        return {key.split("/")[0] for key in self.contradictory_segments}

    @property
    def ceiling(self) -> float:
        """Upper bound on pass rate imposed by the data, not the model."""
        if not self.n_segments:
            return 0.0
        return (self.n_segments - self.n_contradictory) / self.n_segments

    def as_dict(self) -> dict:
        return {
            "n_segments": self.n_segments,
            "n_contradictory_segments": self.n_contradictory,
            "contradictory_segments": list(self.contradictory_segments),
            "n_contradictory_pairs": len(self.contradictions),
            "affected_games": sorted(self.affected_games),
            "ceiling": self.ceiling,
        }

    def summary(self) -> str:
        return (
            f"determinism census: {self.n_contradictory}/{self.n_segments} segments "
            f"({self.n_contradictory / self.n_segments:.0%}) contain a step where an "
            f"identical board and action gave a different result, across "
            f"{len(self.affected_games)} game(s). "
            f"Pass-rate ceiling from the data alone: {self.ceiling:.0%}."
            if self.n_segments
            else "determinism census: no segments"
        )


def census_segment(segment: LevelSegment) -> list[Contradiction]:
    """Every contradictory pair within one segment."""
    found: list[Contradiction] = []
    seen: dict[tuple, tuple[int, tuple]] = {}

    for index, t in enumerate(segment.transitions):
        key = _state_key(t)
        outcome = _outcome_key(t)
        if key in seen:
            first_index, first_outcome = seen[key]
            if first_outcome != outcome:
                found.append(
                    Contradiction(
                        segment_key=segment.key,
                        first_index=first_index,
                        second_index=index,
                        action=str(t.action),
                        cells_changed_first=_cells_changed(
                            segment.transitions[first_index]
                        ),
                        cells_changed_second=_cells_changed(t),
                    )
                )
        else:
            seen[key] = (index, outcome)

    return found


def _cells_changed(t: Transition) -> int:
    return sum(
        1
        for rb, ra in zip(t.frame_before[0], t.frame_after[0])
        for a, b in zip(rb, ra)
        if a != b
    )


def census(segments: Iterable[LevelSegment]) -> DeterminismCensus:
    segments = list(segments)
    result = DeterminismCensus(n_segments=len(segments))
    for segment in segments:
        found = census_segment(segment)
        if found:
            result.contradictory_segments.append(segment.key)
            result.contradictions.extend(found)
    return result


def deterministic_only(segments: Sequence[LevelSegment]) -> list[LevelSegment]:
    """Segments a pure `predict(state, action)` could in principle pass."""
    return [s for s in segments if not census_segment(s)]
