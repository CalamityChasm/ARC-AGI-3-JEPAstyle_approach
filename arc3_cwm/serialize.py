"""Compact, lossless export of level segments.

The backtest runs on Kaggle (the model is a sealed appliance and the local
box cannot host it) but the recorded play lives here, so the segments have
to travel as a dataset. Shipping raw `frame_before`/`frame_after` pairs
would mean two full 64x64 grids per step -- ~16M integers across a run.

Instead each segment carries its opening grid once plus, per step, the
cells that changed. Grids are reconstructed by applying the diffs in
order, which is exact: `load_segments(dump_segments(x)) == x` is asserted
in the tests, cell for cell, rather than assumed.

Correctness matters more than size here. A lossy round-trip would shift
the boards the model is asked about without changing anything visible in
the output -- the same silent-corruption shape as this project's
`action_input` bug.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ._engine import Action, Transition
from .extract import LevelSegment

FORMAT_VERSION = 1


def _diff(before: list[list[int]], after: list[list[int]]) -> list[list[int]]:
    return [
        [x, y, new]
        for y, (row_before, row_after) in enumerate(zip(before, after))
        for x, (old, new) in enumerate(zip(row_before, row_after))
        if old != new
    ]


def _apply(grid: list[list[int]], diff: Sequence[Sequence[int]]) -> list[list[int]]:
    out = [list(row) for row in grid]
    for x, y, value in diff:
        out[y][x] = value
    return out


def segment_to_dict(segment: LevelSegment) -> dict[str, Any]:
    if not segment.transitions:
        raise ValueError(f"{segment.key}: refusing to serialise an empty segment")

    steps = []
    previous_after: list[list[int]] | None = None

    for t in segment.transitions:
        step: dict[str, Any] = {
            "a": t.action.name,
            "x": t.action.x,
            "y": t.action.y,
            "d": _diff(t.frame_before[0], t.frame_after[0]),
            "lb": t.levels_completed_before,
            "la": t.levels_completed_after,
            "s": t.state_after,
        }
        # A segment's chain is NOT always continuous: the extractor drops
        # RESET steps and coordinate-less ACTION6 steps, so this step's
        # `frame_before` may not be the previous step's `frame_after` (16
        # RESETs occur in the 2026-09-21 run alone). Chaining diffs
        # blindly across such a gap silently reconstructs the wrong board
        # -- so a discontinuity carries a full resync grid instead.
        if previous_after is not None and t.frame_before[0] != previous_after:
            step["g"] = t.frame_before[0]
        steps.append(step)
        previous_after = t.frame_after[0]

    return {
        "game_id": segment.game_id,
        "level": segment.level,
        "cleared_level": segment.cleared_level,
        "boundary_cells_changed": segment.boundary_cells_changed,
        "opening": segment.transitions[0].frame_before[0],
        "steps": steps,
    }


def segment_from_dict(payload: dict[str, Any]) -> LevelSegment:
    grid = [list(row) for row in payload["opening"]]
    transitions: list[Transition] = []

    for step in payload["steps"]:
        # `g` marks a discontinuity (a dropped RESET, say) -- resync to the
        # stored board rather than chaining a diff across the gap.
        if "g" in step:
            grid = [list(row) for row in step["g"]]
        after = _apply(grid, step["d"])
        transitions.append(
            Transition(
                frame_before=[grid],
                action=Action(name=step["a"], x=step["x"], y=step["y"]),
                frame_after=[after],
                levels_completed_before=step["lb"],
                levels_completed_after=step["la"],
                state_after=step["s"],
            )
        )
        grid = after

    return LevelSegment(
        game_id=payload["game_id"],
        level=payload["level"],
        transitions=transitions,
        cleared_level=payload.get("cleared_level", False),
        boundary_cells_changed=payload.get("boundary_cells_changed"),
    )


def dump_segments(segments: Iterable[LevelSegment], path: Path, source: str = "") -> Path:
    """Write segments to a gzipped JSON file."""
    path = Path(path)
    payload = {
        "format_version": FORMAT_VERSION,
        "source": source,
        "segments": [segment_to_dict(s) for s in segments],
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return path


def load_segments(path: Path) -> list[LevelSegment]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        payload = json.load(fh)

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version {version!r}, expected {FORMAT_VERSION} -- "
            "refusing to guess at an unknown layout"
        )
    return [segment_from_dict(s) for s in payload["segments"]]
