"""Compact diff representation between two frames. Used both to keep LLM
prompts small (cheaper than pasting two full 64x64 grids) and to describe
prediction failures concretely -- see architecture.md's "text/JSON, not
images" and "validate by replay" sections.

**Everything in this module is a total function.** These are diagnostic
formatters, and at least one of their two arguments is routinely the raw
return value of LLM-authored `predict()` code (see replay._check_one),
which is under no obligation to be a well-formed grid. Before
2026-09-07 they assumed well-formed input and raised on anything else:
a real scored run died with `TypeError: object of type 'int' has no
len()` at what was then line 18 of this file, taking a whole game thread
down with it (see experiments/stage7_codeworld_fixes.md). A malformed
grid must produce a *description of how it is malformed* -- which is
exactly the feedback the repair prompt wants -- never an exception.
"""

from __future__ import annotations

from typing import Any, Optional

from .types import Grid


def describe_malformed(grid: Any, label: str = "grid") -> Optional[str]:
    """Return None if `grid` is a well-formed Grid (list of layers, each a
    list of rows, each a list of ints), else a short human/LLM-readable
    description of the first thing wrong with it.

    Deliberately checks structure only, not size -- a 8x8 grid is
    well-formed even though real frames are 64x64, because the shape
    comparison against the real frame is what catches size errors, with a
    much more useful message.
    """
    if not isinstance(grid, list):
        return f"{label} is {type(grid).__name__}, expected a list of layers"
    if not grid:
        return f"{label} is an empty list (no layers)"
    for i, layer in enumerate(grid):
        if not isinstance(layer, list):
            return f"{label} layer {i} is {type(layer).__name__}, expected a list of rows"
        if not layer:
            return f"{label} layer {i} is empty (no rows)"
        for j, row in enumerate(layer):
            if not isinstance(row, list):
                return (
                    f"{label} layer {i} row {j} is {type(row).__name__}, expected a "
                    f"list of ints -- the grid looks like it is missing a nesting "
                    f"level (a grid is [layer][row][col], not [row][col])"
                )
            for k, cell in enumerate(row):
                if not isinstance(cell, int) or isinstance(cell, bool):
                    return (
                        f"{label} layer {i} row {j} col {k} is "
                        f"{type(cell).__name__}, expected an int"
                    )
    return None


def _row_len(layer: list, index: int = 0) -> int:
    """Length of a layer's first row, 0 for an empty layer. Only ever
    called on structurally-validated layers."""
    return len(layer[index]) if layer else 0


def grid_shape_mismatch(a: Grid, b: Grid) -> Optional[str]:
    """Describe how `a` and `b` differ in shape, or None if they match.

    Malformed input is reported as a mismatch rather than raised -- see
    the module docstring.
    """
    for grid, label in ((a, "first grid"), (b, "second grid")):
        malformed = describe_malformed(grid, label)
        if malformed is not None:
            return malformed

    if len(a) != len(b):
        return f"layer count differs: {len(a)} vs {len(b)}"
    for i, (la, lb) in enumerate(zip(a, b)):
        if len(la) != len(lb) or _row_len(la) != _row_len(lb):
            return (
                f"layer {i} shape differs: {len(la)}x{_row_len(la)} "
                f"vs {len(lb)}x{_row_len(lb)}"
            )
    return None


def diff_cells(before: Grid, after: Grid, max_cells: int = 200) -> list[tuple[int, int, int, int, int]]:
    """Return (layer, x, y, old_value, new_value) for every changed cell.

    Truncated at max_cells with the count noted by the caller if a game
    legitimately changes more cells than that in one step (e.g. a full
    screen wipe) -- rare, but shouldn't blow up prompt size when it happens.

    Returns [] rather than raising if either argument is malformed;
    callers that care about *why* should call grid_shape_mismatch first
    (format_diff below does).
    """
    if describe_malformed(before) is not None or describe_malformed(after) is not None:
        return []
    changes: list[tuple[int, int, int, int, int]] = []
    for layer_idx, (layer_before, layer_after) in enumerate(zip(before, after)):
        for y, (row_before, row_after) in enumerate(zip(layer_before, layer_after)):
            for x, (old, new) in enumerate(zip(row_before, row_after)):
                if old != new:
                    changes.append((layer_idx, x, y, old, new))
                    if len(changes) >= max_cells:
                        return changes
    return changes


def format_diff(before: Grid, after: Grid, max_cells: int = 200) -> str:
    mismatch = grid_shape_mismatch(before, after)
    if mismatch:
        return f"<shape mismatch: {mismatch}>"
    changes = diff_cells(before, after, max_cells=max_cells)
    if not changes:
        return "<no cells changed>"
    lines = [f"({layer},{x},{y}): {old}->{new}" for layer, x, y, old, new in changes]
    return ", ".join(lines)


def format_grid(grid: Grid, layer: int = 0) -> str:
    """Compact text rendering of one layer -- one row per line, cells as
    single hex-ish digits (0-15 -> '0'-'9','a'-'f') so a 64-wide row stays
    64 characters instead of a much longer comma-separated list.

    Total, like everything else here: a malformed or short grid renders as
    a `<...>` description rather than raising IndexError/TypeError.
    """
    malformed = describe_malformed(grid)
    if malformed is not None:
        return f"<unrenderable: {malformed}>"
    if layer >= len(grid):
        return f"<unrenderable: layer {layer} requested but grid has {len(grid)} layer(s)>"
    digits = "0123456789abcdef"
    return "\n".join(
        "".join(digits[v] if 0 <= v <= 15 else "?" for v in row)
        for row in grid[layer]
    )
