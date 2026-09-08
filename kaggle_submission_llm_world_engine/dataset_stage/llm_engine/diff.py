"""Compact diff representation between two frames. Used both to keep LLM
prompts small (cheaper than pasting two full 64x64 grids) and to describe
prediction failures concretely -- see architecture.md's "text/JSON, not
images" and "validate by replay" sections.
"""

from __future__ import annotations

from typing import Optional

from .types import Grid


def grid_shape_mismatch(a: Grid, b: Grid) -> Optional[str]:
    if len(a) != len(b):
        return f"layer count differs: {len(a)} vs {len(b)}"
    for i, (la, lb) in enumerate(zip(a, b)):
        if len(la) != len(lb) or (la and lb and len(la[0]) != len(lb[0])):
            return f"layer {i} shape differs: {len(la)}x{len(la[0]) if la else 0} vs {len(lb)}x{len(lb[0]) if lb else 0}"
    return None


def diff_cells(before: Grid, after: Grid, max_cells: int = 200) -> list[tuple[int, int, int, int, int]]:
    """Return (layer, x, y, old_value, new_value) for every changed cell.

    Truncated at max_cells with the count noted by the caller if a game
    legitimately changes more cells than that in one step (e.g. a full
    screen wipe) -- rare, but shouldn't blow up prompt size when it happens.
    """
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
    64 characters instead of a much longer comma-separated list."""
    digits = "0123456789abcdef"
    return "\n".join(
        "".join(digits[v] if 0 <= v <= 15 else "?" for v in row)
        for row in grid[layer]
    )
