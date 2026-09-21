"""Rule-based HUD / status-bar detection over ``segment_layer`` nodes.

This file is the **single source of truth** for the HUD arm's detector. It is
spliced verbatim into the solver bundle's
``inference/utils/segmentation.py`` by ``scripts/_build_hud_bundle.py``, between
the ``_HUD_SPLICE_BEGIN`` / ``_HUD_SPLICE_END`` markers below, so the code the
tests exercise and the code that ships to Kaggle cannot drift apart.

Constraints inherited from the splice target (``segmentation.py``'s own
docstring): **standard library only, no project imports, no ``from __future__``
import**. That module's source is injected into the Python-tool sandbox via
``inspect.getsource``, where project packages are not importable.

Provenance
----------
The rules are a port of ``identify_status_bars_with_rule`` and its helpers from
Rudakov, Shock & Cowley, *"Graph-Based Exploration for ARC-AGI-3 Interactive
Reasoning Tasks"* (AAAI 2026 Workshop on AI for Scientific Research,
arXiv:2512.24156), MIT-licensed. This repo already carries that work at
``ARC-AGI-3-Agents/agents/templates/graph_explorer_agent.py`` with the required
notice at ``ARC-AGI-3-Agents/agents/templates/graph_explorer_THIRD_PARTY_LICENSE``.
The thresholds (edge distance 3, aspect ratio 5, twin count 3) are theirs,
unchanged.

Two deliberate differences from the upstream implementation, both recorded
because they change behaviour:

1. **Twins are matched by the bundle's translation-invariant ``hash``**, not by
   upstream's ``(area, is_rectangle, color)`` triple. Equal hash means equal
   normalized cell set *and* equal color, which implies equal area and equal
   rectangle-ness -- so hash-twins are a strict **subset** of upstream twins.
   The detector therefore fires *less* often than upstream's, never more. That
   is the safe direction for an advisory flag.
2. **Bounding boxes come from a node's ``boundary``** (the traced outer contour
   reduced to corner points) rather than from the raw cell set, because
   ``boundary`` is what the node dict exposes. Every bbox-extreme cell lies on
   the outer perimeter and on a straight run whose endpoints are corners, so
   this is exact; ``tests/test_hud_detect.py`` checks it against an independent
   flood-fill bbox on every frame in the fixture.

The upstream ``left``/``top`` tests (``max < 3``) and ``right``/``bottom`` tests
(``min > extent - 3``) are **not symmetric**: the former admit depths 0-2, the
latter only 0-1. That asymmetry is preserved verbatim rather than "fixed",
because it is the code that was validated against real frames from the
champion run, and correcting it would only make the detector fire *more*.
"""

# _HUD_SPLICE_BEGIN  (everything below this marker is copied into segmentation.py)

# Upstream thresholds (arXiv:2512.24156, MIT). Do not retune without re-running
# the edge-depth validation -- these are what bound the flag to the frame border.
_HUD_EDGE_DISTANCE = 3
_HUD_RATIO_THRESHOLD = 5
_HUD_TWINS_THRESHOLD = 3


def _hud_bbox(boundary):
    """(min_row, min_col, max_row, max_col) of a node's boundary corner points."""
    rows = [p[0] for p in boundary]
    cols = [p[1] for p in boundary]
    return (min(rows), min(cols), max(rows), max(cols))


def _hud_edges(bbox, height, width):
    """Which frame edges this bbox sits *fully* against, upstream's rule verbatim.

    'left'/'top' require the whole bbox within `_HUD_EDGE_DISTANCE` of the edge;
    'right'/'bottom' use upstream's stricter `min > extent - distance` form.
    """
    min_row, min_col, max_row, max_col = bbox
    edges = []
    if max_col < _HUD_EDGE_DISTANCE:
        edges.append("left")
    if min_col > width - _HUD_EDGE_DISTANCE:
        edges.append("right")
    if max_row < _HUD_EDGE_DISTANCE:
        edges.append("top")
    if min_row > height - _HUD_EDGE_DISTANCE:
        edges.append("bottom")
    return edges


def _hud_is_long(bbox, direction):
    """Is the bbox a long thin bar, in an orientation consistent with its edge?"""
    min_row, min_col, max_row, max_col = bbox
    col_extent = max_col - min_col + 1
    row_extent = max_row - min_row + 1
    ratio = col_extent / row_extent
    if ratio >= _HUD_RATIO_THRESHOLD and direction in ("any", "horizontal"):
        return True
    if ratio <= 1.0 / _HUD_RATIO_THRESHOLD and direction in ("any", "vertical"):
        return True
    return False


def detect_hud_nodes(nodes, height, width):
    """Advisory ids of nodes that look like HUD / status-bar chrome.

    Two shapes qualify, both of which must sit fully against a frame edge:
      1. a **line** -- aspect ratio >= 5:1 in the orientation that edge implies;
      2. **dots** -- >= 3 identical objects (same ``hash``) on the same edge.

    Returns a sorted list of node ids. Never raises on well-formed nodes, and
    never inspects anything but the node dicts it was handed: this is a pure
    read-side annotation and removes nothing.
    """
    if not nodes or height <= 0 or width <= 0:
        return []

    bboxes = [_hud_bbox(node["boundary"]) for node in nodes]
    edges_of = [_hud_edges(bbox, height, width) for bbox in bboxes]

    by_hash = {}
    for i, node in enumerate(nodes):
        by_hash.setdefault(node["hash"], []).append(i)

    checked = set()
    hud = set()
    for i, node in enumerate(nodes):
        if i in checked:
            continue
        checked.add(i)
        edges = edges_of[i]
        if not edges:
            continue

        directions = []
        if "left" in edges or "right" in edges:
            directions.append("vertical")
        if "top" in edges or "bottom" in edges:
            directions.append("horizontal")
        direction = "any" if len(directions) == 2 else directions[0]

        if _hud_is_long(bboxes[i], direction):
            hud.add(i)
            continue

        # Not a line: fall back to the "dots" rule. A twin counts only if it is
        # itself on one of the same edges. Upstream consumes twins so they are
        # not re-evaluated as group leaders later; preserved.
        edge_set = set(edges)
        twins = [
            j
            for j in by_hash.get(node["hash"], ())
            if j != i and edge_set.intersection(edges_of[j])
        ]
        for j in twins:
            checked.add(j)
        if len(twins) + 1 < _HUD_TWINS_THRESHOLD:
            continue
        hud.add(i)
        hud.update(twins)

    return sorted(hud)

# _HUD_SPLICE_END
