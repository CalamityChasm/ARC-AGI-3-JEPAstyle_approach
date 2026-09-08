"""Shared feature extraction for jepa/click_effect_model.py -- used by
BOTH scripts/harvest_click_effect_data.py (offline pretraining corpus) and
the live agent (ARC-AGI-3-Agents/agents/templates/graph_explorer_learned_agent.py)
so both compute features identically. Same convention as
jepa/graph_features.py's own docstring rationale for this project's other
learned-signal agents.
"""

import numpy as np

PATCH_RADIUS = 3  # matches scripts/diagnose_state_similarity.py
PAD_VALUE = 255


def extract_patch(frame_np: np.ndarray, x: int, y: int, radius: int = PATCH_RADIUS) -> np.ndarray:
    """(2*radius+1, 2*radius+1) uint8 patch centered at (x, y), pad-valued
    at any part that falls outside the frame -- same edge-safety rationale
    as the diagnostic's own _patch helper (an edge-clipped window must
    never silently collide with an interior one)."""
    h, w = frame_np.shape
    size = 2 * radius + 1
    padded = np.full((size, size), PAD_VALUE, dtype=np.uint8)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    py0, px0 = y0 - (y - radius), x0 - (x - radius)
    sub = frame_np[y0:y1, x0:x1]
    padded[py0:py0 + sub.shape[0], px0:px0 + sub.shape[1]] = sub
    return padded


def extract_segment_features(segments: list[dict], segmented_frame: np.ndarray, x: int, y: int) -> np.ndarray:
    """5-dim normalized [color, area, is_rectangle, width, height] for the
    segment at (x, y), or all-zeros if (x, y) isn't inside any tracked
    segment (shouldn't happen for a real click on the board, but frame
    edges / off-by-one cases are handled defensively rather than raising)."""
    h, w = segmented_frame.shape
    if not (0 <= y < h and 0 <= x < w):
        return np.zeros(5, dtype=np.float32)
    seg_id = int(segmented_frame[y, x])
    if seg_id < 0 or seg_id >= len(segments):
        return np.zeros(5, dtype=np.float32)
    seg = segments[seg_id]
    x1, y1, x2, y2 = seg["bounding_box"]
    width, height = x2 - x1 + 1, y2 - y1 + 1
    return np.array([
        seg["color"] / 15.0,
        min(seg["area"], 4096) / 4096.0,
        float(seg["is_rectangle"]),
        width / 64.0,
        height / 64.0,
    ], dtype=np.float32)
