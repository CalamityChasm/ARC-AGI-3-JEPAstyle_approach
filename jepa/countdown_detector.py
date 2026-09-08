"""Classical-CV (non-learned) detector for a depleting countdown/timer
bar -- the mechanic `scripts/diagnose_countdown_bar_prediction.py`
confirmed the recurrent world model does NOT transfer to held-out games,
despite it plausibly being a common convention across many ARC-3 games:
the model's game-id-conditioned residual has no representation of
"bars near an edge tend to shrink" as an abstraction independent of which
specific game this is, so an unfamiliar game_id gets nothing. A detector
that works directly on raw pixels, with no learned/trained component at
all, sidesteps that failure mode entirely.

Design: watch the four single-pixel-wide edge strips (top row, bottom
row, left column, right column) of the board. On the first observed
frame of an episode, find candidate "bars" -- maximal contiguous runs of
a single color along an edge, long enough to plausibly be a deliberate
UI element rather than noise, excluding whichever color is most common
in the frame overall (background regions abutting an edge produce
spurious same-color runs that have nothing to do with a timer).

On every subsequent frame, two independent checks must both hold or the
candidate is disqualified outright, never reconsidered:
1. **No reversion.** A cell that emptied (changed away from the bar
   color) must never change back to the bar color -- a real timer never
   counts back up.
2. **Contiguous-from-an-edge.** The set of emptied cells must, at every
   observation, form one unbroken block anchored at EITHER end of the
   bar's original span (`0..k` or `n-k..n`) -- exactly the shape a real
   bar/timer depletes in. This is the key discriminator: incidental
   background changes near a board edge (an object wandering past, a
   room boundary shifting) satisfy "never reverts" fairly often over a
   short window by chance, but essentially never satisfy "always exactly
   a contiguous prefix/suffix" too -- validated directly against real
   recordings (see the module's own test comments below): without this
   check, `bp35`/`r11l`/`sp80`/`cd82`/`m0r0` (none of which have a real
   edge countdown bar, confirmed by direct frame inspection in
   `experiments/stage6_bp35_ka59_mechanics.md`) all produced false
   "urgency" signals from background noise; `ka59` (which does have a
   real bar) is the only one that survives this check.

Among surviving candidates, the one with the most confirmed depletion is
reported as the timer.

This is deliberately conservative -- happy to report "no timer detected"
(`urgency() is None`) rather than guess. False positives would corrupt
an agent's behavior; false negatives just mean the agent falls back to
its ordinary behavior, the safe default.
"""

from dataclasses import dataclass, field

import numpy as np

MIN_BAR_FRACTION = 0.25  # a candidate run must span at least this much of its edge


MIN_DEPLETION_EVENTS = 5  # separate steps where the depleted count grew, before trusting this candidate


@dataclass
class _Candidate:
    edge: str  # "top", "bottom", "left", "right"
    color: int
    start: int  # index along the edge where the run begins
    length: int  # original run length
    depleted_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    disqualified: bool = False
    # Number of distinct observation steps where the depleted count grew
    # (not just total depleted count) -- a real per-turn timer deplete
    # in many small, steady increments; a one-off jump from an unrelated
    # object/room change happens in a single event. Requiring several
    # separate events is what actually separates the two in practice
    # (see the module docstring's "no reversion"/"contiguous" checks for
    # why those two alone weren't enough).
    depletion_events: int = 0


def _edge_strip(frame: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return frame[0, :]
    if edge == "bottom":
        return frame[-1, :]
    if edge == "left":
        return frame[:, 0]
    if edge == "right":
        return frame[:, -1]
    raise ValueError(edge)


def _find_runs(strip: np.ndarray, exclude_color: int) -> list[tuple[int, int, int]]:
    """Returns (color, start, length) for every maximal contiguous
    same-color run in `strip` at least MIN_BAR_FRACTION of its length,
    skipping `exclude_color` (the frame's dominant/background color)."""
    n = len(strip)
    min_len = max(2, int(n * MIN_BAR_FRACTION))
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and strip[j] == strip[i]:
            j += 1
        length = j - i
        if length >= min_len and int(strip[i]) != exclude_color:
            runs.append((int(strip[i]), i, length))
        i = j
    return runs


def _is_contiguous_from_edge(mask: np.ndarray) -> bool:
    """True if the True cells in `mask` form one unbroken block anchored
    at either end (a prefix `mask[:k]` or a suffix `mask[-k:]`) -- the
    shape a real depleting bar always has. An all-False mask (nothing
    emptied yet) trivially counts as contiguous."""
    n = len(mask)
    k = int(mask.sum())
    if k == 0:
        return True
    return bool(mask[:k].all()) or bool(mask[-k:].all())


class CountdownBarDetector:
    """Stateful, per-episode. Call `observe(frame)` once per real turn
    (raw (H, W) int grid, NOT the model's tensor encoding); call `reset()`
    on RESET. `urgency()` returns a float in [0, 1] (0 = freshly full,
    1 = fully depleted) for the most-confirmed surviving candidate, or
    `None` if nothing has been confidently identified as a timer yet."""

    def __init__(self) -> None:
        self._candidates: list[_Candidate] = []
        self._initialized = False

    def reset(self) -> None:
        self._candidates = []
        self._initialized = False

    def observe(self, frame) -> None:
        frame = np.asarray(frame)
        if not self._initialized:
            self._init_candidates(frame)
            self._initialized = True
            return
        for cand in self._candidates:
            if cand.disqualified:
                continue
            strip = _edge_strip(frame, cand.edge)
            span = strip[cand.start : cand.start + cand.length]
            now_bar = span == cand.color
            # Disqualify on any cell that was already depleted (not the
            # bar color) reverting back to the bar color -- a real timer
            # never counts back up.
            reverted = cand.depleted_mask & now_bar
            if reverted.any():
                cand.disqualified = True
                continue
            new_mask = cand.depleted_mask | ~now_bar
            # Disqualify unless the emptied region is still a clean
            # prefix/suffix -- the shape a real bar depletes in. Rules
            # out incidental background/object motion near an edge,
            # which reverts rarely but "eats into" the run from random
            # scattered positions, not one consistent end.
            if not _is_contiguous_from_edge(new_mask):
                cand.disqualified = True
                continue
            if new_mask.sum() > cand.depleted_mask.sum():
                cand.depletion_events += 1
            cand.depleted_mask = new_mask

    def _init_candidates(self, frame: np.ndarray) -> None:
        self._candidates = []
        values, counts = np.unique(frame, return_counts=True)
        dominant_color = int(values[np.argmax(counts)])
        for edge in ("top", "bottom", "left", "right"):
            strip = _edge_strip(frame, edge)
            for color, start, length in _find_runs(strip, exclude_color=dominant_color):
                self._candidates.append(
                    _Candidate(edge=edge, color=color, start=start, length=length,
                               depleted_mask=np.zeros(length, dtype=bool))
                )

    def _survivors(self) -> list:
        return [
            c for c in self._candidates
            if not c.disqualified and c.depletion_events >= MIN_DEPLETION_EVENTS
        ]

    def urgency(self) -> float | None:
        survivors = self._survivors()
        if not survivors:
            return None
        best = max(survivors, key=lambda c: c.depleted_mask.sum())
        return float(best.depleted_mask.mean())

    def best_candidate_summary(self) -> dict | None:
        survivors = self._survivors()
        if not survivors:
            return None
        best = max(survivors, key=lambda c: c.depleted_mask.sum())
        return {
            "edge": best.edge,
            "color": best.color,
            "start": best.start,
            "length": best.length,
            "depleted": int(best.depleted_mask.sum()),
            "events": best.depletion_events,
            "urgency": float(best.depleted_mask.mean()),
        }
