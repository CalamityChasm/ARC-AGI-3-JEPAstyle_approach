"""Stage 3 (plan.md): exact (state, action) -> next_state transition graph.

A lossless, deterministic-lookup companion to the *learned* predictor.
ARC-AGI-3 games are turn-based and deterministic (the same state + the
same action always produces the same next state), so any transition seen
once can be recalled exactly forever -- no compounding rollout error, no
wasted re-exploration of something already fully explained. This is
intentionally not a learned component: a plain dict keyed on hashed exact
frame content.

Unlike the learned predictor's coarse 8x8-patch salience map (fine for
*ranking* candidate click regions when the outcome is unknown), ACTION6
entries here are keyed on the *exact* (x, y) pixel -- once we've actually
observed what a specific click does, there's no reason to blur that back
down to patch granularity.
"""

import hashlib
from dataclasses import dataclass


def _hash_frame(frame: list) -> str:
    """Hash a FrameData.frame (list of one (64, 64) grid layer) to a
    compact, exact key -- same content always hashes the same; this only
    needs to distinguish game states, not be cryptographically secure."""
    layer = frame[0]
    flat = bytes(v for row in layer for v in row)
    return hashlib.blake2b(flat, digest_size=16).hexdigest()


@dataclass
class TransitionRecord:
    next_state_key: str
    levels_completed_delta: int
    visits: int = 1


class TransitionGraph:
    """(state_key, action_id, xy) -> TransitionRecord, built up during play.

    `xy` is `None` for simple (non-ACTION6) actions.
    """

    def __init__(self) -> None:
        self._edges: dict[tuple[str, int, tuple[int, int] | None], TransitionRecord] = {}
        # state_key -> (action_id, xy, levels_completed_delta) of the best
        # known outcome from that exact state, so a Memory agent can
        # instantly recall "what worked here" without scanning every edge.
        self._best_action_from: dict[str, tuple[int, tuple[int, int] | None, int]] = {}

    @staticmethod
    def key_for(frame: list) -> str:
        return _hash_frame(frame)

    def record(
        self,
        state_key: str,
        action_id: int,
        xy: tuple[int, int] | None,
        next_frame: list,
        levels_completed_delta: int,
    ) -> None:
        next_key = _hash_frame(next_frame)
        edge_key = (state_key, action_id, xy)
        existing = self._edges.get(edge_key)
        if existing is not None:
            existing.visits += 1
            existing.next_state_key = next_key
            existing.levels_completed_delta = levels_completed_delta
        else:
            self._edges[edge_key] = TransitionRecord(next_key, levels_completed_delta)

        best = self._best_action_from.get(state_key)
        if best is None or levels_completed_delta > best[2]:
            self._best_action_from[state_key] = (action_id, xy, levels_completed_delta)

    def lookup(
        self, state_key: str, action_id: int, xy: tuple[int, int] | None
    ) -> TransitionRecord | None:
        return self._edges.get((state_key, action_id, xy))

    def tried_actions(self, state_key: str) -> set[tuple[int, "tuple[int, int] | None"]]:
        """Every (action_id, xy) pair already tried from this exact state."""
        return {
            (a, xy) for (s, a, xy) in self._edges if s == state_key
        }

    def best_known_action(
        self, state_key: str
    ) -> tuple[int, "tuple[int, int] | None", int] | None:
        """The best (highest levels_completed_delta) action ever observed
        from this exact state, or `None` if this state has never been seen."""
        return self._best_action_from.get(state_key)

    def seen(self, state_key: str) -> bool:
        return state_key in self._best_action_from

    def __len__(self) -> int:
        return len(self._edges)
