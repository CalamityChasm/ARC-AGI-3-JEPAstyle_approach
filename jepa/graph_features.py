"""Structural (non-visual, non-learned) features of an exact-state
exploration graph -- shared between offline training-data harvesting
(scripts/harvest_graph_win_distance_data.py) and live agent inference
(graph_explorer_structural_agent.py). Both must compute features
identically, or a model trained on one distribution sees a different one
at inference time.

Zero torch/learned-model dependencies, same convention as jepa/memory.py:
TransitionGraph -- pure adjacency bookkeeping, safe to reuse from a
"no JEPA world model" agent (see graph_explorer_agent.py's own docstring
for why TransitionGraph reuse there doesn't violate that principle; same
reasoning applies here).

This is deliberately a *different*, simpler graph than either
graph_explorer_core.py's GraphExplorer (which tracks priority-group/
frontier-distance bookkeeping tied to live action-candidate enumeration)
or jepa/memory.py's TransitionGraph (which only tracks single-hop best-
known-action). StructuralGraphTracker exists purely to produce a fixed
feature vector per visited state, cheaply, for a learned win-distance
model -- it doesn't drive action selection itself.
"""

FEATURE_NAMES = [
    "out_degree",
    "in_degree",
    "visit_count",
    "depth",
    "num_available_actions",
    "action6_available",
]


class StructuralGraphTracker:
    """Causal (as-of-decision-time) structural bookkeeping for one game's
    exact-state exploration graph, merged across as many episodes/RESETs
    as observe_state/record_edge are called for -- ARC-3 determinism
    means the same exact state reached via different episodes is the
    same graph node, so merging is valid (same assumption jepa/memory.py:
    TransitionGraph and graph_explorer_agent.py's transition_memory both
    already make)."""

    def __init__(self) -> None:
        self.out_edges: dict[str, list[tuple[int, tuple[int, int] | None, str, int]]] = {}
        self.in_degree: dict[str, int] = {}
        self.visit_count: dict[str, int] = {}
        self.depth: dict[str, int] = {}
        self._start_state: str | None = None

    def observe_state(self, state: str) -> None:
        """Call once per decision point, for the state about to act from --
        registers a visit. Depth is fixed on first *discovery* (via
        record_edge, or here for the very first state of the whole
        tracked graph), not on every visit."""
        self.visit_count[state] = self.visit_count.get(state, 0) + 1
        if self._start_state is None:
            self._start_state = state
            self.depth.setdefault(state, 0)

    def record_edge(
        self,
        state: str,
        action_id: int,
        xy: tuple[int, int] | None,
        next_state: str,
        levels_delta: int,
    ) -> None:
        out = self.out_edges.setdefault(state, [])
        for (a, p, _ns, _d) in out:
            if a == action_id and p == xy:
                return  # already recorded (deterministic -- same result)
        out.append((action_id, xy, next_state, levels_delta))
        self.in_degree[next_state] = self.in_degree.get(next_state, 0) + 1
        if next_state not in self.depth:
            self.depth[next_state] = self.depth.get(state, 0) + 1

    def features(
        self,
        state: str,
        num_available_actions: int = 0,
        action6_available: bool = False,
    ) -> dict[str, float]:
        return {
            "out_degree": len(self.out_edges.get(state, [])),
            "in_degree": self.in_degree.get(state, 0),
            "visit_count": self.visit_count.get(state, 0),
            "depth": self.depth.get(state, 0),
            "num_available_actions": num_available_actions,
            "action6_available": int(bool(action6_available)),
        }

    def feature_vector(
        self,
        state: str,
        num_available_actions: int = 0,
        action6_available: bool = False,
    ) -> list[float]:
        f = self.features(state, num_available_actions, action6_available)
        return [f[name] for name in FEATURE_NAMES]


def hops_to_win_labels(
    out_edges: dict[str, list[tuple[int, tuple[int, int] | None, str, int]]],
    max_hops: int = 20,
) -> dict[str, int]:
    """Hindsight-only label computation (uses the FULL, final graph --
    never call this with a causally-restricted graph, and never call it
    from live inference code; it's a training-label generator, mirroring
    how Stage 5's value head used hindsight discounted returns as
    training targets while never being fed future information at
    inference time).

    Reverse BFS from every win-edge's source state (an edge whose
    levels_completed_delta > 0 -- taking it wins immediately, so its
    source state is 0 hops from a win) outward over the reverse adjacency,
    giving every state in the graph its minimum hop-distance to a win.
    States from which no win is reachable within the graph (as actually
    explored) get no entry -- these are excluded from training, not
    assigned a fake sentinel, so `max_hops` only caps how far reverse-BFS
    is willing to look, not the training label range.
    """
    rev: dict[str, set[str]] = {}
    win_sources: set[str] = set()
    for state, edges in out_edges.items():
        for (_a, _p, next_state, delta) in edges:
            rev.setdefault(next_state, set()).add(state)
            if delta > 0:
                win_sources.add(state)

    from collections import deque

    dist: dict[str, int] = {s: 0 for s in win_sources}
    dq = deque(win_sources)
    while dq:
        cur = dq.popleft()
        d = dist[cur]
        if d >= max_hops:
            continue
        for pred in rev.get(cur, ()):
            if pred not in dist:
                dist[pred] = d + 1
                dq.append(pred)
    return dist
