"""Exact-state exploration graph, ported from Evgenii Rudakov, Ryan Shock,
and Nathan Cowley's "Graph-Based Exploration for ARC-AGI-3 Interactive
Reasoning Tasks" (arXiv:2512.24156, AAAI 2026 Workshop on AI for Scientific
Research) -- a training-free graph exploration agent that placed 3rd on the
ARC-AGI-3 Preview Challenge private leaderboard.

Source: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
(`graph_explorer.py`, commit including their post-evaluation RESET-loop fix,
aad8145). Used under the MIT License -- see
`graph_explorer_THIRD_PARTY_LICENSE` in this directory for the exact license
text and copyright notice required by that license's terms.

This module is a faithful, near-verbatim port of their `GraphExplorer`/
`NodeInfo` classes: exact state hashing (owned by the agent, not this
module), a priority-tiered untested-edge queue per node, and BFS-maintained
shortest-path distances from every explored node to the nearest "frontier"
node (one still holding an untested edge at or below the current priority
threshold) -- so action selection always has a concrete target to route
toward, not just a local ranking. Stripped from the original: the
matplotlib-based grid-world demo/visualization code (`_plot_grid`,
`_visualize_grid`, `run_grid_demo`, the `__main__` smoke tests) -- none of
that participates in real gameplay, only in the paper's own illustrative
grid-world figures. Algorithm logic (`record_test`, `choose_edge`,
`_rebuild_distances`, `_maybe_advance_group`, `_close_node`, `_add_new_node`)
is unchanged from the source.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Set, Tuple
import random
import numpy as np

INFINITY = np.iinfo(np.int32).max

# NOTE: all data formats here chosen crudely, to be optimized later (upstream note)
edge_dtype = np.dtype([
    ("group", "i4"),  # 0-indexed group id
    ("result", "i4"),  # 1 if success, -1 if failed, 0 if not tested yet
    ("target", "U32"),  # target node hash-name, "" if not tested or failed
    ("distance", "i4"),  # distance to the frontier node, 0 means next node is the frontier
    ("errors", "i4"),  # number of errors so far
])


def format_struct_table(arr):
    names = ("idx",) + arr.dtype.names
    cols = []
    for name in names:
        if name == "idx":
            cols.append([str(i) for i in range(len(arr))])
        else:
            cols.append([str(r[name]) for r in arr])
    widths = [max(len(n), *(len(v) for v in col)) for n, col in zip(names, cols)]
    header = " | ".join(n.ljust(w) for n, w in zip(names, widths))
    sep = "-+-".join("-" * w for w in widths)
    lines = []
    for i in range(len(arr)):
        line = " | ".join(cols[j][i].ljust(widths[j]) for j in range(len(names)))
        lines.append(line)
    return "\n".join([header, sep, *lines])


@dataclass
class NodeInfo:
    name: Hashable

    total_candidates: int  # how many exist
    num_groups: int = 1  # FIXME (upstream): is never used
    active_group: int = 0

    group2remaining_candidate_ids: List[Set[int]] = field(default_factory=list)

    edge_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=edge_dtype))

    error_threshold: int = 3
    closed: bool = False  # flips when last probe done
    distance: float | None = 0  # TODO (upstream): how is it initialized?

    def __post_init__(self):

        assert self.name is not None, "Node name must be provided"

        if self.num_groups > 1 and self.group2remaining_candidate_ids is None:
            raise ValueError("group2remaining_candidate_ids must be provided if num_groups > 1")

        if self.num_groups == 1 and self.group2remaining_candidate_ids is None:
            self.group2remaining_candidate_ids = [set(range(self.total_candidates))]

        self.group2remaining_candidate_ids = [set(r_c_ids) for r_c_ids in self.group2remaining_candidate_ids]

        self.edge_data = np.zeros(self.total_candidates, dtype=edge_dtype)

        for group_id, remaining_candidate_ids in enumerate(self.group2remaining_candidate_ids):
            self.edge_data["group"][list(remaining_candidate_ids)] = group_id

    @property
    def has_open(self) -> bool:
        """Still hiding >=1 untested edge?"""
        return len(self.tested) < self.total_candidates

    def record_test(self, edge_idx: int, success: int, target_node: Hashable | None = None) -> bool:

        edge_group_id = self.edge_data[edge_idx]["group"]

        assert self.edge_data["result"][edge_idx] == 0 and \
            self.edge_data["target"][edge_idx] == "" and \
            self.edge_data["distance"][edge_idx] == 0, \
            "Edge result must be untested before recording a test"

        if success == -1:
            self.edge_data["errors"][edge_idx] += 1
            if self.edge_data["errors"][edge_idx] >= self.error_threshold:
                self.edge_data["errors"][edge_idx] = 0
                new_group_id = edge_group_id + 1
                if new_group_id > self.num_groups - 1:
                    # count it as failed and move on
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
                    self.edge_data["result"][edge_idx] = -1
                    self.edge_data["distance"][edge_idx] = INFINITY
                    return True
                else:
                    self.edge_data["group"][edge_idx] = new_group_id
                    self.group2remaining_candidate_ids[new_group_id].add(edge_idx)
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
            return False

        self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)

        if success == 1:
            self.edge_data["target"][edge_idx] = str(target_node)
            self.edge_data["distance"][edge_idx] = -1  # NOTE: distance is maintained by the GraphExplorer class
            self.edge_data["result"][edge_idx] = 1
        elif success == 0:
            self.edge_data["distance"][edge_idx] = INFINITY
            self.edge_data["result"][edge_idx] = -1

        return True

    def has_open_group(self, group_id: int) -> bool:
        """Return True if this node has at least one untested edge belonging to *group_id* or below."""
        for i in range(group_id + 1):
            if len(self.group2remaining_candidate_ids[i]) > 0:
                return True
        return False

    def __repr__(self) -> str:
        edge_data_repr = format_struct_table(self.edge_data)

        return f"""NodeInfo:
name={self.name},
total_candidates={self.total_candidates},
num_groups={self.num_groups},
distance={self.distance},
closed={self.closed},
{edge_data_repr}
"""


class GraphExplorer:

    def __init__(
        self,
        start_node: Hashable | None = None,
        num_candidates: int | None = None,
        group2remaining_candidate_ids: List[Set[int]] | None = None,
        n_groups: int = 1,
        verbose_level: int = 0,
        tie_break_fn=None,
    ) -> None:

        self._verbose_level = verbose_level
        self._n_groups = max(1, n_groups)
        # Optional (node, mode, candidate_edge_indices) -> edge_idx callable,
        # used by choose_edge in place of random.choice. None (default)
        # preserves the exact upstream uniform-random behavior -- this hook
        # exists so a subclass agent (see graph_explorer_jepa_agent.py) can
        # substitute an informed re-rank *within* whatever candidate set the
        # priority-tier/frontier-distance algorithm already produced,
        # without altering that algorithm itself. A plain mutable attribute
        # (not just a constructor arg) since the agent that wants to use it
        # typically can't build the real scoring function until after its
        # own model-loading step, which happens after this object is
        # constructed -- see GraphExplorerJepaAgent.__init__.
        self.tie_break_fn = tie_break_fn

        self.reset()

    def reset(self) -> None:
        self._nodes: Dict[Hashable, NodeInfo] = {}
        self._G: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set)  # (edge_idx, target_node)
        self._G_rev: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set)  # (edge_idx, source_node)
        self._frontier: Set[Hashable] = set()
        self._dist: Dict[Hashable, int] = {}
        self._next: Dict[Hashable, Tuple[int, Hashable]] = {}  # (edge_idx, target_node)
        self._active_group: int = 0  # current priority group

        self.suspicious_transitions: Dict[Tuple[Hashable, int, Hashable], int] = {}  # (source_node, edge_idx, target_node) -> count
        self.suspicious_transitions_threshold: int = 3

        self._empty = True

    def initialize(self, start_node: Hashable | None = None, num_candidates: int | None = None, group2remaining_candidate_ids: List[Set[int]] | None = None) -> None:

        if start_node is not None:
            self._add_new_node(start_node, num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)

        if self._verbose_level >= 1:
            print(f"\nGraph is initialized with node: {self._nodes[start_node]}")
            self.dump()

    def record_test(
        self,
        node: Hashable,
        edge_idx: Hashable,
        success: bool,
        target_node: Optional[Hashable] = None,
        target_num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
        suspicious_transition: bool = False,
    ) -> None:

        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}")  # TODO (upstream): alternatively, add it to the graph
        node_info = self._nodes[node]

        if node_info.closed:
            if target_node == self._nodes[node].edge_data["target"][edge_idx]:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, skipping test {edge_idx!r}")
                return
            else:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, we perform the test only if the target node is closer to frontier than the original target node. It will allow to fix the broken transition.")
                dist_to_frontier = self._dist.get(target_node, 0)  # 0 if it wasn't previously recorded (so it's in the frontier)
                prev_target_node = self._nodes[node].edge_data["target"][edge_idx]
                prev_dist_to_frontier = self._dist.get(prev_target_node, INFINITY)

                if dist_to_frontier < prev_dist_to_frontier:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is closer to frontier than the original target node {prev_target_node!r}, we perform the test")
                else:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is further from frontier than the original target node {prev_target_node!r}, we skip the test")
                    return

        # store metadata immediately
        if self._verbose_level >= 1:
            print(f"Recording action {edge_idx} from {node} to {target_node} with success {success}")

        if suspicious_transition:
            self.suspicious_transitions[(node, edge_idx, target_node)] = self.suspicious_transitions.get((node, edge_idx, target_node), 0) + 1
            if self._verbose_level >= 1:
                print(f"Suspicious transition detected: {node, edge_idx, target_node}, count: {self.suspicious_transitions[(node, edge_idx, target_node)]}")

            if self.suspicious_transitions[(node, edge_idx, target_node)] < self.suspicious_transitions_threshold:
                if self._verbose_level >= 1:
                    print(f"It will be ignored for now, but will be allowed after {self.suspicious_transitions_threshold} attempts")
                return
            else:
                if self._verbose_level >= 1:
                    print("Transition is recorded as permanent")

        node_info.record_test(edge_idx, success, target_node)

        # successful hop => register edge and maybe discover a brand-new node
        if success == 1:
            if target_node is None:
                raise ValueError("target_node required when success=True")

            if target_node not in self._nodes:
                new_node = True
                if target_num_candidates is None:
                    raise ValueError(
                        "target_num_candidates required for a new node"
                    )
                self._add_new_node(target_node, target_num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)
            else:
                new_node = False

            self._G[node].add((edge_idx, target_node))
            self._G_rev[target_node].add((edge_idx, node))

            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)

            if self._nodes[target_node].has_open_group(self.active_group):
                self._rebuild_distances()
            else:
                self._close_node(target_node)
                self._maybe_advance_group(target_node)

        else:
            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)
                self._maybe_advance_group(node)

        if self._verbose_level >= 1:
            if success == 1:
                success_str = "succeeded"
            elif success == -1:
                success_str = "threw an error"
            else:
                success_str = "failed"

            print(f"\n\nNode {node!r} candidate {edge_idx!r} {success_str}:")
            print(f"Source node:\n{self._nodes[node]}")
            if success == 1:
                print(f"{'NEW' if new_node else 'Existing'} target node:\n{self._nodes[target_node]}")
        self.dump()

    def get_distance(self, node: Hashable) -> Optional[int]:
        d = self._dist.get(node)
        return None if d is None or d == float("inf") else d

    def get_next_hop(self, node: Hashable) -> Optional[Hashable]:
        # NOTE (upstream): DEPRECATED
        if node in self._frontier:
            return node
        nxt = self._next.get(node)
        if nxt is None:
            return None
        if isinstance(nxt, tuple) and len(nxt) == 2:
            return nxt[1]
        return nxt

    def edge_info(self, node: Hashable, edge_idx: Hashable) -> np.ndarray:
        return self._nodes[node].edge_data[edge_idx]

    def is_finished(self) -> bool:
        return not self._frontier

    @property
    def active_group(self) -> int:
        return self._active_group

    @property
    def empty(self) -> bool:
        return self._empty

    def _add_new_node(self, node: Hashable,
                       n_candidates: int,
                       group2remaining_candidate_ids: Optional[List[Set[int]]] = None
                       ) -> None:

        if n_candidates < 1:
            raise ValueError("num_candidates must be positive")

        self._nodes[node] = NodeInfo(node, n_candidates, self._n_groups, group2remaining_candidate_ids=group2remaining_candidate_ids)
        self._G[node] = set()
        self._G_rev[node] = set()

        if self._empty:
            self._empty = False

        if self._nodes[node].has_open_group(self.active_group):
            self._frontier.add(node)
        else:
            self._close_node(node)
            self._maybe_advance_group(node)

    def _close_node(self, node: Hashable) -> None:
        node_info = self._nodes[node]
        if node_info.closed:
            return
        node_info.closed = True
        self._frontier.discard(node)
        self._rebuild_distances()  # removal from frontier may increase some distances in the graph

    def _rebuild_distances(self) -> None:
        """
        Rebuild the distances from the frontier nodes in the graph (multi-source
        BFS from every frontier node, over the reverse edge graph -- this is the
        mechanism that lets `choose_edge` always route toward the *nearest*
        unexplored edge, not just react to a win having already happened once).
        """
        self._dist.clear()
        self._next.clear()
        dq = deque(self._frontier)
        for node, node_info in self._nodes.items():
            node_info.distance = INFINITY
            self._dist[node] = INFINITY
        for src in self._frontier:
            self._nodes[src].distance = 0
            self._dist[src] = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                u_info = self._nodes[u]
                u_dist = self._dist.get(u, INFINITY)
                u_info.edge_data["distance"][edge_idx] = v_dist + 1
                if u_dist > u_info.edge_data["distance"][edge_idx]:
                    u_info.distance = u_info.edge_data["distance"][edge_idx]
                    self._dist[u] = u_info.edge_data["distance"][edge_idx]
                    self._next[u] = (edge_idx, v)
                    dq.append(u)

    def _maybe_advance_group(self, current_node: Hashable) -> None:
        """
        If it's not possible to reach any frontier node from the current node,
        given the current active group, advance to the next higher group id and
        rebuild distances.
        """

        distance = self._nodes[current_node].distance
        while distance == INFINITY and self.active_group < self._n_groups - 1:
            if self._verbose_level >= 1:
                print(f"Node {current_node!r} is not reachable from any frontier node under {self.active_group}, advancing to the next group")

            self._active_group += 1
            self._dist.clear()
            self._next.clear()
            self._frontier.clear()

            for node, node_info in self._nodes.items():
                node_info.active_group = self.active_group
                if node_info.has_open_group(self.active_group):
                    self._frontier.add(node)
                    node_info.closed = False

            self._rebuild_distances()
            distance = self._dist.get(current_node)

    def dump(self) -> None:
        if self._verbose_level >= 1:
            print("=== explorer state ===")
            print("frontier :", self._frontier)
            print("N nodes  :", len(self._nodes))
            print("N edged candidates  :", sum(len(node_info.edge_data) for node_info in self._nodes.values()))
            if self._verbose_level >= 2:
                print("Graph    :", self._G)
                print("dist     :", self._dist)
                print("next hop :", self._next)
            print("======================")

    def print_all_nodes(self) -> None:
        for node_info in self._nodes.values():
            print(node_info)

    def get_candidate_edges(self, node: Hashable) -> tuple[str, list]:
        """Factored out of `choose_edge`: returns (mode, candidate_edge_indices)
        without picking one. `mode` is `"explore"` (untested edges in the
        active priority group -- the normal case) or `"travel"` (already-
        tested, successful edges tied for the lowest known distance to the
        nearest frontier node -- used to navigate when this node has nothing
        left to explore itself). Exposed publicly so `tie_break_fn` can see
        which regime it's being asked to break a tie in."""
        node_info = self._nodes[node]
        if node_info.has_open_group(self.active_group):
            untested_edges = []
            for group_id in range(self.active_group + 1):
                untested_edges.extend(node_info.group2remaining_candidate_ids[group_id])
            if not untested_edges:
                raise ValueError("No untested edges in the current group while the group is open")
            return "explore", untested_edges
        else:
            lowest_dist = node_info.distance
            edges_with_lowest_dist = [edge_idx for edge_idx, edge_data in enumerate(node_info.edge_data) if edge_data["distance"] <= lowest_dist and edge_data["result"] == 1 and edge_data["group"] <= self.active_group]
            return "travel", edges_with_lowest_dist

    def choose_edge(self, node: Hashable, return_reasoning: bool = False) -> Hashable:
        """Hierarchical action selection (paper's Algorithm 1): try an
        untested edge at the current priority group first; if none remain at
        this node, route toward the nearest node that still has one (BFS
        distance maintained incrementally by `_rebuild_distances`); if no
        such node is reachable at all, `_maybe_advance_group` (called from
        `record_test`/`_add_new_node`/`_close_node`) has already widened the
        priority threshold before this is called.

        Picks among the candidate set via `self.tie_break_fn(node, mode,
        candidates)` if one is set, else uniform random (upstream's exact
        behavior) -- either way, *which* candidates are even eligible is
        decided entirely by `get_candidate_edges` above, not by the tie
        break; the coverage-first/frontier-distance discipline itself is
        never overridden.
        """
        mode, candidates = self.get_candidate_edges(node)
        node_info = self._nodes[node]

        if self.tie_break_fn is not None:
            edge_idx = self.tie_break_fn(node, mode, candidates)
            reasoning = f"[{mode}] tie_break_fn chose edge {edge_idx} from candidates {candidates}\n"
        elif mode == "explore":
            edge_idx = random.choice(candidates)
            reasoning = f"Randomly chose untested edge {edge_idx} from group {self.active_group} with {node_info.group2remaining_candidate_ids} group2candidates\n"
        else:
            edge_idx = random.choice(candidates)
            reasoning = f"Chose edge {edge_idx} with lowest dist {node_info.distance}\n"

        reasoning += f"Node info: {node_info}\n"

        if return_reasoning:
            return edge_idx, reasoning
        else:
            return edge_idx
