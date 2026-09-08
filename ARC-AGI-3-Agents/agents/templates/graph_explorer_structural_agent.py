"""`GraphExplorerAgent` layered with a small, purpose-trained structural
model that predicts "how many hops to a known win from this exact state,"
used to bias travel-mode routing only.

**Why this is scoped differently from `graph_explorer_jepa_agent.py`.**
That class biased BOTH explore-mode (which untested action to try) and
travel-mode (which known edge to route through) using this project's
general-purpose JEPA world model (InfoGain/value head), and was a clear,
decisive regression (34 vs 46 levels, see that module's own docstring and
CLAUDE.md). Two things are different here, on purpose:
1. Only travel-mode is touched. Explore-mode stays exactly the pure
   agent's uniform-random behavior -- the leading hypothesis for why the
   JEPA variant regressed was that biasing *explore* order changes *when*
   within an episode a GAME_OVER-triggering action gets hit, front-loading
   resets. Travel-mode routing doesn't select untested/risky actions at
   all -- every travel-mode candidate is an edge already confirmed
   successful, so re-ranking among them can't introduce a new failure
   mode of that kind.
2. The scoring signal is a small model purpose-trained on structural
   graph features (out-degree, in-degree, visit count, BFS depth from
   start, per-node candidate richness) to predict hops-to-nearest-known-
   win (scripts/train_graph_win_distance_model.py, cross-validated by
   held-out game) -- not a reused general-purpose value head whose own
   training signal was documented (Stage 5) as barely distinguishable
   from a zero baseline.

**Live feature computation reads GraphExplorer's own already-maintained
state directly** (`self.graph_explorer._G`/`_G_rev`/`_nodes`) rather than
keeping a second, independently-updated graph -- it's already
authoritative and in sync by the time `tie_break_fn` fires each decision
(`_choose_action_inner` records the previous transition's outcome before
calling `choose_edge` for the next one). Two features
(`num_available_actions`, `action6_available`) are approximated from
`NodeInfo.total_candidates` rather than the exact game-level
`available_actions` list scripts/harvest_graph_win_distance_data.py used
offline (that list isn't available for a node we aren't currently
standing on) -- a documented simplification, not a silent one.

If the trained model isn't available (not yet trained, or failed to
load), this degrades gracefully to exactly `GraphExplorerAgent`'s own
already-verified behavior (`tie_break_fn` stays unset).
"""

import logging
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .graph_explorer_agent import GraphExplorerAgent

logger = logging.getLogger()

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODEL_DIR = _REPO_ROOT / "checkpoints_graph_win_distance"

FEATURE_NAMES = [
    "out_degree",
    "in_degree",
    "visit_count",
    "depth",
    "num_available_actions",
    "action6_available",
]


class GraphExplorerStructuralAgent(GraphExplorerAgent):
    """See module docstring. Registered as `graphexplorerstructuralagent`."""

    # Same z-normalized-softmax tie-break pattern and rationale as
    # GraphExplorerJepaAgent.TIE_BREAK_TEMPERATURE -- not swept.
    TIE_BREAK_TEMPERATURE = 0.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_ready = False
        self._visit_counts: dict[str, int] = {}
        self._start_state: str | None = None
        try:
            self._init_model()
            self.graph_explorer.tie_break_fn = self._structural_tie_break
            self._model_ready = True
        except Exception:
            logger.exception(
                f"{self.game_id} - GraphExplorerStructuralAgent: model init "
                "failed, falling back to plain GraphExplorerAgent behavior"
            )

    def _init_model(self) -> None:
        import joblib
        model_path = _MODEL_DIR / "win_distance_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"no trained model at {model_path}")
        self._model = joblib.load(model_path)

    def _compute_depth_map(self) -> dict[str, int]:
        """BFS depth from this episode's start state, over GraphExplorer's
        own forward adjacency (`_G`) -- recomputed fresh each call rather
        than tracked incrementally; graphs at this project's action
        budgets (hundreds to low thousands of nodes) make a full BFS per
        decision cheap, and correctness is worth more than the savings
        here for a first cheap test."""
        depth: dict[str, int] = {}
        if self._start_state is None:
            return depth
        depth[self._start_state] = 0
        dq = deque([self._start_state])
        while dq:
            cur = dq.popleft()
            d = depth[cur]
            for (_edge_idx, nxt) in self.graph_explorer._G.get(cur, ()):
                if nxt not in depth:
                    depth[nxt] = d + 1
                    dq.append(nxt)
        return depth

    def _structural_features_for(self, state: str, depth_map: dict[str, int]) -> list[float]:
        node_info = self.graph_explorer._nodes.get(state)
        out_degree = len(self.graph_explorer._G.get(state, ()))
        in_degree = len(self.graph_explorer._G_rev.get(state, ()))
        visit_count = self._visit_counts.get(state, 0)
        depth = depth_map.get(state, 0)
        total_candidates = node_info.total_candidates if node_info is not None else 0
        num_available_actions = total_candidates
        action6_available = int(total_candidates > 5)
        return [out_degree, in_degree, visit_count, depth, num_available_actions, action6_available]

    def _weighted_sample(self, edge_scores: list[tuple[int, float]]) -> int:
        """Same z-normalized temperature-softmax pattern as
        GraphExplorerJepaAgent._weighted_sample -- see that class's own
        docstring for why a hard argmax/argmin is the wrong default here
        (this project has hit the "deterministic tie-break on a flat/
        near-flat signal defaults to the same handful of indices" bug
        three times already in other contexts)."""
        if len(edge_scores) == 1:
            return edge_scores[0][0]
        scores = np.array([s for _, s in edge_scores], dtype=np.float64)
        std = scores.std()
        if std < 1e-9:
            return edge_scores[random.randrange(len(edge_scores))][0]
        z = (scores - scores.mean()) / std
        logits = z / self.TIE_BREAK_TEMPERATURE
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        idx = np.random.choice(len(edge_scores), p=weights)
        return edge_scores[idx][0]

    def _structural_tie_break(self, node, mode: str, candidates: list) -> int:
        self._visit_counts[node] = self._visit_counts.get(node, 0) + 1
        if self._start_state is None:
            self._start_state = node

        if mode != "travel" or not self._model_ready or len(candidates) <= 1:
            return random.choice(candidates)

        try:
            node_info = self.graph_explorer._nodes[node]
            depth_map = self._compute_depth_map()
            scored = []
            for edge_idx in candidates:
                target_hash = str(node_info.edge_data["target"][edge_idx])
                if not target_hash:
                    continue
                feats = self._structural_features_for(target_hash, depth_map)
                pred_hops = float(self._model.predict([feats])[0])
                scored.append((edge_idx, -pred_hops))  # lower predicted hops = better

            if not scored:
                return random.choice(candidates)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.game_id} - tie_break[travel] scored={scored} of {len(candidates)} candidates")
            return self._weighted_sample(scored)
        except Exception:
            logger.exception(
                f"{self.game_id} - GraphExplorerStructuralAgent: tie_break scoring "
                f"failed (mode={mode}), falling back to random choice for this decision"
            )
            return random.choice(candidates)
