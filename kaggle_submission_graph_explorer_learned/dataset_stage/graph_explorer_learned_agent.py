"""`GraphExplorerAgent` layered with a small, pretrained-then-test-time-
adapted model predicting whether a candidate ACTION6 click will produce
any visible effect -- used to bias EXPLORE-mode ordering (which untested
click to try next), not travel-mode routing.

**Why this is scoped to explore-mode, unlike graph_explorer_structural_
agent.py (which deliberately avoided it).** That earlier design reasoned
explore-mode bias was the likely cause of `graph_explorer_jepa_agent.py`'s
regression (front-loading GAME_OVER-triggering actions). This class
revisits that scope on purpose, at explicit user request, because the
underlying signal is fundamentally different and better-validated this
time:
- `graph_explorer_jepa_agent.py`'s explore-mode signal was InfoGain
  (cross-expert disagreement on the *whole-board* MoE predictor) -- a
  live trace found candidate scores clustered within noise of each other
  (near-flat, not real signal) before that class's own fix.
- This class's signal is `jepa/click_effect_model.py`, trained
  specifically on LOCAL click-effect prediction and directly validated
  (scripts/diagnose_state_similarity.py: local-patch match predicts
  "did anything change" with ~100% consistency in 9/12 games;
  scripts/train_click_effect_model.py's leave-games-out CV: mean AUC
  0.77 on held-out games, not near-chance) before ever touching this
  agent.

The risk this class does NOT eliminate: even a real, non-degenerate bias
in *which* untested action gets tried first can still change *when*
within an episode a GAME_OVER-triggering action gets hit -- that
structural risk is inherent to explore-mode bias, not specific to a weak
signal. This is why the matched backtest against the pure agent is the
real test, not the component-level AUC alone.

**Model + adaptation, mirroring hypothesis_agent.py's TestTimeAdapter
integration:** `jepa/click_effect_model.py`'s ClickEffectModel is loaded
from checkpoints_click_effect/ (pretrained across the local games'
harvested win-adjacent corpus), then `jepa/click_effect_adapter.py`'s
ClickEffectAdapter takes a few real gradient steps on a small subset
(trunk + both heads) using THIS game's own observed ACTION6 outcomes as
they're seen during play -- persists across RESETs of the same game,
never across games (one fresh agent instance per game_id, same as every
other learned-signal agent in this project).

If model construction fails for any reason, this degrades gracefully to
exactly `GraphExplorerAgent`'s own already-verified behavior.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .graph_explorer_agent import GraphExplorerAgent

logger = logging.getLogger()

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.click_effect_adapter import ClickEffectAdapter  # noqa: E402
from jepa.click_effect_features import extract_segment_features  # noqa: E402
from jepa.click_effect_model import ClickEffectModel  # noqa: E402
from jepa.device import get_device  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints_click_effect"


class GraphExplorerLearnedAgent(GraphExplorerAgent):
    """See module docstring. Registered as `graphexplorerlearnedagent`."""

    TIE_BREAK_TEMPERATURE = 0.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_ready = False
        self._pending_click_obs: tuple | None = None  # (frame_np, x, y, seg_features)
        try:
            self._init_model()
            self.graph_explorer.tie_break_fn = self._learned_tie_break
            self._model_ready = True
        except Exception:
            logger.exception(
                f"{self.game_id} - GraphExplorerLearnedAgent: model init "
                "failed, falling back to plain GraphExplorerAgent behavior"
            )

    def _init_model(self) -> None:
        model_path = _CHECKPOINT_DIR / "click_effect_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"no trained model at {model_path}")
        # Debug/isolation lever: forces CPU regardless of what get_device()
        # would pick. Added after confirming this project's only PROVEN
        # torch+enable_gpu:true submission is hypothesis_agent.py (never
        # torch+enable_gpu:false, this class's own original config, which
        # failed 3x for unexplained reasons) -- lets this kernel run under
        # enable_gpu:true (matching the one proven infrastructure profile)
        # while still never touching this project's confirmed-broken P100
        # CUDA compute path (see CLAUDE.md's own gotcha entry).
        if os.getenv("GRAPH_EXPLORER_LEARNED_FORCE_CPU") == "1":
            self.device = torch.device("cpu")
        else:
            self.device = get_device()
        self.model = ClickEffectModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.adapter = ClickEffectAdapter(self.model, self.device)
        # Debug/isolation lever: disables live gradient-step adaptation
        # (predict() still works, observe() just stops short of
        # _adapt_step()) -- same env-var-gated-tunable convention as
        # GRAPH_EXPLORER_MAX_ACTIONS elsewhere in this file. Added to
        # isolate whether the adaptation machinery (real backward()/
        # optimizer.step() calls, the most complex new code here) is
        # implicated in a real-submission-only failure a diagnostic push
        # can't reach, versus the simpler prediction-only path.
        if os.getenv("GRAPH_EXPLORER_LEARNED_ADAPT_DISABLED") == "1":
            self.adapter.enabled = False

    def _segment_features(self, edge_idx: int) -> np.ndarray:
        seg = self._current_frame_segments[edge_idx]
        x1, y1, x2, y2 = seg["bounding_box"]
        width, height = x2 - x1 + 1, y2 - y1 + 1
        return np.array([
            seg["color"] / 15.0,
            min(seg["area"], 4096) / 4096.0,
            float(seg["is_rectangle"]),
            width / 64.0,
            height / 64.0,
        ], dtype=np.float32)

    def _segment_centroid(self, edge_idx: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self._current_frame_segments[edge_idx]["bounding_box"]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _weighted_sample(self, edge_scores: list[tuple[int, float]]) -> int:
        """Same z-normalized temperature-softmax pattern as
        GraphExplorerJepaAgent/GraphExplorerStructuralAgent -- see either
        class's own docstring for why a hard argmax is the wrong default."""
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

    def _record_pending_observation(self, node) -> None:
        """If the previous decision was a click this agent chose, its real
        outcome is now known (node == the resulting hashed_frame, and
        GraphExplorer's own bookkeeping for the previous edge was already
        recorded before choose_edge was called for THIS decision) -- feed
        it to the adapter before scoring the next one."""
        if self._pending_click_obs is None or not self._model_ready:
            return
        prev_frame, x, y, seg_features, prev_hash = self._pending_click_obs
        frame_changed = node != prev_hash
        won = self._last_levels_completed_delta > 0
        try:
            self.adapter.observe(prev_frame, x, y, seg_features, frame_changed, won)
        except Exception:
            logger.exception(f"{self.game_id} - GraphExplorerLearnedAgent: adapter.observe failed")
        self._pending_click_obs = None

    def _learned_tie_break(self, node, mode: str, candidates: list) -> int:
        self._record_pending_observation(node)

        if mode != "explore" or not self._model_ready or len(candidates) <= 1:
            return random.choice(candidates)

        try:
            num_click_actions = self._current_num_click_actions
            scored = []
            click_scores = []
            simple_candidates = []
            for edge_idx in candidates:
                if edge_idx < num_click_actions:
                    x, y = self._segment_centroid(edge_idx)
                    seg_features = self._segment_features(edge_idx)
                    prob = self.adapter.predict(self._current_frame_np, x, y, seg_features)
                    scored.append((edge_idx, prob, (x, y, seg_features)))
                    click_scores.append(prob)
                else:
                    simple_candidates.append(edge_idx)

            # Simple (non-click) actions: this model has nothing to say
            # about them -- give them the neutral (mean-of-clicks, or 0.5
            # if there are no click candidates this decision) score rather
            # than an arbitrary default that would silently bias for/
            # against them relative to click candidates.
            neutral = float(np.mean(click_scores)) if click_scores else 0.5
            for edge_idx in simple_candidates:
                scored.append((edge_idx, neutral, None))

            chosen_idx = self._weighted_sample([(e, s) for e, s, _ in scored])

            chosen_extra = next(extra for e, _s, extra in scored if e == chosen_idx)
            if chosen_extra is not None:
                x, y, seg_features = chosen_extra
                self._pending_click_obs = (self._current_frame_np.copy(), x, y, seg_features, node)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self.game_id} - tie_break[explore] scored={[(e, s) for e, s, _ in scored]}")
            return chosen_idx
        except Exception:
            logger.exception(
                f"{self.game_id} - GraphExplorerLearnedAgent: tie_break scoring "
                f"failed (mode={mode}), falling back to random choice for this decision"
            )
            return random.choice(candidates)
