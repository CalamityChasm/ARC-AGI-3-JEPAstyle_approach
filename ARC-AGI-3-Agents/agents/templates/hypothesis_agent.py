"""Stage 5 (plan.md): hypothesis bundle + directed action selection.

Builds on Stage 3's `Memory` agent (exact transition-graph recall,
exploit-on-score-delta -- reused unchanged here) and replaces Stage 2/3's
EMA-based "observed surprise" ranking with plan.md's actual Stage 5
design, scoped down to reuse already-trained components rather than a
from-scratch hypothesis-search system:

- **N parallel hypotheses** = Stage 4's K=8 MoE experts (the
  MiniGrid-pretrained checkpoint, `checkpoints/moe_predictor.pt`), each
  treated as one hypothesis about "what a given action does"
  (`jepa/hypothesis_bundle.py`,
  `jepa/models/moe_predictor.py: MoEPredictor.predict_all_experts`).
- **Bayesian confidence update** (architecture.md:
  `p(Hi) ∝ p(Hi) · exp(-prediction_error_i / τ)`) after observing each
  transition's actual outcome -- which experts have been reliable *in
  this game so far*.
- **Entropy of the confidence distribution -> beta_t**: uncertain (still
  disagreeing about which expert to trust) -> explore; confident -> exploit.
- **Q(s,a) = (1-beta)*InfoGain(a) + beta*V(next_state(a))** action
  selection. InfoGain(a) is disagreement across the K experts' raw
  (ungated) predictions for candidate action a -- computed in a *single*
  forward pass per action (the per-patch variance map falls out for free
  and doubles as the ACTION6 click-location salience map, no separate
  64-patch scan needed). V is the decoupled value head
  (`jepa/models/value_head.py`, `jepa/train_value_head.py`), confidence-
  weighted across the K experts' predicted next-states.
- **Experiment-designer opening probes**: at the start of each episode/
  reset, try every simple action once (matching `PressOnce`'s pattern)
  before switching to the Q-driven policy, so the hypothesis bundle has
  some real per-action signal before it starts trusting its own
  confidence weights.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.device import get_device  # noqa: E402
from jepa.grid import CANVAS, PATCH, arc3_frame_to_tensor  # noqa: E402
from jepa.hypothesis_bundle import HypothesisBundle, info_gain  # noqa: E402
from jepa.memory import TransitionGraph  # noqa: E402
from jepa.models import CNNEncoder, MoEPredictor, ValueHead  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
_PATCHES_PER_SIDE = CANVAS // PATCH  # 8


class Hypothesis(Agent):
    """Bayesian hypothesis bundle over MoE experts + InfoGain/value-driven action selection."""

    MAX_ACTIONS = 300
    EXPLOIT_REPEATS = 2
    # Bayesian update temperature -- small, since our latent MSE errors sit
    # around 1e-4 to 1e-2 (see CLAUDE.md's Stage 1/4 numbers); a temperature
    # tuned for O(1) losses would barely move the confidence weights at all.
    TAU = 0.01
    # InfoGain(a) is *predicted* expert disagreement, recomputed fresh each
    # turn from a near-deterministic forward pass -- unlike Curiosity's
    # EMA-of-observed-error ranking, it has no built-in decay when an action
    # keeps getting picked without actually producing useful surprise. Without
    # a random fallback this reproduces Curiosity's own first bug (see
    # CLAUDE.md's Stage 2 history): the single highest-scoring action gets
    # picked every single turn, forever, since nothing about a repeat lowers
    # its score. Same fix, same rate.
    EPSILON = 0.25

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = get_device()
        self._rng = random.Random()

        self.encoder = CNNEncoder().to(self.device)
        self.encoder.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "encoder_moe.pt", map_location=self.device)
        )
        self.encoder.eval()

        game_vocab: dict[str, int] = {}
        vocab_path = _CHECKPOINT_DIR / "game_vocab_moe.json"
        if vocab_path.exists():
            game_vocab = json.loads(vocab_path.read_text())
        self.game_idx = game_vocab.get(self.game_id, 0)
        num_games = max(len(game_vocab), 1)

        self.predictor = MoEPredictor(num_games=num_games, num_experts=8).to(self.device)
        self.predictor.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "moe_predictor.pt", map_location=self.device)
        )
        self.predictor.eval()
        self.num_experts = self.predictor.num_experts

        self.value_head = ValueHead().to(self.device)
        value_path = _CHECKPOINT_DIR / "value_head.pt"
        if value_path.exists():
            self.value_head.load_state_dict(torch.load(value_path, map_location=self.device))
        else:
            logger.warning(
                f"{self.game_id} - no value_head.pt found at {value_path}, "
                "using an untrained value head (V will be ~noise)"
            )
        self.value_head.eval()

        self.graph = TransitionGraph()
        self.hypotheses = HypothesisBundle(num_hypotheses=self.num_experts, tau=self.TAU)

        self._last_levels_completed = 0
        self._exploit_remaining = 0
        self._probe_plan: list[int] = []

        self._prev_feat: torch.Tensor | None = None
        self._prev_action_id: int | None = None
        self._prev_xy: tuple[int, int] | None = None
        self._prev_state_key: str | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @torch.no_grad()
    def _encode(self, latest_frame: FrameData) -> torch.Tensor:
        tensor = arc3_frame_to_tensor(latest_frame.frame)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return self.encoder(x)  # (1, C, 8, 8)

    @torch.no_grad()
    def _predict_experts(
        self, feat: torch.Tensor, action_id: int, xy: tuple[int, int] | None
    ) -> torch.Tensor:
        """(K, C, 8, 8) ungated per-expert predicted next-features for one
        (state, action) pair. A single forward pass -- the per-patch
        variance across the K experts (computed by the caller) doubles as
        both the scalar InfoGain(a) ranking signal and, for ACTION6, the
        spatial click-location salience map, without a separate scan."""
        b = feat.shape[0]
        action_t = torch.full((b,), action_id, dtype=torch.long, device=self.device)
        if xy is not None:
            x, y = xy
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=self.device)
        else:
            xy_t = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
        game_t = torch.full((b,), self.game_idx, dtype=torch.long, device=self.device)
        return self.predictor.predict_all_experts(feat, action_t, xy_t, game_t)[0]  # (K, C, 8, 8)

    def _update_hypotheses(self, feat: torch.Tensor, latest_frame: FrameData) -> None:
        """Records the observed transition into the exact graph, and (if
        an action was actually taken) attributes real prediction error to
        each of the K hypotheses/experts, updating their confidence."""
        if self._prev_feat is None or self._prev_action_id is None:
            return

        if self._prev_state_key is not None:
            levels_delta = latest_frame.levels_completed - self._last_levels_completed
            self.graph.record(
                self._prev_state_key, self._prev_action_id, self._prev_xy, latest_frame.frame, levels_delta
            )

        if self._prev_action_id == GameAction.RESET.value:
            return

        with torch.no_grad():
            expert_preds = self._predict_experts(self._prev_feat, self._prev_action_id, self._prev_xy)
            errors = (expert_preds - feat[0].unsqueeze(0)).pow(2).mean(dim=(1, 2, 3)).cpu()  # (K,)
        self.hypotheses.update(errors)

    def _score_action(
        self, feat: torch.Tensor, action_id: int, beta: float
    ) -> tuple[float, tuple[int, int] | None]:
        """Returns (Q(s, action_id), best_xy) -- best_xy is the
        InfoGain-salient click location if action_id is ACTION6, else None."""
        # Neutral xy for the "which action" scoring pass -- xy conditioning
        # broadcasts uniformly across all spatial positions in this
        # architecture (see MoEPredictor._condition), so it doesn't bias
        # which patches look informative; only which action_id does.
        expert_preds = self._predict_experts(feat, action_id, (32, 32))  # (K, C, 8, 8)

        weights = self.hypotheses.weights.to(self.device)  # (K,)
        with torch.no_grad():
            v_per_expert = self.value_head(expert_preds)  # (K,)
        v = (weights * v_per_expert).sum().item()

        xy = None
        if action_id == GameAction.ACTION6.value:
            patch_var = expert_preds.var(dim=0).mean(dim=0)  # (8, 8), variance across experts per patch
            best_patch = int(patch_var.argmax().item())
            row, col = divmod(best_patch, _PATCHES_PER_SIDE)
            xy = (col * PATCH + PATCH // 2, row * PATCH + PATCH // 2)
            # ig must be computed the same way (mean over experts-variance across
            # every patch/channel) as the non-ACTION6 branch below, or ACTION6
            # wins the cross-action comparison almost by construction -- a max
            # over 64 patches is nearly always larger than a global mean, so an
            # earlier version (`patch_var.max()`) made ACTION6 look more
            # "informative" than any other action regardless of what the experts
            # actually predicted. patch_var.mean() here is mathematically
            # identical to info_gain(expert_preds) (both reduce to
            # var(dim=0).mean() over channels+spatial, just in a different
            # order) -- argmax above still picks the best click location, only
            # the scalar used for cross-action comparison changes.
            ig = patch_var.mean().item()
        else:
            ig = info_gain(expert_preds).item()

        q = (1.0 - beta) * ig + beta * v
        return q, xy

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "hypothesis agent: reset"
            self._last_levels_completed = 0
            self._exploit_remaining = 0
            self._prev_feat = None
            self._prev_action_id = None
            self._prev_xy = None
            self._prev_state_key = None
            # Experiment-designer opening probe: try every simple action
            # once before trusting the hypothesis bundle's own confidence
            # weights, matching PressOnce's "press each action once" idea.
            self._probe_plan = [
                a.value for a in GameAction if a.is_simple() and a is not GameAction.RESET
            ]
            return action

        feat = self._encode(latest_frame)
        self._update_hypotheses(feat, latest_frame)

        if latest_frame.levels_completed > self._last_levels_completed:
            self._exploit_remaining = self.EXPLOIT_REPEATS
        self._last_levels_completed = latest_frame.levels_completed

        state_key = TransitionGraph.key_for(latest_frame.frame)

        if self._exploit_remaining > 0 and self._prev_action_id is not None:
            self._exploit_remaining -= 1
            action_id, xy = self._prev_action_id, self._prev_xy
            action = GameAction.from_id(action_id)
            if action.is_complex() and xy is not None:
                action.set_data({"x": xy[0], "y": xy[1]})
            action.reasoning = (
                f"hypothesis agent: exploiting recent level gain ({self._exploit_remaining} repeats left)"
            )
        else:
            remembered = self.graph.best_known_action(state_key)
            available = latest_frame.available_actions or [
                a.value for a in GameAction if a is not GameAction.RESET
            ]
            if remembered is not None and remembered[2] > 0:
                action_id, xy, _delta = remembered
                action = GameAction.from_id(action_id)
                if action.is_complex() and xy is not None:
                    action.set_data({"x": xy[0], "y": xy[1]})
                action.reasoning = "hypothesis agent: recalling a known winning action from this exact state"
                logger.info(f"{self.game_id} - hypothesis agent: recalling known winning action {action_id} at {xy}")
            elif self._probe_plan and self._probe_plan[0] in available:
                action_id = self._probe_plan.pop(0)
                xy = None
                action = GameAction.from_id(action_id)
                action.reasoning = "hypothesis agent: experiment-designer opening probe"
            elif self._rng.random() < self.EPSILON:
                action_id = self._rng.choice(available)
                xy = None
                action = GameAction.from_id(action_id)
                if action.is_complex():
                    x, y = self._rng.randrange(CANVAS), self._rng.randrange(CANVAS)
                    action.set_data({"x": x, "y": y})
                    xy = (x, y)
                action.reasoning = "hypothesis agent: epsilon-random fallback"
            else:
                beta = self.hypotheses.beta()
                best_q, best_action_id, best_xy = -1e18, available[0], None
                for candidate in available:
                    q, xy_candidate = self._score_action(feat, candidate, beta)
                    if q > best_q:
                        best_q, best_action_id, best_xy = q, candidate, xy_candidate
                action_id, xy = best_action_id, best_xy
                action = GameAction.from_id(action_id)
                if action.is_complex():
                    x, y = xy if xy is not None else (32, 32)
                    action.set_data({"x": x, "y": y})
                    xy = (x, y)
                action.reasoning = f"hypothesis agent: Q={best_q:.5f} beta={beta:.3f}"

        self._prev_feat = feat
        self._prev_action_id = action_id
        self._prev_xy = xy
        self._prev_state_key = state_key
        return action
