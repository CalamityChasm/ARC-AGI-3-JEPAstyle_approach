"""Real multi-step lookahead over Stage 3's recurrent world model --
the natural next lever flagged by `experiments/stage6_recurrent_exploration.md`
after `RecurrentCuriosity` (retrospective, history-conditioned surprise
ranking) came back a clean 0/32 on `bp35`/`ka59`. That experiment's own
honest caveat: a better *retrospective* surprise estimate is not the same
claim as "the agent can simulate ahead before committing to a first
action." This agent tests the untested half.

Design: random-shooting Model Predictive Control (MPC), the standard
lightweight approach when you have a learned dynamics model but no
reward model and want a real search rather than a single reactive
prediction:

1. **Candidates.** Every available simple action, plus (if ACTION6 is
   legal) `NUM_CLICK_CANDIDATES` sampled click locations -- there is no
   `HypothesisBundle`-style expert-disagreement salience map available
   for a single monolithic recurrent predictor, so candidates are sampled
   uniformly (same "cover the space with cheap probes" logic as this
   project's own experiment-designer opening probes elsewhere).
2. **Rollout.** For each candidate FIRST action, simulate
   `SEARCH_DEPTH` steps forward *purely in feature space* -- the
   predicted feature from step *k* is fed back in as the "current
   feature" for step *k+1*, so no real environment interaction and, past
   the one real encode already needed for this turn's decision, no CNN
   forward passes either (`RecurrentActionConditionedPredictor` operates
   entirely on features). Steps 2..D within a rollout pick their own
   action uniformly at random -- only the *first* action of each
   candidate rollout is what's actually being evaluated; a full branching
   tree past depth 1 is not implemented, on the standard random-shooting
   argument that broad, cheap single-sample rollouts beat a narrow,
   expensive exhaustive tree at this action-space size.
3. **Scoring: reachable novelty.** There is no value/reward signal that
   means anything on a held-out game (Stage 5's value head doesn't
   transfer across encoders, and this project's own InfoGain mechanism is
   specifically an MoE-ensemble-disagreement signal this single recurrent
   model has no equivalent of) -- so the search objective is *state
   novelty*: does this imagined path reach a point meaningfully different
   from anything actually observed yet this episode? Concretely, each
   candidate's score is the MAX over its simulated trajectory of the
   MINIMUM L2 distance from that step's pooled predicted feature to any
   REAL pooled feature already recorded in `self._memory` (only real
   observations are ever added to it -- imagined states never pollute the
   novelty reference set). The first action of the highest-scoring
   candidate is taken. This is a standard episodic-novelty exploration
   bonus (à la NGU/RND-style episodic memory), just computed via lookahead
   instead of applied post-hoc to the action actually taken.

Hidden-state and memory bookkeeping is real, not simulated, at the top
level: `self._hidden` and `self._memory` only ever advance from actually
observed transitions (see `_advance_real_state`) -- the scratch hidden
state used *inside* a single search call is always a fresh clone of
`self._hidden`, discarded after scoring.

Loads the same held-out-trained checkpoint as `RecurrentCuriosity`
(`checkpoints_recurrent_holdout/`, `--exclude-games
r11l,bp35,m0r0,tr87,ka59`) for a fair zero-shot comparison.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.device import get_device  # noqa: E402
from jepa.grid import CANVAS, arc3_frame_to_tensor  # noqa: E402
from jepa.models import CNNEncoder, RecurrentActionConditionedPredictor  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints_recurrent_holdout"


class RecurrentSearch(Agent):
    """Random-shooting MPC over the recurrent world model, scored by
    reachable novelty against an episodic memory of real observations."""

    MAX_ACTIONS = 300
    EXPLOIT_REPEATS = 2
    EPSILON = 0.25
    # Rollout length in imagined steps. Kept short: compute is cheap per
    # step (pure feature-space forward passes, no CNN re-encode), but a
    # single-model recurrent predictor's own errors compound with every
    # imagined step -- Stage 1/3's own history (predictors that learn to
    # coast toward identity) means trusting a long imagined rollout is
    # riskier than trusting a short one. 3 is a starting point, not swept.
    SEARCH_DEPTH = 3
    # How many candidate ACTION6 click locations to probe per decision,
    # sampled uniformly (no salience map available for a single model --
    # see module docstring). Kept small relative to the 64 available
    # patches to bound per-decision compute (breadth x depth forward
    # passes every turn).
    NUM_CLICK_CANDIDATES = 6
    # Episodic memory cap -- bounded mainly so distance computation stays
    # cheap even across a 2500-action episode, not because older
    # observations stop being relevant.
    MEMORY_CAP = 3000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = get_device()

        self.encoder = CNNEncoder().to(self.device)
        self.encoder.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "encoder_recurrent.pt", map_location=self.device)
        )
        self.encoder.eval()

        game_vocab: dict[str, int] = {}
        vocab_path = _CHECKPOINT_DIR / "game_vocab_recurrent.json"
        if vocab_path.exists():
            game_vocab = json.loads(vocab_path.read_text())
        self.game_idx = game_vocab.get(self.game_id, 0)
        num_games = max(len(game_vocab), 1)

        self.predictor = RecurrentActionConditionedPredictor(num_games=num_games).to(self.device)
        self.predictor.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "recurrent_predictor.pt", map_location=self.device)
        )
        self.predictor.eval()

        self._hidden = self.predictor.init_hidden(1, self.device)
        self._memory: list[torch.Tensor] = []  # pooled (C,) real feature vectors, this episode only

        self._last_levels_completed = 0
        self._exploit_remaining = 0
        self._prev_action_id: int | None = None
        self._prev_xy: tuple[int, int] | None = None
        self._prev_feat: torch.Tensor | None = None
        self._prev_hidden: torch.Tensor | None = None

        seed = hash((self.game_id, id(self))) & 0xFFFFFFFF
        self._rng = random.Random(seed)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @torch.no_grad()
    def _encode(self, latest_frame: FrameData) -> torch.Tensor:
        tensor = arc3_frame_to_tensor(latest_frame.frame)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return self.encoder(x)  # (1, C, 8, 8)

    def _action_tensors(self, action_id: int, xy: tuple[int, int] | None) -> tuple:
        b = 1
        action_t = torch.full((b,), action_id, dtype=torch.long, device=self.device)
        if xy is not None:
            x, y = xy
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=self.device)
        else:
            xy_t = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
        game_t = torch.full((b,), self.game_idx, dtype=torch.long, device=self.device)
        return action_t, xy_t, game_t

    def _random_xy(self) -> tuple[int, int]:
        return self._rng.randrange(CANVAS), self._rng.randrange(CANVAS)

    @torch.no_grad()
    def _advance_real_state(self, observed_feat: torch.Tensor) -> None:
        """Called once per real turn with the actually-observed feature --
        advances `self._hidden` by replaying the REAL previous transition
        (never a simulated one), and records `observed_feat` into the
        episodic novelty memory. No-op on the first decision of an episode
        (nothing to replay yet) or after a RESET action (RESET carries no
        dynamics signal, same convention as every other agent in this
        project)."""
        if self._prev_feat is not None and self._prev_action_id is not None:
            if self._prev_action_id != GameAction.RESET.value:
                action_t, xy_t, game_t = self._action_tensors(self._prev_action_id, self._prev_xy)
                _pred, new_hidden = self.predictor(self._prev_feat, action_t, xy_t, self._prev_hidden, game_t)
                self._hidden = new_hidden

        pooled = observed_feat.mean(dim=(2, 3)).squeeze(0)  # (C,)
        self._memory.append(pooled)
        if len(self._memory) > self.MEMORY_CAP:
            self._memory = self._memory[-self.MEMORY_CAP :]

    @torch.no_grad()
    def _novelty(self, pooled_query: torch.Tensor) -> float:
        """Min L2 distance from `pooled_query` (C,) to any real feature
        observed so far this episode -- larger means more novel. Returns a
        large constant if memory is still empty (everything is novel
        before anything has been seen)."""
        if not self._memory:
            return 1e6
        mem = torch.stack(self._memory, dim=0)  # (N, C)
        dists = torch.cdist(pooled_query.unsqueeze(0), mem).squeeze(0)  # (N,)
        return dists.min().item()

    @torch.no_grad()
    def _rollout_score(
        self, cur_feat: torch.Tensor, first_action: int, first_xy: tuple[int, int] | None,
        available: list[int],
    ) -> float:
        """Simulates SEARCH_DEPTH steps starting with `first_action`,
        entirely in imagined feature space, scoring by the max
        reachable-novelty seen along the trajectory."""
        hidden = self._hidden.clone()
        feat = cur_feat
        action_id, xy = first_action, first_xy
        best = -1.0
        for _step in range(self.SEARCH_DEPTH):
            action_t, xy_t, game_t = self._action_tensors(action_id, xy)
            pred_feat, hidden = self.predictor(feat, action_t, xy_t, hidden, game_t)
            pooled = pred_feat.mean(dim=(2, 3)).squeeze(0)
            best = max(best, self._novelty(pooled))
            feat = pred_feat
            # Continuation action for the NEXT imagined step -- uniform
            # random, this candidate's own identity is only its first
            # action (see module docstring).
            action_id = self._rng.choice(available)
            xy = self._random_xy() if action_id == GameAction.ACTION6.value else None
        return best

    def _pick_action(
        self, cur_feat: torch.Tensor, available: list[int]
    ) -> tuple[int, tuple[int, int] | None]:
        if self._rng.random() < self.EPSILON:
            action_id = self._rng.choice(available)
            xy = self._random_xy() if action_id == GameAction.ACTION6.value else None
            return action_id, xy

        candidates: list[tuple[int, tuple[int, int] | None]] = []
        for a in available:
            if a == GameAction.ACTION6.value:
                for _ in range(self.NUM_CLICK_CANDIDATES):
                    candidates.append((a, self._random_xy()))
            else:
                candidates.append((a, None))

        best_score, best_candidate = -1.0, candidates[0]
        for action_id, xy in candidates:
            score = self._rollout_score(cur_feat, action_id, xy, available)
            if score > best_score:
                best_score, best_candidate = score, (action_id, xy)
        return best_candidate

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "recurrent search agent: reset"
            self._last_levels_completed = 0
            self._exploit_remaining = 0
            self._prev_feat = None
            self._prev_action_id = None
            self._prev_xy = None
            self._prev_hidden = None
            self._hidden = self.predictor.init_hidden(1, self.device)
            self._memory = []
            return action

        feat = self._encode(latest_frame)
        hidden_before_update = self._hidden
        self._advance_real_state(feat)

        if latest_frame.levels_completed > self._last_levels_completed:
            self._exploit_remaining = self.EXPLOIT_REPEATS
        self._last_levels_completed = latest_frame.levels_completed

        if self._exploit_remaining > 0 and self._prev_action_id is not None:
            self._exploit_remaining -= 1
            action_id, xy = self._prev_action_id, self._prev_xy
            action = GameAction.from_id(action_id)
            if action.is_complex() and xy is not None:
                action.set_data({"x": xy[0], "y": xy[1]})
            action.reasoning = (
                f"recurrent search agent: exploiting recent level gain "
                f"({self._exploit_remaining} repeats left)"
            )
        else:
            available = latest_frame.available_actions or [
                a.value for a in GameAction if a is not GameAction.RESET
            ]
            action_id, xy = self._pick_action(feat, available)
            action = GameAction.from_id(action_id)
            if action.is_complex():
                x, y = xy if xy is not None else (32, 32)
                action.set_data({"x": x, "y": y})
                xy = (x, y)
            action.reasoning = "recurrent search agent: lookahead novelty search"

        self._prev_feat = feat
        self._prev_action_id = action_id
        self._prev_xy = xy
        self._prev_hidden = hidden_before_update
        return action
