"""Stage 3's recurrent core (`jepa/models/recurrent_predictor.py:
RecurrentActionConditionedPredictor`), wired into a live exploration
agent for the first time -- it was built and evaluated on its own
changed-patches metric back in Stage 3, but no agent ever actually used
it for real-time action selection (see CLAUDE.md's own "never evaluated
on pure exploration efficacy" note).

Motivation: the whole Stage 6 held-out-games investigation found the
world model's raw one-step prediction accuracy doesn't matter much for
`bp35`/`ka59` specifically (13+ world-model interventions, all null on
breadth) -- but nothing tested so far gave the exploration policy any
memory of what it already tried *this episode*. A GRU hidden state
carried across the episode is the one component in this whole project
built for exactly that and never actually plugged into a decision.

Structurally this is `Curiosity` (Stage 2) with two changes:
1. The Stage 1 stateless one-step predictor is swapped for the Stage 3
   recurrent predictor, with a real hidden state maintained across the
   whole episode (reset only on RESET, i.e. at episode start -- matching
   `RecurrentActionConditionedPredictor.init_hidden`'s own documented
   reset boundary). Loads a checkpoint trained via `--exclude-games
   r11l,bp35,m0r0,tr87,ka59` (`checkpoints_recurrent_holdout/`), so this
   is a fair zero-shot comparison against everything else in the Stage 6
   held-out-games investigation, not a checkpoint that's already seen
   these games.
2. The top-level action ranking uses the SAME temperature-softmax sample
   over `_action_surprise` that `hypothesis_agent.py`'s own argmax-lock
   fix uses (see `experiments/stage6_action_selection_softmax.md`) --
   `Curiosity`'s original hard argmax over its EMA surprise scores has
   the identical "deterministic argmax on a near-flat map always picks
   the same index" bug, just never traced closely enough to notice
   before. Fixed here directly rather than reintroducing a known bug into
   a brand new agent copied from the buggy original.

Everything else (observed-vs-predicted surprise as an EMA per action,
ACTION6 competing as one top-level option with click location resolved
separately via softmax-sampled patch selection, exploit-on-level-up,
epsilon-random fallback) is unchanged from `Curiosity`'s own
already-debugged design -- see that agent's own docstring for the bug
history behind each piece.
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
from jepa.grid import CANVAS, PATCH, arc3_frame_to_tensor  # noqa: E402
from jepa.models import CNNEncoder, RecurrentActionConditionedPredictor  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints_recurrent_holdout"
_PATCHES_PER_SIDE = CANVAS // PATCH  # 8


class RecurrentCuriosity(Agent):
    """Observed-prediction-error-driven exploration on top of Stage 3's
    recurrent (GRU-history-carrying) world model, instead of Stage 1's
    stateless one-step model."""

    MAX_ACTIONS = 300
    EXPLOIT_REPEATS = 2
    SURPRISE_MOMENTUM = 0.7
    OPTIMISTIC_INIT = 1.0
    EPSILON = 0.25
    PATCH_SAMPLE_TEMPERATURE = 0.1
    # Same value and rationale as hypothesis_agent.py's ACTION_SAMPLE_TEMPERATURE.
    ACTION_SAMPLE_TEMPERATURE = 0.1

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

        self._action_surprise: dict[int, float] = {}
        self._patch_surprise: dict[tuple[int, int], float] = {}

        self._last_levels_completed = 0
        self._exploit_remaining = 0

        self._prev_feat: torch.Tensor | None = None
        self._prev_action_id: int | None = None
        self._prev_xy: tuple[int, int] | None = None
        # The hidden state *before* the last step's predictor call -- needed
        # to recompute what the model would have predicted (for the surprise
        # comparison) without re-running the actual state transition twice.
        self._prev_hidden: torch.Tensor | None = None

        seed = hash((self.game_id, id(self))) & 0xFFFFFFFF
        self._rng = random.Random(seed)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @torch.no_grad()
    def _encode(self, latest_frame: FrameData) -> torch.Tensor:
        tensor = arc3_frame_to_tensor(latest_frame.frame)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return self.encoder(x)

    @torch.no_grad()
    def _predict(
        self, feat: torch.Tensor, action_id: int, xy: tuple[int, int] | None, hidden: torch.Tensor
    ) -> tuple:
        b = feat.shape[0]
        action_t = torch.full((b,), action_id, dtype=torch.long, device=self.device)
        if xy is not None:
            x, y = xy
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=self.device)
        else:
            xy_t = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
        game_t = torch.full((b,), self.game_idx, dtype=torch.long, device=self.device)
        return self.predictor(feat, action_t, xy_t, hidden, game_t)

    def _update_surprise(self, feat: torch.Tensor) -> None:
        """Attribute observed change to the action taken last turn, same as
        `Curiosity`'s own version -- but the prediction being compared
        against was made with the hidden state carried into that step, and
        the *new* hidden state (from actually replaying that transition) is
        what carries forward into this turn's decision, so the model's
        history genuinely accumulates real observations, not just replayed
        predictions."""
        if self._prev_feat is None or self._prev_action_id is None or self._prev_hidden is None:
            return
        if self._prev_action_id == GameAction.RESET.value:
            return
        with torch.no_grad():
            predicted, new_hidden = self._predict(
                self._prev_feat, self._prev_action_id, self._prev_xy, self._prev_hidden
            )
            region_err = (predicted - feat).pow(2).mean(dim=1)[0]
            mean_err = region_err.mean().item()
        self._hidden = new_hidden

        old = self._action_surprise.get(self._prev_action_id, self.OPTIMISTIC_INIT)
        self._action_surprise[self._prev_action_id] = (
            self.SURPRISE_MOMENTUM * old + (1 - self.SURPRISE_MOMENTUM) * mean_err
        )

        if self._prev_action_id == GameAction.ACTION6.value and self._prev_xy is not None:
            x, y = self._prev_xy
            patch_key = (y // PATCH, x // PATCH)
            patch_err = region_err[patch_key].item()
            old_patch = self._patch_surprise.get(patch_key, self.OPTIMISTIC_INIT)
            self._patch_surprise[patch_key] = (
                self.SURPRISE_MOMENTUM * old_patch + (1 - self.SURPRISE_MOMENTUM) * patch_err
            )

    def _sample_click(self) -> tuple[int, int]:
        scores = []
        patches = []
        for row in range(_PATCHES_PER_SIDE):
            for col in range(_PATCHES_PER_SIDE):
                scores.append(self._patch_surprise.get((row, col), self.OPTIMISTIC_INIT))
                patches.append((row, col))
        max_score = max(scores)
        weights = [
            pow(2.718281828, (s - max_score) / self.PATCH_SAMPLE_TEMPERATURE) for s in scores
        ]
        row, col = self._rng.choices(patches, weights=weights, k=1)[0]
        x = col * PATCH + self._rng.randrange(PATCH)
        y = row * PATCH + self._rng.randrange(PATCH)
        return x, y

    def _pick_action(self, available: list[int]) -> tuple[int, tuple[int, int] | None, float]:
        if self._rng.random() < self.EPSILON:
            action_id = self._rng.choice(available)
        else:
            # Temperature-weighted softmax sample over the EMA surprise
            # scores, not a hard argmax -- see this module's own docstring
            # for why a hard argmax here would silently reintroduce
            # Curiosity's original (never-noticed) version of the same
            # bug hypothesis_agent.py's ACTION_SAMPLE_TEMPERATURE fixes.
            scores = [self._action_surprise.get(c, self.OPTIMISTIC_INIT) for c in available]
            max_score = max(scores)
            weights = [
                pow(2.718281828, (s - max_score) / self.ACTION_SAMPLE_TEMPERATURE) for s in scores
            ]
            action_id = self._rng.choices(available, weights=weights, k=1)[0]

        if action_id == GameAction.ACTION6.value:
            xy = self._sample_click()
        else:
            xy = None
        return action_id, xy, self._action_surprise.get(action_id, self.OPTIMISTIC_INIT)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "recurrent curiosity agent: reset"
            self._last_levels_completed = 0
            self._exploit_remaining = 0
            self._prev_feat = None
            self._prev_action_id = None
            self._prev_xy = None
            self._prev_hidden = None
            # New episode -- carrying the hidden state across a RESET would
            # mix this attempt's history with the last one's, which is
            # exactly the noise RecurrentActionConditionedPredictor's own
            # `init_hidden` reset boundary exists to avoid.
            self._hidden = self.predictor.init_hidden(1, self.device)
            return action

        feat = self._encode(latest_frame)
        hidden_before_update = self._hidden
        self._update_surprise(feat)

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
                f"recurrent curiosity agent: exploiting recent level gain "
                f"({self._exploit_remaining} repeats left)"
            )
        else:
            available = latest_frame.available_actions or [
                a.value for a in GameAction if a is not GameAction.RESET
            ]
            action_id, xy, score = self._pick_action(available)
            action = GameAction.from_id(action_id)
            if action.is_complex():
                x, y = xy if xy is not None else (32, 32)
                action.set_data({"x": x, "y": y})
                xy = (x, y)
            action.reasoning = f"recurrent curiosity agent: est. surprise {score:.5f}"

        self._prev_feat = feat
        self._prev_action_id = action_id
        self._prev_xy = xy
        self._prev_hidden = hidden_before_update
        return action
