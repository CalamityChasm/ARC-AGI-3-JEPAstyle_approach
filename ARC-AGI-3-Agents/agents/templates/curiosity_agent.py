"""Stage 2 (plan.md): curiosity-driven agent built on the Stage 1 JEPA world model.

Action selection ranks candidate actions by *observed* prediction error
(RND/ICM-style curiosity), not merely *predicted* residual magnitude. Each
turn: encode the current frame, compare it against what the predictor
expected *last* turn (given the action just taken) to get a real surprise
value, and fold that into a running per-action estimate (EMA). The next
action is chosen by ranking those running per-action estimates -- prefer
actions that have historically surprised the model, deprioritize ones that
reliably turn out to be no-ops, and try untried actions first (optimistic
initialization). ACTION6 (needs an (x, y) click target) competes in that
same ranking as a single action; only once it's chosen does a *separate*,
finer-grained per-8x8-patch surprise map (the same per-region granularity
`jepa/eval_stage1.py`'s salience map already uses) get consulted, via
weighted-random sampling, to pick *where* to click.

Two earlier versions of this had real bugs, both worth documenting so a
future agent doesn't reintroduce them:
1. Ranking by *predicted* (not observed) residual got stuck cycling the
   same handful of actions forever with zero levels completed -- a
   consistently-highest-predicted action never got penalized for actually
   producing no change. Fixed by tracking *observed* surprise instead.
2. Ranking all 64 ACTION6 patches as top-level options *alongside* the
   handful of simple actions (all tied at the same optimistic-init value)
   forced a mandatory ~64-action raster-scan of every click location
   before the agent could do anything else -- burning ~20% of a 300-action
   budget on reconnaissance regardless of whether ACTION6 even mattered for
   that game, and empirically performing *worse* than random on a matched
   budget (2 vs 4 levels completed over 3x25-game trials). Fixed by making
   ACTION6 compete as a single option at the top level (mean surprise
   across all patches) and only refining *where* to click, via
   weighted-random sampling, once it's already been chosen -- coverage
   without the mandatory burn-in.

Immediately exploits (repeats the last action for a few more steps) on any
observed increase in `levels_completed`, per plan.md ("exploit immediately
on any observed score delta"). A small epsilon-random fallback guards
against the classic "noisy TV" failure mode of pure curiosity (fixating
forever on something that's genuinely, legitimately surprising every time
but never actually productive) -- plan.md defers a real fix for that to
Stage 5's hypothesis bundle; Stage 2 stays simple by design.

Reuses the (imperfect -- see CLAUDE.md's Stage 1 status) Stage 1 predictor
as a *relative* novelty ranking signal, not a calibrated world model:
directed exploration only needs the predictor to rank candidate actions
sensibly relative to each other, not to predict pixel-perfect outcomes.
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
from jepa.models import ActionConditionedPredictor, CNNEncoder  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
_PATCHES_PER_SIDE = CANVAS // PATCH  # 8


class Curiosity(Agent):
    """Observed-prediction-error-driven exploration on top of the Stage 1 JEPA model."""

    MAX_ACTIONS = 300
    # Keep repeating a just-successful action this many extra steps. Kept
    # small: a level-up usually means the *board* just changed underneath
    # us (new layout), so blindly repeating the old winning action isn't
    # guaranteed to still make sense -- this is a short, cheap bet that it
    # might, not a strategy to lean on heavily.
    EXPLOIT_REPEATS = 2
    SURPRISE_MOMENTUM = 0.7  # EMA weight on the running estimate (higher = forgets slower)
    OPTIMISTIC_INIT = 1.0  # untried actions/locations look this surprising until proven otherwise
    EPSILON = 0.25  # chance of a uniform-random action instead of the surprise ranking
    PATCH_SAMPLE_TEMPERATURE = 0.1  # lower = more sharply peaked toward high-surprise patches

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = get_device()

        self.encoder = CNNEncoder().to(self.device)
        self.encoder.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "encoder_finetuned.pt", map_location=self.device)
        )
        self.encoder.eval()

        game_vocab: dict[str, int] = {}
        vocab_path = _CHECKPOINT_DIR / "game_vocab.json"
        if vocab_path.exists():
            game_vocab = json.loads(vocab_path.read_text())
        # Falls back to index 0 for a game not seen during Stage 1 training
        # (e.g. any private-eval game) -- the per-game embedding just won't
        # mean anything useful there, same limitation Stage 1 already has.
        self.game_idx = game_vocab.get(self.game_id, 0)
        num_games = max(len(game_vocab), 1)

        self.predictor = ActionConditionedPredictor(num_games=num_games).to(self.device)
        self.predictor.load_state_dict(
            torch.load(_CHECKPOINT_DIR / "predictor.pt", map_location=self.device)
        )
        self.predictor.eval()

        # Top-level "which action" ranking -- one scalar per action id.
        self._action_surprise: dict[int, float] = {}
        # Fine-grained "where to click" ranking, only consulted once
        # ACTION6 has already been chosen at the top level.
        self._patch_surprise: dict[tuple[int, int], float] = {}

        self._last_levels_completed = 0
        self._exploit_remaining = 0

        # What we did *last* turn, so this turn can compare the predictor's
        # expectation of it against what actually happened.
        self._prev_feat: torch.Tensor | None = None
        self._prev_action_id: int | None = None
        self._prev_xy: tuple[int, int] | None = None

        seed = hash((self.game_id, id(self))) & 0xFFFFFFFF
        self._rng = random.Random(seed)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @torch.no_grad()
    def _encode(self, latest_frame: FrameData) -> torch.Tensor:
        tensor = arc3_frame_to_tensor(latest_frame.frame)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return self.encoder(x)  # (1, C, 8, 8)

    @torch.no_grad()
    def _predict(self, feat: torch.Tensor, action_id: int, xy: tuple[int, int] | None) -> torch.Tensor:
        b = feat.shape[0]
        action_t = torch.full((b,), action_id, dtype=torch.long, device=self.device)
        if xy is not None:
            x, y = xy
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=self.device)
        else:
            xy_t = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
        game_t = torch.full((b,), self.game_idx, dtype=torch.long, device=self.device)
        return self.predictor(feat, action_t, xy_t, game_t)

    def _update_surprise(self, feat: torch.Tensor) -> None:
        """Attribute the *actual* observed change (`feat` vs. what was
        predicted last turn) to whatever action was taken last turn --
        updating the scalar per-action estimate, and (for ACTION6) the
        specific patch that was clicked."""
        if self._prev_feat is None or self._prev_action_id is None:
            return
        if self._prev_action_id == GameAction.RESET.value:
            return
        with torch.no_grad():
            predicted = self._predict(self._prev_feat, self._prev_action_id, self._prev_xy)
            region_err = (predicted - feat).pow(2).mean(dim=1)[0]  # (8, 8)
            mean_err = region_err.mean().item()

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
        """Weighted-random pick of an 8x8 *patch*, weighted toward
        historically-surprising patches (softmax), with untried patches at
        the optimistic default -- coverage without a forced full scan. The
        actual (x, y) is then a uniform-random pixel *within* that patch,
        not always its exact center -- clicking only ever at patch centers
        would throw away 7/8 of random's pixel-level precision, and the
        target that actually matters for a given game's mechanic may not
        land on a center pixel."""
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
            best_score = -1.0
            action_id = available[0]
            for candidate in available:
                score = self._action_surprise.get(candidate, self.OPTIMISTIC_INIT)
                if score > best_score:
                    best_score = score
                    action_id = candidate

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
            action.reasoning = "curiosity agent: reset"
            self._last_levels_completed = 0
            self._exploit_remaining = 0
            self._prev_feat = None
            self._prev_action_id = None
            self._prev_xy = None
            return action

        feat = self._encode(latest_frame)
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
                f"curiosity agent: exploiting recent level gain "
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
            action.reasoning = f"curiosity agent: est. surprise {score:.5f}"

        self._prev_feat = feat
        self._prev_action_id = action_id
        self._prev_xy = xy
        return action
