"""Stage 3 (plan.md): curiosity-driven exploration + exact memory + a
recurrent world model.

Builds directly on Stage 2's `Curiosity` agent (see curiosity_agent.py for
the bug-fix history behind its exploration design -- observed-not-predicted
surprise, ACTION6 competing as one top-level option, click precision) and
adds the two things plan.md calls out for Stage 3:

1. **Exact (state, action) -> outcome memory** (`jepa/memory.py`'s
   `TransitionGraph`), persisted for the agent's whole lifetime (i.e.
   across every RESET within one game, not just within one level attempt --
   ARC-3 games reliably return to the *exact* same starting frame on
   RESET, so anything learned in a prior attempt is exactly recallable in
   the next one). Before falling back to curiosity-driven exploration,
   check whether this *exact* frame has a known winning action already --
   if so, take it immediately, no re-exploration needed. This is the literal
   "agent remembers what it already tried" milestone, and it's lossless
   (a plain dict lookup), unlike anything the learned predictor provides.
2. **A recurrent memory core** (`RecurrentActionConditionedPredictor` --
   see `jepa/models/recurrent_predictor.py` for why this is a GRUCell
   rather than literal Mamba: `mamba-ssm` doesn't build on this Windows
   box). The hidden state persists across an entire episode (reset only on
   RESET, not every step), so the observed-surprise signal driving
   exploration can, in principle, depend on more than just the current
   frame.

Also uses the exact graph to avoid **re-trying actions already tried from
this exact state** when untried ones remain -- a cheap, correct way to
guarantee full local coverage before ever repeating something whose
outcome is already known, independent of (and complementing) the global
per-action surprise ranking inherited from Stage 2.
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
from jepa.memory import TransitionGraph  # noqa: E402
from jepa.models import CNNEncoder, RecurrentActionConditionedPredictor  # noqa: E402

_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
_PATCHES_PER_SIDE = CANVAS // PATCH  # 8


class Memory(Agent):
    """Curiosity-driven exploration + exact transition memory + a recurrent world model."""

    MAX_ACTIONS = 300
    EXPLOIT_REPEATS = 2
    SURPRISE_MOMENTUM = 0.7
    OPTIMISTIC_INIT = 1.0
    EPSILON = 0.25
    PATCH_SAMPLE_TEMPERATURE = 0.1

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

        # Exact memory -- persists for the agent's whole lifetime (every
        # RESET within this game), not reset per level/attempt.
        self.graph = TransitionGraph()

        self._action_surprise: dict[int, float] = {}
        self._patch_surprise: dict[tuple[int, int], float] = {}
        self._last_levels_completed = 0
        self._exploit_remaining = 0

        # Recurrent hidden state -- persists across steps *within* an
        # episode, reset to zero on RESET (see choose_action).
        self._hidden: torch.Tensor | None = None

        # What we did *last* turn, so this turn can (a) attribute observed
        # surprise, (b) record the transition into the exact graph, and
        # (c) advance the recurrent hidden state.
        self._prev_feat: torch.Tensor | None = None
        self._prev_action_id: int | None = None
        self._prev_xy: tuple[int, int] | None = None
        self._prev_state_key: str | None = None

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
    def _predict(
        self, feat: torch.Tensor, action_id: int, xy: tuple[int, int] | None, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = feat.shape[0]
        action_t = torch.full((b,), action_id, dtype=torch.long, device=self.device)
        if xy is not None:
            x, y = xy
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=self.device)
        else:
            xy_t = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
        game_t = torch.full((b,), self.game_idx, dtype=torch.long, device=self.device)
        return self.predictor(feat, action_t, xy_t, hidden, game_t)

    def _update_from_last_turn(self, feat: torch.Tensor, latest_frame: FrameData) -> None:
        """Attribute the last turn's outcome: record it in the exact
        transition graph, update the observed-surprise estimates, and
        advance the recurrent hidden state."""
        if self._prev_feat is None or self._prev_action_id is None:
            return

        levels_delta = latest_frame.levels_completed - self._last_levels_completed
        if self._prev_state_key is not None:
            self.graph.record(
                self._prev_state_key, self._prev_action_id, self._prev_xy, latest_frame.frame, levels_delta
            )

        if self._prev_action_id == GameAction.RESET.value:
            return

        with torch.no_grad():
            predicted, new_hidden = self._predict(
                self._prev_feat, self._prev_action_id, self._prev_xy, self._hidden
            )
            region_err = (predicted - feat).pow(2).mean(dim=1)[0]  # (8, 8)
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

    def _pick_action(
        self, available: list[int], tried: set[tuple[int, "tuple[int, int] | None"]]
    ) -> tuple[int, tuple[int, int] | None, float]:
        # Prefer actions never tried from this *exact* state -- guaranteed
        # local coverage before ever repeating a known outcome. ACTION6 is
        # effectively always "untried" here (a specific pixel essentially
        # never recurs across draws), which is the correct behavior: it
        # keeps competing on the surprise ranking rather than getting
        # starved by unrelated simple actions having been exhausted.
        untried = [a for a in available if (a, None) not in tried or a == GameAction.ACTION6.value]
        candidates = untried or available

        if self._rng.random() < self.EPSILON:
            action_id = self._rng.choice(candidates)
        else:
            best_score = -1.0
            action_id = candidates[0]
            for candidate in candidates:
                score = self._action_surprise.get(candidate, self.OPTIMISTIC_INIT)
                if score > best_score:
                    best_score = score
                    action_id = candidate

        xy = self._sample_click() if action_id == GameAction.ACTION6.value else None
        return action_id, xy, self._action_surprise.get(action_id, self.OPTIMISTIC_INIT)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "memory agent: reset"
            self._last_levels_completed = 0
            self._exploit_remaining = 0
            self._hidden = None
            self._prev_feat = None
            self._prev_action_id = None
            self._prev_xy = None
            self._prev_state_key = None
            return action

        feat = self._encode(latest_frame)
        if self._hidden is None:
            self._hidden = self.predictor.init_hidden(1, self.device)
        self._update_from_last_turn(feat, latest_frame)

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
                f"memory agent: exploiting recent level gain ({self._exploit_remaining} repeats left)"
            )
        else:
            remembered = self.graph.best_known_action(state_key)
            if remembered is not None and remembered[2] > 0:
                action_id, xy, _delta = remembered
                action = GameAction.from_id(action_id)
                if action.is_complex() and xy is not None:
                    action.set_data({"x": xy[0], "y": xy[1]})
                action.reasoning = "memory agent: recalling a known winning action from this exact state"
                # GameAction.reasoning isn't actually serialized anywhere in
                # this harness (SimpleAction/ComplexAction have no
                # `reasoning` field, so `do_action_request`'s
                # `action.action_data.model_dump()` never carries it) --
                # log this specific event for real, since it's the one
                # thing worth being able to confirm actually fires.
                logger.info(f"{self.game_id} - memory agent: recalling known winning action {action_id} at {xy}")
            else:
                available = latest_frame.available_actions or [
                    a.value for a in GameAction if a is not GameAction.RESET
                ]
                tried = self.graph.tried_actions(state_key)
                action_id, xy, score = self._pick_action(available, tried)
                action = GameAction.from_id(action_id)
                if action.is_complex():
                    x, y = xy if xy is not None else (32, 32)
                    action.set_data({"x": x, "y": y})
                    xy = (x, y)
                action.reasoning = f"memory agent: est. surprise {score:.5f}"

        self._prev_feat = feat
        self._prev_action_id = action_id
        self._prev_xy = xy
        self._prev_state_key = state_key
        return action
