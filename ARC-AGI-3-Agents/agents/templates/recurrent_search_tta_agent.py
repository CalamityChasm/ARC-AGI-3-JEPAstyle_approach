"""Adds real test-time adaptation (gradient steps on the recurrent
predictor's own observed transitions, taken live during play) on top of
`RecurrentSearch`'s random-shooting MPC.

Motivation: `RecurrentSearch` (a frozen world model) came back a clean
0/32 on `bp35`/`ka59` (`experiments/stage6_recurrent_exploration.md`) --
root-caused to residual collapse, the predictor coasting to identity on
unfamiliar games. `scripts/test_time_adaptation_recurrent.py` directly
confirmed TTA substantially closes that gap for THIS predictor on THESE
two games (changed-patches `bp35`: -9.76% -> -0.51%, `ka59`: -6.51% ->
-1.84%, both at n=200 observed transitions) -- a bigger effect than TTA
ever showed on the MoE predictor. This agent tests whether that
representation-level gain translates into real level completions.
Extends `RecurrentSearch` rather than duplicating it (only `__init__` and
`choose_action` differ; rollout/novelty/action-selection logic is reused
unchanged via inheritance).

TTA operating point matches the diagnostic exactly: K=5 (adapt every 5
newly observed real transitions), 8 AdamW steps per adaptation event,
lr=5e-5, adapting only `predictor.net[-1]` (the final Conv2d producing
the residual, ~4.2K params, ANIL-style). Adaptation persists across
RESETs of the same game -- mirrors the MoE agent's `TestTimeAdapter`
design choice (only a fresh agent instance per game resets it), not
reset here.

Because `AVAILABLE_AGENTS` in `agents/__init__.py` is built from
`Agent.__subclasses__()` (direct subclasses only), this class is
manually registered there the same way `ReasoningAgent` already is --
being a subclass of `RecurrentSearch`, not `Agent` directly, it would
not otherwise be discovered.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from arcengine import FrameData, GameAction, GameState

from .recurrent_search_agent import RecurrentSearch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jepa.grid import arc3_frame_to_tensor, patch_change_mask  # noqa: E402
from jepa.losses import weighted_prediction_loss  # noqa: E402


class RecurrentSearchTTA(RecurrentSearch):
    """`RecurrentSearch` + live test-time adaptation on observed transitions."""

    TTA_SEQ_LEN = 16
    TTA_K = 5
    TTA_STEPS = 8
    TTA_LR = 5e-5
    ADAPT_BUFFER_CAP = 2000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for p in self.predictor.parameters():
            p.requires_grad = False
        self._tta_trainable = list(self.predictor.net[-1].parameters())
        for p in self._tta_trainable:
            p.requires_grad = True
        self._tta_opt = torch.optim.AdamW(self._tta_trainable, lr=self.TTA_LR)
        self._adapt_buffer: list = []
        self._prev_raw_frame = None

    def _chunk_loss(self, chunk: list) -> torch.Tensor:
        curs, actions, xys, nxts, masks = [], [], [], [], []
        for frame_t, action_id, xy, frame_t1 in chunk:
            curs.append(arc3_frame_to_tensor(frame_t))
            actions.append(action_id)
            x, y = xy if xy is not None else (32, 32)
            xys.append([x / 63.0, y / 63.0])
            nxts.append(arc3_frame_to_tensor(frame_t1))
            masks.append(patch_change_mask(frame_t, frame_t1))
        cur = torch.from_numpy(np.stack(curs)).to(self.device)
        nxt = torch.from_numpy(np.stack(nxts)).to(self.device)
        action_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        xy_t = torch.tensor(xys, dtype=torch.float32, device=self.device)
        mask_t = torch.from_numpy(np.stack(masks)).to(self.device)
        game_t = torch.full((len(chunk),), self.game_idx, dtype=torch.long, device=self.device)

        hidden = self.predictor.init_hidden(1, self.device)
        losses = []
        for step in range(len(chunk)):
            with torch.no_grad():
                cur_feat = self.encoder(cur[step : step + 1])
                next_feat = self.encoder(nxt[step : step + 1])
            pred_feat, hidden = self.predictor(
                cur_feat, action_t[step : step + 1], xy_t[step : step + 1], hidden, game_t[step : step + 1]
            )
            losses.append(weighted_prediction_loss(pred_feat, next_feat, mask_t[step].unsqueeze(0)))
        return sum(losses) / len(losses)

    def _run_tta_step(self) -> None:
        self.predictor.train()
        for _ in range(self.TTA_STEPS):
            seq_len = min(self.TTA_SEQ_LEN, len(self._adapt_buffer))
            start = (
                0
                if len(self._adapt_buffer) <= seq_len
                else self._rng.randrange(len(self._adapt_buffer) - seq_len + 1)
            )
            chunk = self._adapt_buffer[start : start + seq_len]
            loss = self._chunk_loss(chunk)
            self._tta_opt.zero_grad()
            loss.backward()
            self._tta_opt.step()
        self.predictor.eval()

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_raw_frame = None
            return super().choose_action(frames, latest_frame)

        raw_frame = latest_frame.frame
        if (
            self._prev_raw_frame is not None
            and self._prev_action_id is not None
            and self._prev_action_id != GameAction.RESET.value
        ):
            self._adapt_buffer.append(
                (self._prev_raw_frame, self._prev_action_id, self._prev_xy, raw_frame)
            )
            if len(self._adapt_buffer) > self.ADAPT_BUFFER_CAP:
                self._adapt_buffer = self._adapt_buffer[-self.ADAPT_BUFFER_CAP :]
            if (
                len(self._adapt_buffer) % self.TTA_K == 0
                and len(self._adapt_buffer) >= self.TTA_SEQ_LEN
            ):
                self._run_tta_step()

        action = super().choose_action(frames, latest_frame)
        self._prev_raw_frame = raw_frame
        return action
