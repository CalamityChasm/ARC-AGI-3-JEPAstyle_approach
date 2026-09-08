"""Test-time adaptation for jepa/click_effect_model.py's ClickEffectModel
-- mirrors jepa/test_time_adapter.py's design (same ANIL-style restricted-
subset recipe, same per-game not per-RESET reset boundary, same
buffer-then-periodic-step mechanics) applied to a different, smaller model
and a different signal (local click-effect prediction rather than
whole-board latent prediction).

Why adapt at all, given the pretrained model already showed real
cross-game AUC (~0.77, scripts/train_click_effect_model.py's leave-games-
out CV): per-game click-effect base rates vary enormously in the training
data (some games' clicks are 0% "does anything" in the harvested window,
others 100%) -- a frozen zero-shot model can rank candidates reasonably
but is poorly CALIBRATED to any one game's own regime until it sees that
game's own data. Adaptation is exactly what closes that gap, same
motivation as the production predictor's own test-time adaptation.

**Dose (k/n_steps/lr) swept, not guessed** (scripts/sweep_click_effect_tta.py,
leave-games-out: pretrain on all-but-one game, adapt on that held-out
game's own first ~70% of transitions in order, evaluate AUC on its last
~30%, never adapted on). The original untuned defaults (k=10, n_steps=3,
lr=5e-4) left real signal on the table: frozen (no adaptation) pooled AUC
on held-out games was ~0.51 (barely above chance) and those defaults only
reached ~0.54. A real, clean dose-response effect -- mirroring this
project's own JEPA-side TTA/Reptile dosing history -- jumps sharply to
~0.77 at `k=5, n_steps=8, lr=2e-3` (one game, `ft09`, went from 0.07 --
worse than random -- to 0.99), then plateaus/wobbles in the 0.73-0.76
band for every dose tried past that (up to `k=1, n_steps=40, lr=2e-2`) --
not worth the extra per-decision compute for no further gain. One game
(`m0r0`) never improved and slightly degraded at higher doses across two
independent sweep runs -- consistent enough to be a real per-game
characteristic (this specific local pattern of clicks may just not carry
the kind of signal this model can adapt to), not sweep noise.
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .click_effect_features import extract_patch, extract_segment_features
from .click_effect_model import ClickEffectModel, get_adapter_params


class ClickEffectAdapter:
    """Wraps a live `ClickEffectModel`, streaming real observed ACTION6
    transitions from ONE game and periodically taking a few gradient steps
    on the restricted adapter subset (trunk + both heads).

    Reset boundary -- persists across RESETs of the SAME game, only resets
    on a genuinely new game. Same rationale as jepa/test_time_adapter.py's
    own TestTimeAdapter.reset docstring: the dynamics being adapted to
    don't change across a RESET, and this project's Swarm already
    constructs one fresh agent instance per game_id, so this reset
    boundary is enforced for free by construction -- reset() exists for
    defensiveness/testability, not because it's expected to fire in normal
    play.
    """

    def __init__(
        self,
        model: ClickEffectModel,
        device: torch.device,
        k: int = 5,
        n_steps: int = 8,
        lr: float = 2e-3,
        buffer_size: int = 200,
        batch_size: int = 16,
        min_buffer_for_adapt: int = 6,
    ) -> None:
        self.model = model
        self.device = device
        self.k = k
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.min_buffer_for_adapt = min_buffer_for_adapt

        self.buffer: deque = deque(maxlen=buffer_size)
        self._n_observed = 0
        self.n_adapt_events = 0
        self.last_adapt_latency_s: Optional[float] = None
        self.enabled = True

        self._params = get_adapter_params(model)
        self._snapshot = [p.detach().clone().cpu() for p in self._params]

        for p in model.parameters():
            p.requires_grad = False
        for p in self._params:
            p.requires_grad = True

        self.opt = torch.optim.AdamW(self._params, lr=lr)
        self.bce = nn.BCEWithLogitsLoss()

    def reset(self) -> None:
        with torch.no_grad():
            for p, snap in zip(self._params, self._snapshot):
                p.copy_(snap.to(p.device))
        self.buffer.clear()
        self._n_observed = 0
        self.n_adapt_events = 0
        self.opt = torch.optim.AdamW(self._params, lr=self.lr)

    def observe(self, frame_t: np.ndarray, x: int, y: int, seg_features: np.ndarray,
                frame_changed: bool, won: bool) -> None:
        """Record one real ACTION6 (frame_t, x, y) -> outcome observation
        just made during play, and fire an adaptation step every k
        observations. Cheap to call every ACTION6 decision -- the buffer
        append is O(patch-extraction), most calls don't trigger a step.
        `seg_features` is passed in rather than recomputed here since the
        live agent already runs FrameProcessor.segment_frame for its own
        action-group bookkeeping every decision."""
        patch = extract_patch(frame_t, x, y)
        self.buffer.append((patch, seg_features, float(frame_changed), float(won)))
        self._n_observed += 1
        if not self.enabled:
            return
        if len(self.buffer) < self.min_buffer_for_adapt:
            return
        if self._n_observed % self.k == 0:
            self._adapt_step()

    def _sample_batch(self) -> list:
        n = min(self.batch_size, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def _adapt_step(self) -> None:
        start = time.time()
        batch = self._sample_batch()
        patches = np.stack([b[0] for b in batch])
        seg_feats = np.stack([b[1] for b in batch])
        changed = np.array([b[2] for b in batch], dtype=np.float32)
        win = np.array([b[3] for b in batch], dtype=np.float32)

        patch_t = torch.from_numpy(patches).long().to(self.device)
        seg_t = torch.from_numpy(seg_feats).float().to(self.device)
        changed_t = torch.from_numpy(changed).to(self.device)
        win_t = torch.from_numpy(win).to(self.device)

        self.model.train()
        try:
            for _ in range(self.n_steps):
                changed_logit, win_logit = self.model(patch_t, seg_t)
                loss = self.bce(changed_logit, changed_t) + self.bce(win_logit, win_t)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        finally:
            self.model.eval()

        self.n_adapt_events += 1
        self.last_adapt_latency_s = time.time() - start

    @torch.no_grad()
    def predict(self, frame_t: np.ndarray, x: int, y: int, seg_features: np.ndarray) -> float:
        """Returns P(frame_changed) for a candidate click -- the score
        graph_explorer_learned_agent.py biases explore-mode tie-breaking
        with. win probability isn't exposed here: it's part of the same
        joint loss (regularizing the shared trunk with a genuine, if rare,
        win signal) but this project's own Stage 5 history documents how
        unreliable a model trained on ~2% positive-rate data is when used
        directly as a per-decision score -- frame_changed is the head with
        real, validated held-out-game AUC (~0.77); win's AUC varied more
        (0.33-0.81 across folds) and isn't trusted for live scoring yet."""
        patch = extract_patch(frame_t, x, y)
        patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
        seg_t = torch.from_numpy(seg_features).float().unsqueeze(0).to(self.device)
        changed_logit, _win_logit = self.model(patch_t, seg_t)
        return torch.sigmoid(changed_logit).item()
