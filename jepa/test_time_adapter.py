"""Production test-time adaptation for the MoE predictor
(stage6-test-time-adaptation-agent).

`scripts/test_time_adaptation.py` (stage6-test-time-adaptation) showed that
letting the MoE predictor take a few real AdamW steps on a held-out game's
own observed transitions, DURING simulated play, gives a real (if modest)
changed-patches improvement that a frozen zero-shot forward pass cannot
produce -- see CLAUDE.md's Stage 6 addendum and
experiments/stage6_test_time_adaptation.md for the full diagnostic. This
module turns that diagnostic into a reusable component
`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py` can wire into real
play, with a clean per-game reset mechanism (see `TestTimeAdapter.reset`
docstring for why the reset boundary is "new game", not "every RESET").

Adapted parameters are the SAME restricted ANIL-style subset validated in
the original diagnostic: each of the K experts' LAST `Conv2d` (the layer
directly producing the residual) plus the gate's LAST `Linear`, ~33.8K
params for the production K=8 checkpoint. Everything else (encoder,
action/xy/game embeddings, every expert's earlier layers) stays frozen.
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Optional

import torch

from .grid import arc3_frame_to_tensor, patch_change_mask
from .losses import weighted_prediction_loss
from .models import MoEPredictor


def get_adapter_params(predictor: MoEPredictor) -> list:
    """Each expert's LAST Conv2d + the gate's LAST Linear -- identical
    subset to scripts/test_time_adaptation.py's set_adapter_trainable, kept
    as a single shared definition so the diagnostic script and the real
    agent can never silently drift apart on which params get adapted."""
    params = []
    for expert in predictor.experts:
        last_conv = expert[-1]
        params.extend(last_conv.parameters())
    params.extend(predictor.gate[-1].parameters())
    return params


class TestTimeAdapter:
    """Wraps a live `MoEPredictor`, streaming real observed transitions
    from ONE game and periodically taking a few gradient steps on the
    restricted adapter subset above.

    Reset boundary -- persists across RESETs of the SAME game, only resets
    on a genuinely NEW game:

    ARC-3's RESET returns to the same game's starting board, not a
    different game -- the underlying dynamics being adapted to (what does
    action X do in THIS game) don't change across a RESET, so there's no
    principled reason to throw away gradient progress just because the
    board reset. This mirrors `jepa/memory.py: TransitionGraph`'s existing
    design in this project, which already persists its exact-recall graph
    across RESETs of the same game for the identical reason (see
    CLAUDE.md's Stage 3 section). The counter-argument (compounding
    interference if early adaptation was based on limited/misleading data)
    is real but weaker here: the adapted subset is tiny (~33.8K params),
    the learning rate is small (this project's own sweep -- see
    experiments/stage6_test_time_adaptation_agent.md -- found the
    interference/gain tradeoff is a smooth, well-behaved dial, not a cliff
    that early bad data could fall off), and MORE data (accumulated across
    resets) should if anything make estimates *more* reliable, not less,
    consistent with why persisting was chosen.

    In practice this reset boundary is already enforced for free by how
    `ARC-AGI-3-Agents/agents/swarm.py` runs games: one fresh `Hypothesis`
    instance (and therefore one fresh `TestTimeAdapter`, built from the
    pristine checkpoint) is constructed per game_id in `Swarm.main`'s game
    list, and an agent's `main()` loop only ever plays that ONE game
    (possibly across many RESETs) before exiting. `reset()` below exists
    for defensiveness (in case that per-game-instantiation assumption ever
    changes, e.g. a future harness variant that reuses one long-lived agent
    object across games) and so this can be unit-tested / reasoned about
    directly rather than relying entirely on an external construction
    pattern this module can't see.
    """

    def __init__(
        self,
        predictor: MoEPredictor,
        encoder: torch.nn.Module,
        device: torch.device,
        game_idx: int,
        k: int = 25,
        n_steps: int = 3,
        lr: float = 5e-5,
        buffer_size: int = 400,
        batch_size: int = 16,
        min_buffer_for_adapt: int = 8,
    ) -> None:
        self.predictor = predictor
        self.encoder = encoder
        self.device = device
        self.game_idx = game_idx
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

        self._params = get_adapter_params(predictor)
        # CPU clones -- avoids holding a second GPU copy of the (small)
        # adapter subset just for the reset snapshot.
        self._snapshot = [p.detach().clone().cpu() for p in self._params]

        for p in predictor.parameters():
            p.requires_grad = False
        for p in self._params:
            p.requires_grad = True

        self.opt = torch.optim.AdamW(self._params, lr=lr)

    def reset(self, game_idx: Optional[int] = None) -> None:
        """Restore the pristine (checkpoint) weights for the adapted
        subset, clear the observation buffer, and rebuild the optimizer
        (so stale Adam moment estimates from a different game's gradients
        can't bleed into the new game either). Only call this on an
        actual new game -- see the class docstring."""
        with torch.no_grad():
            for p, snap in zip(self._params, self._snapshot):
                p.copy_(snap.to(p.device))
        self.buffer.clear()
        self._n_observed = 0
        self.n_adapt_events = 0
        self.opt = torch.optim.AdamW(self._params, lr=self.lr)
        if game_idx is not None:
            self.game_idx = game_idx

    def observe(
        self,
        frame_t: list,
        action_id: int,
        xy: Optional[tuple],
        frame_t1: list,
    ) -> None:
        """Record one real (frame_t, action, frame_t1) transition just
        observed during play, and fire an adaptation step if this is the
        k-th observation since the last one (or since reset). Cheap to
        call every turn -- the buffer append is O(1) and most calls do not
        trigger a gradient step."""
        if action_id == 0:  # RESET carries no dynamics signal to learn from
            return
        x, y = xy if xy is not None else (0, 0)
        self.buffer.append((frame_t, action_id, x, y, frame_t1))
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
        cur_np, action_ids, xys, nxt_np, masks = [], [], [], [], []
        for frame_t, action_id, x, y, frame_t1 in batch:
            cur_np.append(arc3_frame_to_tensor(frame_t))
            nxt_np.append(arc3_frame_to_tensor(frame_t1))
            action_ids.append(action_id)
            xys.append([x / 63.0, y / 63.0])
            masks.append(patch_change_mask(frame_t, frame_t1))

        import numpy as np

        cur = torch.from_numpy(np.stack(cur_np)).to(self.device)
        nxt = torch.from_numpy(np.stack(nxt_np)).to(self.device)
        action_t = torch.tensor(action_ids, dtype=torch.long, device=self.device)
        xy_t = torch.tensor(xys, dtype=torch.float32, device=self.device)
        mask_t = torch.from_numpy(np.stack(masks)).to(self.device)
        game_t = torch.full((cur.shape[0],), self.game_idx, dtype=torch.long, device=self.device)

        self.predictor.train()
        try:
            for _ in range(self.n_steps):
                with torch.no_grad():
                    cur_feat = self.encoder(cur)
                    next_feat = self.encoder(nxt)
                pred_feat, _gate = self.predictor(cur_feat, action_t, xy_t, game_t)
                loss = weighted_prediction_loss(pred_feat, next_feat, mask_t)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        finally:
            # Always restore eval mode, even if a gradient step raises --
            # the real Q-scoring forward passes (_predict_experts) assume
            # self.predictor is in eval mode and must never silently run
            # in train mode after a failed adaptation step.
            self.predictor.eval()

        self.n_adapt_events += 1
        self.last_adapt_latency_s = time.time() - start
