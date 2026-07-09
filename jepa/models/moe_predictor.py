"""Stage 4 (plan.md): mixture-of-gated-experts predictor.

Replaces the monolithic one-step predictor with K small expert networks and
a gate that learns which expert(s) apply to a given (state, action):

    s_hat_{t+1} = feat + sum_k g_k(feat, action, xy, game) * E_k(feat, action, xy, game)

Each expert is meant to specialize on one atomic causal pattern (translate,
recolor, appear/disappear, etc. -- architecture.md's list), discovered
during training rather than hand-categorized; the gate learns what context
activates each one. The combiner network architecture.md also describes
(for blending pairs of experts into compound effects) is explicitly
deferred here, per plan.md's own Stage 4 scope ("Defer the combiner
network").

K defaults to 8, not architecture.md's 16-24 -- that range assumes
pretraining on a 96GB box across diverse generated grid envs (MiniGrid/
Sokoban/Crafter/procgen). This project doesn't have that data source built
(deferred, see CLAUDE.md's Stage 1 "Next steps" history) or that compute;
training on the existing ~55k-transition ARC-3 corpus with too many
experts risks exactly the "expert collapse / data dilution" failure mode
architecture.md's own weak-points table warns about. Scale K up if a
future session adds more diverse training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor import NUM_ACTIONS


class MoEPredictor(nn.Module):
    def __init__(
        self,
        feature_channels: int = 64,
        action_embed_dim: int = 16,
        num_games: int = 1,
        game_embed_dim: int = 16,
        num_experts: int = 8,
        expert_hidden: int = 64,
        top_k: int | None = None,
    ):
        """
        top_k: if set (Shazeer et al.-style "noisy top-k gating"), the gate
            keeps only the top_k highest-scoring experts per example
            (softmax over just those, -inf/zero elsewhere) instead of a
            dense soft blend over all K -- forces genuinely sparse,
            per-example routing rather than letting the optimizer settle
            for "blend everyone a little," which is what a first attempt
            at dense gating converged to regardless of load-balance-loss
            tuning (see CLAUDE.md's Stage 4 status, items 1-5). During
            training, per-expert trainable noise is added to the gate
            logits before top-k selection (encourages exploration across
            experts early on); at eval time the raw logits are used
            directly. None (the default) reproduces the original dense
            softmax gate.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.action_embed = nn.Embedding(NUM_ACTIONS, action_embed_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, action_embed_dim),
            nn.GELU(),
        )
        self.game_embed = nn.Embedding(num_games, game_embed_dim)
        cond_dim = action_embed_dim * 2 + game_embed_dim

        # Each expert is a small pointwise (1x1-conv) MLP over
        # [feat; broadcast cond] -- deliberately simple/shallow per-expert,
        # since the point is many *specialized* small functions, not one
        # deep general one (that's what the monolithic predictor already
        # tried).
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feature_channels + cond_dim, expert_hidden, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(expert_hidden, feature_channels, kernel_size=1),
                )
                for _ in range(num_experts)
            ]
        )
        # NOT zero-initialized (unlike the monolithic ActionConditionedPredictor):
        # if every expert's last layer starts at exact zero, every expert
        # produces bit-for-bit identical output (zero) regardless of its
        # earlier layers, so with a uniform gate every expert receives the
        # *exact same* gradient and stays identical forever -- a genuine
        # symmetry that gradient descent can't break on its own. Scaling
        # down (rather than zeroing) the last layer keeps outputs small at
        # init without collapsing that symmetry.
        for expert in self.experts:
            last = expert[-1]
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)

        # Gate: pooled (feat, cond) -> softmax over experts.
        self.gate = nn.Sequential(
            nn.Linear(feature_channels + cond_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, num_experts),
        )
        if top_k is not None:
            if not (1 <= top_k <= num_experts):
                raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
            # Per-expert trainable noise scale (softplus-transformed, so
            # always positive), same pooled input as the gate itself.
            self.noise_gate = nn.Linear(feature_channels + cond_dim, num_experts)
            nn.init.zeros_(self.noise_gate.weight)
            nn.init.zeros_(self.noise_gate.bias)

    def forward(
        self,
        feat: torch.Tensor,
        action_id: torch.Tensor,
        xy: torch.Tensor,
        game_idx: torch.Tensor | None = None,
    ) -> tuple:
        """
        feat: (B, C, H, W) current feature map
        action_id: (B,) long, action ids 0-7
        xy: (B, 2) float, normalized (x, y) in [0, 1]
        game_idx: (B,) long, index into the game vocabulary (0 if omitted)

        Returns (predicted_next_feat, gate_weights) -- gate_weights (B, K)
        is exposed for the load-balancing loss and for inspecting whether
        experts actually specialize (see jepa/train_moe_predictor.py).
        """
        b, c, h, w = feat.shape
        a_embed = self.action_embed(action_id)
        xy_embed = self.coord_mlp(xy)
        if game_idx is None:
            game_idx = torch.zeros(b, dtype=torch.long, device=feat.device)
        g_embed = self.game_embed(game_idx)
        cond = torch.cat([a_embed, xy_embed, g_embed], dim=-1)  # (B, cond_dim)

        pooled_feat = feat.mean(dim=(2, 3))  # (B, C)
        gate_input = torch.cat([pooled_feat, cond], dim=-1)
        gate_logits = self.gate(gate_input)  # (B, K)

        if self.top_k is not None:
            if self.training:
                noise_std = F.softplus(self.noise_gate(gate_input))
                gate_logits = gate_logits + torch.randn_like(gate_logits) * noise_std
            top_vals, top_idx = gate_logits.topk(self.top_k, dim=-1)
            masked_logits = torch.full_like(gate_logits, float("-inf"))
            masked_logits.scatter_(1, top_idx, top_vals)
            gate_weights = F.softmax(masked_logits, dim=-1)  # (B, K), zero outside top_k
        else:
            gate_weights = F.softmax(gate_logits, dim=-1)  # (B, K)

        cond_spatial = cond.view(b, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([feat, cond_spatial], dim=1)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K, C, H, W)
        weights = gate_weights.view(b, self.num_experts, 1, 1, 1)
        residual = (expert_outputs * weights).sum(dim=1)  # (B, C, H, W)

        return feat + residual, gate_weights


def load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer-style auxiliary loss: N * sum_i(f_i * P_i), where
    f_i is the fraction of the batch routed (top-1) to expert i and P_i is
    that expert's average softmax probability over the batch. Minimized
    (value 1) when usage is uniform, maximized (value N) when collapsed to
    a single expert -- penalizes total collapse without forcing perfectly
    even utilization (tolerates a genuinely dominant expert for a common
    pattern, per architecture.md's own framing).

    gate_weights: (B, K) softmax weights from MoEPredictor.forward.
    """
    num_experts = gate_weights.shape[1]
    top1 = gate_weights.argmax(dim=-1)  # (B,)
    f = torch.bincount(top1, minlength=num_experts).float() / gate_weights.shape[0]  # (K,)
    p = gate_weights.mean(dim=0)  # (K,)
    return num_experts * (f * p).sum()
