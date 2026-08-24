"""A genuinely larger MoE predictor, additive to `moe_predictor.py`'s
`MoEPredictor` -- same reasoning as `encoder_scaled.py`: a separate class so
nothing that already loads a `moe_predictor.pt` checkpoint (Kaggle
submission, `hypothesis_agent.py`, `scripts/diagnose_*.py`) is put at risk.

Generalizes two axes `MoEPredictor` fixed:
- `expert_depth`: `MoEPredictor`'s experts are exactly 2 conv layers
  (1 hidden layer). This lets each expert be deeper (more hidden conv
  layers) as well as wider (`expert_hidden`, already configurable upstream
  but now explicitly scaled alongside the encoder's `width_mult`).
- `num_experts`: unchanged as a direct constructor arg, just noting it's the
  other lever this experiment's "more experts" instruction refers to.

`load_balance_loss`, the `top_k` noisy-gating option, and the
gate/expert-symmetry-breaking init are all reused unchanged from
`moe_predictor.py` (imported, not reimplemented) -- those are training-loss
concerns orthogonal to model size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor import NUM_ACTIONS


class ScaledMoEPredictor(nn.Module):
    def __init__(
        self,
        feature_channels: int = 64,
        action_embed_dim: int = 16,
        num_games: int = 1,
        game_embed_dim: int = 16,
        num_experts: int = 8,
        expert_hidden: int = 64,
        expert_depth: int = 1,
        top_k: int | None = None,
    ):
        """
        expert_depth: number of *hidden* conv layers per expert (each
            `Conv2d(1x1) -> GELU`), before the final projection back to
            `feature_channels`. `expert_depth=1` reproduces `MoEPredictor`'s
            exact per-expert shape (conv -> gelu -> conv). Kept at 1x1
            convs throughout (pointwise, per-patch-independent), matching
            `MoEPredictor`'s own design rationale -- experts specialize on a
            causal *pattern* at a given patch, not a spatial-context
            operation; spatial context is the encoder's job.
        """
        super().__init__()
        if expert_depth < 1:
            raise ValueError(f"expert_depth={expert_depth} must be >= 1")
        self.num_experts = num_experts
        self.top_k = top_k
        self.action_embed = nn.Embedding(NUM_ACTIONS, action_embed_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, action_embed_dim),
            nn.GELU(),
        )
        self.game_embed = nn.Embedding(num_games, game_embed_dim)
        cond_dim = action_embed_dim * 2 + game_embed_dim

        def make_expert() -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(feature_channels + cond_dim, expert_hidden, kernel_size=1),
                nn.GELU(),
            ]
            for _ in range(expert_depth - 1):
                layers += [nn.Conv2d(expert_hidden, expert_hidden, kernel_size=1), nn.GELU()]
            layers.append(nn.Conv2d(expert_hidden, feature_channels, kernel_size=1))
            return nn.Sequential(*layers)

        self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])
        # Same symmetry-breaking rationale as MoEPredictor: small random
        # (not zero) init on each expert's last layer.
        for expert in self.experts:
            last = expert[-1]
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)

        self.gate = nn.Sequential(
            nn.Linear(feature_channels + cond_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, num_experts),
        )
        if top_k is not None:
            if not (1 <= top_k <= num_experts):
                raise ValueError(f"top_k={top_k} must be in [1, num_experts={num_experts}]")
            self.noise_gate = nn.Linear(feature_channels + cond_dim, num_experts)
            nn.init.zeros_(self.noise_gate.weight)
            nn.init.zeros_(self.noise_gate.bias)

    def _condition(
        self, feat: torch.Tensor, action_id: torch.Tensor, xy: torch.Tensor, game_idx: torch.Tensor | None
    ) -> tuple:
        b, _c, h, w = feat.shape
        a_embed = self.action_embed(action_id)
        xy_embed = self.coord_mlp(xy)
        if game_idx is None:
            game_idx = torch.zeros(b, dtype=torch.long, device=feat.device)
        g_embed = self.game_embed(game_idx)
        cond = torch.cat([a_embed, xy_embed, g_embed], dim=-1)
        cond_spatial = cond.view(b, -1, 1, 1).expand(-1, -1, h, w)
        expert_input = torch.cat([feat, cond_spatial], dim=1)
        return cond, cond_spatial, expert_input

    def forward(
        self,
        feat: torch.Tensor,
        action_id: torch.Tensor,
        xy: torch.Tensor,
        game_idx: torch.Tensor | None = None,
    ) -> tuple:
        b = feat.shape[0]
        cond, _cond_spatial, x = self._condition(feat, action_id, xy, game_idx)

        pooled_feat = feat.mean(dim=(2, 3))
        gate_input = torch.cat([pooled_feat, cond], dim=-1)
        gate_logits = self.gate(gate_input)

        if self.top_k is not None:
            if self.training:
                noise_std = F.softplus(self.noise_gate(gate_input))
                gate_logits = gate_logits + torch.randn_like(gate_logits) * noise_std
            top_vals, top_idx = gate_logits.topk(self.top_k, dim=-1)
            masked_logits = torch.full_like(gate_logits, float("-inf"))
            masked_logits.scatter_(1, top_idx, top_vals)
            gate_weights = F.softmax(masked_logits, dim=-1)
        else:
            gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        weights = gate_weights.view(b, self.num_experts, 1, 1, 1)
        residual = (expert_outputs * weights).sum(dim=1)

        return feat + residual, gate_weights

    def predict_all_experts(
        self,
        feat: torch.Tensor,
        action_id: torch.Tensor,
        xy: torch.Tensor,
        game_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _cond, _cond_spatial, x = self._condition(feat, action_id, xy, game_idx)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        return feat.unsqueeze(1) + expert_outputs


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
