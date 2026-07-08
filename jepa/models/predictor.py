"""Action-conditioned one-step latent predictor.

Not Mamba yet (plan.md Stage 1 explicitly defers that to Stage 3) -- a
single-step predictor: given the current feature map s_t and the action
taken, predict the next feature map s_hat_{t+1}. The action embedding is
broadcast to every spatial cell and concatenated as extra channels, then a
couple of 1x1 convs (pointwise MLP per patch) produce the prediction. Kept
spatial (rather than collapsing to one global vector) so per-region
prediction error -- the salience map -- falls out directly.
"""

import torch
import torch.nn as nn

NUM_ACTIONS = 8  # RESET=0, ACTION1-5=1-5, ACTION6=6 (complex), ACTION7=7


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        feature_channels: int = 64,
        action_embed_dim: int = 16,
        num_games: int = 1,
        game_embed_dim: int = 16,
    ):
        super().__init__()
        self.action_embed = nn.Embedding(NUM_ACTIONS, action_embed_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, action_embed_dim),
            nn.GELU(),
        )
        # The same action id means a different effect in every game, so the
        # predictor needs to know which game it's looking at -- otherwise
        # it's fitting one shared mapping across mutually-inconsistent
        # per-game action semantics. NUM_GAMES=1 (the default) disables this
        # (single always-index-0 embedding) for callers that don't have a
        # game vocabulary.
        self.game_embed = nn.Embedding(num_games, game_embed_dim)
        cond_dim = action_embed_dim * 2 + game_embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels + cond_dim, feature_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )

    def forward(
        self,
        feat: torch.Tensor,
        action_id: torch.Tensor,
        xy: torch.Tensor,
        game_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        feat: (B, C, H, W) current feature map
        action_id: (B,) long, action ids 0-7
        xy: (B, 2) float, normalized (x, y) in [0, 1] (0 if not a complex action)
        game_idx: (B,) long, index into the game vocabulary (0 if omitted)
        """
        b, _, h, w = feat.shape
        a_embed = self.action_embed(action_id)  # (B, E)
        xy_embed = self.coord_mlp(xy)  # (B, E)
        if game_idx is None:
            game_idx = torch.zeros(b, dtype=torch.long, device=feat.device)
        g_embed = self.game_embed(game_idx)  # (B, G)
        cond = torch.cat([a_embed, xy_embed, g_embed], dim=-1)
        cond = cond.view(b, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([feat, cond], dim=1)
        # residual: predict the *change* in latent state, not the raw next state
        return feat + self.net(x)
