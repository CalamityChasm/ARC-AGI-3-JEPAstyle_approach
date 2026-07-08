"""Stage 3 (plan.md): recurrent memory core layered on the Stage 1
one-step predictor.

plan.md calls for "the Mamba core" here; `mamba-ssm` has no prebuilt wheel
for this Windows box, and building it from source requires a local CUDA
toolkit exactly matching torch's build version -- this box has CUDA 13.0
via `nvcc` vs. torch's `cu126` build, a real mismatch (see CLAUDE.md for
the failed build log), not just a missing wheel to chase down. A
`GRUCell`-based recurrent core satisfies the actual stated requirement --
"carries compressed history across the episode" -- without that fragile
dependency. Swap for real Mamba later if this box's CUDA toolkit and
torch's build ever get aligned.

Design: keep Stage 1's spatial, per-patch residual predictor mostly
unchanged (so the salience-map deliverable and existing intuition still
apply) -- fold a persistent hidden state in as one more conditioning
vector alongside action/xy/game, updated once per step from a pooled
summary of the current feature map, and broadcast back out spatially. The
hidden state must be reset at the start of each new episode (`init_hidden`
gives zeros) -- carrying it across unrelated episodes/games would just be
noise.
"""

import torch
import torch.nn as nn

from .predictor import NUM_ACTIONS


class RecurrentActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        feature_channels: int = 64,
        action_embed_dim: int = 16,
        num_games: int = 1,
        game_embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_embed = nn.Embedding(NUM_ACTIONS, action_embed_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, action_embed_dim),
            nn.GELU(),
        )
        self.game_embed = nn.Embedding(num_games, game_embed_dim)

        cond_dim = action_embed_dim * 2 + game_embed_dim
        self.gru_cell = nn.GRUCell(feature_channels + cond_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Conv2d(feature_channels + cond_dim + hidden_dim, feature_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )
        # Same zero-init rationale as ActionConditionedPredictor: start as
        # an exact identity function so the model never has to first climb
        # out of a random-noise regime before it can commit to real signal.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        feat: torch.Tensor,
        action_id: torch.Tensor,
        xy: torch.Tensor,
        hidden: torch.Tensor,
        game_idx: torch.Tensor | None = None,
    ) -> tuple:
        """
        feat: (B, C, H, W) current feature map
        action_id: (B,) long, action ids 0-7
        xy: (B, 2) float, normalized (x, y) in [0, 1]
        hidden: (B, hidden_dim) recurrent state carried from the previous
            step (zeros at episode start -- see `init_hidden`)
        game_idx: (B,) long, index into the game vocabulary (0 if omitted)

        Returns (predicted_next_feat, new_hidden).
        """
        b, _, h, w = feat.shape
        a_embed = self.action_embed(action_id)  # (B, E)
        xy_embed = self.coord_mlp(xy)  # (B, E)
        if game_idx is None:
            game_idx = torch.zeros(b, dtype=torch.long, device=feat.device)
        g_embed = self.game_embed(game_idx)  # (B, G)
        cond = torch.cat([a_embed, xy_embed, g_embed], dim=-1)  # (B, cond_dim)

        pooled_feat = feat.mean(dim=(2, 3))  # (B, C) -- cheap summary for the GRU
        gru_input = torch.cat([pooled_feat, cond], dim=-1)
        new_hidden = self.gru_cell(gru_input, hidden)  # (B, hidden_dim)

        cond_spatial = cond.view(b, -1, 1, 1).expand(-1, -1, h, w)
        hidden_spatial = new_hidden.view(b, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([feat, cond_spatial, hidden_spatial], dim=1)
        pred_feat = feat + self.net(x)
        return pred_feat, new_hidden
