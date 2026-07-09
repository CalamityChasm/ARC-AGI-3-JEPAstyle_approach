"""Stage 5: decoupled value head (architecture.md's "Decoupled Heads").

A small MLP off a pooled summary of the shared encoder's feature map,
predicting expected discounted future progress (see
jepa/data/value_targets.py for how the target is built). Kept fully
separate from the dynamics predictor -- same rationale architecture.md
gives for DIAMOND-style decoupled reward/termination heads, just applied
to this project's JEPA latent instead of raw pixels.
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, feature_channels: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) encoder feature map -> (B,) scalar value estimate."""
        pooled = feat.mean(dim=(2, 3))
        return self.net(pooled).squeeze(-1)
