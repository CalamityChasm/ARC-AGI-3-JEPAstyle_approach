"""Small CNN encoder: (17, 64, 64) one-hot grid -> (C, 8, 8) spatial feature map.

Deliberately small -- this is a data-bound problem (plan.md), not a
capacity-bound one. Four stride-2 convs take 64x64 down to 8x8, so each
output cell corresponds to an 8x8 pixel patch (matches the salience map's
"per-region" granularity).
"""

import copy

import torch
import torch.nn as nn

from ..grid import NUM_CHANNELS


class CNNEncoder(nn.Module):
    def __init__(self, out_channels: int = 64, in_channels: int = NUM_CHANNELS):
        super().__init__()
        hidden = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, stride=1, padding=1),  # 8->8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_ema_target(encoder: CNNEncoder) -> CNNEncoder:
    """Deep copy of `encoder` with grads disabled, used as the JEPA target."""
    target = copy.deepcopy(encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def update_ema_target(target: CNNEncoder, online: CNNEncoder, momentum: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.mul_(momentum).add_(op, alpha=1.0 - momentum)
