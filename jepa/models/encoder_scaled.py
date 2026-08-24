"""A genuinely larger CNN encoder, additive to `encoder.py`'s `CNNEncoder`.

Why a new class instead of parameterizing `CNNEncoder` in place: `CNNEncoder`
is loaded by name from several already-shipped checkpoint-loading code paths
(Kaggle submission notebook, `hypothesis_agent.py`, various `scripts/
diagnose_*.py`). Changing its architecture in place -- even behind a
default-preserving flag -- would risk a state-dict mismatch the moment
anyone forgets to pass the flag when loading an old checkpoint. A separate
class with its own state-dict shape sidesteps that risk entirely: nothing
that already imports `CNNEncoder` is touched.

Design, addressing CLAUDE.md's brief directly ("more depth ... retaining
higher spatial resolution longer before downsampling ... substantially more
width"):
- `CNNEncoder` is 4 strided convs, one downsample per conv, zero extra
  processing at any resolution before moving to the next. This class keeps
  the same 64->32->16->8 downsampling schedule (so the predictor's existing
  8x8-patch assumption -- `NUM_ACTIONS`-conditioned per-patch residuals,
  ACTION6's salience map, `jepa/grid.py`'s `patch_change_mask` -- all still
  apply unchanged) but inserts `blocks_per_stage` residual blocks *at* each
  resolution before downsampling, so the network actually processes each
  spatial scale rather than passing straight through it. `width_mult` scales
  channel count at every stage.
- `blocks_per_stage=0, width_mult=1.0` is the "size-0" default -- same
  depth/resolution schedule and comparable parameter count to `CNNEncoder`
  (not the literal same state dict; the residual-block generalization
  changes some layer shapes even at zero extra blocks, e.g. the stem is a
  single 3x3 conv here vs. `CNNEncoder`'s first strided 4x4 conv). A
  from-scratch-trained default-config `ScaledCNNEncoder` is a fair
  small-scale regression point, not a drop-in replacement for an existing
  `encoder.pt`/`encoder_moe.pt` checkpoint.
"""

import torch
import torch.nn as nn

from ..grid import NUM_CHANNELS


def _round_to_group(channels: int, group: int = 8) -> int:
    """GroupNorm(8, C) requires C % 8 == 0; round up so any width_mult still
    produces a valid channel count."""
    return max(group, ((channels + group - 1) // group) * group)


class ResBlock(nn.Module):
    """Standard pre-activation-free residual block: two 3x3 convs, GroupNorm,
    GELU, additive skip. Kept deliberately simple (no bottleneck, no SE) --
    the point of this experiment is raw depth/width, not a fancier block."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


def _downsample(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.GELU(),
    )


def _make_stage(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])


class ScaledCNNEncoder(nn.Module):
    """(NUM_CHANNELS, 64, 64) one-hot grid -> (out_channels, 8, 8) feature map,
    same downsampling schedule as `CNNEncoder` but with real per-resolution
    processing and configurable width.

    Args:
        out_channels: feature channels at the 8x8 output (also scaled by
            `width_mult` unless `fixed_out_channels=True`, so a wider network
            hands the predictor a proportionally wider feature map too --
            the predictor's own capacity should scale with this).
        width_mult: multiplies hidden-stage channel count (64 * width_mult,
            rounded to a multiple of 8 for GroupNorm).
        blocks_per_stage: residual blocks inserted at each of the 4
            resolutions (64/32/16/8) before downsampling to the next. 0
            reproduces `CNNEncoder`'s "no extra processing per stage" shape.
    """

    def __init__(
        self,
        out_channels: int = 64,
        in_channels: int = NUM_CHANNELS,
        width_mult: float = 1.0,
        blocks_per_stage: int = 0,
        fixed_out_channels: bool = False,
    ):
        super().__init__()
        hidden = _round_to_group(round(64 * width_mult))
        actual_out = out_channels if fixed_out_channels else _round_to_group(round(out_channels * width_mult))
        self.out_channels = actual_out

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.stage64 = _make_stage(hidden, blocks_per_stage)
        self.down1 = _downsample(hidden, hidden)  # 64 -> 32
        self.stage32 = _make_stage(hidden, blocks_per_stage)
        self.down2 = _downsample(hidden, hidden)  # 32 -> 16
        self.stage16 = _make_stage(hidden, blocks_per_stage)
        self.down3 = _downsample(hidden, hidden)  # 16 -> 8
        self.stage8 = _make_stage(hidden, blocks_per_stage)
        self.out_conv = nn.Conv2d(hidden, actual_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage64(x)
        x = self.down1(x)
        x = self.stage32(x)
        x = self.down2(x)
        x = self.stage16(x)
        x = self.down3(x)
        x = self.stage8(x)
        return self.out_conv(x)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_ema_target(encoder: nn.Module) -> nn.Module:
    """Same pattern as `encoder.py: make_ema_target`, generalized to any
    module (avoids importing `CNNEncoder`'s narrower type hint)."""
    import copy

    target = copy.deepcopy(encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def update_ema_target(target: nn.Module, online: nn.Module, momentum: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.mul_(momentum).add_(op, alpha=1.0 - momentum)
