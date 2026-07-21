"""JEPA loss pieces shared by encoder pretraining and dynamics training.

Prediction loss is plain latent MSE against a stop-gradient EMA target
(standard JEPA/BYOL). That alone can collapse to a constant output; the
EMA target's asymmetry already resists this in practice, but we add a
cheap VICReg-style variance term as a second line of defense: it penalizes
the online encoder's per-channel std dropping below a floor across the
batch, which a collapsed (constant) encoder can't satisfy.
"""

import torch
import torch.nn.functional as F

from .grid import CANVAS, NUM_COLORS, PATCH

VARIANCE_FLOOR = 1.0
VARIANCE_WEIGHT = 0.1


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-patch latent MSE. `target` must already be detached (stop-grad)."""
    return F.mse_loss(pred, target)


CHANGED_PATCH_WEIGHT = 8.0


def weighted_prediction_loss(
    pred: torch.Tensor, target: torch.Tensor, patch_changed: torch.Tensor
) -> torch.Tensor:
    """Like prediction_loss, but upweights patches whose pixels actually
    changed. Even "changed" ARC-3 frames are mostly a static grid plus one
    small moving region, so a plain per-patch mean is dominated by
    trivially-unchanged patches and gives almost no gradient signal for the
    handful that carry real dynamics.

    pred, target: (B, C, H, W); patch_changed: (B, H, W) bool.
    """
    err = per_region_error(pred, target)  # (B, H, W)
    weight = 1.0 + CHANGED_PATCH_WEIGHT * patch_changed.float()
    return (err * weight).sum() / weight.sum()


def per_region_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H, W) L2 error per spatial region -- the salience map."""
    return (pred - target).pow(2).mean(dim=1)


def variance_regularizer(feat: torch.Tensor) -> torch.Tensor:
    """feat: (B, C, H, W). Encourages each channel to stay spread out across
    the batch so the encoder can't collapse to a constant output."""
    b = feat.shape[0]
    flat = feat.permute(1, 0, 2, 3).reshape(feat.shape[1], -1)  # (C, B*H*W)
    std = torch.sqrt(flat.var(dim=1) + 1e-4)
    return F.relu(VARIANCE_FLOOR - std).mean()


# --- Stage 6 "object identity" auxiliary loss ---------------------------
#
# scripts/diagnose_encoder_vs_predictor.py's diagnostic B found the encoder
# barely represents object identity: two same-colored patches at different
# grid positions within one frame are only marginally more similar in
# feature space than two differently-colored patches (production checkpoint:
# +0.011 cosine-sim gap, essentially noise). None of the existing losses
# above give the encoder any reason to represent "these two patches are the
# same kind of thing" -- prediction_loss/weighted_prediction_loss only
# compare a patch to *itself* across time, and variance_regularizer only
# cares about per-channel spread, not cross-patch structure. This term adds
# that signal directly: same-color, different-position patches within a
# frame should look similar in feature space; different-color patches
# should look different.
CONTRAST_MARGIN = 0.3


def patch_dominant_color(cur_onehot: torch.Tensor) -> torch.Tensor:
    """(B, NUM_CHANNELS, CANVAS, CANVAS) one-hot grid -> (B, CANVAS//PATCH,
    CANVAS//PATCH) long dominant ARC color (0..NUM_COLORS-1) per patch.

    Derived directly from the one-hot tensor already used elsewhere in the
    training pipeline (no raw color grid needed): sum each patch's
    per-pixel one-hot counts per color channel, then argmax over the real
    color channels only (excludes the pad channel, index NUM_COLORS -- a
    patch that's mostly padding shouldn't get "counted" as that color).
    """
    b = cur_onehot.shape[0]
    color = cur_onehot[:, :NUM_COLORS]  # (B, NUM_COLORS, CANVAS, CANVAS) -- drop the pad channel
    n = CANVAS // PATCH
    patches = color.view(b, NUM_COLORS, n, PATCH, n, PATCH)
    counts = patches.sum(dim=(3, 5))  # (B, NUM_COLORS, n, n)
    return counts.argmax(dim=1)  # (B, n, n)


def same_color_contrastive_loss(
    feat: torch.Tensor, cur_onehot: torch.Tensor, margin: float = CONTRAST_MARGIN
) -> torch.Tensor:
    """Pulls together same-color, different-position patch features within
    a frame; pushes apart different-color patch features (a simple margin
    term on cosine similarity -- not full InfoNCE, per this project's
    established preference for the simplest thing that could work, see
    CLAUDE.md's Stage 6 object-identity writeup).

    feat: (B, C, H, W) encoder output (H=W=CANVAS//PATCH). cur_onehot:
    (B, NUM_CHANNELS, CANVAS, CANVAS), the same batch's raw one-hot input --
    dominant color is derived from this, not from `feat`, so this loss's
    only path to the encoder's gradient is through the similarity
    computation itself.

    The "background" color (a frame's single most common per-patch
    dominant color -- usually the empty/base color a grid is mostly filled
    with) is excluded from both the positive and negative sets: pulling
    together dozens of background patches that already dominate a frame,
    or contrasting them against every other color, would drown out the
    actual foreground-object signal this loss exists to add.
    """
    b, c, h, w = feat.shape
    patch_colors = patch_dominant_color(cur_onehot).view(b, h * w)  # (B, H*W)

    color_counts = F.one_hot(patch_colors, num_classes=NUM_COLORS).sum(dim=1)  # (B, NUM_COLORS)
    background = color_counts.argmax(dim=1)  # (B,)
    is_foreground = patch_colors != background.unsqueeze(1)  # (B, H*W)

    flat = feat.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
    normed = F.normalize(flat, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))  # (B, H*W, H*W) cosine similarity

    same_color = patch_colors.unsqueeze(2) == patch_colors.unsqueeze(1)  # (B, H*W, H*W)
    not_self = ~torch.eye(h * w, dtype=torch.bool, device=feat.device).unsqueeze(0)
    both_fg = is_foreground.unsqueeze(2) & is_foreground.unsqueeze(1)

    pos_mask = same_color & both_fg & not_self
    neg_mask = (~same_color) & both_fg & not_self

    pos_loss = (1.0 - sim)[pos_mask].mean() if pos_mask.any() else feat.new_zeros(())
    neg_loss = F.relu(sim - margin)[neg_mask].mean() if neg_mask.any() else feat.new_zeros(())
    return pos_loss + neg_loss
