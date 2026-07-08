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
