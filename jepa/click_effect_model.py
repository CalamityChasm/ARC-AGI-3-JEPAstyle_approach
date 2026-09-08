"""Small, purpose-built model predicting whether an ACTION6 click at a
given local board position will produce any visible effect (and,
secondarily, whether it wins a level) -- from LOCAL structural context
(the small pixel neighborhood around the click point, plus the clicked
segment's shape/color/size) rather than the whole-board latent
representation the production MoE predictor uses.

Why this exists, and why it's scoped differently from that predictor:
scripts/diagnose_state_similarity.py found that an EXACT local-patch match
between two otherwise-unrelated board states predicts the click's effect
("did anything change") with ~100% consistency in 9 of 12 games tested --
a real, local, largely game-internal signal, distinct from the whole-board
prediction task the production JEPA world model already struggles to
generalize (Stage 6 addendum: 13 independent interventions, 12 failures on
cross-game transfer). Exact-match is brittle, though -- this model is the
"soft"/learned version: pretrained across many games' local-patch data,
then further specialized to whichever game is actually being played via
jepa/click_effect_adapter.py's test-time adaptation (same ANIL-style
restricted-subset recipe already validated by jepa/test_time_adapter.py
on the production predictor -- the one lever in this project's whole
Stage 6 investigation that showed real, positive, dialable cross-game
signal, precisely because it adapts live rather than trying to generalize
zero-shot).

Deliberately NOT reusing the production CNNEncoder/MoEPredictor -- this
model is small (operates on a 7x7 patch + 5 scalar segment features, not a
64x64 frame) and independent, so pretraining/adapting it can never
interfere with the production checkpoint.
"""

import torch
import torch.nn as nn

PATCH_SIZE = 7  # matches scripts/diagnose_state_similarity.py's PATCH_RADIUS=3
NUM_COLORS = 17  # 0-15 ARC colors + the 255 pad sentinel used for edge-clipped patches
SEGMENT_FEATURE_DIM = 5  # color, area, is_rectangle, width, height (all normalized)


def _color_index(patch_uint8: torch.Tensor) -> torch.Tensor:
    """Maps the raw uint8 patch (0-15 real colors, 255 pad sentinel) to a
    dense 0-16 index for the embedding lookup."""
    idx = patch_uint8.clone().long()
    idx[idx == 255] = 16
    return idx


class ClickEffectModel(nn.Module):
    def __init__(self, color_embed_dim: int = 4, trunk_dim: int = 32) -> None:
        super().__init__()
        self.color_embed = nn.Embedding(NUM_COLORS, color_embed_dim)
        self.patch_conv = nn.Sequential(
            nn.Conv2d(color_embed_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.segment_fc = nn.Sequential(
            nn.Linear(SEGMENT_FEATURE_DIM, 16),
            nn.ReLU(),
        )
        # Adaptable subset (test-time-adapted live, see
        # jepa/click_effect_adapter.py) -- everything above this point
        # (color_embed, patch_conv, segment_fc) stays frozen after
        # pretraining, mirroring jepa/test_time_adapter.py's own
        # "encoder frozen, last layers adapted" split.
        self.trunk = nn.Sequential(
            nn.Linear(16 + 16, trunk_dim),
            nn.ReLU(),
        )
        self.frame_changed_head = nn.Linear(trunk_dim, 1)
        self.win_head = nn.Linear(trunk_dim, 1)

    def forward(self, patch: torch.Tensor, segment_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """patch: (B, 7, 7) uint8/long raw color indices (255 = pad).
        segment_features: (B, 5) float, pre-normalized."""
        idx = _color_index(patch)
        embedded = self.color_embed(idx)  # (B, 7, 7, C)
        embedded = embedded.permute(0, 3, 1, 2)  # (B, C, 7, 7)
        patch_feat = self.patch_conv(embedded).flatten(1)  # (B, 16)
        seg_feat = self.segment_fc(segment_features)  # (B, 16)
        fused = torch.cat([patch_feat, seg_feat], dim=1)
        trunk_out = self.trunk(fused)
        return self.frame_changed_head(trunk_out).squeeze(-1), self.win_head(trunk_out).squeeze(-1)


def get_adapter_params(model: ClickEffectModel) -> list:
    """The restricted, test-time-adaptable subset -- trunk + both output
    heads (small: 16*32+32 + 32*2 + 2 ~= 610 wired to a bias thanks to also
    the trunk's own bias, well under 1K params). Kept as a single shared
    definition, same rationale as jepa/test_time_adapter.py's own
    get_adapter_params -- the pretraining script and the live adapter must
    never silently disagree on which params get adapted."""
    params = []
    params.extend(model.trunk.parameters())
    params.extend(model.frame_changed_head.parameters())
    params.extend(model.win_head.parameters())
    return params
