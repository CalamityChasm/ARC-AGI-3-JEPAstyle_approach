"""Stage 6 continuous-game-embedding investigation, Phase 2B: a small
network that maps the *content* of the current frame's pooled encoder
features into a continuous game/context descriptor, instead of a
categorical `game_id -> embedding table` lookup.

Motivation (see experiments/stage6_continuous_game_embedding.md): the MoE
predictor's held-out-game generalization gap survived four independent
fixes (ablating game-id conditioning, confirming the encoder is fine, an
anti-collapse loss, simulated training-time unfamiliarity -- all
documented in CLAUDE.md's Stage 6 addendum), and Phase 1 of this
investigation found Stage 3's recurrent hidden state doesn't close it
either. The common thread across every categorical/lookup-based approach:
a genuinely novel game_id falls back to a fixed, undertrained index (0),
which carries no information about what's actually on screen. A
*content-derived* descriptor has no such fallback -- the same function is
applied to any input, familiar or not, so there's no discontinuity
between "a game_id I've seen" and "a game_id I haven't."

This is deliberately the cheaper, single-frame version (Phase 2B(a) in
the investigation's decision tree) -- no episode-context bookkeeping, no
new data-loading machinery. If this alone doesn't help, Phase 2B(b) would
extend it to a multi-transition context (a short window of recent
(feat, action, feat') tuples from the current episode, meta-learning
style) -- a materially bigger build, only worth it once this cheaper
version is shown insufficient.
"""

import torch.nn as nn


class FrameContextEncoder(nn.Module):
    """pooled current-frame features (B, feature_channels) -> a continuous
    context embedding (B, embed_dim), the same shape/role `MoEPredictor`'s
    `game_embed` lookup table produces, but computed from content instead
    of looked up by category -- see module docstring for why that matters
    for held-out-game generalization specifically."""

    def __init__(self, feature_channels: int = 64, embed_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, pooled_feat):
        """pooled_feat: (B, feature_channels) -> (B, embed_dim)."""
        return self.net(pooled_feat)
