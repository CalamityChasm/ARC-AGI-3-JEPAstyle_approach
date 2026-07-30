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

import torch
import torch.nn as nn

from .predictor import NUM_ACTIONS


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


class EpisodeContextEncoder(nn.Module):
    """Phase 2B(b): infers a continuous game/context descriptor from a
    small window of K *other* (feat_t, action, feat_t1) transitions
    observed earlier in the same episode -- not the transition being
    predicted -- rather than a single current frame (`FrameContextEncoder`)
    or a category lookup. Meta-learning-style task inference (PEARL/
    VariBAD): infer "what kind of game is this" from a handful of
    observed exemplars.

    Each context transition is summarized as `[pooled_feat_t;
    action_embed; pooled_feat_t1 - pooled_feat_t]` (what state, what
    action, what changed), mapped through a small per-transition MLP,
    then mean-pooled across the K context transitions -- a Deep-Sets-
    style phi-then-pool, deliberately permutation-invariant since which
    order the K context transitions happen to be sampled in shouldn't
    matter.

    This module only *combines* already-encoded, already-pooled
    per-transition summaries -- it does not run the shared CNNEncoder
    itself. The caller (see jepa/train_context_moe_predictor.py) is
    responsible for running each context transition's raw frames through
    the same online encoder used for the target transition, so context
    and target share one latent space.
    """

    def __init__(
        self,
        feature_channels: int = 64,
        action_embed_dim: int = 16,
        embed_dim: int = 16,
        hidden: int = 64,
    ):
        super().__init__()
        self.action_embed = nn.Embedding(NUM_ACTIONS, action_embed_dim)
        summary_dim = feature_channels * 2 + action_embed_dim
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(
        self, pooled_feat_t: torch.Tensor, action_id: torch.Tensor, pooled_feat_t1: torch.Tensor
    ) -> torch.Tensor:
        """pooled_feat_t, pooled_feat_t1: (B, K, feature_channels);
        action_id: (B, K) long. Returns (B, embed_dim)."""
        a_embed = self.action_embed(action_id)  # (B, K, action_embed_dim)
        delta = pooled_feat_t1 - pooled_feat_t  # (B, K, feature_channels)
        summary = torch.cat([pooled_feat_t, a_embed, delta], dim=-1)  # (B, K, summary_dim)
        per_transition_embed = self.net(summary)  # (B, K, embed_dim)
        return per_transition_embed.mean(dim=1)  # (B, embed_dim)
