"""Stage 6 continuous-game-embedding investigation, Phase 2B(b): builds
training examples that pair a target transition with a window of K other
transitions observed *earlier in the same episode* -- the raw material for
`jepa/models/context_encoder.py: EpisodeContextEncoder`'s meta-learning-
style task inference.

Reuses `jepa/data/sequences.py`'s existing per-episode transition lists
(no changes needed there) rather than building a separate loader --
episodes are already loaded/ordered correctly for this purpose.

Local recordings only, same limitation as `jepa/data/sequences.py` itself
(the external arc-3-logs dataset has no clean per-episode boundaries) --
see experiments/stage6_continuous_game_embedding.md's Phase 2B(b) section
for why this deviation from the fully-matched Phase 2B(a) recipe was
accepted rather than building new plumbing for this scoped test.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..grid import arc3_frame_to_tensor, patch_change_mask
from .sequences import build_game_vocab, load_all_episodes

CONTEXT_WINDOW = 8


class EpisodeContextDataset(Dataset):
    """For every transition at episode-position `i >= context_window`,
    returns the target transition (same shape `jepa.data.trajectories.
    TransitionDataset` produces) plus its `context_window` immediately-
    preceding same-episode transitions' raw frames/actions (to be encoded
    by the shared online encoder at train/eval time -- not precomputed
    here, so gradients can flow back through the encoder from context
    frames too, same as they do for the target transition).

    Positions `i < context_window` (the first few transitions of every
    episode, where a full-size context window isn't yet available) are
    skipped entirely -- simpler than padding/masking machinery, and with
    ~150 episodes at ~80 steps each there's no shortage of valid positions
    even after dropping the first `context_window` of each.
    """

    def __init__(self, episodes: list, game_vocab: dict, context_window: int = CONTEXT_WINDOW):
        self.game_vocab = game_vocab
        self.context_window = context_window
        self.index = []  # list of (episode_idx, position_idx)
        self.episodes = episodes
        for ep_idx, ep in enumerate(episodes):
            for pos in range(context_window, len(ep)):
                self.index.append((ep_idx, pos))

    def __len__(self) -> int:
        return len(self.index)

    def sample_weights(self, changed_weight: float = 3.0) -> list:
        """Per-sample weights for a WeightedRandomSampler, oversampling
        target transitions where the frame actually changed -- mirrors
        `jepa.data.trajectories.TransitionDataset.sample_weights` (Stage
        1's fix for changed-transition scarcity, see CLAUDE.md iteration
        #2); only the *target* transition's changed flag matters here,
        not the context window's."""
        weights = []
        for ep_idx, pos in self.index:
            changed = self.episodes[ep_idx][pos][5]
            weights.append(changed_weight if changed else 1.0)
        return weights

    def __getitem__(self, idx: int):
        ep_idx, pos = self.index[idx]
        ep = self.episodes[ep_idx]

        frame_t, action_id, x, y, frame_t1, _changed, game_id = ep[pos]
        target_cur = arc3_frame_to_tensor(frame_t)
        target_action = action_id
        target_xy = np.array([x / 63.0, y / 63.0], dtype=np.float32)
        target_nxt = arc3_frame_to_tensor(frame_t1)
        target_mask = patch_change_mask(frame_t, frame_t1)
        target_game_idx = self.game_vocab[game_id]

        ctx_curs, ctx_actions, ctx_nxts = [], [], []
        for c_pos in range(pos - self.context_window, pos):
            c_frame_t, c_action_id, _cx, _cy, c_frame_t1, _c_changed, _c_game_id = ep[c_pos]
            ctx_curs.append(arc3_frame_to_tensor(c_frame_t))
            ctx_actions.append(c_action_id)
            ctx_nxts.append(arc3_frame_to_tensor(c_frame_t1))

        return (
            torch.from_numpy(target_cur),  # (17, 64, 64)
            torch.tensor(target_action, dtype=torch.long),
            torch.from_numpy(target_xy),  # (2,)
            torch.from_numpy(target_nxt),  # (17, 64, 64)
            torch.from_numpy(target_mask),  # (8, 8)
            torch.tensor(target_game_idx, dtype=torch.long),
            torch.from_numpy(np.stack(ctx_curs)),  # (K, 17, 64, 64)
            torch.tensor(ctx_actions, dtype=torch.long),  # (K,)
            torch.from_numpy(np.stack(ctx_nxts)),  # (K, 17, 64, 64)
        )


def load_episode_context_dataset(
    repo_root: Path, context_window: int = CONTEXT_WINDOW, exclude_games: list | None = None,
    name_substrings: list | None = None,
) -> tuple:
    """Convenience wrapper: loads episodes, builds a shared game vocab,
    and returns (EpisodeContextDataset, game_vocab) -- mirrors the
    load-then-build-dataset pattern every other jepa/data/*.py module
    uses. `exclude_games`/`name_substrings` are passed straight through to
    `jepa.data.sequences.load_all_episodes`."""
    episodes = load_all_episodes(repo_root, exclude_games=exclude_games, name_substrings=name_substrings)
    game_vocab = build_game_vocab(episodes)
    dataset = EpisodeContextDataset(episodes, game_vocab, context_window=context_window)
    return dataset, game_vocab
