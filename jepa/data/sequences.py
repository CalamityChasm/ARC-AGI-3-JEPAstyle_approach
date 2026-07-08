"""Loads ARC-3 recordings as ordered per-episode *sequences* (not shuffled
i.i.d. transitions like `trajectories.py`) for training the Stage 3
recurrent predictor -- a recurrent core only learns anything useful about
carrying history if it's trained on transitions in their original temporal
order, chunked per episode rather than randomly shuffled across episodes.

Local recordings only (not the external arc-3-logs Kaggle dataset): that
dataset is one giant per-game file without clean per-episode boundaries in
the schema (`done` marks episode ends but reconstructing clean chunks from
it wasn't needed for a first working Stage 3 pass -- local recordings
already provide ~150 clean per-episode files, which is enough to validate
whether the recurrent core learns anything at all).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..grid import arc3_frame_to_tensor, patch_change_mask
from .trajectories import RECORDINGS_DIR, _load_frame_lines


def load_all_episodes(repo_root: Path) -> list:
    """Returns a list of episodes; each episode is a list of
    (frame_t, action_id, x, y, frame_t1, changed, game_id) tuples in
    original temporal order, one list per recording file."""
    recordings_dir = repo_root / RECORDINGS_DIR
    episodes = []
    for path in sorted(recordings_dir.glob("*.recording.jsonl")):
        frames = _load_frame_lines(path)
        transitions = []
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x = xy.get("x", 0)
            y = xy.get("y", 0)
            changed = cur["frame"] != nxt["frame"]
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], changed, game_id))
        if transitions:
            episodes.append(transitions)
    return episodes


def build_game_vocab(episodes: list) -> dict:
    game_ids = sorted({t[6] for ep in episodes for t in ep})
    return {g: i for i, g in enumerate(game_ids)}


class EpisodeSequenceDataset(Dataset):
    """Chunks each episode into fixed-length (`seq_len`) windows, dropping
    the ragged remainder at the end of each episode -- simple, and ~150
    episodes at ~80 steps each leaves plenty of full-length chunks without
    needing padding/masking machinery."""

    def __init__(self, episodes: list, game_vocab: dict, seq_len: int = 16):
        self.game_vocab = game_vocab
        self.seq_len = seq_len
        self.chunks = []
        for ep in episodes:
            for start in range(0, len(ep) - seq_len + 1, seq_len):
                self.chunks.append(ep[start : start + seq_len])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        curs, actions, xys, nxts, masks = [], [], [], [], []
        for frame_t, action_id, x, y, frame_t1, _changed, _game_id in chunk:
            curs.append(arc3_frame_to_tensor(frame_t))
            actions.append(action_id)
            xys.append([x / 63.0, y / 63.0])
            nxts.append(arc3_frame_to_tensor(frame_t1))
            masks.append(patch_change_mask(frame_t, frame_t1))
        game_id = chunk[0][6]
        return (
            torch.from_numpy(np.stack(curs)),  # (T, 17, 64, 64)
            torch.tensor(actions, dtype=torch.long),  # (T,)
            torch.tensor(np.array(xys), dtype=torch.float32),  # (T, 2)
            torch.from_numpy(np.stack(nxts)),  # (T, 17, 64, 64)
            torch.from_numpy(np.stack(masks)),  # (T, 8, 8)
            torch.tensor(self.game_vocab[game_id], dtype=torch.long),
        )
