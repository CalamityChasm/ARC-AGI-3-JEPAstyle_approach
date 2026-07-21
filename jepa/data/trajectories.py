"""Loads (frame_t, action_t, frame_t+1) transitions out of the Stage 0
recording JSONL files (ARC-AGI-3-Agents/recordings/*.recording.jsonl).

Within one recording file, line i+1's `action_input` is the action that was
taken *from* the state in line i to produce the state in line i+1 (see
agents/agent.py: append_frame is only called with the FrameData returned by
take_action(action), which already carries that action's action_input).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..grid import arc3_frame_to_tensor, patch_change_mask

RECORDINGS_DIR = "ARC-AGI-3-Agents/recordings"


def _load_frame_lines(path: Path) -> list:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            event = json.loads(raw)
            data = event.get("data", {})
            if "frame" in data and "action_input" in data and data["frame"]:
                lines.append(data)
    return lines


def load_all_transitions(repo_root: Path, name_substrings: list | None = None) -> list:
    """Returns a list of (frame_t, action_id, x, y, frame_t+1, changed, game_id) tuples.

    `changed` is a cheap pixel-level flag (frame_t != frame_t+1). A large
    fraction of random-policy transitions are exact no-ops (action had no
    visible effect), which a naive latent-MSE loss/average can drown in --
    it's exposed here so training can rebalance towards informative
    transitions instead of mostly learning "predict no change".

    `game_id` matters more than it might look: the same action id means a
    completely different effect in each of the 25 games, so a predictor
    that can't tell which game it's in is being asked to fit 25
    mutually-inconsistent action->effect mappings at once.

    `name_substrings` (stage6-object-identity addition): if given, only
    recording files whose name contains at least one of these substrings
    are loaded (e.g. `[".random.", ".solver."]`). Needed because
    `ARC-AGI-3-Agents/recordings/` can end up holding files from several
    different sources at once (random-policy corpus, search-harvest,
    ad-hoc agent-comparison eval runs, ...) -- an apples-to-apples
    comparison against a specific prior checkpoint needs the exact same
    file set that checkpoint was trained on, not "whatever happens to be
    in the directory right now." None (the default) preserves the
    original behavior of loading every recording file present.
    """
    recordings_dir = repo_root / RECORDINGS_DIR
    transitions = []
    for path in sorted(recordings_dir.glob("*.recording.jsonl")):
        if name_substrings is not None and not any(s in path.name for s in name_substrings):
            continue
        frames = _load_frame_lines(path)
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x = xy.get("x", 0)
            y = xy.get("y", 0)
            changed = cur["frame"] != nxt["frame"]
            game_id = cur.get("game_id", "unknown")
            transitions.append(
                (cur["frame"], action_id, x, y, nxt["frame"], changed, game_id)
            )
    return transitions


def build_game_vocab(transitions: list) -> dict:
    game_ids = sorted({t[6] for t in transitions})
    return {g: i for i, g in enumerate(game_ids)}


class TransitionDataset(Dataset):
    def __init__(self, transitions: list, game_vocab: dict):
        self.transitions = transitions
        self.game_vocab = game_vocab

    def __len__(self) -> int:
        return len(self.transitions)

    def sample_weights(self, changed_weight: float = 3.0) -> list:
        """Per-sample weights for a WeightedRandomSampler, oversampling
        transitions where the frame actually changed."""
        return [changed_weight if t[5] else 1.0 for t in self.transitions]

    def __getitem__(self, idx: int):
        frame_t, action_id, x, y, frame_t1, _changed, game_id = self.transitions[idx]
        cur = arc3_frame_to_tensor(frame_t)
        nxt = arc3_frame_to_tensor(frame_t1)
        xy_norm = np.array([x / 63.0, y / 63.0], dtype=np.float32)
        patch_mask = patch_change_mask(frame_t, frame_t1)  # (8, 8) bool
        return (
            torch.from_numpy(cur),
            torch.tensor(action_id, dtype=torch.long),
            torch.from_numpy(xy_norm),
            torch.from_numpy(nxt),
            torch.from_numpy(patch_mask),
            torch.tensor(self.game_vocab[game_id], dtype=torch.long),
        )
