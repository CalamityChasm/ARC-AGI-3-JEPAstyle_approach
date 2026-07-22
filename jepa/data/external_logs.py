"""Loads (frame_t, action, frame_t+1) transitions from the external Kaggle
"arc-3-logs" dataset (data/arc3_logs.zip, gitignored -- pull with
`kaggle datasets download calamitychasm/arc-3-logs -p <tmp>` and copy the
zip to `data/arc3_logs.zip`; see CLAUDE.md).

That dataset is ~2.6GB uncompressed across 25 per-game JSONL files (one
`exploretransitionlogger.jsonl` per public game), each line already a
complete `{"grid", "action", "next_grid", "score", "done"}` transition --
no need to reconstruct transitions from consecutive frames the way
`trajectories.py` does for our own Stage 0 recordings. Read directly out
of the zip (via `zipfile`) rather than extracting ~2.6GB to disk.

This corpus is much larger than the local Stage 0 recordings (~105k vs
~12k transitions across the same 25 games) but it's someone else's bulk
random-policy scrape, not our own controlled corpus -- lower quality/
diversity per transition. `max_per_game` reservoir-samples a bounded
number of transitions per game so this supplements the local corpus
instead of drowning it out 9-to-1.
"""

import json
import zipfile
from pathlib import Path

import numpy as np

EXTERNAL_LOGS_ZIP = "data/arc3_logs.zip"
_SUFFIX = "_exploretransitionlogger.jsonl"


def _game_id_from_entry(name: str) -> str:
    # "arc3_logs/ls20-9607627b_exploretransitionlogger.jsonl" -> "ls20-9607627b"
    return Path(name).name.removesuffix(_SUFFIX)


def load_external_transitions(
    repo_root: Path, max_per_game: int | None = 2000, seed: int = 0, exclude_games: list | None = None
) -> list:
    """Returns a list of (frame_t, action_id, x, y, frame_t1, changed, game_id)
    tuples in the same shape `trajectories.TransitionDataset` expects.

    `frame_t`/`frame_t1` are wrapped as `[grid]` -- a one-element list
    holding a (64, 64) numpy uint8 array -- to match
    `grid.arc3_frame_to_tensor`'s `frame[0]` indexing while avoiding the
    memory cost of nested Python lists (numpy uint8 is ~800MB total for the
    full corpus vs several GB as nested lists of Python ints).

    This dataset has no per-action (x, y) coordinates (unlike our own
    ACTION6 recordings), so x=y=0 for every transition here -- harmless,
    since ACTION6 usage is rare in random-policy data anyway.

    `max_per_game`: reservoir-samples that many transitions per game file.
    `None` loads everything (~105k transitions total -- fine on RAM, but
    much bigger than the local corpus this is meant to supplement).
    Returns `[]` if the zip hasn't been pulled locally (this dataset is
    optional supplementary data, not a hard dependency).

    `exclude_games` (stage6-game-holdout addition): short game codes
    (e.g. `["r11l", "bp35"]`) whose per-game JSONL entry should be
    skipped entirely -- this dataset covers the same 25 ARC games as the
    local recordings, so a leave-some-games-out experiment must exclude
    them here too, not just from `trajectories.load_all_transitions`.
    """
    zip_path = repo_root / EXTERNAL_LOGS_ZIP
    if not zip_path.exists():
        return []

    rng = np.random.default_rng(seed)
    transitions = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".jsonl"))
        for name in names:
            game_id = _game_id_from_entry(name)
            if exclude_games is not None and any(game_id.startswith(f"{g}-") for g in exclude_games):
                continue
            reservoir = []
            with zf.open(name) as f:
                for i, raw in enumerate(f):
                    raw = raw.strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    grid = np.asarray(obj["grid"], dtype=np.uint8)
                    next_grid = np.asarray(obj["next_grid"], dtype=np.uint8)
                    changed = bool(np.any(grid != next_grid))
                    action_id = int(obj["action"])
                    item = ([grid], action_id, 0, 0, [next_grid], changed, game_id)

                    if max_per_game is None or len(reservoir) < max_per_game:
                        reservoir.append(item)
                    else:
                        j = int(rng.integers(0, i + 1))
                        if j < max_per_game:
                            reservoir[j] = item
            transitions.extend(reservoir)
    return transitions
