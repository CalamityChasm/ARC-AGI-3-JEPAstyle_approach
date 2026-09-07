"""Save each WorldModel revision to disk, per architecture.md's "Per-game
artifact" section: an inspectable, diffable trail of what the agent
believed and when it was wrong.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Anchored to the repo root (parent of this package) rather than left
# relative -- main.py runs with cwd=ARC-AGI-3-Agents/, which would
# otherwise scatter saved revisions under the vendored framework dir
# instead of a predictable repo-root-level location.
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "world_models"


def save_revision(game_id: str, version: int, source: str, note: str, out_dir: Path = DEFAULT_DIR) -> Path:
    game_dir = out_dir / game_id
    game_dir.mkdir(parents=True, exist_ok=True)
    path = game_dir / f"v{version}.py"
    path.write_text(source, encoding="utf-8")

    meta_path = game_dir / f"v{version}.json"
    meta_path.write_text(
        json.dumps({"version": version, "note": note, "timestamp": time.time()}, indent=2),
        encoding="utf-8",
    )
    return path
