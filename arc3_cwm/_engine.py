"""Import shim for the rescued `llm_engine` package.

`llm_engine` lives inside the Kaggle dataset staging tree
(`kaggle_submission_llm_world_engine/dataset_stage/`) rather than at the
repo root, because that tree's layout is dictated by how Kaggle mounts a
dataset. Rather than copy it (a second copy is exactly how this project
lost the original to begin with -- see CLAUDE.md's rescue notes), this
module puts that directory on `sys.path` once and re-exports the pieces
the backtest needs.

Nothing here modifies the engine. The backtest is an instrument pointed
at the engine, not a fork of it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Override for environments that do not have this repo's layout -- notably
#: a Kaggle dataset mount, where `llm_engine` sits beside `arc3_cwm` rather
#: than under `kaggle_submission_llm_world_engine/dataset_stage/`.
_ENV_OVERRIDE = "ARC3_CWM_ENGINE_DIR"


def _resolve_stage_dir() -> Path:
    override = os.environ.get(_ENV_OVERRIDE, "").strip()
    if override:
        candidate = Path(override)
        if not (candidate / "llm_engine").is_dir():
            raise RuntimeError(
                f"{_ENV_OVERRIDE}={override!r} but no llm_engine/ directory is there"
            )
        return candidate

    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "kaggle_submission_llm_world_engine" / "dataset_stage",
        here.parent,  # llm_engine/ sitting beside arc3_cwm/ (dataset mount)
    ]
    for candidate in candidates:
        if (candidate / "llm_engine").is_dir():
            return candidate
    raise RuntimeError(
        "cannot locate llm_engine/; looked in "
        + repr([str(c) for c in candidates])
        + f" -- set {_ENV_OVERRIDE} to its parent directory"
    )


_STAGE_DIR = _resolve_stage_dir()

if str(_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE_DIR))

from llm_engine.diff import format_diff, format_grid  # noqa: E402
from llm_engine.replay import ReplayResult, replay  # noqa: E402
from llm_engine.types import (  # noqa: E402
    ALL_ACTIONS,
    Action,
    GameTranscript,
    Grid,
    Transition,
)
from llm_engine.world_model import (  # noqa: E402
    WORLD_MODEL_SKELETON,
    LoadResult,
    WorldModelProtocol,
    load_world_model,
    safe_predict,
)

__all__ = [
    "ALL_ACTIONS",
    "Action",
    "GameTranscript",
    "Grid",
    "LoadResult",
    "ReplayResult",
    "Transition",
    "WORLD_MODEL_SKELETON",
    "WorldModelProtocol",
    "format_diff",
    "format_grid",
    "load_world_model",
    "replay",
    "safe_predict",
    "stage_dir",
]


def stage_dir() -> Path:
    """Where `llm_engine` was imported from -- surfaced so a failing test
    can say *which* copy it loaded, not just that an import failed."""
    return _STAGE_DIR
