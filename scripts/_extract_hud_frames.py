"""Extract a small fixture of **real** boards from a champion (anim) run.

The HUD arm's tests must fire against frames the incumbent actually saw, not
synthetic grids -- a detector that works on hand-made bars proves nothing about
the 25 public games. This pulls a few boards per game out of the run's own
``artifacts/*_p0_events.jsonl`` and writes them gzipped into ``tests/fixtures/``
so the suite is self-contained on a fresh clone.

Boards are stored as one hex character per cell (ARC colors are 0-15), one
string per row, which gzips to a few KB for the whole set.

Usage:
    venv/Scripts/python.exe scripts/_extract_hud_frames.py <run-dir> \\
        [--per-game 3] [--out tests/fixtures/hud_frames.json.gz]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "tests" / "fixtures" / "hud_frames.json.gz"
_HEX = "0123456789abcdef"


def _encode(board: list[list[int]]) -> list[str]:
    return ["".join(_HEX[max(0, min(15, int(v)))] for v in row) for row in board]


def _boards(path: Path, per_game: int) -> list[dict]:
    """Up to ``per_game`` boards from one game, spread across distinct levels
    where the run reached more than one, so the fixture is not all level 0."""
    seen_levels: dict[int, list[list[str]]] = {}
    order: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"board"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        board = event.get("board")
        if not isinstance(board, list) or not board or not isinstance(board[0], list):
            continue
        level = int(event.get("level") or 0)
        if level not in seen_levels:
            seen_levels[level] = _encode(board)
            order.append(level)
        if len(order) >= per_game:
            break
    return [{"level": lv, "rows": seen_levels[lv]} for lv in order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="unpacked `kaggle kernels output` dir")
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    artifacts = sorted((args.run_dir / "artifacts").glob("*_p0_events.jsonl"))
    if not artifacts:
        raise SystemExit(f"no *_p0_events.jsonl under {args.run_dir / 'artifacts'}")

    games = {}
    for path in artifacts:
        game = path.name.split("_p0")[0]
        frames = _boards(path, args.per_game)
        if frames:
            games[game] = frames

    payload = {
        "source_run": str(args.run_dir),
        "note": (
            "Real boards from the champion anim run. One hex char per cell "
            "(ARC color 0-15), one string per row."
        ),
        "games": games,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=1) + "\n").encode("utf-8")
    # mtime=0 so rebuilding the fixture from the same run is byte-reproducible.
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(args.out, "wb"), mtime=0) as fh:
        fh.write(raw)
    total = sum(len(v) for v in games.values())
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes): {len(games)} games, {total} boards")


if __name__ == "__main__":
    main()
