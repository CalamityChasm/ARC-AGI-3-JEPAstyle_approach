"""Would a restart-at-stall trigger ever fire with any runway left?

The stall count that matters is in *analysis turns* (the unit `ToolAgent.analyze`
increments), not acting turns -- 36% of anim's turns take no action at all.
Turn boundaries come from the transcript headers

    --- analysis_step=N | action=M | HH:MM:SS | tool-agent ---

and level membership from `artifacts/*_events.jsonl` (type == "action" rows,
which carry both `analysis_step` and `level`).

For a threshold T and a cap R this replays each game and reports, for every
firing: the turn it fires on, how many analysis turns remain after it, and how
many actions were actually spent in the turns that remain. The last is the
quantity that decides whether the mechanism has headroom at all -- a restart with
two turns left cannot recover anything.

Usage:
    venv/Scripts/python.exe scripts/simulate_restart_trigger.py <run-dir> \
        --thresholds 8,10,12,15,20 --json <out.json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HEADER_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|", re.M)


def turns_from_transcript(path: Path) -> list[int]:
    """Distinct `analysis_step` values, in order.

    The transcript writes one header per *LLM call*, and at
    `yield_seconds = 180` a turn can make two. `analysis_step` is the counter
    `ToolAgent.analyze` increments, which is the unit a stall threshold counts,
    so duplicates must be collapsed -- not collapsing them made "actions
    remaining after a fire" exceed the game's own action total.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    seen: set[int] = set()
    out: list[int] = []
    for m in HEADER_RE.finditer(text):
        s = int(m.group(1))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def actions_by_step(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """(actions per analysis_step, level at each analysis_step)."""
    acts: dict[int, int] = {}
    lvl: dict[int, int] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "action":
                continue
            s = rec.get("analysis_step")
            if s is None:
                continue
            s = int(s)
            acts[s] = acts.get(s, 0) + 1
            if rec.get("level") is not None:
                lvl[s] = int(rec["level"])
    return acts, lvl


def simulate(run_dir: Path, threshold: int, cap: int) -> dict[str, Any]:
    out = []
    for tpath in sorted((run_dir / "transcripts").glob("*_p0.txt")):
        game = tpath.name[: -len("_p0.txt")]
        gid = game.split("-")[0]
        ev = run_dir / "artifacts" / f"{game}_p0_events.jsonl"
        if not ev.exists():
            continue
        turns = turns_from_transcript(tpath)
        if not turns:
            continue
        acts, lvl_at = actions_by_step(ev)

        # Level at every turn: carry the last observed level forward, since a
        # no-op turn does not emit an action row.
        cur = None
        level_seq = []
        for s in turns:
            if s in lvl_at:
                cur = lvl_at[s]
            level_seq.append(cur)

        fires = []
        since_change = 0
        prev_level = level_seq[0]
        for i, (s, lv) in enumerate(zip(turns, level_seq)):
            if lv is not None and prev_level is not None and lv != prev_level:
                since_change = 0
            prev_level = lv if lv is not None else prev_level
            since_change += 1
            if since_change >= threshold and len(fires) < cap:
                remaining_turns = len(turns) - (i + 1)
                remaining_actions = sum(acts.get(t, 0) for t in turns[i + 1 :])
                fires.append(
                    {
                        "turn_index": i + 1,
                        "analysis_step": s,
                        "level": lv,
                        "remaining_turns": remaining_turns,
                        "remaining_actions": remaining_actions,
                    }
                )
                since_change = 0

        out.append(
            {
                "game_id": gid,
                "total_turns": len(turns),
                "total_actions": sum(acts.values()),
                "fires": fires,
            }
        )
    return {"threshold": threshold, "cap": cap, "games": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--thresholds", default="8,10,12,15,20")
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results = {}
    print(
        f"{'T':>3}{'games fire':>12}{'fires':>7}{'med runway':>12}"
        f"{'min runway':>12}{'turns after':>13}{'actions after':>15}"
    )
    print("-" * 74)
    for t in [int(x) for x in args.thresholds.split(",")]:
        res = simulate(args.run_dir, t, args.cap)
        results[str(t)] = res
        allf = [f for g in res["games"] for f in g["fires"]]
        gf = sum(1 for g in res["games"] if g["fires"])
        ra = sorted(f["remaining_actions"] for f in allf)
        rt = sum(f["remaining_turns"] for f in allf)
        print(
            f"{t:3d}{gf:12d}{len(allf):7d}"
            f"{(ra[len(ra) // 2] if ra else 0):12d}{(ra[0] if ra else 0):12d}"
            f"{rt:13d}{sum(ra):15d}"
        )

    # Detail at the middle threshold for the stalled / zero-scoring games.
    mid = [int(x) for x in args.thresholds.split(",")][len(args.thresholds.split(",")) // 2]
    print(f"\ndetail at T={mid}:")
    print(
        f"{'game':6}{'turns':>7}{'act':>6}{'fires':>7}  "
        f"first fire (turn/of, level, turns left, actions left)"
    )
    for g in sorted(results[str(mid)]["games"], key=lambda g: g["game_id"]):
        if not g["fires"]:
            print(f"{g['game_id']:6}{g['total_turns']:7d}{g['total_actions']:6d}{0:7d}")
            continue
        f0 = g["fires"][0]
        extra = (
            f"  +{len(g['fires']) - 1} more"
            if len(g["fires"]) > 1
            else ""
        )
        print(
            f"{g['game_id']:6}{g['total_turns']:7d}{g['total_actions']:6d}"
            f"{len(g['fires']):7d}  turn {f0['turn_index']}/{g['total_turns']}, "
            f"lvl {f0['level']}, {f0['remaining_turns']} turns / "
            f"{f0['remaining_actions']} actions left{extra}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
