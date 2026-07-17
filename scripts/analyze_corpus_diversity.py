"""Stage 6: corpus-quality/diversity check, matching the methodology
`experiments/stage6_selfplay_bootstrap.md` used before committing to a
retrain -- completion rate, frame-changed rate, and (critically, since
"higher activity" isn't automatically "better data") *diversity*: distinct
action ids and distinct ACTION6 click locations actually exercised per
game, not just how often something happened.

Usage: python scripts/analyze_corpus_diversity.py random solver
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"


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


def analyze_file(path: Path) -> dict:
    frames = _load_frame_lines(path)
    if len(frames) < 2:
        return None
    game_id = frames[0].get("game_id", "unknown").split("-")[0]

    n_transitions = 0
    n_changed = 0
    action_ids = set()
    click_xys = set()
    max_levels = 0

    for i in range(len(frames) - 1):
        cur, nxt = frames[i], frames[i + 1]
        action = nxt["action_input"]
        action_id = action["id"]
        if action_id == 0:  # RESET -- not a real probe, skip from action-diversity stats
            continue
        n_transitions += 1
        if cur["frame"] != nxt["frame"]:
            n_changed += 1
        action_ids.add(action_id)
        if action_id == 6:
            xy = action.get("data", {}) or {}
            click_xys.add((xy.get("x", 0), xy.get("y", 0)))
        max_levels = max(max_levels, nxt.get("levels_completed", 0))

    return {
        "game_id": game_id,
        "n_transitions": n_transitions,
        "n_changed": n_changed,
        "changed_rate": n_changed / n_transitions if n_transitions else 0.0,
        "distinct_actions": len(action_ids),
        "distinct_clicks": len(click_xys),
        "n_clicks": sum(1 for i in range(len(frames) - 1) if frames[i + 1]["action_input"]["id"] == 6),
        "max_levels_completed": max_levels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agents", nargs="+", help="Agent name(s) whose recordings to analyze, e.g. random solver")
    args = parser.parse_args()

    for agent in args.agents:
        files = sorted(RECORDINGS_DIR.glob(f"*.{agent}.*.recording.jsonl"))
        if not files:
            print(f"=== {agent}: no recording files found ===\n")
            continue

        per_game = defaultdict(list)
        for f in files:
            r = analyze_file(f)
            if r is not None:
                per_game[r["game_id"]].append(r)

        print(f"=== {agent}: {len(files)} files, {len(per_game)} distinct games ===")
        total_transitions = 0
        total_changed = 0
        total_levels = 0
        games_with_completion = 0
        print(f"{'game':<8} {'files':>5} {'trans':>7} {'chg_rate':>9} {'dist_act':>9} {'dist_click':>11} {'n_click':>8} {'max_lvl':>8}")
        for game, rows in sorted(per_game.items()):
            n_trans = sum(r["n_transitions"] for r in rows)
            n_chg = sum(r["n_changed"] for r in rows)
            dist_act = max((r["distinct_actions"] for r in rows), default=0)
            dist_click = max((r["distinct_clicks"] for r in rows), default=0)
            n_click = sum(r["n_clicks"] for r in rows)
            max_lvl = max((r["max_levels_completed"] for r in rows), default=0)
            total_transitions += n_trans
            total_changed += n_chg
            total_levels += max_lvl
            if max_lvl > 0:
                games_with_completion += 1
            print(
                f"{game:<8} {len(rows):>5} {n_trans:>7} {n_chg / n_trans if n_trans else 0:>9.1%} "
                f"{dist_act:>9} {dist_click:>11} {n_click:>8} {max_lvl:>8}"
            )
        print(
            f"\nTOTAL: {total_transitions} transitions, "
            f"{total_changed / total_transitions if total_transitions else 0:.1%} changed rate, "
            f"{total_levels} total levels completed across {games_with_completion} distinct games\n"
        )


if __name__ == "__main__":
    main()
