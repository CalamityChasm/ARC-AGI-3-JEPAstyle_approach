"""Matched backtest: GraphExplorerAgent (pure) vs GraphExplorerStructuralAgent
(travel-mode routing biased by scripts/train_graph_win_distance_model.py's
model), same protocol this project has used for every prior GraphExplorer
comparison (MAX_ACTIONS=300, all 25 local games, n repeats, one full-swarm
`python main.py --agent=X` invocation per repeat).

Unlike run_agent_backtest_fold.py (per-game subprocess, Hypothesis-
specific), this runs the whole 25-game swarm per repeat -- matching how
every existing GraphExplorerAgent/GraphExplorerJepaAgent number in
CLAUDE.md was actually produced (pooled score needs one shared scorecard
across all 25 games, not per-game files) -- and extracts+deletes
recordings after every repeat, not just at the end, to bound disk usage
regardless of how many repeats run (see this session's own disk-full
crash earlier at MAX_ACTIONS=5000 for why this matters).

Usage (from repo root, inside the venv):
    python scripts/run_graph_structural_backtest.py --repeats 8 --agents graphexploreragent graphexplorerstructuralagent
"""

import argparse
import json
import subprocess
import sys
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"
VENV_PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"
RECORDINGS_DIR = AGENTS_DIR / "recordings"
HARVEST_DIR = REPO_ROOT / "data" / "graph_explorer_harvest"

MIN_FREE_GB = 5.0


def free_gb(drive: str) -> float:
    total, used, free = shutil.disk_usage(drive)
    return free / (1024 ** 3)


def parse_results_from_recordings() -> dict:
    """Parses real per-game levels_completed/win_levels directly from the
    recording files just produced (same method used to recover results
    after this session's earlier disk-full scorecard-JSON loss) -- more
    robust than relying on the scorecard log line, which has already
    failed once this session under disk pressure."""
    import numpy as np

    results = {}
    for path in sorted(RECORDINGS_DIR.glob("*.recording.jsonl")):
        game = path.name.split("-")[0]
        max_levels = 0
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                d = rec.get("data", {})
                lc = d.get("levels_completed")
                if lc is not None:
                    max_levels = max(max_levels, lc)
        results[game] = max(results.get(game, 0), max_levels)
    return results


def try_parse_pooled_score(log_path: Path) -> float | None:
    """Real schema (verified against an intact scorecard from earlier this
    session, /tmp/graphexplorer_sweep1.log): a top-level "score" field is
    the already-pooled Kaggle-formula score across all games -- no need to
    reconstruct it from "environments"."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    idx = text.find("FINAL SCORECARD REPORT")
    if idx == -1:
        return None
    # find the first '{' after the marker, then the matching '}' via a
    # simple brace counter (the payload is pretty-printed multi-line JSON
    # logged as a single logger.info call, no other '{'/'}' before it on
    # the same logical block).
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    card = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return card.get("score")
    return None


def run_one_repeat(agent: str, repeat_idx: int, timeout: int) -> dict:
    total, used, free = shutil.disk_usage("C:\\")
    if free / (1024 ** 3) < MIN_FREE_GB:
        print(f"[disk] ABORTING before repeat {repeat_idx} -- C: {free/(1024**3):.2f}GB free", file=sys.stderr)
        sys.exit(1)

    for f in RECORDINGS_DIR.glob("*.recording.jsonl"):
        f.unlink()

    log_path = REPO_ROOT / f"_backtest_{agent}_repeat{repeat_idx}.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        try:
            subprocess.run(
                [str(VENV_PYTHON), "main.py", f"--agent={agent}"],
                cwd=AGENTS_DIR, timeout=timeout, stdout=logf, stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"  [warn] repeat {repeat_idx} for {agent} timed out after {timeout}s", flush=True)
    elapsed = time.time() - t0

    per_game_levels = parse_results_from_recordings()
    pooled_score = try_parse_pooled_score(log_path)

    # Extract level-up transitions before deleting, per this project's
    # established practice (wins are sparse, don't discard them).
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(VENV_PYTHON), "scripts/extract_level_up_transitions.py",
         "--src", str(RECORDINGS_DIR), "--dst", str(HARVEST_DIR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    for f in RECORDINGS_DIR.glob("*.recording.jsonl"):
        f.unlink()
    log_path.unlink(missing_ok=True)

    total_levels = sum(per_game_levels.values())
    distinct_games = sorted(g for g, lv in per_game_levels.items() if lv > 0)
    print(f"[{agent}] repeat {repeat_idx}: elapsed={elapsed:.0f}s total_levels={total_levels} "
          f"distinct_games={len(distinct_games)} {distinct_games} pooled_score={pooled_score}", flush=True)
    return {
        "repeat": repeat_idx, "elapsed": elapsed, "total_levels": total_levels,
        "distinct_games": distinct_games, "pooled_score": pooled_score,
        "per_game_levels": per_game_levels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "graph_structural_backtest_results.json")
    args = parser.parse_args()

    all_results = {}
    for agent in args.agents:
        repeats = []
        for r in range(1, args.repeats + 1):
            repeats.append(run_one_repeat(agent, r, args.timeout))
        all_repeat_levels = [rep["total_levels"] for rep in repeats]
        all_distinct = set()
        for rep in repeats:
            all_distinct.update(rep["distinct_games"])
        scores = [rep["pooled_score"] for rep in repeats if rep["pooled_score"] is not None]
        summary = {
            "agent": agent, "n_repeats": args.repeats,
            "total_levels_sum": sum(all_repeat_levels),
            "mean_levels_per_repeat": sum(all_repeat_levels) / len(all_repeat_levels),
            "distinct_games_ever": sorted(all_distinct),
            "n_distinct_games_ever": len(all_distinct),
            "mean_pooled_score": (sum(scores) / len(scores)) if scores else None,
            "n_repeats_with_pooled_score": len(scores),
            "repeats": repeats,
        }
        all_results[agent] = summary
        print(f"\n=== {agent} summary ===")
        print(json.dumps({k: v for k, v in summary.items() if k != "repeats"}, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote full results to {args.out}")


if __name__ == "__main__":
    main()
