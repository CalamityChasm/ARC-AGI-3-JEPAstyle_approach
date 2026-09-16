"""Decompose `tokens_per_action` into its two independent factors.

Background
----------
`experiments/stage7_model_search.md` §2 establishes the turn identity

    turns_per_game = agg_gen_tok_s * T / (tokens_per_action * N_games)

and shows that `tokens_per_action` spans 515 -> 1,782 (a factor of 3.5) across
four real runs, cancelling every throughput gain the Stage 7 serving work
produced. It does not explain *why*.

This script answers that from the runs' own `benchmark.json`, which records, for
every single game action, the `generated_tokens` that action cost. Crucially many
actions record **zero** generated tokens: the harness lets one LLM turn emit a
Python program that issues several game actions, and only the first action of
such a batch carries the turn's token cost. So

    tokens_per_action = tokens_per_turn / actions_per_turn

and the two factors are independently controllable: `tokens_per_turn` is
reasoning verbosity, `actions_per_turn` is how much the model batches.

Usage
-----
    venv/Scripts/python.exe scripts/analyze_tokens_per_action.py \
        fp8=<dir> nvfp4=<dir> ctx16k=<dir> dedupe=<dir> anim=<dir> \
        --json experiments/stage7_tokens_per_action.json

`<dir>` is an unpacked `kaggle kernels output` directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_runs(path: Path) -> list[dict[str, Any]]:
    with (path / "benchmark.json").open(encoding="utf-8") as fh:
        return json.load(fh)["game_runs"]


def rhae_level_score(human_actions: int, agent_actions: int) -> float:
    """S_l = min(1.15, h_l / a_l) ** 2 -- the official per-level RHAE term."""
    if agent_actions <= 0:
        return 0.0
    return min(1.15, human_actions / agent_actions) ** 2


def analyse_game(run: dict[str, Any]) -> dict[str, Any]:
    history = run.get("history") or []
    human = run.get("base_actions_per_level") or []
    n_levels = run.get("number_of_levels") or len(human)

    # --- factor the token cost -------------------------------------------
    # A "turn" starts at every action that carries a non-zero token cost.
    # Actions with zero cost were emitted by the same Python program as the
    # preceding costed action, i.e. they are free extra actions inside a turn.
    turn_tokens: list[int] = []
    actions_in_turn: list[int] = []
    for step in history:
        tok = int(step.get("generated_tokens") or 0)
        if tok > 0 or not turn_tokens:
            turn_tokens.append(tok)
            actions_in_turn.append(1)
        else:
            actions_in_turn[-1] += 1

    actions = len(history)
    gen_tokens = sum(int(s.get("generated_tokens") or 0) for s in history)
    turns = len(turn_tokens)
    free_actions = sum(1 for s in history if int(s.get("generated_tokens") or 0) == 0)

    # --- where did the actions go? ---------------------------------------
    # The harness records level transitions implicitly; use the score payload
    # where available, else fall back to counting from the run state.
    levels_done = run.get("levels_completed")
    if levels_done is None:
        # `state` is one of win / gave_up / timeout; the per-level action
        # boundaries are not in benchmark.json, so fall back to the score file.
        levels_done = run.get("level", 0)

    return {
        "game_id": run.get("game_id"),
        "state": run.get("state"),
        "n_levels": n_levels,
        "human_actions_total": sum(human),
        "human_actions_per_level": human,
        "actions": actions,
        "gen_tokens": gen_tokens,
        "turns": turns,
        "free_actions": free_actions,
        "free_action_frac": (free_actions / actions) if actions else 0.0,
        "tokens_per_action": (gen_tokens / actions) if actions else 0.0,
        "tokens_per_turn": (gen_tokens / turns) if turns else 0.0,
        "actions_per_turn": (actions / turns) if turns else 0.0,
        "turn_tokens": turn_tokens,
        "actions_in_turn": actions_in_turn,
    }


def analyse_run(name: str, path: Path) -> dict[str, Any]:
    games = [analyse_game(r) for r in load_runs(path)]

    actions = sum(g["actions"] for g in games)
    gen_tokens = sum(g["gen_tokens"] for g in games)
    turns = sum(g["turns"] for g in games)
    free_actions = sum(g["free_actions"] for g in games)
    all_turn_tokens = [t for g in games for t in g["turn_tokens"] if t > 0]
    all_batch = [a for g in games for a in g["actions_in_turn"]]

    # score.json, if present, is the harness's own public-25 self-eval.
    score = None
    score_path = path / "score.json"
    if score_path.exists():
        with score_path.open(encoding="utf-8") as fh:
            per_game = json.load(fh)["games"]
        vals = [v["score"] for v in per_game.values()]
        score = sum(vals) / len(vals) if vals else None

    return {
        "name": name,
        "games": len(games),
        "score": score,
        "actions": actions,
        "gen_tokens": gen_tokens,
        "turns": turns,
        "tokens_per_action": gen_tokens / actions if actions else 0.0,
        "tokens_per_turn": gen_tokens / turns if turns else 0.0,
        "actions_per_turn": actions / turns if turns else 0.0,
        "free_actions": free_actions,
        "free_action_frac": free_actions / actions if actions else 0.0,
        "turn_tokens_median": statistics.median(all_turn_tokens) if all_turn_tokens else 0,
        "turn_tokens_p90": (
            statistics.quantiles(all_turn_tokens, n=10)[8] if len(all_turn_tokens) > 10 else 0
        ),
        "turn_tokens_max": max(all_turn_tokens) if all_turn_tokens else 0,
        "batch_median": statistics.median(all_batch) if all_batch else 0,
        "batch_max": max(all_batch) if all_batch else 0,
        "per_game": games,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path pairs")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results = []
    for spec in args.runs:
        name, _, raw = spec.partition("=")
        results.append(analyse_run(name, Path(raw)))

    hdr = (
        f"{'run':10} {'score':>7} {'actions':>8} {'turns':>7} "
        f"{'tok/act':>8} {'tok/turn':>9} {'act/turn':>9} {'free%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        score = f"{r['score']:.2f}" if r["score"] is not None else "-"
        print(
            f"{r['name']:10} {score:>7} {r['actions']:>8} {r['turns']:>7} "
            f"{r['tokens_per_action']:>8.0f} {r['tokens_per_turn']:>9.0f} "
            f"{r['actions_per_turn']:>9.2f} {r['free_action_frac'] * 100:>6.1f}%"
        )

    print()
    print("turn-token distribution (costed turns only):")
    for r in results:
        print(
            f"  {r['name']:10} median={r['turn_tokens_median']:>6.0f} "
            f"p90={r['turn_tokens_p90']:>6.0f} max={r['turn_tokens_max']:>6.0f} "
            f"| batch median={r['batch_median']:.0f} max={r['batch_max']:.0f}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
