"""Measure what raising ``LOCAL_ANALYZER_YIELD_SECONDS`` actually did.

Why not just read the score
---------------------------
The public-25 mean is a noisy, weakly-transferring gate: this project's own
local->real ratio is about 0.27, and its real-score sd on a fixed config is
0.295 on n=4. A change that moves productive turns by tens of percent can be
invisible in it. So the primary read-out here is the **mechanism**, measured
from the run's own artifacts, with the score reported alongside for
comparability rather than as the verdict.

What is measured, and from where
--------------------------------
``transcripts/*.txt``       every analyzer turn writes one ``[ANALYZER STATUS]``
                            block ending in ``step_executed: True|False`` and a
                            ``message:``. Blocks with neither are request
                            errors (the per-game read timeout). This gives the
                            no-op fraction and its stop reasons, plus the
                            ``yield_seconds`` the run *actually* used -- the
                            direct confirmation that the knob landed.
``vllm-metrics-final.prom`` the server's own counters: total LLM requests
                            (``request_generation_tokens_count``), generated
                            tokens, and e2e latency. Turns can contain more than
                            one request once the yield budget allows it, so this
                            is what separates "per turn" from "per LLM call".
``benchmark.json``          per-action ``generated_tokens``. Actions costing
                            zero tokens were emitted by the same Python program
                            as the preceding costed action, so grouping on
                            non-zero cost recovers actions-per-acting-call.
``artifacts/*_events.jsonl``every action is stamped with its ``level``, so
                            actions can be split into levels that completed and
                            the one that never did.
``score.json``              the harness's own public-25 self-eval.

Caveat carried from ``analyze_rhae_binding.py``: the events stream counts more
records than ``benchmark.json`` counts actions (it includes animation frames),
so the wasted-action *fraction* is computed within the events stream and the
absolute action total is quoted from ``benchmark.json``. Both are reported.

Usage
-----
    venv/Scripts/python.exe scripts/analyze_yield_effect.py \
        baseline=<dir> yield180=<dir> --json experiments/stage7_yield_fix.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

STATUS_RE = re.compile(r"^\[ANALYZER STATUS\]$", re.MULTILINE)
YIELD_RE = re.compile(r"^yield_seconds: (\S+)$", re.MULTILINE)
EXEC_RE = re.compile(r"^step_executed: (True|False)$", re.MULTILINE)
MESSAGE_RE = re.compile(r"^message: (.+)$", re.MULTILINE)


def read_transcripts(run_dir: Path) -> dict[str, Any]:
    """Turn-level accounting, straight out of the analyzer's own status blocks."""
    per_game: dict[str, dict[str, int]] = {}
    messages: dict[str, int] = {}
    yields: dict[str, int] = {}

    for path in sorted((run_dir / "transcripts").glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        gid = path.stem.removesuffix("_p0")
        blocks = len(STATUS_RE.findall(text))
        execs = EXEC_RE.findall(text)
        per_game[gid] = {
            "status_blocks": blocks,
            "executed": sum(1 for e in execs if e == "True"),
            "no_op": sum(1 for e in execs if e == "False"),
            # Blocks carrying no step_executed line are request errors.
            "errored": blocks - len(execs),
        }
        for msg in MESSAGE_RE.findall(text):
            messages[msg.strip()] = messages.get(msg.strip(), 0) + 1
        for y in YIELD_RE.findall(text):
            yields[y] = yields.get(y, 0) + 1

    executed = sum(g["executed"] for g in per_game.values())
    no_op = sum(g["no_op"] for g in per_game.values())
    return {
        "per_game": per_game,
        "status_blocks": sum(g["status_blocks"] for g in per_game.values()),
        "executed_turns": executed,
        "no_op_turns": no_op,
        "errored_turns": sum(g["errored"] for g in per_game.values()),
        "completed_turns": executed + no_op,
        "no_op_turn_frac": no_op / (executed + no_op) if (executed + no_op) else 0.0,
        "messages": dict(sorted(messages.items(), key=lambda kv: -kv[1])),
        "yield_seconds_seen": dict(sorted(yields.items(), key=lambda kv: -kv[1])),
    }


def read_prom(run_dir: Path) -> dict[str, float]:
    path = run_dir / "vllm-metrics-final.prom"
    if not path.exists():
        return {}
    wanted = {
        "vllm:request_generation_tokens_count": "llm_requests",
        "vllm:generation_tokens_total": "gen_tokens_server",
        "vllm:prompt_tokens_total": "prompt_tokens_server",
        "vllm:e2e_request_latency_seconds_count": "latency_count",
        "vllm:e2e_request_latency_seconds_sum": "latency_sum",
    }
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        name, _, rest = line.partition("{")
        key = wanted.get(name)
        if key is None:
            continue
        out[key] = out.get(key, 0.0) + float(rest.rsplit("}", 1)[1].strip())
    if out.get("latency_count"):
        out["mean_e2e_latency_s"] = out["latency_sum"] / out["latency_count"]
    return out


def read_benchmark(run_dir: Path) -> dict[str, Any]:
    """Actions, and the acting-call grouping recovered from token costs."""
    with (run_dir / "benchmark.json").open(encoding="utf-8") as fh:
        runs = json.load(fh)["game_runs"]

    per_game = {}
    for run in runs:
        history = run.get("history") or []
        costed = sum(1 for s in history if int(s.get("generated_tokens") or 0) > 0)
        per_game[run["game_id"]] = {
            "actions": len(history),
            "gen_tokens": sum(int(s.get("generated_tokens") or 0) for s in history),
            "costed_calls": costed,
            "state": run.get("state"),
            "n_levels": run.get("number_of_levels") or len(run.get("base_actions_per_level") or []),
            "human_actions": sum(run.get("base_actions_per_level") or []),
        }

    actions = sum(g["actions"] for g in per_game.values())
    costed = sum(g["costed_calls"] for g in per_game.values())
    return {
        "per_game": per_game,
        "games": len(per_game),
        "actions": actions,
        "gen_tokens": sum(g["gen_tokens"] for g in per_game.values()),
        "costed_calls": costed,
        "actions_per_acting_call": actions / costed if costed else 0.0,
        "gave_up": sum(1 for g in per_game.values() if g["state"] == "gave_up"),
    }


def read_levels(run_dir: Path) -> dict[str, Any]:
    """Split event-stream actions into completed levels vs the one that never was."""
    per_game = {}
    for path in sorted((run_dir / "artifacts").glob("*_events.jsonl")):
        gid = path.name.removesuffix("_p0_events.jsonl")
        counts: dict[int, int] = {}
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action_num") is None or rec.get("level") is None:
                    continue
                lvl = int(rec["level"])
                counts[lvl] = counts.get(lvl, 0) + 1
        if not counts:
            continue
        deepest = max(counts)
        # Every level before the one we were still on when time ran out is solved.
        per_game[gid] = {
            "levels_solved": deepest - 1,
            "deepest_level": deepest,
            "events_total": sum(counts.values()),
            "events_on_unsolved": counts.get(deepest, 0),
            "per_level": counts,
        }

    total = sum(g["events_total"] for g in per_game.values())
    wasted = sum(g["events_on_unsolved"] for g in per_game.values())
    return {
        "per_game": per_game,
        "events_total": total,
        "events_on_unsolved": wasted,
        "wasted_frac": wasted / total if total else 0.0,
        "levels_solved": sum(g["levels_solved"] for g in per_game.values()),
    }


def read_score(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "score.json"
    if not path.exists():
        return {"mean": None, "per_game": {}}
    with path.open(encoding="utf-8") as fh:
        games = json.load(fh)["games"]
    vals = {k: v["score"] for k, v in games.items()}
    return {
        "mean": sum(vals.values()) / len(vals) if vals else None,
        "zeros": sum(1 for v in vals.values() if v == 0.0),
        # The sota-research write-up's "games scoring 0" column is the sub-1.0
        # bucket, not exact zeros; both are reported so the two agree.
        "sub_one": sum(1 for v in vals.values() if v < 1.0),
        "per_game": vals,
    }


def analyse(name: str, run_dir: Path) -> dict[str, Any]:
    t = read_transcripts(run_dir)
    p = read_prom(run_dir)
    b = read_benchmark(run_dir)
    lv = read_levels(run_dir)
    sc = read_score(run_dir)

    games = b["games"] or 1
    calls = p.get("llm_requests") or t["completed_turns"]
    return {
        "name": name,
        "dir": str(run_dir),
        "games": b["games"],
        # --- the mechanism ------------------------------------------------
        "yield_seconds_seen": t["yield_seconds_seen"],
        "analyzer_turns": t["completed_turns"],
        "executed_turns": t["executed_turns"],
        "no_op_turns": t["no_op_turns"],
        "no_op_turn_frac": t["no_op_turn_frac"],
        "errored_turns": t["errored_turns"],
        "stop_messages": t["messages"],
        "llm_calls": calls,
        "calls_per_turn": calls / t["completed_turns"] if t["completed_turns"] else 0.0,
        "no_op_call_frac": 1 - (t["executed_turns"] / calls) if calls else 0.0,
        # --- the product --------------------------------------------------
        "calls_per_game": calls / games,
        "actions_per_call": b["actions"] / calls if calls else 0.0,
        "actions_per_game": b["actions"] / games,
        "actions": b["actions"],
        "actions_per_acting_call": b["actions_per_acting_call"],
        "gen_tokens": b["gen_tokens"],
        "tokens_per_action": b["gen_tokens"] / b["actions"] if b["actions"] else 0.0,
        "mean_e2e_latency_s": p.get("mean_e2e_latency_s"),
        # --- where the actions went ---------------------------------------
        "levels_solved": lv["levels_solved"],
        "levels_per_game": lv["levels_solved"] / games,
        "wasted_action_frac": lv["wasted_frac"],
        "events_total": lv["events_total"],
        "events_on_unsolved": lv["events_on_unsolved"],
        "gave_up": b["gave_up"],
        # --- comparability -------------------------------------------------
        "public25_mean": sc["mean"],
        "games_scoring_zero": sc.get("zeros"),
        "games_scoring_sub_1": sc.get("sub_one"),
        "per_game": {
            gid: {
                **b["per_game"].get(gid, {}),
                **{f"turn_{k}": v for k, v in t["per_game"].get(gid, {}).items()},
                **{
                    "levels_solved": lv["per_game"].get(gid, {}).get("levels_solved"),
                    "events_total": lv["per_game"].get(gid, {}).get("events_total"),
                    "events_on_unsolved": lv["per_game"].get(gid, {}).get("events_on_unsolved"),
                },
                "score": sc["per_game"].get(gid),
            }
            for gid in sorted(b["per_game"])
        },
    }


def breakeven_model(calls: int, executed: int) -> list[dict[str, float]]:
    """What raising the yield budget can and cannot buy, before seeing the result.

    ``stage7_sota_research.md`` §1.5 puts an upper bound of **+85% productive
    turns** on removing the no-ops. That bound is not reachable by *this* lever,
    and the reason is in the same document's own §1.2: calls per game is pinned
    near 54 by wall-clock / latency, not by the yield budget. A second call
    inside a turn is therefore **not free** -- it consumes a call that would
    otherwise have started a fresh turn.

    Model. Let ``p`` = P(the first call of a turn acts), measured as
    executed/calls on the baseline. Let ``q`` = P(the second call acts, given
    the first did not) -- the unknown this experiment actually measures. With a
    fixed call budget ``C``:

        calls per turn  = p*1 + (1-p)*2 = 2 - p
        turns           = C / (2 - p)
        executed turns  = turns * (p + (1-p)*q)

    So the ceiling is ``q = 1``: **+30%**, not +85%. Break-even is ``q ~ 0.5``:
    below that, raising the budget *reduces* productive turns, because the
    model spends its second call inspecting again rather than acting.

    The one prior datapoint at yield 180 (the anim graft: 1,358 calls, 922
    turns, 586 executed) implies ``q ~ 0.23`` -- well under break-even -- which
    predicts this run lands **negative**. That run changed the solver at the
    same time, which is exactly why this single-variable test exists.
    """
    p = executed / calls if calls else 0.0
    calls_per_turn = 2 - p
    turns = calls / calls_per_turn if calls_per_turn else 0.0
    return [
        {
            "q": q,
            "calls_per_turn": calls_per_turn,
            "turns": turns,
            "executed": turns * (p + (1 - p) * q),
            "vs_baseline": (turns * (p + (1 - p) * q)) / executed - 1 if executed else 0.0,
        }
        for q in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]


def _pct(new: float | None, old: float | None) -> str:
    if not old or new is None:
        return "     -"
    return f"{100 * (new / old - 1):+6.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path pairs; the FIRST is the baseline")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results = []
    for spec in args.runs:
        name, _, raw = spec.partition("=")
        results.append(analyse(name, Path(raw)))

    base = results[0]
    rows = [
        ("no-op turn fraction", "no_op_turn_frac", "{:.1%}"),
        ("no-op LLM-call fraction", "no_op_call_frac", "{:.1%}"),
        ("analyzer turns (total)", "analyzer_turns", "{:.0f}"),
        ("  of which executed", "executed_turns", "{:.0f}"),
        ("LLM calls (server count)", "llm_calls", "{:.0f}"),
        ("LLM calls per turn", "calls_per_turn", "{:.2f}"),
        ("calls per game", "calls_per_game", "{:.1f}"),
        ("actions per call", "actions_per_call", "{:.2f}"),
        ("actions per acting call", "actions_per_acting_call", "{:.2f}"),
        ("actions per game", "actions_per_game", "{:.1f}"),
        ("TOTAL ACTIONS", "actions", "{:.0f}"),
        ("mean e2e latency (s)", "mean_e2e_latency_s", "{:.1f}"),
        ("tokens per action", "tokens_per_action", "{:.0f}"),
        ("levels solved (total)", "levels_solved", "{:.0f}"),
        ("levels per game", "levels_per_game", "{:.2f}"),
        ("wasted-action fraction", "wasted_action_frac", "{:.1%}"),
        ("games ending gave_up", "gave_up", "{:.0f}"),
        ("games scoring exactly 0", "games_scoring_zero", "{:.0f}"),
        ("games scoring < 1.0", "games_scoring_sub_1", "{:.0f}"),
        ("PUBLIC-25 MEAN", "public25_mean", "{:.2f}"),
    ]

    width = max(len(r[0]) for r in rows) + 2
    head = f"{'metric':{width}}" + "".join(f"{r['name']:>14}" for r in results)
    if len(results) > 1:
        head += f"{'vs base':>10}"
    print(head)
    print("-" * len(head))
    for label, key, fmt in rows:
        line = f"{label:{width}}"
        for r in results:
            v = r.get(key)
            line += f"{(fmt.format(v) if v is not None else '-'):>14}"
        if len(results) > 1:
            line += _pct(results[-1].get(key), base.get(key)).rjust(10)
        print(line)

    print()
    print("break-even model on the baseline (see breakeven_model.__doc__):")
    print("  q = P(2nd call acts | 1st did not); calls/game is pinned by latency")
    for row in breakeven_model(int(base["llm_calls"]), int(base["executed_turns"])):
        print(
            f"    q={row['q']:.2f} -> turns {row['turns']:.0f}, "
            f"executed {row['executed']:.0f} ({row['vs_baseline']:+.0%} vs baseline)"
        )
    if len(results) > 1:
        v = results[-1]
        # Recover the realised q from the variant's own turn accounting.
        p = base["executed_turns"] / base["llm_calls"] if base["llm_calls"] else 0.0
        ex_rate = v["executed_turns"] / v["analyzer_turns"] if v["analyzer_turns"] else 0.0
        q = (ex_rate - p) / (1 - p) if p < 1 else 0.0
        print(f"  realised q on {v['name']}: {q:.2f} (break-even ~0.50)")

    print()
    for r in results:
        print(f"{r['name']}: yield_seconds seen in transcripts = {r['yield_seconds_seen']}")
        for msg, n in list(r["stop_messages"].items())[:4]:
            print(f"    {n:>5}x  {msg}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
