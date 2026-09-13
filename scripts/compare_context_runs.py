"""Compare Duck NVFP4 public-25 runs on BOTH axes: score and throughput.

Score alone would have shipped prefix caching (identical actions, 47% score
collapse). Throughput alone would ship anything that makes the model dumber but
faster. Every candidate is therefore reported on:

  * mean self-eval score  -- did quality survive?
  * total actions         -- did residency actually improve?
  * mean prompt tokens, KV residency, turns/game, retained history depth

The progressive-summary trap: these logs emit a running summary roughly every
10 minutes, so a naive grep for "mean score" returns an early, partial figure.
This script recomputes the mean from the per-game `[finished]` lines and
cross-checks it against both `benchmark.json`'s `final_score` fields and the
LAST progressive block, printing all three. They must agree.

Usage:
    python scripts/compare_context_runs.py <label>=<dir> [<label>=<dir> ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

KV_POOL_TOKENS = 105_202

FINISHED = re.compile(
    r"\[finished\] (?P<game>\S+) state=(?P<state>\S+) level=(?P<lvl>\d+)/(?P<lvls>\d+) "
    r"score=(?P<score>[-\d.]+) actions=(?P<actions>\d+) tokens=(?P<tokens>\d+)"
)
MEANLINE = re.compile(r"mean score:\s+([\d.]+)")


def read_log(path):
    """Kernel logs are a JSON array of {stream_name, time, data} records."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    try:
        return "".join(rec.get("data", "") for rec in json.loads(raw))
    except json.JSONDecodeError:
        # Older/plain logs
        return raw


def parse_prom(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            name, _, val = line.strip().rpartition(" ")
            name = name.split("{")[0]
            try:
                out[name] = out.get(name, 0.0) + float(val)
            except ValueError:
                pass
    return out


def scheduler_stats(path):
    """Running / Waiting / preemptions from the vLLM server log."""
    if not os.path.exists(path):
        return {}
    run, wait, preempt = [], [], 0
    pat = re.compile(r"Running:\s*(\d+)\s*reqs.*?Waiting:\s*(\d+)\s*reqs")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                run.append(int(m.group(1)))
                wait.append(int(m.group(2)))
            if "preempt" in line.lower():
                preempt += 1
    if not run:
        return {}
    return {
        "snapshots": len(run),
        "running_p50": statistics.median(run),
        "running_max": max(run),
        "waiting_p50": statistics.median(wait),
        "preempt_lines": preempt,
    }


def summarise(label, root):
    logs = glob.glob(os.path.join(root, "*.log"))
    kernel_log = next((p for p in logs if os.path.basename(p).startswith("arc3-")), None)
    text = read_log(kernel_log) if kernel_log else ""

    finished = [m.groupdict() for m in FINISHED.finditer(text)]
    progressive = MEANLINE.findall(text)

    scores = [float(f["score"]) for f in finished]
    actions = [int(f["actions"]) for f in finished]
    levels = [int(f["lvl"]) for f in finished]

    bench_scores, bench_actions = [], []
    bpath = os.path.join(root, "benchmark.json")
    if os.path.exists(bpath):
        b = json.load(open(bpath, encoding="utf-8"))
        for g in b.get("game_runs", []):
            bench_scores.append(float(g.get("final_score", 0.0)))
            bench_actions.append(sum(g.get("actions_per_level", []) or []))

    prom = parse_prom(os.path.join(root, "vllm-metrics-final.prom"))
    n_req = prom.get("vllm:request_prompt_tokens_count", 0.0)
    prompt_sum = prom.get("vllm:request_prompt_tokens_sum", 0.0)
    gen_sum = prom.get("vllm:request_generation_tokens_sum", 0.0)
    mean_prompt = prompt_sum / n_req if n_req else float("nan")
    mean_gen = gen_sum / n_req if n_req else float("nan")
    per_req = mean_prompt + mean_gen
    residency = KV_POOL_TOKENS / per_req if per_req == per_req and per_req else float("nan")

    # Retained history depth: how many user turns the final snapshot carried.
    plogs = glob.glob(os.path.join(root, "prompts", "*.log"))
    retained = []
    for p in plogs:
        with open(p, encoding="utf-8", errors="replace") as fh:
            body = fh.read()
        try:
            body = body[body.index("[MODEL INPUT]"): body.index("\n[TURN TRANSCRIPT SO FAR]")]
        except ValueError:
            continue
        retained.append(body.count("\n[USER]"))

    return {
        "label": label,
        "root": root,
        "n_games_finished": len(finished),
        "mean_score_from_finished": statistics.mean(scores) if scores else float("nan"),
        "mean_score_from_benchmark": statistics.mean(bench_scores) if bench_scores else float("nan"),
        "last_progressive_mean": float(progressive[-1]) if progressive else float("nan"),
        "n_progressive_blocks": len(progressive),
        "total_actions": sum(actions),
        "total_actions_benchmark": sum(bench_actions),
        "total_levels": sum(levels),
        "zero_score_games": sum(1 for s in scores if s <= 0),
        "requests": n_req,
        "mean_prompt_tokens": mean_prompt,
        "mean_generation_tokens": mean_gen,
        "tokens_per_request": per_req,
        "kv_residency": residency,
        "turns_per_game": (n_req / len(finished)) if finished else float("nan"),
        "retained_user_turns": statistics.mean(retained) if retained else float("nan"),
        **scheduler_stats(os.path.join(root, "vllm-openai-server.log")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=dir")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        label, _, root = spec.partition("=")
        if not root:
            sys.exit(f"expected label=dir, got {spec!r}")
        rows.append(summarise(label, root))

    base = rows[0]

    def fmt(v, spec=",.2f"):
        return "n/a" if v != v else format(v, spec)

    metrics = [
        ("games finished", "n_games_finished", ",d", False),
        ("MEAN SCORE (from [finished])", "mean_score_from_finished", ",.2f", True),
        ("  cross-check: benchmark.json", "mean_score_from_benchmark", ",.2f", False),
        ("  cross-check: last progressive", "last_progressive_mean", ",.2f", False),
        ("TOTAL ACTIONS", "total_actions", ",d", True),
        ("  cross-check: benchmark.json", "total_actions_benchmark", ",d", False),
        ("total levels completed", "total_levels", ",d", True),
        ("games scoring zero", "zero_score_games", ",d", False),
        ("LLM requests", "requests", ",.0f", True),
        ("turns per game", "turns_per_game", ",.1f", True),
        ("mean prompt tokens", "mean_prompt_tokens", ",.0f", True),
        ("mean generation tokens", "mean_generation_tokens", ",.0f", False),
        ("tokens per request", "tokens_per_request", ",.0f", True),
        ("KV residency (105,202/tok)", "kv_residency", ",.2f", True),
        ("retained user turns/request", "retained_user_turns", ",.1f", True),
        ("scheduler Running p50", "running_p50", ",.1f", False),
        ("scheduler Waiting p50", "waiting_p50", ",.1f", False),
    ]

    w = 32
    header = f"{'metric':<{w}}" + "".join(f"{r['label']:>16}" for r in rows)
    if len(rows) > 1:
        header += "".join(f"{'vs ' + base['label']:>16}" for _ in rows[1:])
    print(header)
    print("-" * len(header))
    for name, key, spec, ratio in metrics:
        line = f"{name:<{w}}"
        for r in rows:
            v = r.get(key, float("nan"))
            line += f"{('n/a' if v != v else format(v, spec)):>16}"
        if len(rows) > 1:
            for r in rows[1:]:
                a, b = base.get(key, float("nan")), r.get(key, float("nan"))
                if ratio and a == a and b == b and a:
                    line += f"{b / a:>15.2f}x"
                else:
                    line += f"{'':>16}"
        print(line)

    print()
    print("read this table as a pair: ACTIONS says whether residency actually")
    print("improved, SCORE says whether the model still plays well. The prefix-")
    print("caching failure moved actions by +0.2% and score by -47%; a")
    print("throughput-only benchmark would have shipped it.")

    if args.json_out:
        json.dump(rows, open(args.json_out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()


