"""Cross-run memory / residency / turns / score accounting for the Duck stack.

Purpose
-------
`stage7_kv_residency_levers.md` closed with the recommendation that the one
remaining direction was **a smaller model** -- "frees memory for KV, buys
residency, but costs capability". The brief that produced this script went
further: it believed the FP8 stack had *also* pinned its KV cache to 5 GiB
(`TAAF_VLLM_KV_CACHE_MEMORY_BYTES = 5368709120`), leaving ~55 GiB of the card
unused, and that a large-KV FP8 run was therefore an untested configuration.

**Both halves of that premise are false, and this script proves it from the
runs' own artifacts rather than from reasoning.**

1. Those `TAAF_VLLM_*` variables are read only by the *NVFP4 fork's* bundled
   `serving_setup.py`. Our FP8 bundle contains zero references to any of them
   (VERIFIED, `experiments/stage7_duck_throughput.md` s2.1), so setting them is
   a silent no-op. Our FP8 stack launches vLLM from a hardcoded argv and ships
   the `mtp1+flags` profile, which sets no KV size at all.
2. Consequently the FP8 run took the vLLM default (`gpu_memory_utilization`
   0.9) and got **44.57 GiB / 342,144 tokens** of KV -- 3.25x the NVFP4 pool --
   and ran at **Waiting p50 = 0, Running p50 = 25, zero preemptions**. The
   large-KV small-model configuration is not untested. It is what our 2.57
   submission ran.

What it measures
----------------
For each kernel output directory, from primary artifacts only:

* ``vllm-openai-server.log``  -> weights GiB, KV pool GiB/tokens, and the full
  ``Running`` / ``Waiting`` / ``KV usage`` distribution over every scheduler
  snapshot (the same 10 s cadence `analyze_turn_latency.py` uses).
* the kernel log's **last** progressive summary block -> mean public-25 score,
  total actions, total generated tokens, wallclock.  Taking the last block is
  load-bearing: the harness emits one roughly every 10 games' worth of
  progress, and reading the first inverts the conclusion
  (`stage7_duck_nvfp4.md`, "A reporting trap worth recording").

It then checks the turn identity from `stage7_turn_latency.md` s5:

    turns_per_game = agg_generation_tok_s * T / (tokens_per_turn * N_games)

which reduces to ``actions == total_generated_tokens / tokens_per_action`` and
is therefore trivially true *given* tokens_per_action -- the point of printing
it is that **tokens_per_action is not a constant of the harness.** It is a
property of the model, it varies 3.5x across these runs, and it is what
cancels every residency gain measured so far.

Usage
-----
    venv/Scripts/python.exe scripts/analyze_model_tradeoff.py \
        fp8=<dir> nvfp4=<dir> ctx16k=<dir> dedupe=<dir> [--json out.json]

Each ``<dir>`` is an unpacked ``kaggle kernels output`` directory.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# "Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 9.4%"
_SCHED = re.compile(
    r"Running:\s*(\d+)\s*reqs?,\s*Waiting:\s*(\d+)\s*reqs?,"
    r"\s*GPU KV cache usage:\s*([\d.]+)%"
)
_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
# Two spellings, because the two stacks take different paths into the
# allocator: ours lets vLLM profile the card and reports what was left over;
# the NVFP4 bundle pins a byte count and skips profiling entirely.
_KV_GIB = re.compile(
    r"Available KV cache memory:\s*([\d.]+)\s*GiB"
    r"|reserved\s*([\d.]+)\s*GiB memory for KV [Cc]ache"
)
# vLLM does not log preemptions as text on either stack; the count lives only
# in the Prometheus scrape. Absent scrape => unknown, which must NOT render as 0.
_PREEMPT = re.compile(r"^vllm:num_preemptions_total\{[^}]*\}\s*([\d.eE+]+)", re.MULTILINE)
_WEIGHTS = re.compile(r"Model loading took\s*([\d.]+)\s*GiB")
_MAXLEN = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")

# Progressive summary block fields. The harness writes these into the kernel
# log with literal "\n" escapes inside JSON stream records, so match loosely.
_SUMMARY_FIELDS = {
    "score": re.compile(r"mean score:\s*([\d.]+)"),
    "actions": re.compile(r"total actions:\s*(\d+)"),
    "gen_tokens": re.compile(r"total tokens:\s*(\d+)"),
    "duration": re.compile(r"duration:\s*(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?"),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def parse_server_log(path: Path) -> dict:
    """Memory facts + the scheduler occupancy distribution."""
    if not path.is_file():
        return {}
    text = _read(path)
    out: dict = {}
    if (m := _WEIGHTS.search(text)):
        out["weights_gib"] = float(m.group(1))
    if (m := _KV_GIB.search(text)):
        out["kv_gib"] = float(m.group(1) or m.group(2))
    if (m := _KV_TOKENS.search(text)):
        out["kv_tokens"] = _int(m.group(1))
    if (m := _MAXLEN.search(text)):
        out["max_model_len"] = _int(m.group(1))
        out["vllm_reported_concurrency"] = float(m.group(2))

    running, waiting, kv_pct = [], [], []
    for match in _SCHED.finditer(text):
        running.append(int(match.group(1)))
        waiting.append(int(match.group(2)))
        kv_pct.append(float(match.group(3)))
    if running:
        out["snapshots"] = len(running)
        out["running_p50"] = statistics.median(running)
        out["running_mean"] = round(statistics.mean(running), 2)
        out["running_max"] = max(running)
        out["waiting_p50"] = statistics.median(waiting)
        out["waiting_mean"] = round(statistics.mean(waiting), 2)
        out["waiting_max"] = max(waiting)
        out["kv_usage_mean_pct"] = round(statistics.mean(kv_pct), 1)
        out["snapshots_queue_empty"] = sum(1 for w in waiting if w == 0)
    return out


def parse_metrics(path: Path) -> dict:
    """Preemption count from the Prometheus scrape, if the stack wrote one."""
    if not path.is_file():
        return {}
    if (m := _PREEMPT.search(_read(path))):
        return {"preemptions": int(float(m.group(1)))}
    return {}


def parse_kernel_log(path: Path) -> dict:
    """The LAST progressive summary block -- never the first (see module docstring)."""
    text = _read(path)
    out: dict = {}
    for key, pattern in _SUMMARY_FIELDS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        last = matches[-1]
        if key == "duration":
            hours, minutes, seconds = (int(g or 0) for g in last.groups())
            out["wallclock_s"] = hours * 3600 + minutes * 60 + seconds
        elif key == "score":
            out["score"] = float(last.group(1))
        else:
            out[key] = int(last.group(1))
    return out


def analyze(name: str, directory: Path) -> dict:
    server_logs = list(directory.glob("vllm-openai-server.log"))
    kernel_logs = [p for p in directory.glob("*.log") if p.name != "vllm-openai-server.log"]
    if not kernel_logs:
        raise SystemExit(f"{name}: no kernel log in {directory}")

    row: dict = {"name": name, "dir": str(directory)}
    row.update(parse_server_log(server_logs[0]) if server_logs else {})
    row.update(parse_metrics(directory / "vllm-metrics-final.prom"))
    row.update(parse_kernel_log(kernel_logs[0]))

    actions = row.get("actions")
    gen = row.get("gen_tokens")
    wall = row.get("wallclock_s")
    if actions and gen:
        row["gen_tokens_per_action"] = round(gen / actions, 1)
    if gen and wall:
        row["agg_gen_tok_s"] = round(gen / wall, 1)
    if actions:
        row["actions_per_game"] = round(actions / 25, 1)

    # Closure check on the turn identity (stage7_turn_latency.md s5).
    if all(k in row for k in ("agg_gen_tok_s", "wallclock_s", "gen_tokens_per_action")):
        predicted = (
            row["agg_gen_tok_s"] * row["wallclock_s"]
            / (row["gen_tokens_per_action"] * 25)
        )
        row["turns_identity_predicted"] = round(predicted, 1)
        if row.get("actions_per_game"):
            row["turns_identity_residual_pct"] = round(
                100.0 * (predicted - row["actions_per_game"]) / row["actions_per_game"], 2
            )

    # Residency the pool actually permits, vs. what the scheduler achieved.
    if row.get("kv_tokens") and row.get("gen_tokens_per_action"):
        # Resident tokens per request = prompt + generation. Prompt is not in
        # this log, so report the pool and the achieved occupancy separately
        # rather than inventing a prompt length.
        row["kv_tokens_per_running_req"] = (
            round(row["kv_tokens"] * (row.get("kv_usage_mean_pct", 0) / 100.0)
                  / row["running_mean"], 0)
            if row.get("running_mean") else None
        )
    return row


COLUMNS = [
    ("name", "run", "s"),
    ("weights_gib", "weights GiB", ".2f"),
    ("kv_gib", "KV GiB", ".2f"),
    ("kv_tokens", "KV tokens", ","),
    ("max_model_len", "max len", ","),
    ("running_p50", "Run p50", "g"),
    ("waiting_p50", "Wait p50", "g"),
    ("preemptions", "preempt", "d"),
    ("agg_gen_tok_s", "gen tok/s", ".1f"),
    ("actions", "actions", ","),
    ("gen_tokens_per_action", "tok/action", ".0f"),
    ("score", "public-25", ".2f"),
]


def render(rows: list[dict]) -> str:
    header = " | ".join(f"{label:>12s}" for _, label, _ in COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _, fmt in COLUMNS:
            value = row.get(key)
            if value is None:
                cells.append(f"{'-':>12s}")
            elif fmt == "s":
                cells.append(f"{value:>12s}")
            else:
                cells.append(f"{value:>12{fmt}}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", metavar="NAME=DIR")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    rows = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected NAME=DIR, got {spec!r}")
        name, _, directory = spec.partition("=")
        rows.append(analyze(name, Path(directory)))

    print(render(rows))
    print()
    for row in rows:
        residual = row.get("turns_identity_residual_pct")
        if residual is not None:
            print(
                f"{row['name']:>10s}: turn identity predicts "
                f"{row['turns_identity_predicted']:.1f} turns/game vs observed "
                f"{row['actions_per_game']:.1f} (residual {residual:+.2f}%)"
            )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
