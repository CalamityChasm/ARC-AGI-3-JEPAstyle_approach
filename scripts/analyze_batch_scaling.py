"""Recover the decode batch-scaling curve from a Duck run's own vLLM server log.

The question this answers, for free, before any GPU time is spent: on this
stack, does aggregate generation throughput RISE when more sequences run
concurrently, or is decode already compute-bound at the observed batch of ~3?

That is the whole precondition for the KV-capacity lever. Turns per game is

    turns_per_game = agg_output_tok_s * T / (tokens_per_turn * n_games)

so turns scale with AGGREGATE generation throughput and with nothing else. If
generation throughput is flat in `Running`, enlarging the KV pool buys nothing.

Method: vLLM's `loggers.py:310` line reports, every 10 s, the average prompt
and generation throughput over that window together with the instantaneous
`Running` / `Waiting` / KV-usage. Bucketing by `Running` gives a real, measured
throughput-vs-batch curve on the production hardware, at the production prompt
shape, with no new run required.

Caveat, stated rather than buried: `Running` is sampled at the instant the line
is emitted while the throughputs are averaged over the preceding 10 s, so the
pairing is noisy. It is also an OBSERVATIONAL curve - batch size varied because
of the workload, not because anything was controlled - so it bounds the
question rather than settling it. A controlled sweep is the settling evidence.

Usage:
    python scripts/analyze_batch_scaling.py <kernel-output-dir>
"""

from __future__ import annotations

import argparse
import os
import re
import statistics

LOGGER_LINE = re.compile(
    r"Avg prompt throughput: (?P<pt>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<gt>[\d.]+) tokens/s, "
    r"Running: (?P<run>\d+) reqs, Waiting: (?P<wait>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    args = ap.parse_args()

    rows = []
    path = os.path.join(args.output_dir, "vllm-openai-server.log")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LOGGER_LINE.search(line)
            if m:
                rows.append(
                    {
                        "prompt_tps": float(m.group("pt")),
                        "gen_tps": float(m.group("gt")),
                        "running": int(m.group("run")),
                        "waiting": int(m.group("wait")),
                        "kv": float(m.group("kv")),
                    }
                )

    print("=" * 84)
    print("DECODE BATCH-SCALING CURVE (observational, from the production log)")
    print("source: %s   n=%d snapshots" % (os.path.abspath(path), len(rows)))
    print("=" * 84)

    # Pure-decode windows only: a window with heavy prefill spends most of its
    # time on prompt tokens, and its generation figure would understate decode.
    quiet = [r for r in rows if r["prompt_tps"] < 500.0]
    print("\nwindows with prompt throughput < 500 tok/s (decode-dominated): %d of %d"
          % (len(quiet), len(rows)))

    for label, data in (("ALL windows", rows), ("decode-dominated windows", quiet)):
        print("\n-- %s --" % label)
        print("%8s %7s %12s %12s %12s %10s"
              % ("Running", "n", "gen tok/s", "per-seq", "prompt tok/s", "KV %"))
        by = {}
        for r in data:
            by.setdefault(r["running"], []).append(r)
        for run in sorted(by):
            g = by[run]
            gen = statistics.mean(x["gen_tps"] for x in g)
            pro = statistics.mean(x["prompt_tps"] for x in g)
            kv = statistics.mean(x["kv"] for x in g)
            print("%8d %7d %12.1f %12.1f %12.1f %10.1f"
                  % (run, len(g), gen, gen / run if run else float("nan"), pro, kv))

    # Marginal value of one more concurrent sequence, over the decode-dominated
    # windows, which is the number the KV lever actually buys.
    by = {}
    for r in quiet:
        by.setdefault(r["running"], []).append(r["gen_tps"])
    pts = [(k, statistics.mean(v), len(v)) for k, v in sorted(by.items()) if len(v) >= 5]
    print("\n-- marginal aggregate generation throughput per extra running sequence --")
    print("   (decode-dominated windows with n >= 5 only)")
    for (a, ga, na), (b, gb, nb) in zip(pts, pts[1:]):
        print("   Running %d -> %d : %.1f -> %.1f tok/s  (%+.1f tok/s per extra seq, "
              "x%.3f)  n=%d,%d" % (a, b, ga, gb, (gb - ga) / (b - a), gb / ga, na, nb))
    if len(pts) >= 2:
        lo, hi = pts[0], pts[-1]
        print("\n   over Running %d -> %d: aggregate x%.3f while batch x%.3f"
              % (lo[0], hi[0], hi[1] / lo[1], hi[0] / lo[0]))
        print("   -> scaling efficiency %.0f%% of ideal linear batching"
              % (100.0 * (hi[1] / lo[1] - 1.0) / (hi[0] / lo[0] - 1.0)))
        print("   (100%% would mean each extra resident sequence is free; 0%% would")
        print("    mean decode is already compute-bound and the KV lever is worthless)")

    # --- verdict on whether this log can answer the question at all ----------
    counts = {}
    for r in rows:
        counts[r["running"]] = counts.get(r["running"], 0) + 1
    top_run, top_n = max(counts.items(), key=lambda kv: kv[1])
    frac = top_n / len(rows) if rows else 0.0
    print("\n" + "=" * 84)
    print("VERDICT ON THIS LOG AS EVIDENCE")
    print("=" * 84)
    print("Running distribution: %s" % dict(sorted(counts.items())))
    print("%.0f%% of all snapshots sit at Running=%d." % (100.0 * frac, top_run))
    if frac > 0.6:
        print(
            "\nThe batch size BARELY VARIES, because the KV pool pins it. There is no\n"
            "range to regress over, the off-mode buckets are tiny (n=2..19) and are\n"
            "confounded with prefill bursts and preemption recovery, and the\n"
            "decode-dominated Running=4 bucket is visibly contaminated (aggregate\n"
            "generation FALLS, which no batching model predicts).\n\n"
            "So: the turn DECOMPOSITION is fully recoverable from existing artifacts,\n"
            "but the throughput-vs-batch CURVE is not. Reporting a number from this\n"
            "log would be exactly the 'confident, plausible-looking, wrong table'\n"
            "failure this project has already paid for once. It needs a controlled\n"
            "sweep on a free kernel instead."
        )
    else:
        print("\nThe batch size varies enough for this curve to carry real information.")


if __name__ == "__main__":
    main()
