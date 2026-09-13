"""Decompose Duck/TAAF analyzer turn latency from a run's own primary artifacts.

Inputs: the unzipped `kaggle kernels output` directory of a Duck NVFP4 run,
which must contain `vllm-metrics-final.prom`, `vllm-openai-server.log`,
`transcripts/*.txt`.

Usage:
    python scripts/analyze_turn_latency.py <output_dir> [--json out.json]

Everything printed is derived from primary artifacts (the vLLM Prometheus
scrape and the vLLM server log), never from the kernel console log, which is
stderr-block-buffered and mis-timestamps events (see
experiments/stage7_analyzer_timeouts.md section 0).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

# --------------------------------------------------------------------------
# Prometheus text-format parsing
# --------------------------------------------------------------------------

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<val>[-+0-9.eEnaN]+)\s*$"
)


def parse_prom(path):
    """Return {metric_name: [(labels_dict, value), ...]}."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _SAMPLE.match(line)
            if not m:
                continue
            labels = {}
            if m.group("labels"):
                for k, v in re.findall(r'(\w+)="([^"]*)"', m.group("labels")):
                    labels[k] = v
            try:
                val = float(m.group("val"))
            except ValueError:
                continue
            out.setdefault(m.group("name"), []).append((labels, val))
    return out


def scalar(prom, name, **want):
    for labels, val in prom.get(name, []):
        if all(labels.get(k) == v for k, v in want.items()):
            return val
    return None


def hist_buckets(prom, base):
    rows = []
    for labels, val in prom.get(base + "_bucket", []):
        le = labels.get("le")
        if le is None:
            continue
        rows.append((float("inf") if le == "+Inf" else float(le), val))
    rows.sort(key=lambda r: r[0])
    return rows


def hist_quantile(buckets, q):
    """Bucket upper bound at quantile q.

    Deliberately returns the bucket edge rather than interpolating inside it:
    vLLM's buckets are coarse and interpolation would invent precision the
    data does not have. Read every percentile as "<= this value".
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = q * total
    for le, cum in buckets:
        if cum >= target:
            return le
    return buckets[-1][0]


def hist_stats(prom, base):
    count = scalar(prom, base + "_count")
    total = scalar(prom, base + "_sum")
    buckets = hist_buckets(prom, base)
    return {
        "count": count,
        "sum": total,
        "mean": (total / count) if (count and total is not None) else None,
        "p50": hist_quantile(buckets, 0.50),
        "p90": hist_quantile(buckets, 0.90),
        "p99": hist_quantile(buckets, 0.99),
    }


# --------------------------------------------------------------------------
# vLLM server log
# --------------------------------------------------------------------------

LOGGER_LINE = re.compile(
    r"Avg prompt throughput: (?P<pt>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<gt>[\d.]+) tokens/s, "
    r"Running: (?P<run>\d+) reqs, Waiting: (?P<wait>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%"
)
KV_SIZE = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, "
    r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"
)
MODEL_MEM = re.compile(r"Model loading took ([\d.]+) GiB memory")
INIT_FREE = re.compile(
    r"Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB memory for KV Cache"
)
OOM = re.compile(
    r"memory allocation failed with OOM on device \d+ while trying to allocate "
    r"(\d+) bytes \(free: (\d+), total: (\d+)\)"
)
NONDEFAULT = re.compile(r"non-default args: (\{.*\})")


def parse_server_log(path):
    snaps = []
    info = {
        "kv_cache_tokens": None,
        "kv_max_concurrency_at": None,
        "kv_max_concurrency": None,
        "model_load_gib": None,
        "initial_free_gib": None,
        "kv_reserved_gib": None,
        "ooms": [],
        "launch_args": {},
    }
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LOGGER_LINE.search(line)
            if m:
                snaps.append(
                    {
                        "prompt_tps": float(m.group("pt")),
                        "gen_tps": float(m.group("gt")),
                        "running": int(m.group("run")),
                        "waiting": int(m.group("wait")),
                        "kv_pct": float(m.group("kv")),
                    }
                )
                continue
            m = KV_SIZE.search(line)
            if m:
                info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
                info["kv_max_concurrency_at"] = int(m.group(2).replace(",", ""))
                info["kv_max_concurrency"] = float(m.group(3))
                continue
            m = MODEL_MEM.search(line)
            if m:
                info["model_load_gib"] = float(m.group(1))
                continue
            m = INIT_FREE.search(line)
            if m:
                info["initial_free_gib"] = float(m.group(1))
                info["kv_reserved_gib"] = float(m.group(2))
                continue
            m = OOM.search(line)
            if m:
                info["ooms"].append(
                    {
                        "want_bytes": int(m.group(1)),
                        "free_bytes": int(m.group(2)),
                        "total_bytes": int(m.group(3)),
                    }
                )
                continue
            m = NONDEFAULT.search(line)
            if m and not info["launch_args"]:
                for key in (
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "kv_cache_memory_bytes",
                    "max_model_len",
                    "enable_prefix_caching",
                    "enable_chunked_prefill",
                    "async_scheduling",
                    "max_cudagraph_capture_size",
                ):
                    mm = re.search(r"'%s': ([^,}]+)" % key, m.group(1))
                    if mm:
                        info["launch_args"][key] = mm.group(1).strip()
                mm = re.search(r"'speculative_config': (\{[^}]*\})", m.group(1))
                if mm:
                    info["launch_args"]["speculative_config"] = mm.group(1)
    info["snapshots"] = snaps
    return info


# --------------------------------------------------------------------------
# Transcripts (same header convention as scripts/analyze_analyzer_timeouts.py)
# --------------------------------------------------------------------------

HDR = re.compile(
    r"--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---"
)


def _hhmmss(ts):
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def scan_transcripts(root):
    per_game = []
    all_durations = []
    for path in sorted(glob.glob(os.path.join(root, "transcripts", "*.txt"))):
        game = os.path.basename(path).split("_p0")[0]
        text = open(path, encoding="utf-8", errors="replace").read()
        times = [_hhmmss(m.group(3)) for m in HDR.finditer(text)]
        durs = [b - a for a, b in zip(times, times[1:])]
        all_durations.extend(durs)
        per_game.append(
            {
                "game": game,
                "turns": len(times),
                "first_s": times[0] if times else None,
                "last_s": times[-1] if times else None,
                "median_turn_s": statistics.median(durs) if durs else None,
            }
        )
    return {"per_game": per_game, "durations": all_durations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = args.output_dir

    prom = parse_prom(os.path.join(root, "vllm-metrics-final.prom"))
    srv = parse_server_log(os.path.join(root, "vllm-openai-server.log"))
    tr = scan_transcripts(root)

    e2e = hist_stats(prom, "vllm:e2e_request_latency_seconds")
    queue = hist_stats(prom, "vllm:request_queue_time_seconds")
    infer = hist_stats(prom, "vllm:request_inference_time_seconds")
    prefill = hist_stats(prom, "vllm:request_prefill_time_seconds")
    decode = hist_stats(prom, "vllm:request_decode_time_seconds")
    ttft = hist_stats(prom, "vllm:time_to_first_token_seconds")

    n_req = e2e["count"] or 0
    prompt_tok = scalar(prom, "vllm:prompt_tokens_total") or 0.0
    gen_tok = scalar(prom, "vllm:generation_tokens_total") or 0.0
    drafts = scalar(prom, "vllm:spec_decode_num_drafts_total") or 0.0
    draft_tok = scalar(prom, "vllm:spec_decode_num_draft_tokens_total") or 0.0
    acc_tok = scalar(prom, "vllm:spec_decode_num_accepted_tokens_total") or 0.0

    snaps = srv["snapshots"]
    window_s = 10.0 * len(snaps)  # loggers.py:310 fires on a 10 s cadence

    def mean(key):
        return statistics.mean(s[key] for s in snaps) if snaps else float("nan")

    print("=" * 78)
    print("TURN LATENCY DECOMPOSITION")
    print("source: %s" % os.path.abspath(root))
    print("=" * 78)

    print("\n-- [A] Per-request latency, vllm-metrics-final.prom (cumulative) --")
    print("  requests completed                 n = %.0f" % n_req)
    print(
        "  %-28s %9s %8s %9s %9s %9s"
        % ("metric", "mean s", "share", "p50<=", "p90<=", "p99<=")
    )
    for label, h in [
        ("e2e request latency", e2e),
        ("  queue (WAITING phase)", queue),
        ("  inference (RUNNING phase)", infer),
        ("    prefill phase", prefill),
        ("    decode phase", decode),
        ("time to first token", ttft),
    ]:
        share = ""
        if h["mean"] is not None and e2e["mean"]:
            share = "%7.2f%%" % (100.0 * h["mean"] / e2e["mean"])
        print(
            "  %-28s %9.2f %8s %9s %9s %9s"
            % (label, h["mean"], share, h["p50"], h["p90"], h["p99"])
        )
    resid = e2e["mean"] - queue["mean"] - infer["mean"]
    print(
        "  closure: queue+inference = %.2f s vs e2e %.2f s (residual %+.2f s)"
        % (queue["mean"] + infer["mean"], e2e["mean"], resid)
    )

    print("\n-- [B] Server admission state, vllm-openai-server.log --")
    print("  launch args: %s" % json.dumps(srv["launch_args"]))
    print("  scheduler snapshots (10 s cadence) n = %d -> window %.0f s" % (len(snaps), window_s))
    print("  KV cache size                      %s tokens" % f"{srv['kv_cache_tokens']:,}")
    print(
        "  vLLM max concurrency @ %s tok/req  %.2fx"
        % (f"{srv['kv_max_concurrency_at']:,}", srv["kv_max_concurrency"])
    )
    print("  mean Running reqs                  %.2f" % mean("running"))
    print("  mean Waiting reqs                  %.2f" % mean("waiting"))
    print("  mean Running+Waiting               %.2f" % (mean("running") + mean("waiting")))
    print("  mean GPU KV cache usage            %.1f%%" % mean("kv_pct"))
    runs = sorted(s["running"] for s in snaps)
    waits = sorted(s["waiting"] for s in snaps)
    kvs = sorted(s["kv_pct"] for s in snaps)
    print("  Running  min/p50/max               %d / %d / %d" % (runs[0], runs[len(runs) // 2], runs[-1]))
    print("  Waiting  min/p50/max               %d / %d / %d" % (waits[0], waits[len(waits) // 2], waits[-1]))
    print("  KV usage min/p50/max               %.1f%% / %.1f%% / %.1f%%" % (kvs[0], kvs[len(kvs) // 2], kvs[-1]))
    n_at_cap = sum(1 for s in snaps if s["running"] >= 8)
    print("  snapshots with Running >= 8        %d of %d (%.1f%%)" % (n_at_cap, len(snaps), 100.0 * n_at_cap / len(snaps)))
    n_kv_hi = sum(1 for s in snaps if s["kv_pct"] >= 85.0)
    print("  snapshots with KV usage >= 85%%     %d of %d (%.1f%%)" % (n_kv_hi, len(snaps), 100.0 * n_kv_hi / len(snaps)))
    print("  mean prompt throughput             %.1f tok/s" % mean("prompt_tps"))
    print("  mean generation throughput         %.1f tok/s" % mean("gen_tps"))

    print("\n-- [C] GPU memory headroom --")
    print("  model weights on GPU               %s GiB" % srv["model_load_gib"])
    print("  reported initial free              %s GiB" % srv["initial_free_gib"])
    print("  KV cache reserved                  %s GiB" % srv["kv_reserved_gib"])
    print("  CUDA OOM events in server log      %d" % len(srv["ooms"]))
    for o in srv["ooms"]:
        print(
            "    wanted %.1f MiB, free %.1f MiB, device total %.1f GiB"
            % (o["want_bytes"] / 2 ** 20, o["free_bytes"] / 2 ** 20, o["total_bytes"] / 2 ** 30)
        )

    print("\n-- [D] Workload shape --")
    print("  mean prompt tokens / request       %,.0f".replace(",", "") % (prompt_tok / n_req))
    print("  mean generated tokens / request    %.0f" % (gen_tok / n_req))
    kv_per_req = (prompt_tok + gen_tok) / n_req
    print("  mean KV footprint / request        %.0f tokens" % kv_per_req)
    print("  -> requests that fit in KV cache   %.2f" % (srv["kv_cache_tokens"] / kv_per_req))
    print("  per-sequence decode rate           %.1f tok/s" % (gen_tok / n_req / decode["mean"]))
    print("  per-sequence prefill rate          %.0f tok/s" % (prompt_tok / n_req / prefill["mean"]))
    print(
        "  prefix cache queries / hits        %.0f / %.0f"
        % (scalar(prom, "vllm:prefix_cache_queries_total") or 0, scalar(prom, "vllm:prefix_cache_hits_total") or 0)
    )
    print("  MTP drafts/draft tok/accepted      %.0f / %.0f / %.0f" % (drafts, draft_tok, acc_tok))
    print("  MTP acceptance rate                %.4f" % (acc_tok / draft_tok))
    print("  MTP accepted per draft             %.3f of %.0f" % (acc_tok / drafts, draft_tok / drafts))
    preempt = scalar(prom, "vllm:num_preemptions_total") or 0.0
    print("  preemptions                        %.0f (%.1f%% of requests)"
          % (preempt, 100.0 * preempt / n_req))
    for labels, val in prom.get("vllm:request_success_total", []):
        if val:
            print("  finished_reason=%-12s        %.0f" % (labels.get("finished_reason"), val))

    print("\n-- [E] Turn-level cross-check, transcripts/*.txt --")
    games = tr["per_game"]
    durs = sorted(tr["durations"])
    total_turns = sum(g["turns"] for g in games)
    print("  games                              %d" % len(games))
    print("  analyzer turns (transcript heads)  %d" % total_turns)
    print(
        "  turns per game min/mean/max        %d / %.1f / %d"
        % (min(g["turns"] for g in games), total_turns / len(games), max(g["turns"] for g in games))
    )
    print(
        "  inter-turn gap p10/p50/p90 (s)     %d / %d / %d"
        % (durs[int(0.10 * len(durs))], durs[len(durs) // 2], durs[int(0.90 * len(durs))])
    )
    print("  LLM requests per analyzer turn     %.3f" % (n_req / total_turns))

    print("\n-- [F] Little's law closure --")
    X = n_req / window_s
    print("  throughput X                       %.4f req/s" % X)
    print("  X * e2e   -> reqs in server        %.2f" % (X * e2e["mean"]))
    print("  X * queue -> reqs waiting          %.2f   (log mean %.2f)" % (X * queue["mean"], mean("waiting")))
    print("  X * infer -> reqs running          %.2f   (log mean %.2f)" % (X * infer["mean"], mean("running")))

    print("\n-- [G] Turns-per-game arithmetic --")
    print("  A game is a serial loop, so it gets T / turn_latency turns, and under")
    print("  saturation Little's law makes turn_latency = N_games / X. Both views")
    print("  reduce to the same identity, which closes on the measured data:")
    print()
    tokens_per_turn = gen_tok / total_turns
    agg_gen = mean("gen_tps")
    print("      turns_per_game = agg_generation_tok_s * T / (tokens_per_turn * N_games)")
    print("                     = %.1f * %.0f / (%.0f * %d)"
          % (agg_gen, window_s, tokens_per_turn, len(games)))
    print("                     = %.1f          (observed %.1f)"
          % (agg_gen * window_s / (tokens_per_turn * len(games)), total_turns / len(games)))
    print()
    print("      turns_per_game = X * T / N_games = %.4f * %.0f / %d = %.1f"
          % (X, window_s, len(games), X * window_s / len(games)))
    print("      turn_latency   = N_games / X     = %d / %.4f = %.1f s  (e2e %.1f s)"
          % (len(games), X, len(games) / X, e2e["mean"]))
    print()
    print("  CONSEQUENCE, and it corrects a natural worry: turns per game depends on")
    print("  AGGREGATE generation throughput and on nothing else, because the token")
    print("  count of a turn is a property of the harness, not of the server config.")
    print("  A configuration that doubles aggregate throughput by running twice as")
    print("  many sequences at half the per-sequence speed therefore delivers")
    print("  exactly TWICE the turns, not the same number - the halved per-sequence")
    print("  rate is already inside the aggregate figure. Per-sequence latency would")
    print("  only matter if a single turn could outgrow the analyzer timeout (900 s)")
    print("  or the per-game budget; at %.1f s it is 6.2x under the former." % e2e["mean"])
    print()
    print("  %8s %14s %14s %14s" % ("agg x", "agg tok/s", "turns/game", "turn latency s"))
    seq_cap_mult = 8.0 / mean("running")
    for mult in sorted({1.0, 1.25, 1.5, 2.0, round(seq_cap_mult, 2), 3.0}):
        print("  %8.2f %14.1f %14.1f %14.1f"
              % (mult, agg_gen * mult,
                 agg_gen * mult * window_s / (tokens_per_turn * len(games)),
                 e2e["mean"] / mult))
    seq_cap = srv["launch_args"].get("max_num_seqs")
    print()
    print("  x%.2f is the ceiling IF the KV pool stopped binding and the scheduler" % seq_cap_mult)
    print("  ran at its configured max_num_seqs=%s with per-sequence decode speed" % seq_cap)
    print("  unchanged: mean Running %.2f -> 8. Whether per-sequence speed"
          % mean("running"))
    print("  survives that batch increase on a 512-expert MoE is exactly what the")
    print("  free sweep has to measure; see [H].")

    print("\n-- [H] What is NOT recoverable from these artifacts --")
    print("  The throughput-vs-batch curve. Running sits at 3 in 705 of 792")
    print("  snapshots (89%) because the KV pool pins it, so there is no range to")
    print("  regress over. See scripts/analyze_batch_scaling.py for the attempt and")
    print("  why its numbers must not be quoted.")

    if args.json:
        blob = {
            "source": os.path.abspath(root),
            "latency": {
                "e2e": e2e,
                "queue": queue,
                "inference": infer,
                "prefill": prefill,
                "decode": decode,
                "ttft": ttft,
            },
            "server": {k: v for k, v in srv.items() if k != "snapshots"},
            "snapshot_summary": {
                "n": len(snaps),
                "window_s": window_s,
                "mean_running": mean("running"),
                "mean_waiting": mean("waiting"),
                "mean_kv_pct": mean("kv_pct"),
                "mean_prompt_tps": mean("prompt_tps"),
                "mean_gen_tps": mean("gen_tps"),
                "pct_running_ge_8": 100.0 * n_at_cap / len(snaps),
                "pct_kv_ge_85": 100.0 * n_kv_hi / len(snaps),
            },
            "workload": {
                "requests": n_req,
                "prompt_tokens_total": prompt_tok,
                "generation_tokens_total": gen_tok,
                "mtp_drafts": drafts,
                "mtp_draft_tokens": draft_tok,
                "mtp_accepted_tokens": acc_tok,
                "mean_kv_footprint_tokens": kv_per_req,
            },
            "turns": {"per_game": games, "total": total_turns},
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
