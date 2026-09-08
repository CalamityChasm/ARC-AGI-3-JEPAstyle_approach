# =============================================================================
# stage7-duck-concurrency: vLLM aggregate-throughput vs. concurrency benchmark
# =============================================================================
# This cell REPLACES the Duck harness's own `bm.run(...)` game loop. Cells 1-5
# above are byte-identical to the production submission notebook, so vLLM is
# booted with the real Qwen3.8-27B-FP8 model, on the real RTX PRO 6000, with
# the real bundled server arguments. Nothing about the model, the machine
# shape, or the attention backend is changed.
#
# The question being measured (see experiments/stage7_duck_concurrency.md):
#
#   Total wall-clock in a competition rerun is fixed (~9h) and aggregate
#   throughput is a property of the vLLM server, so total tokens generated is
#   roughly `throughput x wall_clock` REGARDLESS of concurrency. Raising
#   concurrency 28 -> 37 gives each game more wall-clock but a proportionally
#   thinner slice of the GPU. Concurrency is therefore only a win if AGGREGATE
#   throughput actually RISES with more concurrent sequences.
#
# So: hold the workload fixed, sweep concurrency, measure aggregate output
# tokens/sec. Everything else (KV-cache utilisation, queueing, preemption) is
# recorded to explain whatever the throughput curve does.

import json
import os
import random
import time
import urllib.request

import asyncio

CONCURRENCY_LEVELS = [14, 28, 37, 48, 64]
PROMPT_CHARS_TARGET = 42000  # ~11-13K tokens; representative of the analyzer's
# 32K rolling window at steady state (see notebook 0 markdown for the caveat)
MAX_OUTPUT_TOKENS = 512
WARMUP_CONCURRENCY = 4
WARMUP_OUTPUT_TOKENS = 32

_base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "").rstrip("/")
MODEL_ID = os.environ.get("INFERENCE_ANALYZER_MODEL") or ""
if _base.endswith("/v1"):
    CHAT_URL = f"{_base}/chat/completions"
    METRICS_URL = f"{_base[:-3].rstrip('/')}/metrics"
else:
    CHAT_URL = f"{_base}/v1/chat/completions"
    METRICS_URL = f"{_base}/metrics"

print("=" * 78, flush=True)
print("stage7-duck-concurrency :: vLLM throughput-vs-concurrency benchmark")
print("=" * 78, flush=True)
print(f"LOCAL_ANALYZER_BASE_URL = {_base!r}")
print(f"INFERENCE_ANALYZER_MODEL = {MODEL_ID!r}")
print(f"chat url    = {CHAT_URL}")
print(f"metrics url = {METRICS_URL}")
print(f"levels      = {CONCURRENCY_LEVELS}")
print(f"max_tokens  = {MAX_OUTPUT_TOKENS}", flush=True)


# --- what the bundled setup actually launched vLLM with ----------------------
# The server's own --max-num-seqs is the single most decisive number for this
# question: if it sits below a swept level, that level cannot actually run
# that many sequences concurrently no matter what the client does.
def _dump_vllm_launch_args() -> None:
    print("\n--- bundled vLLM launch arguments (grepped from setup_commands.json) ---")
    path = BUNDLE_DIR / "setup_commands.json"  # noqa: F821  (defined in cell 3)
    try:
        commands = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  could not read {path}: {exc!r}")
        return
    needles = (
        "max-num-seqs",
        "max_num_seqs",
        "max-model-len",
        "max_model_len",
        "gpu-memory-utilization",
        "gpu_memory_utilization",
        "max-num-batched-tokens",
        "max_num_batched_tokens",
        "enable-prefix-caching",
        "enable_prefix_caching",
        "tensor-parallel",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_FLASHINFER_SAMPLER",
    )
    hits = 0
    for command in commands:
        for line in str(command).splitlines():
            if any(n in line for n in needles):
                print(f"  {line.strip()[:200]}")
                hits += 1
    if not hits:
        print("  (no matching lines -- vLLM args are probably defaults)")


_dump_vllm_launch_args()


# --- server-side metrics -----------------------------------------------------
_METRIC_KEYS = (
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:gpu_prefix_cache_hit_rate",
)


def _scrape_metrics() -> dict[str, float]:
    """Scrape the vLLM Prometheus endpoint. Returns {} if unavailable."""
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as response:
            body = response.read().decode("utf-8", "ignore")
    except Exception:
        return out
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        for key in _METRIC_KEYS:
            if line.startswith(key):
                try:
                    out[key] = max(out.get(key, float("-inf")), float(line.rsplit(" ", 1)[1]))
                except (ValueError, IndexError):
                    pass
    return out


# --- workload ----------------------------------------------------------------
_WORD_POOL = (
    "grid cell frame segment component adjacency containment transition action "
    "reset click hypothesis observe predict verify colour region boundary shape "
    "object move rotate reflect fill count index level score attempt policy "
    "state node edge path search branch prune candidate evidence contradiction"
).split()


def _make_prompt(seed: int) -> str:
    """A unique, realistically-long prompt.

    Deliberately UNIQUE per request (seeded RNG) so vLLM's prefix cache cannot
    collapse the prefill across the concurrent batch -- in the real harness
    every game holds its own independent conversation, so a shared-prefix
    workload would flatter the server in a way the real run never sees.
    """
    rng = random.Random(seed * 7919 + 13)
    parts = [
        f"Session {seed}. You are analysing an unfamiliar grid puzzle environment.\n",
        "Below is a transcript of observations, segmentations and attempted actions.\n\n",
    ]
    total = sum(len(p) for p in parts)
    step = 0
    while total < PROMPT_CHARS_TARGET:
        step += 1
        words = " ".join(rng.choice(_WORD_POOL) for _ in range(rng.randint(14, 30)))
        chunk = (
            f"[step {step}] observation: {words}. "
            f"segmentation: {rng.randint(2, 40)} components, "
            f"largest={rng.randint(3, 900)} px at ({rng.randint(0, 63)},{rng.randint(0, 63)}). "
            f"action taken: ACTION{rng.randint(1, 7)} -> "
            f"{'frame changed' if rng.random() < 0.5 else 'no change'}.\n"
        )
        parts.append(chunk)
        total += len(chunk)
    parts.append(
        "\nWrite a detailed step-by-step analysis of what mechanic this environment "
        "most likely implements, and what you would try next. Be thorough and specific.\n"
    )
    return "".join(parts)


async def _one_request(session, seed: int, max_tokens: int, allow_ignore_eos: bool):
    """Stream one completion; return per-token arrival timestamps."""
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": _make_prompt(seed)}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if allow_ignore_eos:
        # Forces every sequence to generate exactly max_tokens, so the batch is
        # a fixed, identical workload at every concurrency level rather than
        # one whose size depends on when the model happens to stop.
        payload["ignore_eos"] = True

    t_send = time.perf_counter()
    raw_lines: list[str] = []
    arrivals: list[float] = []
    usage = None
    status = None
    ctype = None
    error = None
    try:
        async with session.post(CHAT_URL, json=payload) as response:
            status = response.status
            ctype = response.headers.get("Content-Type", "")
            if status != 200:
                error = (await response.text())[:400]
                return {
                    "ok": False, "status": status, "error": error, "seed": seed,
                    "t_send": t_send, "t_end": time.perf_counter(), "arrivals": [],
                    "usage": None,
                }
            async for raw in response.content:
                line = raw.decode("utf-8", "ignore").strip()
                if len(raw_lines) < 25:
                    raw_lines.append(line[:300])
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # This server runs with reasoning_parser=qwen3 /
                    # preserve_thinking, so generated tokens arrive in
                    # `reasoning_content` while the model is thinking and only
                    # move to `content` once it emits its final answer. Counting
                    # only `content` undercounts real decode work -- and, at the
                    # small max_tokens used by the readiness probe, produces
                    # ZERO arrivals for a perfectly healthy server (observed:
                    # status=200, error=None, yet ok=False -> "FATAL: server not
                    # answering", which aborted the first benchmark run before
                    # it measured anything). Both fields are decoded output
                    # tokens and both count toward throughput.
                    if delta.get("content") or delta.get("reasoning_content"):
                        arrivals.append(time.perf_counter())
    except Exception as exc:  # pragma: no cover - diagnostic only
        error = repr(exc)[:400]
    return {
        "ok": error is None and bool(arrivals),
        "status": status,
        "error": error,
        "raw_lines": raw_lines,
        "content_type": ctype,
        "seed": seed,
        "t_send": t_send,
        "t_end": time.perf_counter(),
        "arrivals": arrivals,
        "usage": usage,
    }


async def _sampler(stop_event, samples: list[dict]):
    """Poll /metrics once a second for the duration of a level."""
    while not stop_event.is_set():
        metrics = _scrape_metrics()
        if metrics:
            samples.append(metrics)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


async def _run_level(session, concurrency: int, max_tokens: int, seed_base: int,
                     allow_ignore_eos: bool) -> dict:
    """Fire `concurrency` simultaneous requests and measure the batch."""
    samples: list[dict] = []
    stop_event = asyncio.Event()
    sampler_task = asyncio.create_task(_sampler(stop_event, samples))

    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        _one_request(session, seed_base + i, max_tokens, allow_ignore_eos)
        for i in range(concurrency)
    ])
    t1 = time.perf_counter()
    stop_event.set()
    await sampler_task

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    # Token accounting: prefer the server's own usage numbers; fall back to
    # counting streamed content deltas (1 delta ~= 1 token in vLLM).
    usage_out = sum((r["usage"] or {}).get("completion_tokens", 0) for r in ok)
    delta_out = sum(len(r["arrivals"]) for r in ok)
    total_out = usage_out or delta_out
    prompt_toks = [
        (r["usage"] or {}).get("prompt_tokens", 0) for r in ok if r["usage"]
    ]

    # END-TO-END: what the harness actually experiences -- includes prefill,
    # queueing and ramp. This is the number that governs "tokens generated in
    # a fixed 9h wall-clock".
    e2e_wall = t1 - t0
    e2e_tps = total_out / e2e_wall if e2e_wall > 0 else 0.0

    # STEADY DECODE: the window in which ALL `concurrency` sequences are
    # simultaneously decoding -- from the last request's first token to the
    # first request's last token. Isolates decode from prefill/ramp.
    decode_tps = float("nan")
    decode_window = float("nan")
    decode_tokens = 0
    firsts = [r["arrivals"][0] for r in ok if r["arrivals"]]
    lasts = [r["arrivals"][-1] for r in ok if r["arrivals"]]
    if firsts and lasts:
        w_start, w_end = max(firsts), min(lasts)
        if w_end > w_start:
            decode_window = w_end - w_start
            for r in ok:
                decode_tokens += sum(1 for t in r["arrivals"] if w_start <= t <= w_end)
            decode_tps = decode_tokens / decode_window

    ttfts = [r["arrivals"][0] - r["t_send"] for r in ok if r["arrivals"]]

    def _agg(key, fn=max):
        vals = [s[key] for s in samples if key in s]
        return fn(vals) if vals else float("nan")

    return {
        "concurrency": concurrency,
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "first_error": (failed[0]["error"] if failed else None),
        "e2e_wall_s": e2e_wall,
        "e2e_out_tokens": total_out,
        "e2e_agg_tps": e2e_tps,
        "e2e_per_seq_tps": e2e_tps / concurrency if concurrency else 0.0,
        "decode_window_s": decode_window,
        "decode_tokens": decode_tokens,
        "decode_agg_tps": decode_tps,
        "decode_per_seq_tps": decode_tps / concurrency if concurrency else float("nan"),
        "ttft_p50_s": sorted(ttfts)[len(ttfts) // 2] if ttfts else float("nan"),
        "ttft_max_s": max(ttfts) if ttfts else float("nan"),
        "prompt_tokens_mean": (sum(prompt_toks) / len(prompt_toks)) if prompt_toks else 0,
        "kv_cache_usage_max": _agg("vllm:gpu_cache_usage_perc"),
        "running_max": _agg("vllm:num_requests_running"),
        "waiting_max": _agg("vllm:num_requests_waiting"),
        "preemptions_total_end": _agg("vllm:num_preemptions_total"),
        "metric_samples": len(samples),
    }


async def _main() -> list[dict]:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp unavailable -- cannot run the benchmark.", flush=True)
        return []

    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=0)
    rows: list[dict] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Probe once to find out whether the server accepts `ignore_eos`.
        probe = await _one_request(session, 999_000, 8, True)
        allow_ignore_eos = probe["ok"]
        if not allow_ignore_eos:
            print(f"note: ignore_eos probe failed (status={probe['status']}, "
                  f"err={probe['error']!r}); retrying without it", flush=True)
            probe = await _one_request(session, 999_001, 8, False)
            if not probe["ok"]:
                # Do NOT guess at the cause -- dump exactly what came back.
                # A previous run failed here with status=200/err=None and a
                # first fix (counting reasoning_content) did not change it,
                # so the raw stream is the only reliable evidence.
                print(f"PROBE FAILED: status={probe['status']} "
                      f"err={probe['error']!r} "
                      f"content_type={probe.get('content_type')!r} "
                      f"n_raw_lines={len(probe.get('raw_lines') or [])}", flush=True)
                print("---- RAW PROBE RESPONSE (first 25 lines) ----", flush=True)
                for _l in (probe.get("raw_lines") or []):
                    print(f"  | {_l}", flush=True)
                print("---- END RAW PROBE RESPONSE ----", flush=True)
                print("continuing anyway -- measuring is the point; a probe "
                      "that cannot parse the stream does not prove the server "
                      "is down.", flush=True)
        print(f"\nignore_eos accepted = {allow_ignore_eos}", flush=True)

        print(f"warmup: {WARMUP_CONCURRENCY} x {WARMUP_OUTPUT_TOKENS} tokens ...", flush=True)
        await _run_level(session, WARMUP_CONCURRENCY, WARMUP_OUTPUT_TOKENS,
                         900_000, allow_ignore_eos)
        print("warmup done.\n", flush=True)

        base_pre = _scrape_metrics()
        print(f"idle metrics: {base_pre}\n", flush=True)

        for i, level in enumerate(CONCURRENCY_LEVELS):
            print(f"--- level {level} (concurrency) starting ...", flush=True)
            row = await _run_level(session, level, MAX_OUTPUT_TOKENS,
                                   seed_base=1000 + i * 1000,
                                   allow_ignore_eos=allow_ignore_eos)
            rows.append(row)
            print(
                f"    conc={row['concurrency']:>3}  "
                f"ok={row['requests_ok']}/{row['concurrency']}  "
                f"e2e_agg={row['e2e_agg_tps']:.1f} tok/s  "
                f"decode_agg={row['decode_agg_tps']:.1f} tok/s  "
                f"per_seq={row['decode_per_seq_tps']:.2f} tok/s  "
                f"kv={row['kv_cache_usage_max']:.3f}  "
                f"waiting_max={row['waiting_max']}",
                flush=True,
            )
            # Let the scheduler drain fully so the next level starts clean.
            await asyncio.sleep(10)
    return rows


_rows = await _main()  # noqa: F704 - notebook top-level await, as in cell 9

# --- clearly-parseable result block ------------------------------------------
print("\n\n" + "=" * 78, flush=True)
print("BEGIN_CONCURRENCY_BENCHMARK_RESULTS")
print("=" * 78)
_hdr = (
    f"{'conc':>5} {'ok':>6} {'e2e_agg':>9} {'dec_agg':>9} {'per_seq':>8} "
    f"{'kv_max':>7} {'wait_max':>9} {'ttft_p50':>9} {'preempt':>8} {'prompt_tok':>11}"
)
print(_hdr)
print("-" * len(_hdr))
for _r in _rows:
    print(
        f"{_r['concurrency']:>5} "
        f"{_r['requests_ok']:>3}/{_r['concurrency']:<2} "
        f"{_r['e2e_agg_tps']:>9.1f} "
        f"{_r['decode_agg_tps']:>9.1f} "
        f"{_r['decode_per_seq_tps']:>8.2f} "
        f"{_r['kv_cache_usage_max']:>7.3f} "
        f"{_r['waiting_max']:>9.1f} "
        f"{_r['ttft_p50_s']:>9.2f} "
        f"{_r['preemptions_total_end']:>8.0f} "
        f"{_r['prompt_tokens_mean']:>11.0f}"
    )
print("-" * len(_hdr))
print("JSON_ROWS " + json.dumps(_rows))
print("=" * 78)
print("END_CONCURRENCY_BENCHMARK_RESULTS")
print("=" * 78, flush=True)

# --- the verdict arithmetic, computed in-kernel from the measured numbers ----
if len(_rows) >= 2:
    _by_conc = {r["concurrency"]: r for r in _rows}
    _r28, _r37 = _by_conc.get(28), _by_conc.get(37)
    print("\nVERDICT_INPUTS")
    if _r28 and _r37 and _r28["decode_agg_tps"] > 0:
        _ratio = _r37["decode_agg_tps"] / _r28["decode_agg_tps"]
        print(f"  aggregate decode throughput 37 / 28 = {_ratio:.4f}")
        print(f"  (>1.00 => raising concurrency generates MORE total tokens in a")
        print(f"   fixed 9h wall-clock; ~1.00 => token-neutral, only redistributes;")
        print(f"   <1.00 => raising concurrency generates FEWER total tokens)")
        print(f"  per-game token share 28 -> 37: waves 4 -> 3, so per-game "
              f"wall-clock x {4/3:.3f}, GPU slice x {_ratio*28/37:.3f}, "
              f"net per-game tokens x {_ratio*(28/37)*(4/3):.3f}")
    _best = max(_rows, key=lambda r: (r["decode_agg_tps"] if r["decode_agg_tps"] == r["decode_agg_tps"] else -1))
    print(f"  peak aggregate decode throughput at concurrency={_best['concurrency']}: "
          f"{_best['decode_agg_tps']:.1f} tok/s")
print("\nbenchmark complete.", flush=True)
