# =============================================================================
# stage7-turn-latency: what actually limits Duck analyzer turns on the NVFP4 stack
# =============================================================================
# Sibling of `kaggle_submission_duck/notebook_serving_diag/serving_benchmark_cell.py`.
# The server-lifecycle and streaming-measurement code is carried over from it
# (including the hard-won detail that this vLLM build streams generated tokens
# as `delta.reasoning`, not `delta.content`, and that a request must be counted
# ok when the server reports `completion_tokens > 0`).
#
# WHAT IS BEING TESTED (see experiments/stage7_turn_latency.md)
# -------------------------------------------------------------
# The free NVFP4 public-25 run was 100% time-bound: all 25 games hit the 7920 s
# wall, each getting only ~52 analyzer turns at a median 153 s per turn. Its own
# `/metrics` scrape decomposes that turn as:
#
#     e2e 145.70 s = queue 126.71 s (86.97%) + inference 18.61 s (12.77%)
#                                               prefill  1.60 s ( 1.10%)
#                                               decode  17.01 s (11.67%)
#
# So turn latency IS queueing. The obvious suspect was
# `TAAF_VLLM_MAX_NUM_SEQS = 8` against 28 game threads. That suspect is
# REFUTED by the same run's server log, which never once shows the scheduler at
# its sequence cap:
#
#     Running min/p50/max = 2 / 3 / 6      (0 of 792 snapshots reached 8)
#     Waiting min/p50/max = 18 / 22 / 22
#     GPU KV cache size: 105,202 tokens, Maximum concurrency for
#         32,768 tokens per request: 3.21x
#     mean KV cache usage 80.3%, 191 preemptions
#     mean prompt 20,175 tok, mean generation 1,433 tok -> 21,608 tok resident
#     105,202 / 21,608 = 4.87 requests fit in the KV cache
#
# The binding constraint is KV-CACHE CAPACITY, not sequence slots. This sweep
# therefore measures both axes, most-informative-first, and prints a direct
# falsification test of the sequence-slot hypothesis (`seqs40` vs `baseline`)
# before anything else.
#
# The hybrid-allocator detail that makes the KV axis non-obvious, from the same
# log, is why `kv-cache-dtype fp8` is measured rather than reasoned about:
#
#     Setting attention block size to 1600 tokens to ensure that attention page
#     size is >= mamba page size.
#     Padding mamba page size by 0.25% to ensure that mamba page size and
#     attention page size are exactly equal.
#
# This is a hybrid GDN linear-attention model. Halving attention KV bytes does
# NOT halve the pool's per-request cost, because each sequence also holds a
# fixed-size mamba page, and vLLM re-equalises the page sizes. The net effect is
# not predictable from the log alone.
#
# WHAT IS REPORTED, AND WHY
# -------------------------
# Aggregate tok/s is NOT the decision quantity. Turns per game is:
#
#     turns_per_game = request_throughput [req/s] * 7920 s / n_games_in_flight
#
# A config that doubles aggregate throughput by running twice as many sequences
# at half the per-sequence speed delivers exactly the same number of turns. So
# every row prints request throughput and the turns-per-game it implies, next to
# the token rates.
#
# Queue / prefill / decode are read from the server's own Prometheus HISTOGRAM
# SUMS, delta'd across each level - the identical quantities the production run
# was decomposed with, so rows here are directly comparable to it.

import asyncio
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- workload, matched to the real public-25 run ------------------------------
# Production measured 20,175 prompt tokens and 1,433 generated tokens per
# request at 25 games in flight. Those are reproduced here rather than reusing
# the older benchmarks' 9.4K/512 shape, because the whole question is KV
# footprint and the footprint is set by exactly these two numbers.
CLIENT_CONCURRENCY = 25        # public-25: 25 games, each with one request in flight
REQUESTS_PER_LEVEL = 50        # two full waves through the pipe, semaphore-gated
TARGET_PROMPT_TOKENS = 20175
MAX_OUTPUT_TOKENS = 1433
WARMUP_CONCURRENCY = 4
WARMUP_OUTPUT_TOKENS = 32
GAME_BUDGET_S = 7920.0         # bm.solver.max_runtime_s_per_game
N_GAMES_PUBLIC25 = 25

SWEEP_DEADLINE_S = float(os.environ.get("DUCK_KV_SWEEP_DEADLINE_S", 6.5 * 3600))
SERVER_READY_TIMEOUT_S = 2400.0   # NVFP4 boot is ~7 min (135 GB model over NFS)

WORKING_DIR = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working"))
SERVER_LOG = WORKING_DIR / "vllm-openai-server.log"
IDENTITY_FILE = WORKING_DIR / "vllm-server-identity.json"
SWEEP_LOG_DIR = WORKING_DIR / "sweep-logs"
SWEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)

_base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "").rstrip("/")
MODEL_ID = os.environ.get("INFERENCE_ANALYZER_MODEL") or ""
if _base.endswith("/v1"):
    CHAT_URL = f"{_base}/chat/completions"
    MODELS_URL = f"{_base}/models"
    METRICS_URL = f"{_base[:-3].rstrip('/')}/metrics"
else:
    CHAT_URL = f"{_base}/v1/chat/completions"
    MODELS_URL = f"{_base}/v1/models"
    METRICS_URL = f"{_base}/metrics"

_SWEEP_T0 = time.time()

print("=" * 78, flush=True)
print("stage7-turn-latency :: KV-capacity vs sequence-slot benchmark (NVFP4 stack)")
print("=" * 78, flush=True)
print(f"LOCAL_ANALYZER_BASE_URL  = {_base!r}")
print(f"INFERENCE_ANALYZER_MODEL = {MODEL_ID!r}")
print(f"client concurrency       = {CLIENT_CONCURRENCY}  ({REQUESTS_PER_LEVEL} requests/level)")
print(f"target prompt tokens     = {TARGET_PROMPT_TOKENS}")
print(f"max_tokens               = {MAX_OUTPUT_TOKENS}")
print(f"sweep deadline           = {SWEEP_DEADLINE_S:.0f}s", flush=True)


# =============================================================================
# Step 0 -- stop the bundle's own vLLM watchdog.
# =============================================================================
# The serving bundle starts `vllm_server_watchdog.py`, which polls /v1/models
# every 15 s and RESTARTS the server after 4 consecutive failures (verified in
# the production run's `vllm-watchdog-events.jsonl`:
# {"failure_threshold": 4, "interval_seconds": 15.0, "max_restart_attempts": 2}).
# Every config change below takes the server down for minutes, so an active
# watchdog would race us and relaunch the ORIGINAL argv underneath a mutated
# config -- silently measuring the wrong server. Kill it first and say so.
def _kill_watchdog() -> list[int]:
    killed = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except (OSError, FileNotFoundError):
            continue
        if "vllm_server_watchdog" in cmd:
            pid = int(entry.name)
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except OSError:
                pass
    return killed


_WATCHDOG_PIDS = _kill_watchdog()
print(f"\nwatchdog processes terminated: {_WATCHDOG_PIDS or 'none found'}", flush=True)
time.sleep(3)
print(f"watchdog still present: {_kill_watchdog() or 'no'}", flush=True)


# =============================================================================
# Step 1 -- recover the REAL baseline argv.
# =============================================================================
# Preferred source is the bundle's own `vllm-server-identity.json`, which
# records the launch argv verbatim along with the resolved tuning dict. Falling
# back to /proc/<pid>/cmdline keeps the older benchmarks' method available if
# the bundle ever stops writing that file.
#: pid -> Popen for servers WE started, so they can be reaped. A killed child
#: stays a zombie until waited on, and a zombie still answers os.kill(pid, 0) --
#: that exact confusion aborted run 1 of the serving benchmark.
_OWNED: dict = {}

#: Our own process group. Never signal this one: it would kill the notebook.
_OWN_PGID = os.getpgrp()


def _alive(pid: int) -> bool:
    proc = _OWNED.get(pid)
    if proc is not None and proc.poll() is not None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return raw[raw.rfind(")") + 2:].split()[0] != "Z"
    except (FileNotFoundError, OSError, IndexError):
        return False


def _pgid_of(pid: int):
    try:
        return os.getpgid(pid)
    except OSError:
        return None


BASELINE_ARGV: list = []
BASELINE_PID = None
BASELINE_TUNING = {}
try:
    _ident = json.loads(IDENTITY_FILE.read_text(encoding="utf-8"))
    BASELINE_ARGV = list(_ident.get("argv") or [])
    BASELINE_PID = _ident.get("pid")
    BASELINE_TUNING = _ident.get("vllm_tuning") or {}
    print(f"\nbaseline argv source: {IDENTITY_FILE}", flush=True)
    print(f"  identity pid={BASELINE_PID} pgid={_ident.get('pgid')} "
          f"workers={[w.get('pid') for w in (_ident.get('workers') or [])]}", flush=True)
except Exception as exc:
    print(f"\ncould not read {IDENTITY_FILE}: {exc!r}", flush=True)

if not BASELINE_ARGV:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (OSError, FileNotFoundError):
            continue
        parts = [p.decode("utf-8", "replace") for p in raw.split(b"\x00") if p]
        if any("api_server" in p or ("serve" == p) for p in parts) and any(
                "vllm" in p for p in parts):
            BASELINE_ARGV, BASELINE_PID = parts, int(entry.name)
            print(f"baseline argv source: /proc/{entry.name}/cmdline", flush=True)
            break

print("--- baseline vLLM argv (read from a primary artifact, not assumed) ---", flush=True)
for _tok in BASELINE_ARGV:
    print(f"  {_tok}")
print(f"baseline resolved tuning: {json.dumps(BASELINE_TUNING, sort_keys=True)}", flush=True)
print(f"baseline pid alive: {bool(BASELINE_PID) and _alive(BASELINE_PID)}", flush=True)


def _gpu_free_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


# =============================================================================
# Step 2 -- argv mutation.
# =============================================================================
def _drop_flag(argv: list, flag: str, has_value: bool) -> list:
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2 if has_value else 1
            continue
        if argv[i].startswith(flag + "="):
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def _set_flag(argv: list, flag: str, value) -> list:
    """Set `flag` to `value` exactly once. `value=None` means a boolean flag.

    Implemented as drop-then-append rather than in-place substitution: an
    earlier in-place version silently DELETED a boolean flag that was already
    present (it consumed the existing token and then skipped the append because
    it had been "seen"), which would have quietly measured prefix caching OFF in
    the config whose entire point is turning it ON.
    """
    out = _drop_flag(argv, flag, has_value=value is not None)
    out.extend([flag] if value is None else [flag, str(value)])
    return out


# =============================================================================
# Step 3 -- server lifecycle.
# =============================================================================
def _server_ready() -> bool:
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _stop_server(pid, timeout_s: float = 300.0) -> bool:
    """Stop a vLLM server AND its worker processes.

    Signalling the API-server pid alone is not enough: this stack runs the
    engine core, the GPU worker and the PLE-offload worker as separate
    processes (verified in the production run's `vllm-server-identity.json`,
    which lists workers 155/223/224/242/305/388 under one pgid). The GPU
    allocation lives in the WORKER, so killing only the leader would leave
    ~82 GiB resident and make the next config look like it cannot fit.
    """
    if not pid or not _alive(pid):
        return True
    proc = _OWNED.get(pid)
    pgid = _pgid_of(pid)
    use_group = pgid is not None and pgid != _OWN_PGID
    if not use_group:
        print(f"    !! pgid({pid})={pgid} equals our own group {_OWN_PGID}; "
              f"signalling the pid only (workers may survive)", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            if use_group:
                os.killpg(pgid, sig)
            else:
                os.kill(pid, sig)
        except OSError:
            break
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        deadline = time.time() + (timeout_s if sig != signal.SIGKILL else 60.0)
        while time.time() < deadline:
            if not _alive(pid):
                break
            time.sleep(2)
        if not _alive(pid):
            break
    if _alive(pid):
        return False
    # The process is gone long before its ~82 GiB of weights are. Waiting on the
    # port alone is not enough: starting the next server while the previous
    # allocation is still resident is an OOM that would be misread as "this KV
    # size does not fit".
    port_deadline = time.time() + 180.0
    while time.time() < port_deadline and _server_ready():
        time.sleep(2)
    mem_deadline = time.time() + 300.0
    while time.time() < mem_deadline:
        free = _gpu_free_mib()
        if free != free or free > 80_000:  # NaN -> cannot check; proceed
            break
        time.sleep(5)
    print(f"    gpu free after stop: {_gpu_free_mib():.0f} MiB", flush=True)
    time.sleep(15)
    return True


def _start_server(argv: list, log_path: Path):
    env = os.environ.copy()
    handle = log_path.open("w", encoding="utf-8")
    try:
        # start_new_session puts the server in its own process group, so
        # _stop_server can killpg it (reaching the engine core and the GPU
        # workers) without ever signalling this notebook's own group.
        proc = subprocess.Popen(argv, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
    except Exception as exc:
        return None, f"spawn failed: {exc!r}"
    _OWNED[proc.pid] = proc
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            return None, f"process exited early rc={proc.returncode}"
        if _server_ready():
            return proc.pid, "ready"
        time.sleep(5)
    _stop_server(proc.pid)
    return None, "timed out waiting for readiness"


_KV_LINE = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, "
    r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_OOM_LINE = re.compile(r"memory allocation failed with OOM")


def _log_facts(path: Path) -> dict:
    """Pull the scheduler-capacity facts out of a server log."""
    facts = {"kv_cache_tokens": None, "kv_max_concurrency": None, "ooms": 0,
             "block_size_note": None}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return facts
    m = _KV_LINE.search(text)
    if m:
        facts["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        facts["kv_max_concurrency"] = float(m.group(3))
    facts["ooms"] = len(_OOM_LINE.findall(text))
    m = re.search(r"Setting attention block size to (\d+) tokens", text)
    if m:
        facts["block_size_note"] = int(m.group(1))
    return facts


def _log_tail(path: Path, lines: int = 40) -> list:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except Exception:
        return []


# =============================================================================
# Step 4 -- measurement.
# =============================================================================
# Histogram SUM/COUNT pairs, delta'd across a level. This is the same
# decomposition applied to the production run, so the rows are comparable.
_HIST = (
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:time_to_first_token_seconds",
)
_COUNTERS = (
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting",
           "vllm:kv_cache_usage_perc")


def _scrape() -> dict:
    out = {}
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=10) as response:
            body = response.read().decode("utf-8", "ignore")
    except Exception:
        return out
    for line in body.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        wanted = (name in _COUNTERS or name in _GAUGES
                  or any(name == h + s for h in _HIST for s in ("_sum", "_count")))
        if wanted:
            out[name] = out.get(name, 0.0) + val if name.endswith("_total") or \
                name.endswith("_sum") or name.endswith("_count") else max(
                    out.get(name, float("-inf")), val)
    return out


def _delta(before: dict, after: dict, key: str) -> float:
    if key not in after:
        return float("nan")
    return after[key] - before.get(key, 0.0)


_WORD_POOL = (
    "grid cell frame segment component adjacency containment transition action "
    "reset click hypothesis observe predict verify colour region boundary shape "
    "object move rotate reflect fill count index level score attempt policy "
    "state node edge path search branch prune candidate evidence contradiction"
).split()

_CHARS_PER_TOKEN = 4.45  # recalibrated in-kernel from a real usage report


def _make_prompt(seed: int, target_tokens: int) -> str:
    """Unique-per-request prompt so prefix caching cannot collapse the prefill.

    NOTE the deliberate limitation: production prompts are multimodal
    (MULTIMODAL_CONTEXT=current_grid, MULTIMODAL_UPSCALE=4), so part of their
    20,175 tokens are image tokens. These are text-only. KV footprint per token
    is identical either way, and prefill is only 1.10% of a production turn, so
    this matters for the vision-encoder cost and nothing else in this question.
    """
    rng = random.Random(seed * 7919 + 13)
    target_chars = int(target_tokens * _CHARS_PER_TOKEN)
    parts = [
        f"Session {seed}. You are analysing an unfamiliar grid puzzle environment.\n",
        "Below is a transcript of observations, segmentations and attempted actions.\n\n",
    ]
    total = sum(len(p) for p in parts)
    step = 0
    while total < target_chars:
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


async def _one_request(session, seed: int, max_tokens: int, allow_ignore_eos: bool,
                       target_tokens: int):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": _make_prompt(seed, target_tokens)}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if allow_ignore_eos:
        payload["ignore_eos"] = True

    t_send = time.perf_counter()
    raw_lines, arrivals, usage, status, error = [], [], None, None, None
    try:
        async with session.post(CHAT_URL, json=payload) as response:
            status = response.status
            if status != 200:
                error = (await response.text())[:400]
                return {"ok": False, "status": status, "error": error, "seed": seed,
                        "t_send": t_send, "t_end": time.perf_counter(),
                        "arrivals": [], "usage": None, "raw_lines": []}
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
                    # VERIFIED on this exact server family: the Qwen3 reasoning
                    # parser emits generated tokens as `delta.reasoning`, NOT
                    # `delta.content` and NOT `delta.reasoning_content`.
                    # Counting only `content` silently zeroed three prior GPU
                    # runs (see experiments/stage7_duck_concurrency.md).
                    if (delta.get("content") or delta.get("reasoning")
                            or delta.get("reasoning_content")):
                        arrivals.append(time.perf_counter())
    except Exception as exc:
        error = repr(exc)[:400]
    return {
        # A server-reported completion_tokens > 0 means the request genuinely
        # succeeded even if the delta field name went unrecognised.
        "ok": error is None and (bool(arrivals)
                                 or bool((usage or {}).get("completion_tokens", 0))),
        "status": status, "error": error, "raw_lines": raw_lines, "seed": seed,
        "t_send": t_send, "t_end": time.perf_counter(),
        "arrivals": arrivals, "usage": usage,
    }


async def _sampler(stop_event, samples: list):
    while not stop_event.is_set():
        s = _scrape()
        if s:
            samples.append(s)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def _run_level(session, concurrency: int, n_requests: int, max_tokens: int,
                     seed_base: int, allow_ignore_eos: bool,
                     target_tokens: int) -> dict:
    """Closed-loop level: `concurrency` requests in flight at all times.

    A semaphore, not a single gather of N, because the real harness is a closed
    loop -- each game thread issues its next request the moment the previous one
    returns. A plain gather would drain the pipe at the tail and understate
    steady-state queueing.
    """
    before = _scrape()
    samples: list = []
    stop_event = asyncio.Event()
    sampler_task = asyncio.create_task(_sampler(stop_event, samples))
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(i):
        async with sem:
            return await _one_request(session, seed_base + i, max_tokens,
                                      allow_ignore_eos, target_tokens)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[_guarded(i) for i in range(n_requests)])
    t1 = time.perf_counter()
    stop_event.set()
    await sampler_task
    after = _scrape()

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    wall = t1 - t0

    usage_out = sum((r["usage"] or {}).get("completion_tokens", 0) for r in ok)
    delta_out = sum(len(r["arrivals"]) for r in ok)
    total_out = usage_out or delta_out
    prompt_toks = [(r["usage"] or {}).get("prompt_tokens", 0) for r in ok if r["usage"]]

    # Per-request decode rate (degrades gracefully; the "all sequences decoding
    # simultaneously" window can vanish under speculative decoding).
    per_req_tps = []
    for r in ok:
        if len(r["arrivals"]) >= 2:
            span = r["arrivals"][-1] - r["arrivals"][0]
            if span > 0:
                per_req_tps.append((len(r["arrivals"]) - 1) / span)
    ttfts = sorted(r["arrivals"][0] - r["t_send"] for r in ok if r["arrivals"])
    e2es = sorted(r["t_end"] - r["t_send"] for r in ok)

    n_hist = _delta(before, after, "vllm:e2e_request_latency_seconds_count")

    def _mean_hist(base):
        c = _delta(before, after, base + "_count")
        s = _delta(before, after, base + "_sum")
        return (s / c) if (c and c == c and c > 0) else float("nan")

    drafted = _delta(before, after, "vllm:spec_decode_num_draft_tokens_total")
    accepted = _delta(before, after, "vllm:spec_decode_num_accepted_tokens_total")
    drafts = _delta(before, after, "vllm:spec_decode_num_drafts_total")
    gen_tok = _delta(before, after, "vllm:generation_tokens_total")

    def _gauge_mean(key):
        vals = [s[key] for s in samples if key in s]
        return (sum(vals) / len(vals)) if vals else float("nan")

    req_per_s = len(ok) / wall if wall > 0 else 0.0
    return {
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "first_error": (failed[0]["error"] if failed else None),
        "wall_s": wall,
        "out_tokens": total_out,
        "agg_out_tps": total_out / wall if wall > 0 else 0.0,
        "per_seq_tps": (sum(per_req_tps) / len(per_req_tps)) if per_req_tps else float("nan"),
        "req_per_s": req_per_s,
        # THE decision quantity: how many analyzer turns one game would get in
        # its 7920 s budget if `n_games` games shared this server.
        "turns_per_game": req_per_s * GAME_BUDGET_S / N_GAMES_PUBLIC25,
        "client_e2e_p50_s": e2es[len(e2es) // 2] if e2es else float("nan"),
        "client_ttft_p50_s": ttfts[len(ttfts) // 2] if ttfts else float("nan"),
        # server-side decomposition, the directly comparable numbers
        "srv_n": n_hist,
        "srv_e2e_s": _mean_hist("vllm:e2e_request_latency_seconds"),
        "srv_queue_s": _mean_hist("vllm:request_queue_time_seconds"),
        "srv_inference_s": _mean_hist("vllm:request_inference_time_seconds"),
        "srv_prefill_s": _mean_hist("vllm:request_prefill_time_seconds"),
        "srv_decode_s": _mean_hist("vllm:request_decode_time_seconds"),
        "srv_ttft_s": _mean_hist("vllm:time_to_first_token_seconds"),
        "prompt_tokens_mean": (sum(prompt_toks) / len(prompt_toks)) if prompt_toks else 0,
        "gen_tokens_delta": gen_tok,
        "preemptions": _delta(before, after, "vllm:num_preemptions_total"),
        "mtp_drafts": drafts,
        "mtp_draft_tokens": drafted,
        "mtp_accepted_tokens": accepted,
        "mtp_accept_rate": (accepted / drafted) if (drafted and drafted > 0) else float("nan"),
        "running_mean": _gauge_mean("vllm:num_requests_running"),
        "waiting_mean": _gauge_mean("vllm:num_requests_waiting"),
        "kv_usage_mean": _gauge_mean("vllm:kv_cache_usage_perc"),
        "metric_samples": len(samples),
    }


# =============================================================================
# Step 5 -- configurations, ordered most-informative-first.
# =============================================================================
# The sweep stops cleanly at the deadline, so the primary hypothesis and its
# falsification test are measured first and nothing important can be cut.
GB = 1024 ** 3
CONFIGS = [
    {"name": "baseline", "desc": "production argv (kv 5 GiB bf16, seqs 8, mtp3)",
     "mutate": None},
    {"name": "seqs40",
     "desc": "FALSIFICATION TEST: raise the sequence cap ONLY (8 -> 40)",
     "mutate": {"set": [("--max-num-seqs", 40)]}},
    {"name": "kvfp8-seqs40",
     "desc": "fp8 KV cache + seqs 40 -- more resident requests at ZERO extra GPU memory",
     "mutate": {"set": [("--kv-cache-dtype", "fp8"), ("--max-num-seqs", 40)]}},
    {"name": "kv8-seqs40",
     "desc": "KV pool 5 -> 8 GiB + seqs 40 -- does the card have room at all?",
     "mutate": {"set": [("--kv-cache-memory-bytes", 8 * GB), ("--max-num-seqs", 40)]}},
    {"name": "kv8-fp8-seqs40",
     "desc": "8 GiB + fp8 + seqs 40 -- the combined ceiling",
     "mutate": {"set": [("--kv-cache-memory-bytes", 8 * GB),
                        ("--kv-cache-dtype", "fp8"), ("--max-num-seqs", 40)]}},
    {"name": "seqs16",
     "desc": "fills in the sequence-cap axis between 8 and 40",
     "mutate": {"set": [("--max-num-seqs", 16)]}},
    {"name": "prefix-seqs40",
     "desc": "prefix caching ON + seqs 40 -- block sharing across games' common prefix",
     "mutate": {"drop": [("--no-enable-prefix-caching", False)],
                "set": [("--enable-prefix-caching", None), ("--max-num-seqs", 40)]}},
    {"name": "kv12-seqs40",
     "desc": "KV pool 12 GiB + seqs 40 -- how far does GPU memory actually go?",
     "mutate": {"set": [("--kv-cache-memory-bytes", 12 * GB), ("--max-num-seqs", 40)]}},
]


def _mutate(argv: list, spec: dict) -> list:
    out = list(argv)
    for flag, has_value in (spec.get("drop") or []):
        out = _drop_flag(out, flag, has_value)
    for flag, value in (spec.get("set") or []):
        out = _set_flag(out, flag, value)
    return out


async def _main() -> list:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp unavailable -- cannot run the benchmark.", flush=True)
        return []

    rows: list = []
    current_pid = BASELINE_PID
    timeout = aiohttp.ClientTimeout(total=7200)
    target_tokens = TARGET_PROMPT_TOKENS

    for index, config in enumerate(CONFIGS):
        elapsed = time.time() - _SWEEP_T0
        if elapsed > SWEEP_DEADLINE_S:
            print(f"\n!! deadline reached ({elapsed:.0f}s) -- stopping before "
                  f"{config['name']!r}; {len(CONFIGS) - index} config(s) unmeasured",
                  flush=True)
            break

        print("\n" + "=" * 78, flush=True)
        print(f"CONFIG {index + 1}/{len(CONFIGS)}: {config['name']}  --  {config['desc']}")
        print("=" * 78, flush=True)

        row = {"config": config["name"], "desc": config["desc"],
               "elapsed_at_start_s": elapsed}
        log_path = SERVER_LOG

        if config["mutate"] is None:
            row["argv"] = BASELINE_ARGV
            row["server_status"] = "baseline (already running)"
        else:
            if not BASELINE_ARGV:
                row["server_status"] = "skipped: baseline argv unknown"
                rows.append(row)
                print("  skipped -- no baseline argv to mutate.", flush=True)
                continue
            argv = _mutate(BASELINE_ARGV, config["mutate"])
            row["argv"] = argv
            log_path = SWEEP_LOG_DIR / f"vllm-{config['name']}.log"
            print(f"  stopping server pid={current_pid} ...", flush=True)
            if not _stop_server(current_pid):
                row["server_status"] = "failed to stop previous server"
                rows.append(row)
                print("  FAILED to stop previous server -- aborting sweep.", flush=True)
                break
            print(f"  starting with: {' '.join(str(t) for t in argv[4:])[:400]}", flush=True)
            t_boot = time.time()
            pid, status = _start_server(argv, log_path)
            row["boot_s"] = time.time() - t_boot
            row["server_status"] = status
            if pid is None:
                row["log_tail"] = _log_tail(log_path, 45)
                row.update(_log_facts(log_path))
                rows.append(row)
                print(f"  BOOT FAILED ({status}) after {row['boot_s']:.0f}s. Log tail:",
                      flush=True)
                for line in row["log_tail"][-25:]:
                    print(f"   | {line}", flush=True)
                print("  restoring baseline server ...", flush=True)
                pid, status = _start_server(BASELINE_ARGV, SERVER_LOG)
                current_pid = pid
                if pid is None:
                    print(f"  !! baseline restore failed ({status}) -- aborting.", flush=True)
                    break
                continue
            current_pid = pid
            print(f"  server ready in {row['boot_s']:.0f}s (pid={pid})", flush=True)

        row.update(_log_facts(log_path))
        print(f"  KV cache: {row.get('kv_cache_tokens')} tokens, "
              f"vLLM max concurrency {row.get('kv_max_concurrency')}x, "
              f"attn block {row.get('block_size_note')} tok, "
              f"log OOMs {row.get('ooms')}", flush=True)

        async with aiohttp.ClientSession(
                timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
            probe = await _one_request(session, 999_000 + index, 8, True, 2000)
            allow_ignore_eos = probe["ok"]
            if not allow_ignore_eos:
                probe = await _one_request(session, 999_500 + index, 8, False, 2000)
                if not probe["ok"]:
                    print(f"  PROBE FAILED: status={probe['status']} "
                          f"err={probe['error']!r}", flush=True)
                    for line in (probe.get("raw_lines") or [])[:15]:
                        print(f"   | {line}", flush=True)
            row["ignore_eos"] = allow_ignore_eos

            # Calibrate chars-per-token ONCE from a real usage report rather
            # than trusting a constant, so the prompt really is ~20,175 tokens.
            if index == 0 and (probe.get("usage") or {}).get("prompt_tokens"):
                got = probe["usage"]["prompt_tokens"]
                chars = len(_make_prompt(999_000, 2000))
                ratio = chars / got if got else _CHARS_PER_TOKEN
                target_tokens = TARGET_PROMPT_TOKENS
                globals()["_CHARS_PER_TOKEN"] = ratio
                print(f"  calibration: {chars} chars -> {got} prompt tokens "
                      f"({ratio:.2f} chars/token); prompts will target "
                      f"{TARGET_PROMPT_TOKENS} tokens", flush=True)

            await _run_level(session, WARMUP_CONCURRENCY, WARMUP_CONCURRENCY,
                             WARMUP_OUTPUT_TOKENS, 900_000 + index * 100,
                             allow_ignore_eos, 4000)
            measured = await _run_level(
                session, CLIENT_CONCURRENCY, REQUESTS_PER_LEVEL, MAX_OUTPUT_TOKENS,
                seed_base=1000 + index * 1000, allow_ignore_eos=allow_ignore_eos,
                target_tokens=target_tokens)
        row.update(measured)
        rows.append(row)
        print(
            f"  -> ok={row['requests_ok']}/{REQUESTS_PER_LEVEL}  "
            f"req/s={row['req_per_s']:.4f}  TURNS/GAME={row['turns_per_game']:.1f}  "
            f"agg={row['agg_out_tps']:.1f} tok/s  per_seq={row['per_seq_tps']:.1f}  "
            f"queue={row['srv_queue_s']:.1f}s infer={row['srv_inference_s']:.1f}s  "
            f"run={row['running_mean']:.2f} wait={row['waiting_mean']:.2f}  "
            f"preempt={row['preemptions']:.0f}  accept={row['mtp_accept_rate']:.3f}",
            flush=True)
        await asyncio.sleep(10)

    return rows


_rows = await _main()  # noqa: F704 - notebook top-level await, as in cell 9


# =============================================================================
# Step 6 -- parseable result block.
# =============================================================================
print("\n\n" + "=" * 78, flush=True)
print("BEGIN_KV_BENCHMARK_RESULTS")
print("=" * 78)
_hdr = (f"{'config':<16} {'ok':>6} {'req/s':>7} {'turns/gm':>9} {'agg_tps':>8} "
        f"{'per_seq':>8} {'queue_s':>8} {'infer_s':>8} {'run':>5} {'wait':>5} "
        f"{'kv_tok':>8} {'preempt':>8} {'accept':>7} {'boot_s':>7}")
print(_hdr)
print("-" * len(_hdr))
for _r in _rows:
    if "req_per_s" not in _r:
        print(f"{_r['config']:<16} {'--':>6}  {_r.get('server_status', '?')}")
        continue
    _ar = _r.get("mtp_accept_rate", float("nan"))
    print(f"{_r['config']:<16} {_r['requests_ok']:>2}/{REQUESTS_PER_LEVEL:<3} "
          f"{_r['req_per_s']:>7.4f} {_r['turns_per_game']:>9.1f} "
          f"{_r['agg_out_tps']:>8.1f} {_r['per_seq_tps']:>8.1f} "
          f"{_r['srv_queue_s']:>8.1f} {_r['srv_inference_s']:>8.1f} "
          f"{_r['running_mean']:>5.2f} {_r['waiting_mean']:>5.1f} "
          f"{str(_r.get('kv_cache_tokens')):>8} {_r['preemptions']:>8.0f} "
          f"{_ar:>7.3f} {_r.get('boot_s', float('nan')):>7.0f}")
print("-" * len(_hdr))
print("JSON_ROWS " + json.dumps(_rows, default=str))
print("=" * 78)
print("END_KV_BENCHMARK_RESULTS")
print("=" * 78, flush=True)

# --- the arithmetic that actually decides anything ---------------------------
_by = {r["config"]: r for r in _rows if "req_per_s" in r}
_bl = _by.get("baseline")
print("\nTURNS-PER-GAME (the decision quantity; aggregate tok/s is NOT)")
print("  turns_per_game = req_per_s * 7920 s / 25 games")
if _bl and _bl["req_per_s"] > 0:
    print(f"  {'config':<16} {'turns/game':>11} {'vs baseline':>12} {'agg tok/s':>11} {'vs base':>9}")
    for _name in [c["name"] for c in CONFIGS]:
        _r = _by.get(_name)
        if not _r:
            print(f"  {_name:<16} {'not measured':>11}")
            continue
        _t = _r["turns_per_game"] / _bl["turns_per_game"]
        _g = _r["agg_out_tps"] / _bl["agg_out_tps"] if _bl["agg_out_tps"] else float("nan")
        print(f"  {_name:<16} {_r['turns_per_game']:>11.1f} {_t:>11.3f}x "
              f"{_r['agg_out_tps']:>11.1f} {_g:>8.3f}x")
    _s40 = _by.get("seqs40")
    if _s40:
        _ratio = _s40["turns_per_game"] / _bl["turns_per_game"]
        print(f"\n  FALSIFICATION TEST -- raising max_num_seqs 8 -> 40 alone: "
              f"x{_ratio:.3f} turns/game")
        print(f"    baseline running_mean={_bl['running_mean']:.2f}  "
              f"seqs40 running_mean={_s40['running_mean']:.2f}")
        print("    If these are equal, the sequence cap was never the constraint "
              "and the KV pool is.")
else:
    print("  baseline row missing -- no comparison possible.")

print("\nPRODUCTION REFERENCE (free public-25 run, 2026-09-10, same stack):")
print("  req/s 0.1691   turns/game 53.6 (52.5 observed)   agg 242.1 tok/s")
print("  queue 126.71 s (86.97%)   inference 18.61 s   running_mean 3.13   "
      "waiting_mean 21.62")
print("  KV 105,202 tokens   21,608 tokens resident per request   191 preemptions")
print("\nbenchmark complete.", flush=True)
