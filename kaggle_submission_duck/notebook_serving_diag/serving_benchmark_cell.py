# =============================================================================
# stage7-duck-throughput: vLLM SERVING-CONFIGURATION benchmark at fixed concurrency
# =============================================================================
# Sibling of `notebook_concurrency_diag/benchmark_cell.py`. That cell held the
# serving configuration fixed and swept client concurrency; this one holds
# concurrency FIXED at the measured production optimum (37, see
# experiments/stage7_duck_concurrency.md) and sweeps the SERVER configuration.
#
# The measurement code (`_make_prompt`, `_one_request`, `_run_level`,
# `_scrape_metrics`) is carried over unchanged so the numbers this cell prints
# are directly comparable to that experiment's table -- same prompt generator,
# same seeds, same `max_tokens=512`, same `ignore_eos`, same accounting.
#
# WHAT IS BEING TESTED (see experiments/stage7_duck_throughput.md)
# ---------------------------------------------------------------
# A public fork (`keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp`) states that its
# only changes versus the stock Duck harness are to model serving, and it
# reports a large leaderboard gain. Its recipe is: NVFP4 weights + native
# 3-token NEXTN MTP speculative decoding + async scheduling + chunked prefill +
# CUDA graphs + prefix caching OFF + a 5 GiB KV cache + 8 vLLM sequences.
#
# Two facts found before writing this cell, both of which change the design:
#
#  1. The fork's `TAAF_VLLM_*` environment variables are read ONLY by its own
#     source bundle's `serving_setup.py`. Our bundle
#     (`jakobbrggen/taaf-kaggle-source-anim-20260807-anim`) contains ZERO
#     references to them -- setting them on our stack is a silent no-op. The
#     fork's "flags" are not flags we can set; they are a different serving
#     stack. So this cell mutates the REAL vLLM argv directly instead.
#
#  2. Our EXISTING production model already ships MTP weights. Its
#     `config.json` has `text_config.mtp_num_hidden_layers = 1` and the
#     checkpoint contains `mtp.safetensors` (our own notebook asserts that file
#     is present). The NVFP4 model's MTP block is the same shape
#     (`mtp.num_hidden_layers = 1`). So speculative decoding -- the recipe's
#     single biggest lever -- may be reachable with NO model swap at all.
#     That is the primary hypothesis this cell exists to test.
#
# METHOD
# ------
# Cells 1-5 above are byte-identical to the production submission notebook, so
# the bundled setup boots the real Qwen3.8-27B-FP8 server with the real
# production argv. This cell then:
#
#   * dumps `vllm serve --help` so flag availability is VERIFIED from the
#     installed vLLM, never assumed (three GPU runs were previously burned on
#     an assumed field name -- see the concurrency experiment's process note);
#   * recovers the baseline argv from `/proc/<pid>/cmdline` of the ALREADY
#     RUNNING server, so the baseline is what production actually launches
#     rather than a hand-copied approximation;
#   * measures the baseline at concurrency 37;
#   * then, for each subsequent config, stops the server, restarts it with a
#     mutated argv, waits for readiness, and measures the identical workload.
#
# A config whose flags the installed vLLM rejects is recorded as a failed row
# WITH its server-log tail, and the sweep continues (the next config starts
# from the baseline argv again, so one bad config cannot poison the rest).

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

# --- fixed workload (identical to the concurrency benchmark) ------------------
CONCURRENCY = 37  # the measured production optimum; held fixed here
PROMPT_CHARS_TARGET = 42000  # ~9.4K tokens, as measured in the concurrency run
MAX_OUTPUT_TOKENS = 512
WARMUP_CONCURRENCY = 4
WARMUP_OUTPUT_TOKENS = 32

# --- sweep budget ------------------------------------------------------------
# Each config costs a full server restart (model reload). Kaggle GPU sessions
# are finite and contended, so the sweep is ordered most-informative-first and
# stops cleanly when the budget is spent rather than being cut off mid-write.
SWEEP_DEADLINE_S = float(os.environ.get("DUCK_SWEEP_DEADLINE_S", 5.5 * 3600))
SERVER_READY_TIMEOUT_S = 1800.0

WORKING_DIR = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working"))
SERVER_PID_FILE = WORKING_DIR / "vllm-openai-server.pid"
SERVER_LOG = WORKING_DIR / "vllm-openai-server.log"
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
print("stage7-duck-throughput :: vLLM serving-configuration benchmark")
print("=" * 78, flush=True)
print(f"LOCAL_ANALYZER_BASE_URL  = {_base!r}")
print(f"INFERENCE_ANALYZER_MODEL = {MODEL_ID!r}")
print(f"concurrency (fixed)      = {CONCURRENCY}")
print(f"max_tokens               = {MAX_OUTPUT_TOKENS}")
print(f"sweep deadline           = {SWEEP_DEADLINE_S:.0f}s", flush=True)


# =============================================================================
# Step 0 -- VERIFY which flags this vLLM build actually supports.
# =============================================================================
# Instrument before theorising. Every flag in every config below is checked
# against this set; an unsupported flag is reported as such instead of being
# silently dropped or blindly sent.
def _vllm_serve_help() -> tuple[str, set[str]]:
    """Parse `vllm serve --help` for the flags this build actually accepts.

    Tries both entry points, because the bundled setup launches the server via
    `vllm.entrypoints.openai.api_server` while newer builds expose the CLI at
    `vllm.entrypoints.cli.main`.
    """
    env = os.environ.copy()
    attempts = [
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "--help"],
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
    ]
    last = ""
    for argv in attempts:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=300, env=env)
            text = (proc.stdout or "") + (proc.stderr or "")
        except Exception as exc:  # pragma: no cover - diagnostic only
            last = f"<{argv[2]}: {exc!r}>"
            continue
        flags = set(re.findall(r"(--[a-z0-9][a-z0-9\-]*)", text))
        # A real help dump has dozens of flags; a traceback has ~none.
        if len(flags) >= 20:
            return text, flags
        last = text[-500:]
    return f"<no usable help output; last={last!r}>", set()


_HELP_TEXT, SUPPORTED_FLAGS = _vllm_serve_help()
# CRITICAL: if help parsing failed, SUPPORTED_FLAGS is empty. Treating that as
# "no flag is supported" would skip EVERY config and waste the whole GPU run on
# a table full of "skipped". Empty means "unknown", so fall back to attempting
# each config and letting a real boot failure be the evidence.
HELP_USABLE = len(SUPPORTED_FLAGS) >= 20
print(f"\nvllm serve --help: {len(SUPPORTED_FLAGS)} distinct flags parsed "
      f"(usable={HELP_USABLE})", flush=True)
if not HELP_USABLE:
    print("  !! help output unusable -- flag-availability pre-checks DISABLED; "
          "every config will be attempted and judged by whether it boots.",
          flush=True)
    print(f"  raw: {_HELP_TEXT[:400]}", flush=True)
_FLAGS_OF_INTEREST = [
    "--speculative-config", "--async-scheduling", "--enable-prefix-caching",
    "--no-enable-prefix-caching", "--kv-cache-dtype", "--kv-cache-memory-bytes",
    "--max-num-seqs", "--max-num-batched-tokens", "--enable-chunked-prefill",
    "--no-enable-chunked-prefill", "--max-cudagraph-capture-size",
    "--gpu-memory-utilization", "--compilation-config", "--quantization",
    "--moe-backend", "--no-enable-log-requests", "--disable-uvicorn-access-log",
]
if HELP_USABLE:
    print("flag availability in THIS build (VERIFIED, not assumed):", flush=True)
    for _f in _FLAGS_OF_INTEREST:
        print(f"   {_f:<34} {'YES' if _f in SUPPORTED_FLAGS else 'no'}", flush=True)

try:
    import vllm as _vllm_mod  # noqa: F401
    print(f"\nvllm version: {_vllm_mod.__version__}", flush=True)
except Exception as _exc:  # pragma: no cover
    print(f"\nvllm import failed in notebook process: {_exc!r}", flush=True)


# =============================================================================
# Step 1 -- recover the REAL baseline argv from the running server.
# =============================================================================
def _read_pid() -> int | None:
    try:
        return int(SERVER_PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _argv_of(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [p.decode("utf-8", "replace") for p in raw.split(b"\x00") if p]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


BASELINE_PID = _read_pid()
BASELINE_ARGV: list[str] = []
if BASELINE_PID and _alive(BASELINE_PID):
    try:
        BASELINE_ARGV = _argv_of(BASELINE_PID)
    except Exception as exc:
        print(f"could not read /proc/{BASELINE_PID}/cmdline: {exc!r}", flush=True)

print("\n--- baseline vLLM argv (read from the LIVE process, not assumed) ---", flush=True)
if BASELINE_ARGV:
    for _tok in BASELINE_ARGV:
        print(f"  {_tok}")
else:
    print("  !! could not recover baseline argv -- restarts will be skipped", flush=True)


# =============================================================================
# Step 2 -- argv mutation helpers.
# =============================================================================
def _drop_flag(argv: list[str], flag: str, has_value: bool) -> list[str]:
    out: list[str] = []
    i = 0
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


def _apply(argv: list[str], *, drop: list[tuple[str, bool]] | None = None,
           add: list[str] | None = None) -> list[str]:
    out = list(argv)
    for flag, has_value in (drop or []):
        out = _drop_flag(out, flag, has_value)
    out.extend(add or [])
    return out


def _unsupported(add: list[str]) -> list[str]:
    """Which of the flags this config wants are absent from `vllm serve --help`.

    Returns [] when the help dump was unusable -- "unknown" must not be
    reported as "unsupported", or a parsing failure would silently cancel the
    entire sweep.
    """
    if not HELP_USABLE:
        return []
    return [tok for tok in add if tok.startswith("--") and tok not in SUPPORTED_FLAGS]


# =============================================================================
# Step 3 -- server lifecycle.
# =============================================================================
def _stop_server(pid: int | None, timeout_s: float = 180.0) -> bool:
    if not pid or not _alive(pid):
        return True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except OSError:
            return True
        deadline = time.time() + (timeout_s if sig != signal.SIGKILL else 30.0)
        while time.time() < deadline:
            if not _alive(pid):
                # The process is gone, but the port and the GPU allocation can
                # lag behind it. Wait until the old server genuinely stops
                # answering before letting a replacement bind the same port --
                # otherwise the next config could be measured against the
                # PREVIOUS server and silently report the wrong configuration.
                port_deadline = time.time() + 120.0
                while time.time() < port_deadline and _server_ready():
                    time.sleep(2)
                time.sleep(15)
                return True
            time.sleep(2)
    return not _alive(pid)


def _server_ready() -> bool:
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _start_server(argv: list[str], log_path: Path) -> tuple[int | None, str]:
    """Launch vLLM with `argv`; return (pid, status). Blocks until ready/failed."""
    env = os.environ.copy()
    handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(argv, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return None, f"spawn failed: {exc!r}"
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            return None, f"process exited early rc={proc.returncode}"
        if _server_ready():
            return proc.pid, "ready"
        time.sleep(5)
    _stop_server(proc.pid)
    return None, "timed out waiting for readiness"


def _log_tail(path: Path, lines: int = 40) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except Exception:
        return []


def _log_warnings(path: Path, limit: int = 12) -> list[str]:
    """Serving warnings/errors worth reporting alongside the throughput number."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    hits: list[str] = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in ("warning", "error", "not supported", "unsupported",
                                  "falling back", "fallback", "disabl", "ignor")):
            line = line.strip()
            if line and line not in hits:
                hits.append(line[:220])
        if len(hits) >= limit:
            break
    return hits


# =============================================================================
# Step 4 -- measurement (carried over from the concurrency benchmark unchanged).
# =============================================================================
_METRIC_KEYS = (
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:gpu_prefix_cache_hit_rate",
    # Speculative-decoding accounting: acceptance rate is the number that says
    # whether MTP is doing real work or merely burning draft compute.
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_emitted_tokens_total",
)


def _scrape_metrics() -> dict[str, float]:
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
                    out[key] = max(out.get(key, float("-inf")),
                                   float(line.rsplit(" ", 1)[1]))
                except (ValueError, IndexError):
                    pass
    return out


_WORD_POOL = (
    "grid cell frame segment component adjacency containment transition action "
    "reset click hypothesis observe predict verify colour region boundary shape "
    "object move rotate reflect fill count index level score attempt policy "
    "state node edge path search branch prune candidate evidence contradiction"
).split()


def _make_prompt(seed: int) -> str:
    """Unique-per-request prompt so prefix caching cannot collapse the prefill.

    Identical to the concurrency benchmark's generator (same seeding), so rows
    from the two experiments are directly comparable.
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
        payload["ignore_eos"] = True

    t_send = time.perf_counter()
    raw_lines: list[str] = []
    arrivals: list[float] = []
    usage = None
    status = None
    error = None
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
                    # VERIFIED on this exact server: the Qwen3 reasoning parser
                    # emits thinking tokens as `delta.reasoning` -- NOT
                    # `delta.content` and NOT `delta.reasoning_content`.
                    # Counting only `content` silently zeroed three prior GPU
                    # runs. Accept all three so a rename degrades gracefully.
                    if (delta.get("content") or delta.get("reasoning")
                            or delta.get("reasoning_content")):
                        arrivals.append(time.perf_counter())
    except Exception as exc:  # pragma: no cover - diagnostic only
        error = repr(exc)[:400]
    return {
        # Belt and braces: a server-reported completion_tokens > 0 means the
        # request genuinely succeeded even if the delta field went unrecognised.
        "ok": error is None and (bool(arrivals)
                                 or bool((usage or {}).get("completion_tokens", 0))),
        "status": status, "error": error, "raw_lines": raw_lines, "seed": seed,
        "t_send": t_send, "t_end": time.perf_counter(),
        "arrivals": arrivals, "usage": usage,
    }


async def _sampler(stop_event, samples: list[dict]):
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

    usage_out = sum((r["usage"] or {}).get("completion_tokens", 0) for r in ok)
    delta_out = sum(len(r["arrivals"]) for r in ok)
    total_out = usage_out or delta_out
    prompt_toks = [(r["usage"] or {}).get("prompt_tokens", 0) for r in ok if r["usage"]]

    e2e_wall = t1 - t0
    e2e_tps = total_out / e2e_wall if e2e_wall > 0 else 0.0

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

    drafted = _agg("vllm:spec_decode_num_draft_tokens_total")
    accepted = _agg("vllm:spec_decode_num_accepted_tokens_total")
    accept_rate = (accepted / drafted) if (drafted and drafted == drafted and drafted > 0) else float("nan")

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
        "spec_draft_tokens": drafted,
        "spec_accepted_tokens": accepted,
        "spec_accept_rate": accept_rate,
        "metric_samples": len(samples),
    }


# =============================================================================
# Step 5 -- the configurations, ordered most-informative-first.
# =============================================================================
# Each entry: (name, description, mutation dict applied to the BASELINE argv).
# Ordering matters: the sweep stops at the deadline, so anything that would be
# dropped is the least informative thing, never the primary hypothesis.
#
# Attribution is deliberate. `mtp3` and `flags` are each measured ALONE against
# the same baseline, so a combined result can never be reported as if it were
# attributable to one of them.
_SPEC_MTP = json.dumps({"method": "mtp", "num_speculative_tokens": 3},
                       separators=(",", ":"))
_SPEC_MTP2 = json.dumps({"method": "mtp", "num_speculative_tokens": 2},
                        separators=(",", ":"))
_SPEC_NGRAM = json.dumps(
    {"method": "ngram", "num_speculative_tokens": 3,
     "prompt_lookup_max": 4, "prompt_lookup_min": 2},
    separators=(",", ":"))

CONFIGS: list[dict] = [
    {
        "name": "baseline",
        "desc": "production argv, unchanged (already running)",
        "mutate": None,  # measured on the server the bundle already started
    },
    {
        "name": "mtp3",
        "desc": "baseline + 3-token MTP speculative decoding (NO model change)",
        "mutate": {"add": ["--speculative-config", _SPEC_MTP]},
    },
    {
        "name": "flags",
        "desc": "baseline + async sched, prefix caching OFF, fp8 KV (NO spec decode)",
        "mutate": {
            "drop": [("--enable-prefix-caching", False)],
            "add": ["--async-scheduling", "--no-enable-prefix-caching",
                    "--kv-cache-dtype", "fp8"],
        },
    },
    {
        "name": "mtp3+flags",
        "desc": "both of the above together (combined, NOT attributable alone)",
        "mutate": {
            "drop": [("--enable-prefix-caching", False)],
            "add": ["--speculative-config", _SPEC_MTP, "--async-scheduling",
                    "--no-enable-prefix-caching", "--kv-cache-dtype", "fp8"],
        },
    },
    {
        "name": "mtp2",
        "desc": "2-token MTP (is 3 past the acceptance sweet spot?)",
        "mutate": {"add": ["--speculative-config", _SPEC_MTP2]},
    },
    {
        "name": "ngram3",
        "desc": "n-gram spec decode -- the model-free fallback if MTP is unsupported",
        "mutate": {"add": ["--speculative-config", _SPEC_NGRAM]},
    },
]


async def _main() -> list[dict]:
    try:
        import aiohttp
    except ImportError:
        print("aiohttp unavailable -- cannot run the benchmark.", flush=True)
        return []

    rows: list[dict] = []
    current_pid = BASELINE_PID
    timeout = aiohttp.ClientTimeout(total=3600)

    for index, config in enumerate(CONFIGS):
        elapsed = time.time() - _SWEEP_T0
        if elapsed > SWEEP_DEADLINE_S:
            print(f"\n!! sweep deadline reached ({elapsed:.0f}s) -- stopping before "
                  f"config {config['name']!r}; {len(CONFIGS) - index} config(s) unmeasured",
                  flush=True)
            break

        print("\n" + "=" * 78, flush=True)
        print(f"CONFIG {index + 1}/{len(CONFIGS)}: {config['name']}  --  {config['desc']}")
        print("=" * 78, flush=True)

        row: dict = {"config": config["name"], "desc": config["desc"],
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
            add = config["mutate"].get("add", [])
            missing = _unsupported(add)
            if missing:
                # Report rather than attempt: an unsupported flag is a real,
                # reportable finding about this build, not a reason to guess.
                row["server_status"] = f"skipped: flags absent from this build: {missing}"
                row["argv"] = None
                rows.append(row)
                print(f"  SKIPPED -- this vLLM build has no {missing}", flush=True)
                continue

            argv = _apply(BASELINE_ARGV, drop=config["mutate"].get("drop"), add=add)
            row["argv"] = argv
            log_path = SWEEP_LOG_DIR / f"vllm-{config['name']}.log"
            print(f"  stopping current server (pid={current_pid}) ...", flush=True)
            if not _stop_server(current_pid):
                row["server_status"] = "failed to stop previous server"
                rows.append(row)
                print("  FAILED to stop previous server -- aborting sweep.", flush=True)
                break
            print(f"  starting: {' '.join(argv[-12:])}", flush=True)
            t_boot = time.time()
            pid, status = _start_server(argv, log_path)
            row["boot_s"] = time.time() - t_boot
            row["server_status"] = status
            if pid is None:
                row["log_tail"] = _log_tail(log_path, 40)
                rows.append(row)
                print(f"  BOOT FAILED ({status}) after {row['boot_s']:.0f}s. Log tail:",
                      flush=True)
                for line in row["log_tail"][-25:]:
                    print(f"   | {line}", flush=True)
                # Restore the baseline so later configs still have a server.
                print("  restoring baseline server ...", flush=True)
                pid, status = _start_server(BASELINE_ARGV, SERVER_LOG)
                current_pid = pid
                if pid is None:
                    print(f"  !! baseline restore failed ({status}) -- aborting sweep.",
                          flush=True)
                    break
                continue
            current_pid = pid
            print(f"  server ready in {row['boot_s']:.0f}s (pid={pid})", flush=True)

        row["warnings"] = _log_warnings(log_path)

        async with aiohttp.ClientSession(
            timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
        ) as session:
            probe = await _one_request(session, 999_000 + index, 8, True)
            allow_ignore_eos = probe["ok"]
            if not allow_ignore_eos:
                probe2 = await _one_request(session, 999_500 + index, 8, False)
                if not probe2["ok"]:
                    # Dump raw evidence rather than guessing -- see the process
                    # note in experiments/stage7_duck_concurrency.md.
                    print(f"  PROBE FAILED: status={probe2['status']} "
                          f"err={probe2['error']!r}", flush=True)
                    for line in (probe2.get("raw_lines") or [])[:15]:
                        print(f"   | {line}", flush=True)
                    print("  continuing anyway -- a probe that cannot parse the "
                          "stream does not prove the server is down.", flush=True)
            row["ignore_eos"] = allow_ignore_eos

            await _run_level(session, WARMUP_CONCURRENCY, WARMUP_OUTPUT_TOKENS,
                             900_000 + index * 100, allow_ignore_eos)
            measured = await _run_level(session, CONCURRENCY, MAX_OUTPUT_TOKENS,
                                        seed_base=1000 + index * 1000,
                                        allow_ignore_eos=allow_ignore_eos)
        row.update(measured)
        rows.append(row)
        print(
            f"  -> ok={row['requests_ok']}/{CONCURRENCY}  "
            f"e2e_agg={row['e2e_agg_tps']:.1f}  "
            f"decode_agg={row['decode_agg_tps']:.1f}  "
            f"per_seq={row['decode_per_seq_tps']:.2f}  "
            f"ttft_p50={row['ttft_p50_s']:.2f}  "
            f"preempt={row['preemptions_total_end']}  "
            f"accept={row['spec_accept_rate']}",
            flush=True,
        )
        await asyncio.sleep(10)

    return rows


_rows = await _main()  # noqa: F704 - notebook top-level await, as in cell 9


# =============================================================================
# Step 6 -- parseable result block, same style as the concurrency benchmark.
# =============================================================================
print("\n\n" + "=" * 78, flush=True)
print("BEGIN_SERVING_BENCHMARK_RESULTS")
print("=" * 78)
_hdr = (
    f"{'config':<12} {'ok':>7} {'e2e_agg':>9} {'dec_agg':>9} {'per_seq':>8} "
    f"{'ttft_p50':>9} {'preempt':>8} {'accept':>7} {'kv_max':>7} {'boot_s':>7}"
)
print(_hdr)
print("-" * len(_hdr))
for _r in _rows:
    if "e2e_agg_tps" not in _r:
        print(f"{_r['config']:<12} {'--':>7}  {_r.get('server_status', '?')}")
        continue
    _ar = _r.get("spec_accept_rate", float("nan"))
    _ar_s = f"{_ar:.3f}" if _ar == _ar else "n/a"
    print(
        f"{_r['config']:<12} "
        f"{_r['requests_ok']:>3}/{CONCURRENCY:<3} "
        f"{_r['e2e_agg_tps']:>9.1f} "
        f"{_r['decode_agg_tps']:>9.1f} "
        f"{_r['decode_per_seq_tps']:>8.2f} "
        f"{_r['ttft_p50_s']:>9.2f} "
        f"{_r['preemptions_total_end']:>8.0f} "
        f"{_ar_s:>7} "
        f"{_r['kv_cache_usage_max']:>7.3f} "
        f"{_r.get('boot_s', float('nan')):>7.0f}"
    )
print("-" * len(_hdr))
print("JSON_ROWS " + json.dumps(_rows, default=str))
print("=" * 78)
print("END_SERVING_BENCHMARK_RESULTS")
print("=" * 78, flush=True)

# --- attribution arithmetic, computed in-kernel from the measured numbers ----
_by = {r["config"]: r for r in _rows if "e2e_agg_tps" in r}
_bl = _by.get("baseline")
print("\nATTRIBUTION (vs. baseline, e2e aggregate tok/s)")
if _bl and _bl["e2e_agg_tps"] > 0:
    for _name in ("mtp3", "mtp2", "flags", "mtp3+flags", "ngram3"):
        _r = _by.get(_name)
        if not _r:
            print(f"  {_name:<12} not measured")
            continue
        _ratio = _r["e2e_agg_tps"] / _bl["e2e_agg_tps"]
        print(f"  {_name:<12} {_r['e2e_agg_tps']:>7.1f} tok/s  "
              f"x{_ratio:.3f}  ({(_ratio - 1) * 100:+.1f}%)")
    _m, _f = _by.get("mtp3"), _by.get("flags")
    if _m and _f:
        print("\n  Model-swap-free attribution:")
        print(f"    spec decoding alone : {(_m['e2e_agg_tps'] / _bl['e2e_agg_tps'] - 1) * 100:+.1f}%")
        print(f"    serving flags alone : {(_f['e2e_agg_tps'] / _bl['e2e_agg_tps'] - 1) * 100:+.1f}%")
else:
    print("  baseline row missing -- no attribution possible.")

print("\nWARNINGS PER CONFIG")
for _r in _rows:
    _w = _r.get("warnings") or []
    print(f"  --- {_r['config']} ({_r.get('server_status')})")
    for _line in _w[:8]:
        print(f"      {_line}")

print("\nbenchmark complete.", flush=True)
