"""Live diagnostic driver for CodeWorldAgent on real ARC-AGI-3 games.

This answers ONE question that no local test can answer, because the local
box (RTX 2070, 8GB) cannot host Qwen3-Coder-30B-A3B:

    Can the coder model actually write a replay-passing WorldModel for a
    real 64x64 ARC-AGI-3 game?

Everything in PR #8 (the unconditional-skeleton-fallback fix, the sandbox
allowlist, extract_code, transcript-preserving retries) is worthless if the
answer is no.

Why this is a driver and not `main.py`
--------------------------------------
`main.py` gets its game list from a live HTTP call to `{ROOT_URL}/api/games`
*before* anything else runs, regardless of OPERATION_MODE. A free diagnostic
push has `enable_internet: false` and no gateway sidecar, so that call always
fails and main.py exits with "No games available to play". This driver keeps
every other part of the real path identical -- same `Swarm`, same
`Arcade`-created environments, same `CodeWorldAgent` -- and only replaces the
game-listing step with a direct read of the scanned OFFLINE environments.

Deliberate deviations from the real submission, all printed at runtime:
  * OPERATION_MODE=offline against the competition's own `environment_files`
    (no gateway available on a free push).
  * 2 games, not the full roster; MAX_ACTIONS and the coder budget are
    reduced so the run fits comfortably inside a free GPU session.
  * ACTION_MODEL_DIR is pointed at the *coder* model directory so only one
    model is resident. The action head is separately budget-disabled
    (ACTION_LLM_CALL_BUDGET=0), so it is never consulted; this only avoids
    holding a second multi-GB model in VRAM alongside the 30B coder. The
    Gemma mount path is still resolved and printed for parity.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback

T0 = time.time()


def el() -> str:
    return f"{time.time() - T0:7.1f}s"


def say(*parts: object) -> None:
    print(f"[{el()}]", *parts, flush=True)


def banner(title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"== {title}", flush=True)
    print("=" * 78, flush=True)


# --- knobs ----------------------------------------------------------------
N_GAMES = int(os.getenv("DIAG_N_GAMES", "2"))
MAX_ACTIONS = int(os.getenv("DIAG_MAX_ACTIONS", "40"))
CODER_BUDGET = int(os.getenv("DIAG_CODER_BUDGET", "3"))
DRAFT_ATTEMPTS = int(os.getenv("DIAG_DRAFT_ATTEMPTS", "5"))
REPAIR_ATTEMPTS = int(os.getenv("DIAG_REPAIR_ATTEMPTS", "2"))
# Hard wall-clock guards, so a slow model degrades into "we have partial
# evidence" instead of "the kernel was killed and we have none".
LLM_DEADLINE_S = float(os.getenv("DIAG_LLM_DEADLINE_MIN", "240")) * 60.0
RUN_DEADLINE_S = float(os.getenv("DIAG_RUN_DEADLINE_MIN", "300")) * 60.0

EVIDENCE_PATH = os.getenv("DIAG_EVIDENCE", "/kaggle/working/diag_evidence.json")

EVIDENCE: dict = {
    "config": {
        "n_games": N_GAMES,
        "max_actions": MAX_ACTIONS,
        "coder_budget": CODER_BUDGET,
        "draft_attempts": DRAFT_ATTEMPTS,
        "repair_attempts": REPAIR_ATTEMPTS,
    },
    "llm_calls": [],      # one entry per raw LLM completion
    "load_results": [],   # one entry per load_world_model() call
    "replay_results": [],  # one entry per replay() call
    "rounds": [],         # one entry per draft/repair round
    "games": [],          # per-game final state
    "errors": [],
}


def flush_evidence() -> None:
    try:
        with open(EVIDENCE_PATH, "w", encoding="utf-8") as fh:
            json.dump(EVIDENCE, fh, indent=2)
    except Exception as exc:  # noqa: BLE001
        say("WARNING: could not write evidence file:", exc)


# --- setup ----------------------------------------------------------------
HARNESS = os.getenv("DIAG_HARNESS_DIR", "/kaggle/working/ARC-AGI-3-Agents")
WORKDIR = os.getenv("DIAG_WORK_DIR", "/kaggle/working")
for p in (HARNESS, WORKDIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(HARNESS)

banner("STEP 1 -- environment")
say("python", sys.version.split()[0])
try:
    import torch

    say("torch", torch.__version__, "cuda_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        say(
            "gpu:", props.name,
            f"| capability sm_{props.major}{props.minor}",
            f"| {props.total_memory / 1e9:.1f} GB",
        )
        say("bf16 supported:", torch.cuda.is_bf16_supported())
except Exception:  # noqa: BLE001
    say("torch import failed:")
    traceback.print_exc()

say("CODER_MODEL_DIR  =", os.getenv("CODER_MODEL_DIR"))
say("ACTION_MODEL_DIR =", os.getenv("ACTION_MODEL_DIR"))
say("GEMMA_MODEL_DIR (resolved, not loaded) =", os.getenv("GEMMA_MODEL_DIR_PARITY"))
say("LLM_BACKEND      =", os.getenv("LLM_BACKEND"))
say("ENVIRONMENTS_DIR =", os.getenv("ENVIRONMENTS_DIR"))
say("OPERATION_MODE   =", os.getenv("OPERATION_MODE"))

banner("STEP 2 -- verify the deployed code is the FIXED code")
from llm_engine import drafting, replay as replay_mod, world_model as wm_mod  # noqa: E402
from llm_engine.world_model import WORLD_MODEL_SKELETON  # noqa: E402

_wm_src = open(wm_mod.__file__, encoding="utf-8").read()
_draft_src = open(drafting.__file__, encoding="utf-8").read()
checks = {
    "sandbox allowlist contains 'super'": '"super"' in _wm_src,
    "sandbox allowlist contains 'map'": '"map"' in _wm_src,
    "sandbox allowlist contains 'filter'": '"filter"' in _wm_src,
    "drafting has NO unconditional skeleton fallback": "fallback = load_world_model" not in _draft_src,
    "retry prompt re-includes transcript": "_render_transcript(transcript)" in _draft_src,
    "DRAFT_MAX_TOKENS >= 4096": getattr(drafting, "DRAFT_MAX_TOKENS", 0) >= 4096,
}
for k, v in checks.items():
    say(("  OK   " if v else "  FAIL "), k)
EVIDENCE["fixed_code_checks"] = checks
say("WORLD_MODEL_SKELETON length =", len(WORLD_MODEL_SKELETON))

banner("STEP 3 -- instrument drafting")

_orig_safe_complete = drafting._safe_complete
_orig_load = drafting.load_world_model
_orig_replay = drafting.replay
_call_counter = {"n": 0}
# A real import statement, not a docstring line that happens to start with "from".
_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+[A-Za-z_.][\w.]*"
    r"|from\s+[A-Za-z_.][\w.]*\s+import\s)"
)


def _instr_safe_complete(client, system, user, max_tokens):  # type: ignore[no-untyped-def]
    n = _call_counter["n"] = _call_counter["n"] + 1
    if time.time() - T0 > LLM_DEADLINE_S:
        say(f"LLM call #{n}: PAST DEADLINE, returning None (aborting this round)")
        EVIDENCE["llm_calls"].append({"call": n, "skipped": "past deadline"})
        return None
    say(f"[{threading.current_thread().name}] --- LLM call #{n} begins (prompt {len(system) + len(user)} chars, max_new_tokens={max_tokens})")
    t = time.time()
    out = _orig_safe_complete(client, system, user, max_tokens)
    dt = time.time() - t
    say(f"--- LLM call #{n} returned in {dt:.1f}s: {('None (client failure)' if out is None else str(len(out)) + ' chars')}")
    rec = {
        "call": n,
        "seconds": round(dt, 1),
        "prompt_chars": len(system) + len(user),
        "response_chars": None if out is None else len(out),
        "response": out,
    }
    EVIDENCE["llm_calls"].append(rec)
    if out is not None:
        print(f"----- RAW RESPONSE #{n} (verbatim, first 6000 chars) -----", flush=True)
        print(out[:6000], flush=True)
        print(f"----- END RAW RESPONSE #{n} -----", flush=True)
    flush_evidence()
    return out


def _instr_load(source):  # type: ignore[no-untyped-def]
    res = _orig_load(source)
    is_stub = source.strip() == WORLD_MODEL_SKELETON.strip()
    hazard = sorted(
        n for n in ("super", "map", "filter", "reversed", "type", "object", "divmod", "pow")
        if f"{n}(" in source
    )
    imports = [ln for ln in source.splitlines() if _IMPORT_RE.match(ln)]
    rec = {
        "call": _call_counter["n"],
        "thread": threading.current_thread().name,
        "source_chars": len(source),
        "load_ok": res.ok,
        "load_error": res.error,
        "is_exact_skeleton": is_stub,
        "builtins_used_beyond_original_allowlist": hazard,
        "import_statements": imports,
    }
    EVIDENCE["load_results"].append(rec)
    say(
        f"load_world_model: ok={res.ok} err={res.error!r} "
        f"is_exact_skeleton={is_stub} extra_builtins={hazard} imports={imports}"
    )
    print(f"----- EXTRACTED CANDIDATE SOURCE (after LLM call #{_call_counter['n']}), verbatim -----", flush=True)
    print(source, flush=True)
    print("----- END CANDIDATE SOURCE -----", flush=True)
    flush_evidence()
    return res


def _instr_replay(transcript, model):  # type: ignore[no-untyped-def]
    res = _orig_replay(transcript, model)
    first = res.first_failure
    rec = {
        "call": _call_counter["n"],
        "game_id": transcript.game_id,
        "passed": res.passed,
        "pass_count": res.pass_count,
        "total": res.total,
        "first_failure_index": None if first is None else first.index,
        "first_failure_reason": None if first is None else (first.reason or "")[:2000],
    }
    EVIDENCE["replay_results"].append(rec)
    say(
        f"REPLAY {transcript.game_id}: passed={res.passed} matched={res.pass_count}/{res.total}"
        + ("" if first is None else f" first_divergence=#{first.index}")
    )
    if first is not None:
        print("----- FIRST DIVERGENCE -----", flush=True)
        print((first.reason or "")[:2500], flush=True)
        print("----- END FIRST DIVERGENCE -----", flush=True)
    flush_evidence()
    return res


drafting._safe_complete = _instr_safe_complete
drafting.load_world_model = _instr_load
drafting.replay = _instr_replay
say("patched drafting._safe_complete / load_world_model / replay")

banner("STEP 4 -- import + configure the agent")
from agents.templates import code_world_agent as cwa_mod  # noqa: E402

CodeWorldAgent = cwa_mod.CodeWorldAgent
say("imported CodeWorldAgent from", cwa_mod.__file__)

CodeWorldAgent.MAX_ACTIONS = MAX_ACTIONS
CodeWorldAgent.CODER_LLM_CALL_BUDGET = CODER_BUDGET
CodeWorldAgent.ACTION_LLM_CALL_BUDGET = 0  # action head disabled for this diagnostic
CodeWorldAgent.DRAFT_MAX_ATTEMPTS = DRAFT_ATTEMPTS
CodeWorldAgent.REPAIR_MAX_ATTEMPTS = REPAIR_ATTEMPTS
say(
    "overrides: MAX_ACTIONS=%d CODER_BUDGET=%d DRAFT_ATTEMPTS=%d REPAIR_ATTEMPTS=%d ACTION_BUDGET=0"
    % (MAX_ACTIONS, CODER_BUDGET, DRAFT_ATTEMPTS, REPAIR_ATTEMPTS)
)

# Round demarcation: the agent module bound these names at import time, so
# patching drafting.* alone would not catch them.
_orig_draft_fn = cwa_mod.draft_world_model
_orig_repair_fn = cwa_mod.repair_world_model


def _wrap_round(kind, fn):  # type: ignore[no-untyped-def]
    def inner(client, transcript, *a, **kw):  # type: ignore[no-untyped-def]
        banner(f"{kind.upper()} ROUND -- game {transcript.game_id}, transcript len {len(transcript)}")
        t = time.time()
        outcome = fn(client, transcript, *a, **kw)
        installed_is_stub = bool(
            outcome.source and outcome.source.strip() == WORLD_MODEL_SKELETON.strip()
        )
        rec = {
            "kind": kind,
            "game_id": transcript.game_id,
            "transcript_len": len(transcript),
            "seconds": round(time.time() - t, 1),
            "ok": outcome.ok,
            "attempts": outcome.attempts,
            "returned_source_is_exact_skeleton": installed_is_stub,
            "returned_source_chars": len(outcome.source) if outcome.source else 0,
            "rejected_candidate_chars": (
                len(outcome.last_candidate_source) if outcome.last_candidate_source else 0
            ),
            "replay_pass_count": (
                None if outcome.replay_result is None else outcome.replay_result.pass_count
            ),
            "replay_total": (
                None if outcome.replay_result is None else outcome.replay_result.total
            ),
        }
        EVIDENCE["rounds"].append(rec)
        say(f"{kind.upper()} ROUND RESULT for {transcript.game_id}:", json.dumps(rec))
        if not outcome.ok:
            say(
                f"  >>> {kind} FAILED. outcome.world_model is None: {outcome.world_model is None}. "
                "Nothing will be installed (this is the PR#8 behaviour; the OLD code "
                "would have installed WORLD_MODEL_SKELETON here)."
            )
        flush_evidence()
        return outcome

    return inner


cwa_mod.draft_world_model = _wrap_round("draft", _orig_draft_fn)
cwa_mod.repair_world_model = _wrap_round("repair", _orig_repair_fn)

# Wall-clock guard on the game loop itself.
_orig_is_done = CodeWorldAgent.is_done


def _is_done_guarded(self, frames, latest_frame):  # type: ignore[no-untyped-def]
    if time.time() - T0 > RUN_DEADLINE_S:
        say(f"{self.game_id}: RUN DEADLINE reached, stopping this game")
        return True
    return _orig_is_done(self, frames, latest_frame)


CodeWorldAgent.is_done = _is_done_guarded

SELFTEST = os.getenv("DIAG_SELFTEST") == "1"
if SELFTEST:
    # Exercise every instrumentation + reporting path with a synthetic
    # transcript and a fake agent, so this driver can be validated end to
    # end on a box with no GPU and no environment_files. Never enabled in
    # the real diagnostic run.
    banner("SELFTEST -- synthetic transcript through the real draft path")
    from llm_engine.types import Action, GameTranscript, Transition  # noqa: E402

    def _blank():
        return [[[0] * 8 for _ in range(8)]]

    _tr = GameTranscript(game_id="selftest")
    for _i in range(3):
        _a, _b = _blank(), _blank()
        _b[0][_i][_i] = 5
        _tr.append(Transition(_a, Action(name="ACTION1"), _b, 0, 0, "NOT_FINISHED"))

    class _FakeClient:
        def complete(self, system, user, max_tokens=1024):
            return "```python\n" + WORLD_MODEL_SKELETON + "```"

    _outcome = cwa_mod.draft_world_model(_FakeClient(), _tr, max_attempts=2)

    class _FakeAgent:
        game_id = "selftest"
        action_counter = 12
        _init_failed = False
        transcript = _tr
        model_version = 0
        model_source = _outcome.source
        coder_budget = type("B", (), {"calls_used": 1, "call_log": ["draft"]})()

        @property
        def levels_completed(self):
            return 0

        @property
        def state(self):
            return "NOT_FINISHED"

    swarm = type("S", (), {"agents": [_FakeAgent()]})()
else:
    banner("STEP 5 -- discover OFFLINE games")
    from arc_agi import Arcade  # noqa: E402

    arc = Arcade()
    all_games = sorted(e.game_id for e in arc.available_environments)
    say(f"{len(all_games)} environments scanned")
    for g in all_games:
        print("   ", g, flush=True)
    if not all_games:
        say("FATAL: no environments found; nothing to run")
        EVIDENCE["errors"].append("no environments scanned")
        flush_evidence()
        sys.exit(1)

    preferred = [p.strip() for p in os.getenv("DIAG_GAMES", "").split(",") if p.strip()]
    games: list[str] = []
    for pref in preferred:
        games += [g for g in all_games if g.startswith(pref) and g not in games]
    for g in all_games:
        if len(games) >= N_GAMES:
            break
        if g not in games:
            games.append(g)
    games = games[:N_GAMES]
    EVIDENCE["config"]["games"] = games
    say("SELECTED GAMES:", games)

    banner("STEP 6 -- run the agent")
    from agents.swarm import Swarm  # noqa: E402

    swarm = None
    try:
        swarm = Swarm("codeworldagent", "http://localhost:8001", games)
        swarm.main()
        say("swarm.main() returned normally")
    except Exception:  # noqa: BLE001
        say("swarm.main() RAISED:")
        traceback.print_exc()
        EVIDENCE["errors"].append(traceback.format_exc())

banner("STEP 7 -- final per-game state (THE DECISIVE CHECK)")
agents = list(getattr(swarm, "agents", []) or [])
for a in agents:
    src = getattr(a, "model_source", None)
    installed = src is not None
    is_stub = bool(src and src.strip() == WORLD_MODEL_SKELETON.strip())
    try:
        levels = a.levels_completed
    except Exception:  # noqa: BLE001
        levels = None
    try:
        final_state = str(a.state)
    except Exception:  # noqa: BLE001
        final_state = None
    rec = {
        "game_id": a.game_id,
        "actions_taken": getattr(a, "action_counter", None),
        "levels_completed": levels,
        "final_state": final_state,
        "init_failed": getattr(a, "_init_failed", None),
        "transcript_len": len(getattr(a, "transcript", []) or []),
        "world_model_installed": installed,
        "world_model_is_template_stub": is_stub,
        "model_version": getattr(a, "model_version", None),
        "coder_calls_used": getattr(getattr(a, "coder_budget", None), "calls_used", None),
        "coder_call_log": getattr(getattr(a, "coder_budget", None), "call_log", None),
        "model_source": src,
    }
    EVIDENCE["games"].append(rec)
    banner(f"GAME {a.game_id}")
    say("actions taken            :", rec["actions_taken"])
    say("levels completed         :", rec["levels_completed"])
    say("final state              :", rec["final_state"])
    say("coder calls used         :", rec["coder_calls_used"], rec["coder_call_log"])
    say("init_failed              :", rec["init_failed"])
    say("transcript length        :", rec["transcript_len"])
    say("world model INSTALLED    :", installed)
    say("installed model IS STUB  :", is_stub, "  <-- False + installed=True means a REAL drafted model")
    say("model version            :", rec["model_version"])
    if src:
        print("----- INSTALLED WORLD MODEL SOURCE (verbatim) -----", flush=True)
        print(src, flush=True)
        print("----- END INSTALLED WORLD MODEL SOURCE -----", flush=True)
    else:
        say("no world model was installed for this game")

banner("STEP 8 -- verdict summary")
n_llm = len([c for c in EVIDENCE["llm_calls"] if c.get("response_chars")])
n_loaded = len([r for r in EVIDENCE["load_results"] if r["load_ok"]])
n_load_fail = len([r for r in EVIDENCE["load_results"] if not r["load_ok"]])
best = max(
    (r for r in EVIDENCE["replay_results"]),
    key=lambda r: (r["pass_count"] / r["total"]) if r["total"] else 0.0,
    default=None,
)
n_pass = len([r for r in EVIDENCE["replay_results"] if r["passed"]])
say("LLM completions that returned text :", n_llm)
say("candidates that COMPILED+loaded    :", n_loaded)
say("candidates that FAILED to load     :", n_load_fail)
say("candidates that PASSED replay      :", n_pass)
if best is not None:
    say(
        "best replay match                  :",
        f"{best['pass_count']}/{best['total']} on {best['game_id']}",
    )
say(
    "any real (non-stub) model installed :",
    any(g["world_model_installed"] and not g["world_model_is_template_stub"] for g in EVIDENCE["games"]),
)
say("errors captured                    :", len(EVIDENCE["errors"]))
EVIDENCE["summary"] = {
    "llm_completions_with_text": n_llm,
    "candidates_loaded": n_loaded,
    "candidates_load_failed": n_load_fail,
    "candidates_replay_passed": n_pass,
    "best_replay": best,
    "any_real_model_installed": any(
        g["world_model_installed"] and not g["world_model_is_template_stub"] for g in EVIDENCE["games"]
    ),
    "wall_clock_seconds": round(time.time() - T0, 1),
}
flush_evidence()
say("evidence written to", EVIDENCE_PATH)
say("DONE")
