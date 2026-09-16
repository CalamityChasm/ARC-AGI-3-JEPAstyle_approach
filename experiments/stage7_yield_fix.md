# Stage 7 — `LOCAL_ANALYZER_YIELD_SECONDS` 60 → 180, tested alone

Date: 2026-09-15. Branch: `stage7-yield-fix`. Kernel:
`calamitychasm/arc3-duck-nvfp4-yield180` v1 (free run, **no competition
submission was made by this work**).

Labels follow the repo convention. **[VERIFIED]** = read directly from a primary
artifact (the bundle source we actually run, or one of our own run's output
files, or a local computation over them). **[INFERRED]** = reasoning over
verified facts.

This implements shortlist item **#2** of `experiments/stage7_sota_research.md`
§6, isolated to a single variable. Baseline to beat:
`calamitychasm/arc3-duck-nvfp4-baseline` v2 — public-25 **10.69**, **3,633**
actions, **46.0%** no-op turns.

Reproduce with:

```
venv/Scripts/python.exe scripts/_build_duck_nvfp4_yield.py
venv/Scripts/python.exe -m pytest tests/test_duck_nvfp4_yield_patch.py -q
venv/Scripts/python.exe scripts/analyze_yield_effect.py \
    baseline=<baseline-out> yield180=<yield180-out> \
    --json experiments/stage7_yield_fix.json
venv/Scripts/python.exe scripts/parse_finished_lines.py \
    <baseline-out> <yield180-out> --per-game
```

---

## 1. The constant, and exactly how it gates the loop [VERIFIED]

All line numbers are in the bundle this kernel mounts and runs,
`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`, read from the downloaded
dataset — not from a fork or a paraphrase.

| what | where |
|---|---|
| the value is persisted | `serving_setup.py:2940` — `"LOCAL_ANALYZER_YIELD_SECONDS": "60"` inside `persist_analyzer_environment()` |
| it is bound, once, at **module import** | `src/ARC3-Inference/inference/agent/tool_agent.py:143` — `_LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)` |
| each agent copies it at construction | `tool_agent.py:934` — `self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(...)` |
| it becomes the stop predicate | `tool_agent.py:1770-1778` — `control_yield_reason()` returns `"turn_time_budget"` once `(time.monotonic() - turn_started_at) >= self._yield_seconds` |
| the predicate is checked **before every model call** | `tool_agent.py:1782-1786` — top of `while self._tool_steps is None or turn_count < self._tool_steps` |
| …and **again after every non-acting tool dispatch** | `tool_agent.py:1968-1972` — inside the `for tool_index, tool_call in ...` loop, right after `if dispatch.step_executed: break` |

The loop, reduced to its control flow (`tool_agent.py:1782-1978`):

```python
while self._tool_steps is None or turn_count < self._tool_steps:
    if control_yield_reason() is not None:
        break                          # (A) budget already gone -> turn over
    turn_count += 1
    ...                                # one LLM request, ~146 s
    for tool_index, tool_call in enumerate(tool_calls):
        dispatch = self._dispatch_tool(state_path, tool_name, arguments)
        if dispatch.step_executed:
            step_executed = True
            break                      # (B) acted -> turn over, productively
        yielded_control_reason = control_yield_reason()
        if yielded_control_reason is not None:
            break                      # (C) inspected, budget gone -> turn wasted
```

Two things make the 60 s value binding rather than advisory:

1. **The step count is not the gate — time is.** `serving_setup.py:2938`
   persists `LOCAL_ANALYZER_TOOL_STEPS = "0"`, and `tool_agent.py:932` maps that
   to `self._tool_steps = None`, i.e. the `while` condition is permanently true.
   (`stage7_sota_research.md` §1.3 describes a 12-step allowance; the persisted
   value is 0/unbounded. The conclusion is unchanged and in fact strengthened —
   nothing but the clock ever ends this loop.)
2. **One round-trip already exceeds the budget.** The baseline's own server
   counters give mean end-to-end request latency
   `195,093.28 s / 1,339 requests = 145.7 s` [VERIFIED,
   `vllm-metrics-final.prom`], against a 60 s budget. So branch (A) is reached
   on the *second* iteration of every turn without exception: exactly one model
   call per turn, and if that call only inspected, branch (C) fires and the turn
   produces nothing.

That is the whole mechanism behind the headline pathology. In the baseline,
**every single one of the 592 no-op turns carries the same stop reason**, and
nothing else appears [VERIFIED, all 25 transcripts]:

```
  695x  Step executed.
  592x  Yielded control to solver: turn_time_budget.
```

At 180 s a second call fits (145.7 < 180) and a third does not (291.4 > 180),
so a turn can afford to inspect *and then act* with the tool result already in
context.

### 1.1 The change

Two lines of value, in a guarded block appended to the setup cell (cell 10) of
the baseline notebook:

```python
_YIELD_KNOB = {"LOCAL_ANALYZER_YIELD_SECONDS": "180"}
...
_persisted.update(_YIELD_KNOB)
SETUP_ENV_PATH.write_text(json.dumps(_persisted, indent=2, sort_keys=True) + "\n")
os.environ.update(_YIELD_KNOB)
```

**Why the tail of cell 10 specifically.** The constant is bound at *import*
time (`tool_agent.py:143`), and the first import of the solver is one cell
later: cell 12 unpickles `benchmark_initial.pkl`, whose
`inference.framework.solver` does `from inference.agent.tool_agent import
ToolAgent` at line 37. Cell 10 is the last point that is both *after*
`serving_setup.py` has persisted the analyzer environment and *before* that
import, so the module global binds 180.0 naturally rather than by a post-hoc
reassignment. Verified in the notebook: no cell before 12 imports `taaf` or
`inference` at all [VERIFIED].

The solver constructs one `ToolAgent` per game
(`solver.py:1180-1204`, `_make_analyzer`) on a `ThreadPoolExecutor` in this same
process, and `__init__` re-reads the module global each time, so one patch
covers all 25 games. The analyzer is not re-exec'd in a subprocess; the only
`subprocess.run` calls in `solver.py` (lines 951, 1008) are `make server` /
`make stop-server`, which this NVFP4 stack does not use.

**Guards** (pattern from `sahasawatt/thui-rs-v0` cell 9, *minus* their
`LOCAL_ANALYZER_SEED = 20260825`, which would have been a second variable):
the block asserts the persisted value is still `"60"`, that eleven other
analyzer knobs are exactly what the baseline had, that no seed exists, that
nothing is imported yet, and finally that
`_tool_agent._LOCAL_ANALYZER_YIELD_SECONDS == 180.0`. Any upstream drift fails
the run loudly instead of silently measuring something else.

**Single-variable enforcement.** `scripts/_build_duck_nvfp4_yield.py` asserts
the baseline notebook's md5 (`57ffcd51…`) and that exactly cells `[1, 10]`
differ — cell 1 being the fork-header markdown, cell 10 by append only.
`tests/test_duck_nvfp4_yield_patch.py` executes the override block itself
against a transcribed stand-in for the bundle's import-time binding: 8 tests,
including that it changes exactly one persisted key, introduces no seed, and
raises if the solver was already imported. Full suite 142 passed.

---

## 2. What this lever can and cannot buy — pre-registered [INFERRED]

Committed before the run resolved (`scripts/analyze_yield_effect.py:
breakeven_model`, commit `f6fd31f`), so it is a prediction, not a
post-hoc story.

`stage7_sota_research.md` §1.5 bounds removing the no-ops at **+85% productive
turns**. **That bound is not reachable by this lever**, and the reason is in the
same document's §1.2: calls per game is pinned near 54 by wall clock ÷ latency,
*not* by the yield budget. A second call inside a turn is therefore not free —
it consumes a call that would otherwise have started a fresh turn.

With `p` = P(first call of a turn acts) = 695/1339 = **0.519** measured on the
baseline, and `q` = P(second call acts | first did not) the unknown this run
measures, at a fixed call budget `C`:

```
calls per turn = 2 - p          turns = C / (2 - p)
executed turns = turns * (p + (1-p) * q)
```

| q | turns | executed | vs baseline |
|---:|---:|---:|---:|
| 0.00 | 904 | 469 | −32% |
| 0.25 | 904 | 578 | −17% |
| **0.50** | 904 | **687** | **−1%** (break-even) |
| 0.75 | 904 | 795 | +14% |
| 1.00 | 904 | 904 | +30% (ceiling) |

So the honest ceiling for item #2 is **+30%**, not +85%, and it goes *negative*
below `q ≈ 0.5`. The one prior datapoint at yield 180 — the anim graft, 1,358
calls / 922 turns / 586 executed — implies **`q ≈ 0.23`**, well under break-even.
That run swapped the solver at the same time, which is exactly why this
single-variable test is worth a free push either way.

### 2.1 The model is validated against an independent run [VERIFIED]

Before this run resolved, the same model was checked against the anim graft —
the only other run in this project at `yield_seconds = 180`, and the one whose
solver swap made it uninterpretable on its own:

| | model prediction | anim, measured |
|---|---:|---:|
| LLM calls per turn | 1.48 | **1.47** |
| analyzer turns (at its 1,358 calls) | 917 | **922** |
| executed turns at the realised `q` | 575 | **586** |

Its realised `q` is **0.24** — within rounding of the 0.23 inferred from the
research doc's summary table, and far under the 0.50 break-even. Mean e2e
latency barely moved (145.7 s → 144.2 s), confirming the model's fixed-call-
budget assumption: the knob regroups calls, it does not buy any.

So the model reproduces an independent run to within 2% on all three
quantities. That makes it a real predictor rather than an arithmetic identity,
and it predicts this single-variable run lands **negative** on productive turns.

---

## 3. Method — and why the score is not the gate

The public-25 mean has misled this project repeatedly: the local→real ratio is
about 0.27, and the real-score sd on a fixed config is 0.295 (n=4, mean 2.83).
A change that moves productive turns by tens of percent can sit entirely inside
that noise. So the primary read-out is the mechanism, measured from the run's
own artifacts, with score and actions reported alongside for comparability.

`scripts/analyze_yield_effect.py` reads four artifacts per run: `transcripts/*.txt`
(one `[ANALYZER STATUS]` block per analyzer turn, each ending
`step_executed: True|False` plus a `message:`, and stating the `yield_seconds`
actually in force), `vllm-metrics-final.prom` (the server's own request count,
which is what separates "per turn" from "per LLM call" once a turn can hold
two), `benchmark.json` (per-action `generated_tokens`), and
`artifacts/*_events.jsonl` (per-action `level` stamps).

**Validation.** Every published baseline figure reproduces exactly: 46.0% no-op,
1,287 completed turns (695 executed + 592 yielded, plus 25 request-error blocks
= the 25 documented per-game read timeouts), 53.6 calls/game, 2.71 actions/call,
145.3 actions/game, 3,633 actions, 48/183 levels, 10.69 mean. Two independent
artifacts also agree exactly on the productive-call count: `benchmark.json` has
**695** actions with non-zero `generated_tokens`, and the transcripts have
**695** `step_executed: True` blocks.

**The progressive-summary trap, avoided.** These logs emit a running summary
block roughly every ~10 games, so grepping "the summary" yields a partial-run
snapshot. `scripts/parse_finished_lines.py` instead parses the **per-game
`[finished]` lines** — exactly one per game, written when that game ends:

```
[finished] m0r0-492f87ba state=gave_up level=1/6 score=4.76 actions=88
           tokens=78191 per-level=29/30,59/111,0/203,0/26,0/500,0/237
```

`per-level` is ours/human for each level in order, so the never-completed level
is exactly the (L+1)-th entry. This is both an independent recomputation of
`score.json`/`benchmark.json` from a different artifact and the cleanest source
for the wasted-action split. On the baseline it returns public-25 **10.69**,
**3,633** actions, **48/183** levels, human baseline **17,135** (matching
arXiv:2607.15439), all 25 `gave_up`, and wasted actions **1,825/3,633 =
50.2%** — the brief's figure exactly, where the events-stream approximation
gives 51.4% because it also counts animation frames.

---

<!-- RESULTS -->

---

## 6. Recommendation

<!-- PENDING -->
