# Stage 7 — CodeWorldModel backtest: can this model write a world model that replays?

**Branch:** `stage7-codeworld-backtest`
**Status:** instrument **built and calibrated**; the real measurement has **not** been run.
**Cost so far:** zero GPU, zero submission quota.

---

## 1. Why this exists

`CodeWorldAgent` (`kaggle_submission_llm_world_engine/`) is the one genuinely
original asset in this repo aimed at published SOTA — an LLM writes the Python
source of a world model, an LLM-free beam search plans against it
(arXiv:2605.05138 reports 58.12% mean RHAE with that mechanism). It scored
**0.00**.

Its diagnosed defects were fixed on `stage7-codeworld-fixes`, except one, recorded
there as *"still binding"*: throughput of 0.02–0.27 actions/sec. But that figure was
measured under `LLM_BACKEND=transformers`, which cannot multiplex games; GitHub
issue #3 already names the fix as the in-kernel vLLM path we operate daily. So
throughput is a backend config change, not a property of the design.

Which leaves the question nobody has answered, and the one the write-up itself
flags as **untested**:

> Whether the coder model can write a replay-passing model for a real 64x64 game
> is untested — the local RTX 2070 cannot host it.

Everything about `CodeWorldAgent` is downstream of that. If the model cannot
produce a world model that reproduces observed play, the planner, the throughput
work and the RHAE upside are all moot. This arm builds the instrument that
settles it.

**It is a counted measurement, not a scored one.** A segment either yields a
replay-passing model or it does not. That puts it entirely outside the ±2.46 SE
that makes a single free public-25 run unable to rank anything — the constraint
that invalidated a "seven interventions, seven regressions" conclusion earlier in
this project.

## 2. What is measured

> Given the first N observed steps of one real level, can the served model write a
> `WorldModel` whose `predict()` reproduces all N exactly?

Data is real recorded play: the 25-game anim free run of 2026-09-21, our current
champion configuration. No simulation, no synthetic games.

**Replay is teacher-forced.** `llm_engine.replay` scores each step from that
step's own real `frame_before`, never from the model's previous prediction, so
errors cannot compound. This is strictly *easier* than the closed-loop rollout a
beam search actually needs. That is deliberate: it makes the test a floor. A
failure here is decisive; a pass is necessary, not sufficient.

## 3. Two structural findings, both from reading the code and data

### 3.1 The agent's own replay gate is unsatisfiable after any level-up

`code_world_agent.py` keeps **one transcript per game**, appends every transition
to it (`_handle_new_transition`), and never segments or truncates it.
`draft_world_model` requires a candidate to reproduce **all** of it.

A level-clearing transition's `frame_after` is the **next level's opening
layout**. Measured on the anim run: boundary steps rewrite **693–1,054 of 4,096
cells**, against a median of **109** for an ordinary move. No inferred rule
produces a fresh layout, and the drafting prompt explicitly forbids hardcoding
grids from the examples.

The only way to satisfy such a step is to emit that grid as a literal — and it
does not fit. One 64x64 board as a Python literal is **12,720 characters, ~4,240
to 6,360 tokens**, against `DRAFT_MAX_TOKENS = 4096`. A single boundary step
therefore costs **1.0–1.6x the entire response budget**, before any class
structure, for one step.

**So once a game clears one level, no candidate that fits the response budget can
pass replay again**, and the only candidate that could in principle is one doing
exactly what the prompt forbids.
Drafting burns all 5 attempts, returns `ok=False`, and the agent plays on with no
world model — permanently, for exactly the games it was doing well on. This is a
plausible contributor to the 0.00 that is independent of every defect already
diagnosed.

This backtest therefore segments **per level** and excludes the boundary step,
matching the engine's own stated contract ("a Python simulator for one level").
Fixing the agent to do the same is a separate change and is **not** made here.

### 3.2 Some games are not a function of the visible board

3 of 61 windowed segments (**5%**, 2 games) contain two steps with a
**byte-identical** `frame_before`, an identical action, and **different
outcomes**. For those, no pure `predict(state, action)` exists at all.

The affected games are animation-driven, which is consistent with the anim solver
bundle — the graft worth +20% — being specifically animation-aware. The engine's
contract does permit small hidden counters on `self`, so these are not strictly
unmodelable, but inferring invisible state from the visible board is categorically
harder than inferring a rule.

This is a **hard ceiling on the pass rate that has nothing to do with the model**,
so it is measured up front, for free, and reported alongside every result.

## 4. A third finding: the engine's prompt renderer cannot fit

`llm_engine.drafting._render_transcript` emits `format_grid(t.frame_before)` for
**every** transition it shows, up to 40. A 64x64 grid is 64 lines of 64
characters, and consecutive `frame_before` grids differ only by the previous
step's diff — which is already printed beside them.

Measured on the real run at a matched 40-step window:

| renderer | total (61 segments) | largest single prompt |
|---|---:|---:|
| `llm_engine.drafting` | 8,403,519 chars | 232,139 chars (~58k tokens) |
| this arm's `render.py` | 1,832,996 chars | 80,258 chars (~20k tokens) |

**4.6x smaller**, median ~5.9k tokens. Running the backtest through the engine's
renderer would have measured context overflow and reported it as a capability
result.

## 5. Calibration — the part that makes a result readable

Both bounds are established **before** any GPU time, on the same 61 real segments
the measurement will use.

| bound | value | meaning |
|---|---:|---|
| **null floor** — do-nothing model (`WORLD_MODEL_SKELETON`) | **0 / 61** | no segment is passable by predicting "nothing changes"; every pass is real |
| **proven ceiling** — generated lookup-table oracle | **58 / 61 (95.1%)** | the harness demonstrably *can* report a pass on real 64x64 data |

The 3 segments the oracle cannot pass are exactly the 3 with a proven
board/action contradiction. Nothing is unexplained.

The null floor matters because this repo learned the hard way (all of Stage 1)
that a metric dominated by trivially-unchanged content is beatable by predicting
no change. Here the median identity prefix is **0%** of steps, mean 0.4%, max 14%.

The oracle matters because **a 0% result is uninterpretable without it** — "the
model never produced a passing world model" and "the harness cannot recognise
one" would look identical. This project has already produced one complete,
plausible-looking, entirely empty results table from exactly that ambiguity (the
`delta.reasoning` gotcha). The oracle is a memorised lookup table, is labelled as
such, is never shown to a model, and is evidence about the *instrument* only —
never that the task is learnable.

## 6. Pre-registered metrics and decision rule

Registered **before** the run, and registered on an **outcome**, not only a
mechanism — the sharper lesson from the AVO arm, which passed its mechanism
falsifier and lost anyway.

**Primary (counted):** pass rate on *informative* segments — those the do-nothing
model does not already pass. Currently 61 of 61.

**Secondary (graded):** median fraction of steps reproduced before the first
mismatch. A model getting 80% of a level's dynamics is a different situation from
one getting 2%, and both read as "failed".

**Diagnostic:** failure taxonomy — `no_code` (no `class WorldModel` produced at
all), `load_error`, `predict_raised`, `bad_shape`, `grid_mismatch`,
`signal_mismatch`.

| informative pass rate | verdict |
|---|---|
| **< 10%** | **#4 is dead.** The model cannot write world models for real games. Do not build the planner, do not do the vLLM backend work. |
| **10–40%** | Partial. Worth pursuing only on the subset of games that pass, and only with the repair loop. Not a general mechanism. |
| **> 40%** | **#4 is on.** The backend swap and planner integration become worth their cost. |

**The graded measure can overturn a binary reading, and that is pre-registered
too:** if pass rate is near zero *but* median prefix exceeds 50%, the conclusion
is not "the model cannot do this" but "exact full-transcript replay is the wrong
gate" — which would point at a tolerance-based or prefix-scored acceptance rule
as the next arm, not at abandoning #4. Recording this now so it cannot be decided
after seeing the number.

## 7. What was built

| file | role |
|---|---|
| `arc3_cwm/extract.py` | event log → per-level `Transition` segments; all skips counted |
| `arc3_cwm/determinism.py` | exact board/action contradiction census; the data-imposed ceiling |
| `arc3_cwm/render.py` | compact prompt (opening grid once + per-step diffs) |
| `arc3_cwm/harness.py` | draft/replay/repair loop, failure taxonomy, identity baseline |
| `arc3_cwm/oracle.py` | generated guaranteed-passing model — the positive control |
| `arc3_cwm/report.py` | counted aggregation, informative-vs-free-pass separation |
| `arc3_cwm/_engine.py` | import shim; the engine is used, never forked or copied |
| `scripts/run_cwm_backtest.py` | CLI, including `--self-check` (no LLM contacted) |
| `tests/test_cwm_backtest.py` | 36 tests, weighted toward the extractor |

Nothing in `llm_engine/` or `code_world_agent.py` is modified. This is an
instrument pointed at the engine.

**Tests: 36 new, 183 total, all passing.** The weighting toward the data pipeline
is deliberate: every prior data bug in this project was silent. The
`action_input` bug made every recorded action look like RESET and invalidated
four stages of results; a stale checkpoint sat in `checkpoints/` for weeks. A
backtest with a subtly wrong extractor produces a confident, plausible,
meaningless number — worse than no number. The ACTION6 row/col → (x, y)
mapping is asserted against the real recorded string format, because transposing
it would move every click and would be invisible in any aggregate.

## 8. How to run it

Free, no LLM, proves the instrument on the data about to be measured:

```bash
python scripts/run_cwm_backtest.py --artifacts <run>/artifacts --self-check
```

The real measurement needs the served model. Local is impossible (RTX 2070, 8 GB);
this runs on Kaggle against the in-kernel vLLM server the Duck/anim stack already
starts — `LLM_BACKEND=openai` plus `CODER_LLM_BASE_URL`, a config change, no code
change:

```bash
python scripts/run_cwm_backtest.py --artifacts <run>/artifacts \
    --max-segments 10 --out results.json
```

Start with `--max-segments 10` as a pilot. At ~3 attempts per segment this is
tens of model calls, not a full game run — it is cheap next to a 9-hour free run,
and it consumes **no submission quota**.

## 9. Result

*Deliberately left open. To be filled only from a real run.*

## 10. Caveats

- Teacher-forced replay is easier than the closed-loop rollout a planner needs.
  A pass here does not establish planning viability.
- One free run (25 games, 61 segments). Segment counts per game are uneven.
- The 40-step window is a choice, not a measurement. Longer windows are harder
  and were not tested.
- The `changed_fraction` range across segments is 0.56–1.00; segments are not
  equally difficult and the pass rate is unweighted.
- The two structural findings in §3 are read from code and data, not from a
  controlled experiment. §3.1 in particular predicts a failure mode in the live
  agent that has not been observed in a scored run — the 0.00 run predates the
  fixes and its logs are gone.
