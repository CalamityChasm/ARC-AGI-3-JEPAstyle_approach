# Stage 7 — CodeWorldAgent prototype on the served model

**Kernel:** `calamitychasm/arc3-cwm-prototype` (free; no submission quota)
**Date:** 2026-09-23
**Verdict:** the 2026-09-09 verdict is **overturned**. The model *can* write
replay-passing world models. It still does not convert them into levels.

---

## 1. The claim being re-tested

Commit `0603d60`, merged to `master`, states:

> **VERDICT: the coder model cannot write a replay-passing world model.**
> 25 LLM completions -> 14 compiled -> 0 passed replay. Best match ever:
> 2 of 9 transitions.
> *Recommendation: stop fixing CodeWorldAgent in its current form.*

Three things were different there, and all three mattered:

1. It used **Qwen3-Coder-30B-A3B** over `transformers`, not the
   **Qwen3.8-Flash-Next-NVFP4** we actually serve.
2. The transcript was never reset per level, so clearing one level made
   the replay gate permanently unsatisfiable.
3. **Its prompts were over the context window.** That run recorded
   "prompts growing to 142 KB" as a *throughput* note. At ~47k tokens
   against 32,768, those prompts were being rejected, not answered.

Point 3 means the original verdict was measuring rejection in part, not
only capability.

## 2. What was fixed first

| fix | evidence it was needed |
|---|---|
| per-level transcript reset | boundary steps rewrite 693-1054 of 4096 cells; no rule reproduces a fresh layout |
| reasoning-field fallback in the client | backtest saw `{'reasoning': 46}` with `content` never populated |
| `enable_thinking=False` | with thinking on, 27/27 replies truncated, **zero** contained a `class WorldModel` |
| **compact transcript rendering** | **93 of 109 calls rejected in ~0.1s; prompts to 281,603 chars (~94k tokens)** |

The last one is the decisive fix, and it was **my own known defect**: the
backtest write-up recorded this exact renderer problem as finding #4 on
2026-09-21, and the fix was not carried into the engine the live agent
uses.

## 3. Result — the renderer fix changes everything

Identical kernel, identical games, identical model. Only the prompt
encoding differs.

| | v1 (per-step grids) | v2 (one grid + diffs) |
|---|---:|---:|
| prompt chars, median | 160,900 | **21,446** |
| prompt chars, max | 281,603 | **53,469** |
| prompts over context | **76 / 109** | **0 / 308** |
| calls returning text | 16 / 109 (15%) | **284 / 308 (92%)** |
| candidates loaded | 13 | **227** |
| **replay passes** | **0** | **16** |
| games with a passing model | 0 | **4 of 12** |
| real model installed | **no** | **yes, 3 games** |
| wall clock | 140 s | 1,650 s |

**Longest passing replay: 28 of 28 transitions** (`lp85-305b61c3`). That is
not a lucky one-liner on a 3-step transcript; it is a model that
reproduces 28 consecutive observed steps exactly.

Graded signal across all 227 replays: median prefix **28%**, mean
**40.2%**, and **86 of 227** reproduced at least half the transcript.

Games with at least one passing model: `dc22`, `lp85`, `ls20`, `m0r0`.

## 4. What this does NOT show

**Zero levels were completed, in any of the 12 games.** A correct model of
the first 9-28 transitions did not convert into progress within 120
actions. The capability question is answered; the *usefulness* question is
not, and nothing here says the planner turns a good model into score.

Remaining generation waste is also real: **77 of 304 candidates failed to
load** — 30 missing `predict`/`goal_hint`, 16 unclosed parenthesis, 11
indentation errors. The unclosed-paren and indentation cases are
truncation, so some reply budget is still being lost.

## 5. Honest status of the design

- **Settled:** the served model writes replay-passing world models for
  real 64x64 ARC-AGI-3 games. The merged verdict saying otherwise was
  confounded by context overflow and should not be cited.
- **Not settled:** whether that converts to RHAE. 0 levels in 12 games is
  not encouraging, but 120 actions with a model installed for only 3 of
  them is a weak test of conversion.
- **Not a submission candidate.** With no model installed the agent plays
  random legal actions, which is the ~0.06 floor. 3 of 12 games installing
  a model does not change that arithmetic.

## 6. Next, in order

1. **Raise the install rate.** 4/12 games produce a passing model; the
   binding losses are truncation (still) and missing-method candidates.
2. **Then test conversion**, with a longer action budget on the games that
   *do* install a model — the only place a planner can show anything.
3. Only after both: consider a scored submission.

## 7. Caveats

- One free run, 12 games, 120 actions. No repeats.
- Public-25 games, which the whole community iterates against.
- `enable_thinking=False` throughout; the thinking arm was not re-tested
  after the renderer fix and may now behave differently, since its
  truncation was measured under the oversized prompts.
- Replay here is teacher-forced, as in the backtest: each step is scored
  from its own real `frame_before`, so errors cannot compound. A planner
  needs closed-loop rollout, which is strictly harder.

---

## 8. Coder-model arm (2026-09-25): a coder-specialist is NOT better

Kernel `calamitychasm/arc3-cwm-backtest-coder`, `COMPLETE`, 70 min.
**Qwen3-Coder-30B-A3B-Instruct** via `transformers`, on the same 12
segments, same prompt, same calibration as the Flash-Next arms. Only the
model differs.

### Result: 0 / 12

| arm | passed | reached the sandbox and ran |
|---|---:|---:|
| Flash-Next, thinking, 16k | 1/12 | 0 (27/27 truncated) |
| Flash-Next, no thinking, 8k | 0/12 | **4** |
| **Qwen3-Coder-30B** | **0/12** | **1** |

Attempt-level taxonomy across all 36 attempts:

| outcome | count |
|---|---:|
| `bad_shape` (malformed return triple) | **16** |
| `load_error` | 10 |
| `predict_raised` | 9 |
| `grid_mismatch` (ran, wrong) | 1 |
| `no_code` / `no_response` | **0 / 0** |

### What this actually says

**Compliance is perfect and capability is not.** The coder produced a
`class WorldModel` on every one of 36 attempts -- zero `no_code`, zero
`no_response`, unlike Flash-Next which frequently emitted nothing usable.
The preflight confirmed it too (179-char reply, contained the class).

But **35 of 36 candidates never produced a usable prediction**. The
dominant failure is `bad_shape` (16): `predict()` returns something that
is not a well-formed `(next_state, levels_delta, done)` triple -- the
exact slip `world_model.py` documents, returning one layer instead of the
list of layers. Nine more crashed at predict time.

So on the measure that matters -- candidates that load, run, and produce a
gradeable prediction -- the coder scores **1**, against Flash-Next's **4**.
It writes more code and less *working* code.

### Strategic consequence

The appliance is sealed; we serve Flash-Next and cannot swap it. A coder
win here would have been an awkward result -- a ceiling we could measure
but not act on. **This removes that awkwardness: the model we are allowed
to serve is not the bottleneck relative to a coder-specialist.**

### The one concrete lead

`bad_shape` at 16/36 is a **contract** failure, not a reasoning failure --
the model understands the task and botches the return type. That is the
kind of thing a sharper prompt (an explicit worked example of the exact
return shape, rather than a prose description) could plausibly fix, and it
would apply to *both* models. Untested; recorded as a lead, not a plan.

### Caveats

- One run, 12 segments, no repeats.
- `transformers`, so ~95 s/attempt and a 607 s model load. Slow, but the
  backtest makes only 36 calls so throughput is not a confound here.
- `max_tokens=8192`; responses ran to 29 KB, so truncation is not the
  story this time the way it was for Flash-Next's thinking arm.

---

## 9. Prompt fix (2026-09-25): the contract failure is solved, and a
## capability wall is what was underneath it

Single-variable A/B. Only `SYSTEM_PROMPT` differs -- same model
(Qwen3-Coder-30B-A3B), same 12 segments, same everything else. The old
prompt showed an elliptical skeleton and described the return in prose;
the new one states the contract explicitly and carries a **complete,
runnable** worked example.

### The fix worked, and by a lot

| attempt-level, 36 each | old | new | delta |
|---|---:|---:|---:|
| `bad_shape` | 16 | **0** | **-16** |
| `predict_raised` | 9 | **0** | **-9** |
| `load_error` | 10 | 8 | -2 |
| `grid_mismatch` (loaded, ran, gradeable) | 1 | **28** | **+27** |

**Attempts producing a usable prediction went from 1/36 to 28/36.** The
contract failure is not merely reduced, it is *gone*: zero malformed
triples, zero predict-time crashes. Responses also got shorter (median
10,601 -> 7,938 chars), consistent with less flailing.

That confirms the 2026-09-25 diagnosis exactly: `bad_shape` was a
**contract** failure, not a reasoning one, and a prompt that *shows* the
shape rather than describing it fixes it.

### And the pass rate did not move at all

**Still 0/12.** Best prefix across all 12 segments:

    old  [0,0,0,0,0,0,0,0,0,0,0,0]
    new  [0,0,0,0,0,0,0,0,0,0,0,1]

Eleven of twelve segments are wrong at **step 0**. The one exception
reproduced a single step.

So removing the confound did not reveal a near-miss underneath. It
revealed a **capability wall**: the coder now reliably writes valid,
loadable, runnable world models that are wrong about the rule
immediately and essentially always.

### Coder vs the served model, now on equal footing

| | reached the sandbox | best prefix |
|---|---:|---:|
| Flash-Next, no thinking | 4/12 segments | **9/40** |
| Qwen3-Coder-30B, new prompt | **9/12 segments** | 1/27 |

The coder produces **more than twice as many runnable candidates** and
gets **nowhere near as close to the rule**. Flash-Next reached 9
consecutive correct steps once; the coder never exceeds 1. And the live
prototype on Flash-Next produced **16 replay passes**, which neither
coder run came close to.

**The coder hypothesis is now firmly dead** -- tested twice, the second
time with the interface confound removed, and it lost on the measure that
matters both times.

### What to keep

The prompt change is a real, isolated, large improvement in code validity
and should be kept for **both** models -- Flash-Next's own runs lost
candidates to `load_error` too. It just is not what stands between this
design and a working world model.

`load_error` at 8/36 is the remaining validity loss and was barely touched
by the contract block (10 -> 8), so it is a different problem (syntax /
truncation), not the same one.
