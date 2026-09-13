# Stage 7 — Duck analyzer read timeouts: measured, root-caused, and (honestly) not worth fixing

Date: 2026-09-12. Branch: `stage7-analyzer-timeouts`. Labels follow this repo's
convention: **[VERIFIED]** = read directly from a primary artifact (run
artifacts, bundle source, or a locally-run computation over them);
**[INFERRED]** = reasoning over verified facts, not directly observed.

**No competition submission was made.** Everything here is read-only Kaggle API
calls against an already-completed free run plus two public dataset downloads.
The daily submission slot was untouched.

Reproduce every number with:

```
venv/Scripts/python.exe scripts/analyze_analyzer_timeouts.py <kernel-output-dir>
```

Raw output is committed at `experiments/stage7_analyzer_timeouts_measurements.txt`;
per-game records at `experiments/stage7_analyzer_timeouts_data.json`.

---

## TL;DR

The `analyzer request failed … Read timed out` warnings are **not a bug and not
recoverable lost work.** They are the *mechanism* by which the harness enforces
its per-game time budget: `request_timeout_seconds()` clamps each HTTP read
timeout to the time left in the game's budget, so the one analyzer request that
happens to be in flight when the budget expires always dies. There is exactly
**one per game, in every game, all expiring inside the same 0.98-second window**.

Realistic recoverable upside: **+0.058 points on a public-25 mean of 10.69
(+0.54% relative)**; absolute upper bound **+0.226 (+2.12%)**. Scaled onto our
real hidden-set score of ~2.9 that is roughly **+0.016 to +0.06**, against the
**+0.4** needed to reach the 3.31 top-10% bar.

**Recommendation: do not spend a submission slot, or further engineering, on
this.** The real number this investigation surfaced is the one next to it:
every game got only **~52 analyzer turns** in its 7920 s, at a **median 153 s
per turn**, and **all 25 games hit the wall** — the harness is 100% latency-bound.

---

## 0. A measurement trap worth recording first

The obvious way to do this task — grep the kernel console log — gives two wrong
answers, and this project has been burned by exactly this class of mistake
before (see CLAUDE.md's vLLM `delta.reasoning` gotcha).

**[VERIFIED]** In the structured kernel log
(`/api/v1/kernels/output`, `log` field, 1218 entries):

- The string appears on **50** lines, not 25. Each warning is emitted twice with
  a bare `'\n'` entry between the copies. Counting lines double-counts the loss.
- **All 50 carry a timestamp of 8494.14–8494.55 s**, i.e. *after* the final
  per-game summary block at 8374.98 s. stderr is block-buffered and flushed at
  cell end, so the console log **cannot** place these events chronologically.
  Read literally it says "a burst of 50 failures after the run finished".

Every number below is therefore derived from the run's own primary artifacts —
`transcripts/*.txt`, `artifacts/*_events.jsonl`, `prompts/*.log`, `score.json` —
which carry real per-turn wall-clock timestamps.

Source run: `calamitychasm/arc3-duck-nvfp4-baseline` v2, `COMPLETE`,
public-25 path, 2026-09-10, mean self-eval 10.69.

---

## 1. The measured loss [VERIFIED]

| quantity | value |
|---|---:|
| games | 25 |
| analyzer turns (`ANALYZER STATUS` records) | 1312 |
| … succeeded (`model: …`) | 1287 |
| … failed (`request_error: … Read timed out`) | **25** |
| failure rate | **1.905% of turns** |
| distribution across games | **`{1: 25}`** — exactly one per game, all 25 games |
| total actions | 3633 |
| total levels completed | 48 |

Observed `read timeout=` values (s): n=25, min **3.92**, median **76.89**,
mean **75.71**, max **155.70**, sum **1892.8**.

**Where they cluster: all of them are each game's *last* turn.** The failing
`action` number is, in every single game, exactly `actions + 1` — the turn that
was starting when the budget ran out (e.g. `ar25` 411 actions, fails at 412;
`sp80` 263 actions, fails at 264). Not "early or late in the sequence": *at the
end*, always, by construction.

And `prompts/*.log` shows **`request_index_within_turn: 1`** in all 25 snapshots
— the timeout hit the turn's *first* LLM call, so the doomed turn produced
literally zero output. **[VERIFIED]**

### The single most decisive number

Reconstructing, per game, `issue_time + read_timeout`:

```
wall-clock instant each doomed request expired (s past midnight):
  min=64436.21  max=64437.19  range=0.98s  stdev=0.27s
```

All 25 requests, issued anywhere in the final 156 seconds, **expired within one
second of each other at 17:53:57**. And:

```
first transcript turn across all 25 games: 15:41:57 (spread 0 s)
17:53:57 − 7920 s                        = 15:41:57
```

The deadline is exactly `game start + max_runtime_s_per_game`. Concurrency is
28 ≥ 25 games, so every game started together and every game's budget therefore
expired together. **[VERIFIED]**

---

## 2. The code that sets the timeout [VERIFIED]

Bundle `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` (the mount our NVFP4
stack runs), file `src/ARC3-Inference/inference/framework/solver.py`, **lines
227–244**:

```python
    def request_timeout_seconds(self) -> float | None:
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.timing_payload()["time_remaining_seconds"]
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        return max(0.1, min(candidates))
```

Answer to the question as posed: **(b) and (c) together — it is
`min(analyzer_timeout, per-game remaining wall-clock, global soft-deadline
remaining)`, floored at 0.1 s.**

Call path, all verified in the same bundle:

- `solver.py:300` — every analyzer turn passes it:
  `request_timeout_seconds=self.request_timeout_seconds()`
- `solver.py:219–225` — `timing_payload()` is
  `max(0.0, max_runtime_s_per_game − (time.monotonic() − started_at))`. This is
  the source of the odd precision (`23.048428580999825`).
- `tool_agent.py:1302–1308` — it becomes the `requests.post` timeout verbatim:

  ```python
        def post_chat(request_payload: dict[str, Any]) -> requests.Response:
            return requests.post(
                f"{self._model.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=request_payload,
                timeout=request_timeout_seconds if request_timeout_seconds is not None else self._timeout,
            )
  ```

- `tool_agent.py:1995–1996` — the warning itself, and note `retryable_failure`:

  ```python
            log.warning("analyzer request failed at action %d: %s", display_action_num, exc)
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=captured_reasoning)
  ```

- `solver.py:312–317` — a retryable failure re-runs the *same* analysis step
  after `ANALYZER_RETRY_BACKOFF_SECONDS = 1.0` (`solver.py:58`) — **but the
  `while not self.should_stop()` loop and the explicit `if self.should_stop():
  break` mean the retry never happens once the runtime limit is reached.** The
  game simply ends.

**Which of the three candidates actually binds here** [VERIFIED from our
notebook, `kaggle_submission_duck_nvfp4/notebook/duck-qwen3-8-anim-base.ipynb`]:

- cell 14: `bm.solver.max_runtime_s_per_game = 7920.0`, `analyzer_timeout = 900.0`
- cell 16: `soft_end = NOTEBOOK_START_EPOCH + (32400 − 600)` = ~31800 s ≈ 00:22
  next day — far beyond the observed 17:53:57.

So the binding term is **`max_runtime_s_per_game = 7920 s`**, not
`analyzer_timeout=900` and not the soft end. This confirms the orchestrator's
own reading that the 900 s figure in `PUBLIC25_SETTINGS` is not what these
requests used: 900 only binds in the first 7020 s of a game, during which no
request ever came close to it.

**This is upstream Tufa code, not something the NVFP4 fork introduced.**
`request_timeout_seconds()` is byte-equivalent in the older Duck bundle
`jakobbrggen/taaf-kaggle-source-anim-20260807-anim` at
`src/ARC3-Inference/inference/framework/solver.py:267–284`. Our FP8 stack
(2.57) behaves identically. **[VERIFIED]**

---

## 3. Bug or symptom? — **Symptom, and a deliberate one**

**Verdict: symptom.** Four independent lines of evidence, not one:

1. **The timeout value is by construction "time left in this game".** Code
   quoted above. It is not a fixed value that happens to be too small.
2. **All 25 expire in a 0.98 s window** at exactly `start + 7920 s`. A
   miscomputed or too-short timeout would produce failures scattered through
   the run; a shared deadline produces exactly this.
3. **Exactly one per game, in every game — and it is always the turn after the
   last recorded action.** Every game ran to its budget wall; none finished
   early. The timeout is the *last event of the game*, not an interruption
   inside it.
4. **The abandoned turn had produced nothing to lose**
   (`request_index_within_turn: 1`, so no tool calls had run yet), and the game
   was going to stop one line later regardless — `should_stop()` →
   `runtime_limit_reached()` was already true at the same instant.

### Would the server have answered shortly?

This is the "recoverable lost work" branch of the question, and the answer is
**no, not within the budget** — measured, not assumed:

```
turn duration, pooled over all 1287 successful turns (s):
  n=1287  min=25  p10=98  median=153  p90=205  max=294
mean leftover at the deadline = 75.7 s
  -> only 4.7% of turns ever completed that fast
```

Per game, comparing the leftover against that game's *own fastest turn ever*:

- **In 9 of 25 games the leftover was strictly below the game's own minimum
  turn duration** (`ar25` 59.7 s vs min 61 s; `ft09` 10.0 vs 60; `r11l` 27.4 vs
  72; `sp80` 3.9 vs 60; also `ka59`, `ls20`, `m0r0`, `sc25`, `sk48`).
  Recovery there was **provably impossible** — no turn that game ever ran could
  have fit.
- Only `lp85` (44%) and `wa30` (31%) had a leftover long enough that a
  meaningful fraction of their turns would have fitted.

**[INFERRED, flagged as unresolved]** `vllm:request_success_total{finished_reason="abort"}`
reads `0.0` in `vllm-metrics-final.prom`, which *could* mean the server ran
those 25 generations to completion after the client hung up (wasted GPU work).
It could equally mean the metrics were scraped before those requests resolved —
the server shut down at 17:53:58, one second later. This run cannot distinguish
the two, and in a single-wave public-25 run it does not matter (every game ends
at the same instant, so nobody is starved). It *could* matter at a wave boundary
in a real 110-game rerun, where up to 28 doomed generations would overlap the
next wave's opening turns. **Untested; noted, not claimed.**

---

## 4. What recovering it would actually be worth, under RHAE

RHAE per environment is `E_e = min(completion_term, efficiency_term)` with
`S_l = min(1.15, h_l/a_l)^2` and `w_l = l`. **More actions is emphatically not
linearly more score:**

- Actions spent on a level that never completes contribute **nothing** — no
  `S_l` term exists for an unsolved level. So the marginal action is worth
  **zero** unless it happens to be the one that finishes a level.
- Where the **efficiency** term is the binding `min`, an extra completed level
  **cannot raise `E_e` at all**. The run's own scorer lets us see which term
  binds: where reported score equals the completion term exactly, completion
  binds. **In 9 of 25 games efficiency binds** (`bp35`, `dc22`, `ft09`, `ls20`,
  `sc25`, `sp80`, `tu93`, `vc33`, `wa30`) — and `wa30`, the game with the
  *second* largest leftover and the highest P(fit) after `lp85`, is one of them,
  so its recoverable actions are worth literally nothing.

Two estimates, both computed per game and then averaged over the 25:

| estimate | assumption | recoverable actions | recoverable levels | Δ mean public-25 score |
|---|---|---:|---:|---:|
| **realistic** | a turn only lands if that game's turn-duration distribution says it fits in its leftover | **5.5** / 3633 (0.15%) | **0.11** / 48 | **+0.058** on 10.69 (**+0.54%**) |
| **absolute upper bound** | every leftover second converts to actions at that game's own average rate (ignores that a turn cannot complete in <25 s) | 33.9 (0.93%) | 0.45 | **+0.226** on 10.69 (**+2.12%**) |

**[INFERRED]** Our public-25-to-hidden-set ratio has been ~0.27 (10.69 local →
2.84/2.95 real). Applying the *relative* deltas to our real score of ~2.9 gives
**+0.016 (realistic) to +0.062 (upper bound)**. The top-10% bar is 3.31 and
rising ~0.1/day; we need ~+0.4. **This closes between 4% and 15% of a single
day's drift in the bar.**

---

## 5. Recommendation

**Do not pursue this, and do not spend a submission slot on it.** There is no
minimal change worth proposing: the timeout is the budget-enforcement mechanism,
the abandoned work is a first request with zero partial output, the game ends at
the same instant either way, and in 9 of 25 games the leftover was shorter than
anything that game could ever have completed. Raising the timeout would only let
a request outlive the budget and return an answer the harness must discard.

The one change that is *defensible* is also almost worthless: skip starting an
analyzer turn when `request_timeout_seconds()` is below some quantile of that
game's own observed turn durations, and spend the tail on a zero-LLM heuristic
action instead. Ceiling **+2.12% relative**, realistic **+0.54%**, plus it adds
a new failure mode (a bad heuristic action at the buzzer can lose a level as
easily as gain one). **Not recommended.**

### Where the headroom this investigation actually found is

The same artifacts say something much louder than the timeouts do, and it lines
up exactly with Tufa's own "cost is mostly dictated by the harness":

- **All 25 games hit the 7920 s wall. Zero finished early.** The run is 100%
  time-bound — solvability was never the limiting factor for *any* game.
- **Each game completed only ~52 analyzer turns** (51–54, remarkably uniform)
  in 2 h 12 m — a **median 153 s per turn**, with even the fastest decile at
  **98 s**.
- Those 52 turns produced 42–411 actions depending on the game (mean 2.77
  actions/turn), and 48 levels across 25 games.

**[INFERRED]** Turn latency, not timeouts, is the whole ballgame: halving the
153 s median would roughly double the number of decisions every game gets, which
is two orders of magnitude more leverage than the ≤2% here. The obvious suspect
is queueing — 25–28 concurrent game threads against
`TAAF_VLLM_MAX_NUM_SEQS = 8` is a 3.1–3.5× oversubscription, so most of a turn's
153 s is plausibly wait, not generation. That is measurable for free (the
`/metrics` scrape the bundle already writes carries queue-time histograms) and
is the recommended next target.

### Free validation, if anyone wants to re-check this

Everything above is reproducible at zero cost and zero quota:
`scripts/analyze_analyzer_timeouts.py` runs against any `kaggle kernels output`
directory of a Duck run. Pointing it at the next free public-25 run is the whole
validation — if the pattern ever deviates from "exactly one timeout per game,
all expiring in the same second", *that* would be a real bug and worth
re-opening this.

---

## Appendix — per-game detail

See `experiments/stage7_analyzer_timeouts_measurements.txt` for the full tables
(per-game turns/actions/levels/game-overs, failing action, leftover, minimum
turn duration, P(fit), and the marginal RHAE value of one more level).
