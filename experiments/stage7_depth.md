# Stage 7 — Where the agent loses depth, and two mechanisms against it

Date: 2026-09-19. Branch: `stage7-depth`. Kernels
`calamitychasm/arc3-duck-nvfp4-anim-cf` and `-nt` (free pushes, no submission).

**No competition submission was made by this investigation, and none will be.**
Everything here is read-only analysis of completed runs, plus two free
`kaggle kernels push`es. The daily slot was untouched.

Labels follow this repo's convention. **[VERIFIED]** = read directly from a
primary artifact (a run's own `benchmark.json`, `artifacts/*_events.jsonl`,
`transcripts/*.txt`, bundle source, or a locally-run computation over them).
**[INFERRED]** = reasoning over verified facts, not directly observed.

Reproduce every number here with:

```
venv/Scripts/python.exe scripts/analyze_depth.py      anim=<dir> wg=<dir> rs=<dir> \
    --json experiments/stage7_depth_data.json
venv/Scripts/python.exe scripts/analyze_dead_turns.py anim=<dir> wg=<dir> rs=<dir> \
    --json experiments/stage7_depth_dead_turns.json
venv/Scripts/python.exe scripts/read_duck_public25_log.py <dir>
venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim_depth.py --check
venv/Scripts/python.exe -m pytest tests/test_commit_floor.py tests/test_no_thinking.py -q
```

`<dir>` is an unpacked `kaggle kernels output` directory for
`arc3-duck-nvfp4-anim` (the incumbent), `-anim-wg` and `-anim-rs` (its two
sibling free runs, used here as independent replicates of the measurement).

---

## 0. TL;DR

1. **The agent does not mainly lose depth by dying, by wasting actions, or by
   being starved of actions. It loses depth by not taking turns.** 46.1% of the
   run's entire wall clock — 91,295 of 198,233 seconds — goes on `analyze()`
   turns that execute **no game action at all**, and a dead turn is the
   *expensive* kind (median 258 s against 173 s for one that acts). Replicated
   at 48.6% and 49.8% on the two sibling runs, so this is structural, not noise.
   [VERIFIED, §2]
2. **Deaths are not the lever.** 16 in-level game overs, 520 actions (19.9%) —
   but they sit almost entirely in games whose next level is worth little under
   `w_l = l`. Every death-affected game gaining a level is worth **+2.12**
   public-25, against **+5.30** for the ten games already two or more levels
   deep. [VERIFIED, §3]
3. **The deep games ran out of clock, not out of ideas.** `re86` had cleared 4
   levels and spent **1 action** on level 5 when time ran out; `tu93` 0; `ar25`
   9 of a 89-action human baseline; `ft09` — the run's best game at 47.62 — 21
   of 65. These are the games where one more level is worth +11 to +24 each.
   [VERIFIED, §4]
4. **The two losses are the same loss.** The games holding the largest terminal
   dead streaks *are* the deep games: `r11l` spent its last 16 turns and 4,703 s
   (59% of its clock) on a single analysis step it never acted on, `cn04` 18
   turns and 4,750 s (60%), `ft09` 1,771 s (22%). [VERIFIED, §5]
5. Built and free-ran two single-variable arms on the anim graft: a **commit
   floor** (§7.1) and a **no-thinking** arm (§7.2). §8 has their telemetry.

---

## 1. Method, and one correction carried forward

Three sources, in descending authority:

* `benchmark.json` — the harness's own per-level accounting
  (`actions_per_level`, `base_actions_per_level`, `final_score`, and a per-action
  `history` with cumulative wall clock). This is what produces `score.json`.
* `artifacts/*_events.jsonl` — per-action `game_over` / `level_completed` /
  `level` flags.
* `transcripts/*_p0.txt` — one `--- analysis_step=N | action=M | HH:MM:SS ---`
  header and one `[ANALYZER STATUS]` block per `analyze()` turn, the block
  reporting `step_executed` and the yield reason.

**The event stream mirrors every action.** `scripts/analyze_depth.py` counts
only `type == "action"` rows. Without that filter the anim run reads 3,605 rows
against 2,615 real actions, and every game over twice [VERIFIED]:

```
anim rows by type : initial 25 | action 2,615 | analysis 940 | experiment 25
GAME_OVER by type : action 16  | analysis 16
benchmark.json    : 2,615 actions
```

This is the same defect that produced the 4,970-against-3,633 figure on the
nvfp4 baseline. `load_run()` asserts `sum(actions_per_level) == len(history) ==
len(action rows)` for all 25 games before computing anything, on all three runs.

**A free cross-check fell out of it**: the `type == "analysis"` row count (940 /
921 / 929) equals the transcript header count exactly, on all three runs. The
turn counts in §2 therefore have two independent sources that agree.

---

## 2. The dominant depth sink: 46.1% of the clock takes no action

Every game runs the full 7,920 s and ends `gave_up` [VERIFIED,
`final_wallclock_seconds` 7,920.1–7,965.4, `state` for all 25]. **Wall clock,
not the action cap, is what ends every game**, so the unit that matters is the
turn, and a turn that executes nothing is a turn the game did not get.

`scripts/analyze_dead_turns.py`, over three independent runs of the same
chassis [VERIFIED]:

| run | turns | executed | dead | clock in dead turns | terminal streaks |
|---|---:|---:|---:|---:|---:|
| **anim** (incumbent) | 940 | 586 | **354 (37.7%)** | **91,295 s / 198,233 s = 46.1%** | 89 turns, 21,598 s (10.9%) |
| anim-wg | 921 | 558 | 363 (39.4%) | 96,363 s / 198,128 s = 48.6% | 85 turns, 20,125 s (10.2%) |
| anim-rs | 929 | 553 | 376 (40.5%) | 98,606 s / 198,161 s = 49.8% | 76 turns, 17,363 s (8.8%) |

Median dead turn **258 s**; median turn that acts **173 s** [VERIFIED, anim;
277/163 and 273/164 on the siblings]. A dead turn costs *more* than a productive
one, because it runs until the 180 s yield budget trips and then finishes the
call already in flight.

### 2.1 Why a dead turn is unrecoverable, from the bundle's own source

`inference/agent/tool_agent.py: analyze()` breaks out of its tool loop when
`time.monotonic() - turn_started_at >= self._yield_seconds` and returns
`yielded_control=True`. `inference/framework/solver.py:360-361` then does
[VERIFIED, sha256 `2bef5d6b…7c8e`]:

```python
retry_analysis_step = None
if getattr(result, "yielded_control", False):
    retry_analysis_step = analysis_step
    continue
```

— it replays **the same analysis step**. So consecutive dead turns are
consecutive attempts at one decision, not progress. History is preserved across
them (`preserve_history` stays `True` on a plain yield), so this is not amnesia:
the model accumulates its own investigation and still does not commit.

### 2.2 What it looks like

`r11l`, a game that had cleared 2 of 6 levels, made **22 `analyze()` calls at
`analysis_step=10`**, from 04:37:31 to 05:52:01, every one ending
[VERIFIED, transcript]:

```
step_executed: False
message: Yielded control to solver: turn_time_budget.
```

Each of those turns ran two `python` calls, both inspection-only — real,
careful, converging work ("Level 2 completed on a click into plain black. Let me
compare the level-2 before-frame's group centroids vs ring centers…") — and
never called `action(...)`. The game's last action was at 3,262 s; the remaining
**4,703 s (59% of its clock)** produced nothing.

---

## 3. Deaths: real, measured, and the wrong target

`analyze_depth.py` treats a game over as destroying the *life*, not the level —
verified in this repo's own event streams, where `level` does not revert across
a `GAME_OVER` and the next action is a `RESET` at the same level.

**16 in-level game overs in 8 of 25 games, destroying 520 actions — 19.9% of all
2,615.** [VERIFIED]

| game | deaths | actions in each life that ended in death | levels | score | +1 level worth |
|---|---:|---|---:|---:|---:|
| `sp80` | 6 | 30, 31, 31, 31, 31, 31 | 0/6 | 0.00 | +4.76 |
| `tu93` | 3 | 4, 14, 9 | 4/9 | 21.46 | +11.11 |
| `bp35` | 2 | 3, 7 | 1/9 | 0.85 | +4.44 |
| `dc22` | 1 | 94 | 0/6 | 0.00 | +4.76 |
| `sc25` | 1 | 59 | 0/6 | 0.00 | +4.76 |
| `vc33` | 1 | 50 | 3/7 | 21.22 | +14.29 |
| `su15` | 1 | 25 | 1/9 | 2.03 | +4.44 |
| `wa30` | 1 | 70 | 1/9 | 1.89 | +4.44 |

`sp80`'s six lives of 30–31 actions re-derive Thuitanium's published hidden
budget (`sp80 L1 = 30`) exactly, from our own run.

**Why it is the wrong target.** `w_l = l`, so a level is worth its index. If
*every one* of these eight games gained one level, the public-25 mean rises
**+2.12**. If the ten games already two or more levels deep each gained one, it
rises **+5.30**. The deaths are concentrated where the weight is not:
five of the eight are at level 0–1, where one more level is worth 4.4–4.8 raw,
i.e. under +0.2 each on the mean.

This also re-confirms `stage7_action_budget.md` §2.1 from a different angle: only
one level in the whole run supports budget inference, and its inferred budget
(30) is *below* that level's human baseline (39), so knowing the number does not
produce a clearance.

---

## 4. Where the actions go, against what they are worth

Actions by level index, from `benchmark.json`'s own `actions_per_level`
[VERIFIED, anim]:

| level | actions | share of actions | levels cleared | RHAE weight share |
|---:|---:|---:|---:|---:|
| 1 | 1,240 | 47.4% | 20 | 1.8% |
| 2 | 788 | 30.1% | 10 | 3.6% |
| 3 | 329 | 12.6% | 6 | 5.5% |
| 4 | 201 | 7.7% | 5 | 7.3% |
| 5 | 44 | 1.7% | 1 | 9.1% |
| 6 | 13 | 0.5% | 0 | 10.9% |
| 7–10 | 0 | 0.0% | 0 | 72% combined |

**Deep levels are starved — but not by a schedulable budget.** A level index is
not a slot you can move actions into: you cannot spend an action on level 5
until level 4 is cleared. The honest reading of this table is not "reallocate
actions", it is **"the run terminates two to three levels short of where the
weight lives, in every game"**. 72% of the total weight sits on levels 7–10,
which the run never reaches at all.

**Budget split** [VERIFIED]:

* **43.3%** of the whole run's wall clock is spent before that game's *first*
  level completes (85,821 s of 198,233 s).
* Of that, **20.0%** of the run's total clock belongs to the five games that
  never cleared level 1 at all (`dc22`, `sc25`, `sk48`, `sp80`, `tn36` — 39,608 s,
  612 actions, all five scoring 0.00).
* **35.2%** of the clock falls after each game's last completed level.

---

## 5. How close the scoring games got, and what it was worth

`stall_ratio` is the actions spent on the first uncompleted level against that
level's human baseline `h_l` from `benchmark.json`'s `base_actions_per_level`.
`+1 lvl` is the game's score recomputed with `levels_completed + 1`, through the
same RHAE implementation that reproduces this run's `final_score` values
(`scripts/project_turns_rhae.py: rhae_score`).

[VERIFIED, top of the table by marginal value]

| game | score | levels | stall level | ours/human on it | ratio | clock before 1st clear | **+1 level** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ft09` | 47.62 | 4/6 | 5 | 21/65 | 0.32 | 3% | **+23.81** |
| `lp85` | 39.35 | 5/8 | 6 | 13/60 | 0.22 | 7% | **+18.99** |
| `vc33` | 21.22 | 3/7 | 4 | 51/61 | 0.84 | 9% | **+14.49** |
| `r11l` | 14.29 | 2/6 | 3 | 11/51 | 0.22 | 12% | **+14.29** |
| `ar25` | 27.78 | 4/8 | 5 | 9/89 | 0.10 | 8% | **+13.89** |
| `re86` | 27.78 | 4/8 | 5 | **1**/189 | 0.01 | 15% | **+13.89** |
| `tu93` | 21.46 | 4/9 | 5 | **0**/123 | — | 7% | **+11.11** |
| `m0r0` | 3.71 | 1/6 | 2 | 61/111 | 0.55 | 42% | +10.58 |
| `ka59` | 10.71 | 2/7 | 3 | 75/51 | 1.47 | 26% | +6.56 |
| … | | | | | | | |
| `sp80` | 0.00 | 0/6 | 1 | 215/39 | 5.51 | 100% | +0.16 |
| `sk48` | 0.00 | 0/8 | 1 | 157/61 | 2.57 | 100% | +0.42 |

**Seven of the top eight stalls are at or below a third of the human action
count for the level they died on, and two are at literally 0 and 1 action.**
Those games did not fail their next level. They never started it.

Two aggregates [VERIFIED]:

* if every one of the 12 games whose stall level was under 0.40× the human
  baseline gained one level: **+5.52** public-25;
* if every one of the 25 games gained one: +8.97 (the ceiling, not a forecast).

For scale, `stage7_action_budget.md` §1.5 puts the **entire** efficiency term at
**+0.54** on this chassis. Depth is worth an order of magnitude more than
efficiency, and this is the third independent derivation in Stage 7 saying so.

---

## 6. Ranked candidates

Ranked by (expected effect × confidence) ÷ risk, and every one judged against
`w_l = l`.

### Built — 1. Commit floor (kernel `-cf`)

* **Mechanism.** After two consecutive `analyze()` turns that executed no
  action, the next turn's user prompt carries a harness directive requiring it
  to end in `action(...)`; escalates after five. Nothing is forced, injected or
  reset — the model still chooses the action.
* **Measurement that motivates it.** §2: 46.1% of the clock, 354 of 940 turns,
  replicated at 48.6%/49.8%; 89 terminal-streak turns holding 21,598 s; and
  §5, which shows the games holding those streaks are the ones where a level is
  worth +14 to +24.
* **Expected effect under RHAE.** [INFERRED] The recoverable pool is the 101
  turns of the anim run that sit on a streak of 3+ (26,565 s, 13.4% of the
  clock). At the run's own 4.5 actions per executing turn that is ~450 actions
  landing on the current level of the games that are deepest. If it converted
  in `r11l`, `cn04`, `ft09` and `bp35` alone the mean moves +2.11; a realistic
  fraction of that is far smaller. **Ceiling, not forecast.**
* **Risk.** Acting earlier than the model wants can spend a hidden per-level
  action budget (§3, `sp80` L1 = 30) or walk into a death. Bounded: the four
  games holding the largest streaks (`cn04`, `r11l`, `tr87`, `ft09`) had
  **zero** deaths between them [VERIFIED], and the mechanism is structurally
  inert on any turn not preceded by two dead ones — so most of the run cannot
  be touched at all, which also gives the free run its own null arm.
* **Threshold fixed in advance, not swept.** 2 consecutive dead turns, which is
  `sahasawatt/thui-af-v0`'s published action-floor trigger (a team at 3.74).
  Our own streak histogram endorses it rather than fitting it: 125 of 189
  streaks are a single turn (ordinary deliberation, left alone), 34 are two, and
  the 29 streaks of 3+ hold 71 turns and 18,664 s.

### Built — 2. No-thinking (kernel `-nt`)

* **Mechanism.** One post-import rebind,
  `tool_agent._LOCAL_ANALYZER_ENABLE_THINKING = False`, which
  `_chat_completion` reads per call and `build_chat_payload` turns into the
  vLLM `chat_template_kwargs={"enable_thinking": False}` switch.
* **Measurement that motivates it.** Same 46% sink, attacked from the cost side.
  Across the anim run's 1,358 model responses [VERIFIED, `[MODEL RESPONSE META]`
  blocks]: **4,768,656 reasoning characters against 250,663 of content — 95.0%
  of all generated text is the reasoning block**, median 2,030 characters per
  response. Generation is what makes a turn exceed 180 s before the model acts.
* **Expected effect under RHAE.** [INFERRED] Compounding on this stack: shorter
  responses mean more `python` calls fit inside one 180 s turn *and* fewer
  resident tokens per request, and turn latency here is 86.97% queue wait driven
  by KV residency (`stage7_kv_residency_levers.md`). Plausibly the largest
  single lever on turns-per-game that remains untested.
* **Risk. High, and the direction is known.** This repo's precedent is that every
  intervention reducing what the model can think with (`dedupe`, `ctx16k`) raised
  tokens per turn and lost score. `stage7_sota_research.md` lists it as item 7,
  "high risk, listed for completeness, not recommended first". It is second here
  for exactly that reason, and it is here at all because one free run settles it.

### Not built — 3. Yield budget 180 s → 300–420 s

Same sink, opposite sign: give the turn enough budget to finish deliberating
rather than pressuring it to conclude. Cheap (one knob) but weakly motivated:
`r11l` had 44 inspection calls across 22 retries of one step with history
preserved and still did not act, so more time within a turn is not obviously
what it lacked. It is also a knob sweep, which `stage7_config_locality.md`
specifically warns against fitting to a run. Worth a free push later; not worth
one of the two slots here.

### Ruled out — 4. Death avoidance / death blacklist

**Rejected on measured weight, not on mechanism.** §3: every death-affected game
gaining a level is worth +2.12, half of what the already-deep games are worth,
and the mechanism costs prompt tokens on all 25 games to address 8. The one
level in the run that supports budget inference has a budget below its own human
baseline. Independently reached by `stage7_action_budget.md` §2.3.

### Ruled out — 5. Restart-at-stall

Already built on `stage7-action-budget`; its free run is read here for the first
time [VERIFIED, `read_duck_public25_log.py`]: **8.71 mean, 39 levels (anim: 42),
fired 6 times, 120 turns discarded.** It attacks the same terminal-stall time
(terminal streaks 21,598 s → 17,363 s) by **discarding** the belief state. Its
free-run score cannot rank it (±2.46), but the levels count moved the wrong way,
and it is the mechanistic opposite of the commit floor, which keeps the belief
state and asks for a decision from it.

### Ruled out as breadth, explicitly

* **Action floor via `step_env`** (`thui-af-v0`'s own form): commits a *random*
  exploratory action rather than the model's. On a level whose hidden budget is
  30 actions that is actively harmful, and it buys coverage of click locations,
  not progress on the current level.
* **Anything that reallocates actions toward unreached levels.** §4: a level
  index is not a slot. You cannot spend an action on level 5 before clearing 4.
* **Anything that trades a deep game's clock for a shallow game's.** Under
  `w_l = l` that is a strictly losing trade, which is the reading the wipe-guard
  arm's 2.53 supports (2 levels traded for 2 more games scoring).

### Considered and dropped — 6. The level-transition boundary

Measured rather than assumed: the turn immediately after a level clears is
**50.7% dead** against ~29% at 1–5 turns into a level [VERIFIED, 33 executed /
34 dead at position 0]. Real, and a genuine re-grounding cost — but it is only
34 turns of the 354. The dominant bucket is 6+ turns into a level (230 dead
turns, 40.6%), which is the stall regime the commit floor targets.

---

## 7. What was built

Both arms are **one inserted cell** on top of
`kaggle_submission_duck_nvfp4_anim`, built by
`scripts/_build_duck_nvfp4_anim_depth.py`, which asserts **by per-cell sha256
that all 18 inherited cells are present, in order, unchanged**, and refuses to
build if the arm's cell contains a forbidden token (`os.environ`, `bm.solver`,
`SETUP_ENV_PATH`, and for the commit floor `LOCAL_ANALYZER` as well).
`stage7_config_locality.md`'s warning — a prior attempt carried an unrelated
`seqs=16` through three runs and confounded the only one that ran — is why this
is a mechanical check rather than an intention.

### 7.1 Commit floor — `scripts/commit_floor_cell.py`

Two wrappers on `ToolAgent`:

* `analyze` counts consecutive turns whose `AnalyzerTurnResult.step_executed` is
  false, resets on any executing turn and on a new session, and treats a `None`
  return (missing state file / upstream error) as neutral rather than as the
  model declining to act.
* `_build_user_prompt` appends the directive when that count is ≥ 2, with a
  stronger form at ≥ 5. The base prompt is returned byte-identical otherwise.

Safety:

* **It can never kill a game.** Every line of our own bookkeeping is in a
  `try/except`; the prompt wrapper returns upstream's text unchanged if the
  directive raises. Upstream exceptions are deliberately *not* swallowed:
  `solver.py` raises on a `None` result, so returning one would change
  behaviour.
* **It fails fast rather than degrading silently.** At install time it reads
  `inspect.getsource` of the real `analyze`, `_build_user_prompt` and
  `solver.py` and asserts five control-flow needles plus the prompt's own act
  instruction and the solver's retry branch. An upstream reword raises at
  cell-execution time instead of silently arming nothing. It refuses to install
  twice.
* **A 5-case synthetic probe runs before any real game**, covering the firing
  point, the reset on an executing turn, the escalation point, a new session,
  a `None` return, and an untouched turn being byte-identical to upstream. Probe
  traffic is then zeroed out of the counters.
* **24 tests** (`tests/test_commit_floor.py`), including six that prove the
  upstream-drift assertions fail loudly, one that the cell mentions no knob or
  env var, one that exactly two methods on `ToolAgent` are rebound, and one that
  the notebook carries this exact cell.

Telemetry: `COMMIT_FLOOR_INSTALLED` once, `COMMIT_FLOOR_FIRED n=… consec=…` per
armed turn, `COMMIT_FLOOR_RESULT consec=… executed=0|1` per armed turn — **the
conversion rate is the point**, and it is counted, not sampled — and
`COMMIT_FLOOR_FINAL turns=… dead=… fired=… escalated=… converted=… errors=…` at
exit.

Failure modes and their readings, fixed before the run:

| symptom | reading |
|---|---|
| no `COMMIT_FLOOR_INSTALLED` | a source assertion failed; the arm is invalid, not negative |
| installed but `fired=0` with dead turns in the log | the wrapper did not reach the live instances |
| `errors > 0` | an unanticipated shape; the run is still valid (it fell through) but the count is a defect |
| `fired` high, `converted` ≈ 0 | the mechanism engaged and the model ignored it — a clean negative on the *directive*, not on the diagnosis |

### 7.2 No-thinking — `scripts/no_thinking_cell.py`

One `setattr`. Asserts the flag exists, is still `True` (so nothing else set it
and the arm is genuinely one variable), that `_chat_completion` still reads it
at call time, that `ToolAgent.__init__` does **not** snapshot it (which would
make a post-import rebind silently inert on the per-game agents the solver
builds later), and that `build_chat_payload` still emits the switch. Then it
probes the **real** payload builder before and after the flip and asserts that
`chat_template_kwargs` is the only key that moved. **10 tests**
(`tests/test_no_thinking.py`).

### 7.3 The falsifier, stated before the run — and it is a *counted* one

`stage7_noise_floor.md` establishes that a single public-25 pass carries
SE ±2.46, so the score cannot rank these arms. The commit floor does not need
it to, because the chassis provides its own comparator.

`scripts/analyze_dead_turns.py` now reports **P(a turn executes | k consecutive
dead turns immediately before it)** — what the *untreated* chassis does at
exactly the point the mechanism intervenes. It is remarkably stable
[VERIFIED, three independent runs]:

| k dead turns before | anim | anim-wg | anim-rs | pooled |
|---:|---:|---:|---:|---:|
| 0 | 69% | 69% | 71% | 1234/1772 = **69.6%** |
| 1 | 64% | 62% | 51% | 299/505 = 59.2% |
| 2 | 52% | 39% | 42% | 87/196 = 44.4% |
| 3 | 30% | 25% | 29% | 27/98 = 27.6% |
| 4 | 44% | 33% | 26% | 22/66 = 33.3% |
| 5 | 0% | 27% | 38% | 10/39 = 25.6% |
| 6+ | 10% | 17% | 20% | 18/114 = 15.8% |
| **k ≥ 2 (where the floor fires)** | **33.6%** | **30.4%** | **32.1%** | **164/513 = 32.0%** |

**So: without the directive, a turn arriving with two or more dead turns behind
it acts 32.0% of the time.** The free run prints
`COMMIT_FLOOR_RESULT consec=… executed=0|1` for every armed turn, so
`converted / result_lines` is directly comparable to that number, counted rather
than sampled. At the ~100 armed turns the anim run's streak histogram predicts,
a binomial at p = 0.32 has sd ≈ 4.7 pp.

**Falsifier, fixed before the run:**

> If the directive fires on the expected order of turns and the armed-turn
> conversion rate is not materially above **32%**, the model is ignoring an
> explicit instruction and this mechanism does not work on this chassis. The
> line closes, regardless of what the public-25 score says in either direction.

And symmetrically: **a conversion rate well above 32% is evidence the mechanism
works even if the score moves the wrong way**, because the score at n=1 cannot
resolve anything at this effect size and the conversion rate can.

The no-thinking arm has no comparable internal comparator — its telemetry is the
dead-turn fraction, turns per game, actions and reasoning characters, all
counted, but its *capability* cost can only be read from outcomes, which is
exactly what a free run cannot resolve. That asymmetry is itself a reason it
ranks second.

---

## 8. Free-run results

<!-- RESULT -->

---

## 9. Recommendation

<!-- RECOMMENDATION -->
