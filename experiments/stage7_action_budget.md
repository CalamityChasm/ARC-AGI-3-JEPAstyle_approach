# Stage 7 — The action budget: where the actions go, and restart-at-stall

Date: 2026-09-16. Branch: `stage7-action-budget`. Kernel
`calamitychasm/arc3-duck-nvfp4-anim-rs` (free push, no submission).

**No competition submission was made by this investigation, and none will be.**
Everything here is read-only analysis of completed runs plus one free
`kaggle kernels push`. The daily slot was untouched.

Labels follow this repo's convention. **[VERIFIED]** = read directly from a
primary artifact (bundle source, a run's own artifacts, a Kaggle API response,
or a locally-run computation over them). **[INFERRED]** = reasoning over
verified facts, not directly observed.

Reproduce every number here with:

```
venv/Scripts/python.exe scripts/analyze_action_budget.py \
    anim=<anim-dir> baseline=<nvfp4-dir> --json experiments/stage7_action_budget_runs.json
venv/Scripts/python.exe scripts/measure_stall_and_death.py \
    anim=<anim-dir> baseline=<nvfp4-dir> --json experiments/stage7_action_budget_sizing.json
venv/Scripts/python.exe scripts/measure_turns_to_clear.py anim=<anim-dir>
venv/Scripts/python.exe scripts/measure_stall_repetition.py anim=<anim-dir>
venv/Scripts/python.exe scripts/simulate_restart_trigger.py <anim-dir> --thresholds 10,14,16,18,20,24
venv/Scripts/python.exe scripts/read_duck_public25_log.py <anim-dir> <rs-dir>
venv/Scripts/python.exe -m pytest tests/test_restart_at_stall.py -q
venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim_restart.py --check
```

`<anim-dir>` / `<nvfp4-dir>` are unpacked `kaggle kernels output` directories for
`arc3-duck-nvfp4-anim` (the incumbent) and `arc3-duck-nvfp4-baseline`.

---

## 0. The falsifier, stated before the result

> **If restarts fire but levels completed and sunk actions are both unchanged,
> the mechanism does not work here and the line closes.**

Three numbers fixed before the run was pushed, so a bug cannot be mistaken for a
negative result (§6 reads them back):

- **Expect 6–10 firings, on ~8 games.** §4.3 replays the incumbent and gets
  exactly 8. `fired=0` with stalls present means the class patch never reached
  the live instances; `fired > 15` means the turn counter is wrong.
- **Expect them on `dc22`, `sk48`, `sb26`, `sp80`, `sc25`, `tn36`, `su15`** —
  five of which are the five games that scored 0.00 — **plus `lf52`, which is
  the predicted collateral**: its level 2 took 22 turns and *did* clear, so a
  restart there fires ~6 turns before a success.
- **The ceiling is +1.23 public-25 and the realistic ceiling is +0.71**
  (§4.4). A null result at that size is weak evidence, not strong — and §7 says
  which it was rather than collapsing the two.

---

## 1. Task 1 — the wasted-action figure, re-derived

### 1.1 What was wrong with the published number

`scripts/analyze_rhae_binding.py:level_actions()` counts every row in
`artifacts/*_events.jsonl` with a non-null `action_num`. Three row types carry
one [VERIFIED, by grouping rows on their key sets]:

| row `type` | carries `action_num` | is a real action |
|---|---|---|
| `initial` | yes | no — the opening frame |
| `action` | yes | **yes** |
| `analysis` | yes | no — mirrors the preceding action row |

On the nvfp4 baseline that counts **4,970 rows against 3,633 real actions**
(+36.9%). The published **51.4%** came from that numerator and denominator.

### 1.2 The primary source is the harness's own `[finished]` line

Nothing needs reconstructing. The harness prints one line per game
[VERIFIED, `arc3-duck-nvfp4-anim.log`]:

```
[finished] sp80-589a99af state=gave_up level=0/6 score=0.00 actions=215
           tokens=64810 per-level=215/39,0/58,0/25,0/148,0/96,0/152
```

`per-level` is `ours/human` for every level; `level=k/M` is levels completed out
of total; `score` is the reported per-game `E_e` as a percentage. It passes
three independent checks, all [VERIFIED] by `scripts/analyze_action_budget.py`:

1. the per-level "ours" values sum to `actions`, for all 25 games in both runs;
2. `actions` matches `benchmark.json` for all 25 games in both runs;
3. `score` matches `score.json` for all 25 games in both runs.

### 1.3 The corrected figure

Sunk = actions on levels never completed (`level=k/M` ⇒ everything from level
k+1 on). [VERIFIED]

| | nvfp4 baseline | **anim (incumbent)** |
|---|---:|---:|
| actions | 3,633 | **2,615** |
| sunk actions | 1,825 | **1,270** |
| **sunk fraction** | **50.2%** | **48.6%** |
| productive actions | 1,808 | 1,345 |
| levels solved | 48 / 183 | 42 / 183 |
| public-25 mean | 10.69 | 9.97 |

**The corrected figure is 48.6% on the chassis we actually run, against the
51.4% in circulation.** This is *not* a case of an inflated estimate deflating
on inspection — the pathology is real and close to half of all play. Two
independent implementations agree (`analyze_action_budget.py` from the
`[finished]` lines; `read_duck_public25_log.py` from the same lines by a
different parser: 0.4857).

**Reconciliation with the wipe-guard document**, which reported 49.3% / 50.9%
from the event stream with the `type == "action"` filter applied. Both are
right about the filter; they differ by exactly one action per game, and the
cause is identifiable [VERIFIED, `ar25`: events `{1:17, 2:25, 3:45, 4:23, 5:10}`
vs `[finished]` `18,25,45,23,9`; same ±1 shift on `lp85`, `ft09`, `r11l`]. The
event stream moves one action per game out of level 1 and into the deepest
level; the totals are identical. The `[finished]` line is the harness's own
per-level accounting — the accounting that produces `score.json` — so it is
primary, and 48.6% supersedes 49.3%.

### 1.4 Where the actions are sunk — per game [VERIFIED]

Anim, sorted by sunk actions. `stall` is the level in progress when the clock
ran out; `ours/hum` is that level's action count against the human baseline.

| game | score | levels | actions | sunk | sunk% | stall | ours/hum | ×human |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sp80` | 0.00 | 0/6 | 215 | 215 | 100% | 1 | 215/39 | 5.51 |
| `sk48` | 0.00 | 0/8 | 157 | 157 | 100% | 1 | 157/61 | 2.57 |
| `sb26` | 2.78 | 1/8 | 140 | 129 | 92% | 2 | 129/28 | 4.61 |
| `dc22` | 0.00 | 0/6 | 97 | 97 | 100% | 1 | 97/59 | 1.64 |
| `sc25` | 0.00 | 0/6 | 94 | 94 | 100% | 1 | 94/36 | 2.61 |
| `g50t` | 1.67 | 1/7 | 202 | 88 | 44% | 2 | 88/175 | 0.50 |
| `ka59` | 10.71 | 2/7 | 159 | 75 | 47% | 3 | 75/51 | 1.47 |
| `wa30` | 1.89 | 1/9 | 148 | 71 | 48% | 2 | 71/119 | 0.60 |
| `m0r0` | 3.71 | 1/6 | 95 | 61 | 64% | 2 | 61/111 | 0.55 |
| `vc33` | 21.22 | 3/7 | 111 | 51 | 46% | 4 | 51/61 | 0.84 |
| `tn36` | 0.00 | 0/7 | 49 | 49 | 100% | 1 | 49/32 | 1.53 |
| `s5i5` | 7.47 | 2/8 | 122 | 42 | 34% | 3 | 42/106 | 0.40 |
| … 13 more, each ≤ 32 sunk | | | | | | | | |

**Five games scored 0.00 having never cleared level 1**: `dc22`, `sc25`,
`sk48`, `sp80`, `tn36`. They account for 612 of the 1,270 sunk actions (48%) and
23–30 analysis turns each.

### 1.5 What the sunk actions cost, and what they do not

Per `stage7_sota_research.md` §2, and re-derived here on the incumbent
[VERIFIED]: only *solved* levels enter either RHAE sum, so sunk actions are
**pure opportunity cost, not a scored penalty**. On anim, completion binds in
**14 of 25** games and efficiency in 11; the entire efficiency cap is worth
**0.54** of a 10.52 completion-only ceiling (reported 9.97). On the baseline it
was 16/25 and 1.25 of 11.94.

That number matters for §2: **0.54 public-25 points is the total value of
perfect efficiency on the incumbent.** Any mechanism whose payoff is
efficiency is bidding against that cap.

---

## 2. Task 2 — which mechanism, and why

Both candidates were sized on the incumbent chassis, from its own artifacts.

### 2.1 Death blacklist — measured headroom

`scripts/measure_stall_and_death.py`, counting `game_over` on `type == "action"`
rows only. [VERIFIED]

| | baseline | **anim (incumbent)** |
|---|---:|---:|
| real game overs | 37 | **16** |
| games with ≥1 | 10 / 25 | **8 / 25** |
| levels with an inferrable budget (≥3 lives agreeing ±1) | 2 | **1** |

The one inferrable budget on the incumbent is **`sp80` level 1 = 31**, from six
lives of `30, 31, 31, 31, 31, 31`. **This independently reproduces
Thuitanium's published `sp80 L1 = 30`** from our own run, which is real
corroboration that the mechanism and their table are sound. [VERIFIED]

It is also the whole of the headroom:

- **1 level in 1 game of 25** supports budget inference at all.
- **Four of the five games scoring 0.00 have ≤ 1 death** (`sk48` 0, `tn36` 0,
  `dc22` 1, `sc25` 1). A death blacklist is structurally inert on them.
- On the one game it addresses, the inferred budget (31) is **below the human
  baseline for that level (39)**, so the level cannot be cleared inside one
  life and knowing the number does not by itself produce a clearance.
  [VERIFIED from the `[finished]` line's `215/39`.]
- Its payoff is mostly efficiency, and §1.5 caps the entire efficiency term at
  **+0.54**.
- It costs prompt tokens **on every turn**, in a regime where turn latency sets
  calls per game (median 190 s/turn measured in §3.2, ~42 turns per 7,920 s
  game). That cost is paid on all 25 games to address 8.

### 2.2 Restart-at-stall — measured headroom

| | **anim (incumbent)** |
|---|---:|
| sunk actions | 1,270 (48.6%) |
| exact-repeat rate inside the stalls | **80.5%** |
| games whose final stall ≥ 20 analysis turns | **7 / 25** |
| sunk actions held by those 7 | **773** |
| of those 7, games that scored 0.00 | **5** |
| ceiling if all 7 convert to one more level | **+1.23 public-25** |
| ceiling for the 4 with ≥ median runway | **+0.71 public-25** |

### 2.3 The decision

**Restart-at-stall, on measured headroom, by roughly an order of magnitude.**
Its realistic ceiling (**+0.71**) is larger than the *entire* efficiency term
(**+0.54**) that a death blacklist could at most recover, it touches 7 games
rather than 1, it reaches all five 0.00 games rather than one, and it costs no
tokens. The death blacklist's one verified finding — `sp80 L1 = 31` — is a
good result that does not translate into a lever at this scale.

---

## 3. What the stall actually is

A restart only pays if what it discards was going to be wasted. Two
measurements, both [VERIFIED].

### 3.1 The stall is a fixation

`scripts/measure_stall_repetition.py` counts actions identical to an
`(action, level)` pair already taken on that level. The environment is
deterministic, so a repeat is by construction telling the agent nothing new.

**Across the 25 final stalls: 1,038 of 1,290 actions (80.5%) are exact
repeats.** On the five 0.00 games: `sk48` 97%, `sp80` 95%, `sc25` 78%,
`dc22` 68%, `tn36` 61%.

The shape is visible in the raw action stream [VERIFIED]:

- `sk48` — 157 actions, 94% of them changing the board, but only five distinct
  actions ever used: RIGHT 51, LEFT 51, UP 28, DOWN 26, one MOUSE click. A
  balanced left/right count is an oscillation.
- `sb26` — the same MOUSE cell clicked 17 times; the top six cells account for
  90 of 130 stall actions.
- `dc22` — `MOUSE(19,48)` 19 times; `tn36` — `MOUSE(55,35)` 17 of 49.

**The honest caveat**: repetition is high *before* the stall too on many games
(`g50t` 96%, `ls20` 95%, `re86` 90%). It is a property of this agent's play, not
only of its stalls. What distinguishes the stall is not a higher repeat rate but
that the repeats are no longer bracketed by level completions.

### 3.2 Clearing history does **not** buy extra turns — measured, negative

A tempting second-order claim is that a restart shortens the prompt and so fits
more turns into the remaining wall clock. It does not. Per-turn wall time by
turn index, from the transcript headers [VERIFIED]:

| turns | 0–4 | 5–9 | 10–14 | 15–19 | 20–24 | 25–29 | 30–34 |
|---|---:|---:|---:|---:|---:|---:|---:|
| median s | 112 | 257 | 230 | 203 | 178 | 199 | 141 |

Latency peaks early and **falls** afterwards — consistent with the harness's own
context eviction already bounding history. **The runway after a restart is
whatever the wall clock already allowed, and no more.** This is the mechanism's
single biggest limitation and §4.4 sizes it.

---

## 4. Choosing the threshold without fitting it to the outcome

### 4.1 The turn unit

`ToolAgent.analyze` is called **more than once per turn**. Verified in the anim
bundle itself, not by analogy with the June duck — `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`,
`src/ARC3-Inference/inference/framework/solver.py` (sha256
`2bef5d6bc23c0312675f0c7203194c94e93d056ac06bf6419acd5142a4ea7c8e`) [VERIFIED]:

```python
# line 333-342
result = self.analyzer.analyze(
    self.state_path, self.action_count, ...,
    analysis_step=analysis_step, ...)          # passed by KEYWORD
...
# line 359-362
retry_analysis_step = None
if getattr(result, "yielded_control", False):
    retry_analysis_step = analysis_step        # the SAME step runs again
    continue
```

A yielded turn — the 36% of calls that take no action, and the thing
`yield_seconds = 180` makes common — reruns `analyze` at the same
`analysis_step`. On the anim run `sp80` made **41 `analyze` calls across 26
distinct `analysis_step` values** [VERIFIED]. Counting calls would fire ~1.5×
too early, so turns are the **set** of distinct `analysis_step` values.

The same lines settle the keyword question: `analysis_step=analysis_step` is a
keyword argument, so `kwargs.get("analysis_step")` reaches it. The `no_step`
counter in §5.5 exists in case that ever changes.

Run-wide [VERIFIED, transcripts]: 940 `analyze` calls, 922 with a
`step_executed` line, 586 executed. ~610 distinct turns over 25 games, i.e.
**~24 turns per game** — the whole budget a threshold has to live inside.

### 4.2 The distribution that sets it

`scripts/measure_turns_to_clear.py`. [VERIFIED]

| | levels that cleared (42) | levels that never cleared (25) |
|---|---:|---:|
| median turns | **5** | 10 |
| p75 / p84 / p90 | 8 / 12 / 17 | — |
| max | 26 | 30 |

| T | cleared ≥ T | (% of 42) | stalls ≥ T | (% of 25) |
|---:|---:|---:|---:|---:|
| 12 | 8 | 19% | 11 | 44% |
| 16 | 5 | 12% | 10 | 40% |
| 18 | 3 | 7% | 8 | 32% |
| **20** | **1** | **2%** | **7** | **28%** |
| 24 | 1 | 2% | 4 | 16% |

**T = 20 is used.** At that value one cleared level in 42 would have been
interrupted against 7 of 25 stalls — a 12:1 separation. It is also
`sahasawatt/thui-rs-v0`'s published value, derived from the same distribution on
their own chassis ("cleared levels take median 10 turns, 84% ≤ 20; the stalled
level burns median 32"). **It is an externally-fixed number that our data
independently endorses, not one swept here for best result** — which is what
`stage7_config_locality.md` requires. Lower thresholds buy more runway (T=12
gives 1,352 actions of runway against T=20's 287) at a rising cost in
interrupted healthy levels; that trade was not taken, because taking it would be
fitting a knob to the measurement run.

### 4.3 Where it fires, replayed [VERIFIED, `simulate_restart_trigger.py`]

**Eight** games, one firing each (a second needs 20 *more* turns; the longest
stall is 30, so the per-level cap is never reached on this run). Total runway:
47 turns / 287 actions.

| game | score | level | turns on it | fires at | runway after |
|---|---:|---:|---:|---:|---:|
| `dc22` | 0.00 | 1 | 30 | 20/30 | 10 turns, 32 actions |
| `sk48` | 0.00 | 1 | 29 | 20/29 | 9 turns, 42 actions |
| `sb26` | 2.78 | 2 | 28 | 24/32 | 8 turns, 39 actions |
| `sp80` | 0.00 | 1 | 26 | 20/26 | 6 turns, 108 actions |
| `sc25` | 0.00 | 1 | 23 | 20/23 | 3 turns, 9 actions |
| `tn36` | 0.00 | 1 | 23 | 20/23 | 3 turns, 6 actions |
| `su15` | 2.03 | 2 | 20 | 27/27 | 0 turns, 0 actions |
| **`lf52`** | **4.21** | **2** | **22** | **24/32** | **8 turns, 51 actions** |

**`lf52` is the collateral, and it is worth naming before the run rather than
explaining afterwards.** Its level 2 is the one cleared level in 42 that ran past
20 turns — it cleared at turn ~30, so the restart fires about six turns before a
success and throws away the belief state that was about to produce it. That is
the measured price of T=20, and the mechanism has to beat it.

### 4.4 The honest sizing, including the deflation

Median runway after a firing is **6 turns**, against a median turns-to-clear of
**5**. So the mechanism is viable — barely — on the four stall firings with ≥
median runway (`dc22` 10, `sk48` 9, `sb26` 8, `sp80` 6) and near-inert on the
other three (`sc25` 3, `tn36` 3, `su15` 0).

| if it converts | public-25 gain |
|---|---:|
| all seven firings → one more level each | **+1.23** (9.97 → 11.20) |
| the four with ≥ median runway | **+0.71** (9.97 → 10.68) |
| one firing | ~+0.19 |

**[INFERRED] These are ceilings, not forecasts.** The restarted agent gets
*fewer* turns than the attempt that just failed, with no history — the case for
it resting entirely on §3.1's finding that the failed attempt was spending 80%
of its actions on exact repeats. Against that: 1 of 42 cleared levels is
interrupted, and this project's own record (`GraphExplorerJepaAgent`, 46 → 34
levels) is that adding informed behaviour to a working algorithm can cost more
than it gains.

---

## 5. The mechanism

`scripts/restart_at_stall_cell.py`, installed as one added cell. The behavioural
change is one branch inside a wrapper on `ToolAgent.analyze`:

```python
if len(st["steps"]) >= RESTART_STALL_TURNS and st["restarts"] < RESTART_STALL_MAX_PER_LEVEL:
    carried = _rs_forget(self)
    _rs_tool_agent._LOCAL_ANALYZER_SEED = int(_rs_tool_agent._LOCAL_ANALYZER_SEED) + 1
```

### 5.1 What it clears, and what it deliberately does not

`ToolAgent._ensure_session` (`tool_agent.py:1140-1155`) is upstream's own
definition of forgetting, and clears seven things [VERIFIED]. The restart clears
the **four that are belief** — `_history_messages`, `_summarized_knowledge`,
`_last_step_summary`, `_last_action_result` — and leaves the rest:

| left alone | why |
|---|---|
| `_session_total_tokens`, `_session_generated_tokens` | accounting, read by the `total_tokens`/`generated_tokens` properties into `benchmark.json`; zeroing them corrupts the run's own metrics |
| `_noop_guard`, `animation_counters`, `_reset_animation_hint_state()` | the anim solver's safety machinery, not belief |
| **`cross_level_notes`** | see below |

**Three deliberate deviations from `thui-rs-v0`**, each with upstream evidence:

1. **`cross_level_notes` is preserved.** `thui-rs-v0` assigns
   `_empty_world_model()`, zeroing all seven fields. Upstream never erases that
   field — the wipe at `tool_agent.py:1343-1356` spares it on all three of its
   triggers, including a full level transition [VERIFIED, and independently
   verified in `stage7_wipe_guard.md` §1.3]. It holds cross-level mechanics, not
   this level's failed plan, so a within-level restart has no reason to destroy
   it. Two of the seven firings are on level 2 or later, where it can matter.
2. **Token counters are untouched** (`thui-rs-v0` does not touch them either;
   stated because it is a real trap in `_ensure_session`'s full reset).
3. **A missing `analysis_step` is loud, not silent.** `thui-rs-v0` reads it with
   `kwargs.get("analysis_step")`; if the solver ever stopped passing it by
   keyword the counter would never advance and the entire arm would be a
   no-op that still printed "installed". Ours counts and prints `no_step`.

The seed bump is load-bearing: `_LOCAL_ANALYZER_SEED` is read at request-build
time (`tool_agent.py:1536`), not captured at construction, so rebinding the
module global takes effect on the next call [VERIFIED, and asserted at install
time]. Without it, a re-draw against a deterministic environment would repeat
the same opening.

### 5.2 Safety

- **It can never kill a game.** All bookkeeping is inside `try/except`; the call
  to the original is *outside* it and deliberately unguarded, because `analyze`
  returns the turn's result to the solver and swallowing an exception there
  would change control flow rather than protect it.
- **It fails fast rather than degrading silently.** At install time it reads
  `inspect.signature`/`getsource` of `analyze`, `_ensure_session` and the class
  body and asserts: `analyze` still takes `state_path` and `analysis_step`;
  it still resolves the frame via `load_runtime_state(state_path)`;
  `_ensure_session` still resets all four belief attributes and still rebuilds
  the world model; the request payload still reads `seed=_LOCAL_ANALYZER_SEED`;
  the world model still has exactly seven fields, one of them
  `cross_level_notes`. An upstream reword raises at cell-execution time. It also
  refuses to install twice.
- **A 9-case synthetic probe runs before any real game**, exercising:
  below-threshold, at-threshold, a duplicated `analysis_step` (the solver's
  retry path), a level change, the per-level cap, a new session, a missing
  `analysis_step`, a bookkeeping error, and an unreadable runtime state —
  asserting belief state, `cross_level_notes` survival, the seed, and that
  upstream is called exactly once every time. Its marker lines are suppressed
  and its counters zeroed, so neither the log nor the tally is polluted.

### 5.3 One variable, enforced mechanically

`scripts/_build_duck_nvfp4_anim_restart.py` inserts the cell after the anim
notebook's customization hook and asserts, **by per-cell sha256, that all 18
inherited cells are present, in order, unchanged**. It refuses to build if the
cell text mentions `os.environ`, `bm.solver`, `SETUP_ENV_PATH`, any solver
setting, or any `LOCAL_ANALYZER_*` global other than the seed — the seed being
the one the mechanism legitimately moves, and only *after* a restart fires, so
the baseline configuration is bit-identical to the incumbent's.

### 5.4 Provenance of the artifact

The notebook this repo builds is **byte-identical to the kernel version that
produced §6's numbers**: `scripts/_build_duck_nvfp4_anim_restart.py` reports
cell sha256 `eeba749e24437d3d`, and `git diff` of the built notebook against the
commit that pushed it is empty. One consequence is visible in the log: the
synthetic probe drives the *real* wrapper, so it emits three
`RESTART_STALL_FIRED` lines and one `RESTART_STALL_ERROR` **before** the
`RESTART_STALL_INSTALLED` banner. `scripts/read_duck_public25_log.py` slices at
the banner and reports how many lines it ignored; `RESTART_STALL_FINAL`'s
counters were never affected, because the probe zeroes them.

### 5.5 Tests

`tests/test_restart_at_stall.py` — **30 tests, all passing.** They stand up a
stub `inference.agent.tool_agent` whose `_ensure_session` and
`_empty_world_model` are **byte-for-byte the anim bundle's** (sha256
`856bf9b8…f8be7`) and whose `analyze` reproduces the real signature, the real
`load_runtime_state` call and the real per-request seed read, then exec the real
cell source against it. They check: the one case that changes; the four belief
attributes; `cross_level_notes` survival; token counters untouched; the seed
bump reaching the payload builder; upstream called on every turn including the
firing one; five cases that must **not** fire (retried step, level change, new
session, cap, missing `analysis_step`); four no-raise paths; that an upstream
exception is **not** swallowed; six fail-loud paths; double-install; that the
cell mentions no config knob; that the notebook carries this exact cell; and
that `scripts/read_duck_public25_log.py`'s regexes match the cell's **own
printed output** rather than a hand-written sample.

### 5.6 How we will know it fired, and what failure looks like

`RESTART_STALL_INSTALLED` once at setup, `RESTART_STALL_FIRED n=…` per firing,
`RESTART_STALL_FINAL fired=… turns_discarded=… capped=… no_step=… errors=…` at
exit. `scripts/read_duck_public25_log.py` recovers all three.

| symptom | reading |
|---|---|
| no `RESTART_STALL_INSTALLED` | the cell raised — a source assertion failed; the arm is invalid, not negative |
| installed, `fired=0`, `no_step` large | the solver stopped passing `analysis_step` by keyword; the arm is inert |
| installed, `fired=0`, `no_step=0` | the class patch did not reach the live instances — cell 11 unpickles `bm.solver`, so a pickled *bound* method is the plausible route |
| `fired` ≈ 30 | the turn counter is counting `analyze` calls, not distinct steps |
| `errors > 0` | an unanticipated state shape; the run is still valid (it fell through to upstream) but the count is a defect to chase |

---

## 6. Free-run result

Kernel `calamitychasm/arc3-duck-nvfp4-anim-rs`, free push, public-25 path.
Compared against the **anim** arm (**9.97 / 2,615 actions / 42 levels**), which
is the incumbent — *not* the keithtyser baseline.

<!-- RESULT -->

---

## 7. Recommendation

<!-- RECOMMENDATION -->

---

## 8. Corrections to earlier Stage 7 documents

- `stage7_sota_research.md` §2 and §5.2 — "actions on the never-completed level:
  2,557 / 4,970 (51.4%)". The denominator counts the event stream's mirrored
  `analysis` rows and its `initial` row; the real figure on that same baseline
  run is **1,825 / 3,633 = 50.2%**, and on the chassis we now run it is
  **1,270 / 2,615 = 48.6%** (§1). The conclusion it supports — completion binds,
  not efficiency — is unaffected and is re-derived here on the incumbent
  (14/25 completion-bound, whole efficiency term worth 0.54).
- `stage7_sota_research.md` §6.6 — "[the death blacklist] depends on #1 —
  without the wipe guard the agent has no memory to accumulate budgets into."
  Not so, for the published implementation: `thui-db-v1` keeps its own per-level
  death record outside `_summarized_knowledge` and injects it into the user
  prompt, so it is independent of the harness's wipe. The reason not to build it
  is §2.1's measured headroom — one inferrable budget in one game of 25 — not a
  dependency.
- `stage7_wipe_guard.md` §2.1 — "wasted-action fraction 50.9% / 49.3%". Right
  about the row filter; off by exactly one action per game, which the event
  stream attributes to the deepest level rather than to level 1 (§1.3). The
  harness's own `[finished]` accounting gives 50.2% / 48.6%.
- `stage7_sota_research.md` §1.3 — "46% of turns take no action". That is 46% of
  **LLM calls**, not turns: 940 calls carry 922 `step_executed` lines of which
  586 executed, but those calls span only ~610 distinct `analysis_step` values,
  because `solver.py` retries a yielded step (§4.1). The per-*turn* no-action
  rate on anim is ~4%. The call-level finding and its cause stand; the unit
  in the sentence does not.
