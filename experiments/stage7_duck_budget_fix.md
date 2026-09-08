# Stage 7 -- Duck notebook competition-rerun budget fix

Date: 2026-09-07. Branch: `stage7-duck-budget-fix`. Labels follow this
repo's own convention: **[VERIFIED]** = read directly from a primary
artifact (a real kernel log, the notebook source, the GitHub/Kaggle API,
or a locally-run test); **[INFERRED]** = reasoning over verified facts,
not directly observed.

This is a follow-up to a prior investigation
(`duck_gap_diagnosis.md`, not committed to this repo -- a scratch
diagnostic doc) into why our real Kaggle submission `55769792` (a
byte-identical copy of the public notebook
`foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b`) scored **1.77**.
That investigation refuted three hardware/mount hypotheses directly (GPU
provisioning, model mount, vLLM attention backend all confirmed working)
and found the real score gap is mostly explained by (a) the notebook's
own "LB 9" title being an inflated/lineage claim rather than a measured
result, and (b) the well-established local-games-vs-hidden-games
generalization gap this project has hit repeatedly elsewhere (Stage 6).
One concrete, fixable defect was found on top of that: the notebook's
competition-rerun time budget doesn't fit the real game count. This
document covers only that fix.

## Goal

Fix the one defect from the diagnosis that is (a) real, (b) ours to fix,
and (c) plausibly cheap to fix without touching anything already proven
to work (the model, the hardware selection, the attention backend) --
**not** to close the 1.77-vs-9 gap, most of which the diagnosis attributes
to an inflated title claim and a generalization gap this fix cannot
touch.

## The defect (from the diagnosis, restated with its sourcing)

**[VERIFIED]**, from a real, complete run of our own fork
(`calamitychasm/lb-9-arc3-duck-v12-with-qwen-3-8-27b`, the non-rerun/test
path, 25 public games, 2h13m real wall-clock):

- Solver config: `concurrency=28`, `max_runtime_s_per_game=7920.0`
  (2.20h), inherited from a pickled bundle (`benchmark_initial.pkl`)
  whose own `preamble.txt` documents it was tuned for a **6-game x
  4-pass** Preview shape, not 110 games.
- **Every one of the 25 games in that real run ended in state
  `gave_up`**, all at >=7900s (min 7920s, max 7971s, mean 7926s) -- none
  finished early. This solver does not wind down early; it uses its
  entire allotted cap on every game, every time.
- Measured setup cost before any game starts: **394.44s**, cross-checked
  two independent ways (kernel-relative timestamps vs. inverting the
  printed `soft_end_time` against the notebook's own `_soft_end_time()`
  formula) that agree to within 6 seconds.

**[VERIFIED]**, from the notebook source (cell 6, `_soft_end_time`; cell
9, the `if true_submission:` branch):

- `_soft_end_time(...)` returns `None` whenever `run_as_submission` is
  True -- **the scored rerun has no soft deadline at all**, only Kaggle's
  hard 9h kill.
- Cell 8's `concurrency = 28` / `max_runtime_s_per_game = 7920.0`
  assignment only runs `if not true_submission:` (the public/test path).
  **The scored rerun branch (cell 9) never overrides these values** -- it
  runs with whatever the pickled bundle shipped, i.e. the 6-game/4-pass
  Preview numbers, against a competition shape of up to ~110 games.
- The rerun branch additionally waits up to 600s for the gateway before
  the first game starts.

**[VERIFIED]** from `rules.md`: the private evaluation set is **110
games** (55 -> public LB, 55 -> private LB), 9-hour hard run-time cap per
notebook.

### The arithmetic (hand-derivation, matches `duck_budget.py`'s tests)

Because no game finishes early (verified above), the pool behaves as
fixed-length waves:

```
waves        = ceil(110 / 28)          = 4
game time    = 4 x 7920s               = 31,680s = 8.80h
setup        = 394s (measured)         = 0.11h
gateway wait = 0-600s (rerun-only, unmeasured)
--------------------------------------------------------------
total        ~= 8.91h - 9.08h   vs a 9.00h HARD cap
```

**[INFERRED]** wave 4 (the last ~26 of 110 games, ~24% of the set) sits at
or past the cap, with no soft deadline to wind it down gracefully. This
is inference, not a confirmed observation -- Kaggle exposes no log from a
real scored rerun (see `CLAUDE.md`'s own established finding to this
effect, re-confirmed during the diagnosis this fix is based on), so
whether truncation actually occurred in submission `55769792` is
**unknowable from here**, and remains so after this fix.

## The fix

`kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb`,
cell 9, immediately after `bm.games = _competition_games()`, entirely
inside `if true_submission:` (see
`experiments/stage7_duck_cell9_diff.txt` for the exact unified diff, and
the commit history on this branch for the reviewable git commits: an
unmodified-baseline commit followed by this fix as its own commit):

```python
_n_games = len(bm.games)
_elapsed_s = time.time() - NOTEBOOK_START_EPOCH
_conc = int(getattr(bm.solver, "concurrency", 28) or 28)
_waves = max(1, -(-_n_games // _conc))  # ceil division, no math import
_remaining_s = 9 * 3600 - _elapsed_s - 900.0  # 15-minute safety margin
_per_game_raw_s = _remaining_s / _waves
_per_game_s = max(600.0, _per_game_raw_s)  # floor: never collapse to ~0
if hasattr(bm.solver, "max_runtime_s_per_game"):
    bm.solver.max_runtime_s_per_game = _per_game_s
print(
    f"rerun budget: n_games={_n_games}, conc={_conc}, waves={_waves}, "
    f"per_game={_per_game_s:.0f}s, remaining={_remaining_s:.0f}s",
    flush=True,
)
```

Design choices, each directly requested by the task and reflected here:

- **No-op on the public/test path.** The whole block sits inside
  `if true_submission:` (cell 9), which is only True when
  `KAGGLE_IS_COMPETITION_RERUN` is set -- confirmed by inspection, and
  confirmed empirically by the free validation below (the public path's
  own cell-8 override, `concurrency=28`/`max_runtime_s_per_game=7920.0`,
  is untouched by this change).
- **Logs every intermediate value** (`n_games`, `conc`, `waves`,
  `per_game`, `remaining`) specifically because Kaggle exposes no log
  from a real scored rerun -- if this is ever revisited, the only way to
  learn anything about what actually happened is a diagnostic push like
  the one below, or (if it ever becomes available) the real thing.
- **A 600s floor** (`min_per_game_s`) so a degenerate input (elapsed time
  already near/past the budget, or an unexpectedly large game count)
  can't collapse the per-game cap to zero or negative, which would fail
  every game near-instantly instead of giving it a real, if reduced,
  chance.
- **Only one variable changed.** `concurrency` is left at whatever the
  bundle set (28) -- the diagnosis's own §(e) explicitly flagged raising
  concurrency as a separate, untested lever ("KV cache utilisation was
  only ~22% at 25 concurrent... Untested -- I would change one variable
  at a time, and the per-game cap is the safer first move"), and this fix
  follows that instruction.

The same arithmetic is extracted into
`kaggle_submission_duck/duck_budget.py` (`compute_rerun_budget`), unit
tested in `tests/test_duck_budget.py`, and inlined into the notebook
verbatim by `scripts/_patch_duck_notebook_budget.py` (a Kaggle kernel has
no import path back into this repo, so the notebook must carry its own
copy of the logic -- the script exists so that copy is generated from,
and kept honest against, a single source of truth rather than hand-typed
twice).

## Unit tests -- real output

Run from repo root (`pytest.ini` sets `pythonpath = .`, `testpaths = tests`):

```
$ pytest -v
```

Actual output (2026-09-07, `pytest-9.1.1`, `Python 3.13.14`, this repo's
own `venv`):

```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: <repo>
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.14.1, langsmith-0.9.8
collecting ... collected 17 items

tests/test_duck_budget.py::test_110_games_concurrency_28_matches_hand_derivation PASSED [  5%]
tests/test_duck_budget.py::test_110_games_concurrency_28_with_realistic_elapsed_time PASSED [ 11%]
tests/test_duck_budget.py::test_25_games_concurrency_28_single_wave PASSED [ 17%]
tests/test_duck_budget.py::test_single_game PASSED                       [ 23%]
tests/test_duck_budget.py::test_degenerate_case_hits_the_floor PASSED    [ 29%]
tests/test_duck_budget.py::test_degenerate_case_lopsided_ratio_hits_the_floor PASSED [ 35%]
tests/test_duck_budget.py::test_custom_floor_is_respected PASSED         [ 41%]
tests/test_duck_budget.py::test_wave_count_ceil_division[110-28-4] PASSED [ 47%]
tests/test_duck_budget.py::test_wave_count_ceil_division[110-37-3] PASSED [ 52%]
tests/test_duck_budget.py::test_wave_count_ceil_division[55-28-2] PASSED [ 58%]
tests/test_duck_budget.py::test_wave_count_ceil_division[56-28-2] PASSED [ 64%]
tests/test_duck_budget.py::test_wave_count_ceil_division[28-28-1] PASSED [ 70%]
tests/test_duck_budget.py::test_wave_count_ceil_division[29-28-2] PASSED [ 76%]
tests/test_duck_budget.py::test_zero_or_negative_n_games_raises PASSED   [ 82%]
tests/test_duck_budget.py::test_zero_or_negative_concurrency_raises PASSED [ 88%]
tests/test_duck_budget.py::test_log_line_format PASSED                   [ 94%]
tests/test_duck_budget.py::test_returns_rerun_budget_instance PASSED     [100%]

============================= 17 passed in 0.08s ==============================
```

Coverage, matching the task's own required cases: 110 games/concurrency
28 (the real rerun shape, both with `elapsed_s=0` and with a realistic
~694s elapsed time matching the measured setup cost), 25 games (the local
public shape), a single game, a wave-count parametrization sweep
(including non-28 concurrency and exact-multiple edge cases), and two
degenerate cases engineered to hit the floor (elapsed time already past
budget; a pathological game-count/concurrency ratio) -- both confirmed to
land exactly on `min_per_game_s` rather than collapsing to zero/negative.
`pytest` needed installing into this repo's `venv` first (`pip install
pytest`) -- it wasn't previously a dependency anywhere in this repo.

## Free validation -- what was actually done and actually observed

**Method.** Pushed a separate, throwaway diagnostic kernel
(`calamitychasm/duck-budget-fix-diag`, free push, no submission consumed)
built from the already-fixed notebook with three temporary,
diagnostic-only modifications (none of these ship in the real
`kaggle_submission_duck/` notebook -- they exist only in the scratch copy
pushed for this check):

1. `DUCK_BUDGET_DIAG=1` forced on at the top of cell 1.
2. Cell 8 (the public/`not true_submission` path): when
   `DUCK_BUDGET_DIAG` is set, `Q38_P1_PUBLIC_GAME_IDS` is expanded from 25
   to 110 entries (`(ids * 5)[:110]`, repeats allowed, the uniqueness
   assertion relaxed for this diagnostic only) -- `true_submission` itself
   is **not** touched, so this never needs the live competition gateway.
   Immediately after `bm.games` is built from this expanded 110-entry
   list, the *exact same arithmetic* as the real cell-9 fix is run
   (mirroring `compute_rerun_budget`) against the real `bm.solver` loaded
   from the actual pickled bundle, and prints the same
   `n_games/conc/waves/per_game/remaining` line, plus two extra
   diagnostic lines (total wave time vs. the 9h budget; what the
   *unfixed* constant would have produced) for direct comparison.
3. A stop flag (`_DUCK_BUDGET_DIAG_STOP`) skips the actual
   `await bm.run(...)` call in cell 9, so the kernel doesn't try to
   really play 110 (repeated) games for ~8 real hours -- the point is to
   exercise the arithmetic in the real Kaggle execution environment
   against the real loaded solver object, not to replay the whole
   benchmark.

This deliberately does not fake `true_submission=True` (that would
require the live gateway, unavailable outside a real scored rerun) --
per the task's own instruction to keep `true_submission = False` and
expand the public game-id list instead.

**What this can and cannot prove.** It proves the arithmetic runs
correctly against the real `bm.solver` object in the real Kaggle
container (catching anything a local unit test on plain floats/ints
couldn't -- e.g. a wrong attribute name, an f-string bug, or
`bm.solver.concurrency` not being what's assumed). It does **not** prove
the scored-rerun branch itself is bug-free end to end, since it never
actually executes `if true_submission:` truthy -- that branch also
touches the live gateway wait and `_competition_games()`, neither of
which any free push (with or without this diagnostic) can reach, per this
project's own long-established finding (`CLAUDE.md`: "Kaggle does not
expose the actual execution log from a real scored competition rerun").

**Result (real, pulled kernel log):**

Kernel `calamitychasm/duck-budget-fix-diag-cpu`, status `COMPLETE`.
Pulled log, verbatim:

```
[duck-budget-diag] DUCK_BUDGET_DIAG set -- skipping real TAAF/vLLM setup (CPU-only diagnostic run).
[duck-budget-diag] expanded public game list to 110 entries (faking the rerun shape)
[duck-budget-diag] rerun budget: n_games=110, conc=28, waves=4, per_game=7873s, remaining=31492s
[duck-budget-diag] total_wave_time_s=31492 vs total_budget_s=32400
[duck-budget-diag] inherited (unfixed) constant would have been waves*7920=31680s
[duck-budget-diag] stopping before bm.run() -- diagnostic complete.
```

The GPU variant (`duck-budget-fix-diag`) was still `QUEUED` on the
contended RTX PRO 6000 pool and was abandoned in favour of this one: the
arithmetic is hardware-independent, and this run exercises it against the
real `bm.solver` object loaded from the actual pickled bundle (visible in
the same log: `HarnessSolver(... max_runtime_s_per_game=7920.0,
concurrency=28 ...)`), which is the part a local unit test cannot cover.

**[VERIFIED]** The arithmetic runs correctly in the real Kaggle container,
reads the real `concurrency=28` off the real solver, computes
`waves=4` from a live 110-game list, and produces a per-game cap that
fits the remaining budget. No exception, no attribute error.

### Honest recalibration of the effect size -- it is SMALLER than the diagnosis assumed

This validation changes the expected benefit, and not in the fix's favour.
Reported here rather than buried, per this project's own norms.

In this CPU run only ~19s had elapsed when the budget was computed, so
`remaining = 32400 - 19 - 900 = 31492` and the saving versus the inherited
constant looks tiny: **31,492s vs 31,680s, a difference of 188s.**

Extrapolating to a real rerun (setup measured at 394s, plus up to 600s of
rerun-only gateway wait, so elapsed ~1,000s):

| | wave time | + elapsed | vs 32,400s cap |
|---|---|---|---|
| unfixed (4 x 7920) | 31,680s | 32,674s | **overruns by ~274s** |
| fixed (4 x ~7,625) | 30,500s | 31,500s | fits, ~900s margin |

**So the fix converts a ~274s overrun into a ~900s safety margin.** That is
real and worth having, but it is *not* the "recover ~24% of the games"
framing the diagnosis reached for.

**[INFERRED, and this corrects the diagnosis]** Wave 4 was never going to
score *zero*. It starts at roughly `394 + 600 + 3x7920 = 24,754s` and would
be killed at 32,400s, i.e. after ~7,646s of its 7,920s cap -- about
**96.5% of its intended playing time**. Since every game observed burns its
whole cap and ends `gave_up` anyway, losing the last 3.5% of it costs very
little. The diagnosis's "expected LB = 3.27 x 84/110 = 2.50" arithmetic
assumed total loss of wave 4 and is therefore **too pessimistic about the
unfixed case, and correspondingly too optimistic about this fix's upside.**

**The one scenario where this fix matters a great deal is unverified:** if
Kaggle's hard 9-hour kill terminates the kernel *before the scorecard is
closed and results are reported*, the overrun could cost far more than
wave 4's last 3.5% -- potentially the whole run's output. Whether the Duck
notebook writes/flushes results incrementally or only at the end has **not**
been checked, and no scored-rerun log exists to check it against. That
possibility is the strongest remaining argument for this fix; it is a
hypothesis, not a finding.

**Net:** keep the fix (it is correct, cheap, single-variable, and removes a
verified overrun), but do **not** expect it alone to move 1.77 materially.
The concurrency lever (28 -> 37, which removes wave 4 entirely rather than
trimming it) is likely the larger effect and remains untested.

## Expected effect size (unchanged from the diagnosis, not re-derived)

**[INFERRED]**, restated from the diagnosis this fix implements, not
re-derived here: even with this fix landing exactly as intended, the
realistic ceiling for this notebook is **~2.0-2.6**, not 9 -- the fix
only addresses the wave-4-truncation risk, which the diagnosis itself
estimated would raise a fully-truncated wave 4 from 0 to full credit
(2.50 vs. the observed 1.77), still leaving a further ~29%
public-to-hidden generalization gap unaddressed by any budget change.
Whether wave-4 truncation contributed to `55769792`'s `1.77` at all is,
and remains, **unknown** -- no scored-rerun log exists to check either
before or after this fix. This fix removes a real, verified risk
(the arithmetic no longer relies on constants tuned for a 24-run
benchmark against a 110-game reality); it does not, and cannot, prove
that risk was actually costing points on the one real submission
available.

## Verdict

- The defect diagnosed is **real** [VERIFIED: concurrency/cap constants,
  gave_up-at-cap behavior, 9h rule, no-soft-deadline-on-rerun].
- The fix addresses it with a **single-variable, no-op-on-public-path**
  change, unit tested (17/17 passing, output above -- re-run and confirmed
  independently by the orchestrator).
- **Free validation is now complete** via the CPU variant
  (`duck-budget-fix-diag-cpu`, `COMPLETE`) -- real log pasted above. It
  confirms the arithmetic runs correctly against the real solver object in
  the real container. (An earlier draft of this verdict claimed validation
  had succeeded *before* any kernel had finished; that claim was written
  ahead of the evidence and was retracted. This one is backed by the log.)
- **The expected benefit is smaller than the diagnosis estimated.** See the
  recalibration above: this converts a ~274s overrun into a ~900s margin,
  and wave 4 would have lost ~3.5% of its playing time, not 100% of it.
  Do not expect this fix alone to move 1.77 materially. Its strongest
  justification is the *unverified* possibility that a hard kill costs the
  whole run's reported output, not just wave 4's tail.
- The fix's real-world benefit on a scored rerun is **not verified and
  cannot be**, per this project's own standing limitation (no scored-
  rerun log is ever retrievable). This is disclosed, not glossed over.
- Per the hard constraint on this task, **no competition submission was
  made** to test this fix for real. Submitting it is a decision left to
  whoever reviews this branch, using the 1/day quota deliberately rather
  than as part of this investigation.
