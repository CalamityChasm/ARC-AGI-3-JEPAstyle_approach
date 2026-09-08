# Stage 6: TransitionGraph health check + tried_actions dead-end fix

**Scope:** production-scale checkpoints only (encoder_moe.pt/moe_predictor.pt/
value_head.pt, ~297K params), not the 85.2M-param scaled run -- per explicit
instruction, this investigation is deliberately kept separate from that
axis. Worth revisiting at scale later if this holds up.

## Why this was checked

Stage 3's plan.md-specified "exact visited-transition graph"
(`jepa/memory.py: TransitionGraph`) was built, unit-tested, and wired into
both `Memory` and `Hypothesis` (the current production agent) -- but had
never been directly measured for whether it's actually doing anything
useful in real play, only whether it's mechanically correct in isolation.

## Method: replay, not re-run

`scripts/diagnose_transition_graph_health.py` replays existing recorded
gameplay through a fresh `TransitionGraph`, in order, reproducing exactly
what the live agent's graph state would have been at every decision point
-- the graph's hashing is a pure, deterministic function of frame content,
so this is equivalent to instrumenting the live agent without needing to
re-run games. First pass used the already-existing winning-transition
harvest (`E:/jepa_overflow/winning_harvest/recordings/`, 26 files,
`MAX_ACTIONS=2000`, 49,448 decisions across 25 games) -- no new data
generation needed for the initial diagnosis.

Three things measured per decision: **state-revisit rate** (has this exact
frame hash been seen before -- if near-zero, the graph structurally can't
help), **exact-repeat rate** (is this the *exact same* (state, action, xy)
already tried from here -- something `tried_actions()`/`lookup()` could
answer), and **useful-recall rate** (did `best_known_action` at a revisited
state ever point to something with `levels_completed_delta > 0`).

## Finding 1 (ruled out): countdown bars don't poison exact-hash matching

Hypothesized that games with a depleting counter baked into the visible
frame (found earlier this session, `jepa/countdown_detector.py`) would
have crippled revisit rates, since every frame differs by at least the
counter. **Tested directly, wrong**: mean revisit% for games WITH a
countdown bar (66.0%, 19 games) was actually *lower* than games WITHOUT one
(72.8%, 6 games) -- the wrong direction for the hypothesis, and the gap is
small. Not the bottleneck.

## Finding 2 (real, but not what it looks like): useful-recall is 0.00% everywhere

Every one of the 25 games showed exactly 0.00% useful-recall. Traced to
signal sparsity, not a broken mechanism: this harvest only ever recorded 4
level-up events total across all 25 games (see
`experiments/stage6_winning_transition_harvest.md`), and a level-up state
is, almost by construction, a late-episode state a budget-limited
exploration run reaches once if at all -- there's rarely a second visit to
recall against. This is the same "value signal is ~98% zero-target"
sparsity problem Stage 5 already documented for the value head, showing up
again here. Not something this component can fix on its own.

## Finding 3 (the real bug): Hypothesis never inherited Memory's dead-end avoidance

Direct code comparison: `Memory` (`ARC-AGI-3-Agents/agents/templates/
memory_agent.py`) calls `self.graph.tried_actions(state_key)` before
falling back to exploration, to avoid re-trying an `(action, xy)` already
known, from this exact state, to be unproductive -- "while untried ones
remain," per its own docstring. `Hypothesis`, built on top of `Memory` in
Stage 5 ("reused unchanged" per its own module docstring), only kept the
*win*-recall path (`best_known_action` gated on `delta > 0`) -- the
`tried_actions()` call was never carried over. `Hypothesis`'s full
candidate-scoring loop (`_score_action` over every `available` action) has
zero awareness of what it already tried from this exact state.

**Directly explains the measured pooled exact-repeat rate of 41%** across
the 25-game harvest -- more than a third of all actions taken were exact
repeats of something already known to do nothing, wastable budget with no
mechanism stopping it.

## Fix

`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`'s candidate loop:
compute `tried = self.graph.tried_actions(state_key)`, filter `available`
down to actions not yet tried as `(action_id, None)` from this state
*while untried ones remain* (falls back to the full set if everything's
been tried -- matches Memory's exact behavior, doesn't forbid a repeat when
there's genuinely nothing else to do). **ACTION6 is deliberately exempt**:
a single click doesn't exhaust its outcome space the way a simple action's
one possible effect does, so "tried at some xy" isn't "exhausted" the way
it is for a 0-5/7-id action.

## Verification (not assumed -- measured before and after)

Smoke-tested first (single game, normal 300-action budget, no crash, 1
level completed). Then re-ran the same 5 highest-original-exact-repeat
games at the same `MAX_ACTIONS=2000` budget with the fix in place, replayed
through the same diagnostic:

| game | exact-repeat % before | after |
|---|---|---|
| bp35 | 70.5% | 7.0% |
| ka59 | 64.1% | 12.5% |
| lf52 | 76.1% | 17.8% |
| sb26 | 85.0% | 14.9% |
| sc25 | 67.8% | 9.8% |
| pooled (these 5) | ~72.7% | 12.4% |

Consistent, large drop on every single game -- roughly 5-10x fewer wasted
repeats. Revisit rate also dropped alongside it (e.g. bp35: 80.3% -> 43.6%),
a corroborating signal that the agent is genuinely reaching more distinct
states rather than retreading the same ground, not just a measurement
artifact. `useful_recalls` stayed 0.00% in this small sample, as expected
-- this fix targets wasted repetition, not win-signal sparsity.

## Agent-level backtest: 5-fold, n=30 total -- the early-fold win evaporates, same pattern as novelty-aware-beta

Matched before/after comparison (`HYPOTHESIS_DEADEND_FILTER=0/1`, same
committed code, toggled via env var -- mirrors this project's existing
`FORCE_BETA`/`DIAG_MODE` pattern for controlled ablations), 5 independent
folds of 6 repeats each (25 games, `MAX_ACTIONS=300`, the normal
production budget), 750 runs per condition total.

**Real bug hit and fixed in the test harness itself before any of this
data is trustworthy**: `main.py` does `load_dotenv(dotenv_path=".env",
override=True)`, which silently clobbered a `RECORDINGS_DIR` passed via
subprocess env with the `.env` default -- fold 1's first attempt dumped
all 300 recordings into one shared folder instead of splitting them by
condition. Recovered safely (each game's before/after pair is created
back-to-back with nothing else interleaved, so sorting by file mtime
within each game exactly reconstructs the split) rather than discarded;
fixed `scripts/run_deadend_backtest_fold.py` to move each file into place
immediately after creation instead of relying on the env var for folds
2-5. Also hit one isolated transient failure (fold 5, `sk48` repeat 3
AFTER produced no recording at all) -- re-ran just that one game rather
than leave the sample asymmetric.

**Per-fold level-completion delta (after minus before):**

| fold | before | after | delta |
|---|---|---|---|
| 1 | 4 | 7 | +3 |
| 2 | 4 | 7 | +3 |
| 3 | 5 | 6 | +1 |
| 4 | 6 | 6 | 0 |
| 5 | 10 | 6 | **-4** |

**Final cumulative (n=30 repeats, 750 runs/condition):**

| metric | before | after |
|---|---|---|
| total levels completed | 29 | 32 (+3) |
| distinct games solved | 4 (ar25, cd82, ft09, r11l) | 5 (+lp85, solved in *every* fold under "after", never under "before") |
| avg actions to first completion | 136.6 | 170.8 (worse) |

**Verdict: the same "early folds look like a real win, then decay to
noise at a larger sample" pattern this project already hit with
novelty-aware beta (n=8 clean win on every metric, n=30 exactly tied) and
the game-id-ablation result (+64.9% single run, didn't reproduce on
reseeding).** The per-fold trend (+3, +3, +1, 0, -4) is a clean decay,
not a plateau -- folds 1-2 alone would have looked like a strong,
unambiguous win and would have been the wrong conclusion. Final
cumulative is a modest +3 levels (~10% relative) with a *worse*
average-actions-to-first-completion, not the clear win the early data
suggested.

**What this does and doesn't call into question.** The component-level
result (exact-repeat rate dropping 5-10x on every game tested, via direct
replay against real gameplay) is not in doubt -- that's a mechanical
measurement, not a noisy agent-level outcome, and stands on its own. What
remains genuinely unresolved is whether that mechanical improvement
reliably translates into completing more levels, at this sample size.
`lp85` being solved in every fold under "after" and never under "before"
is the one piece of evidence that didn't wash out -- worth noting, though
n=1 newly-reachable game isn't strong evidence by itself.

**Recommendation: keep the fix.** It's real, low-risk (ACTION6 exempted,
falls back to the full candidate set when nothing untried remains, matches
Memory's already-validated design), and there's no evidence it hurts --
worst case by this backtest is "no detectable agent-level effect yet,"
not a regression. Don't treat "meaningfully improves completions" as
validated, on the same standard this project has applied to every other
result that showed this exact decay pattern.

## What's still open

- **Win-signal sparsity is untouched.** The graph's win-recall path is
  correct but starved -- closing that gap needs denser win data (e.g. the
  teacher-policy/search-harvest approaches already tried elsewhere in this
  project), not a change to the graph itself.
- **Not yet tested at the 85.2M-param scaled-architecture size** -- kept
  out of scope for this pass per explicit instruction; worth revisiting
  once/if the scaled architecture's own trained-games underperformance
  (see `experiments/stage6_scaled_architecture_prep.md`) is understood.
