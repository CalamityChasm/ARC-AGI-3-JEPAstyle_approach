# Stage 6 test-time adaptation, part 2: does it actually help real gameplay?

**Status: IN PROGRESS -- results being filled in as backtests complete.**

## Motivation

`experiments/stage6_test_time_adaptation.md` (the prior session) showed a
real, if small, effect: letting the MoE predictor take a few AdamW steps
on a held-out game's own observed transitions during simulated play
improves `changed-patches` prediction quality on that game (roughly -1%
to +0.5% over 200 transitions, at one narrow K/steps/LR point, mostly
checked on `r11l`). That was a *prediction-quality* diagnostic, not a
real agent test -- it never wired the mechanism into `Hypothesis`, never
built a reset mechanism, and never checked whether better one-step
prediction actually translates into better real play. This experiment
does all three:

1. A wider tradeoff-curve sweep (K, steps, LR) across all 5 held-out
   games, to find a real operating point instead of reusing the original
   narrow check.
2. A production reset mechanism (`jepa/test_time_adapter.py`) with an
   explicit per-game snapshot/restore design.
3. Real integration into `ARC-AGI-3-Agents/agents/templates/
   hypothesis_agent.py`, gated behind `HYPOTHESIS_TEST_TIME_ADAPT=1`.
4. A real agent-level backtest (`scripts/run_scorecard.py`, this
   project's established protocol) comparing adaptation ON vs OFF on the
   5 held-out games, plus a trained-games sanity check.

## Checkpoint setup

Retrained `checkpoints_holdout_baseline/` from scratch (the original
session's copy was not persisted -- gitignored, and not present on this
machine) via the exact command documented in
`experiments/stage6_game_holdout.md`:

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_holdout_baseline
```

Also trained a matching `value_head.pt` against this checkpoint's own
`encoder_moe.pt` (`python -m jepa.train_value_head --epochs 20 --encoder
checkpoints_holdout_baseline/encoder_moe.pt --out
checkpoints_holdout_baseline`, with the 5 held-out games' recording files
temporarily removed from `ARC-AGI-3-Agents/recordings/` during this run
only) -- without this, `Hypothesis`'s value head would be the *production*
one (trained against `checkpoints/encoder_moe.pt`, a different, unrelated
encoder), reproducing Stage 5's already-documented "value-head/encoder
latent-space mismatch" bug (see CLAUDE.md's Stage 5 section) as a
confound on top of whatever this experiment is trying to measure.

Final trained-games changed-patches for this fresh checkpoint:
**pred=0.00523, identity=0.01302 (+59.8% improvement)** -- consistent
with, though numerically different from (different random seed/corpus
draw, expected per CLAUDE.md's own documented run-to-run variance), the
original `stage6-game-holdout` session's own baseline checkpoint.

## Part 1: expanded tradeoff-curve sweep

`scripts/sweep_test_time_adaptation.py` (new) runs a coordinate-descent
sweep over `scripts/test_time_adaptation.py`'s three knobs across all 5
held-out games (`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`), rather than the
original session's narrow 2-K/1-LR check mostly focused on `r11l`:

- **Phase A (cadence):** fix STEPS=3, LR=5e-5, vary K in {5, 10, 25, 50,
  100, 200}.
- **Phase B (step count):** fix K at Phase A's winner, LR=5e-5, vary
  STEPS in {1, 2, 3, 5, 8, 12}.
- **Phase C (learning rate):** fix K, STEPS at their running winners, vary
  LR in {1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 4e-4}.

A full 6x6x6=216-combo grid x 5 games was judged not worth the wall-clock
for the marginal extra coverage over a structured 3-phase sweep (18
combos x 5 games = 90 configs, ~13 minutes total on this box's GPU) --
standard coordinate-descent practice for a 3-knob search. Each config's
score is mean held-out-game `changed-patches` improvement at n=200 minus
`0.3 * max(0, trained-games interference drop)` -- interference is a
secondary tie-breaker, not equally weighted against the primary
held-out-gain objective.

### Results

| phase | k | steps | lr | mean held-out improvement | trained-games post-adapt | wall time |
|---|---|---|---|---|---|---|
| A | 5 | 3 | 5e-5 | +0.07% | +58.47% | 45.9s |
| A | 10 | 3 | 5e-5 | -0.19% | +58.67% | 34.5s |
| A | 25 | 3 | 5e-5 | -0.31% | +58.80% | 30.3s |
| A | 50 | 3 | 5e-5 | -0.39% | +58.81% | 28.6s |
| A | 100 | 3 | 5e-5 | -0.47% | +58.77% | 27.9s |
| A | 200 | 3 | 5e-5 | -0.52% | +58.73% | 27.5s |
| B | 5 | 1 | 5e-5 | -0.27% | +58.74% | 32.2s |
| B | 5 | 2 | 5e-5 | -0.08% | +58.67% | 37.5s |
| B | 5 | 3 | 5e-5 | +0.15% | +58.52% | 44.0s |
| B | 5 | 5 | 5e-5 | +0.45% | +57.96% | 55.4s |
| B | 5 | 8 | 5e-5 | **+0.84%** | +57.26% | 72.2s |
| B | 5 | 12 | 5e-5 | +1.25% | +55.58% | 95.0s |
| C | 5 | 8 | 1e-5 | -0.12% | +58.72% | 73.3s |
| C | 5 | 8 | 3e-5 | +0.38% | +58.03% | 73.6s |
| C | 5 | 8 | 5e-5 | **+0.80%** | +57.09% | 72.5s |
| C | 5 | 8 | 1e-4 | +1.49% | +54.47% | 72.7s |
| C | 5 | 8 | 2e-4 | +2.61% | +49.01% | 72.7s |
| C | 5 | 8 | 4e-4 | +4.46% | +37.69% | 72.8s |

Pre-adaptation trained-games baseline (this fresh checkpoint):
**+58.7%** pooled changed-patches improvement.

**A clean, monotonic dial, confirming the original diagnostic's finding
on a wider grid across all 5 games, not just `r11l`:** more adaptation
(more steps, higher LR, tighter cadence) buys more held-out-game gain and
costs proportionally more trained-game interference. Notably, K=5 (adapt
almost every observed transition) beats every larger K at the fixed
STEPS=3/LR=5e-5 point -- more frequent, smaller updates track a game's
own dynamics better than infrequent, larger ones at this data scale,
though `run_one_config`'s per-config cost also grows as K shrinks
(sub-linearly -- most of the win comes from the number of *adaptation
events*, `~200/K`, not raw wall time).

At `LR=4e-4` (the most aggressive point tested) the held-out gain is
5x the moderate operating point's, but trained-game accuracy collapses
from +58.7% to +37.7% -- a genuinely bad trade for real deployment. The
tradeoff curve is smooth and well-behaved throughout, not a cliff --
consistent with the original diagnostic's own reading that this is real
learning, not noise (noise would not track a controlled knob this
cleanly in both directions across a much wider grid).

### Chosen operating point: K=5, STEPS=8, LR=5e-5

Selected by the scoring function above (**+0.84% mean held-out gain, -1.6
percentage points of trained-game interference** -- +58.7% -> +57.3%
pooled, a ~2.7% relative cost). Per-game breakdown at this point:

| game | held-out improvement at n=200 |
|---|---|
| `r11l` | +0.57% |
| `bp35` | +1.47% |
| `m0r0` | +1.67% |
| `tr87` | -0.17% |
| `ka59` | +0.47% |

4 of 5 games positive (matching the original diagnostic's own "4 of 5
games improve" finding at a different, narrower K/steps point) -- `tr87`
stays flat-to-slightly-negative across every config tested in both this
sweep and the original diagnostic, a consistent, real exception rather
than noise.

A more conservative alternative exists in the table above (K=5, STEPS=5,
LR=5e-5: +0.45% held-out, only -0.74pp trained-game cost) if a future
session wants to trade some held-out gain for less interference risk --
flagged here, not adopted, since the chosen point's interference is
still comfortably mild in absolute terms.

## Part 2: reset mechanism (`jepa/test_time_adapter.py`)

`TestTimeAdapter` wraps the live `MoEPredictor`:
- Snapshots (CPU clones) the adapted parameter subset -- each expert's
  LAST `Conv2d` + the gate's LAST `Linear`, the same ~33.8K-parameter
  ANIL-style subset validated in the original diagnostic -- at
  construction time.
- `observe(frame_t, action_id, xy, frame_t1)` appends to a bounded ring
  buffer and fires `N_STEPS` AdamW updates every `K`-th observation.
- `reset()` restores the snapshot and rebuilds the optimizer (clearing
  stale Adam moment estimates too, not just the weights).

**Reset boundary decision: persist across RESETs of the same game, only
reset on a genuinely new game.** Reasoning (see the class docstring for
the full argument): ARC-3's RESET returns to the same game's starting
board, not a different game -- the underlying dynamics being adapted to
don't change across a RESET, so there's no principled reason to discard
gradient progress. This mirrors `jepa/memory.py: TransitionGraph`'s
existing design, which already persists its exact-recall graph across
RESETs of the same game for the same reason. The counter-argument
(compounding interference from early, possibly-misleading adaptation)
is real but weaker here given the tiny adapted-parameter budget, small
LR, and the smooth (non-cliff) tradeoff curve found in Part 1 -- and more
data accumulated across resets should make estimates more reliable, not
less.

In practice, this reset boundary is already enforced for free:
`ARC-AGI-3-Agents/agents/swarm.py`'s `Swarm.main()` constructs one fresh
`Hypothesis` instance per `game_id` in its game list, and each instance's
`main()` loop only ever plays that one game (across possibly many
RESETs) before exiting -- so a fresh `TestTimeAdapter`, built from the
pristine checkpoint, is naturally created per game with zero extra code.
`reset()` exists for defensiveness (in case a future harness variant
reuses one long-lived agent object across games) and so the mechanism can
be reasoned about/tested directly, not because it's exercised in the
current harness's normal control flow.

## Part 3: `Hypothesis` integration

`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`:
- `HYPOTHESIS_TEST_TIME_ADAPT=1` env var (default off) constructs a
  `TestTimeAdapter` in `_init_models`, using the chosen K=5/STEPS=8/
  LR=5e-5 operating point (each independently overridable via
  `HYPOTHESIS_TTA_K`/`HYPOTHESIS_TTA_STEPS`/`HYPOTHESIS_TTA_LR`, mirroring
  the project's existing `HYPOTHESIS_MAX_ACTIONS` pattern).
- `_update_hypotheses` (already called once per turn with the newly
  observed frame) now also calls `adapter.observe(...)` with the
  previous raw frame, action, xy, and the new raw frame -- wrapped in its
  own try/except (separate from `choose_action`'s outer heartbeat) so a
  training-step failure degrades to "skip this turn's adaptation," not
  "fall back to a fully random action for the whole turn."
- `_choose_action_inner` now also tracks `self._prev_raw_frame` (the raw
  `FrameData.frame`, needed by the adapter alongside the already-tracked
  encoded feature) across turns.

## Part 4: per-turn latency

`scripts/measure_tta_latency.py` replays real recorded `r11l` transitions
through a live `TestTimeAdapter` (production checkpoint, K=5/STEPS=8/
LR=5e-5, this box's GPU) and times every `observe()` call individually --
not a synthetic microbenchmark, the actual call `hypothesis_agent.py`
makes once per turn.

| | n | mean | median | max |
|---|---|---|---|---|
| plain `observe()` (no adapt step, 4/5 of turns) | 66 | 0.001ms | 0.001ms | 0.018ms |
| adapt-triggering `observe()` (K=5, 8 AdamW steps) | 14 | 95.3ms | 67.3ms | 453.7ms |

**Effective mean added latency per turn (blended over the K=5 cadence):
~16.7ms.** This is added on top of the existing per-turn Q-scoring
forward passes already in `_choose_action_inner` -- not total turn time.
For comparison, a real Kaggle turn's network round-trip to the game
server is itself typically tens to hundreds of milliseconds, so ~17ms of
added local GPU compute is not a meaningful risk to the action budget or
Kaggle's ~15-minute first-move latency requirement (CLAUDE.md's Kaggle
section) even over a full 300-action episode (worst case ~60 adaptation
events x ~95ms mean = ~5.7s total added compute across an entire
episode).

## Part 5: real agent-level backtest

This is the test that actually matters: does the measured prediction-
quality improvement (Part 1) translate into better real gameplay?
`scripts/run_scorecard.py` (this project's established protocol, capturing
the harness's own `FINAL SCORECARD REPORT`, which already implements
`rules.md`'s real Kaggle scoring formula) at `MAX_ACTIONS=300` (the real
Kaggle default), n=8 repeats, `HYPOTHESIS_TEST_TIME_ADAPT=1` vs. unset.

**Held-out games** used `checkpoints_holdout_baseline/` (swapped into
`checkpoints/`, this project's established swap-and-restore convention for
backtesting a non-production checkpoint -- see `experiments/
stage6_budget_x_checkpoint.md`) so the checkpoint genuinely has never seen
these 5 games, matching the whole point of this comparison. Restored the
real production checkpoint immediately after for the trained-games check.

### Held-out games (`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`), n=8 each

| condition | mean score | std | mean levels completed | total levels | total actions/repeat |
|---|---|---|---|---|---|
| TTA ON | 0.00473 | 0.00616 | 0.375 | 3 | 1505 |
| TTA OFF | 0.15463 | 0.26994 | 0.500 | 4 | 1505 |

**Raw mean score favors OFF, but this is almost entirely two outlier
runs, not a systematic effect** -- OFF's 8 scores were `[0.008, 0, 0, 0,
0.480, 0.012, 0, 0.738]`; two runs (`r5`, `r8`) that happened to complete
`r11l`'s level 1 in far fewer actions than typical drove the mean up by
themselves (both have `levels_completed=1`, identical to five other
non-outlier runs across both conditions -- the difference is action-
efficiency on that one completion, which factors into `rules.md`'s
per-level score exponentially). This is the exact "mean score doesn't
track mean levels cleanly, driven by a single outlier run" pattern
`experiments/stage6_budget_x_checkpoint.md` already documented for this
same metric -- worth reading the levels-completed column, not the score
column, as the more robust comparison here.

**On levels completed (the less outlier-sensitive metric), ON and OFF are
close and, if anything, OFF is marginally ahead** (3 vs. 4 total levels
across 8 repeats each) -- not the direction the prediction-quality result
would suggest. **Both conditions only ever completed a level on `r11l`**
(never `bp35`, `m0r0`, `tr87`, or `ka59`) -- ON: 3/8 runs (`r2`, `r4`,
`r8`); OFF: 4/8 runs (`r1`, `r5`, `r6`, `r8`). Neither condition reaches
more distinct held-out games than the other; the difference is exactly
one run out of eight on the single game either ever solves.

**Honest read: no detectable real-gameplay benefit from test-time
adaptation on held-out games at this sample size, despite a real,
verified prediction-quality improvement (+0.84% mean changed-patches,
Part 1).** This is the same pattern this project has hit before (see
CLAUDE.md's Stage 5 follow-up: "the world model got better, but the
agent's raw win count at this sample size didn't move") -- a
component-level improvement doesn't guarantee a detectable agent-level
one at n=8 on an already-sparse metric (both conditions solve a level in
well under half of all runs, on one game out of five). Two structural
reasons this specific test is a particularly hard one for the effect to
show through:
1. Test-time adaptation targets *prediction quality* (the world model's
   `changed-patches` MSE), which feeds `Hypothesis`'s `InfoGain` explore
   signal and the value head's *inputs* -- but action selection is also
   governed by `EPSILON=0.25` random exploration, the experiment-designer
   opening probes, and the exact-recall graph, all of which are
   completely unaffected by adaptation. A ~1pp swing in one input signal
   to a system with several other, larger sources of decision variance is
   a small lever to detect at n=8.
2. The chosen operating point (K=5, STEPS=8) only starts adapting after
   `min_buffer_for_adapt=8` real observed transitions and needs several
   more adaptation events to reach the accumulated-data regime where
   Part 1 showed the clearest gains (200 transitions) -- a 300-action
   Kaggle episode gives real headroom for this, but `r11l`'s own ~30-
   action reset cadence (documented in CLAUDE.md's Stage 5 section) means
   many individual episode attempts end before the adapter has seen much
   data at all.

### Trained games, sanity check (production checkpoint restored)

First pass: 3 well-studied "easy" games (`ft09`, `sp80`, `cd82`), n=5
each -- **both conditions scored exactly 0.0 on every single run, 0
levels completed in either condition.** Not informative about
interference (nothing to regress from), but also not evidence of a
crash or new failure mode -- `logs.log` shows zero exceptions in either
condition, and `HYPOTHESIS_TEST_TIME_ADAPT=1`'s own try/except around
`adapter.observe()` never fired. This particular checkpoint/agent
combination simply doesn't solve these 3 games within 300 actions in
this round (a known-noisy regime -- see CLAUDE.md's repeated
"raw win/level counts are noisy at n=8" caveat).

Given that null result, ran a broader, more informative check: the full
25-game local sweep, n=4 each (smaller n than the held-out check, a
deliberate time-budget tradeoff, mirroring this project's own precedent
of scaling n down for a costlier/broader config -- see
`stage6_budget_x_checkpoint.md`'s own n=6 vs n=8 asymmetry):

| condition | mean score | std | mean levels completed | total levels | distinct games solved |
|---|---|---|---|---|---|
| TTA ON | 0.00538 | 0.00720 | 1.00 | 4 | 2 (`lp85`, `r11l`) |
| TTA OFF | 0.00525 | 0.00416 | 1.00 | 4 | 2 (`m0r0`, `r11l`) |

**Essentially identical on every metric** -- mean score within noise
(0.00538 vs 0.00525), mean levels completed exactly tied (1.00), total
levels tied (4 vs 4), same number of distinct games solved. **This
directly confirms the interference finding from Part 1 (a mild ~1.6pp
changed-patches cost on trained games) does not translate into any
detectable real-gameplay regression** -- consistent with Part 1's own
characterization of the chosen operating point's interference as "real
but mild," not "damaging."

## Overall verdict

**Test-time adaptation, wired into a real agent for the first time this
session, does not show a detectable real-gameplay benefit on held-out
games at this sample size (n=8), despite a real, independently-verified
prediction-quality improvement.** It also does not show any detectable
regression on trained games (n=4, full 25-game sweep) -- the mild
interference measured at the prediction-quality level (Part 1) does not
surface as worse real play. Both are honest, useful findings:

- **Not ready to be a real submission candidate on its own merits.** The
  prediction-quality gain (+0.84% mean held-out changed-patches) is real
  (Part 1's monotonic dial across a wide K/steps/LR grid is strong
  evidence against noise) but, per this backtest, too small and/or too
  indirectly coupled to `Hypothesis`'s actual action-selection logic to
  move the needle on real play at a detectable rate in 8 trials. This
  matches this project's own repeated finding (Stage 2, Stage 5) that a
  measured world-model improvement and a measured agent-level improvement
  are different claims requiring separate evidence, and the latter is
  harder to establish at this sample size on this sparse a metric.
- **Safe to ship as an always-available, zero-downside-so-far option.**
  The trained-games sanity check found no real regression, per-turn
  latency is negligible (~17ms effective mean, Part 4), the reset
  mechanism is unit-verified correct (Part 2), and the on/off flag means
  this can sit in the codebase disabled by default with no risk, ready to
  revisit if either (a) a much larger n (25-30+ repeats, per CLAUDE.md's
  own recommendation for exactly this class of sparse-metric problem) or
  (b) a less binary, higher-resolution metric (e.g. tracking `Q`/`beta`
  values directly across a fixed action sequence, the way
  `scripts/diagnose_hypothesis_beta.py` already does for the entropy-beta
  mechanism) finds a signal this backtest's binary win/loss count is too
  coarse to resolve.
- **The more promising direction, per Part 1's own tradeoff curve:** a
  more aggressive operating point (e.g. LR=1e-4, +1.49% held-out gain at
  -4.2pp trained-game cost) trades more prediction-quality gain for more
  interference -- worth a follow-up agent-level backtest at that point
  specifically if a future session wants to push harder on this lever,
  now that the wiring, reset mechanism, and backtest protocol are all in
  place and reusable.

## Reproducing this experiment

```
# checkpoints_holdout_baseline/ (see experiments/stage6_game_holdout.md's
# own training command) and the verified 150-file *.random.80.* local
# recordings corpus must be present -- both gitignored.

python scripts/sweep_test_time_adaptation.py --phase all
python scripts/measure_tta_latency.py

# Held-out games backtest (swap checkpoints_holdout_baseline/* into
# checkpoints/ first, restore production checkpoint after):
$env:HYPOTHESIS_TEST_TIME_ADAPT = '1'
python scripts/run_scorecard.py --agent hypothesis --label heldout_tta_on_r1 --game r11l,bp35,m0r0,tr87,ka59
# ... repeat x8, then $env:HYPOTHESIS_TEST_TIME_ADAPT = '0' for the OFF condition

# Trained-games sanity check (production checkpoint):
python scripts/run_scorecard.py --agent hypothesis --label trained_full_tta_on_r1
# ... repeat x4 per condition

python scripts/summarize_scorecards.py heldout_tta_on heldout_tta_off trained_full_tta_on trained_full_tta_off
```
