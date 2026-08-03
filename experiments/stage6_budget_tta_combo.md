# Stage 6 experiment: does MAX_ACTIONS=900 compound with test-time adaptation?

**Status: COMPLETE (local backtest only, n=8 per condition). Preliminary
read, not a validated result -- see the honest-limits section below
before acting on this.**

## Motivation

Two separate Stage 6 levers each showed *some* positive signal on their
own, on different evaluation setups, but were never combined:

- `MAX_ACTIONS=900` (vs. the real Kaggle default of `300`) was shown in
  `experiments/stage6_budget_x_checkpoint.md` to be a reliability lever
  across every checkpoint tested -- more mean levels completed, fewer
  zero-completion runs -- though not a clean mean-score lever.
- Test-time adaptation (`experiments/stage6_test_time_adaptation_agent.md`)
  showed a real, verified *prediction-quality* improvement on held-out
  games (a monotonic dial across a wide K/steps/LR sweep), but its own
  agent-level backtest at `MAX_ACTIONS=300` (n=8) did not show a clearly
  detectable gameplay benefit -- plausibly because 300 actions simply
  isn't enough budget for the adaptation to accumulate meaningfully within
  one episode before the episode ends.

The natural next question: does giving test-time adaptation more budget
to work with (via `MAX_ACTIONS=900`) let its prediction-quality gain
actually show up in real play? This experiment runs the one untested
cell in the 2x2 matrix.

## Setup

- Checkpoint: `checkpoints_holdout_baseline/` (the `stage6-game-holdout`
  fold-1 checkpoint, encoder + MoE predictor + value head all trained
  with `r11l`, `bp35`, `m0r0`, `tr87`, `ka59` excluded -- copied byte-
  identical from the `stage6-test-time-adaptation-agent` worktree, so the
  two conditions below that reuse that experiment's own numbers are a
  fair, same-checkpoint comparison, not just "close enough").
- Games: the same 5 held-out games, all in one `scripts/run_scorecard.py`
  call per repeat (`--game r11l,bp35,m0r0,tr87,ka59`).
- `Hypothesis.MAX_ACTIONS` was ported to the `HYPOTHESIS_MAX_ACTIONS`
  env-var-override pattern already established on
  `stage6-budget-x-checkpoint` (default kept at `300`, unchanged
  production behavior; this backtest sets it explicitly per condition).
- `HYPOTHESIS_TEST_TIME_ADAPT=1` / unset controls test-time adaptation,
  at its already-chosen operating point (K=5, STEPS=8, LR=5e-5 --
  unchanged from `stage6_test_time_adaptation_agent.md`'s own sweep).
- n=8 repeats per condition, matching this project's established default
  for a quick preliminary read.
- Two of the four conditions are **reused directly**, not rerun: same
  checkpoint, same 5 games, same `MAX_ACTIONS=300` default, same n=8 --
  from `experiments/stage6_test_time_adaptation_agent.md`'s Part 5. Only
  the two `MAX_ACTIONS=900` conditions were run fresh for this
  experiment.

## Results

| condition | n | mean score | std | mean levels completed | total levels | distinct games solved |
|---|---|---|---|---|---|---|
| baseline (300, TTA off) | 8 | 0.15463 | 0.26994 | 0.500 | 4 | 1 (`r11l`) |
| `MAX_ACTIONS=900` alone | 8 | 0.00980 | 0.01272 | 0.750 | 6 | 1 (`r11l`) |
| TTA alone (300) | 8 | 0.00473 | 0.00616 | 0.375 | 3 | 1 (`r11l`) |
| **combo (900 + TTA)** | 8 | 0.06523 | 0.10133 | **1.000** | **8** | 1 (`r11l`) |

Raw per-run scores, in run order:

```
baseline (300, TTA off):   0.008, 0.000, 0.000, 0.000, 0.480, 0.012, 0.000, 0.738
                            (reused from stage6_test_time_adaptation_agent.md)
budget=900 alone:          0.01778, 0.04026, 0.00666, 0.00000, 0.00166, 0.00418, 0.00000, 0.00787
TTA alone (300):           (reused from stage6_test_time_adaptation_agent.md, see that doc)
combo (900 + TTA):         0.00204, 0.00572, 0.00879, 0.30306, 0.00133, 0.05108, 0.14699, 0.00284
```

Per-run levels completed:

```
budget=900 alone:  1, 1, 1, 0, 1, 1, 0, 1   (6/8 runs nonzero)
combo (900 + TTA): 1, 1, 1, 1, 1, 1, 1, 1   (8/8 runs nonzero)
```

Every held-out-game level completed across all four conditions, in every
run, was on `r11l` -- the same game every prior held-out-games backtest
in this project has solved almost exclusively (see
`stage6_test_time_adaptation_agent.md`'s own Part 5). No condition here
reached `bp35`, `m0r0`, `tr87`, or `ka59` even once.

## What this shows

**On the levels-completed metric -- the one this project has repeatedly
found more robust than raw score at this sample size -- there is a real,
monotonic-looking, non-additive pattern:**

- Baseline: 4/8 total levels (some runs zero-completion).
- Budget alone: 6/8 -- a real lift, consistent with
  `stage6_budget_x_checkpoint.md`'s finding that the budget bump is a
  reliability lever across every checkpoint tested.
- TTA alone (at the tight 300-action budget): 3/8 -- *slightly worse*
  than baseline, consistent with `stage6_test_time_adaptation_agent.md`'s
  own finding of "no detectable benefit, if anything marginally behind."
- **Combo: 8/8 -- every single one of the 8 repeats completed at least
  one level.** This is the first time in this project's entire Stage 6
  held-out-games backtesting history (across dozens of n=8 runs
  documented in CLAUDE.md and this experiments directory) that a
  condition has shown zero zero-completion runs on these 5 games.

Naive addition of the two individual effects (+2 levels from budget,
-1 level from TTA, relative to baseline's 4) would predict roughly 5.
The observed 8 is noticeably higher than that naive sum -- consistent
with a real interaction, not just "the budget effect dominates and TTA
is along for the ride." The mechanistic story fits: TTA's own prior
negative result was hypothesized to be budget-starved (not enough
observed transitions within a 300-action episode for the K=5/STEPS=8
adaptation cadence to accumulate before the episode ends) -- a 900-action
episode gives roughly 3x the raw transition count for the adapter to
learn from before the episode terminates, which is exactly the axis that
explanation predicts should matter.

**On mean score, the picture is muddier, for the same outlier-driven
reason this project has flagged repeatedly (`stage6_budget_x_checkpoint.md`,
`stage6_max_actions.md`).** The combo's mean score (0.06523) is higher
than both single-lever conditions but still well below the baseline's own
mean (0.15463) -- and the baseline's mean is itself known to be driven
almost entirely by two outlier fast completions (0.480 and 0.738 out of 8
runs), not a real central tendency (see
`stage6_test_time_adaptation_agent.md`'s own note on this exact baseline
data). The combo condition has its own outlier pair (0.30306 and 0.14699)
pulling its mean up; the other 6 of 8 runs score under 0.01. Score should
not be read as confirming or contradicting the levels-completed finding
here -- it is simply a noisier metric at this n, as this project has
established repeatedly.

**Breadth did not improve.** All four conditions, including the combo,
only ever solved `r11l` -- zero progress on the other 4 held-out games in
any condition. The combo's gain is entirely a *reliability* improvement
on the one game these agents already have some traction on, not a
breadth-of-generalization improvement. This matters for how the result
should be read: it is not evidence that test-time adaptation (even with
more budget) unlocks genuinely new games out of held-out-game
generalization's broader gap -- it looks more like "more budget, and TTA
riding on that budget, make an already-partially-solvable game more
reliably solvable."

## Honest limits -- read this before trusting the 8/8 result

**This is n=8 per condition -- a preliminary read, not a validated
result, by this project's own explicit standard.** This project's most
recent local-backtest history is a direct cautionary tale for exactly
this situation: the novelty-aware beta override showed an unambiguous,
every-metric win at n=8 (mean score, mean levels, total levels all
favored the change), and that result completely failed to replicate at
n=30 -- the levels-completed metric came back *exactly tied* and the
score-metric direction *reversed*. An n=8 result this clean deserves the
same skepticism, not more trust just because the effect size looks large
or the story has a plausible mechanism attached. A perfect 8/8 could
plausibly regress to something like 5-7/8 (still likely better than
budget-900-alone's 6/8, unless that also regresses) or could hold up --
this backtest alone cannot tell those apart.

Additional reasons for caution specific to this result:

1. **Every completion is on the same single game.** With only one game
   ever contributing any signal, this is effectively a repeated-measures
   comparison on one Bernoulli-like outcome, not a broad multi-game
   result -- exactly the kind of sparse, single-game-dominated metric
   this project has flagged as noisy before (see CLAUDE.md's Stage 2/5
   sections).
2. **The combo condition costs roughly 3x the wall-clock and compute of
   the baseline** (4505 actions/repeat vs. 1505, plus TTA's own per-turn
   adaptation overhead) -- even if the effect is real, it is not a free
   lever, and any recommendation to ship it should weigh that cost.
3. **No trained-games sanity check was run in this experiment** (out of
   scope, matching this project's practice of scaling scope down for a
   quick preliminary read) -- `stage6_test_time_adaptation_agent.md`
   already found TTA alone costs a mild ~1.6pp of trained-game prediction
   quality at its chosen operating point; whether that interference
   changes at `MAX_ACTIONS=900` (more adaptation events per episode) is
   untested here.

## Verdict

**A real, non-additive compounding effect on the levels-completed
reliability metric, not just "no better than either lever alone" and not
"nothing detectable."** The combo (8/8 zero-completion-free) is a
qualitatively different result from every single-lever condition tested
today or in prior Stage 6 backtests -- but it rests on n=8 with all
signal concentrated on one game, and this project's own novelty-aware
beta result is direct proof that an equally clean n=8 win can evaporate
at n=30. **Treat this as a promising lead worth a larger confirmatory
backtest (25-30 repeats, matching this project's own stated bar for
trusting a result at this sparsity) before it influences a real
submission decision** -- not as a validated finding on its own.

## Reproducing this backtest

```
# checkpoints_holdout_baseline/ (see experiments/stage6_game_holdout.md's
# training command, or copy the byte-identical copy already produced on
# stage6-test-time-adaptation-agent) must be present in checkpoints/ --
# gitignored, not committed.

$env:HYPOTHESIS_MAX_ACTIONS = '900'
$env:HYPOTHESIS_TEST_TIME_ADAPT = '0'
python scripts/run_scorecard.py --agent hypothesis --label budget900_tta_off_r1 --game r11l,bp35,m0r0,tr87,ka59
# ... repeat x8

$env:HYPOTHESIS_TEST_TIME_ADAPT = '1'
python scripts/run_scorecard.py --agent hypothesis --label budget900_tta_on_r1 --game r11l,bp35,m0r0,tr87,ka59
# ... repeat x8

python scripts/analyze_budget_tta_combo.py budget900_tta_off budget900_tta_on
```

The `MAX_ACTIONS=300` TTA-off and TTA-on conditions are documented
directly in `experiments/stage6_test_time_adaptation_agent.md`'s Part 5
and were not rerun here.
