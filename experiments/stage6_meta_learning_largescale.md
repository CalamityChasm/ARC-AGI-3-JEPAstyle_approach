# Stage 6 meta-learning, large-scale follow-up: does the high-dose Reptile checkpoint beat plain TTA at real statistical power?

**Status: DONE. At n=30/condition (60 real scorecard runs, both conditions
run fresh in the same round), the high-dose meta-learning checkpoint and
the plain baseline checkpoint -- both under test-time adaptation -- are
statistically indistinguishable on every metric checked: identical total
levels completed (15 vs 15), identical solve rate on the only game either
ever solves (`r11l`, 15/30 = 50% both sides), and Mann-Whitney p-values
of 0.90-1.0 across per-run score, per-solved-run score, and solve-
efficiency. This is a fourth instance of this project's now-familiar
pattern: a real, dosage-confirmed representation-level improvement
(`experiments/stage6_meta_learning.md`'s high-dose ablation) does not
show up as a detectable agent-level advantage at practical sample sizes.**

## Motivation

`experiments/stage6_meta_learning.md` found a real, dosage-confirmed
representation-level result: a "high-dose" Reptile meta-training recipe
(3x more meta-updates/epoch, no epsilon annealing) produces a checkpoint
that, after test-time adaptation, shows a larger held-out-games
prediction-quality improvement (+0.98%/+1.28% simple-mean/pooled) than
plain test-time adaptation on a normally-trained baseline (+0.78%/+0.66%).
A preliminary n=8 agent-level backtest of the high-dose checkpoint came
back exactly tied with the already-published baseline+TTA-ON numbers on
levels completed (3 total, 0.375 mean, all on `r11l`) -- explicitly
flagged in that doc as inconclusive, not negative, since this project has
now repeatedly found n=8 too small to trust on this exact class of
question. Most pointedly: `experiments/stage6_novelty_aware_beta.md`'s
Part 3 found an n=8 win that led on every single metric and then
completely evaporated at n=30 (levels tied exactly, mean-score gap
reversed direction, Mann-Whitney p=0.86/0.59 on the two distributional
checks). This experiment gives the high-dose meta checkpoint the same
properly-powered treatment before treating its n=8 result as anything
more than a lead worth following up.

## Protocol

Matches `scripts/run_novelty_backtest_largescale.py`'s methodology as
closely as possible for direct comparability, adapted to swap full
checkpoint sets (not just an env-var toggle) between conditions:

- 5 held-out games: `r11l`, `bp35`, `m0r0`, `tr87`, `ka59`.
- `MAX_ACTIONS=300` (`Hypothesis`'s unmodified default).
- `HYPOTHESIS_TEST_TIME_ADAPT=1` on **both** conditions -- the question
  here is not "does TTA help" (already established), it's "does
  meta-training for post-adaptation performance beat plain TTA on a
  normally-trained checkpoint," so TTA must be on in both arms and the
  checkpoint is the only thing that differs. K=5/STEPS=8/LR=5e-5 (the
  validated `stage6-test-time-adaptation-agent` operating point,
  `Hypothesis`'s unmodified defaults).
- **Condition "metahd":** `checkpoints_meta_fold1_highdose/` (the exact
  checkpoint `experiments/stage6_meta_learning.md`'s high-dose ablation
  produced -- `--meta-iters-per-epoch 60 --no-epsilon-anneal`, fold-1
  exclude-games recipe). Already ships a `value_head.pt` trained against
  its own `encoder_moe.pt` (per that doc's "Preliminary agent-level
  backtest" section).
- **Condition "baseline":** `checkpoints_holdout_baseline/` at
  `.claude/worktrees/agent-a0f09770086c096a6/checkpoints_holdout_baseline`
  (the `stage6-game-holdout` fold-1 baseline checkpoint used by the
  *original* `stage6-test-time-adaptation-agent` backtest whose published
  TTA-ON numbers this experiment is checking the high-dose checkpoint
  against). Did not ship a `value_head.pt` (not persisted -- gitignored),
  so one was trained fresh against its own `encoder_moe.pt`
  (`python -m jepa.train_value_head --epochs 20 --encoder
  checkpoints_holdout_baseline/encoder_moe.pt --out
  checkpoints_holdout_baseline`), with the 5 held-out games' recording
  files temporarily moved out of `ARC-AGI-3-Agents/recordings/` during
  that run only (same methodology as the original checkpoint setup in
  `stage6-test-time-adaptation-agent`, to avoid a value head fit partly
  on held-out-game reward events).
- n=30 per condition (60 total scorecard runs, matching the novelty-beta
  large-scale precedent's own n).
- Tooling: `scripts/run_meta_largescale_backtest.py` (new, this
  experiment) -- backs up the 4 production checkpoint files
  (`encoder_moe.pt`, `game_vocab_moe.json`, `moe_predictor.pt`,
  `value_head.pt`), swaps each condition's files into `checkpoints/` in
  turn, runs `scripts/run_scorecard.py --agent hypothesis` x30 per
  condition on the 5 held-out games, restores the production files when
  done (in a `finally` block, so a mid-run crash can't leave `checkpoints/`
  swapped).
- Analysis: `scripts/analyze_meta_largescale.py` (new) -- per-condition
  score/levels distribution, per-game breakdown, and Mann-Whitney U tests
  on (a) per-run levels completed, (b) per-run score, (c) per-solved-run
  score, (d) per-solved-run action-count-to-solve -- mirrors
  `experiments/stage6_novelty_aware_beta.md` Part 3's significance-check
  methodology exactly.

## Checkpoint provenance sanity checks (done before running anything)

Both checkpoints' `game_vocab_moe.json` confirmed to exclude all 5
held-out games (21 entries each, `r11l`/`bp35`/`m0r0`/`tr87`/`ka59`
absent from both) -- both are genuinely fold-1 held-out, not accidentally
retrained on the test games. Both checkpoints' `encoder_moe.pt` and
`value_head.pt` confirmed to load cleanly into fresh `CNNEncoder`/
`ValueHead` instances with no shape/key mismatches before committing to
the full run.

## Results

Both conditions run in the same round, back to back, same machine, same
day (2026-08-07, 05:53-06:48 wall clock, ~55 minutes for all 60 runs --
this box had at least one other, unrelated GPU job running concurrently
for part of that window, which is consistent with the per-run pace being
somewhat slower, ~65-90s/run, than the ~34s/run average the original
novelty-beta n=30 backtest reported on a presumably less contended box;
did not affect correctness, only wall-clock).

### Summary (n=30 each)

| condition | mean score | median score | std | mean levels | total levels | distinct games solved |
|---|---|---|---|---|---|---|
| metahd (high-dose Reptile + TTA) | 0.0756 | 0.0026 | 0.2359 | 0.500 | **15** | 1 (`r11l`) |
| baseline (normal training + TTA) | 0.0588 | 0.0033 | 0.1963 | 0.500 | **15** | 1 (`r11l`) |

**Levels completed is exactly tied: 15/30 vs 15/30, identical mean
(0.500), identical 50% solve rate on the identical single game
(`r11l`).** Neither condition ever completed a level on `bp35`, `m0r0`,
`tr87`, or `ka59` in any of the 60 runs -- consistent with every other
held-out-games agent-level backtest in this project's Stage 6 history
(the novelty-beta backtest, the original TTA-agent backtest, and the
n=8 preliminary meta-learning backtest all found the same thing: `r11l`
is essentially the only held-out game this agent ever solves within a
300-action budget, regardless of checkpoint or mechanism).

Per-run scores (in run order):

- metahd: `0.0197, 0.0, 0.0141, 0.0, 0.0, 0.0, 0.0653, 0.0052, 0.0086, 0.137, 0.0, 0.0, 0.0, 0.0056, 0.0235, 0.0443, 0.0, 0.0, 0.0053, 0.0, 0.0111, 0.9524, 0.0, 0.9524, 0.0, 0.0, 0.0064, 0.0, 0.0, 0.0167`
- baseline: `0.0, 0.9524, 0.0, 0.048, 0.0189, 0.0, 0.0, 0.0, 0.0, 0.0069, 0.0, 0.0084, 0.0, 0.0, 0.0067, 0.0086, 0.0119, 0.0147, 0.0085, 0.0171, 0.5879, 0.0, 0.01, 0.0, 0.0557, 0.0, 0.0093, 0.0, 0.0, 0.0`

Mean score nominally favors metahd (0.0756 vs 0.0588), but this is
exactly the same "one or two outlier fast completions dominate the mean"
pattern this project has flagged repeatedly (Stage 5's teacher-policy
backtest, the original TTA-agent backtest, the n=8 and n=30 novelty-beta
backtests): metahd has two runs at the game's max per-level score band
(`0.9524` each, r22/r24) vs baseline's one (`0.9524`, r2, plus a second
partial `0.5879`, r21). With levels-completed exactly tied, these are
score-magnitude outliers among an equal number of successes, not a
difference in how often either condition wins.

### Significance checks (mirroring `stage6_novelty_aware_beta.md` Part 3's methodology exactly)

| check | n (metahd / baseline) | statistic | p-value |
|---|---|---|---|
| levels-completed (per-run) | 30 / 30 | U=450.0 | **p=1.0000** |
| per-run score | 30 / 30 | U=453.5 | **p=0.9622** |
| per-solved-run score | 15 / 15 | U=116.0 | **p=0.9009** |
| solve-efficiency (actions to solve level 1) | 15 / 15 | U=110.0 | **p=0.9339** (mean 166.5 vs 171.8 actions) |

**Every single check comes back with p > 0.90 -- this is about as clean
a null result as this project's Stage 6 backtests have produced.** The
levels-completed p-value of exactly 1.0000 reflects the two groups having
literally identical rank distributions (15/30 successes each, in the same
binary pattern up to permutation) -- not a coincidence of rounding, a
genuinely symmetric outcome. Solve-efficiency, the metric that would most
directly reflect a "starts from a better-adapted place" effect if one
existed, shows metahd solving marginally *faster* on average (166.5 vs
171.8 actions) but the gap is small relative to the spread and nowhere
near significant.

## Verdict

**The high-dose meta-learning checkpoint's confirmed, dosage-validated
representation-level edge over plain test-time adaptation on a normally-
trained checkpoint (+0.98%/+1.28% vs +0.78%/+0.66% post-adaptation
changed-patches improvement, per `experiments/stage6_meta_learning.md`)
does not translate into a detectable agent-level advantage at n=30 real
gameplay repeats.** Every metric checked -- raw levels completed, mean
score, per-solved-run score, and solve efficiency -- lands within noise
of the baseline, with Mann-Whitney p-values all comfortably above 0.90.
This is not a reversal of the representation-level finding (nothing here
contradicts that the high-dose checkpoint genuinely adapts to better
predictions on held-out games than the plain baseline does) -- it is the
same "component measurably improved, agent-level backtest at practical
power can't detect it" pattern this project has now hit four separate
times this investigation: the teacher-policy value head (Stage 5), test-
time adaptation itself (`stage6-test-time-adaptation-agent`), the
novelty-aware beta cap (`stage6-novelty-beta-largescale`), and now
Reptile meta-learning. In all four cases the underlying measurement
(prediction quality, a beta distribution, or here, post-adaptation
changed-patches) showed a real, verified, non-noise effect -- and in all
four cases, 25-30 repeats on this project's 5-held-out-game,
300-action-budget, `levels_completed`-as-outcome protocol wasn't enough
statistical power to confirm it moved real play.

**Why this specific protocol keeps failing to detect real component-level
effects, and what would actually resolve it:** as `stage6_novelty_aware_
beta.md`'s own Part 3 already diagnosed, the structural bottleneck is
that `levels_completed` only ever fires on one of the five held-out games
(`r11l`) regardless of intervention -- a single Bernoulli(~0.5) trial
repeated 30 times per condition has an inherent floor on how small an
effect it can resolve (getting from "50% vs 50%, exactly tied" to a
detectably different split would need either a much larger n, a real
effect size far bigger than anything found so far in this investigation,
or a higher-resolution outcome metric that doesn't require an actual game
win to register signal at all -- e.g. directly comparing post-adaptation
`changed-patches` or `Q`-value trajectories across matched action
sequences during real play, the way `scripts/diagnose_hypothesis_beta_
holdout.py` already does for beta). More repeats on the same binary
win/loss metric on the same one game is unlikely to be the fix; a
different, denser metric is.

**Practical recommendation:** do not adopt the high-dose meta-learning
checkpoint over the plain baseline+TTA setup for any near-term submission
decision -- there is no evidence at this sample size that it plays
better, despite genuinely adapting better on the underlying prediction
task. The meta-learning training recipe and its representation-level
result remain worth keeping documented (a real, dosage-confirmed,
mechanistically-understood effect), but this closes the loop the n=8
preliminary backtest in `experiments/stage6_meta_learning.md` left open:
with proper statistical power, the answer is a clean, well-powered null,
not an inconclusive "maybe."

## Reproducing this experiment

```
# Baseline value head (one-time prerequisite -- checkpoints_meta_fold1_highdose
# already ships its own value_head.pt, no action needed there):
python scripts/move_holdout_recordings.py out   # move r11l/bp35/m0r0/tr87/ka59 recording files aside
python -m jepa.train_value_head --epochs 20 \
  --encoder checkpoints_holdout_baseline/encoder_moe.pt \
  --out checkpoints_holdout_baseline
python scripts/move_holdout_recordings.py back

python scripts/run_meta_largescale_backtest.py --n 30   # sets HYPOTHESIS_TEST_TIME_ADAPT=1 itself
python scripts/analyze_meta_largescale.py meta_ls_metahd_ meta_ls_baseline_
```

## Environment note (not specific to this experiment's conclusions, but real and worth flagging)

This worktree's gitignored asset directories (`checkpoints/`,
`checkpoints_meta_fold1_highdose/`, `checkpoints_holdout_baseline/`,
`data/`) were set up via `ln -s`, which on this box's Git-for-Windows/
MSYS environment does **not** create real symlinks or NTFS junctions for
directories -- confirmed directly via `fsutil reparsepoint query`
(returns "not a reparse point" for all four) and a live write-then-check
test (a marker file written into this worktree's `checkpoints_holdout_
baseline/` never appeared at the source path in another worktree, from
either `bash` or `PowerShell`). `ln -s` reported success and the
resulting directories were pre-populated with a faithful one-time
snapshot of the source content (matching file sizes and timestamps) --
so reads work exactly like a real link at setup time, but writes are
local-only and never propagate back to the source. This did not affect
this experiment's correctness (the value head trained in this section
only ever needed to be read from *this* worktree's own copy, which is
self-consistent), but it means treating any of these four directories in
this worktree as "live-linked to worktree X" is wrong -- they are
point-in-time copies as of 2026-08-06 ~18:46, not live views.
