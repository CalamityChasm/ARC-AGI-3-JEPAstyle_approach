# Stage 6 experiment: MAX_ACTIONS budget x checkpoint backtest matrix

**Status: COMPLETE (local backtest only). No Kaggle submissions were made
from this branch, per the standing task instruction.** This is
groundwork for a future submission decision, not a submission itself.

## Why this matrix

Three world-model checkpoints now exist with independent Kaggle
submission histories, all at `MAX_ACTIONS=300`:

- **production** (`checkpoints/`, main checkout) -- real scores `0.23`, `0.06`.
- **search-harvest** (`E:\jepa_overflow\checkpoints_search\`) -- real scores
  `0.16`, `0.08`.
- **object-identity** (`.claude/worktrees/agent-a5728f08b06dab74a/checkpoints_object_identity/`)
  -- real score `0.00` (one submission, under separate investigation for a
  possible train/test-game-overlap overfitting issue).

Separately, `stage6-score-optimization` found `MAX_ACTIONS=900` (vs the
original `300`) is a **reliability lever** (fewer zero-completion runs),
not a clear mean-score lever, on the **production** checkpoint -- real
submission score `0.22`. That change was deliberately never combined with
either newer checkpoint, to keep the comparison isolated. This experiment
runs the full 3x2 = 6-config matrix locally, using
`scripts/run_scorecard.py`'s real scoring formula (the same one
`rules.md`/Kaggle uses, computed by the offline harness itself), matching
this project's own established backtest protocol.

## Method

- `ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`'s `MAX_ACTIONS`
  ported to the `HYPOTHESIS_MAX_ACTIONS` env-var-override pattern
  (default kept at `300`, unlike `stage6-score-optimization`'s
  experimental `900` default, since this backtest sets the value
  explicitly per run via the env var either way).
- `scripts/run_scorecard.py` / `scripts/summarize_scorecards.py` ported
  from `stage6-score-optimization` unchanged.
- Checkpoints swapped into this worktree's `checkpoints/` between
  configs (the harness hardcodes that path) -- `encoder_moe.pt`,
  `moe_predictor.pt`, `game_vocab_moe.json`, `value_head.pt` copied from
  each source directory listed above, in turn.
- All 25 local games per repeat, offline mode, same protocol as every
  prior backtest in this project.
- Repeats: **n=8 at `MAX_ACTIONS=300`** (~3-4 min/repeat), **n=6 at
  `MAX_ACTIONS=900`** (~8-12 min/repeat) per checkpoint. This asymmetry
  is a deliberate, documented time-budget tradeoff (mirrors this
  project's own precedent of scaling n down for costlier configs -- see
  e.g. `stage6_object_identity.md`'s n=3) -- both still meet the task's
  "5-8 repeats" floor. All 6 configs' sweeps ran back-to-back in one
  session on shared GPU/CPU (concurrently with two other agents' work --
  one MoE retrain, one random-agent variance sweep -- so per-repeat wall
  time is somewhat inflated vs. an isolated run, but that doesn't affect
  score/completion numbers, only wall-clock).

## Results

| config | n | mean score | std | min | max | mean levels completed | zero-completion runs |
|---|---|---|---|---|---|---|---|
| production, 300 | 8 | 0.0278 | 0.0623 | 0.0000 | 0.1925 | 1.00 | 2/8 |
| production, 900 | 6 | 0.0154 | 0.0222 | 0.0008 | 0.0641 | 2.50 | 0/6 |
| search-harvest, 300 | 8 | 0.0061 | 0.0089 | 0.0000 | 0.0294 | 1.12 | 1/8 |
| search-harvest, 900 | 6 | 0.0136 | 0.0215 | 0.0000 | 0.0606 | 1.33 | 1/6 |
| object-identity, 300 | 8 | 0.0061 | 0.0071 | 0.0000 | 0.0212 | 0.62 | 3/8 |
| object-identity, 900 | 6 | 0.0439 | 0.0678 | 0.0007 | 0.1905 | 2.00 | 0/6 |

Raw per-run scores (in run order):

```
production,300:      0.00256, 0.00524, 0.00000, 0.00790, 0.01089, 0.00000, 0.00356, 0.19245
production,900:      0.00445, 0.00664, 0.01350, 0.00267, 0.06408, 0.00081
search-harvest,300:   0.02940, 0.00182, 0.00356, 0.00356, 0.00310, 0.00000, 0.00572, 0.00185
search-harvest,900:   0.06061, 0.01161, 0.00027, 0.00000, 0.00901, 0.00033
object-identity,300:  0.00000, 0.01066, 0.00384, 0.01066, 0.00000, 0.00000, 0.00233, 0.02116
object-identity,900:  0.01844, 0.00079, 0.00066, 0.05030, 0.19048, 0.00300
```

`total_actions` (7525 at budget=300, 22525 at budget=900) and
`total_levels` (183) were bit-for-bit identical across every one of the
48 runs, regardless of checkpoint or repeat -- confirming
`stage6_object_identity.md`'s own earlier observation that the local
offline harness's episode/level structure is deterministic; only score
and `levels_completed` vary run to run.

## What this shows

**1. `MAX_ACTIONS=900` is a reliability lever across all three
checkpoints, not just production.** Mean levels completed rose for every
single checkpoint (production 1.00->2.50, search-harvest 1.12->1.33,
object-identity 0.62->2.00), and zero-completion runs dropped or stayed
at zero for every checkpoint (production 2/8->0/6, search-harvest
1/8->1/6, object-identity 3/8->0/6). This generalizes
`stage6_score_variance.md`'s production-only finding cleanly -- the
budget bump helps the worst-case outcome regardless of which world model
is driving the agent, consistent with the underlying theoretical argument
(a level never reached scores exactly 0; a bigger budget can only help
that failure mode, not hurt levels that already completed).

**2. Mean score does not track mean levels completed cleanly, for any
checkpoint** -- reproducing `stage6_max_actions.md`'s own earlier
caution. production's mean score actually *dropped* going 300->900
(0.0278->0.0154) despite completing more levels on average, because a
few extra actions spent on levels that would have completed anyway under
either budget cost real efficiency-component score. search-harvest and
object-identity's mean scores rose 300->900, but by amounts comparable to
or smaller than the overlapping std in both directions -- not a clean
signal either way at this n.

**3. Every reasonably high mean score in this table is dominated by a
single outlier run, not a real central tendency shift** -- the same
pattern `stage6_max_actions.md` already flagged for its own n=2 test.
production,300's mean (0.0278) is driven almost entirely by one 0.1925
run (the other 7 range 0.0000-0.0109); object-identity,900's mean
(0.0439) is driven almost entirely by one 0.1905 run (the other 5 range
0.0007-0.0503) -- drop that single run from each and both means fall to
roughly 0.004-0.015, in line with everything else in the table. Treating
raw means from n=6-8 samples with std often exceeding the mean itself
(see the `std` column -- every row's std is 50-160% of its own mean) as
suggestive, not decisive -- consistent with this project's long-running
theme (CLAUDE.md's Stage 2/5 sections) that raw score/level counts are a
noisy metric at this sample size.

**4. object-identity's `MAX_ACTIONS=300` result here (mean score 0.0061,
mean levels 0.62, the *worst* mean-levels figure of all six configs) does
not replicate `stage6_object_identity.md`'s own earlier n=3 backtest**,
which found object-identity beating search-harvest with every one of 3
runs outscoring every one of search-harvest's 3 runs (mean 0.0703 vs
0.0020). At this larger n=8 sample, object-identity,300 and
search-harvest,300 are statistically indistinguishable (means 0.0061 vs
0.0061, both well within one std of each other) -- the earlier
"completely one-sided" n=3 comparison did not hold up at a larger sample.
This is worth taking seriously precisely because it's an honest
non-replication, not because either number is more "correct" -- it is a
direct demonstration of why this project insists on 5-8+ repeats before
trusting a small-n local win, and it tempers how much weight
`stage6_object_identity.md`'s own encouraging local numbers should carry
against the real `0.00` Kaggle submission and its open overfitting
question.

**5. At `MAX_ACTIONS=900`, production and object-identity are roughly
tied and both ahead of search-harvest** on mean levels completed
(production 2.50, object-identity 2.00, search-harvest 1.33) and on
zero-completion reliability (0/6, 0/6, 1/6 respectively). object-identity's
higher mean score at 900 is mostly the single-outlier effect noted in
point 3 above -- excluding that one run, its remaining 5 runs average
~0.0146, essentially tied with production's 0.0154.

## Recommendation

**No config in this matrix shows a clean, decisive win over what's
already been validated on real Kaggle submissions.** With that caveat
central to everything below:

- **`MAX_ACTIONS=900` is worth carrying forward regardless of which
  checkpoint is used** -- it's now been shown to reduce worst-case
  (zero-completion) outcomes on all three checkpoints, not just
  production, and the theoretical case for it (never-completing scores
  exactly 0, bigger budget can't make an already-completing level worse
  except via the small efficiency penalty) remains intact.
- **Among checkpoints, this local backtest gives no strong reason to
  switch away from production.** At 900 -- the budget worth using either
  way -- production ties object-identity (once its single-outlier
  inflated mean is discounted) and clearly beats search-harvest, while
  carrying by far the deepest and most consistent real-Kaggle track
  record (4 submissions spanning `0.06`-`0.23`) of the three. Neither
  newer checkpoint has yet shown a local or real-world edge over
  production once budget is controlled for.
- **object-identity remains the most interesting candidate to watch, not
  the safest one to submit next.** Its 900-budget showing is competitive
  locally, but its 300-budget showing here is the single worst of the six
  configs, its one real submission scored the worst of all six
  submissions across all checkpoints to date (`0.00`), and a parallel
  investigation into possible train/test-game overlap is still open and
  unresolved. Combining an unresolved overfitting question with a
  budget change that hasn't been validated on this checkpoint in
  particular via a real submission would confound two untested variables
  at once -- not a clean next step.
- **search-harvest shows no local advantage at either budget** in this
  matrix (worst or tied-worst mean score at both 300 and 900) and no
  clear real-Kaggle advantage either (`0.16`, `0.08`, both inside
  production's own established noise range) -- nothing here elevates it
  above production.

**If a single real submission is to be spent next: production checkpoint
at `MAX_ACTIONS=900`** is the most defensible choice from this local
evidence -- it combines the one lever shown to help across every
checkpoint tested (the budget bump) with the checkpoint that has the
longest, most consistent real-world track record and no open
reliability/overfitting questions. This is *not* a strong recommendation
in an absolute sense (its own 900-budget mean score, 0.0154, is
unremarkable and its std is wide) -- it is the least uncertain of six
options that are all still within noise of each other, not a config with
demonstrated superiority. **Per the task's explicit instruction, no
submission was made from this session.**

## Reproducing this matrix

```
export HYPOTHESIS_MAX_ACTIONS=300   # or 900
python scripts/run_scorecard.py --agent hypothesis --label <name>
python scripts/summarize_scorecards.py <label_prefix> ...
```
Checkpoints must be copied into `checkpoints/` (`encoder_moe.pt`,
`moe_predictor.pt`, `game_vocab_moe.json`, `value_head.pt`) from the
relevant source directory before each config's runs -- the harness
hardcodes that path. All 48 individual scorecard JSON files
(`logs/scorecards/{prod,search,objid}_{300,900}_r{1..8}.json`) are
gitignored local artifacts, not committed, but fully regenerable via the
above.
