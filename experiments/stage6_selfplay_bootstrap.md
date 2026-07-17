# Stage 6 experiment: self-play bootstrapping with `Hypothesis` -- a real, verified negative result

**Status: one full iteration complete, real negative result on the MoE
predictor.** Per the scope this was given, value-head retraining and the
scorecard backtest were *not* run -- both are gated on the predictor eval
showing a real improvement, and it didn't. `checkpoints/moe_predictor.pt`
(production) is untouched; `checkpoints_selfplay/` (gitignored, not
committed) holds the negative-result checkpoint for anyone who wants to
inspect it locally.

## The idea

Stage 5's "teacher policy" follow-up (see CLAUDE.md) already tried
harvesting denser training data with a smarter-than-random policy, but
only for the value head, and using `Memory` (Stage 3) rather than
`Hypothesis` (Stage 5, today's best agent) as the harvester. This
experiment asks the more ambitious version of that question: does using
`Hypothesis` to harvest a self-play corpus, then retraining the **MoE
world model itself** (`encoder_moe.pt` + `moe_predictor.pt`) on
random-policy + self-play data combined, produce a better world model
than the production checkpoint (trained on random-policy data only)?

## Method

1. `HYPOTHESIS_MAX_ACTIONS` env var added to `hypothesis_agent.py`
   (mirrors the existing `DIAG_MODE` pattern) so the harvest budget could
   be bumped without hand-editing the class attribute. One pass across
   all 25 local games, `HYPOTHESIS_MAX_ACTIONS=2500` (matching Stage 5's
   `Memory`-teacher precedent), production checkpoints unchanged.
2. Compared the harvested corpus against the existing random-policy
   corpus and the `Memory`-teacher precedent on completion rate, frame-
   changed rate, and diversity (distinct states/actions/clicks) --
   *before* spending any retraining time on it.
3. Retrained the MoE predictor (`jepa/train_moe_predictor.py
   --pretrain-epochs 20 --epochs 60 --num-experts 8`, i.e. the exact
   command that built the current production checkpoint) on the combined
   corpus (150 random-policy files + 25 new self-play files), to a
   separate `checkpoints_selfplay/` directory.
4. Extended `jepa/benchmark.py` with a `--moe` flag (it previously only
   knew how to evaluate Stage 1's monolithic predictor) to get an
   apples-to-apples `changed-patches` comparison between
   `checkpoints/moe_predictor.pt` and `checkpoints_selfplay/moe_predictor.pt`
   on the *same* held-out split.
5. Value-head retrain + local scorecard backtest were scoped as
   conditional on step 4 showing a real improvement. It didn't, so
   neither ran -- see "What was skipped, and why" below.

### A deliberate corpus-composition choice worth flagging up front

The main checkout's `ARC-AGI-3-Agents/recordings/` currently holds 975
files: the 150-file/~332MB random-policy corpus that actually produced
the current production checkpoint, plus ~850 `hypothesis`-agent files
(~13GB) that are leftover evaluation-run byproducts from
`stage6-score-optimization`'s ongoing sweep in a different, concurrently-
running session -- not part of any documented training corpus. This
experiment's worktree was seeded with only the 150-file random corpus
(re-copied fresh) plus this experiment's own 25 self-play files, *not*
the other 850 -- both to stay out of a concurrently-running session's
data and because those files were never part of what actually trained
`checkpoints/moe_predictor.pt` in the first place, so including them
would break the "does self-play data help *relative to the current
production setup*" comparison this experiment is actually asking.

## Step 1: corpus quality and diversity, before touching the retrain

One pass, `HYPOTHESIS_MAX_ACTIONS=2500`, all 25 games: **4 levels
completed across 4 distinct games** (`ft09`, `lp85`, `m0r0`, `r11l`),
avg. 649.5 actions to first completion, 62,525 total actions, 25 files
totaling 2.8GB. Compared against the two baselines this was scoped to
beat or match:

| harvester | levels | distinct games | games |
|---|---|---|---|
| `Memory` teacher (Stage 5 follow-up, CLAUDE.md) | 5 | 5 | `ar25`, `cd82`, `lp85`, `m0r0`, `r11l` |
| `Hypothesis` (this experiment) | 4 | 4 | `ft09`, `lp85`, `m0r0`, `r11l` |

**Slightly below the `Memory` precedent on raw count**, not above it --
`Hypothesis` picked up `ft09` that `Memory`'s single pass didn't, but
missed `ar25` and `cd82` that `Memory` got. This alone is a fair,
reportable finding: despite `Hypothesis` being the more sophisticated,
better-performing agent on short matched-budget runs (CLAUDE.md's Stage 5
follow-up 3: 14 total levels / 5 distinct games vs `Memory`'s own numbers
elsewhere), that edge didn't clearly carry over to a single long
(2500-action) harvesting pass. Plausibly explained by `Hypothesis`'s
own documented reliability-over-consistency tradeoff (Stage 5: fast but
unreliable) not mattering much at n=1 pass -- not investigated further,
flagged as a real, if modest, data point against assuming a "better
agent" trivially means "better harvester."

**Frame-level changed rate and diversity** (full per-game breakdown
available via the ad hoc analysis script's output, summarized here):

| corpus | files | transitions | frame-changed rate |
|---|---|---|---|
| random (existing, 150 files) | 150 | 12,000 | 63.5% |
| `Hypothesis` self-play (this experiment) | 25 | 62,500 | **76.2%** |

A real, meaningful jump in the fraction of steps that actually do
something -- consistent with the hypothesis that a directed policy
produces more informative transitions per action than random search.
Three games make this especially concrete: `s5i5`, `tn36`, and `vc33`
were near-frozen under random play (12.1%, 12.5%, 13.1% changed rate --
CLAUDE.md's own Stage 1 history already flagged these as "identity
baseline is tiny" games) but hit **100% changed rate** under
`Hypothesis` -- directed clicking triggers real visible dynamics on
these games that random search essentially never found. Distinct-state
counts back this up: `tn36` went from 39 states in 480 random steps to
1,358 states in 2,500 self-play steps (0.081 -> 0.543 states/step, a real
~6.7x density increase, not just "more steps").

**But there's a real echo-chamber signal too, and it's not subtle.** The
random-policy corpus visits 7-8 distinct action ids on essentially every
game (expected from near-uniform sampling). The self-play corpus's
action diversity is *much* narrower on many games -- some of that is
legitimately explained (CLAUDE.md's Stage 5 follow-up 3 already
documented that `ft09`/`lp85`/`r11l` genuinely only have ACTION6
available, so "2 distinct actions" there is a correct reflection of the
game, not narrowing), but several games where random used the *full*
7-8-action space saw `Hypothesis` narrow to noticeably fewer: `su15`
7->3, `tu93` 8->5, `ls20` 7->5, `tr87` 7->5, `sc25` 8->6. And on the
click-location side specifically, the picture is genuinely mixed rather
than uniformly good: `ft09`/`lp85`/`r11l`/`s5i5`/`tn36`/`vc33` show a
striking increase in distinct click locations (dozens under random, up
to ~1,850 out of ~2,500 clicks under `Hypothesis` -- direct, welcome
confirmation that Stage 5 follow-up 3's softmax click-sampling fix is
doing its job) -- but `g50t`, `ls20`, `re86`, `tr87`, `tu93`, and `wa30`
collapsed to **exactly 1** distinct click location each, versus
58-80 distinct clicks under random on those same games. On those six
games, `Hypothesis` essentially never chose to click at all (or clicked
exactly once, likely via the epsilon-random fallback) despite ACTION6
being available and random policy exploring it substantially --
consistent with the agent's Q-scoring having decided ACTION6 isn't
worth it on those particular games and then never revisiting that
belief, a directly self-referential narrowing rather than exploration.

**Read on Step 1 alone:** genuinely denser, more-changed data on
average, and real evidence of triggering previously-unseen dynamics on
a handful of games -- but also a real, measurable narrowing of action-
space and click-space coverage on a different handful of games. Exactly
the mixed picture the echo-chamber risk predicts: not uniformly "more
informative," partly "reflects the current policy's own beliefs back at
itself." Worth remembering when reading the retrain result below.

## Step 2: MoE retrain -- real operational issues, both fixed and kept

Retraining hit two real infrastructure problems on this shared machine,
both fixed with general (not one-off) code changes:

1. **A `MemoryError` inside a DataLoader worker's `pickle.load`**,
   reproducing CLAUDE.md's own documented full-disk gotcha almost
   exactly -- except this time the disk pressure came from a *different*,
   concurrently-running session on the same machine (not something this
   experiment could fix at the source, per the standing instruction to
   stay out of the other session's work). Added a `JEPA_NUM_WORKERS`
   env var override to `jepa/train_moe_predictor.py`'s `_make_loaders`
   (defaults to the original 4-on-cuda/0-on-cpu behavior when unset) so a
   run can fall back to `num_workers=0` -- slower, but sidesteps the
   Windows spawn/pickle path entirely rather than depending on pagefile
   headroom that isn't always available.
2. **The background training process was killed by something outside
   this session's control at almost exactly the ~55-60 minute mark,
   three separate times in a row** (confirmed via GPU utilization and
   growing log output right up until each kill -- these were not natural
   crashes). Since a full pretrain+60-epoch run on this branch's larger,
   self-play-augmented corpus (74,500 local transitions, up from the
   production run's implicit ~55k combined-corpus scale) took
   meaningfully longer than that ceiling, losing an entire run's progress
   to one external kill would have made this experiment impractical.
   Added `--checkpoint-every N` (periodic in-progress saves, plus one at
   the pretrain/finetune phase boundary) and `--resume-from PATH`
   (reloads a checkpoint's weights + game vocab, skips pretrain, treats
   `--epochs` as *additional* epochs from there) to
   `jepa/train_moe_predictor.py`. Verified directly that resuming
   reproduced the exact same loss trajectory as an uninterrupted run
   would have (epoch-1-after-resume's `val_pred_mse=0.00030` picks up
   essentially exactly where the killed run's last logged epoch left off)
   -- training was stitched across three resume boundaries (original run
   to epoch 10/60, resumed to epoch 30/60, resumed to epoch 50/60,
   resumed to completion at 60/60) with no discontinuity. Both additions
   are kept as general, permanent features of the script, not reverted
   after use -- a future long run on this same machine will hit the same
   ~1-hour ceiling regardless of what's being trained.

Final training config matched production exactly: `--pretrain-epochs 20
--epochs 60 --num-experts 8`, MiniGrid pretrain (67,200 transitions) +
ARC-3 fine-tune, warm-started from the same `checkpoints/encoder.pt`.

## Step 3: the apples-to-apples eval -- a clear, real regression

`jepa/benchmark.py`'s `eval` command only supported Stage 1's monolithic
predictor before this experiment; added a `--moe` flag
(`run_eval_moe`/`_load_moe_checkpoint`/`_evaluate_per_game_moe`) so a
`checkpoints/` vs `checkpoints_selfplay/` comparison could run on the
*exact same* held-out split (`TransitionDataset` + `random_split` with a
fixed seed, over whatever's currently in `ARC-AGI-3-Agents/recordings/`
-- deterministic as long as the file set doesn't change between the two
eval calls, which it didn't here).

| checkpoint | pred_changed_mse | identity_changed_mse | changed-patches improvement |
|---|---|---|---|
| `checkpoints/` (production, random-policy-only training data) | 0.02753 | 0.03551 | **+22.5%** |
| `checkpoints_selfplay/` (this experiment, combined corpus) | 0.00389 | 0.00412 | **+5.7%** |

Both numbers technically "pass" the milestone's own bar (predictor beats
identity), but the self-play checkpoint is **far weaker** than
production on the identical held-out data -- not a wash, a clear
regression.

**Investigated why, rather than just reporting the number.** The
absolute MSE values for the self-play checkpoint are uniformly ~10-50x
smaller than production's across nearly every game in the per-game
breakdown -- the same suspicious pattern CLAUDE.md's Stage 1 history
(item 7) once flagged as a possible representation-collapse symptom
before ruling it out for the original monolith. Checked directly this
time:

- **Feature variance is healthy on both checkpoints -- not classic
  collapse.** Per-channel std on a held-out batch: production
  mean=1.643 (min 1.165), self-play mean=1.754 (min 1.468) -- self-play's
  is if anything *higher*, comfortably above the `VARIANCE_FLOOR=1.0`
  this project already uses as its collapse threshold.
- **But the encoder has become far more invariant specifically to real
  frame-to-frame change.** Measured `(feat(frame_t) - feat(frame_t+1))^2`
  directly at changed patches on the same held-out batch: production
  0.013686, self-play **0.000130** -- about **105x smaller**. This isn't
  "features are constant" (ruled out above), it's "the encoder maps two
  frames that visibly differ in pixel space to nearly the same point in
  feature space, specifically at the patches that changed" -- a
  transition-specific degeneracy, not a global one. This is also visible
  directly in the benchmark eval's own aggregate numbers without any
  extra diagnostic: `identity_changed_mse` on the exact same held-out
  patches is 0.0355 for production vs. 0.0041 for self-play, an ~8.6x
  gap that has nothing to do with the predictor at all -- it's purely a
  property of the two encoders.

**Working hypothesis for the mechanism (not fully confirmed, flagged
honestly as such):** given Step 1's own finding that a real slice of the
self-play corpus is low-diversity, exploitation-heavy data (the six
games that collapsed to one click location; the games with narrowed
action coverage), a plausible story is that a meaningful fraction of the
self-play corpus's "changed" transitions are small, repeated, near-
duplicate perturbations clustered around whatever states `Hypothesis`'s
current policy already gravitates to -- which would bias the encoder
toward learning "collapse frame_t and frame_t+1 together" as an
increasingly safe generalization specifically in the neighborhood of
those frequently-revisited near-duplicate states, without collapsing
its *global* representational capacity (which the healthy overall
variance number confirms it didn't). This is a coherent story consistent
with every number measured this session, but it was not independently
verified beyond what's reported here (e.g., by directly measuring
pairwise state similarity within the self-play corpus specifically) --
a fair target for a future session if this line is revisited, not
claimed as proven here.

## The echo-chamber risk, addressed directly

The task this experiment was scoped under asked explicitly not to treat
"higher completion/changed rate" as self-evidently "better data," and to
say plainly if the risk can't be fully ruled out. It can't, and the
result here is actually a fairly clean illustration of exactly that
risk materializing:

- **Surface-level activity metrics went the "good" direction**: +12.7pp
  frame-changed rate, real new dynamics triggered on 3 previously-near-
  frozen games, large click-diversity gains on 6 click-heavy games.
- **Diversity metrics that specifically guard against the echo-chamber
  risk went the "bad" direction on a different, real subset**: narrowed
  action-space coverage on several non-click games, complete ACTION6
  abandonment (1 distinct click each) on 6 other games.
- **The actual downstream retrain result is a clear regression** on the
  metric this whole project treats as ground truth (`changed-patches`,
  not naive whole-grid MSE) -- and the mechanistic finding (encoder
  becoming locally invariant specifically around changed patches, while
  staying globally healthy) is *consistent with* the self-play corpus's
  own narrower, exploitation-shaped coverage being the actual problem,
  though not independently proven to be the specific cause.

**Honest bottom line: one iteration of this loop does not show that a
self-play-harvested corpus from the current best agent improves the
world model -- it shows a real regression, with a plausible and
partially-evidenced (but not fully confirmed) link to the exact
circularity risk this approach was warned to watch for.** This is a
genuine, useful negative result, not an inconclusive one.

## What was skipped, and why

Per this experiment's own scope: value-head retraining
(`jepa.train_value_head`) and the local scorecard backtest
(`scripts/run_scorecard.py`, copied into this branch from
`stage6-score-optimization` via a read-only `git show` -- that branch's
own working tree was never touched) were both explicitly conditional on
the MoE predictor eval showing a real improvement. It showed a clear
regression instead, so neither ran. Retraining the value head on top of
a *worse* world model, or running a multi-repeat agent-level backtest
against a checkpoint already known to have degraded on the component-
level metric that predicts agent quality, would have been spending
compute to confirm a foregone conclusion rather than testing anything
new -- consistent with this project's own standing practice of not
chasing further evidence once a lever has a clean negative result (see
e.g. Stage 4's Sokoban ablation, or the EPSILON/PATCH_TEMP sweep in
`stage6_score_variance.md`).

## Recommendation

**Do not adopt this checkpoint.** `checkpoints/moe_predictor.pt`
(production) remains the right one to use; `checkpoints_selfplay/` is
kept locally (gitignored, not committed) purely for anyone who wants to
inspect the regression directly, not as a candidate.

If a future session wants to revisit self-play bootstrapping, the most
promising next lever isn't "harvest more" -- it's **filtering the
harvested corpus for diversity before retraining on it**, e.g.
downweighting or capping near-duplicate consecutive states within an
episode (mirroring this project's own established "weight what's
actually informative" pattern from `TransitionDataset.sample_weights()`
and `ValueDataset.sample_weights()`), rather than mixing the harvested
data in unfiltered. That would directly target the specific failure mode
this session found (narrow, exploitation-clustered data diluting the
encoder's sensitivity to real change) instead of re-running the same
experiment hoping for a better draw. Also worth trying: harvesting with
a *higher* `EPSILON` than `Hypothesis`'s production default specifically
for data-collection runs (more random exploration mixed into the
harvest, directly counteracting the self-referential narrowing Step 1
measured) -- untried this session, flagged as the more targeted fix
given the diagnosis above, ahead of a from-scratch harvesting redesign.

## Reproducing this experiment

```
# 1. Harvest (from ARC-AGI-3-Agents/, budget bumped via env var)
HYPOTHESIS_MAX_ACTIONS=2500 python main.py --agent=hypothesis

# 2. Corpus quality check (compare against the existing random corpus)
python scripts/compare_agents.py hypothesis

# 3. Retrain (same hyperparams as production; resumable if interrupted)
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 \
    --num-experts 8 --out checkpoints_selfplay --checkpoint-every 5
# if killed partway (JEPA_NUM_WORKERS=0 helps on a disk-constrained shared
# machine): resume with
python -m jepa.train_moe_predictor --epochs <remaining> --num-experts 8 \
    --out checkpoints_selfplay --checkpoint-every 5 \
    --resume-from <checkpoints_selfplay-snapshot-dir>

# 4. Apples-to-apples eval (same held-out split, both checkpoints)
python -m jepa.benchmark eval --moe --checkpoint-dir checkpoints --tag prod
python -m jepa.benchmark eval --moe --checkpoint-dir checkpoints_selfplay --tag selfplay
```
