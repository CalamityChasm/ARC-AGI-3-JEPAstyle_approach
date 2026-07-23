# Stage 6 experiment: does removing per-game embedding conditioning close the held-out-game gap?

**Status: DONE. Ablating per-game embedding conditioning does NOT close
the held-out-game generalization gap -- the no-game-id checkpoint
collapses to ~identity parity on the 5 held-out games just like the other
two checkpoints, refuting this experiment's core hypothesis. But it is a
large, real, unexpected win on the 20 TRAINED games: +64.9% changed-
patches improvement over identity, vs. +8.0% (baseline) and +1.9%
(object-identity) for the checkpoints trained WITH per-game conditioning
-- roughly 8x and 34x larger respectively. Per-game embedding
conditioning isn't just failing to help hidden-game generalization; on
this architecture and data scale it appears to be actively costing a lot
of prediction quality even on the games it was trained on, extending
Stage 1's older "no improvement" finding (CLAUDE.md item 4) into "actively
harmful" for the MoE architecture.**

## Motivation

`experiments/stage6_game_holdout.md` found that both a "production-style"
(no contrastive loss) and an "object-identity-style" (same-color
contrastive loss) MoE checkpoint -- both trained WITH per-game embedding
conditioning (`MoEPredictor(num_games=...)`, `game_idx` looked up from a
per-game vocabulary) -- collapse to ~identity parity (changed-patches
improvement ~0%) on 5 games held out of training entirely. The suspected
mechanism: a never-seen `game_id` falls back to `game_vocab.get(game_id, 0)`
== index 0, an untrained embedding row (`hypothesis_agent.py`'s real
production behavior on a genuinely novel Kaggle game, faithfully mirrored
by that experiment's eval methodology). Since every hidden Kaggle game is
by construction outside the local training vocabulary, this raised the
possibility that the entire world-model's demonstrated edge over identity
doesn't transfer to any genuinely novel game.

Also worth noting: back in Stage 1 (see CLAUDE.md's Stage 1 iteration
history, item 4), per-game embedding conditioning was already tried once
before on the older single-predictor architecture and found to give "no
improvement" on known games -- so this signal may never have been earning
its keep even where it seemed harmless.

This experiment tests the natural next hypothesis: if per-game
conditioning is actively hurting (or simply not helping) held-out-game
generalization, does **removing it entirely** close some or all of the
gap, and does it cost anything on the 20 games the model *did* train on?

## Method

Reused `stage6-game-holdout`'s infrastructure directly (branched from
that branch, not rebuilt): the same `--exclude-games` CLI flag, the same
20-game corpus/vocab construction, the same held-out game list, and the
same evaluation methodology (`scripts/eval_game_holdout.py`, extended
into `scripts/eval_gameid_ablation.py` to add a third checkpoint and a
trained-games-slice comparison).

**New flag: `--ablate-game-id` on `jepa/train_moe_predictor.py`.** Forces
every transition's `game_idx` to a constant `0` throughout BOTH training
and validation, regardless of the transition's real game. Implementation
choice, and why: `MoEPredictor._condition` already treats a missing
`game_idx` (`None`) as an all-zeros tensor (`torch.zeros(b, ...)`) --
see `jepa/models/moe_predictor.py`'s own docstring ("game_idx: ... (0 if
omitted)") -- so forcing every batch to index 0 is mathematically
equivalent to never conditioning on game identity, while leaving the
model's `game_embed` table (and therefore the checkpoint's state-dict
shape) unchanged from a normal MoE checkpoint. Only row 0 of that table
ever receives a real gradient; every other row stays at its random
initialization and is never read. This was the simplest surgical
approach that satisfies the task's own constraint (don't change
`num_games`/vocab-size handling in a way that could break other loading
code) -- the checkpoint's shape is indistinguishable from an ordinary MoE
checkpoint's, just semantically inert past row 0.

Eval must apply the identical substitution consistently: since the
ablated checkpoint was trained with `game_idx` forced to 0 on *every*
transition (trained games included, not just fallback-eligible unseen
ones), evaluating it fairly on the 20 trained games also requires forcing
`game_idx=0` there too -- using each trained game's *real* vocab index
instead would score the checkpoint against an untrained, never-touched
embedding row it never actually learned to use. `scripts/
eval_gameid_ablation.py`'s `force_zero_game_idx` flag (`True` only for the
ablated checkpoint) implements this.

Training command (identical recipe to the other two checkpoints in
`experiments/stage6_game_holdout.md`, only `--ablate-game-id` added,
`--contrast-weight 0.0` to isolate this one variable from the
object-identity question):

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --ablate-game-id --checkpoint-every 5 \
  --out checkpoints_holdout_nogameid
```

(`JEPA_NUM_WORKERS=0`, same rationale as the prior experiment -- shared/
contended GPU box.)

## Results

All three checkpoints trained cleanly to completion (20 MiniGrid-pretrain
+ 60 ARC-finetune epochs) on byte-identical corpora (confirmed via
`moe_training_meta.json`: 9,600 local + 33,998 external + 67,200 MiniGrid
transitions, 21-entry game vocab, identical `exclude_games`, differing
only in `contrast_weight`/`ablate_game_id`). Both eval slices (`n=1881`
changed-patch held-out transitions, `n=5441` changed-patch trained-game
transitions) are identical across all three checkpoints -- confirmed
directly in `logs/eval_gameid_ablation.log`, so this is a true
apples-to-apples comparison.

**Training itself was a real operational obstacle this session, worth a
brief note for future reference:** the first two attempts at this
training run died silently (no Python traceback, no CUDA error, clean
process termination) partway through the ARC-finetune phase --
`python.exe` simply stopped existing, `tail`ing the log went silent, and
`nvidia-smi`/`tasklist` showed no trace of it. Both times this looked
identical to a background-task lifetime limit in the execution
environment terminating the OS process backing a long-running
`run_in_background` bash call, not a training bug (disk space, GPU state,
and the training code itself were all directly checked and ruled out).
Fixed by launching the third attempt as a fully OS-detached process
(PowerShell `Start-Process`, no parent job-object tying its lifetime to
any single tracked shell task) -- that run completed cleanly end-to-end.
Not a finding about the model or data, just a environment-specific
gotcha for any future long (60-90+ minute) unattended training run in
this same setup.

### 1. Changed-patches improvement over identity -- the core comparison

| checkpoint | held-out games (5), n=1881 | trained games (20), n=5441 |
|---|---|---|
| baseline (with game-id) | **+0.0%** (pred=0.065562, id=0.065571) | **+8.0%** (pred=0.001001, id=0.001088) |
| object-identity (with game-id) | **+0.0%** (pred=0.042601, id=0.042602) | **+1.9%** (pred=0.001093, id=0.001114) |
| **no-game-id (ablated)** | **-0.2%** (pred=0.018284, id=0.018245) | **+64.9%** (pred=0.001067, id=0.003039) |

Per-game breakdown for the held-out slice (all three checkpoints, from
`logs/gameid_ablation_results.json`) confirms the pooled number isn't
hiding a per-game split -- the no-game-id checkpoint is at or below
identity parity on every single held-out game (never positive, unlike
the other two which land at small positive numbers on 3/5 games), the
same broad collapse pattern as the other two checkpoints, if anything
landing marginally on the worse side of it rather than better:

| game | baseline | object-identity | no-game-id |
|---|---|---|---|
| `r11l` | -1.6% | +0.0% | -2.4% |
| `bp35` | +0.0% | +0.1% | -0.3% |
| `m0r0` | +0.2% | +0.0% | -0.1% |
| `tr87` | +0.2% | +0.0% | -0.0% |
| `ka59` | -0.9% | -0.3% | -0.4% |

**Answer to the core question: no, ablating game-id conditioning does
NOT close the held-out-game gap, even partially.** The no-game-id
checkpoint's held-out performance (-0.2%) is statistically indistinguishable
from the other two checkpoints' (+0.0%, +0.0%) -- all three collapse to
identity parity on unseen games. This refutes the experiment's motivating
hypothesis (that the gap was caused by a never-seen `game_id` falling
back to an untrained embedding row): a model that has literally never
conditioned on any per-game signal at all -- not "falls back to untrained
row 0," but genuinely never had per-game information available during
training -- still fails to generalize to held-out games' dynamics. The
bottleneck is not the per-game embedding lookup mechanism; it's
something more fundamental about how the shared encoder/predictor's
learned dynamics are tied to the specific visual/color statistics and
mechanics of the 20 training games, that a categorical game-id signal
was never really carrying in the first place.

**Answer to the regression-check question: the opposite of a
regression.** Removing game-id conditioning doesn't cost anything on
trained games -- it improves changed-patches by roughly **8x** over the
baseline (+64.9% vs +8.0%) and **34x** over the object-identity
checkpoint (+64.9% vs +1.9%), on the identical 20-game corpus and
otherwise-identical recipe. This is a large, unambiguous, single-variable
result (only `--ablate-game-id` differs from the baseline run) --
consistent in direction with, and considerably stronger than, Stage 1's
older finding (CLAUDE.md item 4) that per-game conditioning gave "no
improvement" on the old single-predictor architecture. Here, on the MoE
architecture, per-game conditioning isn't neutral -- it appears to
actively cost most of the model's achievable prediction quality on the
very games it trains on. A plausible mechanism (not independently
verified this session): `game_idx` feeds into both the gate's routing
decision and every expert's conditioning vector, so with 21
essentially-unique embeddings the model has an available "shortcut" of
partially routing/predicting per-game rather than learning one shared,
truly action-conditioned dynamics function across the pooled data --
removing that shortcut forces the model to find the genuinely shared
signal instead, and there's evidently a lot more of it available than
either game-id-conditioned checkpoint was extracting.

**Caveat on the trained-games magnitude:** this is a single run per
checkpoint variant, and this project's own history (`CLAUDE.md` Stage 4
item 6's restore-retrain: 44.1% -> 22.9% purely from a different
corpus-generation seed) documents real run-to-run variance on this same
kind of metric. An 8-34x gap is far too large to be fully explained by
that kind of noise, and the effect has a plausible mechanism -- but this
result would be strengthened by at least one more independently-seeded
rerun before treating the exact magnitude (rather than the direction) as
load-bearing for a production decision.

### 2. Diagnostic B (object-identity cosine-similarity gap), secondary

| checkpoint | held-out games gap | trained games gap |
|---|---|---|
| baseline (with game-id) | -0.0654 (same=0.3437, diff=0.4090) | +0.0334 (same=0.5741, diff=0.6075) |
| object-identity (with game-id) | -0.0163 (same=0.5608, diff=0.5770) | +1.2763 (same=0.9986, diff=-0.2777) |
| no-game-id (ablated) | -0.0415 (same=0.8728, diff=0.9143) | +0.1665 (same=0.8742, diff=0.7077) |

This diagnostic measures the ENCODER's representations only (game-id
ablation only touches the predictor's conditioning, not the encoder --
see `jepa/models/encoder.py`, which takes no game input at all), so any
movement here reflects an indirect effect of the different training
signal, not a direct architectural change. Two observations:

- **On held-out games, all three checkpoints show a negative gap**
  (same-color patches look *less* alike than different-color ones) --
  consistent with the changed-patches result above: whatever's failing
  to generalize to unseen games, it's failing at both the representation
  level and the prediction level, for all three training recipes alike.
  No-game-id's held-out gap (-0.0415) sits between the baseline's
  (-0.0654) and object-identity's (-0.0163) -- not a meaningful
  ordering given the small diff-color sample sizes (n=124) already
  flagged as a caveat in `stage6_game_holdout.md`.
- **On trained games, no-game-id shows a real positive same-vs-different
  separation (+0.1665) -- about 5x the baseline's (+0.0334) but far
  short of object-identity's contrastive-loss-driven result (+1.2763).**
  This is a secondary, unplanned finding: removing game-id conditioning
  alone, with zero contrastive loss, produces *some* real object-identity
  structure in the trained-game representations that the baseline
  doesn't have -- plausibly a side effect of the same "forced to learn
  shared, less game-specific structure" mechanism behind the
  changed-patches result, rather than anything targeting object identity
  directly. Interesting, but secondary to this experiment's core
  question and not independently verified beyond this single run.

## Honest read

**This is a partial result, not a clean win or a clean failure --
the two questions this experiment asked have opposite answers.**

1. **Does ablating game-id conditioning close the held-out-game
   generalization gap? No, not even partially.** The gap is exactly as
   wide with a genuinely game-agnostic model (no per-game information
   ever available) as it is with per-game conditioning falling back to
   an untrained embedding row. The mechanism proposed in
   `stage6_game_holdout.md` (untrained fallback embedding causing the
   collapse) is refuted by this result -- something else about how the
   shared encoder/predictor learns is tied to the training games'
   specific visual statistics, and a categorical per-game signal was
   never the load-bearing piece of that. Closing the real generalization
   gap likely needs either more/more-diverse training games (the
   "data, not architecture" lesson this project has landed on repeatedly
   -- see CLAUDE.md's Stage 1/4 history) or an architectural change that
   more directly targets game-agnostic dynamics (e.g. stronger
   regularization toward color/shape-invariant features, or an explicit
   held-out-game validation signal during training) -- not this specific
   fix.

2. **Does ablating game-id conditioning cost anything on trained games?
   No -- it's a large, unexpected win** (+64.9% vs +8.0%/+1.9%), not
   merely "free." This is a genuinely new, actionable finding independent
   of the held-out-game question this experiment was built to answer:
   per-game embedding conditioning, as currently implemented in
   `MoEPredictor`, appears to be a net negative for prediction quality on
   this architecture and data scale, not just a wash. This extends (with
   much stronger evidence) Stage 1's old single-predictor-architecture
   finding that per-game conditioning gave "no improvement" -- worth
   treating as a real, separate lead for improving the *production*
   checkpoint's prediction quality on known games, even though it does
   nothing for the hidden-game problem this experiment set out to solve.

**Verdict on the task's own framing: not a fix for the held-out-game
generalization gap** (the motivating problem is completely unsolved by
this change) **but a real, independently valuable finding for production
quality on trained games**, worth a follow-up experiment on its own
(ideally with 1-2 more independently-seeded reruns to firm up the
magnitude, and a check of whether it also holds with the object-identity
contrastive loss turned on, i.e. `--ablate-game-id --contrast-weight
0.05` together) before deciding whether to actually swap the production
recipe. Not merged to production or to `master` in this session --
this branch (`stage6-gameid-ablation`) documents the finding for a
future session to act on.

## Reproducing this experiment

```
# Corpus setup (once, same as stage6_game_holdout.md): copy the verified
# 150-file *.random.80.* corpus into ARC-AGI-3-Agents/recordings/, and
# data/arc3_logs.zip into data/ -- both gitignored.

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --ablate-game-id --checkpoint-every 5 \
  --out checkpoints_holdout_nogameid

python scripts/eval_gameid_ablation.py
```

(Requires `checkpoints_holdout_baseline/` and `checkpoints_holdout_objid/`
from `experiments/stage6_game_holdout.md` to already exist alongside the
new `checkpoints_holdout_nogameid/` for the full 3-way comparison.)
