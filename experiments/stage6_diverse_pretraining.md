# Stage 6 experiment: does MinAtar pretraining (more diverse synthetic data) close the held-out-ARC-games generalization gap?

**Status: DONE (MinAtar only; Procgen not attempted -- see "Why Procgen
was skipped" below). Negative result: adding MinAtar as a second
synthetic pretraining source alongside MiniGrid does not close the
held-out-ARC-games generalization gap, and on this one fold it is
directionally slightly worse than the MiniGrid-only baseline, both on
the held-out games and on the standard (trained-games) sanity check.
This is the 8th independent intervention this project has tried against
this specific gap, and the 8th to fail to close it.**

## Motivation

CLAUDE.md's "Stage 6 addendum" documents 7 independent, well-targeted
interventions against the held-out-ARC-games generalization collapse
(the world model shows ~0% changed-patches improvement over identity on
any local game it wasn't trained on, confirmed via 5-fold
cross-validation): game-id ablation, an encoder change-sensitivity
audit, an anti-collapse residual loss, simulated training-time game-id
dropout, and three different continuous-conditioning mechanisms (a
recurrent hidden-state, single-frame content conditioning, multi-
transition episode-context pooling). All 7 converged on the same null
result. The working conclusion was that this is a **data-bound** limit
-- the training corpus (20-25 ARC-3 games, +/- MiniGrid pretraining)
doesn't contain enough distinct causal mechanics to generalize
zero-shot to an unseen game, echoing Stage 4's own finding that MiniGrid
pretraining (genuinely new mechanics), not more ARC-3-only data or loss
tuning, was what fixed *that* stage's problem (gate collapse).

This experiment is the first "more/more-diverse pretraining data" lever
tried specifically against the held-out-game generalization gap (as
opposed to Stage 4's original gate-specialization problem, which
MiniGrid did fix). **MinAtar** (`pip install MinAtar`) is a clean-room,
no-ROM reimplementation of 5 classic-Atari-style games (breakout,
asterix, freeway, seaquest, space_invaders) as small
grid-based multi-channel binary environments -- architecturally much
closer to this project's discrete color-grid representation than real
Atari's RGB frames, and containing genuinely different mechanics
(ball-and-paddle physics, lane-crossing timing, shoot-em-up
projectiles) than either ARC-3 or MiniGrid exercise. No copyrighted
ROM data is used anywhere in this pipeline (confirmed: MinAtar ships its
own from-scratch game logic, no ROM files are downloaded or referenced).

## What was built

`jepa/data/minatar_data.py` -- the MinAtar analogue of
`jepa/data/minigrid_data.py`/`jepa/data/sokoban_data.py`. Key details:

- **Translation**: each `(10, 10, C)` bool state (C = 4-10 channels,
  one bool plane per object type, e.g. breakout has
  `paddle/ball/trail/brick`, seaquest has 10 channels including
  oxygen/diver gauges) is collapsed to a single `(10, 10)` int grid
  (colors 0-15) via `channel_idx -> color channel_idx+1` (0 reserved for
  "nothing active"), with higher-index channels overwriting lower ones
  on any cell where more than one channel is simultaneously active.
  Verified directly (not assumed) that the max channel count across all
  5 games (seaquest, 10) stays comfortably under `NUM_COLORS=16`.
- **Action space**: verified directly via
  `minatar.Environment(...).num_actions()` that all 5 games expose a
  constant 6 actions (no-op/left/up/right/down/fire) -- well within
  `jepa/models/predictor.py`'s `NUM_ACTIONS=8`, so (unlike Sokoban's
  9-action space, see CLAUDE.md's Stage 4 item 8 gotcha) **no action-id
  remapping was needed here**, and this was checked *before* any
  training run, not discovered via a CUDA crash.
- **Shared `game_id="minatar"`** across all 5 sub-games (mirroring
  MiniGrid's own choice), on the reasoning that the action *interface*
  is byte-identical across all 5 games (unlike MiniGrid, where "forward"
  still means different displacement depending on current facing) --
  documented in the module docstring as a deliberate choice, not a
  default left unexamined.
- **Data volume**: default `episodes_per_game=160`, `steps_per_episode=80`
  -> 5 * 160 * 80 = 64,000 transitions, chosen to land in the same order
  of magnitude as MiniGrid's own default (21 envs * 40 * 80 = 67,200)
  despite MinAtar having far fewer distinct sub-environments -- a
  controlled "similar data budget, different mechanics" comparison, not
  a data-volume confound.
- Sanity-checked before any training run (per CLAUDE.md's own gotcha
  about Sokoban's silent-until-a-CUDA-crash action-space overflow):
  `max(action_ids) < NUM_ACTIONS` (5 < 8), `patch_change_mask`/
  `arc3_frame_to_tensor` compatibility with the smaller-than-canvas
  10x10 grids, changed-frame rate (~75% on a smoke sample, healthy).

`jepa/train_moe_predictor.py` -- added `--minatar-episodes-per-game N`
(0 = skip, matching the existing `--sokoban-episodes-per-config` pattern)
so MinAtar transitions are added to the synthetic pretrain phase
alongside MiniGrid (and, if used, Sokoban) via the exact same two-phase
pretrain-then-finetune curriculum, sharing one encoder/predictor/
optimizer/game-vocabulary.

`scripts/eval_diverse_pretraining.py` -- adapted from
`scripts/eval_multifold.py`, comparing two normally-game-id-conditioned
checkpoints (neither is the `--ablate-game-id` variant, so no
`force_zero_game_idx` handling is needed) on a fold's held-out games.

## Method: controlled ablation, mirroring Stage 4's Sokoban methodology exactly

Two checkpoints trained via `jepa.train_moe_predictor`, differing
**only** in whether MinAtar pretraining data is added, using fold 1's
exact held-out games (`r11l, bp35, m0r0, tr87, ka59`) and recipe from
`experiments/stage6_multifold_generalization.md` so the held-out-games
evaluation is directly comparable to that document's own fold-1 numbers:

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_baseline

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --minatar-episodes-per-game 160 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_minatar
```

**Confirmed byte-identical shared-corpus transition counts before
trusting the comparison** (directly from both runs' own log output):
9,600 local ARC-3 transitions, 33,998 external `arc-3-logs` transitions,
67,200 MiniGrid transitions -- identical across both runs. The only
difference: the MinAtar run additionally generated 64,000 MinAtar
transitions (22 games in the shared vocab vs. 21 for the baseline, the
one extra entry being `"minatar"`).

Corpus setup used a canonical, isolated copy of the verified 150-file
`*.random.80.*` local recordings corpus (copied from the project's own
`E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\` backup into this
worktree's own `ARC-AGI-3-Agents/recordings/`, rather than the shared
main checkout's live `recordings/` directory, which currently holds a
different, non-canonical mix of files from other concurrent sessions'
work) -- this reproduces the exact 9,600-local-transition/20-game count
`stage6_multifold_generalization.md` itself reports for fold 1, directly
verified before launching either training run.

Both runs used `JEPA_NUM_WORKERS=0` (shared/contended GPU, per this
project's established gotcha) and ran as detached background processes,
polled to completion. Wall-clock: baseline ~100 minutes, MinAtar variant
~127 minutes (pretrain-phase epochs take roughly 2x as long with MinAtar
added, since the pretrain-phase corpus nearly doubles from 67,200 to
131,200 transitions; the finetune phase, unaffected by MinAtar, took
about the same time in both runs).

## Results

### Standard-corpus sanity check (does MinAtar help or hurt on the 20 trained games, matching Stage 4's own metric)

Final-epoch (60/60) changed-patches improvement over identity, on the
held-out-*transitions* validation split of the 20 *trained* games (not
the 5 held-out *games* -- this is the same in-distribution check Stage 4
always used):

| variant | pred_changed_mse | identity_changed_mse | improvement (epoch 60) | improvement (mean, epochs 51-60) |
|---|---|---|---|---|
| baseline (MiniGrid-only) | 0.00194 | 0.00201 | +3.5% | +4.0% |
| MiniGrid+MinAtar | 0.00125 | 0.00125 | +0.0% | +2.1% |

Both numbers are noisy epoch-to-epoch (consistent with this project's
own repeated observation that this metric swings meaningfully run to
run and epoch to epoch -- see CLAUDE.md's Stage 1 item 5 and the
multifold experiment's own `ft09` outlier) -- the 10-epoch trailing mean
is a steadier read than any single epoch. On both readings, **MinAtar
does not help the standard-corpus metric either; it's directionally
somewhat worse** (+4.0% -> +2.1% on the 10-epoch mean). Note the
baseline number here (+3.5-4.0%) is itself much smaller than Stage 4
item 6's original MiniGrid-only result on the *full* 25-game corpus
(+44.1%) -- expected, since this is a smaller 20-game corpus (5 games
excluded) with a different vocab size, not a like-for-like comparison to
that number; the baseline-vs-MinAtar comparison *within this experiment*
is the fair one.

### The test that matters: fold-1 held-out-games generalization

Using the identical evaluation methodology and held-out game set as
`experiments/stage6_multifold_generalization.md`'s fold 1
(`scripts/eval_diverse_pretraining.py`, natural `game_vocab.get(id, 0)`
fallback for the never-seen held-out game_ids, matching
`hypothesis_agent.py`'s real production behavior on a novel Kaggle
game):

| variant | pred_changed_mse | identity_changed_mse | improvement over identity (n=1881) |
|---|---|---|---|
| baseline (MiniGrid-only) | 0.040336 | 0.040285 | **-0.1%** |
| MiniGrid+MinAtar | 0.006381 | 0.006296 | **-1.4%** |

Per held-out game:

| game | baseline | MiniGrid+MinAtar |
|---|---|---|
| r11l | -0.1% | **-10.5%** |
| bp35 | -0.1% | -0.3% |
| m0r0 | -0.1% | -2.6% |
| tr87 | -0.4% | -0.3% |
| ka59 | +0.1% | -1.1% |

Two things worth calling out honestly:

1. **The baseline (-0.1%) closely reproduces the previously-published
   fold-1 baseline number (+0.01%) from the multifold experiment** --
   both are ~0%, well within that experiment's own established
   fold-to-fold noise band (mean -0.30%, std 0.66pp across 5 folds).
   This is a useful cross-check that this experiment's freshly-retrained
   baseline checkpoint (a different random seed/training run than the
   original fold-1 checkpoint) reproduces the same qualitative
   "collapse to ~identity parity" finding, not just a coincidence of
   one specific checkpoint.
2. **Adding MinAtar does not close the gap, and lands somewhat more
   negative (-1.4% vs -0.1%) than the baseline on the exact same
   held-out games in the same run** -- a same-fold, paired comparison,
   which is the more informative read than comparing against a
   different fold's own noise band. The -1.3 percentage-point gap
   between the two variants is modestly larger than the multifold
   study's own established per-fold std (0.66-0.76pp) but not
   dramatically so -- consistent with "no real effect, within extended
   noise" rather than a clear, alarming regression. `r11l`'s -10.5% is
   the one number that stands out individually (both checkpoints have
   n=480 changed patches for this game, not a tiny-sample artifact the
   way CLAUDE.md's `ft09`/`vc33`/`s5i5` caveat describes) -- worth
   flagging as the most concrete single data point against MinAtar
   helping, though it's one game out of five and could still be noise
   at this sample size (n=1 fold).

## Verdict

**Negative result. MinAtar pretraining does not close the
held-out-ARC-games generalization gap on this one fold, and if
anything is directionally worse than the MiniGrid-only baseline on both
the standard-corpus sanity check and the held-out test.** This is the
8th independent, well-targeted intervention this project has tried
against this specific gap (following the 7 documented in CLAUDE.md's
Stage 6 addendum: game-id ablation, encoder audit, anti-collapse loss,
simulated unfamiliarity, and three continuous-conditioning mechanisms)
and the 8th to fail to move it.

Unlike the 7 prior interventions (all architecture/conditioning
changes), this was the first *data-diversity* lever tried against this
*specific* gap -- and it still didn't help, despite MinAtar containing
genuinely different causal mechanics (ball-paddle physics, timed lane
crossing, projectile combat) than anything in ARC-3 or MiniGrid, and
despite Stage 4's own precedent that a new-mechanics data source
(MiniGrid) was exactly what fixed *that* stage's different problem
(gate collapse). Read together with Stage 4's Sokoban ablation (also
negative, also a new-mechanics synthetic source, also added on top of a
working MiniGrid baseline), the emerging pattern is that **not every
addition of "more diverse data" helps** -- MiniGrid's original win
(Stage 4 item 6) may have been doing much of its work by being cheap,
high-changed-rate, consistent-action-semantics data in a regime that
badly needed exactly that (Stage 4's specific gate-collapse problem),
rather than "diversity" in the abstract generalizing to every other
problem more data might plausibly help with.

**A real limitation of this specific result: only 1 of 5 folds was
tested.** The task's own guidance ("if time allows, validate on 1-2 more
folds before trusting any positive result") is explicitly gated on a
*positive* result needing extra scrutiny against a lucky single-fold
draw -- this result is negative and closely matches the already
5-fold-validated baseline pattern (CLAUDE.md's own multifold study), so
the same asymmetric skepticism doesn't obviously apply here in the same
way. But it remains true that this is one fold, one training run per
variant, no replicate seeds -- a future session with more time could
still usefully check 1-2 more folds (particularly given `r11l`'s
notably negative single-game number) before treating "MinAtar doesn't
help, and might mildly hurt" as fully settled rather than "the single
most likely reading of the evidence so far."

## Why Procgen was skipped

The task's own scoping was explicit: attempt Procgen "if MinAtar shows
real promise and you have time left." MinAtar did not show promise --
it did not close the held-out-games gap and was directionally negative
on both the standard-corpus check and the held-out check. Per that
explicit gating, Procgen (a strictly larger lift: RGB observations
needing a real color-quantization translation step, unlike MinAtar's
already-discrete grid) was not attempted this session. `jepa/data/
minatar_data.py` and the `--minatar-episodes-per-game` flag remain
available and reusable if a future session wants to revisit MinAtar
with more folds, more pretraining epochs, or a different data-volume
ratio before concluding this data source is a dead end -- none of that
further tuning was attempted here, consistent with this project's own
practice of not chasing further gains on an already-negative,
already-diagnosed result without new evidence motivating it.

## Reproducing this experiment

```
# Corpus setup: copy the verified 150-file *.random.80.* corpus from
# E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\ into
# ARC-AGI-3-Agents/recordings/, and data/arc3_logs.zip into data/ --
# both gitignored. pip install MinAtar into the project venv.

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_baseline

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --minatar-episodes-per-game 160 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_minatar

python scripts/eval_diverse_pretraining.py --fold 1 \
  --heldout-games r11l,bp35,m0r0,tr87,ka59 \
  --baseline-ckpt checkpoints_diverse_baseline --minatar-ckpt checkpoints_diverse_minatar
```

Each training run took roughly 100-130 minutes on a shared RTX 2070
(`JEPA_NUM_WORKERS=0`, both runs launched concurrently as detached
processes -- 2 concurrent GPU jobs, matching this project's established
concurrency cap). `eval_diverse_pretraining.py` runs in well under a
minute.
