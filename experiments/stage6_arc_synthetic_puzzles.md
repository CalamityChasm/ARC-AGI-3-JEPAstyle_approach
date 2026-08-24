# Stage 6: ARC-3-shaped synthetic puzzles as a pretraining source

**Status: a clean negative result, consistent with the broader pattern.**
Genuinely ARC-3-*shaped* procedural puzzles (not a borrowed game engine
from an unrelated genre) do not close the held-out-games generalization
gap, at matched pretrain scale, on fold 1 of the established 5-fold split.
This is the first data-diversity attempt in this whole investigation built
directly out of ARC-3's own puzzle-logic vocabulary rather than adapting
an existing game (MiniGrid, Sokoban, MinAtar, Procgen, OpenSpiel) -- see
CLAUDE.md's Stage 6 addendum for the full prior history this extends.

## Motivation

CLAUDE.md's Stage 6 addendum documents 13 independent interventions
against the held-out-games gap by the time this experiment started: 7
conditioning/architecture fixes, and 6 data-diversity attempts spanning
MinAtar, Procgen, and a 26-game OpenSpiel roster (~2.14M transitions) --
12 failures, 1 modest success (test-time adaptation). Every one of the
data-diversity attempts borrowed an *existing* game engine from a
genre only loosely related to ARC-3's actual puzzle-logic character
(navigation, block-pushing, reflex arcade, board games). The doc's own
recommended next step explicitly named this gap: "ARC-3-*shaped* synthetic
puzzle generation (rather than borrowed game engines from other genres)
might be a better-matched data source than anything tried so far."

## What was built

`jepa/data/arc_synthetic_data.py`: five hand-built procedural puzzle
families, each its own `game_id`, grounded in direct visual inspection of
local games (`scripts/render_frames_png.py` renders of `bp35`, `ka59`,
`r11l`, `sp80`, `ft09`, `s5i5` -- see `experiments/stage6_bp35_ka59_mechanics.md`,
pulled from prior project history, for `bp35`/`ka59`'s own mechanics) and
`rules.md`'s Action Space description (RESET, ACTION1-5 simple,
ACTION6 complex-with-(x,y), ACTION7 -- semantics vary per game):

- **`arcsyn_route`** -- move a token through sparse random walls to a
  goal cell (routing/reaching -- echoes `ka59`'s dual-token routing and
  `r11l`'s path-tracing).
- **`arcsyn_toggle`** -- ACTION6 click toggles a cell + its 4 orthogonal
  neighbors between two colors (Lights-Out-style local cause-effect
  rule, a classic ARC mechanic).
- **`arcsyn_match`** -- ACTION6 click the on-board object whose color
  matches a displayed target swatch to clear it (object/color identity +
  counting).
- **`arcsyn_symmetry`** -- one action mirrors a half-drawn pattern onto
  its blank other half in a single shot (symmetry/pattern completion --
  one of the most common ARC-1/2 transformation types).
- **`arcsyn_enclosure`** -- move a cursor and trigger a flood-fill that
  only does anything if the cursor sits inside a fully walled-in region
  (spatial containment/enclosure).

Each type gets real procedural variation per episode (random grid size
8-20, random wall/object layout, random color draws) and its own
`game_id` -- applying the per-sub-game-id lesson the MinAtar retry
already established (pooling mechanically-dissimilar sub-games under one
shared id was a real, large confound there), not re-discovering it.
Verified `max(action_ids)=6 < NUM_ACTIONS=8` via an in-module self-check
before any training run touched the data (Sokoban's own 9-action overflow
crashing deep inside a CUDA kernel is exactly the failure this guards
against -- see CLAUDE.md's gotchas). Default
`episodes_per_type=168 * steps_per_episode=80 * 5 types = 67,200`
transitions, deliberately matched to `minigrid_data.py`'s own default
total so a pretrain-phase comparison holds total samples-seen constant
(the Procgen attempts' curriculum-imbalance confound, avoided from the
start rather than re-discovered).

`jepa/train_moe_predictor.py` gained `--pretrain-source
{minigrid,arc_synthetic}` (default `minigrid`, unchanged behavior) --
`arc_synthetic` swaps the new puzzles in for MiniGrid in the pretrain
phase, mutually exclusive rather than additive, for a clean like-for-like
swap rather than a scale-up.

## Methodology deviation, disclosed upfront

**Finetune epochs: 40, not the standard 60**, and via `--resume-from`
(a new capability added this session, see below) rather than a single
uninterrupted run. This environment killed the first from-scratch attempt
(`--pretrain-epochs 20 --epochs 60`) at ~45 minutes wall-clock, ~40/80
epochs in (some external wall-clock limit on this session's background
tasks, not a code crash -- the process had zero errors in its log, was
mid-epoch when killed, and `nvidia-smi`/CPU checks before the kill showed
it actively training). Its `--checkpoint-every 10` insurance meant the
completed 20-epoch pretrain phase plus 20 finetune epochs survived on
disk. Rather than discard that progress, added `--resume-from` to
`train_moe_predictor.py`: loads encoder/predictor weights and the exact
`game_vocab` from an existing checkpoint dir, skips the pretrain phase
(assumed complete), and resumes finetune from the epoch count recorded in
the checkpoint's `checkpoint_tag`. Resumed to a reduced target of 40 total
finetune epochs (20 more, ~20 min) rather than the original 60, to stay
comfortably under the observed kill threshold. The matched no-diversity
baseline below was trained fresh with the same 40-epoch target for a fair
comparison -- **the deviation from 60 epochs is symmetric across both
conditions being compared**, so it doesn't bias the comparison itself,
just makes the absolute numbers not directly comparable to older 60-epoch
entries in CLAUDE.md (only to each other, and to freshly-run baselines at
the same 40-epoch target, both done here).

Optimizer momentum is not preserved across the resume (a fresh AdamW) --
an acceptable minor blip at this coarse a resume granularity, not
expected to materially affect a 20-epoch continuation.

Only **fold 1** of the established 5-fold split was tested this round
(`r11l, bp35, m0r0, tr87, ka59` held out), not the full 5 folds --
consistent with how several other single-source diversity attempts in
this project's history (MinAtar, initial Procgen) were first reported on
one fold before broader confirmation. Flagged as a real limitation, not
glossed over -- see "Recommended next steps" below.

## Setup

Both checkpoints trained on the identical local-only ARC-3 corpus
(19,200 transitions across the 20 non-held-out games, no external
`arc-3-logs` data mixed in, `--exclude-games r11l,bp35,m0r0,tr87,ka59`),
`JEPA_NUM_WORKERS=0` (avoids a real per-epoch DataLoader-worker-respawn
overhead at this small a corpus scale -- discovered this session, see
below), 40 finetune epochs, `num_experts=8`, dense gate:

- **`checkpoints_arc_synthetic_fold1`**: 20 pretrain epochs on the new
  67,200-transition ARC-synthetic corpus, then 40 finetune epochs (20
  fresh + 20 resumed).
- **`checkpoints_holdout_baseline_fold1`**: 0 pretrain epochs (no
  synthetic data at all), 40 finetune epochs, otherwise identical --
  the matched no-diversity control, trained fresh this session rather
  than reused from an older run, so it's drawn from the exact same
  local-recordings corpus draw as the treatment condition.

Evaluated with `scripts/eval_arc_synthetic_holdout.py` (built this
session, mirrors `scripts/eval_game_holdout.py`'s established
changed-patches methodology): changed-patches pred-vs-identity MSE on
transitions from the 5 held-out games only (4,800 local transitions, none
ever seen by either checkpoint), plus a trained-games sanity check
(changed-patches on the 20 trained games, using the same
`game_vocab.get(id, 0)` fallback `hypothesis_agent.py` uses on a real
novel Kaggle game) to catch a degenerate/collapsed run before trusting
the held-out number.

## Result

| checkpoint | held-out overall | trained-games sanity |
|---|---|---|
| **arc-synthetic** (20 pretrain + 40 finetune) | **-0.26%** | **+12.96%** |
| no-diversity-baseline (0 pretrain + 40 finetune) | **-0.03%** | **-0.08%** |

Per-game held-out breakdown:

| game | arc-synthetic | baseline | note |
|---|---|---|---|
| `r11l` | -3.20% | +0.31% | |
| `bp35` | +0.34% | -0.06% | |
| `m0r0` | -0.69% | +0.05% | |
| `tr87` | -4.44% | +0.93% | |
| `ka59` | -16.71% | -0.55% | small-absolute-MSE artifact (see below) |

`ka59`'s -16.71% looks alarming in isolation but is exactly the
small-absolute-denominator pattern CLAUDE.md's own Stage 1 history
already warns about (item 5): absolute MSEs there are `pred=0.000602` vs.
`identity=0.000516`, a difference of `0.0000859` -- tiny in absolute
terms, large as a percentage of a tiny denominator. Not a uniquely bad
result, just a noisy metric on a low-signal game.

**Trained-games sanity check confirms the run is real, not degenerate.**
arc-synthetic pretraining clearly helped on *familiar* games (+12.96% vs.
the matched baseline's -0.08%, itself weak but consistent with this
project's own established "local-only, no external data, is a known-weak
recipe" history -- see CLAUDE.md Stage 1 item 5) -- so real learning
happened, the checkpoint isn't collapsed, and the encoder/predictor
genuinely benefited from the synthetic pretraining on the games it had
seen during finetuning.

**The held-out result does not close the gap.** -0.26% sits squarely
inside the same near-zero noise band every other intervention in
CLAUDE.md's Stage 6 addendum has landed in:

| source | fold-1 held-out changed-patches |
|---|---|
| No-diversity 5-fold mean | -0.30% (std 0.66%, range -1.8% to +0.11%) |
| MiniGrid-only baseline (established) | +0.01% |
| OpenSpiel, width=1.0 | -1.22% |
| OpenSpiel, width=2.0 | +0.02% |
| No-diversity baseline (this session, fresh) | -0.03% |
| **ARC-synthetic puzzles (this session)** | **-0.26%** |

The fresh no-diversity baseline trained this session (-0.03%) replicates
the established 5-fold band well, confirming this session's setup is
consistent with prior methodology, not an artifact of a different corpus
draw or epoch count. ARC-synthetic's -0.26% is worse than the MiniGrid
baseline and OpenSpiel width=2.0, better than OpenSpiel width=1.0 --
in other words, unremarkable, not a standout in either direction.

## Interpretation

This is the same pattern the whole Stage 6 investigation keeps finding:
pretraining diversity (of any kind tried so far, including one now
genuinely built out of ARC-3's own mechanic vocabulary rather than
borrowed from elsewhere) measurably helps the model on *games it's
already seen* during finetuning, but that improvement does not transfer
to genuinely unseen games. Being closer in "shape" to ARC-3's puzzle
logic than MiniGrid/Sokoban/MinAtar/Procgen/OpenSpiel did not turn out to
be the missing ingredient either.

One real difference worth noting: this is a much *smaller*-scale
diversity attempt (67,200 transitions, same order as the original
MiniGrid recipe) than the later, larger pushes in this investigation
(the 358k-transition 29-game mix, the 2.14M-transition OpenSpiel roster)
-- so this result doesn't rule out that a much larger ARC-synthetic
corpus, or a wider variety of puzzle types beyond the 5 built here, might
behave differently. But given OpenSpiel's own scale-up (6 games/358k ->
26 games/2.14M transitions) didn't move its held-out number in a positive
direction either, scale alone is not a strong prior for a different
outcome here without first confirming the *direction* is right at the
current scale.

## Recommended next steps, not done this round

1. **A second-fold confirmation** (fold 2's 5 held-out games) before
   treating -0.26% as more than a single data point -- this project's own
   standing lesson (see the multi-fold-CV section, and OpenSpiel's own
   fold-1/fold-2 pair) is not to trust one fold alone.
2. **A wider or larger ARC-synthetic roster** if a second fold also comes
   back negative but the underlying hypothesis (genre/shape match matters)
   still seems worth another shot at a different scale -- 5 puzzle types
   is a small roster next to OpenSpiel's 26 games or MiniGrid's 21
   environments; more variety within the "genuinely ARC-3-shaped" category
   (e.g. object permanence/appearance-disappearance, multi-step causal
   chains, more elaborate symmetry variants) is untested.
3. Given the accumulating pattern (now 14+ interventions against this
   specific gap, 13+ failures, only test-time adaptation showing any real
   positive signal), it's reasonable to keep prioritizing test-time
   adaptation's own further development (larger adaptation budgets,
   per-game resets) over yet another pretraining-diversity variant, per
   CLAUDE.md's own running recommendation.

## Real infrastructure kept from this session, independent of the result

- `jepa/data/arc_synthetic_data.py` -- reusable, real procedural puzzle
  generator; cheap (67,200 transitions in ~2 seconds), independently
  useful for any future ARC-3-shaped-data experiment even though this
  round's result was negative.
- `--pretrain-source arc_synthetic` on `jepa/train_moe_predictor.py`.
- `--resume-from` on `jepa/train_moe_predictor.py` -- recovers a
  long-running training job from a `--checkpoint-every`-saved checkpoint
  after an interruption, without losing the completed pretrain phase.
  Reusable for any future long training run in this environment, not
  specific to this experiment.
- `scripts/eval_arc_synthetic_holdout.py` -- checkpoint-agnostic held-out
  evaluation (any `--checkpoint dir:label` pairs), with a trained-games
  sanity check built in. Reusable for future held-out-games comparisons.

## A new environment gotcha found this session

**This session's background bash tasks appear to have a wall-clock kill
limit somewhere north of ~45 minutes.** A from-scratch training run
(`--pretrain-epochs 20 --epochs 60`, ~1 min/epoch observed, ~80 min
total expected) was killed by the environment partway through, with no
error in its own log -- confirmed via direct process inspection (the
training process was gone, `nvidia-smi`/CPU showed it had been actively
working right up to the point checked) that this was an external kill,
not a crash. Any long training run in this environment should either (a)
budget comfortably under this limit, or (b) use `--checkpoint-every` +
the new `--resume-from` to make an interruption cheap rather than fatal.
Also worth checking `JEPA_NUM_WORKERS=0` for training runs at this
project's typical small-corpus scale (tens of thousands of transitions,
not the hundreds-of-thousands-plus scale the existing `JEPA_NUM_WORKERS`
gotcha was written for) -- `num_workers=4, persistent_workers=False`
respawns 4 fresh worker processes (each re-importing the whole
`train_moe_predictor` module, including `gym_sokoban`) every single
epoch, and direct timing this session found `JEPA_NUM_WORKERS=0` no
slower per-epoch at this scale while avoiding that respawn overhead
entirely.
