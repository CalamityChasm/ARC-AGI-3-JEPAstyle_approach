# Stage 6 experiment: does Procgen (maze, heist) pretraining close the held-out-ARC-games generalization gap?

**Status: DONE. Negative result, and a sharper one than MinAtar's: adding
Procgen pretraining data doesn't just fail to close the held-out-games
gap, it actively collapses the model's representational dynamic range,
making it *worse than identity even on the trained games* (+48.2% ->
-2.0%) -- a regression the MinAtar experiment didn't show. Diagnosed the
mechanism directly (not just observed the symptom): batch-level feature
variance is not collapsed, but the encoder's sensitivity to frame-to-
frame *change* collapses ~150x in absolute scale and never recovers
during the 60-epoch ARC-3 finetune phase, unlike the baseline's own
checkpoint, which passes through an even deeper collapse during MiniGrid-
only pretraining and then fully re-expands within 1 epoch of seeing
ARC-3 data. This is the 9th independent intervention against the
held-out-ARC-games gap (following CLAUDE.md's 7 conditioning/architecture
attempts and the MinAtar data-diversity attempt) and the 9th to fail to
close it -- but it also surfaces a distinct, more actionable failure mode
(pretrain-phase corpus imbalance) worth flagging for anyone revisiting
multi-source synthetic pretraining.**

## Motivation

CLAUDE.md's "Stage 6 addendum" documents 7 independent conditioning/
architecture interventions, all failing to close the held-out-ARC-games
generalization collapse (the world model shows ~0% changed-patches
improvement over identity on any local game it wasn't trained on,
confirmed via 5-fold cross-validation), plus one prior data-diversity
attempt: **MinAtar** pretraining (`experiments/stage6_diverse_pretraining.md`),
also negative -- and, on the standard-corpus sanity check, directionally
worse than the MiniGrid-only baseline. MinAtar's own 5 sub-games
(breakout, asterix, freeway, seaquest, space_invaders) are reflex/
physics-driven arcade games -- a plausible poor genre match for ARC-3's
static, discrete puzzle-logic mechanics, and a live open question flagged
in that document's own follow-up section.

This experiment tests **Procgen** (`pip install procgen`), one of plan.md's
*originally*-intended pretraining sources (alongside MiniGrid/Sokoban/
Crafter) that had never been attempted before this session, specifically
using the two Procgen environments most genre-matched to ARC-3:
**`maze`** (navigate a procedurally-generated maze to a goal) and
**`heist`** (multi-step key/lock/gem puzzle logic, no combat) -- both
state-transformation/navigation puzzles, not reflexes, unlike MinAtar or
most of Procgen's other 14 games (bigfish, bossfight, chaser, dodgeball,
starpilot, etc., all combat/reflex-driven).

## What was built

`jepa/data/procgen_data.py` -- the Procgen analogue of
`jepa/data/minigrid_data.py`/`sokoban_data.py`/`minatar_data.py`, with
two real structural differences from all three:

1. **A genuine environment/dependency incompatibility, not just a missing
   pip install.** Procgen has no PyPI wheel for Python 3.13 (this
   project's main venv) -- confirmed directly against PyPI's file
   listing: procgen 0.10.7's newest wheels top out at `cp310`. Rather
   than downgrading the whole project's Python version, a **dedicated
   Python 3.10 venv** (`C:\pgvenv`) was used *only* to run
   `generate_transitions()` once via `scripts/generate_procgen_corpus.py`,
   producing a pickled transitions cache
   (`data/procgen_corpus.pkl`, gitignored) that `jepa/train_moe_predictor.py`
   loads via a new `--procgen-cache PATH` flag -- no `procgen` import is
   ever needed in the main training venv. This mirrors Stage 3's
   documented Mamba->GRU substitution: a deliberate, documented
   workaround for a genuine local-environment incompatibility, not a
   silent shortcut. (Task scoping originally suggested a live
   `--procgen-episodes-per-env` flag matching the other three sources'
   pattern; the cache-based `--procgen-cache` flag is the necessary
   adaptation once this incompatibility was discovered.)
2. **Real color quantization**, since Procgen's native observation is raw
   RGB `(64, 64, 3)` uint8 (confirmed via `ProcgenGym3Env(...).ob_space`)
   -- unlike every sibling source's already-categorical grid. A
   dependency-free numpy k-means (`k=16`, no new `scikit-learn`
   dependency) fits one shared 16-color palette on a pooled sample of
   `maze`+`heist` frames; every frame is then quantized via nearest-color
   (Euclidean, RGB) bucketing against that fixed palette. One shared
   palette (not per-env) mirrors MiniGrid's own "one consistent
   vocabulary across sub-environments" choice.

**Action space**: Procgen's raw action space is `Discrete(15)` for every
game (confirmed via `.ac_space`) -- Procgen's own `BaseProcgenEnv.
get_combos()` source (read directly, not guessed) shows action ids 0-8
are a 3x3 grid of directional movement combos (4 being the true no-op)
and ids 9-14 are single special-purpose buttons (jump/shoot/interact)
that `maze`/`heist` don't use. `MOVEMENT_COMBOS = [0,1,2,3,5,6,7,8]`
drops just the no-op -- exactly Sokoban's own precedent ("nothing
changes" is already the trivial baseline everywhere else in this
pipeline) -- leaving exactly 8 real movement actions, a clean fit under
`NUM_ACTIONS=8` with no arbitrary truncation of anything mechanically
relevant to these two games. Verified directly (`max(action_ids) < 8`)
in `scripts/generate_procgen_corpus.py` before any training run, per
CLAUDE.md's own Sokoban-CUDA-crash gotcha.

**game_id scheme: one id per environment** (`procgen_maze`,
`procgen_heist`), not one shared id for all of Procgen -- `heist` has
real state-dependent interaction (walking into a key/lock/gem changes
what touching the next one does) that `maze` doesn't have at all, so
forcing one embedding to serve both would repeat the exact
inconsistent-semantics confound Stage 1 and `sokoban_data.py` already
reasoned through for other sources.

**Data volume**: `steps_per_env=33_600` x 2 envs = 67,200 total
transitions, matching MiniGrid's own total budget and mirroring
`minatar_data.py`'s explicit "similar total data budget, different
mechanics" reasoning -- concentrated into 2 environments rather than
spread across 21 (MiniGrid) or 5 (MinAtar), so deeper-but-narrower
coverage than either sibling source. Frame-level changed rate: **80.7%**
(54,239/67,200) -- higher than MiniGrid's ~43% or Sokoban's ~47%, since a
movement action in `maze`/`heist` almost always displaces the agent by
at least one pixel.

`jepa/train_moe_predictor.py` -- added `--procgen-cache PATH` (loads a
pre-generated cache rather than generating live, unlike the
`--sokoban-episodes-per-config`/`--minatar-episodes-per-game` pattern, for
the dependency reason above), wired into the same two-phase
pretrain-then-finetune curriculum alongside MiniGrid/Sokoban/MinAtar.

`scripts/eval_procgen_pretraining.py` -- direct sibling of
`scripts/eval_diverse_pretraining.py` (the MinAtar version), renamed
flags for clarity (`--procgen-ckpt` instead of `--minatar-ckpt`).

`scripts/eval_procgen_standard_corpus.py` -- full-float-precision
standard-corpus (trained-games) check, needed because the training log's
5-decimal print rounds the Procgen checkpoint's collapsed-magnitude
numbers to `0.00000`. `scripts/diagnose_procgen_collapse.py` /
`scripts/diagnose_procgen_sensitivity.py` -- the two collapse-mechanism
diagnostics described in "Diagnosing the mechanism" below.

## Method: controlled ablation, mirroring the MinAtar experiment's methodology exactly

Two checkpoints trained via `jepa.train_moe_predictor`, differing
**only** in whether Procgen pretraining data is added, using fold 1's
exact held-out games (`r11l, bp35, m0r0, tr87, ka59`) and the identical
recipe `experiments/stage6_diverse_pretraining.md` used, so results are
directly comparable to that document's own numbers:

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_baseline

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --procgen-cache data/procgen_corpus.pkl \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_procgen
```

Corpus setup: the verified canonical 150-file `*.random.80.*` local
recordings corpus (copied from `E:\ARC-AGI-3-JEPAstyle_data\
recordings_archive\`, matching `stage6_diverse_pretraining.md`'s own
corpus provenance) plus `data/arc3_logs.zip` (external `arc-3-logs`
corpus, hardlinked from the main checkout). Both runs launched as
detached background processes, `JEPA_NUM_WORKERS=0` (shared/contended
GPU, per this project's established gotcha).

**Confirmed byte-identical shared-corpus transition counts before
trusting the comparison** (directly from both runs' own
`moe_training_meta.json`): 9,600 local ARC-3 transitions, 33,998 external
`arc-3-logs` transitions, 67,200 MiniGrid transitions -- identical across
both. The only difference: the Procgen run additionally used 67,200
Procgen transitions (23 games in the shared vocab vs. 21 for the
baseline, the two extra entries being `procgen_maze`/`procgen_heist`).
Both runs: `--pretrain-epochs 20 --epochs 60`, 60 total wall-clock minutes
each (baseline finished first; Procgen run's pretrain phase took longer
per epoch since its corpus is double the size at the same epoch count).

## Results

### Standard-corpus sanity check (does Procgen help or hurt on the 20 trained games)

Read directly from both checkpoints via `scripts/eval_procgen_standard_corpus.py`
(rebuilds the exact same ARC-3 train/val split `train_moe_predictor.py`'s
own `train()` used, at full float precision -- the training log's
5-decimal print rounds the Procgen run's collapsed-magnitude numbers to
`0.00000`, unreadable at that precision):

| variant | pred_changed_mse | identity_changed_mse | improvement over identity |
|---|---|---|---|
| baseline (MiniGrid-only) | 4.198e-03 | 8.111e-03 | **+48.2%** |
| MiniGrid+Procgen | 5.298e-05 | 5.196e-05 | **-2.0%** |

**This is a real regression, not just "no help": the Procgen-added
checkpoint is worse than identity on the standard, in-distribution,
trained-games check** -- and the baseline's own +48.2% here is
considerably healthier than the MinAtar experiment's own baseline number
on the identical fold-1 recipe (+3.5-4.0%), so this isn't a case of a
weak baseline making the comparison look worse than it is; the baseline
checkpoint in this run is a strong, clearly-working reference point that
Procgen pretraining then breaks. Note the two variants' absolute error
magnitudes differ by roughly 150x (8.1e-3 vs 5.2e-5) -- see "Diagnosing
the mechanism" below; this scale collapse, not just the sign flip, is
the more informative part of this result.

### The test that matters: fold-1 held-out-games generalization

Via `scripts/eval_procgen_pretraining.py`, same fold-1 held-out games as
every other Stage 6 experiment this session (`r11l, bp35, m0r0, tr87,
ka59`):

| variant | pred_changed_mse | identity_changed_mse | improvement over identity (n=1881) |
|---|---|---|---|
| baseline (MiniGrid-only) | 0.030355 | 0.029901 | **-1.5%** |
| MiniGrid+Procgen | 0.001603 | 0.001600 | **-0.2%** |

Per held-out game:

| game | baseline | MiniGrid+Procgen |
|---|---|---|
| r11l | -2.1% | -0.7% |
| bp35 | -0.7% | +0.2% |
| m0r0 | -1.5% | -0.3% |
| tr87 | -6.8% | -0.2% |
| ka59 | -4.0% | -3.0% |

Both variants collapse to ~identity parity on held-out games, consistent
with every prior intervention this project has tried against this gap
(CLAUDE.md's 7 conditioning fixes, plus MinAtar). The baseline's -1.5%
here is a bit more negative than the MinAtar experiment's own fold-1
baseline reading (-0.1%) but both sit within the multifold study's
established per-fold noise band (mean -0.30%, std 0.66pp) -- expected
run-to-run variance from a fresh retrain, not a new finding on its own.
**The more informative number here is the same ~19x absolute-scale
collapse seen on the standard-corpus check** (identity_changed_mse
0.0299 baseline vs. 0.0016 Procgen) -- present on held-out games too, so
it isn't specific to in-distribution data.

## Diagnosing the mechanism (not just the symptom)

The ~150x scale collapse was concerning enough to verify directly rather
than just report the percentage, using two diagnostics
(`scripts/diagnose_procgen_collapse.py`, `scripts/diagnose_procgen_sensitivity.py`):

1. **Batch-level feature variance is *not* collapsed** -- directly
   measured encoder output std across a held-out batch: baseline
   overall_std=1.997, Procgen overall_std=2.578 (Procgen's is actually
   *higher*). This rules out classic representation collapse (features
   going globally constant, Stage 1 item 7's original -- ultimately
   wrong -- hypothesis) as the explanation. The encoder still
   discriminates strongly between different game states.
2. **What *is* collapsed is temporal-change sensitivity specifically**:
   mean feature-space delta between `frame_t` and `frame_t+1` at
   genuinely pixel-changed patches is 0.001147 for baseline vs. 0.000008
   for Procgen (~140x smaller), and at unchanged patches 0.000292 vs.
   0.000002 (~140x smaller too) -- both shrink by roughly the same
   factor, so the *relative* contrast between changed and unchanged
   patches is roughly preserved (3.93x baseline vs. 3.29x Procgen -- the
   encoder still knows more changed than unchanged, just at a hugely
   compressed absolute scale). This is a different, narrower failure
   than global collapse: the encoder still discriminates *between
   different scenes* fine, but has lost most of its dynamic range for
   representing *how much a scene changed frame-to-frame*.
3. **Traced to a pretrain/finetune-phase interaction, not something
   Procgen's content does in isolation.** Comparing the 20-epoch
   pretrain-phase logs for both runs: **both** collapse similarly during
   pretraining itself (baseline's own `val_identity_mse` shrinks from
   0.00001 at epoch 1 to 0.00000 by epoch 6 and stays there through
   epoch 20 -- MiniGrid's simpler, much-more-static dynamics naturally
   drive this down over 20 epochs, a benign convergence pattern, not
   evidence of a problem by itself). **The difference is what happens
   next.** At ARC-3 finetune epoch 1, baseline's `val_identity_mse` jumps
   back up to a healthy 0.00801 (full recovery within a single epoch of
   real ARC-3 data) and stays in that healthy range for the remaining 59
   epochs. Procgen's own finetune epoch 1 only reaches 0.00009 -- roughly
   90x smaller than baseline's recovery -- and it never meaningfully
   recovers further across all 60 finetune epochs, ending at ~0.00001.
   **Doubling the pretrain-phase corpus (67,200 -> 134,400 transitions,
   MiniGrid+Procgen combined) at the same 20 pretrain epochs and the same
   60 ARC-3 finetune epochs pushes the shared encoder into a
   small-dynamic-range regime that 60 epochs of ARC-3 finetuning isn't
   enough to undo**, whereas the same finetune budget is clearly
   sufficient to undo MiniGrid-only pretraining's own (milder, or at
   least more escapable) version of the same convergence pattern.

**This reframes the finding in an important way.** The original
hypothesis behind this experiment was "MinAtar's genre mismatch (reflex-
arcade vs. ARC-3's puzzle logic) is why it failed; Procgen's maze/heist
are better genre-matched and might do better." The result doesn't
actually test that hypothesis cleanly: the failure mode found here looks
like a **pretrain-phase corpus-size/curriculum-balance problem**
(doubling the synthetic-pretrain corpus without adjusting epoch counts),
not obviously a content/genre problem -- it's plausible the same
collapse would occur adding *any* second 67,200-transition synthetic
source at this same epoch balance, genre-matched or not, and equally
plausible that Procgen's content is fine but got a curriculum that
doesn't suit it. This experiment cannot distinguish those two
explanations as run. A genre-match verdict would need a rerun with a
curriculum controlled for total pretrain volume (e.g. reducing
pretrain-epochs proportionally, or interleaving MiniGrid/Procgen within
the same total transition budget rather than concatenating both at full
size) -- not attempted this session, flagged as the natural next step if
Procgen is revisited.

## Verdict

**Negative result -- the 9th independent intervention against the
held-out-ARC-games generalization gap to fail, and the first of the two
data-diversity attempts (MinAtar, Procgen) to actively regress the
standard trained-games metric, not just fail to improve the held-out one.**
Unlike MinAtar (modest, direction-only regression: +4.0% -> +2.1% on the
standard check), Procgen pretraining flips the standard-corpus result
from strongly positive to negative (+48.2% -> -2.0%) and collapses the
model's representational dynamic range for frame-to-frame change by
~150x, verified directly via two independent diagnostics rather than
inferred from the headline percentage alone.

The mechanism traced above (pretrain/finetune-phase imbalance, not
obviously a genre-match problem) means this result is **not** strong
evidence against the original genre-matching hypothesis that motivated
testing Procgen over more MinAtar tuning -- it's evidence that
*concatenating a second full-size synthetic source into the pretrain
phase at unchanged epoch counts* is a bad recipe, a narrower and more
actionable finding. Combined with Sokoban's own negative result (Stage 4
item 8, a third synthetic-source addition, different diagnosed cause --
deadlock-polluted data) and MinAtar's (genre mismatch), the pattern
across all three "add a second/third synthetic source on top of
MiniGrid" attempts is now 3-for-3 negative, each for a different
diagnosed reason -- worth treating as a real caution against naively
stacking more synthetic sources into this project's current two-phase
curriculum shape, independent of whether the *content* being added is a
good genre match.

**Limitations of this specific result**: one fold (fold 1 only, matching
the MinAtar experiment's own scope decision for a negative result -- the
task's own guidance for extra-fold validation is explicitly gated on a
*positive* result needing scrutiny against a lucky draw, which doesn't
apply here); one training run per variant, no replicate seeds; the
curriculum-imbalance hypothesis above is a plausible mechanistic account
consistent with the direct evidence gathered, not independently verified
by a controlled curriculum-balance rerun. `jepa/data/procgen_data.py`,
`--procgen-cache`, and `scripts/eval_procgen_pretraining.py` all remain
available and reusable if a future session wants to revisit with a
volume-controlled curriculum before concluding Procgen's content itself
is unhelpful.

## Reproducing this experiment

```
# Corpus setup: copy the verified 150-file *.random.80.* corpus from
# E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\ into
# ARC-AGI-3-Agents/recordings/, and data/arc3_logs.zip into data/ --
# both gitignored.
#
# Procgen corpus generation needs a *separate* Python 3.10 venv (see
# jepa/data/procgen_data.py's module docstring and
# scripts/generate_procgen_corpus.py's docstring for exact setup steps
# -- procgen has no wheel for this project's main Python version):
#   C:\pgvenv\Scripts\python.exe scripts\generate_procgen_corpus.py \
#     --envs maze,heist --steps-per-env 33600 --out data\procgen_corpus.pkl

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_baseline

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --procgen-cache data/procgen_corpus.pkl \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_procgen

python scripts/eval_procgen_pretraining.py --fold 1 \
  --heldout-games r11l,bp35,m0r0,tr87,ka59 \
  --baseline-ckpt checkpoints_diverse_baseline --procgen-ckpt checkpoints_diverse_procgen

# Full-precision standard-corpus check + collapse diagnostics (JEPA_NUM_WORKERS=0
# avoids a Windows multiprocessing crash from spawning DataLoader workers
# outside an `if __name__ == "__main__":` guard):
JEPA_NUM_WORKERS=0 python scripts/eval_procgen_standard_corpus.py
python scripts/diagnose_procgen_collapse.py
python scripts/diagnose_procgen_sensitivity.py
```
