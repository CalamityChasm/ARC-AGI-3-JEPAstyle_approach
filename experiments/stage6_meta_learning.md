# Stage 6 meta-learning: does explicitly training for post-adaptation performance beat a normally-trained checkpoint?

**Status: DONE. The standard-dose recipe is a clean negative (post-
adaptation performance not better than the normally-trained baseline).
A 3x-higher-dose variant (more Reptile updates/epoch, no epsilon
annealing) DOES show a real post-adaptation edge over baseline at the
representation level (+1.28% pooled / +0.98% simple-mean held-out
improvement vs baseline's +0.66%/+0.78%) -- but a preliminary n=8
agent-level backtest shows no detectable difference from the already-
published baseline+TTA-ON numbers on levels completed (the more robust
metric), with the same single held-out game (`r11l`) the only one either
configuration has ever solved. One real implementation bug (representation
collapse from a pure ANIL split) was found and fixed along the way.**

## Motivation

Every intervention in CLAUDE.md's Stage 6 addendum targeted a *frozen,
zero-shot* forward pass -- 14 failures. The one exception, test-time
adaptation (`stage6-test-time-adaptation-agent`,
`jepa/test_time_adapter.py: TestTimeAdapter`), showed that a few real
AdamW steps on a held-out game's own observed transitions, applied to a
small ANIL-style parameter subset (~33.8K params: each MoE expert's LAST
Conv2d + the gate's LAST Linear), genuinely improves prediction quality
on that game (+0.84% mean held-out changed-patches at n=200 adaptation
transitions, K=5/STEPS=8/LR=5e-5). But the checkpoint that adaptation
starts from was never trained with adaptability as a goal -- just
ordinary supervised training. This experiment builds a Reptile
(first-order) meta-learning objective that explicitly optimizes the SAME
adapter subset for post-adaptation performance, and checks whether
starting `TestTimeAdapter` from a meta-learned base does better than
starting it from the existing normally-trained base.

## Why Reptile, not MAML

Full (second-order) MAML backpropagates through the inner-loop
optimization itself -- more compute, and a real implementation-
correctness risk (differentiating through an AdamW inner loop is easy to
get subtly wrong). Reptile only needs the inner-loop *endpoint*, no
second-order gradients. Given this project's hardware (one RTX 2070) and
the priority on correctness over squeezing out marginal sample
efficiency, Reptile was the right default; nothing found while building
this suggested MAML would have been worth its extra complexity here.

## Design

`jepa/train_meta_predictor.py` (new script). Standard MiniGrid-pretrain +
ARC-finetune curriculum, identical to `jepa/train_moe_predictor.py`'s own
recipe in every respect except the ARC-finetune phase, where the adapter
subset ("head" -- the exact same params `TestTimeAdapter` adapts,
imported directly from `jepa/test_time_adapter.py: get_adapter_params` so
the two can't drift apart) gets an ADDITIONAL periodic Reptile nudge on
top of the ordinary per-batch joint gradient step every parameter
receives:

1. Sample a batch of training-pool ARC-3 games (never the held-out fold,
   never MiniGrid/external -- see "Reptile task pools" below).
2. For each, snapshot the head, run real AdamW inner-loop steps (default:
   the SAME K=5-derived STEPS=8/LR=5e-5 operating point
   `stage6-test-time-adaptation-agent` validated) on ONLY that game's own
   transitions, record the delta (adapted - snapshot), restore the
   snapshot.
3. Average the deltas across the sampled game batch, then interpolate the
   REAL head toward that average by `epsilon` (linearly annealed to 0
   over the ARC-finetune phase by default -- the standard Reptile
   schedule).

## A real bug found and fixed: pure ANIL split causes representation collapse

The first full implementation excluded the head entirely from the
ordinary joint optimizer (textbook ANIL: head updated ONLY via Reptile).
The resulting checkpoint's `val_pred_mse` and `val_identity_mse` shrank
together in lockstep from ~0.0027 to ~0.00005 over 60 epochs, staying
nearly equal at *every* epoch -- never opening the real, stable gap the
baseline recipe shows throughout (baseline epoch 60: pred=0.00070 vs
identity=0.00087). Root cause: with the head frozen at small near-random
values, the predictor structurally cannot produce a meaningful nonzero
residual (`feat + residual ~= feat`), so the only way left for ordinary
joint SGD to reduce the loss is for the ENCODER itself to make a frame's
before/after encoding trivially close together -- `variance_regularizer`
only floors per-channel std *across the batch*, it says nothing about
temporal sensitivity within one state's own before/after pair, so
nothing punishes this shortcut once the head can't supply real dynamics.
This is the same failure shape as Stage 1's original "predictor learns to
approximate identity" bug, one level further out.

**Fix:** the head now gets the same ordinary per-batch joint gradient
step as everything else (the SAME `joint_opt` built for the MiniGrid
phase is reused, unmodified, for the ARC phase too -- matching
`train_moe_predictor.py`'s own optimizer-reuse pattern), with the Reptile
update layered on top as a periodic additional nudge rather than being
the head's only training signal. This is a legitimate, well-precedented
Reptile variant (auxiliary regularizer on top of ordinary training, not
the sole training signal for the affected params), not a watered-down
compromise -- verified via a 6-epoch smoke test (healthy "predictor
starts worse than identity, improves toward and past it" shape, matching
baseline's own early-epoch curve) before committing to the full 60-epoch
retrain.

## Corpus / recipe (fold-1 split, matching `experiments/stage6_game_holdout.md` exactly)

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 10 --out checkpoints_holdout_baseline

python -m jepa.train_meta_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --checkpoint-every 10 --meta-iters-per-epoch 20 --meta-tasks-per-batch 4 \
  --inner-steps 8 --inner-lr 5e-5 --inner-batch-size 16 --meta-epsilon 1.0 \
  --out checkpoints_meta_fold1
```
27,900 local + 33,998 external ARC-3 transitions (20 training-pool games,
`r11l`/`bp35`/`m0r0`/`tr87`/`ka59` excluded from every source), 67,200
MiniGrid pretrain transitions, 21-entry game vocab (`checkpoints_holdout_baseline`
was not persisted from the original `stage6-game-holdout` session --
gitignored, not transferred -- so it was retrained fresh here from the
documented command; final trained-games changed-patches for this fresh
reproduction, +36.5% on its own internal val split, is in the same
ballpark as that session's own number, consistent with this project's
established run-to-run corpus-draw variance, not a regression).

Reptile task pools (ARC-finetune phase only, train split only): 20
distinct games, pool sizes 1,138-5,173 transitions/game.

## Results

`scripts/eval_meta_learning.py` (new): mirrors
`scripts/test_time_adaptation.py`'s methodology exactly (same per-file
stream/eval split, same K/steps/LR semantics) for direct comparability.

### 1. Zero-shot (no adaptation) on the 5 held-out games -- full population, n=1,881

| checkpoint | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| baseline | 0.072646 | 0.072507 | **-0.19%** |
| meta-reptile | 0.012811 | 0.012769 | **-0.33%** |

Both at ~identity parity, consistent with every other checkpoint tested
under CLAUDE.md's Stage 6 addendum -- **meta-training alone, with no
adaptation applied at eval time, does not close the zero-shot gap.**
Expected (plausible but not guaranteed per the task's own framing) --
this experiment's real test is #3 below, not this one.

### 2. Trained-games zero-shot sanity check, n=898 (8-game probe: `ft09`, `s5i5`, `vc33`, `ar25`, `cd82`, `cn04`, `lp85`, `sp80`)

| checkpoint | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| baseline | 0.002880 | 0.006166 | **+53.29%** |
| meta-reptile | 0.001232 | 0.002286 | **+46.10%** |

A real but moderate cost on trained-game accuracy from the periodic
Reptile nudge (-7.2 percentage points, roughly a 13% relative reduction)
-- not catastrophic, comparable in spirit to `TestTimeAdapter`'s own
documented trained-game interference cost from adaptation itself.

### 3. Post-adaptation on the 5 held-out games, K=5/STEPS=8/LR=5e-5, n=200 -- the real test

| checkpoint | simple per-game mean improvement | pooled (transition-weighted) improvement |
|---|---|---|
| baseline | **+0.72%** | **+0.64%** |
| meta-reptile | **+0.57%** | **+0.42%** |

Per-game (n_observed=200):

| game | baseline | meta-reptile |
|---|---|---|
| `r11l` | +1.18% | -0.06% |
| `bp35` | +0.59% | +0.41% |
| `m0r0` | +1.32% | +0.51% |
| `tr87` | +0.04% | +0.61% |
| `ka59` | +0.46% | +1.39% |

The baseline's own reproduction here (+0.72% simple mean) lands close to,
slightly below, `stage6-test-time-adaptation-agent`'s originally-reported
+0.84% at the same K/steps/LR/n=200 point -- expected run-to-run variance
from a freshly-regenerated corpus (this project's own well-established
pattern), and a useful sanity check that this reproduction is drawing
from the same underlying effect, not something different.

**The central finding: post-adaptation improvement from the meta-learned
base is NOT larger than from the normally-trained baseline -- if
anything, it is somewhat smaller on both the simple-mean and pooled
summary statistics.** Per-game, the two checkpoints trade wins (meta
ahead on `tr87`/`ka59`, baseline ahead on `r11l`/`bp35`/`m0r0`) rather
than one uniformly beating the other, and the gaps involved (fractions of
a percentage point on both sides) are small relative to the run-to-run
variance this project has already documented for this exact metric. **By
this experiment's own stated standard ("if post-adaptation performance
... isn't meaningfully better ... the meta-learning objective hasn't
earned its complexity") -- it hasn't, at this dose.** Not a "meta-hurts"
finding either; the honest read is "no detected benefit," not "detected
harm," given the small magnitudes involved.

## High-dose ablation: 3x more Reptile updates/epoch, epsilon annealing disabled

The standard-dose negative result (above) is confounded by dose: the fix
that avoided representation collapse (giving the head an ordinary
joint-SGD update every batch, not just the periodic Reptile nudge) may
have diluted the Reptile signal too much to matter -- at the standard
dose, the head receives one Reptile-averaged nudge (mean delta norm
~0.01-0.05) roughly every ~500+ ordinary joint-SGD batches (~2,500+
batches/epoch, 5 Reptile updates/epoch), with `epsilon` also annealing to
0 by the final epoch -- meaning the LAST several epochs are, for the
head, effectively pure joint training with zero meta-pressure left. To
separate "this design has a real ceiling" from "the standard dose was
just too weak," retrained with `--meta-iters-per-epoch 60` (3x more
Reptile updates/epoch: 15/epoch instead of 5) and `--no-epsilon-anneal`
(constant `epsilon=1.0` for all 60 epochs, not decayed to 0) --
everything else identical (same corpus, same exclude-games, same
inner-loop K=5-derived STEPS=8/LR=5e-5 operating point).

Training was healthy throughout (`mean_epsilon=1.0000` confirmed constant
every epoch as configured; `mean_delta_norm` stayed in the 0.02-0.05
range across all 60 epochs, not collapsing; `changed-patches` pred beat
identity at every epoch, e.g. epoch 60: pred=0.00328 vs identity=0.00802,
consistent with the fixed design's healthy shape, not the earlier
collapse signature).

### Zero-shot and trained-games (no adaptation)

| checkpoint | zero-shot held-out improvement | trained-games improvement |
|---|---|---|
| baseline | -0.19% | +53.29% |
| meta-reptile (standard dose) | -0.33% | +46.10% |
| meta-reptile (high dose) | **-0.69%** | **+74.69%** |

Zero-shot stays near parity (expected -- not the point of this design).
Trained-games accuracy is *higher* for the high-dose checkpoint than
either the standard-dose meta checkpoint or the baseline itself in this
particular run -- a real number from this run, though with only one
training run per condition it isn't independently confirmed against
run-to-run corpus variance the way this project's larger sweeps are;
noted as a bonus observation, not the focus of this ablation.

### Post-adaptation at n=200 (K=5/STEPS=8/LR=5e-5) -- the central test

| checkpoint | simple per-game mean | pooled (transition-weighted) |
|---|---|---|
| baseline | +0.78% | +0.66% |
| meta-reptile (standard dose) | +0.54% | +0.42% |
| **meta-reptile (high dose)** | **+0.98%** | **+1.28%** |

(Baseline's numbers shifted marginally from the first table earlier in
this doc, e.g. +0.72%->+0.78% simple mean -- this is the SAME baseline
checkpoint re-evaluated, not retrained; the small shift is from
stochastic mini-batch resampling inside the adaptation steps themselves,
same source of noise `experiments/stage6_test_time_adaptation_agent.md`
already documents for this exact metric.)

Per-game (n_observed=200):

| game | baseline | meta (standard) | meta (high dose) |
|---|---|---|---|
| `r11l` | +1.49% | -0.13% | **-1.39%** |
| `bp35` | +0.61% | +0.40% | **+1.18%** |
| `m0r0` | +1.27% | +0.63% | **+2.75%** |
| `tr87` | +0.09% | +0.50% | **+2.99%** |
| `ka59` | +0.44% | +1.28% | **-0.63%** |

**At 3x the dose, the meta-learned checkpoint DOES beat baseline on both
post-adaptation summary statistics** (+1.28% pooled / +0.98% simple mean
vs baseline's +0.66%/+0.78%) -- unlike the standard dose, which trailed
baseline on both. This is a real, directionally-consistent effect, not
noise dressed up as one: the high-dose checkpoint wins decisively on 3 of
5 games (`bp35`, `m0r0`, `tr87` -- all by a wider margin than baseline's
own gains on those same games) and loses on 2 (`r11l`, `ka59`, both
worse than baseline). The pooled statistic favors high-dose more strongly
than the simple mean because `bp35`'s absolute MSE scale is far larger
than the other games' (identity~0.165 vs. identity~0.001-0.018
elsewhere), so its outsized high-dose gain dominates the transition-
weighted average -- worth knowing when reading "+1.28%" as a headline
number, since it isn't evenly earned across games.

**Honest read: the negative standard-dose result WAS a dosing artifact,
not a fundamental ceiling for this design** -- a real, if modest and
unevenly-distributed, post-adaptation improvement exists at a high
enough Reptile dose. This is consistent with, not contradicting, the
standard-dose section above: that result correctly showed the standard
dose specifically doesn't work, not that Reptile meta-learning can't work
here at all.

## Preliminary agent-level backtest (high-dose checkpoint, n=8)

Since the high-dose result looks genuinely promising at the component
level, ran the agent-level backtest the task calls for in that case.
Trained a matching `value_head.pt` against
`checkpoints_meta_fold1_highdose/encoder_moe.pt` first (the same
methodology note from `experiments/stage6_test_time_adaptation_agent.md`
applies: `Hypothesis`'s value head must be trained against the SAME
encoder it's paired with, or a latent-space mismatch reproduces Stage 5's
original bug) -- `val_mse` (0.0068-0.0117 across 20 epochs) sits close to
the zero-baseline (0.0030), the same honest "barely distinguishable from
predict-zero" limitation this project has already documented for every
value head trained on this reward density; not a new problem.

Swapped `checkpoints_meta_fold1_highdose`'s 4 files into `checkpoints/`
(this project's established swap-and-restore convention), ran
`scripts/run_scorecard.py --agent hypothesis --game
r11l,bp35,m0r0,tr87,ka59` x8 with `HYPOTHESIS_TEST_TIME_ADAPT=1` (K=5/
STEPS=8/LR=5e-5, the defaults, matching every other TTA backtest in this
project), restored the production checkpoint immediately after.

| repeat | score | levels completed | game(s) solved |
|---|---|---|---|
| 1 | 0.0000 | 0 | -- |
| 2 | 0.0197 | 1 | `r11l` |
| 3 | 0.4797 | 1 | `r11l` |
| 4 | 0.0000 | 0 | -- |
| 5 | 0.0000 | 0 | -- |
| 6 | 0.0000 | 0 | -- |
| 7 | 0.0156 | 1 | `r11l` |
| 8 | 0.0000 | 0 | -- |

Mean score 0.0644, mean levels 0.375, total levels 3, all 3 completions
on `r11l` (the only held-out game either configuration has ever solved,
same as every prior comparison in this project's Stage 6 history).

**Comparing directly against `stage6-test-time-adaptation-agent`'s
already-published baseline+TTA-ON numbers at the identical n=8, same 5
games, same protocol (TTA ON: mean score 0.00473, mean levels 0.375,
total levels 3):** total levels and mean levels completed are **exactly
tied** (3/3, 0.375/0.375) between the meta-learned-high-dose checkpoint
and the original baseline checkpoint, both under TTA. Mean score differs
substantially (0.0644 vs 0.00473) but this is driven entirely by one
outlier run (repeat 3's 0.48) -- the exact "mean score dominated by a
single high-efficiency completion, not a systematic effect" pattern this
project's own `stage6_test_time_adaptation_agent.md` and
`stage6_budget_x_checkpoint.md` already documented for this identical
metric. Levels completed (the outlier-resistant metric this project has
repeatedly preferred for exactly this reason) shows **no detectable
difference** between the two checkpoints at this sample size.

**This is explicitly preliminary, per the task's own framing.** n=8 is
the same small sample size this project has now hit a real "component
improved, agent-level result didn't move" pattern with three separate
times this session alone (Stage 5's teacher-policy value head, the
original `stage6-test-time-adaptation-agent`'s own backtest, the
novelty-aware beta cap's n=30 retraction of an n=8 finding) -- a real
representation-level gain of this size is not guaranteed to be
detectable in 8 binary-ish trials on an already-sparse metric (both
conditions solve a level in well under half of all runs, on one game out
of five). Not evidence the component-level gain is fake; evidence this
particular test doesn't have the power to confirm or deny it. A larger
sample (25-30 repeats, matching this project's own standing
recommendation for exactly this class of problem) would be needed for a
real answer.

## Working interpretation

The core finding survives the dosing check: **a Reptile meta-learning
objective explicitly targeting post-adaptation performance CAN produce a
checkpoint that adapts better than a normally-trained one -- but only at
a high enough dose, and the gain (while real and directionally
consistent at the representation level) is modest, unevenly distributed
across games, and did not show up as a detectable agent-level win at
n=8.** The practical takeaway for a future session: the standard dose
used in most meta-learning papers' analogous settings was not the right
default here; if this direction is revisited, start from the high-dose
recipe (`--meta-iters-per-epoch 60 --no-epsilon-anneal`) rather than
re-deriving that finding, and prioritize either a larger agent-level
backtest sample or an even higher dose (this ablation only tried one
step up from standard, not a full sweep) before concluding further.

## Reproducing this experiment

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 10 --out checkpoints_holdout_baseline

python -m jepa.train_meta_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --checkpoint-every 10 --out checkpoints_meta_fold1

# High-dose variant (the one that actually beats baseline post-adaptation):
python -m jepa.train_meta_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --checkpoint-every 10 --meta-iters-per-epoch 60 --meta-tasks-per-batch 4 \
  --inner-steps 8 --inner-lr 5e-5 --inner-batch-size 16 --meta-epsilon 1.0 \
  --no-epsilon-anneal --out checkpoints_meta_fold1_highdose

python scripts/eval_meta_learning.py   # add the highdose dir to CHECKPOINTS first

# Agent-level backtest (high-dose checkpoint, swap into checkpoints/ first):
python -m jepa.train_value_head --epochs 20 \
  --encoder checkpoints_meta_fold1_highdose/encoder_moe.pt \
  --out checkpoints_meta_fold1_highdose
# ... copy encoder_moe.pt/moe_predictor.pt/value_head.pt/game_vocab_moe.json/
#     moe_training_meta.json into checkpoints/, backing up the originals first
$env:HYPOTHESIS_TEST_TIME_ADAPT = '1'
python scripts/run_scorecard.py --agent hypothesis --label heldout_tta_on_metahd_r1 --game r11l,bp35,m0r0,tr87,ka59
# ... repeat x8, then restore the backed-up production checkpoint files
```
(`JEPA_NUM_WORKERS=0` recommended on a shared/contended GPU box, per
CLAUDE.md's own gotcha.) Each training run took roughly 65-90 minutes on
a shared RTX 2070; `eval_meta_learning.py` runs in a few minutes; the
8-repeat agent backtest took about 30-45 minutes total.
