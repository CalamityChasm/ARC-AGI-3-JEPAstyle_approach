# Stage 6 meta-learning: does explicitly training for post-adaptation performance beat a normally-trained checkpoint?

**Status: DONE (standard-dose recipe). A clean negative result on the
central question, plus one real implementation bug found and fixed along
the way. A higher-dose follow-up is in progress to check whether the
negative result is a dosing artifact or a real ceiling.**

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

## Agent-level backtest: not run

The task's own framing was conditional ("if the result looks genuinely
promising, also run a real agent-level backtest"). Since the component-
level result shows no advantage for the meta-learned checkpoint over the
existing baseline, an agent-level backtest would not be informative here
-- there is no local signal for it to confirm or fail to confirm, and
this project has repeatedly found that agent-level backtests need a real
component-level effect behind them to have any chance of showing
something at practical sample sizes (Stage 5's teacher-policy value head,
`stage6-test-time-adaptation-agent`'s own agent-level check, the novelty-
aware beta cap's n=30 retraction). Skipped per the task's own stated
scope, not from time pressure alone.

## Working interpretation

The fix required to avoid representation collapse (giving the head an
ordinary joint-SGD update every batch, not just the periodic Reptile
nudge) may have diluted the Reptile signal too much to matter: at the
standard dose, the head receives one Reptile-averaged nudge (mean delta
norm ~0.01-0.05) roughly every ~500+ ordinary joint-SGD batches (~2,500+
batches/epoch, 5 Reptile updates/epoch), with `epsilon` also annealing to
0 by the final epoch -- meaning the LAST several epochs are, for the
head, effectively pure joint training with no meta-pressure left at all.
It's plausible ordinary joint SGD's own optimization pressure simply
outweighs (or erases, epoch over epoch) whatever the Reptile step nudges
toward. A higher-dose ablation (more Reptile updates/epoch, no epsilon
annealing) is in progress to check whether this is a genuine ceiling for
this design or a dosing artifact that a stronger nudge would move --
appended below once complete.

## Reproducing this experiment

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 10 --out checkpoints_holdout_baseline

python -m jepa.train_meta_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --checkpoint-every 10 --out checkpoints_meta_fold1

python scripts/eval_meta_learning.py
```
(`JEPA_NUM_WORKERS=0` recommended on a shared/contended GPU box, per
CLAUDE.md's own gotcha.) Each training run took roughly 65-90 minutes on
a shared RTX 2070; `eval_meta_learning.py` runs in a few minutes.
