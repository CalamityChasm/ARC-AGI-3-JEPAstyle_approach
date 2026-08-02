# Stage 6 experiment: does data augmentation (color permutation) close the held-out-game generalization gap?

**Status: DONE. Negative result. Color-permutation augmentation does not
improve held-out-game generalization (-0.19% vs. baseline's +0.01%, both
inside the established ~0.7-point noise band) and measurably *hurts*
prediction quality on trained games (+7.97% -> -1.76% on the same local-
games check). This is the 14th independent intervention against the
held-out-games gap documented in `CLAUDE.md`'s Stage 6 addendum, and the
13th failure. Spatial (rotation/flip) augmentation was investigated but
deliberately not implemented -- see "Why spatial augmentation was
skipped" below.**

## Motivation

`CLAUDE.md`'s Stage 6 addendum documents 13 independent interventions
against the held-out-game generalization gap (the world model shows no
measurable prediction advantage over identity on any local ARC-3 game it
wasn't trained on), 12 of them negative, one (test-time adaptation) a
real but modest positive. The clearest concrete diagnosis of *why*:
`stage6-object-identity`'s same-color contrastive loss produced a huge
local win (+1.18 object-identity representation gap) that completely
reversed sign on held-out games (-0.02) -- direct evidence the model was
learning the *local* 25 games' own specific color statistics, not a
transferable notion of "same color = same object."

Despite that diagnosis, nobody had tried the single most standard fix for
exactly this failure mode: data augmentation. This project has never used
color-permutation or spatial (rotation/flip) augmentation anywhere in its
training pipeline. This experiment tests color-permutation augmentation
as a direct, well-established remedy for a model over-relying on
incidental, non-transferable statistics of its training distribution.

## Implementation

### Color-permutation augmentation

`jepa/data/trajectories.py`: `TransitionDataset` gained a `color_augment`
constructor flag. When set, every `__getitem__` call draws a fresh random
permutation of all `NUM_COLORS=16` color ids (`np.random.permutation(16)`)
and applies it *identically* to `frame_t` and `frame_t1` via the new
`_permute_frame_colors` helper, before either frame is converted to a
tensor. Applying the same permutation to both frames of one transition
keeps the causal "this action applied to this permuted-color state
produces that permuted-color next-state" relationship truthful -- only
the arbitrary color-id labels change, not the recorded dynamics. This
also leaves `patch_change_mask`'s changed/unchanged verdict exactly
unchanged, for a simple reason: a color permutation is a bijection
applied identically to both frames being compared, and equality between
two grids is preserved under any shared bijective relabeling of the
values being compared -- so it doesn't matter whether patch-change is
computed before or after permuting.

**Whether to include color 0 in the permutation: included, deliberately.**
Classic ARC-1/2 puzzles have an informal convention that color 0
(black) often represents background/empty space, which would be a
reasonable argument for holding it fixed to avoid introducing label
noise on a meaningfully-different color. But ARC-3 *game* frames have no
documented, cross-game convention that color 0 means anything in
particular -- each of the 25 (and ~110 hidden Kaggle) games defines its
own visual language over the same 16-color palette (`rules.md`'s own
framing: "Action semantics are per-game and must be discovered through
exploration -- not documented in advance" makes the same general point
about this project's local games not being a reliable guide to universal
conventions, and the same reasoning applies to color semantics). The
specific failure mode this augmentation targets -- the model memorizing
this project's own 25 games' specific color usage/statistics rather than
a transferable notion of color/object identity -- argues for treating
all 16 ids as equally arbitrary labels, not privileging one as special
without cross-game evidence that it deserves to be. This is a real
design choice, not a proven-optimal one; see "Honest limitations" below
for why it may have backfired, and holding color 0 fixed is flagged as a
natural, cheap follow-up ablation for a future session.

**Augmentation scope, deliberately narrow to isolate one variable:**
applied only to the ARC-3 fine-tuning phase's *training* split (local
recordings + external `arc-3-logs`), never to the MiniGrid pretrain
phase and never to validation. `jepa/train_moe_predictor.py`'s
`_make_loaders` was refactored to split train/val indices first (via the
same `random_split` call, same seed, on an unaugmented base dataset --
so the train/val partition itself is byte-identical to the
pre-refactor code), then build a *separate* `TransitionDataset` for the
training split with `color_augment=True` when requested. When
`--color-augment` is not passed, this refactor is a no-op: the training
split is the exact same `Subset` object the original code produced,
verified by construction (`train_ds = train_split` in that branch).
Validation always evaluates on real, unaugmented frames -- the same
standard this project's evaluation has always held (e.g. never training
the identity baseline itself) -- and MiniGrid pretraining is untouched
so the *only* deliberate difference between a baseline and
color-augmented checkpoint is what happens during ARC-3 fine-tuning.

New CLI flag: `--color-augment` on `jepa/train_moe_predictor.py`.

### Why spatial augmentation was skipped

Investigated directly before writing any code, per the task's own
explicit gating ("if you're not confident the transform is truthful,
it's better to skip"). `ACTION6` (the click action) is mechanically
straightforward -- rotating/flipping a frame requires the corresponding
`(x, y)` transform, which is safe and well-defined. The harder question
was whether any of `ACTION1`-`ACTION5`/`ACTION7` (the "simple" actions,
no coordinate) carry an implicit fixed spatial meaning that a rotation/
flip would need to relabel to stay truthful.

Checked three sources directly:

1. **The ARC engine's own schema** (`venv/Lib/site-packages/arcengine/enums.py`):
   `GameAction.ACTION1`-`ACTION5` and `ACTION7` are all typed
   `SimpleAction` (`game_id: str` only -- no coordinate, no direction
   field, no metadata of any kind). Nothing in the engine's own data
   model encodes a fixed spatial meaning for these ids.
2. **`rules.md`** (this competition's own rules document), Action Space
   section: *"ACTION1-ACTION5 | Simple actions (move/interact, meaning
   varies per game)"* and, explicitly: *"Action semantics are per-game
   and must be discovered through exploration -- not documented in
   advance."* This confirms directional meaning is neither universal nor
   knowable in advance for any specific game -- exactly the thing a
   correct spatial-augmentation relabeling scheme would need to know.
3. **This project's own history** (`CLAUDE.md`): no stage ever
   discovered or hardcoded a per-game action-to-direction mapping;
   Stage 1 item 4 found per-game action *embeddings* didn't even help
   accuracy, and every agent (`Curiosity`, `Memory`, `Hypothesis`) treats
   all simple actions as opaque, undifferentiated options to be
   discovered live, never assuming any of them means "up."

**The concrete correctness risk:** if a given game's `ACTION1` really
does mean, say, "move right" as a fixed property of that game's engine
logic (independent of the rendered frame), then keeping `action_id`
unchanged while rotating both `frame_t` and `frame_t1` by a fixed
transform *per transition* is still individually self-consistent (it's
a truthful relabeling of pixel coordinates for that one sample) -- but
across the *whole* augmented dataset, examples where two different,
independently-rotated frames happen to render similarly (a real
possibility on frames with large uniform/symmetric regions, which many
of these games have) could present the *same-looking* `frame_t` paired
with the *same* `action_id` but *different* required movement
directions in the target, purely because of which random rotation was
drawn for each sample -- an unresolvable ambiguity for a model whose
only inputs are `(frame_t, action_id, xy, game_id)`, none of which
record which transform was applied. Since this project has no reliable,
per-game ground truth for which of `ACTION1`-`ACTION5`/`ACTION7` (if any)
carry this kind of fixed spatial meaning, and no way to verify it in
advance across even the 25 local games, let alone ~110 largely-unseen
Kaggle games, implementing spatial augmentation would risk introducing
exactly the kind of subtly-incorrect training signal the task explicitly
called out as a reason to skip. **Decision: skipped.** If a future
session wants to revisit this, the lowest-risk version would restrict
augmentation to games/actions where `ACTION6` is the *only* available
action (no simple actions to worry about at all) -- a meaningfully
smaller but unambiguously-safe subset.

## Test protocol

Reused `stage6-multifold-cv`'s fold 1 exactly (`experiments/
stage6_game_holdout.md` / `stage6_multifold_generalization.md`): held-out
games `r11l, bp35, m0r0, tr87, ka59`; trained games the other 20
(`ar25, cd82, cn04, dc22, ft09, g50t, lf52, lp85, ls20, re86, s5i5, sb26,
sc25, sk48, sp80, su15, tn36, tu93, vc33, wa30`).

**Baseline checkpoint**: reused directly from `stage6-multifold-cv`
(`checkpoints_fold1_baseline`, copied byte-for-byte from that branch's
worktree) rather than retrained -- avoids ~80-100 minutes of redundant
GPU time for a checkpoint whose exact recipe and numbers are already
published. Verified before trusting it: `scripts/eval_augmentation.py`
reproduces `stage6_multifold_generalization.md`'s exact published
held-out number (**+0.01%**) to the decimal before any new training was
run, confirming the recordings corpus, checkpoint copy, and evaluation
logic were all correctly set up.

**Augmented checkpoint**: trained fresh on this branch, identical recipe
plus `--color-augment`:
```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --color-augment --checkpoint-every 5 \
  --out checkpoints_fold1_augment
```
20 MiniGrid-pretrain epochs (67,200 transitions, unaugmented) + 60
ARC-finetune epochs (9,600 local + 33,998 external `arc-3-logs`
transitions, held-out games excluded from both sources) -- confirmed via
`moe_training_meta.json` that this checkpoint's recipe is identical to
the baseline's in every recorded field except `color_augment` (`null`
vs. `true`). `JEPA_NUM_WORKERS=0` (shared/contended GPU, this project's
established practice). Completed cleanly in ~80 minutes, 60/60 epochs,
`checkpoint_tag: "final"`, no errors.

**Evaluation**: `scripts/eval_augmentation.py` (new, adapted from
`scripts/eval_multifold.py`) computes changed-patches (pred-vs-identity
MSE on changed 8x8 patches) on two populations for each checkpoint: (1)
the 5 held-out games (2,400 transitions, the number that actually
matters), and (2) the 20 trained games' local recordings (9,600
transitions, a regression check). Evaluation never applies color
augmentation to either checkpoint, regardless of how it was trained --
the point is to measure real-input-distribution performance.

## Results

### Held-out games (the number that matters)

| checkpoint | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| baseline (no augmentation) | 0.065562 | 0.065571 | **+0.01%** |
| color-augment | 0.016049 | 0.016018 | **-0.19%** |

Per-game:

| game | baseline | color-augment |
|---|---|---|
| r11l | -1.62% | -62.31% |
| bp35 | +0.04% | -0.13% |
| m0r0 | +0.18% | -0.61% |
| tr87 | +0.21% | -96.54% |
| ka59 | -0.88% | -334.60% |

The `r11l`, `tr87`, `ka59` triple-digit-negative-percent numbers under
color-augment are the same "tiny absolute-MSE denominator" artifact
`CLAUDE.md` (Stage 1 item 5, and again in the multifold experiment) has
repeatedly flagged: absolute MSEs for those games are ~2-4 orders of
magnitude smaller under the augmented checkpoint (e.g. `ka59`:
identity_mse 0.000004 vs. baseline's 0.010435) purely because the
augmented checkpoint's *overall* feature-space error scale is much
smaller (see the trained-games discussion below for why) -- a tiny
absolute swing there produces a huge relative swing. The pooled,
denominator-weighted number (-0.19%) is the honest headline figure; it
sits comfortably inside the ~0.7-percentage-point standard deviation
`stage6-multifold-cv`'s own 5 folds established for this exact
architecture/recipe with *no* intervention at all (mean -0.30%, std
0.66pp for the with-game-id variant). **Color-permutation augmentation
shows no real edge over the established near-zero baseline on held-out
games.**

### Trained games (regression check)

| checkpoint | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| baseline (no augmentation) | 0.001001 | 0.001088 | **+7.97%** |
| color-augment | 0.000610 | 0.000600 | **-1.76%** |

**This is a real, unexpected regression, not noise.** Two independent
signals agree on both direction and rough magnitude: (1) this pooled,
20-game local-recordings check (+7.97% -> -1.76%), and (2) the
training run's own internal validation metric (a 10%-held-out slice of
the combined local+external training corpus, always unaugmented per this
experiment's own design) drifting from a real early-training edge
(epoch 1: pred=0.27160 vs. identity=0.27825, **+2.4%**) down to
near-identity-parity by the end (epoch 60: pred=0.00182 vs.
identity=0.00183, **-0.5%**) -- a steady, monotonic erosion across all
60 epochs, not a late spike. Both checks used different data slices
(one external-inclusive/10%-split, one local-only/full-population) and
both tell the same story: color augmentation measurably degrades
prediction quality on the very games the model *does* have real signal
for, without buying any of that back on held-out games.

## Honest read

**Color-permutation augmentation does not close the held-out-game
generalization gap, and it costs real accuracy on trained games.** This
is a clean negative result on the primary question, not a mixed or
inconclusive one: the held-out number is statistically indistinguishable
from the established no-intervention baseline (both essentially zero,
well inside the known noise band), while the trained-game cost is large
and corroborated by two independent measurements.

**Why might this have backfired, rather than just done nothing?** The
most likely explanation, worth stating plainly rather than hand-waved:
specific color ids plausibly *are* a real, exploitable predictive signal
within this project's 20-25 trained games at this data scale (e.g. a
particular sprite reliably rendered in a particular color within a given
game) -- not a transferable notion of object identity, but a genuine
regularity the model can and does fit given enough data on a fixed set
of games. Randomizing color labels during training directly destroys
that regularity as a *learnable* signal without providing this
architecture, at this data scale (~55k combined ARC-3 transitions, 60
fine-tuning epochs), enough alternative structural/shape-based signal to
fully compensate within the same training budget -- so the predictor's
residual branch, over the course of training, increasingly retreats
toward the safe "predict no change" identity fallback (visible directly
in the internal validation curve above), the same failure shape Stage 1
originally diagnosed as "the predictor learns to approximate identity"
(`CLAUDE.md` Stage 1 item 8), here re-triggered by a different mechanism
(a destroyed label shortcut, rather than an inherently weak signal) but
landing at the same symptom. This also explains why held-out numbers
don't move: a predictor that's retreated toward identity trivially
"transfers" to held-out games in the sense that identity applies
everywhere equally, not because it learned anything more general about
game dynamics.

A second, more specific candidate explanation for *why this particular
augmentation choice* backfired rather than a milder version: **including
color 0 in the permutation** (this experiment's deliberate choice, see
"Implementation" above) means the majority-class background pixels
present in most frames get relabeled on every single training sample,
every epoch -- a substantially more aggressive perturbation than typical
ARC augmentation practice (which often holds 0 fixed). This wasn't
tested against a 0-held-fixed variant this session (out of scope for the
primary test as specified, and this session's negative headline result
didn't meet the task's own bar for triggering a second fold, let alone a
new ablation) but is flagged here as the most promising, cheap next
lever if a future session wants to isolate whether "permute all 16
colors" specifically (rather than color-permutation augmentation in
general) is the reason this backfired.

**Why fold 2 validation was not run:** the task's own protocol gates
second-fold validation on the augmented run showing "real, meaningful
improvement" on held-out games. It didn't -- the held-out result is
flat/negative, not positive, and sits inside the already-established
5-fold noise band from `stage6-multifold-cv` (which showed *every*
architectural/data intervention that session landing in the same
near-zero band regardless of what was tried). Spending another ~80
minutes of training to confirm a negative result that's already
consistent with 13 prior independent negative results on the same
question would not meaningfully change the conclusion. `checkpoints_
fold2_baseline` was copied over from `stage6-multifold-cv` in case a
future session wants to pick this back up cheaply (e.g. to test the
"hold color 0 fixed" variant, which *would* be a new hypothesis worth
a proper fold-1-then-fold-2 validation if it showed any real promise).

**What this does and doesn't establish.** This is the 14th independent
intervention this project has tried against the held-out-games gap
(counting `CLAUDE.md`'s own tally of 13 as of the start of this session)
and the 13th failure, joining conditioning fixes, architecture changes,
and five prior data-diversity attempts. It specifically rules out "the
model just needs to be discouraged from memorizing specific color ids"
as a sufficient fix on its own, at this data scale -- a real, useful
negative result, since it was the most standard, lowest-effort remedy
for exactly the failure mode this project's own diagnosis (the
object-identity checkpoint's collapsed representation gap) pointed at,
and it still didn't work. Combined with the accumulating pattern across
all 14 interventions, this strengthens (does not newly establish) the
working conclusion already in `CLAUDE.md`: the ceiling looks
data-bound -- specifically, bound by how many *genuinely different game
mechanics* the model has ever seen, not by which incidental statistics
(color ids, screen positions, or now confirmed: color-relabeling
robustness) it happens to be overfit to. Test-time adaptation remains
the only lever in this entire investigation that has shown any real,
positive, dialable signal.

## Reproducing this experiment

```
# Corpus setup (once, same as stage6_game_holdout.md): copy the verified
# 150-file *.random.80.* corpus into ARC-AGI-3-Agents/recordings/, and
# data/arc3_logs.zip into data/ -- both gitignored.

# Baseline: reuse checkpoints_fold1_baseline from stage6-multifold-cv
# (or retrain with that branch's own fold-1 command).

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --color-augment --checkpoint-every 5 \
  --out checkpoints_fold1_augment

python scripts/eval_augmentation.py --fold 1 \
  --heldout-games r11l,bp35,m0r0,tr87,ka59 \
  --trained-games ar25,cd82,cn04,dc22,ft09,g50t,lf52,lp85,ls20,re86,s5i5,sb26,sc25,sk48,sp80,su15,tn36,tu93,vc33,wa30 \
  --baseline-ckpt checkpoints_fold1_baseline \
  --augment-ckpt checkpoints_fold1_augment
```
(`JEPA_NUM_WORKERS=0` recommended on a shared/contended GPU box.) The
augmented training run took ~80 minutes on a shared RTX 2070 (contended
with at least one other concurrent agent session); `eval_augmentation.py`
runs in under a minute. Full raw numbers: `logs/augmentation_results.json`.
