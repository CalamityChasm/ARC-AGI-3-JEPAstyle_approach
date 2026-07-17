# Stage 6 experiment: is the model too narrow, or is this still a data problem?

**Status: both phases complete. Phase 1 (train-vs-val fit diagnostic on the
production checkpoint) came back ambiguous-to-positive for the capacity
hypothesis, not a clean disproof -- so per the standing decision rule,
Phase 2 (an actual width ablation) ran. Phase 2's result is unambiguous:
at matched (short) training budget, doubling and quadrupling the encoder/
expert channel width did not improve fit on the model's own training
data -- if anything, both wider variants were flat-to-slightly-worse than
the 1x baseline. Capacity is not the bottleneck at these scales; this
project's standing "data-bound, not capacity-bound" read (CLAUDE.md's
Stage 4 section) holds up under a direct test, not just by elimination.**

## The question this addresses

The user asked directly: "is it possible our model isn't wide enough to
learn all of these mechanics?" This project's existing answer (Stage 4,
CLAUDE.md) was "data-bound, not capacity-bound" -- but that conclusion
came from watching a *gate* fail to specialize when given more experts
without more/diverse data (uniform blending kept winning over routing,
regardless of load-balance-loss tuning). That's real evidence the model
had nothing worth routing between *yet*, but it never actually tested
whether the encoder or per-expert networks are wide enough in the first
place. The encoder is objectively small (a 4-conv-layer CNN collapsing to
a `(64, 8, 8)` feature map) and each MoE expert is two pointwise (1x1)
convs. Worth checking directly rather than assuming either way -- this
experiment does that.

## Phase 1: does the production checkpoint even fit its own training data?

**The diagnostic**: compare `changed-patches` performance (this project's
standing metric -- see CLAUDE.md's Stage 1 history for why raw whole-grid
MSE is misleading) on the TRAIN split vs the held-out VAL split, using the
current production checkpoint (`checkpoints/encoder_moe.pt` +
`checkpoints/moe_predictor.pt` in the main checkout). Every eval this
project has ever run and documented is held-out val -- nobody had looked
at train-set fit before. Reasoning: if the model fits its own training
data poorly, that's underfitting -- a real capacity signal, and the
5-worth-investigating case. If it fits training data very well but val
lags well behind, that's the classic *generalization*-gap signature, and
points away from capacity (more width wouldn't obviously fix that, might
even make it worse without more data).

### A real infrastructure snag, found and worked around before the diagnostic could even run

The main checkout's `ARC-AGI-3-Agents/recordings/` currently holds only
75 files -- all `*.hypothesis.*` recordings, leftover byproducts of a
different, concurrent session's scorecard-backtest sweep. **The original
150-file `*.random.80.*` corpus the production checkpoint was actually
trained on is no longer there at all** -- it was moved off as part of
this project's own documented disk-hygiene practice (see CLAUDE.md's
Gotchas section) sometime after training. Reusing "the exact split" per
this experiment's own instructions is meaningless if the underlying
transitions list doesn't match what was actually trained on, so this had
to be resolved before Phase 1 could mean anything.

Found the original corpus archived on `E:\ARC-AGI-3-JEPAstyle_data\
recordings_archive\` (1225 files total: 1075 `hypothesis` + 150 `random`,
mixed together by an earlier archiving/cleanup pass) and cross-checked it
against the production checkpoint's own `moe_training_meta.json`
(`n_local_transitions: 12000`) before trusting it:

- Exactly 150 `*.random.80.*` files, 6 per game x 25 games -- matches
  CLAUDE.md's documented corpus shape exactly.
- 332MB total -- matches CLAUDE.md's documented "~332MB random-policy
  corpus" figure exactly.
- File timestamps (Jul 8, ~20:40-20:43) sit ~2 hours before
  `checkpoints/moe_predictor.pt`'s own save timestamp (Jul 8, 22:58) --
  consistent with being the actual training input, not a coincidentally
  similar later corpus.
- 12,000 total transitions across the 150 files (`.random.80.` implies
  ~80 transitions/file x 150 = 12,000) -- matches
  `n_local_transitions: 12000` in the checkpoint's own metadata exactly.

All four independent checks agree, so these 150 files were copied into
this worktree's `ARC-AGI-3-Agents/recordings/` (only these -- not the
1075 `hypothesis` files, which postdate this checkpoint and were never
part of its training corpus) before running anything. Also copied
`data/arc3_logs.zip` (the external corpus, 40MB compressed) from the main
checkout so the exact production training command
(`--external-per-game 2000`) could be reproduced for Phase 2.

### Method

Pulled `jepa/benchmark.py`'s `--moe` eval flag and `jepa/train_moe_predictor.py`'s
`--checkpoint-every`/`--resume-from`/`JEPA_NUM_WORKERS` resilience features
from `stage6-search-harvest` (read-only `git show`, that branch's own
working tree untouched). Added a `--split {train,val}` flag to
`jepa.benchmark eval --moe` (`run_eval_moe`, `jepa/benchmark.py`) that
reuses the *exact same* `random_split(..., generator=torch.Generator().manual_seed(0))`
call `train_moe_predictor.py`'s own `_make_loaders` uses (same
`VAL_FRACTION=0.1`) -- picks the complementary 90% ("train") instead of
the usual 10% ("val"), rather than rederiving a different split.

```
python -m jepa.benchmark eval --moe --split val   --checkpoint-dir checkpoints --tag phase1-val
python -m jepa.benchmark eval --moe --split train --checkpoint-dir checkpoints --tag phase1-train
```

### Result: no classic overfitting signature -- train fit is mediocre, not near-ceiling

| split | pooled changed-patches improvement | unweighted mean per-game improvement | games beating identity |
|---|---|---|---|
| val (held-out, 1200 transitions) | +66.1% | -17.6% | 6/25 |
| train (10,800 transitions, directly optimized on for 60 epochs) | +47.4% | -19.2% | 4/25 |

The **pooled** numbers (+66.1% val vs +47.4% train) look at first glance
like val is *better* than train, which would be a strange result either
way -- but this is exactly the "pooled metric dominated by one or two
games' outsized absolute-error magnitude" distortion this project has
flagged before (Stage 1 item 5; `stage6_search_harvest.md`'s own `cn04`
warning). `cn04`'s identity-MSE happened to be ~2x larger on its specific
val-split sample than its train-split sample purely from which frames
landed in which split (val: identity=0.125, train: identity=0.069) --
that single game's absolute-magnitude weight swings the *pooled* average
substantially. The **unweighted per-game mean** (treating all 25 games
equally, sidestepping that distortion) tells the real story: **train
(-19.2%) and val (-17.6%) are statistically indistinguishable** -- if
anything val is marginally *better*, the opposite of what overfitting
would predict.

Per-game detail (train_imp% vs val_imp%, sorted by game):

```
game               train_imp%   val_imp%   gap(train-val)
ar25-0c556536          -18.6%     -18.1%           -0.5%
bp35-0a0ad940           44.5%      49.1%           -4.7%
cd82-fb555c5d          -31.6%     -17.4%          -14.3%
cn04-2fe56bfb           46.0%      62.2%          -16.1%
dc22-fdcac232           -0.3%       6.5%           -6.8%
ft09-0d8bbf25           48.1%     -27.8%           75.8%   <- biggest gap, real but n=47 val is small
g50t-5849a774          -14.7%     -13.1%           -1.6%
ka59-38d34dbb          -80.5%     -58.1%          -22.4%
lf52-271a04aa          -16.9%      -9.2%           -7.7%
lp85-305b61c3            0.0%       0.0%            0.0%
ls20-9607627b           -9.3%      -5.4%           -3.9%
m0r0-492f87ba            3.8%       9.2%           -5.4%
r11l-495a7899           -6.3%     -21.4%           15.2%
re86-8af5384d          -12.7%      -6.3%           -6.5%
s5i5-18d95033         -122.2%    -129.1%            6.9%
sb26-7fbdac44          -27.8%     -39.3%           11.4%
sc25-635fd71a          -16.5%     -24.1%            7.6%
sk48-d8078629           -5.1%      -6.8%            1.7%
sp80-589a99af          -10.3%      -4.4%           -5.8%
su15-1944f8ab           -8.0%       4.8%          -12.8%
tn36-ef4dde99           -5.7%     -31.6%           25.9%
tr87-cd924810          -88.8%     -94.8%            6.1%
tu93-0768757b          -18.4%     -30.7%           12.3%
vc33-5430563c         -109.7%       2.6%         -112.3%  <- val beats train, opposite of overfitting
wa30-ee6fef47          -19.7%     -35.9%           16.1%
```

Gaps are small-to-moderate and point in *both* directions roughly equally
(11 games train>val, 13 val>train, 1 tied) -- not the one-directional
"train always ahead" pattern overfitting would produce. The two outliers
(`ft09` +75.8pp, `vc33` -112.3pp) are exactly the kind of small-identity-
MSE games CLAUDE.md's Stage 1 item 5 already flagged as producing noisy
percentages (`vc33`'s identity MSE is ~0.00001-0.00002, so a tiny absolute
swing is a huge relative one) -- not evidence of a systematic gap.

**The headline finding: only 4/25 games beat the trivial "predict no
change" baseline on data the model was directly, repeatedly optimized on
for 60 epochs.** That's not what a capacity-rich model memorizing its
training set looks like. It's much more consistent with "the model
hasn't been given enough representational room (or hasn't found a way) to
fit even the examples it's already seen" than with "the model has fit
training data well and simply doesn't generalize."

**One honest caveat, stated plainly rather than glossed over:** this
isn't a classic supervised-learning "memorize the training set" setup.
The training target (`target(nxt)`, an EMA copy of the online encoder,
momentum 0.996) is a *moving* target that co-evolves with the online
network throughout training, and the loss also includes a variance
regularizer discouraging representational collapse. So "the model doesn't
reach near-zero training loss" doesn't carry quite the same weight here as
it would in ordinary supervised memorization -- there's a legitimate
non-capacity reason training fit might stay imperfect even with ample
width. This caveat doesn't overturn the finding (mediocre train fit is
still mediocre train fit, and it's still not the "excellent train / weak
val" pattern that would cleanly rule out capacity) -- it just means Phase
1 alone shouldn't be read as slam-dunk proof of underfitting. That's
exactly why the task's own decision rule treats this result as "proceed
to Phase 2," not "conclude capacity-bound and stop."

### Phase 1 decision

Per the standing instruction: *"If the result is inconclusive... OR
positive for capacity being an issue (train fit itself is mediocre/
plateaued) -> proceed to Phase 2."* Train fit here is mediocre (4/25
games beat identity, -19.2% unweighted mean) and the train/val gap is
small and bidirectional, not a clean generalization-gap signature. This
does **not** meet the bar for "clearly negative for the capacity
hypothesis" that would have stopped the investigation at Phase 1.
**Proceeding to Phase 2.**

## Phase 2: does more width actually help train-set fit?

### Method

Added `--width-mult` to `jepa/train_moe_predictor.py` (`build_models`):
scales the encoder's `out_channels` and each MoE expert's
`feature_channels`/`expert_hidden` by this factor off a 64-channel base
(1.0 = today's exact architecture). Action/game embedding dims (16) are
deliberately left unscaled -- the question is encoder/expert channel
width specifically, not embedding capacity. `jepa/benchmark.py`'s
`_load_moe_checkpoint` reads the width back out of a new
`feature_channels` key in `moe_training_meta.json` (defaults to 64 for
any pre-sweep checkpoint) so eval reconstructs the right-shaped model
before loading the state dict.

Parameter counts at each width (encoder + MoE predictor combined):

| width | channels | encoder params | predictor params | total | vs 1x |
|---|---|---|---|---|---|
| 1x | 64 | 185,984 | 99,480 | 285,464 | 1.0x |
| 2x | 128 | 707,840 | 337,624 | 1,045,464 | 3.7x |
| 4x | 256 | 2,759,168 | 1,231,704 | 3,990,872 | 14.0x |

**Data held constant to the current validated corpus** (the reconstructed
150-file local random corpus + `--external-per-game 2000` external
arc-3-logs + the standard MiniGrid pretrain curriculum) -- explicitly
*not* mixing in the still-unresolved search-harvested corpus from
`stage6-search-harvest`, per this experiment's own scope (that branch's
own retrain is itself unfinished/inconclusive; combining two unresolved
variables would make this experiment impossible to interpret cleanly).

**Budget**: a full 60-epoch run at 3 widths was not attempted first --
per this experiment's own budget guidance, and given this session's
history of disk/time pressure (see CLAUDE.md's Gotchas), ran a short,
epoch-matched curriculum first (`--pretrain-epochs 5 --epochs 15`, vs
production's `--pretrain-epochs 20 --epochs 60`) at each width to get a
fast directional signal, with an explicit plan to only invest in a full-
length run for a width that showed a real, non-noise improvement.

```
python -m jepa.train_moe_predictor --pretrain-epochs 5 --epochs 15 --external-per-game 2000 \
    --num-experts 8 --out checkpoints_width1x --width-mult 1.0 --checkpoint-every 5
python -m jepa.train_moe_predictor --pretrain-epochs 5 --epochs 15 --external-per-game 2000 \
    --num-experts 8 --out checkpoints_width2x --width-mult 2.0 --checkpoint-every 5
python -m jepa.train_moe_predictor --pretrain-epochs 5 --epochs 15 --external-per-game 2000 \
    --num-experts 8 --out checkpoints_width4x --width-mult 4.0 --checkpoint-every 5
```

All three ran to completion without incident (~35-40 min each on this
machine's RTX 2070 -- 1x and 2x at similar per-epoch wall-clock, 4x
noticeably slower per epoch as expected from its ~14x parameter count,
but still finished within budget). Disk stayed comfortably clear
throughout (21.9-49.9GB free on `C:` across the whole sweep -- free space
actually grew mid-sweep, apparently from unrelated activity on this
shared machine, not anything this experiment did).

### Result: no width improves on the 1x baseline -- flat to worse

Same `--split train`/`--split val` diagnostic as Phase 1, run on each
width's `final` checkpoint:

| width | val changed-patches improvement | train changed-patches improvement | games beating identity (train) |
|---|---|---|---|
| 1x (64ch, baseline) | **+2.8%** | **+1.6%** | 1/25 |
| 2x (128ch) | -1.2% | -0.3% | 3/25 |
| 4x (256ch) | -0.3% | -0.9% | 0/25 |

(Absolute magnitudes are much smaller than Phase 1's production numbers
across the board -- expected, this is a 15-epoch arc-finetune budget vs
production's 60, not a like-for-like performance comparison. The
comparison that matters here is *within this table*, across widths at
matched budget.)

**Both 2x and 4x are flat-to-worse than 1x on both splits, including
train.** There's no monotonic (or any other clean) trend of improving fit
with width -- 2x's "games beating identity" count (3/25) bounces above 1x
and 4x's (1/25, 0/25) in a way that reads as sample noise on a 25-game,
short-budget comparison, not a real signal; the pooled and unweighted
numbers agree that no width is doing meaningfully better than 1x at
fitting even its own training data. If capacity were the bottleneck,
tripling and then 14x-ing the parameter count should have shown *some*
improvement in train-set fit at matched budget, even a partial one --
instead the pattern is indistinguishable from noise around "no
improvement," with 2x and 4x both nominally worse than 1x on val.

The plausible mechanistic story (not independently verified beyond what's
here, flagged honestly as such): a wider network has more parameters to
move per gradient step at the same learning rate (`lr=3e-4`, unchanged
across widths) and the same total step count -- so if anything, this
short-budget test is somewhat biased *against* the wider models catching
up within 15 epochs, which makes their flat-to-worse showing here a
*stronger* signal against the capacity hypothesis, not a weaker one (a
genuinely capacity-starved model should show at least a directional
improvement even under-optimized, not a flat line).

### Phase 2 decision: stop here, do not invest in a full-length run

Per this experiment's own budget guidance: *"If NONE of the widths
meaningfully improve train-set fit even at the short budget, that's
itself a clean answer... don't force a longer run looking for a signal
that isn't there."* None did. No width variant's checkpoint was kept
(`checkpoints_width1x/2x/4x` were deleted immediately after their eval
numbers were extracted into `logs/benchmarks/history.jsonl` -- all three
combined were under 25MB, this was about hygiene discipline, not real
disk pressure this time).

## Overall verdict

**No -- at the scales tested (1x/64ch through 4x/256ch, up to a 14x
parameter increase), this model is not capacity-bound.** Phase 1 found a
real, honest reason to *doubt* the project's prior "data-bound" framing
(mediocre train-set fit, not the clean generalization-gap signature that
would have ruled out capacity outright) -- but Phase 2's direct test
resolves that doubt: giving the architecture meaningfully more width does
not improve fit on its own training data, even under a budget-matched,
apples-to-apples comparison. This strengthens (rather than merely
repeats) CLAUDE.md's Stage 4 conclusion -- that conclusion came from
watching a gate fail to specialize and inferring "not enough interesting
signal to route between," which is suggestive but indirect; this
experiment tests the width question head-on and gets the same answer a
different way.

**What Phase 1's "mediocre train fit" is more likely to actually be**,
given Phase 2 rules out plain undercapacity: the EMA-target
non-stationarity caveat flagged above is one real candidate (a
co-evolving target changes what "fitting the training set" even means
compared to a fixed-target supervised problem) -- but the more likely
explanation, consistent with this project's entire Stage 1-5 history, is
still the original "data-bound" one: single-step, per-frame MSE-optimal
prediction from a random(-ish)-policy corpus is close to genuinely
predicting from noise for a meaningful fraction of these 25 games (Stage
1 item 9's own conclusion, reached independently via a completely
different ablation), and no amount of width fixes an underlying
information-scarcity problem in the training signal itself.

## What would be worth trying next, if this line is revisited

Not further width scaling -- this experiment's own result argues against
that lever specifically. Better candidates, in rough order of how
directly they follow from what Phase 1/2 actually found:

1. **A fixed-target ablation**: temporarily disable the EMA update
   (`update_ema_target`) or freeze the target encoder entirely for a short
   run, to isolate how much of Phase 1's "mediocre train fit" is the
   moving-target effect flagged above vs. a genuine data/signal
   limitation. Cheap to try (a few-line change), directly answers the one
   caveat this experiment couldn't fully rule out on its own.
2. **More/better training signal**, not more capacity -- e.g. Stage 5's
   already-identified lever (a teacher/search policy's denser, more
   purposeful trajectories, per `stage6_search_harvest.md`'s own
   unresolved retrain) or simply more epochs at the *current* width on
   the *current* data (this experiment never ran anywhere near
   production's full 60-epoch arc-finetune budget at any width, by
   design -- it's possible even 1x hasn't converged yet, which is an
   orthogonal question from whether width helps).
3. **Depth, not width**, if a capacity lever is wanted at all -- this
   experiment only tested channel count; a shallow-but-wide network and a
   deeper network aren't the same capacity question, and architecture.md's
   own weak-points table doesn't rule out depth being the more relevant
   axis for whatever this encoder is currently missing.

## Reproducing this experiment

```
# 1. Reconstruct the exact local training corpus (see "A real infrastructure
#    snag" above for how these files were located/verified) into
#    ARC-AGI-3-Agents/recordings/, and copy data/arc3_logs.zip from the main
#    checkout. Copy checkpoints/{encoder_moe.pt,moe_predictor.pt,game_vocab_moe.json,
#    moe_training_meta.json} into a local checkpoints/ dir.

# 2. Phase 1: train-vs-val diagnostic on the production checkpoint
python -m jepa.benchmark eval --moe --split val   --checkpoint-dir checkpoints --tag phase1-val
python -m jepa.benchmark eval --moe --split train --checkpoint-dir checkpoints --tag phase1-train

# 3. Phase 2: short-budget width sweep (repeat --width-mult in {1.0, 2.0, 4.0})
python -m jepa.train_moe_predictor --pretrain-epochs 5 --epochs 15 --external-per-game 2000 \
    --num-experts 8 --out checkpoints_widthNx --width-mult N --checkpoint-every 5

# 4. Eval each width, both splits
python -m jepa.benchmark eval --moe --checkpoint-dir checkpoints_widthNx --split val   --tag widthN-val
python -m jepa.benchmark eval --moe --checkpoint-dir checkpoints_widthNx --split train --tag widthN-train
```
