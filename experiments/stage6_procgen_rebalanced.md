# Stage 6 experiment: Procgen pretraining, rebalanced -- does fixing the
# curriculum-imbalance confound change the verdict? No. A different,
# better-diagnosed negative result.

**Status: DONE. Still negative, and the mechanism check shows the original
"pretrain-phase corpus-size imbalance" hypothesis was wrong.** Rebalancing
the pretrain-phase corpus so Procgen+MiniGrid totals the same ~67,200
transitions as the MiniGrid-only baseline (instead of concatenating a full
second 67,200-transition source on top of the full baseline corpus) does
**not** fix the collapse `experiments/stage6_procgen_pretraining.md`
found, and does not close the held-out-ARC-games generalization gap
either. The mechanism trace shows *why* the original hypothesis doesn't
hold up: the rebalanced run's encoder gives a healthier **initial**
recovery at the start of ARC-3 finetuning (~5.7x better than the original
imbalanced run) but then the same collapse reasserts itself and
progresses through the *entire* 60-epoch finetune phase, ending at
essentially the same terminal magnitude as the original, uncontrolled
experiment. Corpus-size balance was not the deciding factor. This is the
**10th** independent intervention against the held-out-ARC-games
generalization gap today (following CLAUDE.md's 7 conditioning/
architecture attempts, the MinAtar data-diversity attempt, and the
original unbalanced Procgen attempt) -- the 9th to fail to close it
outright (test-time adaptation, intervention 8 in the day's running
count, remains the sole positive-but-modest result).

## Motivation and what this experiment fixes

`experiments/stage6_procgen_pretraining.md` found that adding Procgen
(`maze`, `heist`) as a second synthetic pretraining source alongside
MiniGrid **regressed** the standard trained-games sanity check
(+48.2% -> -2.0%) and collapsed the encoder's frame-to-frame
temporal-change sensitivity ~150x, never recovering across 60 ARC-3
finetune epochs (unlike the MiniGrid-only baseline, which fully recovers
within 1 finetune epoch). That document's own diagnosis traced this to a
**curriculum-balance confound**, not necessarily a genre-match problem:
the original run concatenated Procgen's full 67,200 transitions on top of
the full 67,200-transition MiniGrid corpus, **doubling the total
pretrain-phase corpus size (67,200 -> 134,400) at an unchanged 20
pretrain epochs** -- meaning half the gradient exposure per unique
transition relative to the baseline. That document explicitly flagged
this as untested and recommended a volume-controlled rerun before drawing
a genre-matching conclusion either way.

This experiment runs that rerun.

## The fix applied, and why

**Chosen approach: subsample both sources to keep the total pretrain
corpus at ~67,200 transitions (matching the single-source baseline),
rather than scaling up pretrain epochs to accommodate the full doubled
corpus.** Concretely:

- **MiniGrid**: halved via a new `--minigrid-episodes-per-env` CLI flag
  added to `jepa/train_moe_predictor.py` (previously a hardcoded-default
  function argument, not exposed on the command line) -- `20` instead of
  the default `40`, giving `21 envs * 20 episodes * 80 steps = 33,600`
  transitions instead of `67,200`.
- **Procgen**: a fresh half-size cache generated via the existing
  `scripts/generate_procgen_corpus.py` (unchanged) with
  `--steps-per-env 16800` (instead of `33,600`), giving
  `16,800 * 2 envs = 33,600` transitions instead of `67,200`. Generated
  from the dedicated Python 3.10 `C:\pgvenv` venv (procgen has no wheel
  for this project's main Python 3.13 venv -- see
  `jepa/data/procgen_data.py`'s module docstring); the original full-size
  cache from the prior experiment no longer existed on this machine, so
  this cache was generated fresh at the target size directly rather than
  generating full-size and subsampling after the fact (equivalent in
  expectation for a random-policy corpus, and avoids wasted generation
  compute). Verified action-id range and per-game/changed-rate stats
  before use, per this project's own gotcha about sanity-checking new
  data before a training run: 33,600 transitions, action ids evenly
  spread 0-7, 16,800 each for `procgen_maze`/`procgen_heist`,
  frame-level changed rate 80.7% (matches the original full-size cache's
  own reported rate almost exactly, as expected for a random-policy
  rollout).
- **Pretrain epochs unchanged at 20** (both conditions) -- with the total
  pretrain corpus now matched at ~67,200 transitions in both conditions,
  per-transition gradient exposure during pretrain is comparable between
  baseline and treatment, which is the specific variable this rerun needs
  controlled.

**Why subsample rather than scale up epochs**: scaling pretrain epochs up
to match the doubled corpus (e.g. 40 pretrain epochs for the 134,400-
transition combined corpus) would have kept total *gradient steps*
comparable but not total *unique-transition* pretrain-phase composition,
and would have made the two runs harder to compare cleanly against the
existing baseline number `experiments/stage6_procgen_pretraining.md`
already established (which used 20 pretrain epochs). Subsampling to a
fixed total transition budget is the more direct, single-variable
isolation of "does *diversity* of pretraining source help, at a fixed
total pretrain-phase data budget" -- the actual question this whole
Stage 6 data-diversity line of experiments has been asking since MinAtar.

**Tradeoff accepted, stated plainly**: unlike the Sokoban/MinAtar
ablations' "byte-identical shared MiniGrid+ARC portions" methodology,
this design means the baseline's MiniGrid corpus (67,200, full) and the
treatment's MiniGrid corpus (33,600, half) are *not* byte-identical to
each other -- the task's own guidance explicitly permitted this tradeoff
given the alternative (byte-identical MiniGrid, doubled total corpus) is
exactly the confound being fixed here.

## Controlled comparison, corpus provenance confirmed

Same fold-1 held-out games as every other Stage 6 experiment this
session (`r11l, bp35, m0r0, tr87, ka59`), same local+external ARC-3
corpus provenance as `stage6_procgen_pretraining.md` (150-file canonical
`*.random.80.*` local recordings corpus + `data/arc3_logs.zip` external
corpus, `--external-per-game 2000`), same `--num-experts 8
--contrast-weight 0.0 --checkpoint-every 5`, both runs warm-started from
the same `checkpoints/encoder.pt`:

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_baseline_rebal

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --minigrid-episodes-per-env 20 --procgen-cache data/procgen_corpus_half.pkl \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_procgen_rebal
```

Confirmed directly from each run's own log/meta, not assumed:

| | baseline (rebal) | minigrid+procgen (rebal) |
|---|---|---|
| local ARC-3 transitions | 9,600 | 9,600 |
| external ARC-3 transitions | 33,998 | 33,998 |
| MiniGrid transitions | 67,200 (`--minigrid-episodes-per-env 40`, default) | 33,600 (`--minigrid-episodes-per-env 20`) |
| Procgen transitions | 0 | 33,600 |
| **total pretrain-phase transitions** | **67,200** | **67,200** |
| games in shared vocab | 21 | 23 |

The total-pretrain-corpus match (67,200 = 67,200) is the one thing this
rerun changed relative to the original experiment (134,400 vs 67,200,
confounded); everything else (ARC-3 portions, epoch counts, exclude-games,
external cap, encoder warm-start) matches the original recipe.

## Results

### Standard-corpus sanity check (trained games), full float precision

Via `scripts/eval_procgen_standard_corpus.py` (parameterized this session
to accept checkpoint-dir overrides rather than hardcoding the original
experiment's directory names):

| variant | pred_changed_mse | identity_changed_mse | improvement over identity |
|---|---|---|---|
| baseline (MiniGrid-only, 67,200) | 9.360e-03 | 2.890e-02 | **+67.6%** |
| MiniGrid+Procgen (rebalanced, 33,600 + 33,600) | 3.574e-05 | 3.562e-05 | **-0.3%** |

**Still a regression, and the absolute-scale collapse is if anything
larger than the original unbalanced attempt's** (identity_changed_mse
ratio baseline/procgen: 2.890e-02 / 3.562e-05 ≈ **811x**, vs. the
original experiment's own ~150x). This baseline run happens to be a
stronger reference point than the original experiment's baseline
(+67.6% vs. +48.2% -- ordinary run-to-run corpus-draw variance, consistent
with this project's own repeated observation that a freshly-regenerated
corpus doesn't reproduce exact prior numbers), so the raw ratio isn't
directly comparable across the two experiments in isolation -- but within
*this* experiment's own matched pair, the collapse is real, large, and in
the same direction as before.

### Held-out fold-1 generalization

Via `scripts/eval_procgen_pretraining.py` (unmodified, reused directly):

| variant | pred_changed_mse | identity_changed_mse | improvement over identity (n=1881) |
|---|---|---|---|
| baseline (MiniGrid-only) | 0.030785 | 0.030736 | **-0.2%** |
| MiniGrid+Procgen (rebalanced) | 0.001539 | 0.001538 | **-0.1%** |

Per held-out game:

| game | baseline | MiniGrid+Procgen (rebalanced) |
|---|---|---|
| r11l | -0.6% | +0.1% |
| bp35 | -0.1% | +0.0% |
| m0r0 | -0.1% | -0.4% |
| tr87 | -0.2% | -0.4% |
| ka59 | -0.2% | -1.2% |

Both variants sit at ~identity parity, consistent with every prior
intervention this project has tried against this gap -- rebalancing did
not open a gap where the imbalanced version didn't have one, and did not
close one either. Absolute-scale collapse is present here too (identity
ratio 0.030736 / 0.001538 ≈ **20x**), smaller than the standard-corpus
check's ~811x but still substantial, and in the same direction.

## Mechanism check: did rebalancing actually fix what it was diagnosed to fix? No.

This is the part of the task that matters most: confirm the *mechanism*,
not just re-report top-line numbers.

**1. Batch-level feature variance: still not collapsed (rules out classic
representation collapse again).** Via `scripts/diagnose_procgen_collapse.py`
on a held-out batch of trained-games transitions:

| variant | mean std/channel | overall_std | overall_mean_abs |
|---|---|---|---|
| baseline | 1.842 | 2.644 | 2.036 |
| MiniGrid+Procgen (rebalanced) | 1.672 | **4.670** | 4.364 |

The Procgen-rebalanced encoder's overall feature std is *higher* than
baseline's, same qualitative finding as the original unbalanced
experiment -- the encoder is not going globally constant.

**2. Temporal-change sensitivity: collapsed, and the relative
changed-vs-unchanged contrast degrades too (not just absolute scale).**
Via `scripts/diagnose_procgen_sensitivity.py` (mean feature-space delta
between `frame_t`/`frame_t+1` at genuinely pixel-changed vs. unchanged
8x8 patches):

| variant | changed_mean_delta | unchanged_mean_delta | ratio (changed/unchanged) |
|---|---|---|---|
| baseline | 0.005392 (n=4771) | 0.000499 (n=60765) | **10.80x** |
| MiniGrid+Procgen (rebalanced) | 0.000010 (n=4771) | 0.000004 (n=60765) | **2.44x** |

Absolute scale collapses ~539x at changed patches specifically
(0.005392 -> 0.000010), and unlike the original unbalanced experiment
(where relative contrast was roughly *preserved*, 3.93x baseline vs.
3.29x Procgen -- "both shrink by roughly the same factor"), **this
rebalanced run's relative contrast degrades substantially too** (10.80x
-> 2.44x, a 4.4x reduction in the changed/unchanged ratio, not just a
proportional scale-down). The encoder still discriminates changed from
unchanged patches somewhat, but far less cleanly than either baseline.

**3. The pretrain -> finetune recovery trace directly falsifies the
original corpus-size hypothesis.** Read from each run's own epoch-by-epoch
training log:

| | pretrain end (epoch 20) `val_identity_mse` | finetune epoch 1 | finetune epoch 60 |
|---|---|---|---|
| baseline (rebal) | ~0.00000 (deep collapse -- benign, per Stage 4/6's established pattern) | **0.00897** (healthy recovery) | 0.00255 (stays healthy) |
| MiniGrid+Procgen (rebalanced) | 0.00014 (partial collapse -- *less* deep than baseline's, plausibly because Procgen's ~80.7% intrinsic changed-rate keeps some residual signal alive through pretrain) | **0.00051** (partial recovery -- ~5.7x *better* than the original unbalanced experiment's own epoch-1 value of 0.00009) | **0.00001** (collapsed again -- ~50x smaller than its own epoch-1 value) |

This is the key mechanistic finding. Rebalancing the pretrain corpus size
**did** give the encoder a measurably better starting point once ARC-3
finetuning begins (epoch-1 recovery of 0.00051 vs. the original
experiment's 0.00009) -- so the original "less gradient exposure per
transition during pretrain" mechanism was a real, correctly-identified
contributor to *that specific* symptom. But it does **not** prevent the
collapse from reasserting itself: `val_identity_mse` shrinks
*monotonically* across essentially the entire 60-epoch finetune phase in
the Procgen-rebalanced run (not just failing to recover further -- actively
continuing to shrink from an already-partial recovery), ending at
essentially the **same terminal magnitude (~0.00001) as the original,
uncontrolled experiment** despite starting finetuning from a
better-conditioned encoder. Whatever is driving the collapse during
ARC-3 finetuning is not fixed by giving pretraining more relative
exposure to each unique transition -- it re-emerges and dominates once
ARC-3 finetuning is underway, specifically when Procgen data was present
in the pretrain mix (at either tested proportion: 100%-on-top-of MiniGrid,
or 50/50 with it).

## What this rules in and rules out

**Ruled out**: the original "pretrain-phase corpus-size imbalance" was
not the (sole, or dominant) cause of Procgen's collapse. A properly
volume-controlled pretrain curriculum still produces the same qualitative
failure -- worse, in some respects (relative changed/unchanged contrast
degrades here where it didn't in the original run). The genre-matching
hypothesis that motivated testing Procgen over further MinAtar tuning
remains untested by *this* fix and is looking less promising than
`stage6_procgen_pretraining.md`'s own "inconclusive, could still be a
curriculum artifact" framing suggested -- the curriculum artifact,
specifically, has now been controlled for and the failure persists.

**Not ruled out, and worth flagging for any future revisit**: something
about Procgen's *content* itself -- plausibly the raw-RGB-to-16-color
k-means quantization (`jepa/data/procgen_data.py: fit_palette`,
`_translate_frame`), which is qualitatively different from every sibling
source's already-categorical grid representation (MiniGrid's `object_idx`
lookup, Sokoban's room-state code, MinAtar's boolean planes) -- may be
producing frames whose *quantized* pixel-level differences don't carry
the same kind of signal the encoder was built and previously pretrained
to exploit, even when the underlying maze/heist state genuinely changed.
This is a plausible mechanistic hypothesis consistent with the evidence
gathered (the encoder's changed/unchanged contrast is real but weak
specifically on this source, 2.44x vs. 10.80x, not zero), not
independently verified this session (e.g. by inspecting quantized-frame
diffs directly against raw-RGB diffs at the same patches) -- flagged as
the natural next diagnostic if Procgen is revisited a third time, ahead
of another curriculum-shape change.

## Verdict

**Negative result, now with the curriculum-imbalance confound closed
off.** This is the **10th** independent intervention against the
held-out-ARC-games generalization gap (CLAUDE.md's 7
conditioning/architecture fixes + MinAtar + the original Procgen attempt
+ this rebalanced Procgen retest), and the **9th to fail to close the
gap outright** (test-time adaptation remains the sole modestly-positive
result across all interventions tried this session). Genre-matched
Procgen pretraining data does not help close the held-out-ARC-games gap,
and now more specifically: **the reason it doesn't help is not simply
"the pretrain curriculum was unbalanced"** -- that was a real, partially-
correct diagnosis (it explains the better epoch-1 recovery seen here) but
not the dominant cause, since the same collapse fully reasserts itself
across the finetune phase regardless. Combined with Sokoban's own
negative result (a third distinct diagnosed cause -- deadlock-polluted
data) and MinAtar's (genre mismatch), the pattern across all attempts to
add a second/third synthetic pretraining source on top of MiniGrid remains
uniformly negative, each for its own distinct, now reasonably
well-understood reason -- not a single unified failure mode, but a
consistent empirical outcome regardless of cause.

## Reproducing this experiment

```
# Half-size Procgen cache (dedicated Python 3.10 venv -- see
# jepa/data/procgen_data.py's module docstring for environment setup):
C:\pgvenv\Scripts\python.exe scripts\generate_procgen_corpus.py \
  --envs maze,heist --steps-per-env 16800 --out data\procgen_corpus_half.pkl

JEPA_NUM_WORKERS=0 python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_baseline_rebal

JEPA_NUM_WORKERS=0 python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --minigrid-episodes-per-env 20 --procgen-cache data/procgen_corpus_half.pkl \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_procgen_rebal

python scripts/eval_procgen_pretraining.py --fold 1 \
  --heldout-games r11l,bp35,m0r0,tr87,ka59 \
  --baseline-ckpt checkpoints_diverse_baseline_rebal --procgen-ckpt checkpoints_diverse_procgen_rebal

JEPA_NUM_WORKERS=0 python scripts/eval_procgen_standard_corpus.py \
  checkpoints_diverse_baseline_rebal checkpoints_diverse_procgen_rebal
JEPA_NUM_WORKERS=0 python scripts/diagnose_procgen_collapse.py \
  checkpoints_diverse_baseline_rebal checkpoints_diverse_procgen_rebal
JEPA_NUM_WORKERS=0 python scripts/diagnose_procgen_sensitivity.py \
  checkpoints_diverse_baseline_rebal checkpoints_diverse_procgen_rebal
```

## Limitations

One fold (fold 1 only, matching the original Procgen experiment's own
scope decision for a negative result); one training run per variant, no
replicate seeds -- the baseline's own stronger-than-before numbers
(+67.6% vs. the original experiment's +48.2%) are a reminder that a
single run's exact magnitude is noisy even before considering the
treatment's effect. The content-quantization hypothesis in "What this
rules in and rules out" is plausible and consistent with the evidence
but not independently verified this session. `jepa/train_moe_predictor.py
--minigrid-episodes-per-env`, `data/procgen_corpus_half.pkl`, and the
parameterized eval/diagnostic scripts are all available and reusable if a
future session wants to test the quantization hypothesis directly (e.g.
by comparing raw-RGB frame diffs against quantized-frame diffs at the
same patches) before trying a third curriculum shape.
