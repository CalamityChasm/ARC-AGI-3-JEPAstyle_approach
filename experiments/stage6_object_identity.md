# Stage 6 experiment: does the encoder represent object identity?

**Status: DONE. Stage 1 (same-color within-frame contrastive loss) clears
the success bar decisively on the first attempt.** Stages 2 and 3 were not
needed. This document covers the finding that motivated the campaign, the
Stage 1 fix, robustness checks, and a full-pipeline backtest (changed-
patches per-game, retrained value head, local scorecard comparison against
the current best checkpoint). Kaggle submission was deliberately **not**
made -- checkpoint promotion and real submissions remain a human decision
point, per this project's established practice.

## The finding this responds to

`scripts/diagnose_encoder_vs_predictor.py`'s diagnostic B tests whether
two same-colored patches at *different* grid positions within a frame map
to similar encoder feature vectors, vs. two differently-colored patches
(the control) -- i.e. whether the encoder represents object identity at
all, not just "did a pixel here change." Before this experiment:

| checkpoint | same-type cos sim | diff-type cos sim | gap |
|---|---|---|---|
| production (`checkpoints/`) | 0.506 | 0.516 | **-0.0095** (noise-level; sign itself is unstable across sampling) |
| search-harvest (`E:/jepa_overflow/checkpoints_search`) | 0.628 | 0.549 | **+0.0789** (real, ~8x the noise floor, still modest) |

(These are this session's own fresh measurements on the same diagnostic
script; production's gap flipped sign vs. the +0.011 figure quoted in the
task brief -- both are within noise of each other, consistent with "noise
level" being the right read either way. Search-harvest's +0.0789 vs. the
brief's +0.092 is the same story: same order of magnitude, minor draw-to-
draw variance in which frames got probed.)

## Success bar (applied exactly, per the task's own instructions)

A fix is declared successful only if the same-type/different-type gap on
the resulting checkpoint is **clearly, robustly above noise and
comparable to or exceeding +0.092** (search-harvest's own gap) -- not
just "better than production," which is too low a bar to mean much.
Ambiguous results (+0.02-0.05, or inconsistent across reruns) would count
as **not fixed** and require escalating to Stage 2/3.

## Stage 1: same-color within-frame contrastive loss

### What was built

`jepa/losses.py: same_color_contrastive_loss` (plus a small helper,
`patch_dominant_color`) -- pulls together same-color, different-position
patch features within a frame; pushes apart different-color patch
features. A simple cosine-similarity margin term (`CONTRAST_MARGIN=0.3`),
**not** full InfoNCE, per this project's established preference for the
simplest thing that could plausibly work (and it did -- no need to
escalate the loss's own sophistication).

- `patch_dominant_color` derives each 8x8 patch's dominant ARC color
  directly from the existing one-hot input tensor already flowing through
  training (sum each patch's per-pixel one-hot counts per color channel,
  argmax over the 16 real color channels, excluding the pad channel) --
  no new data pipeline, no raw grid needed.
- The per-frame "background" color (the single most common per-patch
  dominant color) is excluded from both the positive and negative sets:
  pulling together dozens of background patches that already dominate a
  frame, or contrasting them against everything else, would drown out
  the actual foreground-object signal this loss exists to add.
- Fully vectorized per batch (pairwise cosine-similarity matrix over the
  64 patches x batch, `torch.bmm`) -- no Python loop over batch/color
  groups, cheap relative to the encoder/predictor forward-backward pass.

**Where the loss is applied:** verified first (per the task's own
instruction) that the online encoder is *not* detached during ARC-3
finetuning in `jepa/train_moe_predictor.py`'s training loop (`cur_feat =
online(cur)` flows gradients, `weighted_prediction_loss` and
`variance_regularizer` already train it) -- so a new loss term on
`cur_feat` reaches the same encoder weights. Wired in at
`CONTRAST_WEIGHT=0.05` (deliberately conservative starting point, same
risk-calibration reasoning this project already uses for
`LOAD_BALANCE_WEIGHT` -- cosine-similarity terms live on a much larger
natural scale than the tiny latent-MSE main loss), applied identically in
both the MiniGrid-pretrain and ARC-finetune phases (no special-casing --
the loss is data-source-agnostic, working directly off any batch's
one-hot tensor).

### Infrastructure added alongside (reusable, not one-off)

- `jepa/data/trajectories.py: load_all_transitions(..., name_substrings=...)`
  -- optional filename-substring filter, threaded through both
  `train_moe_predictor.py` and `benchmark.py` via a new
  `--recording-substrings` flag. Needed because `E:\jepa_overflow\recordings`
  (reused read-only from `stage6-search-harvest`'s own worktree) had
  accumulated ~400 `.hypothesis.` eval-run recordings from other sessions
  on top of the 175-file (150 random + 25 solver) corpus
  `checkpoints_search` was actually trained on -- an apples-to-apples
  comparison needs the *exact* file set that checkpoint saw, not
  "whatever's in the directory today."
- Ported `--checkpoint-every`/`--resume-from`/`JEPA_NUM_WORKERS` from
  `stage6-search-harvest` unchanged (training script), plus `--moe`
  eval mode in `jepa/benchmark.py` and `scripts/run_scorecard.py`.

### Corpus and command

Apples-to-apples against `checkpoints_search`: same 175-file corpus
(`--recording-substrings ".random.,.solver."`), same curriculum
(`--pretrain-epochs 20 --epochs 60 --num-experts 8`), same warm-start
encoder (`checkpoints/encoder.pt`, copied read-only from the main
checkout). `--checkpoint-every 5` throughout for resilience.

### A real infrastructure fight, resolved

First attempt (`JEPA_NUM_WORKERS` unset, i.e. the default `num_workers=4`
on CUDA) hit **exactly** the `MemoryError` gotcha this project's own
CLAUDE.md already documents for this scale of corpus (112k transitions,
150 random + 25 solver-harvest files, some solver files up to 676MB) --
each of 4 DataLoader worker processes pickles its own full copy of the
transitions list on Windows (spawn, not fork), and this box's ~32GB RAM
wasn't enough headroom at the moment it crashed (~5GB free, consistent
with the documented pattern). Killed cleanly (RAM recovered to ~27GB
free immediately), retried at `JEPA_NUM_WORKERS=2` as a middle ground --
that came *closer* to a repeat crash (available memory dropped to
**0.7GB** momentarily) than the original 4-worker attempt's own crash
point, so reverted fully to `JEPA_NUM_WORKERS=0` (the documented safe
fallback, single-process data loading, slower but no duplication). This
config ran the full 20 pretrain + 60 finetune epochs to completion with
**zero errors**, at a sustained pace of roughly 2-3 minutes/epoch once
past the initial ~5-6 minute corpus-parse phase -- not the severe
wall-clock penalty CLAUDE.md's own num_workers=0 gotcha warns about for
larger corpora, apparently because this corpus/model size doesn't hit
that particular bottleneck as hard. A second, unrelated RAM squeeze
appeared mid-run (available memory briefly down to ~0.7-1GB, traced
directly to an unrelated process -- `starwarsbattlefrontii.exe`, i.e. a
game someone launched on this shared machine, not anything this session
did) but the run's own process footprint stayed small (~0.4-1.5GB) and
survived it without incident; disk stayed comfortably clear (26-40GB
free throughout) so the documented "full disk prevents pagefile growth"
failure mode never triggered. `--resume-from` was exercised for real
(not just built-and-never-used) across this sequence -- confirmed it
correctly reloads weights, skips the already-completed pretrain phase,
and resumes exactly where the last `--checkpoint-every` save left off.

### Result: PASS, decisively

`scripts/diagnose_encoder_vs_predictor.py` diagnostic B on the resulting
checkpoint (`checkpoints_object_identity/`):

| checkpoint | same-type cos sim | diff-type cos sim | gap |
|---|---|---|---|
| production | 0.506 | 0.516 | -0.0095 |
| search-harvest | 0.628 | 0.549 | +0.0789 |
| **object-identity-stage1** | **0.975** | **-0.225** | **+1.2003** |

**~13-15x the search-harvest gap this same script run measured, and the
bar (+0.092) by more than an order of magnitude.** Same-color patches at
different positions are now *nearly identical* in feature space (cosine
similarity ~0.98); different-color patches are *anti-correlated*
(~-0.22) -- a qualitatively different regime from either prior
checkpoint, not an incremental nudge.

**Reproducibility check** (a result this large invites suspicion of a
sampling fluke or a bug, so this was checked directly rather than taken
on faith): reran diagnostic B on the same checkpoint at 3 more random
seeds before trusting the headline number.

| seed | same-type | diff-type | gap |
|---|---|---|---|
| 0 (original) | 0.975 | -0.225 | +1.2003 |
| 1 | 0.978 | -0.269 | +1.2467 |
| 2 | 0.970 | -0.263 | +1.2323 |
| 3 | 0.977 | -0.221 | +1.1983 |

Tight (+1.20 to +1.25 across 4 independent draws) -- this is a stable
property of the checkpoint, not sampling noise landing favorably once.

**Checked for a degenerate-collapse explanation before trusting the
result further** (a suspiciously large number is exactly the situation
where a trivial failure mode -- e.g. "the encoder just outputs a
per-color constant, discarding everything else, including the dynamics
signal the whole rest of the system depends on" -- deserves a direct
look, not an assumption):
- **Diagnostic A (encoder change-sensitivity)**: changed/unchanged
  feature-delta ratio is **53.6x** on the new checkpoint, vs. 17.5x
  (production) and 7.7x (search-harvest) *in this same run*. If the
  contrastive loss had come at the cost of the encoder's ability to
  register that a patch actually changed, this ratio should have
  dropped, not risen to the highest of the three.
- **Diagnostic C (predictor residual commitment)**: 0.342, within the
  same broad range as production (0.490) and search-harvest (0.121) --
  not collapsed to near-zero (the "predictor learned to ignore its own
  residual and coast on the skip connection" failure mode Stage 1's
  original history describes).
- **The training run's own held-out changed-patches metric**
  (`train_moe_predictor.py`'s own eval, same encoder used for both sides
  of the comparison): final epoch **pred=0.00168 vs identity=0.02909**,
  a **+94.2%** improvement -- comfortably beating both production's
  historical ~+22-29% and search-harvest's own ~+26-47% pooled numbers
  from earlier in this project. The core dynamics-prediction task the
  whole pipeline exists for did not regress; if anything this run's
  world-model quality looks unusually strong even before object-identity
  is factored in.

None of these look like a degenerate trivial solution -- the object-
identity gain appears to be a genuine additional capability, not a
trade-off against everything else the model needs to do.

## Backtest: does this hold up through the full pipeline?

Per the task's own instruction to not stop at the diagnostic alone
(the self-play-bootstrap experiment's cautionary tale: a flattering
*pooled* number once hid 24/25 games individually regressing).

### 1. Changed-patches, per-game, apples-to-apples on the exact 175-file corpus

`jepa/benchmark.py eval --moe --recording-substrings ".random.,.solver."`,
production baseline established fresh in this same session (not reusing
an old number from a different corpus draw):

| checkpoint | pooled changed-patches improvement |
|---|---|
| production (fresh baseline, this session) | +26.4% |
| **object-identity-stage1** | **+94.2%** |

**Per-game breakdown — does the pooled number hold up, or hide
regressions?** It holds up. The highest-signal games (largest absolute
identity-baseline error, i.e. the ones where the percentage isn't just
noise on a tiny denominator -- see this project's own established caveat
about that) show large, genuine wins:

| game | production improve% | object-identity improve% |
|---|---|---|
| bp35-0a0ad940 | +11.2% | **+92.7%** |
| tr87-cd924810 | -69.1% (a loss for production) | **+86.3%** |
| cn04-2fe56bfb | +42.8% | **+60.0%** |
| sp80-589a99af | -21.1% (a loss for production) | **+44.8%** |

The remaining games split roughly evenly between small positive and
small negative swings on very small absolute-MSE games (the same
"tiny-identity-baseline -> noisy percentage" pattern production's own
breakdown already shows -- e.g. `tn36`, `s5i5` are negative for *both*
checkpoints, at absolute MSE values under 0.0002 either way). Positive-
game count: production 8/25, object-identity 10/25 -- comparable breadth,
but object-identity's wins are concentrated on the games that matter
most by magnitude, including two (`tr87`, `sp80`) production was
outright losing on. This is the opposite of the self-play-bootstrap
failure mode: the per-game story actively supports the pooled number
rather than being hidden by it.

### 2. Value head retrain

`python -m jepa.train_value_head --epochs 20 --encoder
checkpoints_object_identity/encoder_moe.pt` -- required regardless of
outcome, since the value head must match whichever encoder's latent
space the agent actually uses (Stage 5's own "value-head/encoder
latent-space mismatch" bug is exactly what this step exists to avoid
reintroducing). 232,876 value-target samples loaded (all local
recordings, unfiltered -- matches how this project has always built the
value head, distinct from the world-model's own filtered-corpus training
run). 0.9% nonzero targets, consistent with this project's established
reward sparsity. Final epoch: train_loss=0.0132, val_mse=0.0033 vs.
zero-baseline=0.0009 -- val_mse sits *above* the zero-baseline on the
full population, same as Stage 5's own documented finding that this is
the wrong population to judge on (98%+ near-zero-target samples
dominate any population-wide average regardless of what the meaningful
minority looks like). Did not repeat Stage 5's own meaningful-subset
re-evaluation here (time-scoped decision, see Gotchas below) -- the value
head's job in this backtest is just to support the scorecard run below,
not to be independently re-validated to Stage 5's own depth.

### 3. Local scorecard backtest vs. current best (`checkpoints_search`)

`scripts/run_scorecard.py --agent hypothesis`, same protocol this
project's own prior backtests use (real Kaggle scoring formula, computed
by the offline harness itself, not a hand-built proxy). Matched, same
session, same 25 local games. **n=3 per checkpoint** (a deliberate,
documented reduction from the task's own "n=8 if time allows" ceiling --
by this point in the session, Stage 1's diagnostic result was already
unambiguous by more than an order of magnitude, and each 25-game
scorecard pass costs several minutes; 3 was judged sufficient to confirm
directional consistency without a multi-hour tail on an already-decisive
result). Checkpoints swapped into `checkpoints/` between runs (the
harness's `Hypothesis` agent hardcodes that path).

| checkpoint | run scores | mean score | levels completed | mean levels |
|---|---|---|---|---|
| search-harvest (current best) | 0.00123, 0.00106, 0.00365 | 0.0020 | 1, 1, 1 | 1.0 |
| **object-identity-stage1** | **0.192, 0.0152, 0.00379** | **0.0703** | **2, 2, 1** | **1.67** |

Every one of object-identity's 3 runs outscored every one of
search-harvest's 3 runs (its *worst* run, 0.00379, still edges out
search-harvest's *best*, 0.00365) -- a small n, but a completely
one-sided one, not an overlapping-with-noise picture. Treating this as
**real, encouraging, directional evidence, not statistical proof** --
consistent with this project's own established discipline about not
overselling small-n comparisons (see e.g. the Stage 5 teacher-policy
follow-up's explicit discussion of exactly this issue).

**An observation worth flagging, not fully explained:** `total_actions`
(7525) and `total_levels` (183) were *bit-for-bit identical* across all
6 scorecard runs (both checkpoints, all 3 repeats each), even though
score and `levels_completed` varied. This suggests the local offline-mode
harness's episode structure (how many actions each of the 25 games'
`Hypothesis` run takes before terminating, and the total level count
across all games) is more deterministic locally than the "genuinely
differs run to run" framing CLAUDE.md's Kaggle-submission section
documents for the real hosted environment -- plausibly a local-harness
property (fixed level generation, or the agent's own action budget
always being exhausted per game) rather than something specific to
either checkpoint. Not investigated further here; flagging for whoever
next needs to reason about local-backtest variance.

## Final verdict

**Stage 1 passes the success bar decisively and holds up through the
full pipeline.** Stages 2 (cross-frame instance-persistence loss) and
Stage 3 (cross-patch attention architecture change) were **not
attempted** -- the task's own escalation rule is to stop as soon as a
stage clears the bar and validate it, not to keep going past a clean
win. The gap (+1.20, reproducible across 4 seeds) exceeds the +0.092 bar
by more than an order of magnitude, the core changed-patches dynamics
metric improved rather than regressed (+94.2% vs. production's own fresh
+26.4% on the identical corpus), the per-game breakdown shows the gains
concentrated on the highest-signal games rather than a pooled-average
illusion, and a small but completely one-sided local scorecard
comparison favors the new checkpoint over the current best.

`checkpoints_object_identity/` (gitignored, not committed -- encoder_moe.pt,
moe_predictor.pt, game_vocab_moe.json, moe_training_meta.json,
value_head.pt) is the candidate for promotion. **Per the task's explicit
instruction, this session does not promote it to `checkpoints/`
production, does not merge this branch anywhere, and does not submit to
Kaggle** -- flagging it here as ready for that human decision, with the
full reproduction command:

```
JEPA_NUM_WORKERS=0 python -m jepa.train_moe_predictor \
  --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --out checkpoints_object_identity --checkpoint-every 5 \
  --recording-substrings ".random.,.solver."
python -m jepa.train_value_head --epochs 20 \
  --encoder checkpoints_object_identity/encoder_moe.pt \
  --out checkpoints_object_identity
```

(Corpus: the 175-file, 150-random + 25-solver-harvest local recordings
corpus documented in `stage6-search-harvest`'s own
`experiments/stage6_search_harvest.md` -- reused read-only from that
branch's worktree/`E:\jepa_overflow\recordings` this session, per the
task's own instructions. `checkpoints/encoder.pt`, the Stage 1 ARC-1/2-
pretrained warm-start encoder, is a prerequisite input, copied read-only
from the main checkout.)

## Gotchas learned this session (in the spirit of this project's existing list)

- **`JEPA_NUM_WORKERS=2` was *closer* to reproducing the documented
  `MemoryError` crash than the default `num_workers=4` had been** on
  this specific corpus/machine state (available memory dropped to 0.7GB
  before it was killed) -- don't assume "fewer workers" is a safe linear
  dial between 4 and 0 for this failure mode; on a RAM-constrained shared
  machine, `0` may be the only genuinely safe setting even though the
  gotcha's own framing suggests intermediate values as a compromise.
- **Available memory (via `Get-Counter '\Memory\Available MBytes'`) is a
  more honest signal than `FreePhysicalMemory`** when Windows is holding
  a lot of reclaimable standby cache -- `FreePhysicalMemory` alone made
  a healthy machine look like it was about to crash at one point this
  session (5GB) when the real number once the standby cache was
  accounted for was materially different. Check both before deciding
  whether to intervene.
- **A shared machine's RAM pressure can come from something with nothing
  to do with this project at all** (a video game launched by whoever
  else uses this box) -- confirmed directly by process-listing the top
  RAM consumers rather than assuming it was this session's own training
  run. Don't reflexively kill your own process on an external RAM spike
  without checking who's actually responsible first.
- **`load_value_targets`/value-head training reads the *entire*
  unfiltered `ARC-AGI-3-Agents/recordings/` directory**, unlike the
  world-model corpus loader this session added filtering to -- 576 files
  (~9GB) this session, not the 175-file corpus the encoder/predictor
  were actually trained on. This is consistent with how this project has
  always built the value head (a broader, unfiltered signal is
  arguably fine/desirable for a value estimator), but it means "the
  value head's training corpus" and "the world model's training corpus"
  are *not* the same 175 files -- worth knowing if a future session ever
  needs to reason precisely about what data went into which component.
