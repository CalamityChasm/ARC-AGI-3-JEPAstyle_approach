# Stage 6 follow-up: is the ENCODER's basic change-sensitivity the reason held-out-game generalization collapses?

**Status: DONE. No -- the encoder's diagnostic-A change-sensitivity does
NOT collapse on held-out games; if anything it's LARGER than on trained
games, for both checkpoints. The bottleneck is downstream, at the
PREDICTOR: directly measured residual-commitment collapses to ~0.000 on
held-out games (vs 0.235 / 0.010 on trained games), meaning the predictor
defaults to "predict identity" on unseen games even though the encoder's
own features clearly still distinguish changed from unchanged patches
there. This refutes the leading hypothesis going into this experiment and
redirects suspicion away from the encoder's representations toward the
predictor's ability to generalize its action/xy-conditioned dynamics.**

## Motivation

`stage6-game-holdout` (see that experiment's own writeup) found the MoE
predictor's changed-patches advantage over identity collapses to ~0% on 5
games held out entirely from training (`r11l`, `bp35`, `m0r0`, `tr87`,
`ka59`). `stage6-gameid-ablation` tested and refuted the obvious
suspect -- the predictor's per-game embedding conditioning -- by training
a checkpoint with game-id conditioning fully ablated; it still collapsed
to ~0% (actually -0.2%) on held-out games. The next most likely suspect,
per the orchestrating session's own reasoning: the ENCODER itself, if it
has learned game-specific visual shortcuts rather than general
shape/motion primitives. Circumstantial support: `stage6-game-holdout`'s
diagnostic B (object-identity cosine-similarity gap) also collapsed/
flipped sign on held-out games for both checkpoints tested there.

This experiment tests the encoder's more fundamental property directly --
not object-identity, but basic change-sensitivity (Stage 1 item 8's
original diagnostic A, `scripts/diagnose_encoder_vs_predictor.py`): does
encoder feature-space distance between `frame_t` and `frame_t+1` differ
at patches that actually changed vs ones that didn't, and does that gap
survive on games the encoder never saw?

## Method

New script `scripts/diagnose_encoder_holdout.py`, reusing
`stage6-game-holdout`'s exact checkpoints and held-out split:

- **Checkpoints**: `checkpoints_holdout_baseline` (no contrastive loss)
  and `checkpoints_holdout_objid` (contrast_weight=0.05), both trained in
  `stage6-game-holdout` on the identical 20-game corpus (5 games
  excluded from local + external training data entirely), found still on
  disk in worktree `agent-a0f09770086c096a6`
  (`C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/agent-a0f09770086c096a6/`)
  -- no retraining needed.
- **Data**: the same verified 150-file / 12,000-transition random-policy
  corpus `scripts/diagnose_encoder_vs_predictor.py` treats as ground
  truth (`E:/ARC-AGI-3-JEPAstyle_data/recordings_archive/*.random.80.*`),
  split into:
  - **held-out**: all 2,400 transitions from the 5 held-out games (full
    population -- neither checkpoint ever trained on any of it, so no
    train/val split is needed).
  - **trained**: a random 2,000-transition sample of the 9,600
    transitions from the other 20 games (matches
    `diagnose_encoder_vs_predictor.py`'s own `SAMPLE_N=2000`, so the two
    scripts' numbers are on a comparable footing).
- **Diagnostic A**: per-8x8-patch feature-space delta
  `(f(frame_t) - f(frame_t+1))**2` (channel-mean), compared at patches
  whose pixels actually changed vs ones that didn't, for each checkpoint,
  on each split. Game-id lookups use the checkpoint's real `game_vocab`
  with a fallback to index 0 for held-out games (`defaultdict(int,
  game_vocab)`) -- irrelevant for the encoder itself (it takes no
  `game_idx`), but kept for interface consistency with the other
  diagnostic scripts.
- **Sanity check**: confirmed `0/5` held-out games appear in either
  checkpoint's `game_vocab` (genuine holdout, not an accidental leak) --
  same check `eval_game_holdout.py` already used.

## Results

### 1. Diagnostic A: held-out games vs. trained games

| checkpoint | split | n changed patches | n unchanged patches | changed delta mean | unchanged delta mean | **ratio** |
|---|---|---|---|---|---|---|
| baseline-holdout | **held-out** | 8,818 | 144,782 | 0.354686 | 0.004422 | **80.21x** |
| baseline-holdout | trained (n=2000 sample) | 4,191 | 123,809 | 0.001074 | 0.000166 | **6.47x** |
| object-identity-holdout | **held-out** | 8,818 | 144,782 | 0.132801 | 0.002760 | **48.12x** |
| object-identity-holdout | trained (n=2000 sample) | 4,191 | 123,809 | 0.002629 | 0.000087 | **30.33x** |

**The ratio does not collapse on held-out games for either checkpoint --
it's larger than on trained games, by a wide margin.** This is the
opposite of the leading hypothesis (encoder-level shortcut learning
causing a held-out collapse). Raw counts: 8,818 changed / 144,782
unchanged patches on held-out data, 4,191 changed / 123,809 unchanged on
the trained sample -- both comfortably large populations, not a
noise-dominated measurement (see `logs/encoder_holdout_diagnostic_a_results.json`
for the full numbers).

### 2. Sanity checks on the surprising result (`scripts/diagnose_encoder_holdout_followup.py`)

Before taking "ratio is bigger on held-out games" at face value, checked
for the obvious confound: the encoder simply producing larger-magnitude
(possibly unstable/OOD) features on unfamiliar games in general, which
would inflate both changed AND unchanged deltas by the same
multiplicative factor without reflecting real sensitivity.

**Feature-norm scale check: refuted.** Mean per-patch feature squared-norm
(`||cur_feat||^2`, averaged over ALL patches, changed and unchanged
alike) is comparable between splits, if anything slightly *smaller* on
held-out data:

| checkpoint | held-out mean ||feat||^2 | trained mean ||feat||^2 |
|---|---|---|
| baseline-holdout | 10.30 | 11.93 |
| object-identity-holdout | 7.40 | 9.71 |

If the encoder were simply blowing up feature magnitude on unfamiliar
visual input, this would be much larger on held-out data -- it isn't.

**Per-game breakdown: not driven by one outlier.** Every one of the 5
held-out games individually shows ratio > 10x for both checkpoints (no
game collapses toward 1x):

| game | baseline ratio | object-identity ratio | mean pixels differing per changed patch (/64) |
|---|---|---|---|
| `r11l` | 38.51x | 62.82x | 5.22 |
| `bp35` | 41.77x | 32.27x | 18.14 |
| `m0r0` | 11.75x | 12.50x | 11.92 |
| `tr87` | 16.08x | 27.14x | 4.13 |
| `ka59` | 10.79x | 29.51x | 5.93 |

The last column also rules out a second innocent explanation ("held-out
games just have bigger, more obvious pixel-level changes per patch"):
the pooled trained-sample average is 12.81 pixels differing per changed
patch (out of 64) vs. the held-out pool's 12.13 -- essentially the same
scale. `bp35` is the one held-out game with a visibly larger per-patch
change (18.14 pixels), consistent with Stage 1's own history flagging it
as the highest-frame-level-changed-rate game in the whole 25-game corpus
-- but the other 4 held-out games have *smaller* average pixel-change
than the trained sample, and still show ratios (10.8x-62.8x) far above
parity.

### 3. Bonus check: where does the ~0% collapse actually come from? (`scripts/diagnose_encoder_holdout_predictor_check.py`)

If the encoder's own change-sensitivity is intact on held-out games, the
`stage6-game-holdout` changed-patches collapse must originate at the
predictor. Reused `diagnose_encoder_vs_predictor.py`'s diagnostic C
(predictor residual-commitment: does the predictor's own residual
magnitude track the true feature-space delta, or default to a near-zero
"predict identity" residual?) on the identical held-out vs. trained
split:

| checkpoint | split | residual mean | true-delta mean | **commitment ratio** |
|---|---|---|---|---|
| baseline-holdout | held-out | 0.000050 | 0.354686 | **0.000** |
| baseline-holdout | trained | 0.000253 | 0.001074 | **0.235** |
| object-identity-holdout | held-out | 0.000007 | 0.132801 | **0.000** |
| object-identity-holdout | trained | 0.000027 | 0.002629 | **0.010** |

**This is the direct explanation.** On held-out games, the predictor's
residual output is essentially zero (commitment ratio 0.000 for both
checkpoints) even though the true feature-space delta at those same
changed patches is large (0.35 / 0.13 -- the biggest deltas measured
anywhere in this experiment, per section 1 above). The predictor is
coasting entirely on its `feat +` identity skip-connection on unseen
games, exactly the "learned to approximate identity" failure mode Stage
1 items 8-9 originally diagnosed (there, for a data-starved single-game
ablation; here, for genuinely novel games) -- while the encoder handed it
raw material that clearly distinguishes changed from unchanged patches.

## Honest read

**The leading hypothesis going into this experiment -- that the encoder
has learned game-specific visual shortcuts and fails to register real
pixel changes on unseen games -- is refuted, cleanly and by a wide
margin, not just narrowly missed.** Diagnostic A's changed/unchanged
ratio is 48x-80x on held-out games (vs. 6x-30x on trained games for the
same two checkpoints) -- the encoder's basic change-sensitivity
generalizes fine, if anything better in this raw sense, to games it never
trained on. This is consistent with (not contradicting)
`stage6-game-holdout`'s diagnostic B finding that object-identity
representation collapses on held-out games: those are two different
properties -- "did something change here" (this experiment, intact) vs.
"do two same-colored patches at different locations look like the same
kind of thing" (diagnostic B, collapsed) -- and this result shows the
coarser property survives while the finer, more structured one doesn't.

**Where the real bottleneck sits, redirected by direct measurement rather
than by elimination alone**: the predictor's residual-commitment
collapses to ~0.000 specifically on held-out games, while the encoder
handed it a change signal that, if anything, is stronger there than on
trained games. Combined with `stage6-gameid-ablation`'s finding that
ablating game-id conditioning didn't fix the collapse either, the
remaining, best-supported explanation is that the predictor's *action/xy*
conditioning (or its MoE gating, which is itself built on the same
encoder features and was already shown in Stage 4 to have only
partial, minority-behavior specialization) isn't generalizing its
learned dynamics to unfamiliar visual contexts -- not that there's no
usable signal available to generalize from. A future session wanting to
close this gap should look at the predictor/gating side specifically
(e.g. a predictor architecture less reliant on absolute per-game visual
statistics, or explicit regularization encouraging the residual branch to
commit on out-of-distribution inputs rather than defaulting to the safe
identity skip) rather than further encoder-side changes (broader
pretraining, different self-supervised objectives) -- those would target
a property (diagnostic A) already shown intact.

## Reproducing this experiment

```
# Requires: E:/ARC-AGI-3-JEPAstyle_data/recordings_archive (the verified
# 150-file corpus) and the two stage6-game-holdout checkpoint directories
# (checkpoints_holdout_baseline / checkpoints_holdout_objid), found still
# on disk in worktree agent-a0f09770086c096a6 as of this writing --
# regenerate per experiments/stage6_game_holdout.md's own commands if gone.

python scripts/diagnose_encoder_holdout.py              # main diagnostic-A result
python scripts/diagnose_encoder_holdout_followup.py      # feature-norm / per-game sanity checks
python scripts/diagnose_encoder_holdout_predictor_check.py  # diagnostic-C bonus check
```
All three run in well under a minute each (inference-only, no training).
