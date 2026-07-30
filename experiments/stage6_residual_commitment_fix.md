# Stage 6 experiment: can a direct residual-commitment loss + simulated-unfamiliarity dropout close the held-out-game gap?

**Status: DONE. Negative result. The fix does NOT close the held-out-game
gap -- changed-patches improvement stays at -0.1% (statistically
identical to the pre-fix checkpoints' +0.0%), and the residual-commitment
ratio on held-out games is still ~0.000, literally unchanged from every
other checkpoint measured so far. It also cost a modest, real regression
on the 20 trained games (+1.4% vs the untouched baseline's +8.0%).**

## Motivation

Three prior experiments in this arc localized where the held-out-game
generalization gap actually lives:

1. `stage6-game-holdout`: the MoE predictor's changed-patches advantage
   over identity collapses to ~0% on 5 games held out of training
   entirely (`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`).
2. `stage6-gameid-ablation`: removing per-game embedding conditioning
   entirely (forcing `game_idx=0` always, in training AND eval) did NOT
   close the collapse (-0.2% vs +0.0%) -- ruling out "untrained fallback
   embedding" as the cause.
3. `stage6-encoder-holdout-diag`: directly measured the encoder's basic
   change-sensitivity on held-out games -- it's actually STRONGER there
   (48-80x changed/unchanged feature-delta ratio) than on trained games
   (6-30x). The encoder is fine. A bonus check localized the real
   bottleneck: the predictor's residual-commitment ratio (residual
   magnitude / true feature-space delta magnitude, at changed patches)
   collapses to **~0.000** on held-out games vs **0.235/0.010** on
   trained games (baseline / object-identity checkpoints respectively) --
   the predictor defaults to "predict identity" specifically on
   unfamiliar games, even though the encoder handed it a clear,
   correctly-detected change signal.

This experiment tests the natural next step: if the predictor's residual
branch is the component collapsing, can it be trained directly to stop
doing that -- via (a) an explicit auxiliary loss penalizing small
residuals at genuinely-changed patches, and (b) exposing the model to a
simulated version of the "conditioning is uninformative" situation during
training (not the *full* removal `stage6-gameid-ablation` already showed
doesn't help, but a partial, per-example dropout that still leaves
game-id conditioning available/useful most of the time)?

## Step 1: gate-entropy diagnostic (before building anything)

Checked whether the MoE gate becomes MORE uniform/hedged specifically on
held-out games, as a candidate mechanistic explanation for the residual
collapse (a model that doesn't know which expert to trust on an
unfamiliar game might hedge toward a uniform blend of small,
not-necessarily-aligned per-expert residuals, which nets out closer to
zero than any single confident expert's output would). Reused the exact
`checkpoints_holdout_baseline`/`checkpoints_holdout_objid` checkpoints
and the verified 150-file/12,000-transition corpus
(`E:/ARC-AGI-3-JEPAstyle_data/recordings_archive`) from
`stage6-encoder-holdout-diag`, computing per-transition gate entropy
(`-sum(p_i * log(p_i))` over the 8 experts) and the fraction of examples
with one expert clearly dominant (weight > 0.3, Stage 4's own convention)
(`scripts/diagnose_gate_entropy_holdout.py`).

| checkpoint | split | mean entropy (% of max) | entropy std | frac dominant (>0.3) |
|---|---|---|---|---|
| baseline-holdout | held-out (n=2400) | 100.00% | 0.0001 | 0.00% |
| baseline-holdout | trained (n=2000) | 99.79% | 0.0192 | 0.30% |
| object-identity-holdout | held-out (n=2400) | 100.00% | 0.0000 | 0.00% |
| object-identity-holdout | trained (n=2000) | 100.00% | 0.0000 | 0.00% |

**Partial support for the hypothesis, from the baseline checkpoint only.**
The baseline checkpoint's small sliver of real per-input gate structure on
trained games (entropy std 0.0192, 0.30% of examples with a dominant
expert -- consistent with Stage 4's own "minority behavior, not the norm"
finding) disappears almost entirely on held-out games (std 0.0001, 0.00%
dominant) -- directionally consistent with "the gate hedges more on
unfamiliar games." But the effect size is small in absolute terms (both
numbers are already >99.7% of the uniform maximum), and the
object-identity checkpoint's gate is already **fully flat (100.00%,
std 0.0000) on BOTH splits** -- it never had any real per-input structure
to lose in the first place, so the diagnostic has nothing to say about
that checkpoint specifically. Treated this as motivating-but-not-
conclusive evidence going into the fix, not proof.

## Step 2: the fix

Added two clean, flagged, reusable options directly to
`jepa/train_moe_predictor.py` (the project's established pattern for
Stage 6 additions -- see `--exclude-games`, `--contrast-weight`,
`--ablate-game-id` on sibling branches -- rather than forking a parallel
script that would fragment future reuse/ablation):

**1. `residual_commitment_loss` (`jepa/losses.py`).** A one-sided hinge:
penalizes the predictor's raw residual (`pred_feat - cur_feat`, before the
`feat +` skip-add) for being SMALLER in per-patch "energy" (channel-mean
squared magnitude, `per_region_error`'s own convention) than the true
observed feature-space delta (`target_feat - cur_feat`), restricted to
patches that actually changed. Does NOT penalize overshoot -- pure
"commit to something with real magnitude" pressure, distinct from
`weighted_prediction_loss`'s existing per-patch reconstruction upweighting
(Stage 1 item 3), which only rewards getting the *right* answer at
changed patches and (per this whole experiment arc's finding) doesn't by
itself stop the model from satisfying that loss via a safe near-zero
residual under genuine uncertainty.

**2. `--game-id-dropout` (`jepa/train_moe_predictor.py`, ARC-finetune-
phase-only).** Per-example Bernoulli: with probability p (0.2 used here),
force that training example's `game_idx` to the fallback index 0.
Deliberately NOT the same thing as `--ablate-game-id`
(`stage6-gameid-ablation`, which forces EVERY example to index 0 for the
whole run and already didn't close the gap on its own) -- this keeps
game-id conditioning available/useful on the 80% majority of examples,
while still exposing the model, during training, to the exact
"conditioning is uninformative" situation a real hidden game produces at
test time, on a fraction of the ALREADY-FAMILIAR 20 trained games. The
point: if the residual-commitment loss above has to survive this
condition during training (not just face it for the first time at eval on
5 never-seen games), maybe it generalizes better.

Both are new CLI flags (`--residual-commit-weight`, `--game-id-dropout`),
default 0.0 (off), so the script reproduces the exact prior recipe
unless explicitly requested.

**Hyperparameter sweep for `--residual-commit-weight`** (short runs, no
MiniGrid pretrain, on the 20-game local-only corpus, before committing to
a full run):

| weight | epochs tested | outcome |
|---|---|---|
| 1.0 | 6 | Residual overshoots massively (commitment ratio 11-16x) and never recovers -- changed-patches ends up 17x *worse* than identity, still diverging at epoch 6. |
| 0.02 | 6 | Still overshoots badly early (commitment ratio up to 11x), recovering slowly -- changed-patches still ~36% worse than identity by epoch 6, trending down but not there. |
| 0.005 | 10 | Converges -- commitment ratio down to 0.40 by epoch 10, changed-patches gap narrowed to -36% (from a much worse start), still trending toward parity. |
| **0.002** | 15 | Cleanest convergence -- commitment ratio settles ~0.15-0.17 by epoch 15 (comparable to baseline's own natural 0.235 on trained games), changed-patches gap narrows steadily from -1483% at epoch 1 to -13% at epoch 15, still improving. |

The one-sided hinge has no penalty for overshoot, so at high weight the
optimizer can trivially satisfy it by inflating the residual far beyond
the true delta's magnitude -- a real, observed instability, not a
hypothetical one. **`--residual-commit-weight 0.002 --game-id-dropout
0.2`** was used for the full run based on this sweep.

## Step 3: full run and evaluation

Trained on the **identical** 20-game corpus/recipe as
`stage6-game-holdout` and `stage6-gameid-ablation` (same command, only
the two new flags added):

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --residual-commit-weight 0.002 --game-id-dropout 0.2 \
  --checkpoint-every 5 --out checkpoints_holdout_rescommit
```

Trained cleanly to completion (20 MiniGrid-pretrain + 60 ARC-finetune
epochs, confirmed via `moe_training_meta.json`'s `checkpoint_tag: "final"`
and byte-identical transition counts to the reference checkpoints: 9,600
local + 33,998 external + 67,200 MiniGrid). Ran as a fully OS-detached
`Start-Process` (file-redirected stdout/stderr, not a piped handle --
avoids the pipe-buffer deadlock risk of an unattended redirect) on a
shared/contended GPU; took a little over an hour wall-clock.

### 1. Changed-patches -- the real test

Evaluated via `scripts/eval_residual_commitment_fix.py` (same methodology
as `scripts/eval_gameid_ablation.py`: held-out games use the
`game_vocab.get(game_id, 0)` fallback, matching real production behavior
on a novel game; trained games use the full local 20-game corpus, not a
held-back val slice):

| checkpoint | held-out games (5), n=1881 | trained games (20), n=5441 |
|---|---|---|
| baseline (no fix) | **+0.0%** (pred=0.065562, id=0.065571) | **+8.0%** (pred=0.001001, id=0.001088) |
| object-identity | **+0.0%** (pred=0.042601, id=0.042602) | **+1.9%** (pred=0.001093, id=0.001114) |
| **residual-commitment-fix** | **-0.1%** (pred=0.027085, id=0.027048) | **+1.4%** (pred=0.000162, id=0.000165) |

Per-game held-out breakdown (all at or below parity, same pattern as
every prior checkpoint in this arc):

| game | improvement |
|---|---|
| `r11l` | -0.2% |
| `bp35` | -0.2% |
| `m0r0` | -0.0% |
| `tr87` | -0.1% |
| `ka59` | -0.1% |

**Answer to the core question: no, this does not close the held-out-game
gap, not even partially.** -0.1% is statistically indistinguishable from
the pre-fix checkpoints' +0.0%/+0.0% -- all three collapse to identity
parity on every held-out game, same as the untouched baseline and the
game-id-ablated checkpoint (`stage6-gameid-ablation`'s -0.2%) before it.

**Trained-games regression check: a real, if modest, cost.** +1.4% is
noticeably below the untouched baseline's +8.0% (though still comfortably
ahead of pure identity, and roughly in the same range as the
object-identity checkpoint's +1.9%). Adding the residual-commitment loss
+ game-id dropout did not come for free on the games the model actually
trains on.

### 2. Diagnostic C -- did the fix move the actual mechanistic lever?

The changed-patches result alone doesn't say whether the fix at least
made the predictor commit MORE on held-out games (just not enough to beat
identity) or whether it had literally zero effect on that specific
quantity. Directly re-measured the same diagnostic-C residual-commitment
ratio `stage6-encoder-holdout-diag` used (`scripts/
diagnose_rescommit_check.py`, identical corpus/split):

| checkpoint | split | residual mean | true-delta mean | **commitment ratio** |
|---|---|---|---|---|
| baseline-holdout | held-out | 0.000050 | 0.354686 | **0.000** |
| baseline-holdout | trained | 0.000253 | 0.001074 | **0.235** |
| object-identity-holdout | held-out | 0.000007 | 0.132801 | **0.000** |
| object-identity-holdout | trained | 0.000027 | 0.002629 | **0.010** |
| **residual-commitment-fix** | **held-out** | **0.000004** | **0.132253** | **0.000** |
| **residual-commitment-fix** | trained | 0.000014 | 0.000165 | 0.086 |

**The fix did NOT move the mechanistic lever it was designed to move, on
the actual held-out games.** The commitment ratio there is still ~0.000
(0.000004 / 0.132253 ≈ 0.00003, rounds to the same 0.000 every other
checkpoint shows) -- despite an auxiliary loss explicitly penalizing
exactly this quantity, and despite training-time exposure to a simulated
version of "conditioning is uninformative" on 20% of ARC training
examples. On the trained games, the fix's own commitment ratio (0.086) is
actually LOWER than the untouched baseline's natural value (0.235) --
consistent with the trained-games changed-patches regression above: the
`game_id_dropout` training regime seems to have made the model *more*
conservative even when real game-id conditioning IS present at eval time,
not just more robust to its absence.

### 3. Gate entropy, revisited on the new checkpoint

For completeness, re-ran `scripts/diagnose_gate_entropy_holdout.py`
against the new checkpoint:

| checkpoint | split | mean entropy (% of max) | entropy std | frac dominant (>0.3) |
|---|---|---|---|---|
| residual-commitment-fix | held-out | 100.00% | 0.0000 | 0.00% |
| residual-commitment-fix | trained | 100.00% | 0.0000 | 0.00% |

The gate is now **completely flat everywhere** -- even less per-input
structure than the untouched baseline had on trained games (99.79%, std
0.0192, 0.30% dominant). If anything, the added loss terms diluted
whatever weak signal was driving the baseline's small amount of real
gate specialization, rather than helping it.

## Honest read

**This is a clean negative result, not a partial win.** Three separate
measurements all point the same direction:

1. **Changed-patches on held-out games: -0.1%, unchanged from ~0%.** The
   real test this experiment was built to move did not move.
2. **The residual-commitment ratio on held-out games is still ~0.000.**
   This is the most informative result of the three -- it rules out "the
   loss just wasn't strong enough yet" as an easy excuse. The mechanism
   this fix directly targets simply did not activate on genuinely unseen
   games, even though the identical loss term (and the `game_id_dropout`
   exposure) DID raise commitment somewhat on the *trained* games it was
   applied to (0.086, nonzero, unlike the ~0.000 baseline-on-held-out
   comparison point).
3. **Gate entropy stayed flat-to-worse**, and **trained-games
   changed-patches regressed** relative to the untouched baseline (+1.4%
   vs +8.0%) -- so this isn't a free, harmless attempt either; it cost
   something real on the games it could have helped.

**Why "simulated unfamiliarity" during training didn't transfer to real
unfamiliarity at test time.** The `game_id_dropout` design's whole premise
was that exposing the model to "game-id conditioning is uninformative"
*during* training (on trained games, with their real visual content still
present) would teach the residual branch a generalizable skill: commit
even when the categorical game signal is missing. That premise turned out
to be wrong, or at least insufficient. The most likely explanation,
consistent with `stage6-gameid-ablation`'s own conclusion ("something
else about how the shared encoder/predictor learns is tied to the
training games' specific visual statistics, and a categorical game-id
signal was never really carrying it in the first place"): a
game-id-dropout example still shows the model a *visually familiar* game
(one of the 20 it has seen thousands of times), just with its
categorical label zeroed -- the model can still recognize "this is
`bp35`'s specific color palette and dynamics" from the raw pixels/features
alone, with or without the id embedding attached. A genuinely held-out
game's pixels/features are unfamiliar in a way that zeroing an id on a
familiar game's own frames never recreates. This reframes the whole
"simulate unfamiliarity via game-id dropout" approach as testing the
wrong axis of unfamiliarity -- the bottleneck isn't really about whether
categorical game-id conditioning is present, it's about whether the
model has ever seen anything resembling this game's *visual content* at
all, which no amount of id-dropout on already-seen games can simulate.

**Does this rule out residual-commitment losses in general?** Not
necessarily -- it rules out this specific combination (a magnitude-only
hinge + id-dropout-on-familiar-games) as a fix for THIS specific
generalization gap. The loss did work as intended in the narrow sense
that it raised commitment on trained games from whatever the counterfactual
would have been at this weight (though the confound with the also-added
dropout, and the fact that it landed below the untouched baseline's own
0.235, makes even that a mixed result, not a clean win). It just never
had a chance to matter on held-out games because the "unfamiliarity" it
trained the model to tolerate wasn't the same kind of unfamiliarity a
truly novel game presents.

**Where this leaves the held-out-game generalization gap, across this
whole experiment arc:** four independent components have now been
checked and cleared or ruled out as fixable via a training-time trick on
the current data --

- game-id embedding conditioning (`stage6-gameid-ablation`): ablating it
  entirely doesn't help.
- the encoder's basic change-sensitivity (`stage6-encoder-holdout-diag`):
  intact, even stronger on held-out games.
- the encoder's object-identity representation
  (`stage6-game-holdout`'s diagnostic B): collapses/reverses on held-out
  games, but that's a *different*, finer property than what changed-
  patches actually needs.
- the predictor's residual-commitment mechanism, targeted directly here:
  still doesn't activate on genuinely unseen games, even under explicit
  loss pressure and simulated-unfamiliarity training.

The consistent pattern across all four is the same one
`stage6-gameid-ablation` already flagged: whatever lets the shared
encoder/predictor commit to real dynamics is tied to having actually seen
a game's specific visual statistics repeatedly during training, not to
any one identifiable conditioning mechanism that a loss-shaping or
dropout trick can substitute for. Consistent with this project's own
recurring "data, not architecture" lesson (Stage 1, Stage 4) -- closing
this specific gap most likely needs genuinely more/more-diverse *games*
during training (so the model has actual practice generalizing across
novel visual content, the way MiniGrid pretraining gave it practice
generalizing across novel *action* semantics in Stage 4), not further
tuning of the current recipe's loss terms on the same 20-game corpus.
Worth flagging as the natural next lever for a future session, rather
than further ablations of this same loss-shaping family.

## Reproducing this experiment

```
# Corpus setup (once, same as stage6_game_holdout.md): copy the verified
# 150-file *.random.80.* corpus into ARC-AGI-3-Agents/recordings/, and
# data/arc3_logs.zip into data/ -- both gitignored.

# Step 1 (reuses stage6-encoder-holdout-diag's checkpoints + corpus,
# no training needed):
python scripts/diagnose_gate_entropy_holdout.py

# Step 2/3 (full training run, ~60-90 min on a shared GPU):
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --residual-commit-weight 0.002 --game-id-dropout 0.2 \
  --checkpoint-every 5 --out checkpoints_holdout_rescommit

python scripts/eval_residual_commitment_fix.py       # changed-patches comparison
python scripts/diagnose_rescommit_check.py           # diagnostic C (commitment ratio)
```

(Requires `checkpoints_holdout_baseline/` and `checkpoints_holdout_objid/`
from `experiments/stage6_game_holdout.md` alongside the new
`checkpoints_holdout_rescommit/` for the full comparison.)
