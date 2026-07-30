# Stage 6 continuous-game-embedding investigation: does replacing the categorical game_id lookup help held-out-game generalization?

**Status: Phase 1 DONE (negative result, 2 folds). Phase 2B in progress.**

## Motivation

`CLAUDE.md`'s Stage 6 addendum found the MoE world-model predictor has no
measurable prediction advantage over identity on any local game it wasn't
trained on (confirmed via 5-fold cross-validation across all 25 games,
`stage6-multifold-cv`). Four independent fixes all failed to close the
gap: ablating the categorical per-game embedding (`stage6-gameid-
ablation`, reseed-verified), confirming the encoder itself is fine
(`stage6-encoder-holdout-diag`), and a direct anti-collapse loss +
simulated training-time unfamiliarity (`stage6-residual-commitment-fix`).
Separately, `scripts/diagnose_infogain_holdout.py` found the *raw*
per-expert disagreement (InfoGain) does NOT collapse on held-out games
even though the *gated* prediction does -- suggesting the gate blending
several genuinely-differentiated expert opinions toward a near-zero
average is the likely mechanism, not a total loss of signal.

The hypothesis under test here: replacing the categorical `game_id ->
embedding table` lookup (which falls back to a fixed, undertrained index
0 for any novel game) with a continuous, **observation-derived** game
descriptor might let the model represent a genuinely novel game as a
point in a learned space, rather than a meaningless constant -- similar
in spirit to context-conditioned meta-RL (PEARL/VariBAD-style task
inference).

This investigation runs a cheap test first (does an already-built
component already show this effect?) before committing to a new model.

## Phase 1: does Stage 3's recurrent predictor already generalize better, thanks to its hidden state?

`jepa/models/recurrent_predictor.py: RecurrentActionConditionedPredictor`
(Stage 3) feeds a `GRUCell` a pooled feature summary + action/xy/game
conditioning each step, and its hidden state -- accumulated from
*observed transitions within the current episode*, not a category lookup
-- is broadcast back in as one more conditioning channel. It still also
conditions on the same categorical `game_id` embedding the MoE predictor
uses, so this is a test of "does adding observation-derived context on
top of the existing categorical lookup help," not a clean ablation of
categorical conditioning alone (that's Phase 2A's job, if Phase 1 had
gone the other way).

### Method

Extended (not rebuilt) existing Stage 3 infrastructure to support the
same leave-N-games-out protocol as `stage6-multifold-cv`:

- `jepa/data/sequences.py: load_all_episodes` -- added `exclude_games`
  (mirrors `trajectories.load_all_transitions`'s flag) and
  `name_substrings` (an include-list, used by the eval script to load
  ONLY a fold's held-out games' episodes).
- `jepa/train_recurrent_predictor.py` -- added `--exclude-games` and
  `--ablate-game-id` CLI flags (the latter mirrors
  `train_moe_predictor.py`'s flag of the same name, built for Phase 2A
  but not exercised this phase since Phase 1 didn't reach that branch),
  plus a `recurrent_training_meta.json` sidecar (this script previously
  wrote no training metadata at all).
- `scripts/eval_recurrent_holdout.py` (new) -- unlike
  `scripts/eval_multifold.py` (which evaluates the MoE predictor on
  i.i.d.-shuffled single transitions), the recurrent predictor's forward
  pass depends on a hidden state accumulated from *preceding in-episode
  transitions*. A bare single-transition batch would always hand it a
  zeroed hidden state, making the "does the hidden state help" question
  untestable. This script instead runs each held-out game's full episode
  **sequentially**, maintaining real accumulated hidden state from
  episode start (zeroed only at episode start, exactly mirroring how a
  live agent would use the model), and measures changed-patches
  pred-vs-identity MSE at every step. Steps are also split into "early"
  (first 4 steps of each episode, hidden state still mostly zeros) vs.
  "warmed_up" (afterward) to isolate whether accumulated history
  specifically helps, not just whether the model overall beats identity.
  Held-out games are looked up with the same `game_vocab.get(id, 0)`
  fallback as every other eval script in this project, mirroring real
  production behavior on a novel Kaggle game.

**Deviation from the MoE fold-1 recipe, decided upfront and documented
rather than silently absorbed:** the MoE checkpoints this project
compares against were trained with MiniGrid pretraining (20 epochs,
67,200 transitions) and external `arc-3-logs` augmentation
(`--external-per-game 2000`, ~32,800 transitions) on top of the ~9,600
local transitions. `jepa/train_recurrent_predictor.py` / `jepa/data/
sequences.py` support **neither** -- by original design (see `jepa/data/
sequences.py`'s own docstring: the external corpus has no clean
per-episode boundaries in its schema, and MiniGrid pretraining would need
new episode-chunking work for `generate_transitions`'s flat output).
Building that support is exactly the kind of "large from-scratch build"
this investigation was told not to commit to before Phase 1 test
evidence justified it -- so Phase 1 ran local-only, matching Stage 3's
own original recipe, not the MoE's combined-data one. This means the
recurrent predictor entering this test had strictly less training data
and weaker synthetic-mechanic exposure than the MoE checkpoints it's
being compared against. If Phase 1 had shown a strong positive effect
despite this handicap, that would have been *stronger* evidence for the
hidden-state hypothesis (winning despite less data); the actual result
(below) is negative, so this deviation is a claimed limitation, not an
explanation to lean on.

Recipe used, both folds: `python -m jepa.train_recurrent_predictor
--epochs 60 --exclude-games <fold's 5 games> --out
checkpoints_fold<N>_recurrent` (default `--seq-len 16 --batch-size 8 --lr
3e-4`, warm-started from `checkpoints/encoder.pt`, same as Stage 3's
original recipe). Fold definitions reused directly from
`stage6-multifold-cv` (`experiments/stage6_multifold_generalization.md`):

| fold | held-out games |
|---|---|
| 1 | `r11l`, `bp35`, `m0r0`, `tr87`, `ka59` |
| 2 | `ar25`, `cd82`, `cn04`, `dc22`, `ft09` |

### Results

Both folds trained cleanly to completion (60/60 epochs, no crashes).
Trained-games (val split) changed-patches improvement at epoch 60: **fold
1 +7.0%** (pred=0.01180, identity=0.01269) -- a real, modest win given
the weaker local-only data recipe. **Fold 2 -5.4%** (pred=0.00213,
identity=0.00202) -- worse than identity, and checking the full epoch
trend (not just the final epoch) shows this isn't a late-training blip:
fold 2's predictor tracks essentially at parity with identity from
around epoch 20 onward (e.g. epoch 20: pred=0.01047/identity=0.01010,
epoch 40: pred=0.00406/identity=0.00396, epoch 60: pred=0.00213/
identity=0.00202 -- consistently ~1-5% worse throughout the second half
of training, never crossing to a real edge). So fold 2's recurrent
predictor didn't clearly beat identity even on its own *trained* games
under this local-only recipe, unlike fold 1 -- a real fold-to-fold
difference worth flagging honestly rather than glossing over, though not
the focus of this test (held-out generalization, below, is the same
near-zero story in both folds regardless of how the trained-game number
came out).

**Held-out games (the actual question):**

| fold | held-out games | changed-patches improvement | early-steps | warmed-up steps |
|---|---|---|---|---|
| 1 | r11l, bp35, m0r0, tr87, ka59 | **-1.7%** (n=1881) | -1.5% (n=92) | -1.7% (n=1789) |
| 2 | ar25, cd82, cn04, dc22, ft09 | **+0.5%** (n=1917) | +0.4% (n=99) | +0.5% (n=1818) |

Per-game breakdown (both folds) shows the same noisy-around-zero pattern
`stage6-multifold-cv` already documented for the MoE predictor -- a few
games with large-looking percentage swings (`dc22` -31.9%, `ft09` -83.6%
in fold 2) that trace back to tiny absolute identity-baseline MSE
(0.0004, 0.0002), the same "small absolute error swing = huge relative
swing" pattern `CLAUDE.md` already flags for `ft09`/`vc33`/`s5i5`
specifically -- not a real generalization signal either direction.
`cn04`'s absolute MSE (~1.36) is nearly 100x every other game's in either
fold and dominates fold 2's pooled percentage almost by itself.

**The early-vs-warmed-up comparison is the most direct test of the
hidden-state hypothesis specifically, and it shows nothing:** in both
folds, the "warmed up" bucket (real accumulated history from several
preceding in-episode transitions) is statistically indistinguishable
from the "early" bucket (hidden state still mostly zeros). If observed
in-episode history were adding useful game-identifying signal beyond
what a zeroed/near-zero hidden state already provides, warmed-up steps
should show a measurably different (most plausibly better) improvement
percentage than early steps. They don't, in either fold.

### Verdict: Phase 1 is a clean negative result

Both folds land within the same near-zero noise band `stage6-multifold-
cv` already established for the MoE predictor across all 5 of its folds
(-1.8% to +0.11%, mean -0.30%/-0.40%) -- fold 1's -1.7% and fold 2's
+0.5% are not distinguishable from that same "identity parity" pattern,
and the within-fold early-vs-warmed-up split shows no benefit from
actually accumulating real history. **Within-episode hidden state alone
is not a rich enough game descriptor to close the held-out-game gap**,
even stacked on top of the same categorical game_id conditioning the MoE
predictor already has. Two folds (not the full 5) is treated as
sufficient corroboration here, not because more folds wouldn't be
informative, but because the task's own decision criterion for the
negative branch doesn't require full 5-fold validation before proceeding
to Phase 2B -- the two folds tested already agree with each other and
with the MoE's own 5-fold pattern, which is the relevant cross-check.

**Per the task's decision tree: this routes to Phase 2B** -- design and
build a purpose-built context-encoder module for the MoE predictor,
rather than Phase 2A (which would have ablated the recurrent predictor's
own game-id conditioning to isolate the hidden state's contribution --
moot, since the hidden state showed no measurable contribution to
isolate).

### A caveat worth stating plainly

This result argues that *this specific implementation* of observation-
derived context (a GRU hidden state pooled from single-frame feature
summaries, updated once per step) doesn't help -- it does not rule out
observation-derived context entirely. Plausible reasons this
implementation specifically might underperform a better-designed context
encoder: (1) the hidden state is a *scalar-pooled* (mean over spatial
dims) summary each step, discarding all spatial structure before it ever
reaches the GRU; (2) it's built from single-step deltas with no explicit
mechanism to represent "what kind of game is this" as opposed to "what
just happened," so it may be tracking short-term dynamics rather than a
stable task identity; (3) it was never trained with any explicit
incentive (e.g. a contrastive or reconstruction objective on the hidden
state itself) to make the hidden state disentangle game identity from
recent-transition noise. Phase 2B's context-encoder design should learn
from this: reduce output dimensions, encode from something inherently
different from what already failed here, and consider Phase 2B's
subsections below.

## Phase 2B: purpose-built context encoder for the MoE predictor

(In progress -- see below for what's built and validated so far.)
