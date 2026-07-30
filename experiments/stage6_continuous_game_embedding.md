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

## Phase 2B(a): single-frame content-derived context (MoE predictor)

Built `jepa/models/context_encoder.py: FrameContextEncoder` -- a small
MLP (`Linear(feature_channels, hidden) -> GELU -> Linear(hidden,
embed_dim)`) mapping the *current frame's* pooled encoder features
directly into a continuous embedding of the same dimension the
categorical `game_embed` table produces. Wired into `MoEPredictor` via a
new `context_mode` constructor argument (`"categorical"` reproduces the
original behavior exactly; `"frame"` swaps `self.game_embed(game_idx)`
for `self.context_encoder(feat.mean(dim=(2,3)))` inside `_condition`,
which both `forward` and `predict_all_experts` already call -- no other
call site anywhere in the codebase needed to change, since `_condition`
already receives `feat`). `jepa/train_moe_predictor.py` gained a
`--context-mode {categorical,frame}` CLI flag and records it in
`moe_training_meta.json`; `scripts/eval_context_holdout.py` (new, mirrors
`scripts/eval_multifold.py`'s structure) loads a checkpoint's
`context_mode` from that meta file and constructs the matching
`MoEPredictor` automatically.

**Unlike Phase 1, this recipe is fully matched to the MoE baseline's
own** (`--pretrain-epochs 20 --epochs 60 --num-experts 8 --external-per-game
2000`, MiniGrid pretrain + external `arc-3-logs` augmentation) --
`_condition` only needed to know how to compute `g_embed` from `feat`,
which is available at every call site the categorical version already
had, so there was no architectural reason to drop MiniGrid/external data
support the way Phase 1's recurrent predictor had to. Two checkpoints
per fold (`categorical` and `frame`) were trained fresh, from the same
warm-started encoder, on identical data, differing only in
`--context-mode`.

### Results (folds 1-2, same fold definitions as Phase 1 and `stage6-multifold-cv`)

| fold | held-out games | categorical | frame |
|---|---|---|---|
| 1 | r11l, bp35, m0r0, tr87, ka59 | -0.5% (n=1881) | -0.1% (n=1881) |
| 2 | ar25, cd82, cn04, dc22, ft09 | -0.0% (n=1917) | -0.0% (n=1917) |

Both variants land at essentially identical, near-zero improvement in
both folds -- consistent with (not better or worse than) every prior
categorical-conditioning result in this project's Stage 6 line of
investigation (`stage6-multifold-cv`'s own 5-fold mean was -0.30%
baseline / -0.40% no-gameid). Frame-mode's fold-1 number (-0.1%) is
marginally less negative than categorical's (-0.5%), but this is well
within the noise band `stage6-multifold-cv` already established (its
5 folds ranged -1.8% to +0.11%) -- not a real effect, and fold 2 shows
no difference at all between the two conditioning schemes (both -0.0%
to one decimal place).

**Verdict: Phase 2B(a) is a clean negative result, corroborated across
both folds tested.** A content-derived, per-frame context embedding --
with no categorical fallback, no undertrained index-0 issue, applied
identically whether or not the game is familiar -- performs no better
than the categorical lookup it was built to replace. This rules out
"the categorical lookup's fallback-to-index-0 discontinuity" as the
mechanism behind the held-out-game gap just as cleanly as
`stage6-gameid-ablation` already ruled out "categorical conditioning
being present at all" (ablating it outright didn't help either -- see
CLAUDE.md's Stage 6 addendum). Combined with Phase 1's finding that
*episode-history* hidden state doesn't help either, both the "different
representation of the same information" (Phase 2B(a)) and "richer
information source, same single-frame content" (Phase 1) axes have now
failed. Per the investigation's decision tree, this routes to Phase
2B(b): a multi-transition, meta-learning-style context encoder that
draws its descriptor from *several other* transitions observed earlier
in the same episode, not just the one frame being predicted from.

## Phase 2B(b): multi-transition episode-context encoder (scoped, single-fold, preliminary)

**Scope decided upfront, given the pattern established by Phase 1 and
2B(a):** two independent conditioning mechanisms (recurrent hidden
state, single-frame content) already failed cleanly and consistently on
2 folds each, on top of `stage6-multifold-cv`'s own 5-fold confirmation
that categorical conditioning (with or without ablation) never closes
this gap. Per this project's own repeated methodological lesson (see
CLAUDE.md's "CRITICAL" gotcha and the Stage 6 addendum's "working
conclusion"), three-for-three convergence on the same negative result is
*more* consistent with a genuine data-bound limit (the model needs real
training-game diversity, not a better way to condition on the same 25
games' worth of signal) than with "the right conditioning mechanism just
hasn't been tried yet." Given that prior, Phase 2B(b) is scoped as a
single-fold, clearly-flagged-preliminary test -- enough to check whether
a materially richer context source changes the picture at all, without
over-investing compute in a fourth confirmation of the same pattern this
investigation's own evidence increasingly points toward.

**Design:** built as new, reusable infrastructure (not a one-off
script), following this project's existing module conventions:

- `jepa/models/context_encoder.py: EpisodeContextEncoder` (added
  alongside `FrameContextEncoder` in the same module) -- takes K
  *other* transitions' already-encoded, pooled features from earlier in
  the same episode (`pooled_feat_t`, `action_id`, `pooled_feat_t1`, each
  `(B, K, ...)`), summarizes each one as `[pooled_feat_t; action_embed;
  pooled_feat_t1 - pooled_feat_t]` (what state, what action, what
  changed), maps each summary through a small MLP, then mean-pools
  across the K context transitions (permutation-invariant -- a Deep
  Sets-style phi-then-pool, since which order the context transitions
  happen to be sampled in shouldn't matter). This is meta-learning-style
  task inference (PEARL/VariBAD): infer "what kind of game is this" from
  a handful of observed exemplars, not a category lookup.
- `jepa/models/moe_predictor.py: MoEPredictor` -- added a third
  `context_mode="external"` plus an optional `context_embed` parameter
  threaded through `forward`/`predict_all_experts`/`_condition`. In this
  mode, `MoEPredictor` builds neither `game_embed` nor an internal
  `context_encoder` -- `g_embed` must be supplied by the caller (raises
  if `context_embed` is `None`), since a genuinely multi-transition
  context can't be derived from `feat` alone the way Phase 2B(a)'s did;
  it requires external orchestration by whatever has access to the full
  episode.
- `jepa/data/episode_context.py` (new) -- `EpisodeContextDataset` wraps
  `jepa/data/sequences.py`'s existing per-episode transition lists (no
  changes needed there) and, for every transition at episode-position
  `i >= CONTEXT_WINDOW`, returns the target transition plus its
  `CONTEXT_WINDOW` immediately-preceding same-episode transitions'
  frames/actions (raw, to be encoded by the shared online encoder at
  train/eval time, not precomputed -- so gradients can flow back through
  the encoder from context frames too, same as the target transition).
  `CONTEXT_WINDOW = 8`.
- `jepa/train_context_moe_predictor.py` (new training script) --
  co-trains the online encoder, `MoEPredictor(context_mode="external")`,
  and `EpisodeContextEncoder` together; encodes both the target and all
  K context frames through the same online encoder each step, computes
  `context_embed` from the pooled context features, and calls
  `predictor(cur_feat, action_id, xy, context_embed=context_embed)`
  before applying the same `weighted_prediction_loss` /
  `variance_regularizer` Stage 1 established. Saves all three components
  (`encoder_context.pt`, `context_moe_predictor.pt`,
  `episode_context_encoder.pt`) plus a training-meta sidecar.
  **Deviation from the fully-matched Phase 2B(a) recipe, same rationale
  as Phase 1's:** episode-context construction depends on
  `jepa/data/sequences.py`'s per-episode ordering, which (like Stage 3)
  only exists for local recordings -- no external `arc-3-logs` or
  MiniGrid-pretrain support for this data shape without materially more
  new plumbing than this scoped test's budget allows. Local-only,
  warm-started from `checkpoints/encoder.pt`, documented explicitly
  rather than silently absorbed, exactly as Phase 1 did.
- `scripts/eval_episode_context_holdout.py` (new) -- for each held-out
  game's full episode, walks forward through it, and at every position
  `i >= CONTEXT_WINDOW` builds context from that *same held-out
  episode's own* preceding transitions (never from a trained game) --
  the direct local proxy for "an agent accumulating real experience
  within one episode of a genuinely novel Kaggle game," which is exactly
  the scenario this whole investigation's context-encoder hypothesis is
  meant to help with.

(Training run and results to follow -- see the next entry in this file
once the fold-1 comparison completes.)
