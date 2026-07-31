# Stage 6 experiment: does test-time adaptation close the held-out-game gap?

**Status: DONE. Real, small, monotonic signal -- distinct from every
representation-based fix tried earlier this session -- but far too small
in magnitude to call the held-out-game generalization gap closed. Mild,
real, dial-able catastrophic interference on trained games, roughly
proportional to how much adaptation was applied.**

## Motivation

CLAUDE.md's Stage 6 addendum documents a robust, 5-fold-cross-validated
finding: this project's MoE world model has **zero measurable prediction
advantage over identity on any local game it wasn't trained on**, and
seven independent conditioning/architecture interventions (categorical
game-id ablation, an encoder audit, an anti-collapse residual loss, and
three different *continuous* embedding/context mechanisms -- including
one that pools multiple in-episode transitions into a context vector) all
failed to close the gap. Every one of those seven was still a **frozen,
zero-shot forward pass** at evaluation time: the model infers a
representation of "which game is this" but its weights never change once
training stops.

This experiment tests a mechanistically different idea: **test-time
adaptation** (a.k.a. test-time training) -- let the model actually take a
few real gradient steps on a held-out game's own observed transitions,
DURING simulated play, rather than asking it to infer everything from a
fixed or learned representation. If the model can extract real signal
from even a handful of a novel game's own transitions via genuine weight
updates, that's a usable mechanism regardless of whether any
representation-based fix ever works.

## Setup

**Checkpoint:** `stage6-game-holdout`'s fold-1 baseline
(`checkpoints_holdout_baseline/`, no contrastive loss, 20 games trained,
5 held out entirely from both local and external training data: `r11l`,
`bp35`, `m0r0`, `tr87`, `ka59` -- see `experiments/stage6_game_holdout.md`
for how this checkpoint was built).

**Data:** the same verified 150-file `*.random.80.*` local corpus used
throughout this session (`E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\`,
copied into this worktree's `ARC-AGI-3-Agents/recordings/` since that
directory is gitignored and worktree-local). Each held-out game has 6
recording files x 80 transitions = 480 transitions. For each game: the
first 5 files (400 transitions, in original within-file order) are the
"adaptation stream" (simulating observing this game's transitions during
play, in order); the 6th file (80 transitions) is a **fixed, never-
adapted-on eval set**, so every reported number measures generalization
within the game to genuinely unseen transitions, not memorization of the
adaptation buffer itself.

**Adapted parameters -- the key design decision.** Full fine-tuning of the
whole predictor on a few dozen-to-a-few-hundred examples risks
catastrophic overfitting/forgetting, and would confound "does adaptation
help" with "did we just break the model." A per-game-embedding-only
adapter (16 floats) was considered and rejected: three different, fully-
trained (60 epochs each) continuous embedding/context mechanisms already
failed earlier this session (see CLAUDE.md's "continuous game-embedding
investigation"), so adapting a tiny embedding for a handful of gradient
steps would likely just reproduce that same negative result for a
different, less interesting reason (too few steps on too small a
parameter), not test whether real weight updates can help at all.
Settled on an **ANIL-style last-layer/expert-output adaptation**: each of
the 8 experts' LAST `Conv2d` layer (the one directly producing the
residual) plus the gate's LAST `Linear` layer are trainable (~33.8K
parameters total); the encoder, action/xy/game embeddings, and every
expert's first layer stay frozen. Expressive enough to plausibly matter,
small and shallow enough to be a low-risk, fast fit.

**Adaptation recipe:** stream transitions one at a time; every K observed
transitions, run `N_STEPS` AdamW steps (each on a random mini-batch drawn
from ALL transitions observed so far from this game, batch size
min(16, buffer size)) at `lr=5e-5` (the midpoint of the task's suggested
1e-5 to 1e-4 range, ~6x below the original training LR of 3e-4). Loss:
the same `weighted_prediction_loss` (8x upweight on changed patches) used
in the original training recipe. Two K conditions tested (K=10, K=50);
`N_STEPS=3` as the default (a robustness check at N_STEPS=1 and 5 is
reported below, not a full grid -- see "Robustness check").

**Measurement:** changed-patches improvement over identity (this
project's standard honest metric throughout) on the FIXED eval split, at
cumulative adaptation-stream sizes of 0 / 10 / 50 / 200 (the task's own
checkpoints). For K=50, no update has fired yet by n_observed=10 (10 % 50
!= 0), so that entry is identical to the n=0 baseline by construction --
kept in the table rather than skipped, since it's itself an honest,
informative data point about update cadence.

**Catastrophic interference check:** the SAME final adapted model (after
streaming the whole 400-transition buffer) is re-evaluated on a fixed
1,600-transition probe drawn from 8 TRAINED games (`ft09`, `s5i5`, `vc33`
-- matching `eval_game_holdout.py`'s original control set -- plus `ar25`,
`cd82`, `cn04`, `lp85`, `sp80` for a wider range of typical MSE scale).
CLAUDE.md's Stage 1 item 5 already flags `ft09`/`s5i5`/`vc33` specifically
as having tiny identity-baseline MSE (~1e-5), which makes the improvement
*percentage* swing wildly from tiny absolute changes -- the wider 8-game
probe and the per-game breakdown below both exist to avoid being misled
by that.

Script: `scripts/test_time_adaptation.py`. Full results:
`logs/test_time_adaptation_results.json`.

## Results: does prediction quality improve with adaptation?

Changed-patches improvement over identity on each game's fixed eval
split, `N_STEPS=3`, `lr=5e-5`:

| game | K | n=0 | n=10 | n=50 | n=200 |
|---|---|---|---|---|---|
| r11l | 10 | -1.2% | -0.9% | -0.2% | **+0.5%** |
| r11l | 50 | -1.2% | -1.2% | -0.9% | -0.4% |
| bp35 | 10 | +0.0% | +0.0% | +0.1% | **+0.3%** |
| bp35 | 50 | +0.0% | +0.0% | +0.0% | +0.1% |
| m0r0 | 10 | -0.3% | -0.3% | -0.2% | **-0.1%** |
| m0r0 | 50 | -0.3% | -0.3% | -0.2% | -0.2% |
| tr87 | 10 | +0.3% | +0.2% | +0.2% | +0.1% |
| tr87 | 50 | +0.3% | +0.3% | +0.2% | +0.2% |
| ka59 | 10 | -1.4% | -1.3% | -0.7% | **+0.2%** |
| ka59 | 50 | -1.4% | -1.4% | -1.3% | -0.8% |

**Real, monotonic, but small.** In 4 of 5 games (`r11l`, `bp35`, `m0r0`,
`ka59`), K=10 shows a consistent, monotonic improvement as more
adaptation transitions accumulate -- `r11l` and `ka59` both flip from
negative (worse than identity, matching the known zero-shot baseline) to
positive by n=200. `tr87` is flat to very slightly worse. K=50 shows the
same *direction* in every game but a smaller magnitude by n=200 (makes
sense: K=10 fires 20 update events x 3 steps = 60 total gradient steps by
n=200, vs. K=50's 4 events x 3 steps = 12 steps) -- **more gradient steps
applied tracks with more improvement**, which is exactly the pattern
you'd want to see from genuine signal rather than noise (noise wouldn't
care how many update events fired).

**This is qualitatively different from every zero-shot mechanism tried
today.** All 7 of those landed at a flat ~0% regardless of what was
changed about the representation. Here, the number visibly *moves* as a
function of how much real adaptation data/steps the model gets -- small,
but a genuinely different failure mode (or rather, non-failure mode).

## Robustness check: step-count sensitivity (r11l, K=10)

| N_STEPS | n=0 | n=10 | n=50 | n=200 | trained-games pooled post-adapt |
|---|---|---|---|---|---|
| 1 | -1.2% | -1.1% | -0.8% | -0.0% | +8.7% (pre: +9.8%) |
| 3 | -1.2% | -0.9% | -0.2% | +0.5% | +6.6% |
| 5 | -1.2% | -0.8% | +0.1% | +0.7% | +6.4% |

A clean, monotonic dial in both directions: more steps -> more held-out
improvement AND more trained-game interference (pooled improvement drops
8.7% -> 6.6% -> 6.4% as steps go 1 -> 3 -> 5). This is the single
strongest piece of evidence in this experiment that the effect is real
rather than run-to-run noise -- noise would not track a controlled
knob this cleanly across two dependent, opposite-direction measurements.

## Results: catastrophic interference

Pooled improvement on the 8-game, 1,600-transition trained-games probe,
pre- vs. post-adaptation (`N_STEPS=3`, `lr=5e-5`) -- pre-adaptation
baseline is **+9.8%** (pred=0.002041, identity=0.002263) for every row
below, since it's the same pristine checkpoint each time:

| game adapted on | K | post-adapt improvement | delta |
|---|---|---|---|
| r11l | 10 | +6.6% | -3.2pp |
| r11l | 50 | +9.2% | -0.6pp |
| bp35 | 10 | +8.7% | -1.1pp |
| bp35 | 50 | +9.9% | +0.1pp |
| m0r0 | 10 | +9.8% | -0.0pp |
| m0r0 | 50 | +10.0% | +0.2pp |
| tr87 | 10 | +8.5% | -1.3pp |
| tr87 | 50 | +9.5% | -0.3pp |
| ka59 | 10 | +7.8% | -2.0pp |
| ka59 | 50 | +9.4% | -0.4pp |

**Real but mild, and clearly scales with how much adaptation was
applied** -- K=10 (60 total gradient steps by the end of the stream)
consistently shows more interference than K=50 (12 steps), and K=50's
interference is mostly within noise (some entries even nominally
*improve*, e.g. `m0r0`/`bp35` at K=50 -- almost certainly noise on a
1,600-transition probe, not a real generalization benefit from adapting
to an unrelated game).

**Per-game breakdown matters here -- the pooled number understates real
heterogeneity.** Looking at individual trained games across all 5 x K=10
adaptation runs:
- `s5i5` shows a large, consistent RELATIVE increase in every run
  (+40% to +144%) -- but its absolute pred MSE is tiny (0.000026 baseline
  -> 0.000036-0.000063 adapted), exactly the tiny-denominator noise
  pattern CLAUDE.md's Stage 1 item 5 already flagged this specific game
  for. Not strong evidence of real damage on its own.
- `vc33` and `cd82` show a consistent, sizeable DECREASE (improvement) in
  every single one of the 5 adaptation runs (`vc33`: -18% to -26%;
  `cd82`: -3% to -25%) -- adapting on an unrelated held-out game's data
  made prediction on these two trained games measurably *better*, not
  worse, every time. Plausibly the last-layer adaptation nudges the
  residual output in a generically more-committal direction (moving away
  from the "predict near-zero residual" collapse this project's Stage 1
  history already documented as the model's default failure mode) that
  happens to help other under-fit games too, independent of which held-
  out game supplied the gradient signal.
- `cn04` and `sp80` (the two largest-absolute-MSE probe games, so their
  percentages are the least denominator-noise-prone) both stay close to
  flat: `cn04` within ±2.2% in every run, `sp80` a small, consistent
  +0.3% to +4.8% increase in every K=10 run. This is the cleanest
  single read on "real" interference magnitude: a few percent, in one
  consistent direction, not a collapse.
- `ft09` and `ar25` are mixed/small, no clear consistent direction.

**Bottom line on interference:** real and directionally consistent for a
few individual games (mild degradation on `sp80`, mild *improvement* on
`vc33`/`cd82`), but nowhere close to "catastrophic" at this parameter
budget (~33.8K params), learning rate (5e-5), and step count (12-60
total steps) -- the largest reliable-scale (non-tiny-denominator) shift
observed on any trained game was `sp80`'s +4.8% at the most aggressive
setting tested (K=10, 5 steps). Scales up cleanly with more
steps/more frequent updates, so it's a real, tunable dial, not a fixed
cost.

## Honest verdict

**Test-time adaptation gives a real, non-zero, controllably-scaling
signal on held-out games that every purely-representational fix tried
this session could not produce.** That's a genuine, mechanistically
distinct finding worth keeping in mind for future work -- it demonstrates
the model's *architecture* isn't fundamentally incapable of learning from
a novel game's own data; it just never gets the chance to in a frozen
zero-shot deployment.

**But the magnitude is far too small to call this "solved," or even
"a usable fix" as currently scoped.** Going from ~-1% to ~+0.5% over 200
observed transitions and up to 60 gradient steps is a small fraction of
the +8% to +30% improvement production's checkpoint gets on games it
actually trained on. At this rate, closing the full gap would need
either dramatically more adaptation data than a real ARC-3 episode
budget provides (Kaggle's real budget is ~300-900 actions per game,
of which this experiment already used up to 200 just for the adaptation
stream, before the eval split) or a much larger learning rate/step count
that the interference results suggest would come at a real, if still
modest, cost to already-learned games.

**Interference is real but mild and dial-able, not catastrophic** at the
tested settings -- a practical deployment would still need SOME mitigation
(see below), but "the model breaks" is not what happened here.

## What a real integration would need (not built this session, per the task's own scope)

If this is revisited with more time/priority:
1. **Reset per game/episode.** `hypothesis_agent.py` already constructs
   fresh model state per Kaggle game session in practice (each game gets
   its own `Hypothesis` instance) -- adaptation state should NOT persist
   across different games, only within one game's repeated RESET/attempt
   cycle, to avoid the interference measured above compounding across
   an entire ~110-game Kaggle run.
2. **A larger step-count/LR sweep** than this experiment's 3-point check
   (1/3/5 steps at a single LR) -- the monotonic trend found here strongly
   suggests there's a real Pareto frontier between held-out-game gain and
   trained-game interference worth mapping out properly, not a single
   fixed operating point.
2. **Periodic fine-tuning calls wired into `choose_action`** (e.g. every
   K real actions taken, mirroring this experiment's K=10/K=50 cadence),
   guarded by a try/except so a bad gradient step can't crash an episode
   -- consistent with this project's existing heartbeat-pattern hardening
   in `hypothesis_agent.py`.
3. **A stability fallback**: since this experiment only tested a benign
   AdamW trajectory, a real deployment should guard against a pathological
   early update (e.g. a single very "surprising" transition producing an
   outsized gradient) -- e.g. gradient clipping, or reverting to the
   frozen checkpoint if a sanity-check eval on a small held-back buffer
   gets meaningfully worse after an update.
4. Given the magnitude found here, this is worth layering ON TOP OF (not
   instead of) continued work on genuinely more diverse pretraining data
   (this session's other two parallel efforts, Procgen and corrected
   MinAtar) -- test-time adaptation looks like a real but modest
   complementary lever, not a standalone fix for the held-out-game gap.

## Reproducing this experiment

```
# checkpoints_holdout_baseline/ and ARC-AGI-3-Agents/recordings/ (the
# verified 150-file *.random.80.* corpus) must be present -- both
# gitignored; see experiments/stage6_game_holdout.md and CLAUDE.md's
# environment-setup section for where to get them.

python scripts/test_time_adaptation.py --games r11l bp35 m0r0 tr87 ka59
python scripts/test_time_adaptation.py --games r11l --n-steps 1
python scripts/test_time_adaptation.py --games r11l --n-steps 5
```
Each game/K-condition pair runs in a few seconds on this box's GPU (a
handful of gradient steps, not a training epoch) -- the full 5-game,
2-K-condition sweep with the wider interference probe takes well under
two minutes total.
