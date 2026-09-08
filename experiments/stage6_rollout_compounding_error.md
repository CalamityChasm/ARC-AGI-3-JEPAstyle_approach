# Stage 6: does compounding rollout error through the learned predictor matter in practice?

**Scope: production-scale checkpoints only**, same as the sibling
`stage6_graph_lookahead.md` and `stage6_transition_graph_health.md`.

## Why

Follow-up to a design discussion about whether a learned action head or
MCTS-style multi-step search through the *learned* predictor could beat
the current one-step InfoGain/value heuristic. We judged multi-step
search through the learned model risky for a specific, named reason:
**compounding rollout error** -- exactly the problem `jepa/memory.py`'s
`TransitionGraph` docstring cites as its own reason to exist ("shields
you from compounding rollout error on seen states"). This is a direct
empirical test of whether that concern is real, using the production
checkpoint, rather than continuing to reason about it in the abstract.

## Real data-quality issue hit and fixed along the way

The first full run of this diagnostic used `checkpoints/` as it stood at
the time -- which turned out to be a stale `stage6-game-holdout` fold-1
checkpoint (21-game vocab, 5 games excluded from training), not the true
26-game production lineage. See CLAUDE.md's new gotcha entry and the
correction on the `55843700` Kaggle submission for the full story --
`checkpoints/` has since been restored to the true production checkpoint.
**The numbers in this section were measured against the wrong (fold-1
holdout) checkpoint before that was caught**, kept here rather than
discarded because the finding is directionally clear and a confirmatory
rerun against the corrected checkpoint is already in progress (see the
Production-checkpoint confirmation section below, filled in once that
lands).

## Method

`scripts/diagnose_rollout_compounding_error.py` (read-only; loads
checkpoints, parses `*.recording.jsonl` into RESET-delimited episodes, no
agent code touched). For each sampled start index across many real
episode segments and each depth d = 1..10:

- **Compounding rollout**: predictor fed its own previous output as the
  next `feat`, conditioned on the *real* action actually taken at each
  step (from the recording), compared against the real encoded frame at
  t+d.
- **Fresh single-step baseline at the same depth**: one predictor call
  from the *real* (ground-truth) encoded frame at t+d-1, compared against
  the same real t+d target -- isolates whether rollout adds error beyond
  what a fresh step would have anyway.
- **Identity baseline**: the frame at t held constant, no prediction at
  all, at the same target -- this project's own standard reference point.

All three scored with the same `changed-patches` methodology
(`jepa/losses.py: per_region_error`) already used everywhere else in this
project, not a new metric invented for this check.

## Results (fold-1 holdout checkpoint -- see caveat above)

**Data:** 764 RESET-delimited episodes across 25 games,
`E:/jepa_overflow/winning_harvest/recordings/` (real `Hypothesis`-agent
play). Episode length min/max/mean = 6/2001/64.8; 738/764 long enough for
depth 10. Split `in_vocab` (20 games this specific checkpoint was trained
on, n=2988 samples/depth) vs `out_of_vocab` (the 5 excluded games,
n=1341/depth, predictor falls back to `game_idx=0`).

**in_vocab, changed-patches MSE:**

| depth | compounding | fresh single-step | identity | gap (comp - fresh) | gap % |
|---|---|---|---|---|---|
| 1 | 0.000830 | 0.000830 | 0.001246 | 0.000000 | 0.0% |
| 2 | 0.002111 | 0.001245 | 0.001718 | 0.000866 | 69.6% |
| 3 | 0.003369 | 0.001185 | 0.002366 | 0.002184 | 184.4% |
| 4 | 0.005060 | 0.001172 | 0.003085 | 0.003888 | 331.8% |
| 5 | 0.006580 | 0.001068 | 0.003690 | 0.005512 | 516.1% |
| 6 | 0.008474 | 0.001036 | 0.004412 | 0.007438 | 718.2% |
| 7 | 0.010215 | 0.000989 | 0.005069 | 0.009226 | 933.2% |
| 8 | 0.012104 | 0.000927 | 0.005528 | 0.011177 | 1205.7% |
| 9 | 0.013752 | 0.000932 | 0.005733 | 0.012820 | 1375.4% |
| 10 | 0.015322 | 0.000887 | 0.006095 | 0.014434 | 1626.7% |

`out_of_vocab` (the 5 games excluded from this checkpoint's training)
shows the same fast-divergence shape at larger absolute magnitude
throughout (gap% 93%->994% from d=2->10) -- expected, since those games'
predictions are already known to be weaker (Stage 6 addendum's
held-out-games collapse), so there's more room to diverge further.

## Verdict

**The compounding-error concern was correct, and it's severe, not a
theoretical worry.** Two clean signatures, not noise:

1. **Fresh single-step error stays roughly flat with depth** (~0.0009-
   0.0013 throughout, in_vocab) -- a single prediction from real
   ground truth is about equally hard regardless of how far into an
   episode it's asked at. **Compounding rollout error grows ~18x** from
   depth 1 to depth 10 over the *same* target frames. The gap between
   them is the compounding signature itself, isolated from "some targets
   are just harder."
2. **By depth ~4-5, compounding rollout error exceeds the identity
   baseline** -- past that point, a frozen no-op prediction (nothing
   changes, ever) is *more accurate* than the model's own multi-step
   rollout. A multi-step planner scoring candidate action sequences this
   way past a handful of steps would be worse than not modeling dynamics
   at all.

**Practical read for the design question that motivated this:** a full
deep MCTS-style search through the learned predictor is not safe --
confirmed, not just plausible. A *shallow* 2-3-step lookahead already
shows a real, non-trivial gap over a fresh prediction at the same depth
(70-184%), so "shallow is free" isn't quite right either -- but it's a
different, much less catastrophic regime than depth 8-10 (1200-1600%).
If model-based multi-step lookahead is ever built, depth 2 is the
outer bound worth trusting without a strong additional justification, and
even that should be treated as a real accuracy cost, not a free lunch.
This is exactly why the graph-based lookahead (`stage6_graph_lookahead.
md`) is the right design for *this* project specifically: every hop
there is a verified fact, not a prediction stacked on a prediction, so it
has none of this failure mode regardless of depth.

## Production-checkpoint confirmation

Rerun of the identical methodology (same 764 episodes, same
`E:/jepa_overflow/winning_harvest/recordings/`, same depths/starts)
against the now-restored true 26-game production checkpoint. All 25
games fall in-vocab (the true checkpoint excludes nothing), so there's
no `out_of_vocab` split this time.

**changed-patches MSE, true production checkpoint:**

| depth | compounding | fresh single-step | identity | gap (comp - fresh) | gap % |
|---|---|---|---|---|---|
| 1 | 0.003037 | 0.003037 | 0.003455 | 0.000000 | 0.0% |
| 2 | 0.006998 | 0.003753 | 0.006442 | 0.003245 | 86.4% |
| 3 | 0.009672 | 0.003511 | 0.007946 | 0.006161 | 175.5% |
| 4 | 0.013881 | 0.003360 | 0.010184 | 0.010521 | 313.1% |
| 5 | 0.017672 | 0.002976 | 0.011900 | 0.014696 | 493.9% |
| 6 | 0.023707 | 0.003810 | 0.014970 | 0.019897 | 522.3% |
| 7 | 0.029155 | 0.003629 | 0.017477 | 0.025527 | 703.5% |
| 8 | 0.036129 | 0.004035 | 0.020903 | 0.032094 | 795.4% |
| 9 | 0.039308 | 0.004331 | 0.020915 | 0.034977 | 807.6% |
| 10 | 0.043496 | 0.003502 | 0.021589 | 0.039994 | 1142.1% |

**The finding holds, and the true checkpoint is if anything more
vulnerable to it, not less.** Same clean signature (fresh single-step
error flat at ~0.003-0.004 throughout while compounding grows ~14x from
depth 1 to 10) -- but the crossover above the identity baseline now
happens by **depth 2** (0.006998 vs 0.006442), a full 2-3 steps earlier
than the fold-1-holdout checkpoint's depth 4-5 crossover. The eventual
depth-10 gap% is somewhat lower on production (1142% vs the holdout
checkpoint's 1627%), but that's a smaller magnitude of an even-earlier
failure, not evidence the problem is milder overall.

**This strengthens, not weakens, the verdict above.** The compounding-
error risk isn't an artifact of testing against the wrong checkpoint --
it reproduces on the real production model, with the practical
implication if anything more severe: a shallow depth-2 lookahead through
the learned predictor is already, on the actual deployed checkpoint,
less accurate than simply assuming nothing changes. The graph-based
lookahead (`stage6_graph_lookahead.md`) remains the right call for
multi-step planning in this project -- every hop there is a verified
fact, immune to this failure mode by construction, regardless of depth.

## Recurrent core: does it compound less?

**Hypothesis being tested:** the Stage 3 `RecurrentActionConditionedPredictor`
(`checkpoints/encoder_recurrent.pt` / `recurrent_predictor.pt`, 25-game
vocab, no MiniGrid pretraining -- a materially different checkpoint
lineage from the MoE production one, not just a different architecture)
carries a compressed GRU hidden state across steps instead of re-deriving
everything from a decaying raw feature map each time. A real,
architecturally-motivated reason it *might* degrade more gracefully under
pure multi-step rollout than the memoryless MoE predictor.

**Method:** `scripts/diagnose_rollout_compounding_error.py --model
recurrent`, extended with a `run_episode_recurrent` path that mirrors the
MoE methodology exactly, generalized to the recurrent model's extra
state:

- **Compounding rollout**: both the feature map *and* the hidden state
  are chained purely from the model's own prior outputs -- `cur_hidden`
  is whatever the previous compounding step's `forward()` call returned,
  never reset to a ground-truth value mid-rollout. This represents what a
  real multi-step planner through this model, with no ground truth
  reinjected along the way, would actually see.
- **Fresh single-step baseline**: one predictor call from the REAL
  observed frame at `t+d-1`, combined with the REAL-history hidden state
  at `t+d-1` -- precomputed by advancing the GRU using only real observed
  frames and real actions from episode start (exactly what
  `memory_agent.py`'s live deployment accumulates in `self._hidden`,
  since it always calls `_predict` with the actually-observed previous
  feat, never a self-generated one). This isolates "does compounding
  through *both* self-predicted features and self-predicted hidden state
  add error beyond a fresh call with perfect real history," the direct
  generalization of the MoE script's own fresh-baseline framing.
- Identity baseline unchanged (frame at `t` held constant).

Same `changed-patches` metric (`jepa/losses.py: per_region_error`), same
`E:/jepa_overflow/winning_harvest/recordings/` data (764 RESET-delimited
episodes, 738/764 long enough for depth 10 -- identical corpus to the
MoE production-checkpoint run above, so the comparison is apples-to-apples
on data, not just on methodology), depths 1-10, default seed/starts.

**A real bug hit and fixed along the way, unrelated to the model
comparison itself:** the first run of this diagnostic crashed with a
`FileNotFoundError` mid-load -- a recording file present when
`recordings_dir.glob(...)` ran had been deleted by some other, external
process before it was opened (this recordings directory is documented in
CLAUDE.md as concurrently modified by other sessions/harvests; the file
count dropped from 153 to 152 between two checks made minutes apart
during this same task). Fixed by wrapping the per-file load in both
`jepa/data/trajectories.py: load_transitions_from_dir` and this script's
own `load_episodes` in a `try/except FileNotFoundError` that skips the
vanished file and continues -- a real, low-risk robustness fix worth
keeping regardless of this specific diagnostic, not a one-off workaround.

**Results (winning_harvest, in_vocab, changed-patches MSE):**

| depth | compounding | fresh single-step | identity | gap (comp - fresh) | gap % |
|---|---|---|---|---|---|
| 1 | 0.004320 | 0.004320 | 0.004416 | 0.000000 | 0.0% |
| 2 | 0.010525 | 0.005627 | 0.007740 | 0.004899 | 87.1% |
| 3 | 0.019729 | 0.005152 | 0.012422 | 0.014577 | 282.9% |
| 4 | 0.031950 | 0.006167 | 0.017626 | 0.025783 | 418.1% |
| 5 | 0.046045 | 0.006115 | 0.022265 | 0.039930 | 653.0% |
| 6 | 0.061737 | 0.006168 | 0.026529 | 0.055569 | 901.0% |
| 7 | 0.079230 | 0.005999 | 0.031122 | 0.073231 | 1220.6% |
| 8 | 0.098301 | 0.006037 | 0.034465 | 0.092264 | 1528.3% |
| 9 | 0.121010 | 0.005505 | 0.036976 | 0.115505 | 2098.1% |
| 10 | 0.143907 | 0.006323 | 0.039566 | 0.137584 | 2176.0% |

(local recordings, in_vocab, a second independent corpus: same shape --
0.0% gap at d=1, crossover above identity at d=2 (0.011322 vs 0.011949 is
still just under identity, but by d=3 the recurrent model is clearly
worse and the gap accelerates the same way -- 1115.8% by d=10. Full table
in the diagnostic's stdout / `scripts_out_recurrent_rollout.json`.)

**Direct comparison to the MoE production checkpoint, same 764-episode
`winning_harvest` corpus, same depths:**

| depth | MoE comp | MoE gap% | recurrent comp | recurrent gap% |
|---|---|---|---|---|
| 1 | 0.003037 | 0.0% | 0.004320 | 0.0% |
| 2 | 0.006998 | 86.4% | 0.010525 | 87.1% |
| 5 | 0.017672 | 493.9% | 0.046045 | 653.0% |
| 10 | 0.043496 | 1142.1% | 0.143907 | 2176.0% |

**Verdict: the hypothesis is rejected. The recurrent core compounds
*more* severely than the memoryless MoE predictor, not less** -- worse on
every axis that matters:

- **Absolute magnitude at depth 10 is over 3x larger** (0.143907 vs.
  0.043496 changed-patches MSE) despite both models starting from
  essentially the same depth-1 error (0.0043 vs. 0.0030 -- comparable
  orders of magnitude, not an unfair head start).
  - **Growth rate d1->d10 is more than double**: recurrent compounding
  error grows **~33x** from depth 1 to depth 10, vs. the MoE predictor's
  own **~14x** (both already documented as severe in this doc's earlier
  sections -- the recurrent core is meaningfully worse on the same
  metric).
- **gap% at depth 10 is roughly double** (2176% vs. 1142%).
- The one place the two models *do* agree: both cross above the identity
  baseline by depth 2, and both show the same clean fresh-vs-compounding
  signature (fresh single-step error stays flat, ~0.005-0.006 for
  recurrent vs. ~0.003-0.004 for MoE, throughout all 10 depths, while
  compounding grows monotonically) -- confirming this is a real
  compounding-error effect for the recurrent model too, not a data
  artifact, just a *worse* one.

**Why, mechanistically -- the architecturally-motivated reason for the
hypothesis turns out to cut the other way.** The GRU hidden state was
expected to insulate the model from feature-map drift by carrying a
"cleaner" compressed summary across steps. But in a pure rollout with no
ground truth ever reinjected, `new_hidden` is computed each step from
`pooled_feat = feat.mean(dim=(2,3))` where `feat` *is* the previous
step's drifting, self-generated prediction (see
`RecurrentActionConditionedPredictor.forward` -- the pooled summary feeds
the GRU cell every call). So instead of one error-accumulating pathway
(the feature map, as in the MoE model), pure rollout through the
recurrent model has **two**: the feature map drifts *and* contaminates
the hidden state each step, and the (now-also-drifting) hidden state
feeds back into the *next* step's prediction as one more conditioning
input alongside the drifting feature map -- compounding on top of
compounding, not a separate clean channel. The architecture's real
strength (carrying real information across steps) only helps when the
hidden state is anchored to ground truth at every step, exactly how
`memory_agent.py` actually uses it in live play (hidden always advances
from the real observed feat, never a self-generated one) -- it is not a
free structural defense against rollout error when ground truth is
absent, which is exactly the scenario a multi-step planner through the
learned model would face.

**Practical takeaway for the design question that motivated this whole
investigation:** the recurrent core does not offer a cheaper or safer
path to multi-step lookahead than the MoE predictor -- if anything it
makes the case for the graph-based lookahead (`stage6_graph_lookahead.md`)
stronger, since neither of this project's two production-scale
architectures tolerates rollout past a couple of steps. Given Task 1's
result answers the motivating question decisively in the negative and
this is a materially different (worse) model, not a variant expected to
need further tuning, this does not change how much effort Task 2's
residual-under-commitment fine-tune deserves on the MoE model
specifically -- that investigation proceeds on its own merits below.

## Does fixing residual under-commitment reduce compounding error?

**Hypothesis:** the production `MoEPredictor` is known (Stage 1 item 8,
and the Stage 6 held-out-games section) to under-commit to real residuals
-- it partly coasts on the `feat +` skip-connection instead of predicting
the true delta. A companion investigation on the *scaled* (85M-param)
architecture found a targeted fine-tune could push its residual/true-delta
ratio up to 0.219-0.382 (from a presumably-lower baseline). If exposure
bias/under-commitment is a real contributor to the compounding-error
problem documented earlier in this doc, fixing it on the production model
should show up as a flatter depth-wise gap% curve, not just a better
single-step number.

**Step 1 -- baseline, measured for the first time this session**
(`scripts/diagnose_production_residual_commitment.py`, same methodology
as `diagnose_scaled_launch_collapse.py`, held-out slice of the local
ARC-3 recordings corpus, true production checkpoint):

- Encoder feature std: mean 1.64 (well above `VARIANCE_FLOOR=1.0` -- no
  collapse), change-sensitivity ratio 22.6x (changed vs. unchanged
  patches) -- the encoder itself is healthy, consistent with every prior
  finding in this project that the encoder was never the bottleneck.
- **Residual/true-delta ratio: 0.129 (all patches), 0.054 (changed
  patches only).** Confirms the production predictor under-commits
  severely -- comparable in severity to what Stage 1 item 8 originally
  found on the monolithic predictor, and well below the scaled model's
  post-fix 0.219-0.382 range.

**Step 2 -- fine-tune** (`scripts/finetune_production_residual.py`):
warm-started fresh `CNNEncoder`/`MoEPredictor` instances directly from
`checkpoints/encoder_moe.pt` / `moe_predictor.pt` (true production
weights, same 26-entry vocab, no new embeddings), then continued training
for 20 epochs on `ARC-AGI-3-Agents/recordings/` (13,800 local transitions,
real data only -- no MiniGrid/external re-pretrain) using the exact same
loss/eval machinery as `train_moe_predictor.py`'s arc-finetune phase
(`_make_loaders`/`_run_epochs`, imported directly, not reimplemented),
lr=1e-4 (lower than the original 3e-4, appropriate for continued
fine-tuning rather than from-scratch training). Training was healthy
throughout -- no collapse, real learning: changed-patches improvement
over identity grew from ~22% at epoch 1 to **~53%** by epoch 20 on this
run's own held-out split (`pred=0.00586` vs. `identity=0.01252`). Saved
to `checkpoints_residual_finetune/` only -- `checkpoints/` was never
touched.

A real, incidental bug hit and fixed here too: the first attempt crashed
with the same concurrent-modification `FileNotFoundError` as Task 1 (a
recording file vanished between glob and open) -- fixed once, centrally,
in `jepa/data/trajectories.py: load_transitions_from_dir` (see Task 1's
section above), which this fine-tune script also depends on, so no
separate fix was needed here.

**Step 3 -- did the ratio actually move? Yes, dramatically:**

| | all patches | changed patches only |
|---|---|---|
| baseline (production) | 0.129 | 0.054 |
| fine-tuned | 0.577 | 0.576 |

A **>10x increase** in the changed-patches ratio, landing well above even
the scaled model's post-fix 0.219-0.382 range -- this fine-tune achieved
its stated, narrow goal decisively. (Side note: the fine-tuned model's
encoder is *not* latent-space-identical to production's, since
`train_moe_predictor.py`'s training loop updates the online encoder
jointly with the predictor -- expected and unavoidable for a same-recipe
continued fine-tune, not a confound in what follows since the comparison
below is behavioral, both checkpoints run through the identical
diagnostic.)

**Step 4 -- does the compounding-error curve actually improve?**
Re-ran `scripts/diagnose_rollout_compounding_error.py --checkpoint-dir
checkpoints_residual_finetune` (new `--checkpoint-dir` flag added this
session), identical methodology/data/depths to the production-checkpoint
run earlier in this doc (same 764-episode `winning_harvest` corpus,
in_vocab):

| depth | production comp | production gap% | fine-tuned comp | fine-tuned gap% |
|---|---|---|---|---|
| 1 | 0.003037 | 0.0% | 0.002339 | 0.0% |
| 2 | 0.006998 | 86.4% | 0.006598 | 98.2% |
| 3 | 0.009672 | 175.5% | 0.010154 | 234.8% |
| 4 | 0.013881 | 313.1% | 0.015348 | 423.3% |
| 5 | 0.017672 | 493.9% | 0.020766 | 745.9% |
| 6 | 0.023707 | 522.3% | 0.026509 | 839.5% |
| 7 | 0.029155 | 703.5% | 0.033028 | 943.8% |
| 8 | 0.036129 | 795.4% | 0.041228 | 1380.8% |
| 9 | 0.039308 | 807.6% | 0.046439 | 1157.9% |
| 10 | 0.043496 | 1142.1% | 0.052666 | 1587.0% |

**Verdict: no, and if anything the opposite happened.** At every single
depth from 2 through 10, the fine-tuned checkpoint's compounding rollout
error is *higher* in absolute magnitude than production's, and its gap%
over the fresh baseline is higher at every depth too. Growth from depth 1
to depth 10 is **~22.5x** for the fine-tuned checkpoint vs. production's
own **~14.3x** -- proportionally *faster* compounding, not slower. The
one place the fine-tune helped is exactly where it was trained to help:
the depth-1 (single-step, ground-truth-conditioned) error is genuinely
lower (0.002339 vs. 0.003037), and the "fresh single-step" baseline stays
lower across most depths too (e.g. depth 5: 0.002455 vs. 0.002976) --
real single-step prediction quality improved, consistent with Step 3's
ratio finding. That improvement just doesn't propagate into the
compounding regime; if anything it's outweighed there.

**Why this makes sense in hindsight, not just "surprising negative
result":** residual under-commitment and rollout compounding turn out to
be more independent than the motivating hypothesis assumed, and the
mechanism cuts the *opposite* direction from what "fix the exposure-bias
symptom" would predict. A predictor that under-commits (coasts toward
`feat +` ~0) is, by construction, numerically conservative under rollout
-- even fed an already-wrong, self-generated input, it can only drift a
small amount per step, since its residuals are small everywhere by
training-time habit. A predictor that has learned to commit confidently
to larger, more accurate residuals *on in-distribution (ground-truth)
inputs* has no such brake once it's fed an out-of-distribution
self-generated input during pure rollout -- the same confident
commitment that improves real single-step accuracy also means each
compounding step can inject a larger, confidently-wrong correction when
the input has already drifted off the training manifold. Under-commitment
was never a *cause* of compounding error in the sense of "if the model
just tried harder, rollout would be more stable" -- it was, if anything,
accidentally *damping* it. Fixing the single-step symptom removed that
accidental damping without addressing the actual root cause (exposure
bias: the model has still never been trained on its own rolled-out
outputs as input, regardless of how confidently it commits to residuals
on real inputs).

**Practical takeaway:** this closes out the residual-under-commitment
hypothesis as a lever for the compounding-error problem -- a real,
useful, honest negative result, not a failed experiment to bury. It does
not change the standing recommendation from earlier in this doc: the
graph-based lookahead (`stage6_graph_lookahead.md`) remains the right
design for multi-step reasoning in this project, since it's immune to
this entire failure mode by construction rather than needing it
tuned away. If compounding error is ever worth revisiting directly
(rather than routing around it via the graph), the more promising lever
per the mechanism identified here would be training the predictor on its
*own* rolled-out outputs as input some fraction of the time (scheduled
sampling / DAgger-style rollout-augmented training) -- directly targeting
exposure bias itself, rather than a proxy metric (residual magnitude)
that turned out not to be causally upstream of it.

## Files touched (Task 1 + Task 2, this session)

- `scripts/diagnose_rollout_compounding_error.py` -- added `--model
  recurrent` (with `run_episode_recurrent`/`load_models_recurrent`) and
  `--checkpoint-dir` (moe model only); fixed a concurrent-modification
  `FileNotFoundError` in `load_episodes`.
- `jepa/data/trajectories.py` -- same concurrent-modification fix in
  `load_transitions_from_dir`.
- `scripts/diagnose_production_residual_commitment.py` -- new; production-
  scale analogue of `diagnose_scaled_launch_collapse.py`'s
  residual-commitment check.
- `scripts/finetune_production_residual.py` -- new; warm-starts from the
  true production checkpoint and continues training on real local ARC-3
  data only. Writes exclusively to `checkpoints_residual_finetune/` --
  `checkpoints/` was never modified by this task.
- `checkpoints_residual_finetune/` -- fine-tuned checkpoint from this
  task (not production, not to be confused with it -- see the checkpoint-
  integrity gotcha in CLAUDE.md before ever pointing production-facing
  code at a directory other than `checkpoints/`).
