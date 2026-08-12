# Stage 6: wiring Stage 3's recurrent core into live exploration for the first time

**Status: COMPLETE. Clean negative for both retrospective memory AND real
multi-step lookahead search, on both games, at both budgets tested --
and mechanistically explained: the recurrent world model itself collapses
to identity on held-out games, so nothing built on top of it (better
ranking, real memory, real search) has real signal to search over. See
the "Follow-up" section below for the search result and root-cause
diagnostic.**

## Motivation

`experiments/stage6_action_selection_softmax.md` found `bp35` and `ka59`
hard-cap actions *per attempt* (`ka59` exactly 100, `bp35` up to 64,
independent of total `MAX_ACTIONS`), while `tr87` -- the one held-out game
that occasionally solves -- has no such tight cap (128-150/attempt). All
exploration tested against these two games so far (`Hypothesis`'s
InfoGain/value blend, `RecurrentCuriosity` below) is otherwise
memoryless *between* candidate-action evaluations within a turn -- it
has no representation of "what have I already tried this attempt" beyond
the `TransitionGraph`'s exact-recall (which only helps once a state has
already been visited with a known outcome, not for finding a first win).

Stage 3 built exactly the component meant to address this --
`jepa/models/recurrent_predictor.py: RecurrentActionConditionedPredictor`,
a GRU-based core carrying compressed history across an episode -- but
per CLAUDE.md's own note, **no agent had ever actually used it for live
action selection**; it was only ever evaluated on its own changed-patches
metric. This experiment wires it in for the first time and tests it
specifically where the per-attempt-cap finding says memory might matter
most.

## What was built

**1. `--exclude-games` support for the recurrent-predictor training
pipeline** (`jepa/data/sequences.py: load_all_episodes`,
`jepa/train_recurrent_predictor.py`) -- mirrors `trajectories.py: load_
all_transitions`'s existing flag exactly. The only prior recurrent
checkpoint (`checkpoints/recurrent_predictor.pt`) was trained on all 25
games, including `bp35`/`ka59` -- not a fair zero-shot test consistent
with the rest of this session's held-out-games methodology. Retrained
excluding `r11l,bp35,m0r0,tr87,ka59` (identical exclusion set to the
MoE checkpoint's own `checkpoints_holdout_baseline`), 30 epochs,
`checkpoints_recurrent_holdout/`. Trained cleanly: changed-patches
improvement +11% at epoch 30 (pred=0.00208 vs identity=0.00234) on the
20 trained games' own held-out split -- a real, modest pass, consistent
with Stage 3's original milestone number.

**2. `RecurrentCuriosity`**
(`ARC-AGI-3-Agents/agents/templates/recurrent_curiosity_agent.py`) --
structurally `Curiosity` (Stage 2) with two changes:
- The stateless Stage 1 predictor is swapped for the held-out recurrent
  predictor above, with a real `GRUCell` hidden state maintained across
  the whole episode (reset only on `RESET`, matching
  `RecurrentActionConditionedPredictor.init_hidden`'s documented reset
  boundary -- carrying it across a `RESET` would mix unrelated attempts'
  histories, the same reasoning `TransitionGraph` and `TestTimeAdapter`
  already use elsewhere in this project for their own reset boundaries).
  Hidden-state bookkeeping: each turn recomputes what the model predicted
  *last* turn (using the hidden state as of *before* that turn) to derive
  the observed-surprise signal, and only then advances `self._hidden` to
  the value that results from actually replaying that transition --
  so the hidden state genuinely accumulates real observed history, not
  replayed predictions.
- The top-level action ranking uses the same temperature-softmax sample
  over the EMA surprise scores that `hypothesis_agent.py`'s own
  `ACTION_SAMPLE_TEMPERATURE` fix uses (see `stage6_action_selection_
  softmax.md`). Worth flagging directly: `Curiosity`'s ORIGINAL top-level
  ranking is a hard argmax over `_action_surprise` -- the exact same
  "deterministic argmax on a near-flat map" bug just fixed in
  `Hypothesis`, never noticed in `Curiosity` because no prior diagnosis
  traced it that closely. Fixed here from the start rather than copying a
  known bug into a brand new agent.

Everything else (ACTION6 competing as one top-level option, click
location resolved via softmax-sampled patch selection, exploit-on-
level-up, epsilon-random fallback) is `Curiosity`'s own already-debugged
design, unchanged.

## Backtest

n=8 repeats each on `bp35` and `ka59`, `checkpoints_recurrent_holdout/`,
matching `Hypothesis`'s own already-established protocol on these games.
Tested at both the standard `MAX_ACTIONS=300` and the `MAX_ACTIONS=2500`
budget (temporarily bumped, then reverted -- same bump-and-revert
pattern used throughout this project) that was necessary to get `tr87`
to show any signal at all with `Hypothesis`, since `MAX_ACTIONS=300`
only buys ~3-4 attempts against these games' own hard per-attempt caps.

| condition | bp35 | ka59 |
|---|---|---|
| `RecurrentCuriosity`, 300 actions (n=8) | 0/8 | 0/8 |
| `RecurrentCuriosity`, 2500 actions (n=8) | 0/8 | 0/8 |
| `Hypothesis` (fixed), 300 actions (n=8, from `stage6_action_selection_softmax.md`) | 0/8 | 0/8 |
| `Hypothesis` (fixed), 2500 actions (n=8, from `stage6_action_selection_softmax.md`) | 0/8 | 0/8 |

**A clean, unambiguous 0/32 across every condition tested for these two
games.** Real accumulated episode history -- the one component in this
whole project built specifically to give exploration a sense of "what
have I already tried" and never previously plugged into a live
decision -- made no detectable difference, at either budget.

## Reading this honestly

This is a real negative result, not an inconclusive one -- 0/8 vs 0/8 at
two different budgets is about as clean a null as this project's sparse
completion metric can produce. Worth being precise about what was and
wasn't actually tested, though: `RecurrentCuriosity`'s hidden state
improves the *retrospective surprise signal* (better history-conditioned
prediction quality feeding the EMA ranking) -- it does NOT do any
explicit forward *planning* or *lookahead* using that hidden state.
Action selection is still "rank cached per-action EMA scores," exactly
like `Curiosity` and `Hypothesis` both already do; the hidden state
changes how good the *retrospective* surprise estimate is, not whether
the agent can simulate "if I do X now, then Y, where does that leave me"
before committing to a first step. If `bp35`/`ka59`'s real bottleneck is
that kind of genuine multi-step lookahead (plausible, given their tight
per-attempt caps demand getting a specific sequence right quickly, not
just noticing surprising outcomes after the fact), this experiment
doesn't rule that out -- it rules out "better within-episode memory for
scoring already-tried actions," a real but narrower claim than "planning
doesn't help."

**Where this leaves the investigation:** the held-out-games breadth gap
on `bp35`/`ka59` has now survived every lever tried across this entire
session -- 13+ world-model interventions (data diversity, conditioning,
TTA, meta-learning), the action-selection argmax fix, budget scaling to
2500 actions, and now real recurrent episode memory. `tr87` is the only
one of the three that has ever moved (2/9 at 2500 actions with the
argmax fix). The next lever worth trying, if this is revisited, is
genuine forward search/lookahead (actually simulating candidate action
sequences with the world model before committing, e.g. a shallow MCTS or
beam search over the recurrent predictor's own rollout) rather than any
further variation on retrospective surprise-based ranking -- this
experiment is reasonably strong evidence that the "smarter ranking of
already-observed outcomes" family of fixes is exhausted for these two
games specifically.

## Follow-up: real lookahead/search, and why it also fails -- mechanistically confirmed

Per this doc's own recommendation, built `RecurrentSearch`
(`ARC-AGI-3-Agents/agents/templates/recurrent_search_agent.py`): genuine
random-shooting MPC over the recurrent predictor, not retrospective
ranking. Each decision: for every candidate first action (all simple
actions + 6 uniformly-sampled ACTION6 click points), simulate
`SEARCH_DEPTH=3` steps forward *purely in feature space* (predicted
features fed back in as the next "current" state -- no CNN re-encoding,
no real environment interaction), scored by the max *reachable novelty*
along the trajectory (min L2 distance from each imagined pooled feature
to any REAL feature already observed this episode, via an episodic
memory that only ever records true observations). The highest-scoring
candidate's first action is taken -- standard random-shooting MPC, the
natural lightweight search when you have a learned dynamics model and no
reward model. `self._hidden`/`self._memory` only ever advance from real
transitions; the scratch hidden state used inside a single search call
is a clone, discarded after scoring.

**Backtest (n=8 each, `bp35`/`ka59`, both 300 and 2500 actions, same
protocol as `RecurrentCuriosity` above): 0/32.** A third fundamentally
different exploration strategy -- after `Hypothesis`'s InfoGain/value
blend and `RecurrentCuriosity`'s retrospective surprise ranking -- comes
back a clean null. Combined total across this whole investigation: **0/48
across three architecturally distinct exploration strategies, at two
budgets each, on these two specific games.**

**Diagnosed the mechanism rather than leaving three unexplained nulls
side by side.** Built `scripts/diagnose_recurrent_residual_holdout.py`,
mirroring the exact diagnostic Stage 6 already used to explain the MoE
predictor's own held-out-game collapse (CLAUDE.md's Stage 6 addendum,
item 3): stream real local-recording transitions with a real accumulated
hidden state through the recurrent predictor, measure the residual
branch's magnitude (pre-skip-connection, i.e. `self.net(x)` before
`feat + self.net(x)`) relative to the true observed feature delta.

| | mean residual^2 | mean true-delta^2 | residual/true-delta ratio |
|---|---|---|---|
| held-out games (`r11l,bp35,m0r0,tr87,ka59`, n=244,900 transitions) | 3.05e-4 | 7.42e-3 | **0.041** |
| trained games (60-episode sample, n=9,420 transitions) | 1.35e-4 | 4.15e-4 | **0.325** |

**The recurrent predictor's residual branch is ~8x weaker relative to the
true change on held-out games than on trained ones -- it is coasting to
identity on unfamiliar games, structurally the same failure Stage 6
already found and named for the MoE predictor.** This is not a new bug;
it's the same root cause reappearing in an independently-trained model,
which is itself informative: two architecturally different predictors
(MoE gating vs. a monolithic GRU-conditioned model), trained on the same
~20-game corpus, both learn to hedge toward "predict no change" on a
genuinely novel game rather than commit to a real residual.

**This is a real, mechanistic explanation for the triple null, not
speculation:** `RecurrentSearch`'s whole exploration signal depends on the
model imagining *meaningfully different* futures for different candidate
actions. If the model instead predicts "almost nothing changes"
regardless of which action or click is imagined, every candidate's
simulated trajectory looks nearly identical to the current state and to
each other -- the novelty-scoring objective has almost no real signal to
discriminate on, and the sophisticated MPC machinery degrades toward
picking among near-tied, uninformative options (functionally close to
the epsilon-random fallback, which is exactly what the 0/32 result looks
like). The bottleneck was never really "the exploration *strategy*
wasn't smart enough" -- retrospective ranking, real episodic memory, and
now genuine multi-step search were all tried and all hit the same wall
-- it's that **the world model these strategies search *over* has
nothing real to offer on a genuinely unseen game**, the exact same
data-bound ceiling this project's Stage 6 addendum already established
for the MoE predictor via 7+ independent conditioning/architecture
interventions, now confirmed to extend to a second, independently-built
predictor architecture too.

**Where this leaves the investigation, for real this time:** every lever
this project can build *on top of* a frozen-or-lightly-adapted world
model trained on ~20-25 ARC-3 games -- better ranking, real memory,
real search -- has now been tried against `bp35`/`ka59` specifically and
failed identically, for the same underlying reason each time. Per Stage
6's own accumulated conclusion (13+ interventions against the broader
held-out-games gap, all converging the same way): closing this requires
either genuinely more diverse *training* data (the one lever that worked
once, for Stage 4's MoE gate specialization via MiniGrid) or a
fundamentally different adaptation mechanism (test-time adaptation showed
a real if modest representation-level effect earlier this session,
`experiments/stage6_test_time_adaptation_agent.md`) -- not another
exploration-strategy variant layered on the current world model. This is
a good, well-evidenced stopping point for the exploration-strategy family
of fixes specifically.

## Follow-up 2: does test-time adaptation actually help at the agent level? Two more negative results, one real training-curve lesson

`scripts/test_time_adaptation_recurrent.py` (built in a prior session)
found TTA substantially narrows the residual-collapse gap on `bp35`/
`ka59` specifically in representation space (changed-patches `bp35`:
-9.76% -> -0.51%, `ka59`: -6.51% -> -1.84%, at n=200 observed
transitions) -- bigger than TTA's effect on the MoE predictor. Two
follow-ups tested whether that translates into real play, per the user's
own ranked list of adaptation methods to try in order, stopping to
investigate only if something looked drastically better or changed the
calculus.

### 1. Wire TTA into `RecurrentSearch`, backtest on `bp35`/`ka59`

Built `RecurrentSearchTTA`
(`ARC-AGI-3-Agents/agents/templates/recurrent_search_tta_agent.py`),
extending `RecurrentSearch` unchanged except for a live adaptation loop:
every 5 newly observed real transitions (`TTA_K=5`), run 8 AdamW steps
(`TTA_STEPS=8`, `lr=5e-5`) on `predictor.net[-1]` only, sampling a fresh
16-step chunk from the accumulated real-transition buffer each step --
the exact operating point the diagnostic validated. Adaptation persists
across RESETs of the same game (mirrors `TestTimeAdapter`'s established
design), only a fresh agent instance per game resets it. Manually
registered in `AVAILABLE_AGENTS` (subclasses `RecurrentSearch`, not
`Agent` directly, so `Agent.__subclasses__()` doesn't auto-discover it --
same pattern `ReasoningAgent` already uses).

**Result: 0/16** (n=8 each, `bp35`/`ka59`, `MAX_ACTIONS=300`) -- identical
null to the frozen-model baseline. The representation-level gain (closing
most of a ~7-10 percentage point gap) did not survive contact with real
play, the same "component measurably improved, agent-level result didn't
move" pattern this project has now hit repeatedly (Stage 5's
teacher-policy value head, the MoE predictor's own TTA backtest, Reptile
meta-learning for the MoE predictor).

### 2. Reptile meta-learning for the recurrent predictor -- a real training-curve lesson, and a second negative

Built `jepa/train_recurrent_meta_predictor.py`, porting
`jepa/train_meta_predictor.py`'s design (built for the MoE predictor) to
`RecurrentActionConditionedPredictor`. Reused the already-learned lesson
from that port directly rather than re-discovering it: the head
(`predictor.net[-1]`, the same subset TTA adapts) stays in ordinary joint
SGD throughout, with a periodic Reptile nudge layered on top -- a pure
ANIL split (head frozen from ordinary training) is already documented to
cause catastrophic representation collapse for the MoE predictor, and
there was no reason to expect a different architecture to be immune.
Per-game Reptile task pools are built from real per-episode sequences
(not flattened transitions), so each inner-loop chunk has genuine
temporal continuity, mirroring the diagnostic's own chunk-sampling
convention exactly. Trained with the "high-dose" recipe already validated
for the MoE predictor (3x updates/epoch, no epsilon annealing) directly,
rather than re-running the "standard dose fails" step of that discovery
first.

**A real, if smaller, collapse signature showed up anyway, despite
avoiding the specific bug already fixed once.** `val_pred_mse` and
`val_identity_mse` shrank together across all 30 epochs to genuine
near-zero (epoch 30: pred=0.00026, identity=0.00027 on the whole-grid
metric -- both ~9x smaller in absolute terms than the non-meta baseline's
own epoch-30 identity MSE of 0.00234). The *relative* changed-patches
improvement tells the same story more precisely: it peaked at **+32.5%**
around epoch 10-11 (where absolute identity MSE, ~0.0022, actually
matches the non-meta baseline's own epoch-30 scale) and eroded steadily
to **+7.0%** by epoch 30 -- the checkpoint kept "improving" on raw MSE
while its *real* predictive edge over identity was quietly shrinking, a
clear sign of over-training into a collapsing regime rather than genuine
convergence. Not a repeat of the documented ANIL bug (the head was never
frozen from joint SGD here) -- more likely the extra 120 Reptile-driven
head-only gradient steps per epoch (15 outer updates x 8 inner steps),
applied on top of a much smaller corpus (181 episodes) than the MoE
version ever trained on, simply over-trained the head faster relative to
the body than the MoE recipe's own dose ratio did. **Lesson for any
future revisit: save intermediate checkpoints (this script doesn't --
a real gap relative to `train_meta_predictor.py`, which has
`--checkpoint-every`) so an earlier, less-collapsed epoch can be
evaluated directly instead of only ever having the final, most-converged
one.**

**Representation-level result on `bp35`/`ka59` (via the same TTA
diagnostic, pointed at this checkpoint): both games cross into positive
changed-patches territory for the first time all session** -- `bp35`:
zero-shot -0.19% -> TTA @ n=200 **+0.94%**; `ka59`: zero-shot -1.01% ->
TTA @ n=200 **+0.71%**. But the absolute margins are tiny and directly
consistent with the collapse reading, not a real breakthrough: `bp35`'s
zero-shot starting point (-0.19%) is already far closer to parity than
the non-meta checkpoint's own zero-shot starting point (-9.76%) --
exactly what a partially-collapsed model would show on *any* game,
trained or held-out, since a model that mostly predicts near-identity
everywhere has almost no relative gap to close in the first place.
Absolute pred-vs-identity MSE gaps at the best (n=200) checkpoint are
correspondingly minuscule (`bp35`: 0.018892 vs 0.019071, a gap of
0.000179; `ka59`: 0.000328 vs 0.000330, a gap of 0.000002) -- on the same
order of magnitude as the "small absolute error swing = huge relative
swing" artifact CLAUDE.md's own Stage 1 history already flags for
low-identity-MSE games.

**Agent-level backtest (`RecurrentSearchTTA` pointed at this checkpoint
via a new `RECURRENT_CHECKPOINT_DIR` env override, same n=8x2-game
protocol): 0/16, identical to every other condition.** Predicted before
running, given a *much larger* representation-level movement from plain
TTA on the non-meta checkpoint already failed to produce a single level
completion -- this much smaller, barely-positive movement was never
likely to do better, and didn't. Confirms this is noise-adjacent, not a
real finding: not investigated further per the "carry on if it proves a
fluke" instruction.

**Net read on adaptation methods 1-2 of the ranked list**: both are clean
negatives at the agent level, for the same underlying reason -- neither
changes the fact that `bp35`/`ka59`'s hard, game-intrinsic per-attempt
action caps (`ka59` exactly 100, `bp35` up to 64) leave too little room
per attempt for any of these small representation-level edges to matter
before the attempt ends. Moving to option 3 (higher TTA dose) next.

## Follow-up 3: higher TTA dose closes the gap decisively in representation space -- and it still doesn't matter

Per the user's ranked list, option 3: retest TTA at a higher dose, per
the same "the conservative dose was needlessly conservative" lesson
already learned once for the MoE predictor's own TTA agent
(`experiments/stage6_test_time_adaptation_agent.md` -- the production
dose was chosen against a fictitious cross-game-interference cost, since
`TestTimeAdapter`/this agent's own adaptation both reset per game
anyway). Bumped `scripts/test_time_adaptation_recurrent.py`'s dose from
`STEPS=8, LR=5e-5` to `STEPS=25, LR=2e-4` (matching the exact values that
worked for the MoE predictor), tested on the ORIGINAL (non-meta,
non-collapsed) `checkpoints_recurrent_holdout` -- not the Reptile
checkpoint above, since that one's own real signal was already
established to be much smaller.

**Representation-level result: the biggest, most unambiguous swing of
this entire investigation.** `bp35`: zero-shot -9.76% -> **+20.74%** at
n=200 observed transitions, with a real, large absolute MSE gap
(pred=0.029881 vs identity=0.037699, a gap of 0.0078 on an absolute scale
of ~0.03-0.04 -- nothing like the meta checkpoint's noise-level margins
above). `ka59` also moved substantially, though non-monotonically and
not fully across zero (-6.51% -> -7.71% at n=50 -> **-0.48%** at n=200) --
still real, directional progress, just slower to converge than `bp35`.
This is not a fluke: the absolute magnitude and the clean, large gap on
`bp35` rule out the "tiny numbers, tiny noise" reading that applied to
the meta checkpoint's own crossing-to-positive result above.

**Bumped `RecurrentSearchTTA`'s `TTA_STEPS`/`TTA_LR` to match (bump-and-
revert), reran the full n=8x2-game agent-level backtest: 0/16, identical
to every prior condition.** Even the single largest, most convincingly
real representation-level improvement of the whole session -- a world
model that now genuinely predicts `bp35`'s dynamics *better than
identity* by a wide, real margin -- produced exactly as many additional
level completions as a completely frozen, never-adapted model: zero,
across all 16 real attempts.

**This is the decisive result, not another data point to average in with
the others.** Every previous negative in this whole `bp35`/`ka59`
investigation could still be explained as "the intervention's
representation-level effect just wasn't big or real enough yet." That
explanation no longer survives: this is about as large and clean a
world-model improvement as this project has ever produced on these two
specific games, deliberately chosen and tested for exactly this reason,
and it still didn't move the needle by even one level completion. The
bottleneck for `bp35`/`ka59` is very unlikely to be world-model quality
at all -- it is much better explained by the structural finding
`experiments/stage6_action_selection_softmax.md` already made and this
doc's own introduction restates: **these two games hard-cap actions per
attempt** (`ka59` exactly 100, `bp35` up to 64) **independent of total
budget.** A better-adapted world model can only make each of those few
dozen actions per attempt marginally better-informed; it cannot buy more
attempts, and if these games require finding a specific, non-obvious
action sequence within a tight per-attempt window (rather than rewarding
"noticing surprising outcomes" or "reaching novel states," which is what
every exploration strategy tried so far actually optimizes for), no
amount of world-model quality improvement targets the real constraint.

**Recommendation, not pursued further this round per the "stop and
postulate" instruction once a calculus-changing result appears:** options
4-6 on the original ranked list (adapt more predictor parameters, warm
the hidden state during TTA, also adapt the encoder) are all further
variations on "make the world model even better/more adapted" -- the same
family of intervention this follow-up just showed doesn't matter even at
a much larger, cleanly-real magnitude than anything tried before. Not
worth running them against `bp35`/`ka59` specifically before addressing
the structural finding directly. Two directions that target the actual
constraint instead: (a) a genuinely tight, per-attempt-budget-aware
search/planning strategy that treats the small per-attempt action count
as a hard planning horizon rather than an incidental budget limit (none
of the strategies tried -- retrospective ranking, real episodic memory,
random-shooting MPC -- were designed with "you get ~60-100 actions and
then it's over, permanently" as an explicit constraint); or (b) directly
inspecting what actually happens within a single `bp35`/`ka59` attempt
(frame-by-frame, informed by this session's now-solid world model) to
understand what those games actually require, rather than continuing to
treat them as generic exploration targets.

## Reproducing this experiment

```
python -m jepa.train_recurrent_predictor --epochs 30 \
  --exclude-games r11l,bp35,m0r0,tr87,ka59 --out checkpoints_recurrent_holdout

python scripts/run_scorecard.py --agent recurrentcuriosity --label rc_bp35_r1 --game bp35
# ... repeat x8 per game, per budget (RecurrentCuriosity.MAX_ACTIONS bump-and-revert
# for the 2500 condition, same pattern as hypothesis_agent.py's own probes)

python scripts/run_scorecard.py --agent recurrentsearch --label rs_bp35_r1 --game bp35
# ... repeat x8 per game, per budget (RecurrentSearch.MAX_ACTIONS bump-and-revert)

python scripts/diagnose_recurrent_residual_holdout.py
```
