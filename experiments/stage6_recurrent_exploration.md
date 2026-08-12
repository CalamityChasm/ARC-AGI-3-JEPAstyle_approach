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
