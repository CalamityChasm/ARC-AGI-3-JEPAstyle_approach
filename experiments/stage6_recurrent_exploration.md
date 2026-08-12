# Stage 6: wiring Stage 3's recurrent core into live exploration for the first time

**Status: COMPLETE. Clean negative -- real accumulated episode memory, in
this design, does not unlock `bp35`/`ka59` either, at either budget
tested.**

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

## Reproducing this experiment

```
python -m jepa.train_recurrent_predictor --epochs 30 \
  --exclude-games r11l,bp35,m0r0,tr87,ka59 --out checkpoints_recurrent_holdout

python scripts/run_scorecard.py --agent recurrentcuriosity --label rc_bp35_r1 --game bp35
# ... repeat x8 per game, per budget (RecurrentCuriosity.MAX_ACTIONS bump-and-revert
# for the 2500 condition, same pattern as hypothesis_agent.py's own probes)
```
