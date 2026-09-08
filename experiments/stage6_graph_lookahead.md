# Stage 6: graph-based multi-hop lookahead

**Scope: production-scale checkpoints/agent only**, same as
`stage6_transition_graph_health.md`.

## Why

Following up on a design discussion about whether a learned action head or
model-based MCTS could beat the current hand-designed Q-scoring heuristic.
Both were judged likely net-negative for this project specifically: a
learned head would need more reward signal than the already-starved value
head has and would likely inherit the same held-out-game generalization
collapse everything else trained on local data has shown; multi-step
search through the *learned* predictor risks compounding rollout error --
literally the reason plan.md called for the exact `TransitionGraph` in
the first place ("shields you from compounding rollout error on seen
states"). The middle path that avoids both problems: extend the *exact*
graph itself to multi-hop lookahead, since every edge is a verified fact,
not a prediction stacked on a prediction.

## What changed

`jepa/memory.py: TransitionGraph`:
- Added an adjacency index (`_outgoing: state_key -> [(action_id, xy,
  next_state_key, delta), ...]`) maintained alongside the existing flat
  `_edges` dict, so per-state queries (`tried_actions`, the new lookahead)
  are O(edges from this state) instead of O(edges in the whole graph) --
  matters since this runs on every single decision.
- New `lookahead_best_path(state_key, max_depth=8)`: BFS over known edges
  from the current exact state, tracking cumulative
  `levels_completed_delta` along each path, returning the *first* action
  of whichever reachable (within `max_depth` hops) path shows the highest
  net progress -- or `None` if no known path from here ever shows
  progress. Strictly generalizes the old `best_known_action` single-hop
  check (BFS naturally finds 1-hop paths too, so nothing that worked
  before stops working).

`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`: the "recall a
known winning action" check now calls `lookahead_best_path` instead of
`best_known_action`, gated by a new `LOOKAHEAD_MAX_DEPTH = 8` class
attribute (same tunable-constant convention as `DEADEND_FILTER`/
`FORCE_BETA`).

## Verification

**Unit-level, on a hand-built 5-state graph** (A -> B -> C -> D, +1 level
at D, 3 hops from A; a separate 1-hop dead end from A): correctly found
the first action of the 3-hop path from A, correctly found the direct
1-hop path from C, correctly returned `None` for a never-seen state,
correctly respected `max_depth` (a depth-2 cap missed the 3-hop path),
and `tried_actions` still works correctly off the new adjacency index.
All passed before touching the live agent.

**Live smoke test** (`r11l`, `MAX_ACTIONS` temporarily bumped to 1500,
`DEBUG=True`): ran cleanly, 1 level completed across 30 resets, no
crash. The new lookahead log line never fired in this one run -- expected
given how rare a multi-hop productive path is to encounter by chance in
a single session.

**Real-data replay, the more informative check**: replayed the full
25-game, 49,448-decision winning-harvest corpus (`E:/jepa_overflow/
winning_harvest/recordings/`, the same dataset `stage6_transition_graph_
health.md` used) through both the old one-hop check and the new
multi-hop lookahead side by side. **Result: zero hits for both, and zero
new opportunities found by the multi-hop version.** Not a bug -- this
directly confirms the same root cause already diagnosed in
`stage6_transition_graph_health.md`: the limiting factor was never
lookahead *depth*, it's that win-adjacent states essentially never get
revisited at all within a budget-limited episode (only 4 level-up events
exist in this entire 49k-decision corpus, and the exact states leading up
to each one are, close to by definition, late-episode states a
budget-limited run reaches once if at all).

## Agent-level backtest: confirms the mechanical finding, not just consistent with it

Matched before/after comparison (`HYPOTHESIS_LOOKAHEAD_MAX_DEPTH=1` vs
`=8` -- mathematically, depth=1 reproduces the old one-hop-only
`best_known_action` check, same env-var-toggle pattern as
`DEADEND_FILTER`), 2 folds of 6 repeats each, 25 games,
`MAX_ACTIONS=300`, 300 runs per condition per fold.

**Fold 1: 2 -> 4 levels, 2 -> 4 distinct games** (a positive-looking
result on its own). **Fold 2: 6 -> 6 levels, flat.** Cumulative (n=12,
600 runs/condition): **8 -> 10 levels (+2), 3 -> 5 distinct games (+2)**.

**The decisive check: did the multi-hop lookahead ever actually fire
differently from the one-hop baseline in either fold's real gameplay?**
Replayed the full AFTER-condition dataset from both folds (90,000 real
decisions total) through the same replay methodology as
`stage6_transition_graph_health.md`, checking specifically for states
where the multi-hop search found something the one-hop check would have
missed. **Result: zero, across all 90,000 decisions.** The BEFORE and
AFTER conditions were mechanistically identical for this feature in
every single game played -- the depth=1 vs depth=8 toggle never once
produced a different candidate action anywhere in this backtest.

**This means the observed level-count deltas are not attributable to the
lookahead at all.** They're ordinary run-to-run variance from
`Hypothesis`'s unseeded RNG (`random.Random()` with no fixed seed,
already documented elsewhere in this project as a real source of score
swings on identical code) -- fold 1's apparent win and fold 2's flat
result are two draws from the same noise distribution, not a real effect
appearing and partially fading. This is a stronger, more direct
conclusion than "no effect detected due to insufficient power" (the
usual caveat on this project's small-sample agent backtests) -- here we
can show *why* no effect could possibly have shown up: the mechanism
being tested never activated.

## Honest verdict

The implementation is correct and strictly better than what it replaces
(same one-hop cases still caught, more general, indexed for performance,
architecturally free of compounding error unlike a model-based
alternative) -- but it doesn't yet show measurable value on the data
available, for the same reason nothing else touching the win-recall path
has: the underlying win signal is too sparse and too rarely revisited,
not the mechanism reading it. **Agent-level impact should be treated as
settled-negative under current exploration budgets, not merely
untested** -- the mechanical replay confirms the feature has essentially
zero opportunity to matter, not just that no effect was detected. Worth
keeping regardless (no downside, real generalization, free), but don't
expect it to move agent-level results until there's meaningfully more
real win data feeding the graph -- which points back at the same lever
this project has already identified: denser, more-directed data
harvesting (teacher policies, search-harvest), not another change to how
the graph is queried.
