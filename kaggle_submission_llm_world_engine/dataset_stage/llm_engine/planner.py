"""Search over a validated WorldModel to choose actions, without calling
the LLM. See architecture.md step 4: "Plan without the LLM" -- this is the
efficiency lever that keeps LLM calls rare while still making a real
decision every step.

Uses beam search: cheap, bounded, and degrades gracefully (fewer/shorter
candidate sequences) rather than blowing up when goal_hint is uninformative
-- a real risk early in a game, before the model has learned much. When
that degradation is total -- no candidate action distinguishes itself at
all -- `plan()` reports `stalled=True` so the caller can fall back to the
sparse action-head LLM (see llm_engine/action_head.py) instead of guessing
via search alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import ALL_ACTIONS, Action, Grid
from .world_model import WorldModelProtocol, safe_goal_hint, safe_predict

# Coordinates to try for ACTION6 -- a coarse grid rather than all 4096
# (x, y) pairs, matching the opening-probe's spread-sample approach. Kept
# small because each candidate multiplies the branching factor, and because
# the competition's own Swarm runs many games concurrently via Python
# threads (GIL-bound) -- a wide beam search here contends for the same
# core across every concurrent game, so this stays intentionally cheap
# rather than exhaustive.
_ACTION6_SAMPLE_POINTS = [(16, 16), (48, 16), (16, 48), (48, 48)]

# If the best first-ply goal_hint and the worst differ by less than this,
# and no rollout predicts winning a level, the search isn't actually
# distinguishing between actions -- treat that as a stall rather than
# confidently acting on noise.
STALL_GOAL_HINT_EPSILON = 0.05


def _candidate_actions() -> list[Action]:
    actions = [Action(name=a) for a in ALL_ACTIONS if a != "ACTION6"]
    actions += [Action(name="ACTION6", x=x, y=y) for x, y in _ACTION6_SAMPLE_POINTS]
    return actions


@dataclass
class PlanResult:
    actions: list[Action]
    predicted_levels_gained: int
    predicted_final_goal_hint: float
    stalled: bool
    goal_hint_spread: float


def plan(
    model: WorldModelProtocol,
    state: Grid,
    depth: int = 2,
    beam_width: int = 4,
    stall_epsilon: float = STALL_GOAL_HINT_EPSILON,
) -> PlanResult:
    """Beam search over predicted futures. Score = (levels gained so far
    in this rollout, goal_hint of the resulting state) -- levels gained
    dominates (it's the real signal), goal_hint only breaks ties among
    rollouts that haven't won anything yet.
    """
    candidates = _candidate_actions()

    # Each beam entry: (score_tuple, action_sequence, resulting_state, cumulative_levels)
    beam: list[tuple[tuple[int, float], list[Action], Grid, int]] = [
        ((0, safe_goal_hint(model, state)), [], state, 0)
    ]

    first_ply_hints: list[float] = []
    first_ply_any_levels = False

    for depth_idx in range(depth):
        expanded: list[tuple[tuple[int, float], list[Action], Grid, int]] = []
        for _score, seq, cur_state, cum_levels in beam:
            if cum_levels > 0:
                # already found a rollout that predicts winning a level --
                # no need to keep expanding it further this planning call.
                expanded.append((_score, seq, cur_state, cum_levels))
                continue
            for action in candidates:
                next_state, levels_delta, done, error = safe_predict(model, cur_state, action)
                if error is not None or next_state is None:
                    continue
                new_cum = cum_levels + (levels_delta or 0)
                hint = safe_goal_hint(model, next_state)
                if depth_idx == 0:
                    first_ply_hints.append(hint)
                    if new_cum > 0:
                        first_ply_any_levels = True
                expanded.append(((new_cum, hint), seq + [action], next_state, new_cum))
                if done:
                    break  # don't bother exploring past a predicted terminal state
        if not expanded:
            break
        expanded.sort(key=lambda e: e[0], reverse=True)
        beam = expanded[:beam_width]

    goal_hint_spread = (max(first_ply_hints) - min(first_ply_hints)) if first_ply_hints else 0.0

    if not beam:
        # Total planning failure (model errors on everything) -- caller
        # should fall back to the action-head LLM or a scripted/random
        # action.
        return PlanResult(
            actions=[], predicted_levels_gained=0, predicted_final_goal_hint=0.0,
            stalled=True, goal_hint_spread=goal_hint_spread,
        )

    best_score, best_seq, _, best_levels = beam[0]
    stalled = best_levels == 0 and not first_ply_any_levels and goal_hint_spread < stall_epsilon

    if not best_seq:
        return PlanResult(
            actions=[], predicted_levels_gained=0, predicted_final_goal_hint=best_score[1],
            stalled=stalled, goal_hint_spread=goal_hint_spread,
        )
    return PlanResult(
        actions=best_seq, predicted_levels_gained=best_levels, predicted_final_goal_hint=best_score[1],
        stalled=stalled, goal_hint_spread=goal_hint_spread,
    )


def next_action(
    model: WorldModelProtocol, state: Grid, depth: int = 2, beam_width: int = 4
) -> tuple[Optional[Action], PlanResult]:
    """Convenience wrapper: plan, then return the first action of the best
    sequence (re-planning every step against the latest real state is
    cheap since this never calls the LLM) alongside the full PlanResult so
    the caller can inspect `stalled` and decide whether to consult the
    action-head fallback."""
    result = plan(model, state, depth=depth, beam_width=beam_width)
    action = result.actions[0] if result.actions else None
    return action, result
