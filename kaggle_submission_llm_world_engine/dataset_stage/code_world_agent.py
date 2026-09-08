"""LLM world-engine agent (plan.md Stages 1-3): draft an executable
per-game world model from an opening-probe transcript, validate it by
replaying the transcript through it, plan actions by searching over the
model (no LLM calls per step), and repair it online when a real
observation diverges from its prediction.

Two LLM roles, two models, both invoked rarely (see architecture.md's
"Two LLM roles" section):
  - "coder" (default: Qwen3-Coder) drafts/repairs WorldModel source --
    events, not every step.
  - "action_head" (default: Gemma) is a sparse fallback consulted only
    when the non-LLM planner stalls (llm_engine.planner.PlanResult.stalled)
    -- it suggests a single action using broader gameplay/visual judgment
    when search has nothing to go on, then control returns to search.

The moment-to-moment default is always the LLM-free beam search in
llm_engine/planner.py -- that's the efficiency lever, unchanged by adding
the action-head fallback.

See ../../../architecture.md and ../../../plan.md for the full design.
This file is intentionally thin -- it's glue between the competition's
Agent interface and the llm_engine/ package, which holds all the actual
logic and has no dependency on this framework.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_engine.action_head import suggest_action  # noqa: E402
from llm_engine.budget import LLMBudget  # noqa: E402
from llm_engine.drafting import draft_world_model, repair_world_model  # noqa: E402
from llm_engine.llm_client import make_client  # noqa: E402
from llm_engine.opening_probes import opening_probe_plan  # noqa: E402
from llm_engine.persistence import save_revision  # noqa: E402
from llm_engine.planner import next_action  # noqa: E402
from llm_engine.types import ALL_ACTIONS, Action as EngineAction, GameTranscript, Transition  # noqa: E402
from llm_engine.world_model import WorldModelProtocol, safe_predict  # noqa: E402

logger = logging.getLogger()


class CodeWorldAgent(Agent):
    MAX_ACTIONS = 200

    # Kept modest by default: the competition's Swarm runs many games
    # concurrently via Python threads (GIL-bound), so search cost here is
    # paid serially across all of them, not just within one game. Increase
    # if profiling on real hardware shows headroom.
    PLAN_DEPTH = 2
    PLAN_BEAM_WIDTH = 4
    DRAFT_MAX_ATTEMPTS = 5
    REPAIR_MAX_ATTEMPTS = 3

    # Separate per-role budgets: drafting/repair is the load-bearing use of
    # the LLM (worth more calls), the action-head is a sparse tie-breaker
    # for stalls only (worth far fewer -- if the planner is stalling this
    # often, the coder model needs a better WorldModel, not more guesses).
    CODER_LLM_CALL_BUDGET = 20
    ACTION_LLM_CALL_BUDGET = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.coder_client = make_client("coder")
        self.action_client = make_client("action_head")
        self.coder_budget = LLMBudget(max_calls_per_game=self.CODER_LLM_CALL_BUDGET)
        self.action_budget = LLMBudget(max_calls_per_game=self.ACTION_LLM_CALL_BUDGET)
        self.transcript = GameTranscript(game_id=self.game_id)
        self._probe_plan = opening_probe_plan()
        self._probe_index = 0

        self.model: Optional[WorldModelProtocol] = None
        self.model_source: Optional[str] = None
        self.model_version = 0

        self._pending_action: Optional[EngineAction] = None
        self._pending_frame_before = None
        self._pending_levels_before: int = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(latest_frame.state is GameState.WIN)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        current_grid = latest_frame.frame
        current_levels = latest_frame.levels_completed

        if self._pending_action is not None:
            t = Transition(
                frame_before=self._pending_frame_before,
                action=self._pending_action,
                frame_after=current_grid,
                levels_completed_before=self._pending_levels_before,
                levels_completed_after=current_levels,
                state_after=latest_frame.state.value,
            )
            self._pending_action = None
            self._handle_new_transition(t)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            action.reasoning = "reset"
            return action

        engine_action = self._choose_engine_action(current_grid, latest_frame.available_actions)
        self._pending_action = engine_action
        self._pending_frame_before = current_grid
        self._pending_levels_before = current_levels
        return self._to_game_action(engine_action)

    # -- internals ---------------------------------------------------

    def _choose_engine_action(self, current_grid: Any, available_actions: list[int]) -> EngineAction:
        if self._probe_index < len(self._probe_plan):
            action = self._probe_plan[self._probe_index]
            self._probe_index += 1
            return action

        if self.model is None:
            self._draft_initial_model()

        chosen: Optional[EngineAction] = None
        if self.model is not None:
            chosen, plan_result = next_action(self.model, current_grid, depth=self.PLAN_DEPTH, beam_width=self.PLAN_BEAM_WIDTH)
            if plan_result.stalled or chosen is None or not self._is_available(chosen, available_actions):
                fallback = self._try_action_head(current_grid, available_actions)
                if fallback is not None:
                    chosen = fallback

        if chosen is None or not self._is_available(chosen, available_actions):
            chosen = self._fallback_action(available_actions)
        return chosen

    def _try_action_head(self, current_grid: Any, available_actions: list[int]) -> Optional[EngineAction]:
        if not self.action_budget.has_budget():
            return None
        self.action_budget.record("stall-fallback")
        outcome = suggest_action(self.action_client, self.transcript, current_grid)
        if not outcome.ok or outcome.action is None:
            logger.info("%s: action_head did not produce a usable action", self.game_id)
            return None
        if not self._is_available(outcome.action, available_actions):
            logger.info("%s: action_head suggested an unavailable action (%s)", self.game_id, outcome.action)
            return None
        logger.info("%s: planner stalled, action_head suggested %s", self.game_id, outcome.action)
        return outcome.action

    def _is_available(self, action: EngineAction, available_actions: list[int]) -> bool:
        if not available_actions:
            return True  # framework not reporting restrictions -- assume all are legal
        return GameAction.from_name(action.name).value in available_actions

    def _fallback_action(self, available_actions: list[int]) -> EngineAction:
        names = ALL_ACTIONS
        if available_actions:
            names = [
                a for a in ALL_ACTIONS
                if GameAction.from_name(a).value in available_actions
            ] or ALL_ACTIONS
        name = random.choice(names)
        if name == "ACTION6":
            return EngineAction(name=name, x=random.randint(0, 63), y=random.randint(0, 63))
        return EngineAction(name=name)

    def _to_game_action(self, engine_action: EngineAction) -> GameAction:
        game_action = GameAction.from_name(engine_action.name)
        if engine_action.name == "ACTION6":
            game_action.set_data({"x": engine_action.x, "y": engine_action.y, "game_id": self.game_id})
        game_action.reasoning = str(engine_action)
        return game_action

    def _draft_initial_model(self) -> None:
        if not self.coder_budget.has_budget():
            logger.warning("%s: no coder LLM budget left, skipping initial draft", self.game_id)
            return
        self.coder_budget.record("draft")
        outcome = draft_world_model(self.coder_client, self.transcript, max_attempts=self.DRAFT_MAX_ATTEMPTS)
        self.model_version += 1
        self.model = outcome.world_model
        self.model_source = outcome.source
        if self.model_source:
            save_revision(self.game_id, self.model_version, self.model_source, note="draft" if outcome.ok else "draft-fallback-skeleton")
        logger.info(
            "%s: initial draft %s after %d attempt(s)",
            self.game_id, "passed replay" if outcome.ok else "FELL BACK to skeleton", outcome.attempts,
        )

    def _handle_new_transition(self, t: Transition) -> None:
        self.transcript.append(t)
        if self.model is None:
            return

        predicted_state, predicted_delta, predicted_done, error = safe_predict(self.model, t.frame_before, t.action)
        mismatch = (
            error is not None
            or predicted_state != t.frame_after
            or predicted_delta != t.levels_delta
            or predicted_done != t.done
        )
        if mismatch and self.coder_budget.has_budget() and self.model_source is not None:
            logger.info("%s: prediction diverged at transition #%d, repairing", self.game_id, len(self.transcript) - 1)
            self.coder_budget.record("repair")
            outcome = repair_world_model(self.coder_client, self.transcript, self.model_source, max_attempts=self.REPAIR_MAX_ATTEMPTS)
            if outcome.ok:
                self.model_version += 1
                self.model = outcome.world_model
                self.model_source = outcome.source
                save_revision(self.game_id, self.model_version, self.model_source, note="repair")
                logger.info("%s: repair succeeded, now v%d", self.game_id, self.model_version)
            else:
                logger.info("%s: repair failed, keeping previous model", self.game_id)
