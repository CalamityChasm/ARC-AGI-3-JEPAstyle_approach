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

    # Once a draft attempt fails, don't immediately retry on the very next
    # step: without a cooldown the agent re-drafts every single turn and
    # burns the whole coder budget on the same transcript within seconds.
    # Wait for this many *new* observed transitions -- new evidence is the
    # only thing that makes a retry worth paying for.
    REDRAFT_AFTER_NEW_TRANSITIONS = 8

    # After this many repair rounds in a row fail, stop patching and
    # re-draft from scratch -- see _handle_new_transition.
    MAX_CONSECUTIVE_REPAIR_FAILURES = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Set before anything that can fail, so _safe_fallback_action is
        # usable even when the rest of construction throws.
        self._rng = random.Random()
        self._init_failed = False

        self.transcript = GameTranscript(game_id=self.game_id)
        self._probe_plan = opening_probe_plan()
        self._probe_index = 0

        self.model: Optional[WorldModelProtocol] = None
        self.model_source: Optional[str] = None
        self.model_version = 0

        self._pending_action: Optional[EngineAction] = None
        self._pending_frame_before = None
        self._pending_levels_before: int = 0
        self._last_draft_attempt_len = -1
        self._consecutive_repair_failures = 0

        self.coder_budget = LLMBudget(max_calls_per_game=self.CODER_LLM_CALL_BUDGET)
        self.action_budget = LLMBudget(max_calls_per_game=self.ACTION_LLM_CALL_BUDGET)
        self.coder_client = None
        self.action_client = None
        try:
            # make_client can raise outright: a misconfigured backend
            # (LLM_BACKEND=transformers with no *_MODEL_DIR set) raises
            # RuntimeError, and the transformers path loads a multi-GB
            # model here, which can fail on dtype/VRAM/driver grounds. Any
            # of those happens during Agent *construction*, before
            # choose_action's own try/except can ever run -- uncaught, it
            # takes down the whole scored run rather than one game.
            self.coder_client = make_client("coder")
            self.action_client = make_client("action_head")
        except Exception:
            logger.exception(
                "%s: code_world agent: LLM client init failed, falling back to "
                "random legal actions for this game",
                self.game_id,
            )
            self._init_failed = True

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        try:
            return bool(latest_frame.state is GameState.WIN)
        except Exception:
            logger.exception("%s: code_world agent: is_done raised, treating as not-done", self.game_id)
            return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Top-level catch-all, mirroring hypothesis_agent.py's heartbeat
        # pattern. Anything raised below propagates out of main.py's agent
        # loop and kills this game's thread outright -- which is exactly
        # what happened on 2026-09-07, when a malformed grid returned by
        # LLM-authored predict() reached a diagnostic formatter and raised
        # TypeError. A safe random legal action keeps the game playing and
        # scoring instead of ending it.
        try:
            if self._init_failed:
                return self._safe_fallback_action(latest_frame)
            return self._choose_action_inner(frames, latest_frame)
        except Exception:
            logger.exception(
                "%s: code_world agent: choose_action raised, falling back to a safe "
                "random action", self.game_id,
            )
            return self._safe_fallback_action(latest_frame)

    def _safe_fallback_action(self, latest_frame: FrameData) -> GameAction:
        """A legal action chosen without touching any of this agent's own
        machinery -- deliberately depends on nothing but `self._rng`, so it
        stays usable when everything else is broken."""
        try:
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                action = GameAction.RESET
                action.reasoning = "code_world: reset (fallback)"
                return action
            available = list(latest_frame.available_actions or [])
        except Exception:
            available = []
        if not available:
            available = [a.value for a in GameAction if a is not GameAction.RESET]
        action = GameAction.from_id(self._rng.choice(available))
        if action.is_complex():
            action.set_data({
                "x": self._rng.randrange(64),
                "y": self._rng.randrange(64),
                "game_id": self.game_id,
            })
        action.reasoning = "code_world: safe fallback after internal error"
        return action

    def _choose_action_inner(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        current_grid = latest_frame.frame
        current_levels = latest_frame.levels_completed

        if self._pending_action is not None:
            pending_action = self._pending_action
            frame_before = self._pending_frame_before
            self._pending_action = None
            # An empty frame teaches the world model nothing and cannot be
            # replayed against, so it is dropped rather than recorded --
            # FrameData.frame defaults to [] and is empty around
            # NOT_PLAYED/reset boundaries.
            if frame_before and current_grid:
                self._handle_new_transition(Transition(
                    frame_before=frame_before,
                    action=pending_action,
                    frame_after=current_grid,
                    levels_completed_before=self._pending_levels_before,
                    levels_completed_after=current_levels,
                    state_after=latest_frame.state.value,
                ))

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
            self._maybe_draft_model()

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
        if self.action_client is None or not self.action_budget.has_budget():
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

    def _maybe_draft_model(self) -> None:
        """Draft a world model, and install it **only if it passed replay**.

        The original `_draft_initial_model` installed whatever
        `draft_world_model` returned, which on failure was a loaded copy
        of WORLD_MODEL_SKELETON -- a `predict()` that returns the input
        unchanged and a `goal_hint()` that returns 0.0. That is worse than
        having no model at all on three counts, and it is the direct cause
        of the 0.00 scored run (see experiments/stage7_codeworld_fixes.md):

          1. the beam search plans against a completely flat objective,
             so every candidate action scores identically;
          2. `self.model is not None` afterwards, so no further draft is
             ever attempted, however much new evidence arrives;
          3. every subsequent transition diverges from 'nothing changes',
             so the whole remaining coder budget is spent repairing a stub
             that cannot pass replay by construction -- the repeated
             'repair failed after 3 attempts' in the log.

        Now: no model is installed unless it reproduces the transcript,
        and a failed attempt leaves `self.model is None` so drafting can be
        retried once enough new evidence has accumulated.
        """
        if self.coder_client is None or not self.coder_budget.has_budget():
            return
        # Re-drafting on identical evidence just repeats the same failure.
        if len(self.transcript) - self._last_draft_attempt_len < self.REDRAFT_AFTER_NEW_TRANSITIONS:
            return
        self._last_draft_attempt_len = len(self.transcript)

        self.coder_budget.record("draft")
        outcome = draft_world_model(self.coder_client, self.transcript, max_attempts=self.DRAFT_MAX_ATTEMPTS)

        if outcome.ok and outcome.world_model is not None and outcome.source:
            self.model_version += 1
            self.model = outcome.world_model
            self.model_source = outcome.source
            save_revision(self.game_id, self.model_version, self.model_source, note="draft")
            logger.info(
                "%s: draft passed replay after %d attempt(s), now v%d",
                self.game_id, outcome.attempts, self.model_version,
            )
            return

        # Keep the diagnostic trail the on-disk revisions were valuable
        # for, without letting an unvalidated candidate near the planner.
        if outcome.last_candidate_source:
            save_revision(
                self.game_id, self.model_version, outcome.last_candidate_source,
                note="rejected-candidate-not-installed",
            )
        logger.warning(
            "%s: draft FAILED after %d attempt(s); playing without a world model "
            "(random legal actions) until enough new evidence to retry",
            self.game_id, outcome.attempts,
        )

    def _handle_new_transition(self, t: Transition) -> None:
        self.transcript.append(t)
        if self.model is None or self.model_source is None:
            return

        predicted_state, predicted_delta, predicted_done, error = safe_predict(self.model, t.frame_before, t.action)
        mismatch = (
            error is not None
            or predicted_state != t.frame_after
            or predicted_delta != t.levels_delta
            or predicted_done != t.done
        )
        if not mismatch:
            self._consecutive_repair_failures = 0
            return

        if self.coder_client is None or not self.coder_budget.has_budget():
            return

        logger.info("%s: prediction diverged at transition #%d, repairing", self.game_id, len(self.transcript) - 1)
        self.coder_budget.record("repair")
        outcome = repair_world_model(self.coder_client, self.transcript, self.model_source, max_attempts=self.REPAIR_MAX_ATTEMPTS)
        if outcome.ok and outcome.world_model is not None and outcome.source:
            self._consecutive_repair_failures = 0
            self.model_version += 1
            self.model = outcome.world_model
            self.model_source = outcome.source
            save_revision(self.game_id, self.model_version, self.model_source, note="repair")
            logger.info("%s: repair succeeded, now v%d", self.game_id, self.model_version)
            return

        # Repeated repair failure means this model is not a near-miss that
        # one more patch will fix -- it is wrong about something
        # structural. Retrying it forever is what produced the scored
        # run's repeated 'repair failed after 3 attempts' while the budget
        # drained. Drop it instead and let drafting start over from a much
        # longer transcript than the opening probes alone provided.
        self._consecutive_repair_failures += 1
        if self._consecutive_repair_failures >= self.MAX_CONSECUTIVE_REPAIR_FAILURES:
            logger.warning(
                "%s: %d consecutive repair failures, discarding model v%d and "
                "re-drafting from the full transcript",
                self.game_id, self._consecutive_repair_failures, self.model_version,
            )
            self.model = None
            self.model_source = None
            self._consecutive_repair_failures = 0
            self._last_draft_attempt_len = -1  # allow an immediate re-draft
        else:
            logger.info("%s: repair failed, keeping previous model", self.game_id)
