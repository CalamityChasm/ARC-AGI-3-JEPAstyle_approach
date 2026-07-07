from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent


class PressOnce(Agent):
    """Stage-0 scripted probe: RESET, then press every action exactly once.

    Simple actions (ACTION1-5, ACTION7) are each pressed once. ACTION6 (the
    coordinate action) is probed at a handful of spread-out grid points since
    its effect can depend on (x, y). Used to generate a first, non-random
    trajectory corpus and a baseline for what each button does.
    """

    MAX_ACTIONS = 20

    COMPLEX_PROBE_POINTS: list[tuple[int, int]] = [(0, 0), (31, 31), (63, 63)]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._plan: list[GameAction] = self._build_plan()
        self._step = 0

    def _build_plan(self) -> list[GameAction]:
        # The initial RESET is issued separately (see choose_action's
        # NOT_PLAYED/GAME_OVER branch) -- this plan only holds the probes
        # that follow it, so indexing into it stays aligned.
        plan: list[GameAction] = []
        for action in GameAction.all_simple():
            if action is GameAction.RESET:
                continue
            plan.append(action)
        for action in GameAction.all_complex():
            for _ in self.COMPLEX_PROBE_POINTS:
                plan.append(action)
        return plan

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(
            latest_frame.state is GameState.WIN or self._step >= len(self._plan)
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = "opening probe: reset before scripted sequence"
        else:
            action = self._plan[self._step]
            self._step += 1
            if action.is_complex():
                x, y = self.COMPLEX_PROBE_POINTS[
                    (self._step - 1) % len(self.COMPLEX_PROBE_POINTS)
                ]
                action.set_data({"x": x, "y": y})
                action.reasoning = f"opening probe: {action.name} at ({x},{y})"
            else:
                action.reasoning = f"opening probe: press {action.name} once"
        return action
