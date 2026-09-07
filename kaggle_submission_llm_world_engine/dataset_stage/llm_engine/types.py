"""Shared data types for the LLM world-engine, independent of the arcengine
package's own pydantic models so that world-model code (LLM-authored, run
via exec()) only ever touches plain Python -- lists, tuples, ints -- never
the framework's classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# A frame is a list of one or more 64x64 grids of integers 0-15. In
# practice almost every public game uses a single grid, but the API allows
# more than one, so this stays a list throughout instead of assuming len==1.
Grid = list[list[list[int]]]

SIMPLE_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]
COMPLEX_ACTIONS = ["ACTION6"]
ALL_ACTIONS = SIMPLE_ACTIONS + COMPLEX_ACTIONS


@dataclass(frozen=True)
class Action:
    """A game action, decoupled from arcengine.GameAction so world-model
    code and the planner can construct/compare actions without importing
    the framework."""

    name: str  # one of ALL_ACTIONS
    x: Optional[int] = None  # only set for ACTION6
    y: Optional[int] = None

    def __post_init__(self) -> None:
        if self.name not in ALL_ACTIONS:
            raise ValueError(f"Unknown action name: {self.name}")
        if self.name == "ACTION6" and (self.x is None or self.y is None):
            raise ValueError("ACTION6 requires x and y")

    def __str__(self) -> str:
        if self.name == "ACTION6":
            return f"ACTION6({self.x},{self.y})"
        return self.name


@dataclass
class Transition:
    """One observed (state, action, next_state) step, plus the score/done
    signals the world model is asked to predict."""

    frame_before: Grid
    action: Action
    frame_after: Grid
    levels_completed_before: int
    levels_completed_after: int
    state_after: str  # "NOT_FINISHED" | "WIN" | "GAME_OVER"

    @property
    def levels_delta(self) -> int:
        return self.levels_completed_after - self.levels_completed_before

    @property
    def done(self) -> bool:
        return self.state_after in ("WIN", "GAME_OVER")


@dataclass
class GameTranscript:
    """The full observed history for one game instance, in order. This is
    what gets serialized into prompts and replayed against candidate
    WorldModel code."""

    game_id: str
    transitions: list[Transition] = field(default_factory=list)

    def append(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)
