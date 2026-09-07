"""Opening-probe action plan: press every simple action once, sample
ACTION6 at a few spread points. Same idea as the sibling JEPA project's
PressOnce agent (architecture.md step 1) -- reimplemented fresh here so
this project has no file-level dependency on that repo.
"""

from __future__ import annotations

from .types import SIMPLE_ACTIONS, Action

# Spread sample rather than exhaustive 64x64 -- consistent with the
# planner's own ACTION6 sampling in planner.py.
PROBE_POINTS = [(0, 0), (31, 31), (63, 63)]


def opening_probe_plan() -> list[Action]:
    plan = [Action(name=a) for a in SIMPLE_ACTIONS]
    plan += [Action(name="ACTION6", x=x, y=y) for x, y in PROBE_POINTS]
    return plan
