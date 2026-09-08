"""Shared fixtures/helpers for the CodeWorldAgent tests.

Two things need arranging before `code_world_agent.py` can be imported
outside the competition harness:

1. `llm_engine` lives inside the Kaggle dataset staging dir, not on the
   repo's own import path.
2. `code_world_agent.py` does `from ..agent import Agent` -- a relative
   import into the vendored `ARC-AGI-3-Agents/agents` package, which is
   not importable from the repo root (and pulls in the whole framework if
   it were). A minimal stand-in package is installed into `sys.modules`
   instead, so the agent under test is the real file, unmodified, with
   only its base class faked.

`arcengine` itself is NOT faked -- it is a real installed dependency, and
using the real `FrameData`/`GameAction`/`GameState` is the whole point of
a test that claims `choose_action` returns a legal action.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_STAGE = REPO_ROOT / "kaggle_submission_llm_world_engine" / "dataset_stage"

if str(DATASET_STAGE) not in sys.path:
    sys.path.insert(0, str(DATASET_STAGE))


def _install_fake_agents_package() -> None:
    """Minimal stand-in for the vendored `agents` package."""
    if "agents.agent" in sys.modules:
        return

    class _StubAgent:
        """Just enough of agents.agent.Agent for CodeWorldAgent.__init__."""

        MAX_ACTIONS = 80

        def __init__(self, *_args, game_id: str = "test-game", **_kwargs) -> None:
            self.game_id = game_id
            self.agent_name = "codeworldagent"
            self.frames = []

    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = []  # type: ignore[attr-defined]
    agent_mod = types.ModuleType("agents.agent")
    agent_mod.Agent = _StubAgent  # type: ignore[attr-defined]
    templates_pkg = types.ModuleType("agents.templates")
    templates_pkg.__path__ = []  # type: ignore[attr-defined]

    sys.modules["agents"] = agents_pkg
    sys.modules["agents.agent"] = agent_mod
    sys.modules["agents.templates"] = templates_pkg


def load_code_world_agent_module():
    """Import the real code_world_agent.py under a package name that makes
    its `from ..agent import Agent` resolve to the stub above."""
    _install_fake_agents_package()
    name = "agents.templates.code_world_agent"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DATASET_STAGE / "code_world_agent.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def code_world_agent_module():
    return load_code_world_agent_module()


# --- fakes -----------------------------------------------------------


class FakeLLMClient:
    """Returns canned responses in order; records every prompt it saw so a
    test can assert on what the model was actually asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.prompts.append((system, user))
        if not self.responses:
            return ""
        return self.responses.pop(0)


class RaisingLLMClient:
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        raise ConnectionError("no LLM server here")


# --- a small deterministic game to draft world models against ---------

GRID_N = 6


def blank_grid(n: int = GRID_N):
    return [[[0] * n for _ in range(n)]]


def with_agent_at(x: int, y: int, n: int = GRID_N):
    g = blank_grid(n)
    g[0][y][x] = 3
    return g


def find_agent(state):
    for y, row in enumerate(state[0]):
        for x, v in enumerate(row):
            if v == 3:
                return x, y
    return None


def true_step(state, action):
    """Ground truth: a single '3' pixel that moves with ACTION1-4 and
    teleports with ACTION6. ACTION5/ACTION7 do nothing."""
    n = len(state[0])
    px, py = find_agent(state)
    nx, ny = px, py
    if action.name == "ACTION1":
        ny = max(0, py - 1)
    elif action.name == "ACTION2":
        ny = min(n - 1, py + 1)
    elif action.name == "ACTION3":
        nx = max(0, px - 1)
    elif action.name == "ACTION4":
        nx = min(n - 1, px + 1)
    elif action.name == "ACTION6":
        nx, ny = action.x % n, action.y % n
    ns = [[row[:] for row in state[0]]]
    ns[0][py][px] = 0
    ns[0][ny][nx] = 3
    return ns


# Source the fake "coder LLM" hands back: a genuinely correct model for
# `true_step`. Deliberately uses `super()` and a dict comprehension --
# both of which the exec sandbox rejected before this branch's fix.
CORRECT_WORLD_MODEL_SOURCE = '''\
class WorldModel:
    def __init__(self):
        super().__init__()
        self.seen = {}

    def _find(self, state):
        for y in range(len(state[0])):
            for x in range(len(state[0][y])):
                if state[0][y][x] == 3:
                    return x, y
        return None

    def predict(self, state, action_name, x=None, y=None):
        n = len(state[0])
        found = self._find(state)
        if found is None:
            return state, 0, False
        px, py = found
        nx, ny = px, py
        if action_name == "ACTION1":
            ny = max(0, py - 1)
        elif action_name == "ACTION2":
            ny = min(n - 1, py + 1)
        elif action_name == "ACTION3":
            nx = max(0, px - 1)
        elif action_name == "ACTION4":
            nx = min(n - 1, px + 1)
        elif action_name == "ACTION6":
            nx, ny = x % n, y % n
        ns = [[row[:] for row in state[0]]]
        ns[0][py][px] = 0
        ns[0][ny][nx] = 3
        return ns, 0, False

    def goal_hint(self, state):
        found = self._find(state)
        if found is None:
            return 0.0
        return float(found[0] + found[1])
'''


def build_transcript(game_id: str = "test-game"):
    """A transcript shaped exactly like the agent's own opening probes."""
    from llm_engine.opening_probes import opening_probe_plan
    from llm_engine.types import Action, GameTranscript, Transition

    transcript = GameTranscript(game_id=game_id)
    state = with_agent_at(2, 2)
    for probe in opening_probe_plan():
        action = probe
        if probe.name == "ACTION6":
            action = Action(name="ACTION6", x=probe.x % GRID_N, y=probe.y % GRID_N)
        nxt = true_step(state, action)
        transcript.append(Transition(
            frame_before=state,
            action=action,
            frame_after=nxt,
            levels_completed_before=0,
            levels_completed_after=0,
            state_after="NOT_FINISHED",
        ))
        state = nxt
    return transcript
