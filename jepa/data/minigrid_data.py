"""MiniGrid data source for Stage 4's dynamics pretraining -- plan.md's
originally-specified fix for Stage 4's data scarcity (see CLAUDE.md's
Stage 4 status: the first MoE attempt, trained on ARC-3 data alone,
consistently collapsed to a uniform expert blend regardless of tuning).

MiniGrid environments have *consistent* action semantics across every
room layout/episode (action 2 always means "move forward", unlike ARC-3
where the same action id means something different in every one of the
25 games) and are cheap to generate in unlimited quantity via a random
policy -- exactly the transfer-learning data source plan.md's "Transfer-
learning curriculum" section calls for pretraining the dynamics model
before fine-tuning on the scarce ARC-3 data.

Translates each MiniGrid frame (`env.unwrapped.grid.encode()`, a
(W, H, 3) array of `[object_idx, color_idx, state_idx]` triples, overlaid
with the agent's own position/direction, which MiniGrid does *not*
include in `grid.encode()`) into the same flat single-channel 0-15 color
grid `jepa/grid.py`'s `grid_to_tensor` expects, so the identical encoder
architecture and training code work on both data sources unchanged.
"""

import random

import gymnasium as gym
import numpy as np

try:
    import minigrid  # noqa: F401  registers the MiniGrid-* environments with gymnasium
except ImportError as e:
    raise ImportError(
        "minigrid is not installed -- run `pip install minigrid` "
        "(see CLAUDE.md's Stage 4 MiniGrid pipeline notes)"
    ) from e

from ..grid import NUM_COLORS

GAME_ID = "minigrid"

# A deliberately diverse set spanning plan.md's target expert vocabulary
# (translate/movement, appear/disappear via pickup/drop, door open/close
# state changes, obstacle/hazard avoidance, multi-room navigation) -- not
# every registered MiniGrid environment (82 of them, many near-duplicate
# size variants of the same mechanic).
DEFAULT_ENV_NAMES = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-DoorKey-16x16-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N2-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-Fetch-5x5-N2-v0",
    "MiniGrid-Fetch-8x8-N3-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-RedBlueDoors-6x6-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",
    "MiniGrid-GoToDoor-8x8-v0",
]

# object_idx (see minigrid.core.constants.OBJECT_TO_IDX) -> base color code.
_OBJECT_COLOR = {
    0: 0,  # unseen -> treat as empty (shouldn't occur; we use the full
    1: 0,  # empty      grid.encode(), not the agent's partial observation)
    2: 1,  # wall
    3: 2,  # floor
    5: 6,  # key
    6: 7,  # ball
    7: 8,  # box
    8: 9,  # goal
    9: 10,  # lava
}
# door (object_idx 4): color depends on state_idx (0=open, 1=closed, 2=locked)
_DOOR_COLOR = {0: 3, 1: 4, 2: 5}
# agent overlay, keyed by agent_dir (0=right, 1=down, 2=left, 3=up) --
# encodes facing direction, which matters for what "forward" will do next.
_AGENT_COLOR = {0: 12, 1: 13, 2: 14, 3: 15}

assert max(_OBJECT_COLOR.values()) < NUM_COLORS
assert max(_DOOR_COLOR.values()) < NUM_COLORS
assert max(_AGENT_COLOR.values()) < NUM_COLORS


def _translate_frame(env) -> list:
    """env (a reset/stepped MiniGrid gym env) -> `[grid]`, a one-element
    list wrapping an (H, W) int grid (colors 0-15) -- the same "single
    layer" convention `arc3_frame_to_tensor`/`patch_change_mask` expect
    (they index `frame[0]`), matching ARC-3's own FrameData.frame shape
    and external_logs.py's `[grid]` wrapping."""
    unwrapped = env.unwrapped
    encoded = unwrapped.grid.encode()  # (W, H, 3): [object_idx, color_idx, state_idx]
    w, h, _ = encoded.shape
    out = np.zeros((h, w), dtype=np.int64)
    for x in range(w):
        for y in range(h):
            obj_idx, _color_idx, state_idx = encoded[x, y]
            if obj_idx == 4:  # door
                out[y, x] = _DOOR_COLOR[int(state_idx)]
            else:
                out[y, x] = _OBJECT_COLOR.get(int(obj_idx), 0)
    ax, ay = unwrapped.agent_pos
    out[ay, ax] = _AGENT_COLOR[int(unwrapped.agent_dir)]
    return [out.tolist()]


def generate_transitions(
    env_names: list | None = None,
    episodes_per_env: int = 40,
    steps_per_episode: int = 80,
    seed: int = 0,
) -> list:
    """Random-policy rollouts across `env_names`, returned as
    `(frame_t, action_id, x, y, frame_t1, changed, game_id)` tuples -- the
    same shape `jepa/data/trajectories.py`'s `TransitionDataset` expects.
    `x, y` are always 0 (MiniGrid has no coordinate-based action); all
    transitions share `game_id="minigrid"` -- deliberately *not* one
    game_id per environment, since the whole point of this data source is
    action semantics that are consistent across layouts, and a per-env
    embedding would let the model route around learning that instead of
    through it.
    """
    env_names = env_names or DEFAULT_ENV_NAMES
    rng = random.Random(seed)
    transitions = []
    for env_name in env_names:
        env = gym.make(env_name)
        for episode in range(episodes_per_env):
            ep_seed = seed * 1_000_000 + (hash(env_name) % 10_000) * 100 + episode
            env.reset(seed=ep_seed)
            frame = _translate_frame(env)
            for _step in range(steps_per_episode):
                action = rng.randrange(env.action_space.n)
                _obs, _reward, terminated, truncated, _info = env.step(action)
                next_frame = _translate_frame(env)
                changed = frame != next_frame
                transitions.append((frame, action, 0, 0, next_frame, changed, GAME_ID))
                frame = next_frame
                if terminated or truncated:
                    env.reset(seed=ep_seed + 500_000)
                    frame = _translate_frame(env)
        env.close()
    return transitions
