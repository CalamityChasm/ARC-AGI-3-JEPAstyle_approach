"""Sokoban data source for Stage 4's dynamics pretraining -- a second
synthetic environment family alongside MiniGrid (`jepa/data/minigrid_data.py`),
added specifically to grow the MoE experts' mechanic vocabulary rather than
to solve the 25 known ARC-3 games better.

MiniGrid pretraining (see CLAUDE.md's Stage 4 status) measurably improved
generalization and gate specialization over ARC-3-only training, but its
own mechanic vocabulary (navigation, key/door, obstacle avoidance,
pickup/carry) still doesn't cover "push a movable object with persistent
consequences" -- a distinct causal pattern architecture.md's own expert
list calls out and neither ARC-3's 25 games nor MiniGrid reliably exercise
in the same way Sokoban's push mechanic does. Each new synthetic
environment family is a shot at mechanics the existing sources don't
contain; Sokoban is the natural next one for that specific reason, not
because it's easier or more data-efficient than what's already in place.

Sokoban's action semantics are *not* the same as MiniGrid's (push/move on a
static overhead grid vs. turn-and-move-forward from a facing direction), so
this data source gets its own `game_id="sokoban"`, distinct from
`"minigrid"` -- consistent with minigrid_data.py's own reasoning for why
MiniGrid environments share *one* id across layouts: the point is
consistent action semantics *within* a source, but Sokoban's and MiniGrid's
semantics are consistent with each other.

Uses `gym_sokoban.envs.sokoban_env.SokobanEnv` directly (not `gym.make`),
sidestepping the legacy `gym` package's environment registry entirely --
gym-sokoban depends on classic `gym` (deprecated in favor of `gymnasium`,
which this project otherwise uses for MiniGrid), and its per-instance
`room_state`/`room_fixed` integer arrays are already exactly the flat
categorical grid representation this module needs, with no rendering
required.
"""

import random

import numpy as np

try:
    from gym_sokoban.envs.sokoban_env import SokobanEnv
except ImportError as e:
    raise ImportError(
        "gym-sokoban is not installed -- run `pip install gym-sokoban` "
        "(see CLAUDE.md's Stage 4 Sokoban pipeline notes)"
    ) from e

from ..grid import NUM_COLORS

GAME_ID = "sokoban"

# (dim_room, num_boxes, max_steps) configs spanning a range of sizes/box
# counts, mirroring minigrid_data.py's DEFAULT_ENV_NAMES size spread --
# gym-sokoban has no equivalent of MiniGrid's named registered environments,
# so these are parametrized directly against SokobanEnv.
DEFAULT_CONFIGS = [
    ((7, 7), 1, 60),
    ((7, 7), 2, 80),
    ((8, 8), 2, 80),
    ((8, 8), 3, 100),
    ((10, 10), 3, 120),
    ((10, 10), 4, 120),
    ((12, 12), 4, 150),
]

# room_state cell codes (see gym_sokoban.envs.room_utils.generate_room's own
# docstring): 0=wall, 1=empty floor, 2=box target (empty), 3=box not on
# target, 4=box on target, 5=player. Mapped 1:1 onto this project's flat
# 0-15 color palette -- the specific values don't need to align with
# minigrid_data.py's choices (the model conditions on game_id to
# disambiguate what a given color means per source), just be internally
# consistent and within NUM_COLORS.
_CELL_COLOR = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5}
# Player standing on an (otherwise-hidden) target cell is visually
# indistinguishable from a plain player cell in room_state alone (both read
# as 5) -- room_fixed still shows the target underneath, so give that case
# its own color the same way a door's state_idx gets its own color in
# minigrid_data.py, rather than silently losing that information.
_PLAYER_ON_TARGET_COLOR = 6

assert max(_CELL_COLOR.values()) < NUM_COLORS
assert _PLAYER_ON_TARGET_COLOR < NUM_COLORS


def _translate_frame(env: SokobanEnv) -> list:
    """env (a reset/stepped SokobanEnv) -> `[grid]`, a one-element list
    wrapping an (H, W) int grid (colors 0-15) -- the same "single layer"
    convention `arc3_frame_to_tensor`/`patch_change_mask` expect (they
    index `frame[0]`), matching minigrid_data.py's own wrapping."""
    state = env.room_state
    fixed = env.room_fixed
    out = np.zeros_like(state, dtype=np.int64)
    for code, color in _CELL_COLOR.items():
        out[state == code] = color
    player_on_target = (state == 5) & (fixed == 2)
    out[player_on_target] = _PLAYER_ON_TARGET_COLOR
    return [out.tolist()]


def generate_transitions(
    configs: list | None = None,
    episodes_per_config: int = 60,
    steps_per_episode: int = 80,
    seed: int = 0,
) -> list:
    """Random-policy rollouts across `configs`, returned as
    `(frame_t, action_id, x, y, frame_t1, changed, game_id)` tuples -- the
    same shape `jepa/data/trajectories.py`'s `TransitionDataset` expects.
    `x, y` are always 0 (Sokoban has no coordinate-based action); all
    transitions share `game_id="sokoban"` (see module docstring for why
    that's a deliberate, separate id from `"minigrid"`).

    gym-sokoban's room generator uses the global `random`/`numpy.random`
    state directly (no per-episode seed hook on `SokobanEnv.reset()`), so
    reproducibility here is at the run level (seeding once up front), not
    the per-episode level `generate_transitions` gets from MiniGrid's
    `env.reset(seed=...)` -- consistent with this project's existing
    tolerance for run-to-run variation in freshly-generated data (see
    CLAUDE.md's gotchas).

    Sokoban's own action space is 9 actions (0=no-op, 1-4=push directions,
    5-8=move directions) -- one more than `jepa/models/predictor.py`'s
    `NUM_ACTIONS=8`, sized for ARC-3's 8-action space (and already a fit
    for MiniGrid's 7). Storing a raw action id of 8 crashes the shared
    action-embedding lookup with an out-of-bounds CUDA assert. Fixed by
    dropping Sokoban's true no-op (action 0 -- redundant anyway, "nothing
    changes" is already the trivial baseline everywhere else in training)
    and remapping the remaining 8 actions (1-8) down to stored ids 0-7;
    `env.step()` itself is still called with the *original*, unshifted
    action id so the push/move semantics stay correct -- only the id
    written into the transition tuple is remapped.
    """
    configs = configs or DEFAULT_CONFIGS
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    transitions = []
    for dim_room, num_boxes, max_steps in configs:
        for _episode in range(episodes_per_config):
            env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)
            frame = _translate_frame(env)
            for _step in range(steps_per_episode):
                action = rng.randrange(1, env.action_space.n)  # skip true no-op (0)
                stored_action = action - 1  # remap 1-8 -> 0-7 to fit NUM_ACTIONS=8
                _obs, _reward, done, _info = env.step(action)
                next_frame = _translate_frame(env)
                changed = frame != next_frame
                transitions.append((frame, stored_action, 0, 0, next_frame, changed, GAME_ID))
                frame = next_frame
                if done:
                    env.reset()
                    frame = _translate_frame(env)
            env.close()
    return transitions
