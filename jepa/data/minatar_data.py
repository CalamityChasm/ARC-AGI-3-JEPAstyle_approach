"""MinAtar data source for Stage 4/Stage 6 dynamics pretraining -- the
first "more diverse pretraining data" attempt at the held-out-ARC-games
generalization gap documented in CLAUDE.md's "Stage 6 addendum" (7
independent conditioning-architecture interventions all failed to close
it; this is the first data-volume/diversity lever tried against that
specific gap, mirroring how MiniGrid pretraining -- not architecture
tuning -- was what actually fixed Stage 4's gate-collapse problem).

MinAtar (`pip install MinAtar`) is a clean-room, no-ROM reimplementation
of 5 classic Atari-style games (breakout, asterix, freeway, seaquest,
space_invaders) as small (10x10xC) *multi-channel binary grids*, not raw
RGB frames -- no copyrighted ROM data is involved anywhere in this
pipeline. Each channel is a boolean plane for one object type (e.g.
breakout has 'paddle'/'ball'/'trail'/'brick'; seaquest has 10 channels).
Action space is a constant 6 actions (no-op, left, up, right, down, fire)
across every game -- `env.num_actions() == 6` for all 5 games, well
within `jepa/models/predictor.py`'s `NUM_ACTIONS=8`, so (unlike Sokoban,
see CLAUDE.md's Stage 4 item 8 gotcha) no action-id remapping is needed
here; verified directly via `minatar.Environment(...).num_actions()`
before writing this module, not assumed.

Translation: each (10, 10, C) bool state is collapsed to a single (10, 10)
int grid (colors 0-15) by mapping channel index i -> color i+1 (0 reserved
for "no channel active at this cell"); where multiple channels are
simultaneously active at one cell (which does happen, e.g. a moving
object's current position overlapping its own 'trail' channel), the
highest-index channel wins by applying channels in ascending order and
overwriting. Max channel count across all 5 games is seaquest's 10 ->
color 10, comfortably under NUM_COLORS=16. This is the direct MinAtar
analogue of jepa/data/minigrid_data.py's object_idx->color mapping and
jepa/data/sokoban_data.py's room_state->color mapping -- same pattern,
new source.

Design choice, revised (2026-07-31, `stage6-minatar-pergame-id` branch):
**per-game ids, not one shared `game_id="minatar"`.** The original
design (see git history / CLAUDE.md's Stage 6 addendum for the full
reasoning) pooled all 5 sub-games under a single shared id, mirroring
MiniGrid's own choice, on the reasoning that the action *interface* is
byte-for-byte identical across all 5 games. That shared-id run was a
negative result (did not close the held-out-ARC-games generalization
gap, and was directionally worse than a MiniGrid-only baseline on both
the standard and held-out metrics -- see
`experiments/stage6_diverse_pretraining.md`). This module now tests
whether that pooling itself was a confound: MinAtar's 5 sub-games don't
share nearly as much *underlying mechanic* with each other as MiniGrid's
21 environments did (paddle-and-ball physics, lane-crossing timing,
submarine-survival, and shoot-em-up projectiles are mutually distinct
causal structures, not variations on one theme the way MiniGrid's
navigation tasks are) -- pooling them under one id may force the shared
encoder/predictor to fit several mutually-inconsistent action->effect
mappings at once, exactly the kind of confound Stage 1 originally
worried about (and found *not* to be the dominant issue) for the 25
ARC-3 games, but never tested for a *synthetic* pretraining source
before. Each game now gets its own id (`minatar_breakout`,
`minatar_asterix`, `minatar_freeway`, `minatar_seaquest`,
`minatar_space_invaders`) via `GAME_ID_PREFIX + game_name`. `GAME_ID`
(the old shared constant) is kept for backward compatibility with any
code that still imports it, but `generate_transitions` no longer uses it
by default.
"""

import random

import numpy as np

try:
    from minatar import Environment
except ImportError as e:
    raise ImportError(
        "MinAtar is not installed -- run `pip install MinAtar` "
        "(see CLAUDE.md's Stage 6 diverse-pretraining notes)"
    ) from e

from ..grid import NUM_COLORS

GAME_ID = "minatar"  # kept for backward compat; no longer used as the default per-transition id
GAME_ID_PREFIX = "minatar_"

DEFAULT_GAMES = ["breakout", "asterix", "freeway", "seaquest", "space_invaders"]

# Verified directly (not assumed) that every game's channel count stays
# well under NUM_COLORS when mapped as channel_idx + 1: breakout=4,
# asterix=4, freeway=7, seaquest=10, space_invaders=6.
_MAX_KNOWN_CHANNELS = 10
assert _MAX_KNOWN_CHANNELS < NUM_COLORS


def _translate_frame(env: "Environment") -> list:
    """env (a reset/stepped MinAtar Environment) -> `[grid]`, a one-element
    list wrapping an (H, W) int grid (colors 0-15) -- the same "single
    layer" convention `arc3_frame_to_tensor`/`patch_change_mask` expect
    (they index `frame[0]`), matching minigrid_data.py's/sokoban_data.py's
    own wrapping. `env.state()` is (H, W, C) bool; channel i -> color i+1,
    later (higher-index) channels win on any cell where more than one
    channel is simultaneously active."""
    state = env.state()  # (H, W, C) bool
    h, w, c = state.shape
    assert c <= _MAX_KNOWN_CHANNELS, (
        f"{env.game_name} has {c} channels, more than the {_MAX_KNOWN_CHANNELS} "
        f"verified at module-write time -- check NUM_COLORS still has headroom"
    )
    out = np.zeros((h, w), dtype=np.int64)
    for ch in range(c):
        out[state[:, :, ch]] = ch + 1
    return [out.tolist()]


def generate_transitions(
    games: list | None = None,
    episodes_per_game: int = 160,
    steps_per_episode: int = 80,
    seed: int = 0,
    per_game_ids: bool = True,
) -> list:
    """Random-policy rollouts across `games`, returned as
    `(frame_t, action_id, x, y, frame_t1, changed, game_id)` tuples -- the
    same shape `jepa/data/trajectories.py`'s `TransitionDataset` expects.
    `x, y` are always 0 (MinAtar has no coordinate-based action).

    `per_game_ids=True` (the new default, see module docstring): each
    sub-game gets its own `game_id` (`GAME_ID_PREFIX + game_name`, e.g.
    `"minatar_breakout"`) instead of one shared `"minatar"` id. Pass
    `per_game_ids=False` to reproduce the original shared-id behavior
    (kept for reproducing the earlier negative result if needed, not used
    by default anymore).
    `jepa/train_moe_predictor.py`'s game-vocabulary construction reads
    `game_id` generically off each transition tuple (`t[6]`), so no
    changes were needed there -- it automatically picks up 5 distinct
    entries instead of 1 when `per_game_ids=True`.

    `episodes_per_game=160` (vs. minigrid_data.py's `episodes_per_env=40`
    across 21 environments) is chosen so the *total* transition volume
    from this source (5 * 160 * 80 = 64,000) lands in the same order of
    magnitude as MiniGrid's own default (21 * 40 * 80 = 67,200) despite
    MinAtar having far fewer distinct sub-environments -- a controlled
    "similar data budget, different mechanics" comparison, not a
    data-volume confound.

    Episode length varies hugely by game under random play (directly
    measured: breakout terminates in ~11 steps on average under random
    actions, freeway effectively never terminates and instead hits
    MinAtar's own internal step cap around 2500) -- exactly like
    minigrid_data.py/sokoban_data.py, this just runs a fixed
    `steps_per_episode`-step budget per nominal "episode" and resets
    immediately whenever the env signals `terminal=True` partway through,
    rather than trying to align "episode" with any game's natural length.
    """
    games = games or DEFAULT_GAMES
    rng = random.Random(seed)
    transitions = []
    for game_name in games:
        game_id = f"{GAME_ID_PREFIX}{game_name}" if per_game_ids else GAME_ID
        env = Environment(game_name)
        for episode in range(episodes_per_game):
            ep_seed = seed * 1_000_000 + (hash(game_name) % 10_000) * 100 + episode
            env.seed(ep_seed)
            env.reset()
            frame = _translate_frame(env)
            for _step in range(steps_per_episode):
                action = rng.randrange(env.num_actions())
                _reward, terminal = env.act(action)
                next_frame = _translate_frame(env)
                changed = frame != next_frame
                transitions.append((frame, action, 0, 0, next_frame, changed, game_id))
                frame = next_frame
                if terminal:
                    env.seed(ep_seed + 500_000)
                    env.reset()
                    frame = _translate_frame(env)
    return transitions
