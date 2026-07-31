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

Design choice, one shared game_id: all 5 MinAtar games share a single
`game_id="minatar"` (mirroring MiniGrid's own choice and reasoning, see
minigrid_data.py's module docstring) -- deliberately *not* one game_id
per sub-game. Two reasons this is the right call here, more so than it
even was for MiniGrid: (1) the action *interface* is byte-for-byte
identical across all 5 games (same 6 actions, same meaning per action id
-- 'l'/'u'/'r'/'d'/'f'/'n' -- unlike MiniGrid where "forward" still means
different displacement depending on current facing), so a shared id lets
the model learn one consistent action vocabulary across genuinely
different game mechanics (paddle-and-ball vs. lane-crossing vs.
shoot-em-up) rather than routing around learning it via 5 separate
per-game embeddings; (2) per-game embeddings for only 5 sub-games would
give each one very little data to fit its own embedding well relative to
what the MiniGrid experiment already found sufficient (21 environments).
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

GAME_ID = "minatar"

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
) -> list:
    """Random-policy rollouts across `games`, returned as
    `(frame_t, action_id, x, y, frame_t1, changed, game_id)` tuples -- the
    same shape `jepa/data/trajectories.py`'s `TransitionDataset` expects.
    `x, y` are always 0 (MinAtar has no coordinate-based action); all
    transitions share `game_id="minatar"` (see module docstring for why).

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
                transitions.append((frame, action, 0, 0, next_frame, changed, GAME_ID))
                frame = next_frame
                if terminal:
                    env.seed(ep_seed + 500_000)
                    env.reset()
                    frame = _translate_frame(env)
    return transitions
