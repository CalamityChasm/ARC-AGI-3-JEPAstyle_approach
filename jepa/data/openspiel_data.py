"""OpenSpiel-backed board/strategy-game data source for Stage 6's scaled
world-model pretraining (see experiments/stage6_scaled_world_model.md).

Every synthetic data source built before this one (MiniGrid, Sokoban,
MinAtar, Procgen -- see CLAUDE.md's Stage 4/Stage 6 history) was a
real-time/navigation/arcade-flavored grid game. This module adds a
genuinely different family: turn-based board/strategy games with
placement, capture, sowing, and push-your-luck-chance mechanics, none of
which the existing sources exercise. `pip install open_spiel` produced a
prebuilt `win_amd64` wheel on this Windows box with zero build-from-source
friction (checked first, per this branch's own scoping instructions,
before designing the rest of this module around it) -- no python-chess /
hand-rolled fallback was needed.

Six games, six distinct `game_id`s (CLAUDE.md's established "one id per
mechanically-dissimilar game" lesson from the MinAtar per-game-id retry --
`stage6-minatar-pergame-id` found pooling mechanically-dissimilar games
under one shared id measurably hurts; applied from the start here instead
of re-discovered):

  connect_four  -- gravity placement, direct column action (7 actions)
  tic_tac_toe   -- simple placement, single click (cell coordinate)
  othello       -- placement + flip-capture, single click
  checkers      -- move + jump-capture, single click (destination cell)
  pig           -- push-your-luck dice/chance, direct action (roll/hold)
  mancala       -- seed-sowing/distribution, direct action (pit index)

Action-space budget (`jepa/models/predictor.py: NUM_ACTIONS=8`, shared
across every data source's action embedding -- see CLAUDE.md's Sokoban
gotcha for what happens if this isn't respected): every game here uses
either a small direct action id (connect_four's 7 columns, pig's 2,
mancala's 6 relative-pit-index) or a single fixed `action_id=6`
("click", matching ARC-3's own ACTION6 id -- not load-bearing, just a
readable convention) with the real choice encoded in `(x, y)` instead --
never both a large direct id space *and* xy. `_sanity_check` at the
bottom of this module asserts `max(action_id) < NUM_ACTIONS` for every
generated corpus before it's ever handed to a training run.

**Design choice: single click per move, not the two-step
select-then-destination pattern floated as an option for board games with
`from -> to` moves (checkers here).** OpenSpiel's checkers exposes moves
(including multi-jump capture chains) as a single atomic
`state.apply_action(...)` -- there is no sub-move granularity to actually
step through, so a "select" transition would have to be a fabricated,
non-environment-driven frame-unchanged transition. This project has
consistently avoided injecting synthetic non-environment transitions
into training corpora, so checkers instead gets one real transition per
move, clicked at the *destination* cell -- the model still gets a
concrete, real xy-conditioned transition tied to a real board change,
just without a synthetic intermediate step. tic_tac_toe/othello use the
same one-click-per-move pattern for consistency (they only need one
click anyway -- placement games don't have a natural "select" step).

**Board rendering**: for the four spatial games (connect_four,
tic_tac_toe, othello, checkers), OpenSpiel's `observation_tensor()` is
confirmed (by direct inspection against each game's own `board_string`
output) to be a one-hot-per-cell stack of `(C, H, W)` planes -- `argmax`
over the channel axis reconstructs the board as a small `(H, W)` int
grid with values in `[0, C)`, well under `NUM_COLORS=16`, with zero
per-game-specific parsing needed. `pig` and `mancala` have no natural
spatial board (`pig`'s own state is two running scores + a turn total;
`mancala`'s `observation_tensor` is a flat 16-value pit-count vector) --
both get a small custom, honestly-synthetic grid rendering (see
`_render_pig`/`_render_mancala` below), documented as such rather than
pretending they're "real" spatial boards.

Boards here are all small enough (<=8x8, or the custom renderings' own
small fixed sizes) to sit top-left of the CANVAS-sized grid uncropped --
the same convention `minigrid_data.py`/`sokoban_data.py` already rely on
(`jepa/grid.py: grid_to_tensor`/`patch_change_mask` both place
smaller-than-canvas grids top-left automatically).
"""

import random
import re

import numpy as np

try:
    import pyspiel
except ImportError as e:
    raise ImportError(
        "open_spiel is not installed -- run `pip install open_spiel` "
        "(see experiments/stage6_scaled_world_model.md: installed cleanly "
        "via a prebuilt win_amd64 wheel on this project's dev box)"
    ) from e

from ..grid import NUM_COLORS
from ..models.predictor import NUM_ACTIONS

CLICK_ACTION_ID = 6  # matches ARC-3's own ACTION6 id; not load-bearing, just readable

GAME_IDS = ["connect_four", "tic_tac_toe", "othello", "checkers", "pig", "mancala"]


def _advance_through_chance(state, rng: random.Random) -> None:
    """Auto-resolve any chance node(s) immediately following a player's
    action (or at game start) -- chance outcomes aren't controllable by
    any action_id, so they're applied transparently rather than being
    surfaced as a labeled "action" the predictor would condition on. A
    no-op for every game in this module except `pig` (dice rolls)."""
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        a = rng.choices(actions, weights=probs)[0]
        state.apply_action(a)


def _render_spatial(state, shape: tuple) -> list:
    """(C, H, W) one-hot observation_tensor -> (H, W) int grid via
    per-cell argmax (see module docstring: verified against each game's
    own board_string before relying on this)."""
    # Explicit player=0 (not the no-arg form, which defaults to
    # state.current_player() internally and raises at terminal states,
    # where current_player() is a sentinel like kTerminalPlayerId=-4).
    # These are all full-information games (no hidden state per player),
    # so which player's "perspective" is requested doesn't change what's
    # observable on the board.
    obs = np.array(state.observation_tensor(0), dtype=np.float32).reshape(shape)
    board = np.argmax(obs, axis=0).astype(np.int64)
    assert board.max() < NUM_COLORS, f"board value {board.max()} exceeds NUM_COLORS={NUM_COLORS}"
    return board.tolist()


_SCORE_RE = re.compile(r"Scores: (\d+) (\d+), Turn total: (\d+)")


def _render_pig(state) -> list:
    """No real spatial board -- pig's entire state is two running scores
    plus the current turn's not-yet-banked total. Rendered as a small,
    honestly-synthetic 4-row scoreboard grid (documented in the module
    docstring, not presented as a "real" board): row 0 = player 0's score
    bar, row 1 = player 1's score bar, row 2 = current turn-total bar,
    row 3 = whose-turn indicator. Winning score is 100 (pyspiel's
    default) -- bars scaled to a 20-cell width."""
    m = _SCORE_RE.search(str(state))
    p0, p1, turn_total = (int(m.group(i)) for i in (1, 2, 3)) if m else (0, 0, 0)
    width = 20
    grid = np.zeros((4, width), dtype=np.int64)
    grid[0, : min(width, round(p0 / 100 * width))] = 1
    grid[1, : min(width, round(p1 / 100 * width))] = 2
    grid[2, : min(width, round(turn_total / 30 * width))] = 3
    cur = state.current_player() if not state.is_terminal() else 0
    grid[3, :] = 4 if cur == 0 else 5
    return grid.tolist()


def _render_mancala(state) -> list:
    """No natural single spatial layout matching the model's flat
    per-cell-color convention -- mancala's own observation_tensor is
    already a flat 16-value pit/store seed-count vector (6 pits + 1
    store per side), reshaped here into a 2x8 grid (row 0 = player 0's
    side, row 1 = player 1's side) mirroring the game's own visual
    layout. Seed counts are clipped to NUM_COLORS-1 (occasionally lossy
    for a pit that's accumulated many seeds -- a documented
    simplification, not a bug)."""
    obs = np.array(state.observation_tensor(0), dtype=np.int64)
    obs = np.clip(obs, 0, NUM_COLORS - 1)
    padded = np.zeros(16, dtype=np.int64)
    padded[: len(obs)] = obs[:16]
    return padded.reshape(2, 8).tolist()


_MOVE_RE = re.compile(r"([a-h])(\d)")


def _checkers_click_xy(state, action: int) -> tuple:
    """Parse action_to_string's algebraic-notation move (e.g. "a3b4", or
    a longer multi-jump chain like "a3b4c5") into the *destination*
    cell's (x, y) -- the last (col, row) pair in the string, which is
    where the moving piece ends up regardless of how many intermediate
    jumps a capture chain made."""
    s = state.action_to_string(state.current_player(), action)
    pairs = _MOVE_RE.findall(s)
    if not pairs:
        return 0, 0
    col_letter, row_digit = pairs[-1]
    x = ord(col_letter) - ord("a")
    y = int(row_digit) - 1
    return max(0, min(7, x)), max(0, min(7, y))


def _pick_action_connect_four(state, rng):
    a = rng.choice(state.legal_actions())
    return a, a, 0, 0  # (raw_action, stored_action_id, x, y)


def _pick_action_tic_tac_toe(state, rng):
    a = rng.choice(state.legal_actions())
    return a, CLICK_ACTION_ID, a % 3, a // 3


def _pick_action_othello(state, rng):
    a = rng.choice(state.legal_actions())
    if a == 64:  # pass
        return a, CLICK_ACTION_ID, 0, 0
    return a, CLICK_ACTION_ID, a % 8, a // 8


def _pick_action_checkers(state, rng):
    a = rng.choice(state.legal_actions())
    x, y = _checkers_click_xy(state, a)
    return a, CLICK_ACTION_ID, x, y


def _pick_action_pig(state, rng):
    a = rng.choice(state.legal_actions())
    # 0="roll", 1="stop" per action_to_string -- both already < NUM_ACTIONS.
    return a, a, 0, 0


def _pick_action_mancala(state, rng):
    a = rng.choice(state.legal_actions())
    # Player 0's pits are ids 1-6, player 1's are 8-13 (verified by direct
    # inspection: legal_actions() at each player's turn). Remapped to a
    # shared 0-5 "my Nth pit from my own store" id via (a-1) % 7 -- loses
    # which player's pit in the stored id (the board itself, which the
    # predictor also conditions on, already encodes whose turn it is), but
    # fits NUM_ACTIONS=8 without an arbitrary truncation.
    stored = (a - 1) % 7
    assert 0 <= stored < NUM_ACTIONS
    return a, stored, 0, 0


_GAME_CONFIG = {
    "connect_four": dict(render=lambda s: _render_spatial(s, (3, 6, 7)), pick=_pick_action_connect_four),
    "tic_tac_toe": dict(render=lambda s: _render_spatial(s, (3, 3, 3)), pick=_pick_action_tic_tac_toe),
    "othello": dict(render=lambda s: _render_spatial(s, (3, 8, 8)), pick=_pick_action_othello),
    "checkers": dict(render=lambda s: _render_spatial(s, (5, 8, 8)), pick=_pick_action_checkers),
    "pig": dict(render=_render_pig, pick=_pick_action_pig),
    "mancala": dict(render=_render_mancala, pick=_pick_action_mancala),
}


def generate_transitions(
    game_name: str,
    num_episodes: int = 400,
    steps_per_episode: int = 60,
    seed: int = 0,
) -> list:
    """Random-policy rollouts for one OpenSpiel game, returned as
    `(frame_t, action_id, x, y, frame_t1, changed, game_id)` tuples --
    the same shape `jepa/data/trajectories.py`'s `TransitionDataset`
    expects. `game_id` is `game_name` itself (one distinct id per game,
    see module docstring)."""
    if game_name not in _GAME_CONFIG:
        raise ValueError(f"unknown game {game_name!r}, expected one of {GAME_IDS}")
    cfg = _GAME_CONFIG[game_name]
    render, pick = cfg["render"], cfg["pick"]
    game = pyspiel.load_game(game_name)
    rng = random.Random(seed)
    transitions = []
    for _episode in range(num_episodes):
        state = game.new_initial_state()
        _advance_through_chance(state, rng)
        frame = [render(state)]
        for _step in range(steps_per_episode):
            if state.is_terminal():
                state = game.new_initial_state()
                _advance_through_chance(state, rng)
                frame = [render(state)]
                continue
            _raw_action, stored_action, x, y = pick(state, rng)
            state.apply_action(_raw_action)
            _advance_through_chance(state, rng)
            next_frame = [render(state)]
            changed = frame != next_frame
            transitions.append((frame, stored_action, x, y, next_frame, changed, game_name))
            frame = next_frame
    return transitions


def generate_all(
    num_episodes: int = 400,
    steps_per_episode: int = 60,
    seed: int = 0,
) -> list:
    """Convenience wrapper: rollouts across all six games, concatenated."""
    out = []
    for i, name in enumerate(GAME_IDS):
        out.extend(generate_transitions(name, num_episodes=num_episodes, steps_per_episode=steps_per_episode, seed=seed + i))
    return out


def _sanity_check(transitions: list) -> dict:
    """Per-game action-id range + changed-rate report -- call before any
    training run (CLAUDE.md's Sokoban gotcha: a silent out-of-range
    action id doesn't fail until a confusing CUDA assert deep inside a
    training batch)."""
    from collections import Counter, defaultdict

    by_game = defaultdict(list)
    for t in transitions:
        by_game[t[6]].append(t)
    report = {}
    for name, ts in by_game.items():
        actions = [t[1] for t in ts]
        changed = sum(1 for t in ts if t[5])
        max_a = max(actions)
        assert max_a < NUM_ACTIONS, f"{name}: action id {max_a} >= NUM_ACTIONS={NUM_ACTIONS}"
        assert min(actions) >= 0, f"{name}: negative action id"
        report[name] = dict(
            n=len(ts),
            max_action=max_a,
            action_hist=dict(Counter(actions)),
            changed_rate=changed / len(ts) if ts else 0.0,
        )
    return report


if __name__ == "__main__":
    import json

    all_t = generate_all(num_episodes=20, steps_per_episode=40)
    rep = _sanity_check(all_t)
    print(json.dumps(rep, indent=2))
