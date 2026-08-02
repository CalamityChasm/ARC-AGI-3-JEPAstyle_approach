"""OpenSpiel-backed board/strategy-game data source for Stage 6's scaled
world-model pretraining (see experiments/stage6_scaled_world_model.md and
experiments/stage6_expanded_roster.md).

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

**stage6-expanded-roster: grew from 6 games to 26** (`pyspiel.
registered_names()` lists 123 games total -- see
experiments/stage6_expanded_roster.md for the full enumeration and
category-by-category exclusion reasoning: simultaneous-move games,
imperfect-information card/hidden-state games, one-shot matrix games, and
a handful of individually-excluded games whose action space genuinely
doesn't compress into this module's click/direct-id conventions, mirroring
the reasoning already used to drop `backgammon` and `nine_mens_morris` in
the original 6-game roster). Every one of the 20 newly-added games gets
its own distinct `game_id` (never pooled, even where two games are
mechanically similar to each other, e.g. `gomoku`/`mnk`/`havannah` --
CLAUDE.md's MinAtar per-game-id lesson is about not pooling *dissimilar*
games under one id to avoid forcing the model to fit inconsistent
action->effect mappings at once; it says nothing against including two
separately-registered, separately-`game_id`d games that happen to share a
mechanic family, and each is still a genuinely distinct OpenSpiel game
with its own board/rules).

Rather than hand-write a bespoke render+action-pick function pair for each
of the 20 new games (this project's original approach for the first 6,
practical at that scale but not at 20+), this module now has three
*generic, reusable* handler families, each verified against every game
that uses it via this module's own `__main__` smoke test before any
full-scale generation (same "verify before relying on it" discipline the
original 6-game module already established for `observation_tensor`'s
one-hot-per-cell layout):

  1. **Cell-index placement** (`_make_pick_cell`): games where the action
     id already IS a board cell (`tic_tac_toe`, `othello`, and now
     `gomoku`, `hex`, `y`, `havannah`, `mnk`, `twixt`, `go9` -- go loaded
     with `board_size=9` rather than OpenSpiel's 19x19 default, to keep
     board size and episode length comparable to this module's other
     games). `x, y = divmod(action_id, width)`; an action id at or past
     `width*height` (a pass move, where one exists) maps to `(0, 0)`,
     mirroring `othello`'s own existing pass handling.
  2. **Destination-click via move-string parsing** (`_make_pick_destination_*`):
     games whose actions are `from -> to` moves with no clean sub-move
     granularity to step through (same reasoning `checkers` was already
     built on) -- extended from one hardcoded algebraic-notation regex to
     two reusable variants, algebraic (`breakthrough`, `clobber`,
     `lines_of_action`, `chess`, `antichess`, `crazyhouse` -- `[a-h]<digit>`
     squares) and numeric-parenthesized (`xiangqi`'s own `"(row,col)"`
     move-string convention, a different board size too: 9 files x 10
     ranks, not 8x8). Both take the *last* coordinate pair in the string
     as the destination, exactly `checkers`'s existing convention for
     multi-jump chains.
  3. **Direct small action id** (no xy needed, action id used as-is):
     `pig`, `mancala` (existing), plus `oware` (6 actions, no remap
     needed -- unlike `mancala`, both players already get ids 0-5
     directly, verified by inspection), `2048` (4 actions), `cliff_walking`
     (4), `catch` (3), `stones_and_gems` (5, a Boulder-Dash-style grid
     game -- direct action id despite having a real spatial board, since
     its action space is small enough to fit `NUM_ACTIONS` without an xy
     detour).

Board rendering follows the same split: **verified one-hot-per-cell
`observation_tensor` reshape** (`_render_spatial`, now clipping to
`NUM_COLORS-1` instead of hard-asserting -- some newly-added games have
more one-hot channels than `NUM_COLORS=16` allows, e.g. `chess` C=20,
`crazyhouse` C=38; clipping collapses rarer channels into the top color
bucket, a documented lossy-but-bounded approximation, not a crash) for
every placement/move game above, plus **generic flat-vector reshape**
(`_render_flat_grid`) for non-spatial-plane sources whose
`observation_tensor` is already a small per-cell value vector rather than
one-hot planes (`oware`'s per-house seed counts, `cliff_walking`/`catch`'s
position indicators, `dots_and_boxes`'s edge/box state), plus one
genuinely bespoke renderer (`_render_2048`, log2-of-tile-value -> color,
since raw tile values like 2048 obviously exceed `NUM_COLORS`).

Six games from the original roster, six distinct `game_id`s (CLAUDE.md's
established "one id per mechanically-dissimilar game" lesson from the
MinAtar per-game-id retry -- `stage6-minatar-pergame-id` found pooling
mechanically-dissimilar games under one shared id measurably hurts;
applied from the start here instead of re-discovered):

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
mancala's 6 relative-pit-index, and the new Tier-3 direct-id games above)
or a single fixed `action_id=6` ("click", matching ARC-3's own ACTION6 id
-- not load-bearing, just a readable convention) with the real choice
encoded in `(x, y)` instead -- never both a large direct id space *and*
xy. `_sanity_check` at the bottom of this module asserts
`max(action_id) < NUM_ACTIONS` for every generated corpus before it's
ever handed to a training run.

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

# Original 6 (unchanged game_ids -- any existing checkpoint's game_vocab_moe.json
# entries for these stay meaningful).
_ORIGINAL_GAME_IDS = ["connect_four", "tic_tac_toe", "othello", "checkers", "pig", "mancala"]

# stage6-expanded-roster: 20 new games, each its own game_id (see module
# docstring for the 3-tier generic-handler design and
# experiments/stage6_expanded_roster.md for the full enumeration/exclusion
# reasoning across all 123 pyspiel.registered_names()).
_NEW_GAME_IDS = [
    # Tier 1: cell-index placement (board size derived from get_parameters()/na)
    "gomoku", "hex", "y", "havannah", "mnk", "twixt", "go9",
    # Tier 2: destination-click via move-string parsing
    "breakthrough", "clobber", "lines_of_action", "chess", "antichess", "crazyhouse", "xiangqi",
    # Tier 3: direct small action id
    "oware", "2048", "cliff_walking", "catch", "stones_and_gems",
    # Tier 1 variant: edge-click (dots_and_boxes' own move-string format)
    "dots_and_boxes",
]

GAME_IDS = _ORIGINAL_GAME_IDS + _NEW_GAME_IDS

# game_id -> pyspiel load string, only where they differ (go9 loads "go"
# with a board_size=9 override -- OpenSpiel's own default go board is
# 19x19, which would make episodes much longer/slower than every other
# source in this roster for no real benefit to mechanical diversity).
_PYSPIEL_LOAD_NAME = {"go9": "go(board_size=9)"}


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
    own board_string before relying on this).

    stage6-expanded-roster: clips to NUM_COLORS-1 instead of hard-
    asserting -- the original 6 games all had C < NUM_COLORS by
    coincidence (never exercised this path), but several newly-added
    games have more one-hot channels than NUM_COLORS allows (chess C=20,
    crazyhouse C=38) since they encode more piece/state types per cell.
    Clipping collapses rarer high-index channels into the top color
    bucket -- a documented lossy-but-bounded approximation (same spirit
    as mancala's own seed-count clipping), not a crash on a game whose
    channel count nobody happened to check in advance."""
    # Explicit player=0 (not the no-arg form, which defaults to
    # state.current_player() internally and raises at terminal states,
    # where current_player() is a sentinel like kTerminalPlayerId=-4).
    # These are all full-information games (no hidden state per player),
    # so which player's "perspective" is requested doesn't change what's
    # observable on the board.
    obs = np.array(state.observation_tensor(0), dtype=np.float32).reshape(shape)
    board = np.argmax(obs, axis=0).astype(np.int64)
    board = np.clip(board, 0, NUM_COLORS - 1)
    return board.tolist()


def _render_flat_grid(state, shape: tuple, scale: float = 1.0) -> list:
    """Generic reshape of a small, already-per-cell-valued
    observation_tensor (NOT one-hot planes -- e.g. oware's normalized
    seed-count-per-house vector, cliff_walking/catch's 0/1 position
    indicators, dots_and_boxes's edge/box-ownership vector) into an
    (H, W) int grid, round+clip into [0, NUM_COLORS-1]. `scale` multiplies
    raw values before rounding (oware's observation_tensor is normalized
    to [0, 1] by total board seed count, not a raw integer count -- see
    module docstring and the game-specific config below)."""
    obs = np.array(state.observation_tensor(0), dtype=np.float64).reshape(shape)
    grid = np.clip(np.round(obs * scale), 0, NUM_COLORS - 1).astype(np.int64)
    return grid.tolist()


def _render_2048(state) -> list:
    """2048's observation_tensor is the raw board of tile values (0, 2,
    4, 8, ..., verified directly -- not one-hot, not normalized), which
    obviously exceeds NUM_COLORS if used as-is. log2(tile_value) maps the
    whole practically-reachable range (up to 2048 = 2^11) into
    [0, NUM_COLORS-1] without collision between distinct tile values,
    unlike a generic clip."""
    obs = np.array(state.observation_tensor(0), dtype=np.float64).reshape(4, 4)
    grid = np.zeros((4, 4), dtype=np.int64)
    nonzero = obs > 0
    grid[nonzero] = np.clip(np.log2(obs[nonzero]).astype(np.int64), 0, NUM_COLORS - 1)
    return grid.tolist()


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


def _make_pick_cell(width: int, height: int):
    """Tier 1 generic: games whose action id already IS a board cell
    (row-major, `x, y = divmod(a, width)`), e.g. tic_tac_toe/othello
    (already hardcoded below, kept as-is) and gomoku/hex/y/havannah/mnk/
    twixt/go9 (new). An action id at or past width*height (a pass move,
    where one exists, e.g. go) maps to (0, 0), mirroring othello's own
    existing pass handling."""

    def pick(state, rng):
        a = rng.choice(state.legal_actions())
        if a >= width * height:
            return a, CLICK_ACTION_ID, 0, 0
        x, y = a % width, a // width
        return a, CLICK_ACTION_ID, x, y

    return pick


_ALGEBRAIC_MOVE_RE = re.compile(r"([a-h])(\d+)")
_NUMERIC_MOVE_RE = re.compile(r"\((\d+),\s*(\d+)\)")


def _make_pick_destination_algebraic(width: int, height: int):
    """Tier 2 generic: `[a-h]<digit>`-style algebraic move strings
    (breakthrough, clobber, lines_of_action, chess, antichess,
    crazyhouse), extending checkers' own hardcoded regex to a reusable,
    per-game-board-size-clipped factory. Takes the LAST coordinate pair
    in the string as the destination (identical convention to checkers'
    multi-jump handling) -- correctly reduces to the single square for
    pawn-style moves like "a3" that have no explicit source square."""

    def pick(state, rng):
        a = rng.choice(state.legal_actions())
        s = state.action_to_string(state.current_player(), a)
        pairs = _ALGEBRAIC_MOVE_RE.findall(s)
        if not pairs:
            return a, CLICK_ACTION_ID, 0, 0
        col_letter, row_digit = pairs[-1]
        x = max(0, min(width - 1, ord(col_letter) - ord("a")))
        y = max(0, min(height - 1, int(row_digit) - 1))
        return a, CLICK_ACTION_ID, x, y

    return pick


def _make_pick_destination_numeric(width: int, height: int):
    """Tier 2 generic, numeric-parenthesized variant: xiangqi's own
    "(row,col)-(row,col)" move-string convention (verified by direct
    inspection -- a different format from every other move game in this
    roster, and a different board shape too: 9 files x 10 ranks, not
    8x8). Takes the LAST (row, col) pair as the destination."""

    def pick(state, rng):
        a = rng.choice(state.legal_actions())
        s = state.action_to_string(state.current_player(), a)
        pairs = _NUMERIC_MOVE_RE.findall(s)
        if not pairs:
            return a, CLICK_ACTION_ID, 0, 0
        r, c = pairs[-1]
        y = max(0, min(height - 1, int(r)))
        x = max(0, min(width - 1, int(c)))
        return a, CLICK_ACTION_ID, x, y

    return pick


_DOTS_AND_BOXES_RE = re.compile(r"\((\w),(\d+),(\d+)\)")


def _pick_dots_and_boxes(state, rng):
    """dots_and_boxes' own move-string format, e.g. "P1(h,0,0)"
    (orientation, row, col) -- a Tier-1-adjacent edge-click game, not a
    cell-placement one, so it gets its own small parser rather than
    reusing _make_pick_cell (action id is an edge index, not a cell
    index)."""
    a = rng.choice(state.legal_actions())
    s = state.action_to_string(state.current_player(), a)
    m = _DOTS_AND_BOXES_RE.search(s)
    if not m:
        return a, CLICK_ACTION_ID, 0, 0
    _orient, r, c = m.groups()
    x = max(0, min(8, int(c)))
    y = max(0, min(8, int(r)))
    return a, CLICK_ACTION_ID, x, y


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


def _pick_action_direct(state, rng):
    """Tier 3 generic: action id used as-is (already < NUM_ACTIONS,
    verified per-game below before being added to this config) -- no xy,
    no CLICK_ACTION_ID indirection. Shared by oware/2048/cliff_walking/
    catch/stones_and_gems (pig/mancala keep their own hand-written
    pick functions -- mancala needs the (a-1) % 7 remap, pig's raw ids
    are already exactly right)."""
    a = rng.choice(state.legal_actions())
    return a, a, 0, 0


_GAME_CONFIG = {
    "connect_four": dict(render=lambda s: _render_spatial(s, (3, 6, 7)), pick=_pick_action_connect_four),
    "tic_tac_toe": dict(render=lambda s: _render_spatial(s, (3, 3, 3)), pick=_pick_action_tic_tac_toe),
    "othello": dict(render=lambda s: _render_spatial(s, (3, 8, 8)), pick=_pick_action_othello),
    "checkers": dict(render=lambda s: _render_spatial(s, (5, 8, 8)), pick=_pick_action_checkers),
    "pig": dict(render=_render_pig, pick=_pick_action_pig),
    "mancala": dict(render=_render_mancala, pick=_pick_action_mancala),

    # --- stage6-expanded-roster: 20 new games ---
    # Tier 1: cell-index placement. Board dims verified via
    # get_parameters()/num_distinct_actions before being hardcoded here
    # (see experiments/stage6_expanded_roster.md) -- NOT guessed.
    "gomoku": dict(render=lambda s: _render_spatial(s, (3, 15, 15)), pick=_make_pick_cell(15, 15)),
    "hex": dict(render=lambda s: _render_spatial(s, (9, 11, 11)), pick=_make_pick_cell(11, 11)),
    "y": dict(render=lambda s: _render_spatial(s, (3, 19, 19)), pick=_make_pick_cell(19, 19)),
    "havannah": dict(render=lambda s: _render_spatial(s, (3, 15, 15)), pick=_make_pick_cell(15, 15)),
    "mnk": dict(render=lambda s: _render_spatial(s, (3, 15, 15)), pick=_make_pick_cell(15, 15)),
    "twixt": dict(render=lambda s: _render_spatial(s, (9, 8, 8)), pick=_make_pick_cell(8, 8)),
    "go9": dict(render=lambda s: _render_spatial(s, (4, 9, 9)), pick=_make_pick_cell(9, 9)),

    # Tier 2: destination-click via move-string parsing.
    "breakthrough": dict(render=lambda s: _render_spatial(s, (3, 8, 8)), pick=_make_pick_destination_algebraic(8, 8)),
    "clobber": dict(render=lambda s: _render_spatial(s, (3, 5, 6)), pick=_make_pick_destination_algebraic(6, 5)),
    "lines_of_action": dict(render=lambda s: _render_spatial(s, (3, 8, 8)), pick=_make_pick_destination_algebraic(8, 8)),
    "chess": dict(render=lambda s: _render_spatial(s, (20, 8, 8)), pick=_make_pick_destination_algebraic(8, 8)),
    "antichess": dict(render=lambda s: _render_spatial(s, (16, 8, 8)), pick=_make_pick_destination_algebraic(8, 8)),
    "crazyhouse": dict(render=lambda s: _render_spatial(s, (38, 8, 8)), pick=_make_pick_destination_algebraic(8, 8)),
    "xiangqi": dict(render=lambda s: _render_spatial(s, (15, 10, 9)), pick=_make_pick_destination_numeric(9, 10)),

    # Tier 1 variant: edge-click.
    "dots_and_boxes": dict(render=lambda s: _render_flat_grid(s, (9, 9)), pick=_pick_dots_and_boxes),

    # Tier 3: direct small action id.
    # oware's observation_tensor is normalized (seed_count / 48, verified
    # by direct inspection -- 4 seeds/house * 6 houses * 2 players = 48
    # total board seeds), hence scale=48 to recover integer-ish seed
    # counts before rounding/clipping (mirrors mancala's own seed-count
    # clipping, just with an extra denormalization step mancala's own
    # observation_tensor didn't need).
    "oware": dict(render=lambda s: _render_flat_grid(s, (2, 7), scale=48.0), pick=_pick_action_direct),
    "2048": dict(render=_render_2048, pick=_pick_action_direct),
    "cliff_walking": dict(render=lambda s: _render_flat_grid(s, (4, 8), scale=5.0), pick=_pick_action_direct),
    "catch": dict(render=lambda s: _render_flat_grid(s, (10, 5), scale=5.0), pick=_pick_action_direct),
    "stones_and_gems": dict(render=lambda s: _render_spatial(s, (31, 12, 20)), pick=_pick_action_direct),
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
    # game_id (used for the training corpus's game_id/vocab) vs. the
    # actual pyspiel load string can differ -- go9's game_id stays "go9"
    # (a stable, self-describing vocab entry) while the real load call
    # needs "go(board_size=9)" (see _PYSPIEL_LOAD_NAME above).
    game = pyspiel.load_game(_PYSPIEL_LOAD_NAME.get(game_name, game_name))
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
    """Convenience wrapper: rollouts across every game in GAME_IDS, concatenated."""
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
