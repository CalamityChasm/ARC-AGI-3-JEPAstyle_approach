"""Procedurally generated ARC-3-*shaped* puzzles for Stage 4/6 dynamics
pretraining -- the one data-diversity direction CLAUDE.md's Stage 6
addendum explicitly flags as untried after 13 independent interventions
(conditioning fixes, and five separate borrowed-game-engine data sources
-- MiniGrid, Sokoban, MinAtar, Procgen, a 26-game OpenSpiel roster) all
failed to move held-out-*game* generalization. Every one of those sources
is a real game engine from an unrelated genre (navigation, block-pushing,
reflex arcade, platformer/maze, board games) borrowed wholesale; none of
them are built out of ARC-3's own puzzle-logic vocabulary. This module
instead hand-builds small procedural puzzle families directly out of
mechanics observed in this project's own local games (see
`experiments/stage6_bp35_ka59_mechanics.md`'s direct frame inspection of
`bp35` -- a maze-navigation puzzle with a rising hazard -- and `ka59` -- a
dual-token routing puzzle -- plus `r11l`'s own diagonal path-tracing
character, `ft09`'s icon-pattern-matching board, and `sp80`'s
platform-reaching structure, all visually inspected via
`scripts/render_frames_png.py` before this module was written) and
`rules.md`'s Action Space section (RESET, ACTION1-5 simple, ACTION6
complex-with-(x,y), ACTION7 -- semantics "vary per game and must be
discovered through exploration", exactly the convention followed here).

Five distinct puzzle TYPES, each its own `game_id` (this project's own
hard-learned lesson from the MinAtar retry: pooling mechanically-
dissimilar sub-games under one shared id was a real, large confound --
only pool under a shared id when there's a MiniGrid-style genuine
shared-semantics reason to, which none of these five have with each
other):

  - `arcsyn_route`     -- move a token through walls to a goal cell
                          (routing/reaching, echoing ka59/r11l/bp35).
  - `arcsyn_toggle`     -- click a cell to toggle it + its 4-neighbors
                          between two colors (Lights-Out-style local
                          cause-effect rule, a classic ARC mechanic).
  - `arcsyn_match`      -- click the on-board object whose color matches
                          a displayed target swatch to clear it (color/
                          object identity + counting).
  - `arcsyn_symmetry`   -- a single action mirrors a half-drawn pattern
                          onto its blank other half (symmetry/pattern
                          completion, one of the most common ARC-1/2
                          transformation types).
  - `arcsyn_enclosure`  -- move a cursor and trigger a flood-fill that
                          only does anything if the cursor sits inside a
                          fully walled-in region (spatial containment/
                          enclosure).

Each type has real procedural variation per generated episode (random
grid size, wall/object layout, and color draws) rather than one fixed
puzzle repeated -- matching how minigrid_data.py/sokoban_data.py vary
room layout per episode. All grids are well under jepa/grid.py's 64x64
CANVAS and get placed top-left by grid_to_tensor/patch_change_mask's
existing general (non-ARC-specific) placement logic -- no changes needed
there.

Action ids used: 0 (no-op/reset marker), 1-4 (cardinal movement, when the
puzzle type has a moving token/cursor), 5 (a single-purpose "primary
interact" trigger -- toggle-independent puzzles use it for their one
non-movement effect: mirror-complete for `arcsyn_symmetry`, flood-fill
for `arcsyn_enclosure`), 6 (ACTION6-style complex click requiring (x, y),
used by `arcsyn_toggle` and `arcsyn_match`), 7 (unused no-op filler,
included because real ARC-3 games routinely leave some action ids
meaningless for a given game -- see rules.md). Max action id used is 6,
comfortably under jepa/models/predictor.py's NUM_ACTIONS=8 ceiling
(sanity-checked at the bottom of this module, per this project's own
"check max(action_ids) < NUM_ACTIONS before training, not after a
confusing CUDA crash" gotcha -- see CLAUDE.md).
"""

import random
from collections import deque

import numpy as np

from ..grid import NUM_COLORS

# ---------------------------------------------------------------------------
# Shared conventions
# ---------------------------------------------------------------------------

NOOP, UP, DOWN, LEFT, RIGHT, INTERACT, CLICK, UNUSED = range(8)
_MOVE_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

GAME_ID_ROUTE = "arcsyn_route"
GAME_ID_TOGGLE = "arcsyn_toggle"
GAME_ID_MATCH = "arcsyn_match"
GAME_ID_SYMMETRY = "arcsyn_symmetry"
GAME_ID_ENCLOSURE = "arcsyn_enclosure"
ALL_GAME_IDS = [GAME_ID_ROUTE, GAME_ID_TOGGLE, GAME_ID_MATCH, GAME_ID_SYMMETRY, GAME_ID_ENCLOSURE]


def _frame(grid: np.ndarray) -> list:
    """(H, W) int array -> `[grid_as_list]`, the one-layer-list convention
    `arc3_frame_to_tensor`/`patch_change_mask` expect (frame[0]), matching
    minigrid_data.py/sokoban_data.py's own wrapping."""
    return [grid.astype(np.int64).tolist()]


def _random_empty_cell(rng: random.Random, occupied: set, h: int, w: int) -> tuple:
    while True:
        r, c = rng.randrange(h), rng.randrange(w)
        if (r, c) not in occupied:
            return r, c


# ---------------------------------------------------------------------------
# 1. arcsyn_route -- token routing to a goal cell through sparse walls
# ---------------------------------------------------------------------------
# Colors: 0=background, 1=wall, 2=token, 3=goal.

_ROUTE_WALL_DENSITY = 0.10


def _route_new_layout(rng: random.Random, h: int, w: int) -> dict:
    grid = np.zeros((h, w), dtype=np.int64)
    n_walls = int(h * w * _ROUTE_WALL_DENSITY)
    occupied = set()
    for _ in range(n_walls):
        r, c = rng.randrange(h), rng.randrange(w)
        grid[r, c] = 1
        occupied.add((r, c))
    tr, tc = _random_empty_cell(rng, occupied, h, w)
    occupied.add((tr, tc))
    gr, gc = _random_empty_cell(rng, occupied, h, w)
    grid[tr, tc] = 2
    grid[gr, gc] = 3
    return {"grid": grid, "token": (tr, tc), "goal": (gr, gc), "h": h, "w": w}


def _route_render(state: dict) -> np.ndarray:
    return state["grid"]


def _generate_route(episodes: int, steps_per_episode: int, rng: random.Random) -> list:
    transitions = []
    for _ep in range(episodes):
        h, w = rng.randint(10, 18), rng.randint(10, 18)
        state = _route_new_layout(rng, h, w)
        frame = _frame(_route_render(state))
        for _step in range(steps_per_episode):
            action = rng.choice([UP, DOWN, LEFT, RIGHT, NOOP, INTERACT, CLICK, UNUSED])
            moved = False
            if action in _MOVE_DELTA:
                dr, dc = _MOVE_DELTA[action]
                tr, tc = state["token"]
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < state["h"] and 0 <= nc < state["w"] and state["grid"][nr, nc] != 1:
                    state["grid"][tr, tc] = 0
                    if (nr, nc) == state["goal"]:
                        state["grid"][nr, nc] = 2  # token overlaps goal cell
                    else:
                        state["grid"][nr, nc] = 2
                    state["token"] = (nr, nc)
                    moved = True
            next_frame = _frame(_route_render(state))
            changed = frame != next_frame
            transitions.append((frame, action, 0, 0, next_frame, changed, GAME_ID_ROUTE))
            frame = next_frame
            if moved and state["token"] == state["goal"]:
                h2, w2 = rng.randint(10, 18), rng.randint(10, 18)
                state = _route_new_layout(rng, h2, w2)
                frame = _frame(_route_render(state))
    return transitions


# ---------------------------------------------------------------------------
# 2. arcsyn_toggle -- click a cell to toggle it + 4-neighbors (Lights Out)
# ---------------------------------------------------------------------------
# Colors: 0=off, 4=on.

_TOGGLE_ON_DENSITY = 0.40


def _toggle_new_layout(rng: random.Random, h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            if rng.random() < _TOGGLE_ON_DENSITY:
                grid[r, c] = 4
    return grid


def _generate_toggle(episodes: int, steps_per_episode: int, rng: random.Random) -> list:
    transitions = []
    for _ep in range(episodes):
        h, w = rng.randint(8, 14), rng.randint(8, 14)
        grid = _toggle_new_layout(rng, h, w)
        frame = _frame(grid)
        for _step in range(steps_per_episode):
            action = rng.choice([CLICK, CLICK, CLICK, NOOP, INTERACT, UP, DOWN, UNUSED])
            x = y = 0
            if action == CLICK:
                y, x = rng.randrange(h), rng.randrange(w)
                for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r, c = y + dr, x + dc
                    if 0 <= r < h and 0 <= c < w:
                        grid[r, c] = 4 if grid[r, c] == 0 else 0
            next_frame = _frame(grid)
            changed = frame != next_frame
            transitions.append((frame, action, x, y, next_frame, changed, GAME_ID_TOGGLE))
            frame = next_frame
        # (episode ends after steps_per_episode; next episode draws a fresh layout)
    return transitions


# ---------------------------------------------------------------------------
# 3. arcsyn_match -- click the object matching a target-color swatch
# ---------------------------------------------------------------------------
# Colors: 0=background, 9=swatch border, {1..8}\{9}-ish object colors
# drawn from a fixed small palette avoiding 0.

_OBJECT_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
_SWATCH_BORDER = 9


def _match_place_objects(rng: random.Random, h: int, w: int, n_objects: int) -> dict:
    grid = np.zeros((h, w), dtype=np.int64)
    colors = rng.sample(_OBJECT_COLORS, k=min(n_objects, len(_OBJECT_COLORS)))
    objects = {}  # color -> (r0, c0) top-left of its 2x2 block
    occupied = set()
    for color in colors:
        for _try in range(50):
            r0, c0 = rng.randrange(2, h - 2), rng.randrange(2, w - 2)
            cells = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
            if not (cells & occupied):
                occupied |= cells
                objects[color] = (r0, c0)
                for rr, cc in cells:
                    grid[rr, cc] = color
                break
    # HUD swatch: fixed 2x2 block at top-left, bordered so it reads distinct
    # from a same-colored on-board object.
    target = rng.choice(list(objects.keys()))
    grid[0, 0:2] = _SWATCH_BORDER
    grid[1, 0:2] = target
    return {"grid": grid, "objects": objects, "target": target, "h": h, "w": w}


def _match_reset_target(rng: random.Random, state: dict) -> None:
    if not state["objects"]:
        return
    state["target"] = rng.choice(list(state["objects"].keys()))
    state["grid"][0, 0:2] = _SWATCH_BORDER
    state["grid"][1, 0:2] = state["target"]


def _generate_match(episodes: int, steps_per_episode: int, rng: random.Random) -> list:
    transitions = []
    for _ep in range(episodes):
        h, w = rng.randint(14, 20), rng.randint(14, 20)
        n_objects = rng.randint(5, 8)
        state = _match_place_objects(rng, h, w, n_objects)
        frame = _frame(state["grid"])
        for _step in range(steps_per_episode):
            action = rng.choice([CLICK, CLICK, CLICK, NOOP, INTERACT, UP, DOWN, UNUSED])
            x = y = 0
            if action == CLICK:
                y, x = rng.randrange(2, state["h"]), rng.randrange(0, state["w"])
                color = int(state["grid"][y, x])
                if color == state["target"] and color in state["objects"]:
                    r0, c0 = state["objects"].pop(color)
                    state["grid"][r0:r0 + 2, c0:c0 + 2] = 0
                    if state["objects"]:
                        _match_reset_target(rng, state)
                    else:
                        h2, w2 = rng.randint(14, 20), rng.randint(14, 20)
                        state = _match_place_objects(rng, h2, w2, rng.randint(5, 8))
            next_frame = _frame(state["grid"])
            changed = frame != next_frame
            transitions.append((frame, action, x, y, next_frame, changed, GAME_ID_MATCH))
            frame = next_frame
    return transitions


# ---------------------------------------------------------------------------
# 4. arcsyn_symmetry -- one action mirrors the left half onto the right
# ---------------------------------------------------------------------------
# Colors: 0=background, 1-8 pattern colors.

_SYMMETRY_DENSITY = 0.20


def _symmetry_new_pattern(rng: random.Random, h: int, w_half: int) -> np.ndarray:
    grid = np.zeros((h, w_half * 2), dtype=np.int64)
    for r in range(h):
        for c in range(w_half):
            if rng.random() < _SYMMETRY_DENSITY:
                grid[r, c] = rng.randint(1, 8)
    return grid


def _generate_symmetry(episodes: int, steps_per_episode: int, rng: random.Random) -> list:
    transitions = []
    for _ep in range(episodes):
        h, w_half = rng.randint(6, 10), rng.randint(6, 10)
        grid = _symmetry_new_pattern(rng, h, w_half)
        completed = False
        frame = _frame(grid)
        for _step in range(steps_per_episode):
            action = rng.choice([INTERACT, INTERACT, NOOP, UP, DOWN, LEFT, RIGHT, CLICK])
            if action == INTERACT and not completed:
                grid[:, w_half:] = grid[:, :w_half][:, ::-1]
                completed = True
            next_frame = _frame(grid)
            changed = frame != next_frame
            transitions.append((frame, action, 0, 0, next_frame, changed, GAME_ID_SYMMETRY))
            frame = next_frame
        # fresh pattern next episode regardless of completion
    return transitions


# ---------------------------------------------------------------------------
# 5. arcsyn_enclosure -- flood-fill only triggers inside a walled region
# ---------------------------------------------------------------------------
# Colors: 0=background, 1=wall, 5=cursor, 6=flooded.

def _enclosure_new_layout(rng: random.Random, h: int, w: int) -> dict:
    grid = np.zeros((h, w), dtype=np.int64)
    # A random rectangular room somewhere inside the grid, walled on all
    # four sides, with a 50% chance of one open gap (so roughly half of
    # generated rooms are genuinely enclosed and half are not -- both
    # cases are informative: flood-fill should fire in one, not the other).
    rh = rng.randint(4, min(8, h - 2))
    rw = rng.randint(4, min(8, w - 2))
    r0 = rng.randint(1, h - rh - 1)
    c0 = rng.randint(1, w - rw - 1)
    grid[r0 - 1:r0 + rh + 1, c0 - 1] = 1
    grid[r0 - 1:r0 + rh + 1, c0 + rw] = 1
    grid[r0 - 1, c0 - 1:c0 + rw + 1] = 1
    grid[r0 + rh, c0 - 1:c0 + rw + 1] = 1
    enclosed = True
    if rng.random() < 0.5:
        # open a gap at a random point on the boundary
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            grid[r0 - 1, rng.randrange(c0, c0 + rw)] = 0
        elif side == "bottom":
            grid[r0 + rh, rng.randrange(c0, c0 + rw)] = 0
        elif side == "left":
            grid[rng.randrange(r0, r0 + rh), c0 - 1] = 0
        else:
            grid[rng.randrange(r0, r0 + rh), c0 + rw] = 0
        enclosed = False
    # cursor starts somewhere outside the room's wall ring
    occupied = {(r, c) for r in range(max(0, r0 - 1), min(h, r0 + rh + 1))
                for c in range(max(0, c0 - 1), min(w, c0 + rw + 1))}
    cr, cc = _random_empty_cell(rng, occupied, h, w)
    grid[cr, cc] = 5
    return {
        "grid": grid, "cursor": (cr, cc), "h": h, "w": w,
        "room": (r0, c0, rh, rw), "enclosed": enclosed, "flooded": False,
    }


def _is_enclosed(grid: np.ndarray, start: tuple, h: int, w: int) -> tuple:
    """BFS flood-fill from `start` over non-wall cells; returns
    (reaches_boundary, region_cells). A region that never touches the
    grid's outer edge is enclosed."""
    seen = {start}
    queue = deque([start])
    touches_edge = start[0] in (0, h - 1) or start[1] in (0, w - 1)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr, nc] != 1:
                seen.add((nr, nc))
                queue.append((nr, nc))
                if nr in (0, h - 1) or nc in (0, w - 1):
                    touches_edge = True
    return touches_edge, seen


def _generate_enclosure(episodes: int, steps_per_episode: int, rng: random.Random) -> list:
    transitions = []
    for _ep in range(episodes):
        h, w = rng.randint(12, 20), rng.randint(12, 20)
        state = _enclosure_new_layout(rng, h, w)
        frame = _frame(state["grid"])
        for _step in range(steps_per_episode):
            action = rng.choice([UP, DOWN, LEFT, RIGHT, INTERACT, INTERACT, NOOP, CLICK])
            if action in _MOVE_DELTA:
                dr, dc = _MOVE_DELTA[action]
                cr, cc = state["cursor"]
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < state["h"] and 0 <= nc < state["w"] and state["grid"][nr, nc] != 1:
                    if state["grid"][cr, cc] == 5:
                        state["grid"][cr, cc] = 0
                    state["grid"][nr, nc] = 5
                    state["cursor"] = (nr, nc)
            elif action == INTERACT and not state["flooded"]:
                touches_edge, region = _is_enclosed(state["grid"], state["cursor"], state["h"], state["w"])
                if not touches_edge:
                    for (rr, cc) in region:
                        if state["grid"][rr, cc] == 0:
                            state["grid"][rr, cc] = 6
                    state["grid"][state["cursor"][0], state["cursor"][1]] = 6
                    state["flooded"] = True
            next_frame = _frame(state["grid"])
            changed = frame != next_frame
            transitions.append((frame, action, 0, 0, next_frame, changed, GAME_ID_ENCLOSURE))
            frame = next_frame
    return transitions


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_GENERATORS = {
    GAME_ID_ROUTE: _generate_route,
    GAME_ID_TOGGLE: _generate_toggle,
    GAME_ID_MATCH: _generate_match,
    GAME_ID_SYMMETRY: _generate_symmetry,
    GAME_ID_ENCLOSURE: _generate_enclosure,
}


def generate_transitions(
    episodes_per_type: int = 168,
    steps_per_episode: int = 80,
    seed: int = 0,
    game_ids: list | None = None,
) -> list:
    """Random-policy rollouts across all 5 synthetic puzzle types (or a
    subset via `game_ids`), returned as `(frame_t, action_id, x, y,
    frame_t1, changed, game_id)` tuples -- same shape every other data
    source in this package uses.

    Default `episodes_per_type=168, steps_per_episode=80` across the 5
    types gives 168 * 80 * 5 = 67,200 transitions total, deliberately
    matched to minigrid_data.py's own default (40 episodes * 80 steps *
    21 envs = 67,200) so a controlled pretrain-phase comparison holds
    total samples-seen constant -- see CLAUDE.md's Stage 6 addendum on
    why curriculum/sample-count balance matters for this kind of
    comparison (the Procgen attempts' first confound).
    """
    game_ids = game_ids or ALL_GAME_IDS
    transitions = []
    for i, game_id in enumerate(game_ids):
        rng = random.Random(seed * 1_000_003 + i)
        transitions += _GENERATORS[game_id](episodes_per_type, steps_per_episode, rng)
    return transitions


if __name__ == "__main__":
    # Quick self-check: generate a small sample and verify the schema +
    # action-space ceiling before any real training run touches this
    # data -- this project's own hard-learned lesson (Sokoban's 9-action
    # space silently overflowing NUM_ACTIONS=8 and crashing deep inside a
    # CUDA kernel on whatever batch first sampled it) says to check this
    # up front, not after a confusing crash.
    from ..models.predictor import NUM_ACTIONS

    sample = generate_transitions(episodes_per_type=4, steps_per_episode=20, seed=0)
    max_action = max(t[1] for t in sample)
    assert max_action < NUM_ACTIONS, f"action id {max_action} >= NUM_ACTIONS={NUM_ACTIONS}"
    changed_rate = sum(1 for t in sample if t[5]) / len(sample)
    by_game = {}
    for t in sample:
        by_game.setdefault(t[6], []).append(t[5])
    print(f"{len(sample)} transitions, max_action={max_action}, overall changed_rate={changed_rate:.3f}")
    for g, changed_flags in by_game.items():
        print(f"  {g}: n={len(changed_flags)} changed_rate={sum(changed_flags) / len(changed_flags):.3f}")
