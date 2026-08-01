"""Hand-rolled Snake and Pong data sources for Stage 6's scaled world-model
pretraining (see experiments/stage6_scaled_world_model.md). Per this
branch's own scoping instructions: no good off-the-shelf source exists for
either (unlike MiniGrid/Sokoban/OpenSpiel), so both are small, direct
implementations rather than a wrapped third-party env. Deliberately simple
(a few dozen lines of state-update logic each) -- the point is genuinely
different *physics/real-time-feel* mechanics (continuous-ish motion,
collision, growth, bouncing) to complement `openspiel_data.py`'s turn-based
board-game mechanics and the existing MiniGrid/Sokoban/MinAtar/Procgen
sources' navigation/arcade mechanics, not a faithful game clone.

Two distinct `game_id`s (`"snake"`, `"pong"`) -- genuinely different
mechanics from each other (growth/self-collision vs. bounce physics),
consistent with this project's "one id per mechanically-dissimilar game"
rule (see CLAUDE.md's MinAtar per-game-id lesson).

Both use small direct action ids (Snake: 4 directions, Pong: 3 paddle
moves) -- well under `jepa/models/predictor.py`'s `NUM_ACTIONS=8` budget,
no click/xy mechanism needed. `x, y` are always 0, matching
minigrid_data.py/sokoban_data.py's convention for sources with no
coordinate-based action.
"""

import random

import numpy as np

from ..grid import NUM_COLORS
from ..models.predictor import NUM_ACTIONS

SNAKE_GAME_ID = "snake"
PONG_GAME_ID = "pong"
GAME_IDS = [SNAKE_GAME_ID, PONG_GAME_ID]

# ---------------------------------------------------------------------------
# Snake
# ---------------------------------------------------------------------------

SNAKE_H = 14
SNAKE_W = 14
_SNAKE_EMPTY, _SNAKE_BODY, _SNAKE_HEAD, _SNAKE_FOOD = 0, 1, 2, 3
_SNAKE_DIRS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right


class _SnakeState:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        cy, cx = SNAKE_H // 2, SNAKE_W // 2
        self.body = [(cy, cx), (cy, cx - 1), (cy, cx - 2)]  # head first
        self._place_food()

    def _place_food(self) -> None:
        occupied = set(self.body)
        free = [(r, c) for r in range(SNAKE_H) for c in range(SNAKE_W) if (r, c) not in occupied]
        self.food = self.rng.choice(free) if free else None

    def render(self) -> list:
        grid = np.zeros((SNAKE_H, SNAKE_W), dtype=np.int64)
        for r, c in self.body[1:]:
            grid[r, c] = _SNAKE_BODY
        hr, hc = self.body[0]
        grid[hr, hc] = _SNAKE_HEAD
        if self.food is not None:
            fr, fc = self.food
            grid[fr, fc] = _SNAKE_FOOD
        return grid.tolist()

    def step(self, action: int) -> bool:
        """Apply one move; returns True if the snake died this step (wall
        or self collision) -- the resulting frame still reflects the fatal
        head position (real transition), reset happens on the *next*
        call's caller, matching sokoban_data.py's own done-then-reset-next
        loop convention."""
        dr, dc = _SNAKE_DIRS[action]
        hr, hc = self.body[0]
        nr, nc = hr + dr, hc + dc
        if not (0 <= nr < SNAKE_H and 0 <= nc < SNAKE_W):
            return True  # wall collision -- head doesn't actually move onto an invalid cell
        if (nr, nc) in self.body[:-1]:  # colliding with own body (tail cell is vacated this step, so excluded)
            return True
        ate = self.food is not None and (nr, nc) == self.food
        self.body.insert(0, (nr, nc))
        if ate:
            self._place_food()
        else:
            self.body.pop()
        return False


def generate_snake_transitions(num_episodes: int = 400, steps_per_episode: int = 80, seed: int = 0) -> list:
    """Random-policy Snake rollouts, `(frame_t, action_id, x, y, frame_t1,
    changed, game_id)` tuples. Death (wall/self-collision) ends the episode
    immediately -- a random policy dies often and quickly (frequent
    resets), which is an honest property of this source, not a bug: it
    means most episodes are short, high-changed-rate bursts rather than
    long wandering, similar in spirit to how ARC-3's own short episodes
    work."""
    rng = random.Random(seed)
    state = _SnakeState(rng)
    transitions = []
    for _episode in range(num_episodes):
        state.reset()
        frame = [state.render()]
        for _step in range(steps_per_episode):
            action = rng.randrange(4)
            died = state.step(action)
            next_frame = [state.render()]
            changed = frame != next_frame
            transitions.append((frame, action, 0, 0, next_frame, changed, SNAKE_GAME_ID))
            frame = next_frame
            if died:
                state.reset()
                frame = [state.render()]
    return transitions


# ---------------------------------------------------------------------------
# Pong (single paddle vs. three walls -- a "squash/warm-up" simplification,
# not two-player -- see module docstring)
# ---------------------------------------------------------------------------

PONG_H = 16
PONG_W = 24
PADDLE_LEN = 4
_PONG_EMPTY, _PONG_PADDLE, _PONG_BALL = 0, 1, 2


class _PongState:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self.paddle_top = PONG_H // 2 - PADDLE_LEN // 2  # paddle sits on the left edge (col 0)
        self.ball_r = float(self.rng.randrange(2, PONG_H - 2))
        self.ball_c = float(PONG_W // 2)
        self.vr = self.rng.choice([-1.0, 1.0]) * 0.7
        self.vc = -1.0  # heads toward the paddle first

    def render(self) -> list:
        grid = np.zeros((PONG_H, PONG_W), dtype=np.int64)
        for r in range(self.paddle_top, self.paddle_top + PADDLE_LEN):
            if 0 <= r < PONG_H:
                grid[r, 0] = _PONG_PADDLE
        br, bc = int(round(self.ball_r)), int(round(self.ball_c))
        br = max(0, min(PONG_H - 1, br))
        bc = max(0, min(PONG_W - 1, bc))
        grid[br, bc] = _PONG_BALL
        return grid.tolist()

    def step(self, action: int) -> bool:
        """action: 0=up, 1=down, 2=stay. Returns True if the ball got past
        the paddle this step (miss -- episode-ending)."""
        if action == 0:
            self.paddle_top = max(0, self.paddle_top - 1)
        elif action == 1:
            self.paddle_top = min(PONG_H - PADDLE_LEN, self.paddle_top + 1)

        self.ball_r += self.vr
        self.ball_c += self.vc
        if self.ball_r <= 0 or self.ball_r >= PONG_H - 1:
            self.vr *= -1
            self.ball_r = max(0.0, min(float(PONG_H - 1), self.ball_r))
        if self.ball_c >= PONG_W - 1:
            self.vc *= -1
            self.ball_c = float(PONG_W - 1)
        if self.ball_c <= 0:
            if self.paddle_top <= self.ball_r <= self.paddle_top + PADDLE_LEN - 1:
                self.vc *= -1
                self.ball_c = 0.0
            else:
                return True  # missed -- ball passes the paddle plane
        return False


def generate_pong_transitions(num_episodes: int = 400, steps_per_episode: int = 100, seed: int = 0) -> list:
    """Random-policy Pong rollouts (paddle vs. bouncing ball), same tuple
    shape as generate_snake_transitions. A missed return ends the episode
    immediately."""
    rng = random.Random(seed)
    state = _PongState(rng)
    transitions = []
    for _episode in range(num_episodes):
        state.reset()
        frame = [state.render()]
        for _step in range(steps_per_episode):
            action = rng.randrange(3)
            missed = state.step(action)
            next_frame = [state.render()]
            changed = frame != next_frame
            transitions.append((frame, action, 0, 0, next_frame, changed, PONG_GAME_ID))
            frame = next_frame
            if missed:
                state.reset()
                frame = [state.render()]
    return transitions


def generate_all(num_episodes: int = 400, seed: int = 0) -> list:
    return generate_snake_transitions(num_episodes=num_episodes, seed=seed) + generate_pong_transitions(
        num_episodes=num_episodes, seed=seed + 1
    )


def _sanity_check(transitions: list) -> dict:
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
        report[name] = dict(
            n=len(ts), max_action=max_a, action_hist=dict(Counter(actions)), changed_rate=changed / len(ts)
        )
    return report


if __name__ == "__main__":
    import json

    all_t = generate_all(num_episodes=20)
    print(json.dumps(_sanity_check(all_t), indent=2))
