"""Procgen data source for Stage 6's diverse-pretraining generalization
test -- one of plan.md's *originally*-intended pretraining sources
(alongside MiniGrid/Sokoban/Crafter), never previously attempted. See
CLAUDE.md's "Stage 6 addendum" for the full context: 7 independent
conditioning/architecture fixes and one prior data-diversity attempt
(MinAtar, `jepa/data/minatar_data.py`) all failed to close the
held-out-ARC-games generalization gap. MinAtar's own 5 sub-games are
reflex/physics-arcade (paddle-and-ball, lane-crossing, shoot-em-up) --
a poor genre match for ARC-3's static puzzle-logic mechanics. Procgen's
`maze` (navigate a generated maze to a goal) and `heist` (multi-step
key/lock/gem puzzle logic, no combat) are a much closer genre match:
state-transformation/navigation puzzles, not reflexes -- the two
environments used here (`DEFAULT_ENVS`), not Procgen's full 16-game
suite, chosen specifically for that genre match rather than for breadth.

**Environment/dependency note, read before trying to import this module
in the main project venv:** Procgen (`pip install procgen`) has no PyPI
wheel for Python 3.13 (this project's main venv, see CLAUDE.md's
Environment setup) -- the newest available wheels (procgen 0.10.7) top
out at `cp310` (confirmed directly against PyPI's file listing, not
assumed), and building from source is a heavy, fragile C++/OpenGL-adjacent
build (`gym3`/`moderngl`-based) not worth taking on for one data source.
Rather than downgrading the whole project's Python version (every other
pin in requirements.txt is validated against 3.13), a **dedicated Python
3.10 venv** (`C:\\pgvenv` on the dev box this was built on -- see
`scripts/generate_procgen_corpus.py`'s own docstring for the exact setup
steps) is used *only* to run this module's `generate_transitions()` once
and serialize its output to a cache file
(`data/procgen_corpus_<env>.pkl`, gitignored under the existing `/data/`
pattern). `jepa/train_moe_predictor.py`'s `--procgen-episodes-per-env`
flag then loads that cache directly via `load_cached_transitions()`,
which is pure-numpy/pickle and needs no `procgen` import at all -- so the
main training venv never needs Procgen installed. This mirrors Stage 3's
documented Mamba->GRU substitution: work around a genuine, unresolvable
local environment incompatibility, as a deliberate, documented deviation,
not silently. Because of this, the `from procgen import ...` used by
`generate_transitions()`/`sample_palette_frames()` below is a *deferred*
(function-local, not module-level) import, unlike every sibling module in
this package (`minigrid_data.py`/`sokoban_data.py`/`minatar_data.py`,
which all guard a module-level import instead) -- deliberately, so this
module still imports cleanly (translation/quantization helpers and
`load_cached_transitions()` all work) in the main venv where `procgen`
itself is genuinely never installable, not just not-yet-installed.

**Observation format, and why this source needs real quantization unlike
every sibling module.** Procgen's native observation is raw RGB,
`(64, 64, 3)` uint8 by default (confirmed directly via
`ProcgenGym3Env(...).ob_space`) -- MiniGrid/Sokoban/MinAtar all expose an
already-categorical/semantic grid (`object_idx`, `room_state` cell code,
per-channel boolean plane) that a simple fixed dict lookup translates 1:1
into a color index. Procgen has no such structure available -- it's
pixels, the same way a real Kaggle ARC-3 game's `frame` is *not* (ARC-3
frames are already a flat 0-15 color-index grid). So this module fits a
**16-color palette via a small, dependency-free numpy k-means** (`k =
NUM_COLORS = 16`, `fit_palette()` below -- no new `scikit-learn`
dependency needed anywhere in this pipeline, since the palette is fit
once at corpus-generation time in the dedicated procgen venv and every
downstream consumer, including the main training venv, only ever sees
already-quantized integer grids, never raw RGB or the palette itself) on
a small sample of frames pooled *across* both `maze` and `heist`
(`sample_palette_frames()`), then does nearest-color (Euclidean, RGB
space) bucketing per pixel (`_translate_frame()`) using that one shared
palette for every frame this module ever generates. One shared palette
(not one per env) was chosen deliberately, mirroring MiniGrid's own
"one consistent color/action vocabulary across sub-environments" choice
-- game_id already disambiguates action *semantics* per environment (see
below), so the color vocabulary doesn't also need to be split, and a
shared palette means a given color index carries roughly the same visual
meaning (e.g. "wall-like dark tone" vs. "floor-like light tone") across
both environments rather than two unrelated 16-color codebooks.

**Action space.** Procgen's raw action space is `Discrete(15)` for every
game (confirmed via `ProcgenGym3Env(...).ac_space`) -- more than
`jepa/models/predictor.py`'s shared `NUM_ACTIONS=8` (see CLAUDE.md's
Sokoban gotcha: a raw action id >= `NUM_ACTIONS` doesn't fail at
data-generation time, only much later inside a confusing CUDA assert on
whatever training batch first samples it -- checked explicitly here,
before this module was ever wired into training). Procgen's own
`BaseProcgenEnv.get_combos()` (read directly from the installed package
source, not assumed/guessed) shows action ids 0-8 form a 3x3 grid of
directional movement combos (`("LEFT","DOWN")`, `("LEFT",)`,
`("LEFT","UP")`, `("DOWN",)`, `()` -- true no-op, `("UP",)`,
`("RIGHT","DOWN")`, `("RIGHT",)`, `("RIGHT","UP")`) and ids 9-14 are
single special-purpose buttons (`D`/`A`/`W`/`S`/`Q`/`E` -- used by other
Procgen games for jumping/shooting/interacting; `maze` and `heist` are
pure walk-into-it navigation games with no combat or jump mechanic, so
these buttons do nothing meaningful in either). `MOVEMENT_COMBOS` below
therefore drops just the true no-op (index 4) -- exactly the same call
Sokoban's own action-remap made ("nothing changes" is already the
trivial baseline everywhere else in this pipeline's training/eval, so a
literal no-op action carries no extra information) -- leaving exactly the
8 real movement combos, a clean, principled fit under `NUM_ACTIONS=8`
with no arbitrary truncation of anything mechanically relevant to these
two games. `env.act()` is still called with the *original*, unshifted
Procgen action id (`MOVEMENT_COMBOS[stored_id]`); only the id written
into the transition tuple is remapped to `0..7`, mirroring
`sokoban_data.py`'s exact convention.

**game_id scheme: one id per environment, not one shared id for all of
Procgen.** Unlike MiniGrid's 21 environments (all sharing one consistent
"turn and move forward" action interface -- see minigrid_data.py's
docstring) or MinAtar's 5 games (literally identical 6-action interface
across all of them -- see minatar_data.py's docstring), `maze` and
`heist` do *not* share a consistent effect-of-action-id mapping in any
deeper sense than "both use the same 3x3 directional-movement combo
grid" -- `heist` additionally has real state-dependent interaction
(walking into a key/lock/gem changes what a given cell means to touch
next) that `maze` doesn't have at all. Rather than force one embedding to
serve two mechanically different games (repeating the exact confound
Stage 1 originally worried about -- and did *not* find to be the
dominant factor -- for ARC-3's 25 games sharing one action vocabulary),
each environment gets its own id (`GAME_IDS`), consistent with how
`sokoban_data.py` justified its own separate id from `"minigrid"`.
"""

import random

import numpy as np

from ..grid import NUM_COLORS

GAME_IDS = {"maze": "procgen_maze", "heist": "procgen_heist"}
DEFAULT_ENVS = ["maze", "heist"]

# Procgen action ids 0-8 are the 3x3 movement-combo grid; 4 is the true
# no-op. Dropping it leaves exactly 8 real movement actions -- see module
# docstring for the full reasoning (verified directly against procgen's
# own BaseProcgenEnv.get_combos(), not assumed).
MOVEMENT_COMBOS = [0, 1, 2, 3, 5, 6, 7, 8]
assert len(MOVEMENT_COMBOS) == 8  # matches jepa.models.predictor.NUM_ACTIONS


def fit_palette(frames_rgb: np.ndarray, k: int = NUM_COLORS, iters: int = 20, seed: int = 0) -> np.ndarray:
    """(N, H, W, 3) uint8 frames -> (k, 3) float32 cluster centers, via a
    plain numpy Lloyd's-algorithm k-means (no scikit-learn dependency --
    see module docstring for why that's deliberate). Subsamples to at most
    200k pixels for speed if given more than that; `iters=20` was enough
    for centers to visibly stop moving on a quick manual check during
    development (not formally convergence-tested further, since the exact
    palette isn't the object of the experiment -- just needs to be a
    reasonable, reproducible 16-color summary of Procgen's actual on-
    screen palette).
    """
    rng = np.random.RandomState(seed)
    pixels = frames_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 200_000:
        idx = rng.choice(len(pixels), 200_000, replace=False)
        pixels = pixels[idx]
    init_idx = rng.choice(len(pixels), k, replace=False)
    centers = pixels[init_idx].copy()
    for _ in range(iters):
        dists = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dists.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = pixels[mask].mean(axis=0)
        centers = new_centers
    return centers


def _translate_frame(rgb: np.ndarray, palette: np.ndarray) -> list:
    """(H, W, 3) uint8 raw Procgen frame + (k, 3) float32 palette -> `[grid]`,
    the same one-element-list "single layer" convention
    `arc3_frame_to_tensor`/`patch_change_mask` expect (they index
    `frame[0]`), matching every sibling translation-layer module's own
    wrapping. Nearest-color (Euclidean, RGB space) bucketing against the
    fixed palette -- see module docstring."""
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    idx = dists.argmin(axis=1).reshape(h, w).astype(np.int64)
    return [idx.tolist()]


def sample_palette_frames(envs: list | None = None, n_frames_per_env: int = 400, seed: int = 0) -> np.ndarray:
    """Short random-policy rollout across `envs`, purely to collect a
    representative sample of raw RGB frames for `fit_palette()`. Deferred
    `procgen` import (see module docstring) -- only callable from inside
    the dedicated procgen venv."""
    from procgen import ProcgenGym3Env

    envs = envs or DEFAULT_ENVS
    rng = random.Random(seed)
    frames = []
    for env_name in envs:
        env = ProcgenGym3Env(
            num=1, env_name=env_name, num_levels=0, start_level=seed, rand_seed=seed,
            distribution_mode="easy", use_backgrounds=False, restrict_themes=True,
        )
        for _step in range(n_frames_per_env):
            _rew, obs, _first = env.observe()
            frames.append(obs["rgb"][0].copy())
            action = np.array([MOVEMENT_COMBOS[rng.randrange(8)]], dtype=np.int32)
            env.act(action)
        env.close()
    return np.stack(frames, axis=0)


def generate_transitions(
    envs: list | None = None,
    steps_per_env: int = 33_600,
    num_parallel: int = 8,
    palette: np.ndarray | None = None,
    seed: int = 0,
) -> list:
    """Random-policy rollouts across `envs` (default: `maze`, `heist`),
    returned as `(frame_t, action_id, x, y, frame_t1, changed, game_id)`
    tuples -- the same shape `jepa/data/trajectories.py`'s
    `TransitionDataset` expects. `x, y` are always 0 (Procgen has no
    coordinate-based action, same as every sibling synthetic source).
    Deferred `procgen` import (see module docstring).

    `steps_per_env=33_600` * 2 default envs = 67,200 total transitions,
    matching MiniGrid's own total budget (`21 envs * 40 episodes * 80
    steps = 67,200`) and mirroring `minatar_data.py`'s explicit "similar
    total data budget, different mechanics" reasoning for its own episode
    count -- except here that budget is concentrated into just 2
    environments rather than spread across 21 (MiniGrid) or 5 (MinAtar),
    so the model sees much deeper experience within `maze`/`heist`
    specifically rather than broad-but-shallow coverage. Worth being
    aware of as a real difference in this ablation's data shape, not
    just its content, when interpreting results.

    No explicit per-episode `reset()` call: Procgen's `gym3` interface
    (`ProcgenGym3Env`) has no `reset()` method at all (confirmed directly
    -- `[m for m in dir(ProcgenGym3Env) if not m.startswith('_')]` has no
    `reset`) -- episodes auto-reset internally, signaled via the `first`
    flag `env.observe()` returns (True on the first observation of a new
    episode). This module doesn't need to track episode boundaries at all
    for the i.i.d.-shuffled-transitions training this feeds into (mirrors
    how `jepa/train_moe_predictor.py` already treats every synthetic
    source), so `first` is intentionally unused here, not overlooked.

    `num_parallel=8`: Procgen's C++ backend runs many sub-envs genuinely
    in parallel within one process (`ProcgenGym3Env(num=N, ...)`), unlike
    MiniGrid/Sokoban/MinAtar's plain-Python single-env loops -- using it
    materially speeds up wall-clock generation. `env.act()`/`env.observe()`
    still process all `num_parallel` sub-envs in lockstep per call; the
    per-worker step count is `steps_per_env // num_parallel`, so
    `steps_per_env` should divide evenly for an exact transition count
    (33_600 / 8 = 4_200, no remainder, with the defaults).

    `distribution_mode="easy"`: simpler level geometry (mirrors this
    project's general preference for pretraining data a random policy can
    meaningfully explore, not a curated/optimal-play corpus). `use_
    backgrounds=False, restrict_themes=True`: reduces visual variety/
    clutter that has nothing to do with the maze/heist *mechanics* this
    source is meant to teach, and makes the fixed 16-color quantization
    more faithful (less unrelated background-art color competing with
    wall/floor/agent/goal/key/lock colors for palette slots).
    """
    from procgen import ProcgenGym3Env

    envs = envs or DEFAULT_ENVS
    if palette is None:
        palette = fit_palette(sample_palette_frames(envs, seed=seed), seed=seed)

    transitions = []
    for env_name in envs:
        game_id = GAME_IDS[env_name]
        env = ProcgenGym3Env(
            num=num_parallel, env_name=env_name, num_levels=0, start_level=seed, rand_seed=seed,
            distribution_mode="easy", use_backgrounds=False, restrict_themes=True,
        )
        rng = random.Random(seed * 1_000_000 + (hash(env_name) % 10_000))

        _rew, obs, _first = env.observe()
        cur_frames = [_translate_frame(obs["rgb"][i], palette) for i in range(num_parallel)]

        steps_per_worker = steps_per_env // num_parallel
        for _step in range(steps_per_worker):
            stored_actions = [rng.randrange(8) for _ in range(num_parallel)]
            real_actions = np.array([MOVEMENT_COMBOS[a] for a in stored_actions], dtype=np.int32)
            env.act(real_actions)
            _rew, obs, _first = env.observe()
            next_frames = [_translate_frame(obs["rgb"][i], palette) for i in range(num_parallel)]

            for i in range(num_parallel):
                changed = cur_frames[i] != next_frames[i]
                transitions.append(
                    (cur_frames[i], stored_actions[i], 0, 0, next_frames[i], changed, game_id)
                )
            cur_frames = next_frames
        env.close()
    return transitions


def load_cached_transitions(cache_path) -> list:
    """Load a transitions list pickled by `scripts/generate_procgen_corpus.py`
    (run once, from inside the dedicated procgen venv). Pure pickle/list --
    no `procgen` import needed, so this is safe to call from the main
    (Python 3.13) training venv, unlike `generate_transitions()`/
    `sample_palette_frames()` above."""
    import pickle
    from pathlib import Path

    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} does not exist -- generate it first via "
            f"`C:\\pgvenv\\Scripts\\python.exe scripts\\generate_procgen_corpus.py` "
            f"(see that script's docstring for the dedicated-venv setup steps; "
            f"procgen has no wheel for this project's main Python version)."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)
