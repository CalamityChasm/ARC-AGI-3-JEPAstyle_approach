# Stage 6: scaled world-model pretraining (capacity + genuinely diverse data)

Testing a hypothesis distinct from anything tried in the single-day
held-out-game-generalization investigation documented in CLAUDE.md's
"Stage 6 addendum": that investigation's 10 interventions either changed
conditioning/architecture (7 of them) or added *modest-scale* data
diversity (MiniGrid/Sokoban already in production, plus MinAtar/Procgen at
~33-67k transitions each, all thematically "grid-navigation/arcade"). A
separate, much earlier capacity ablation (`stage6-capacity-sweep`, 1x/2x/4x
encoder width) found no benefit from more capacity alone, but only ever
tested that capacity against the *same small* data. Neither combination --
meaningfully more capacity *and* genuinely diverse data spanning dozens of
different game mechanics, including mechanics (turn-based, chance-driven,
capture/sowing) nothing else in this project's data has touched -- has
been tried. This experiment tests that combination directly.

Branched from `stage6-multifold-cv` (inherits the 5-fold held-out-games
partition and `--exclude-games` infra) as `stage6-scaled-world-model`.

Scope reality check, stated upfront and honored throughout: genuine
foundation-model-scale training (millions of frames, huge backbones) is
not realistic on a single RTX 2070 in any reasonable timeframe. The goal
is a real order-of-magnitude step up from today's attempts (dozens of
distinct games, corpora meaningfully larger than any single source tried
before), not an unbounded scale-up.

## Phase 1: a genuinely diverse pretraining roster

### OpenSpiel: checked Windows installability first, before designing anything else around it

Per this branch's own scoping instructions (OpenSpiel's Windows support
is historically weaker than Linux/Mac): `pip install open_spiel` was
tried *first*, standalone, before any other Phase 1 design work. Result:
**installed cleanly** via a prebuilt `open_spiel-2.0.1-cp313-cp313-win_amd64.whl`
wheel, zero build-from-source friction. The python-chess / hand-rolled
checkers+connect-four fallback specified as a contingency was not needed.

`pyspiel.registered_names()` lists 123 games. Selected six, each
mechanically distinct from the others and from every existing data
source, each getting its own `game_id` (this project's established
"one id per mechanically-dissimilar game" rule -- see CLAUDE.md's MinAtar
per-game-id retry, applied from the start here rather than re-discovered):

| game | mechanic | board/state | action encoding |
|---|---|---|---|
| `connect_four` | gravity placement | 6x7 | direct id (0-6, the column) |
| `tic_tac_toe` | simple placement | 3x3 | click (fixed id=6, xy=cell) |
| `othello` | placement + flip-capture | 8x8 | click (fixed id=6, xy=cell) |
| `checkers` | move + jump-capture | 8x8 | click (fixed id=6, xy=destination cell) |
| `pig` | push-your-luck dice/chance | scoreboard (synthetic) | direct id (0=roll, 1=hold) |
| `mancala` | seed sowing/distribution | 2x8 pits | direct id (0-5, relative pit index) |

**Games considered and dropped, with reasons (judgment calls made
autonomously, per this task's own instructions):**
- **backgammon** -- explicitly named in this task's brief as the
  chance/dice-like target, and it IS available in OpenSpiel. Dropped
  anyway after inspecting its action space directly: 1,352 distinct
  actions, each encoding a *combination* of several individual checker
  sub-moves, with no clean sub-move-level API to decompose into
  click-sized pieces the way checkers' `action_to_string` output could
  be parsed. `pig` was used instead as a pragmatic, much simpler
  push-your-luck dice game (2 actions: roll or hold) that still delivers
  genuine bust-risk/chance-under-uncertainty dynamics for a fraction of
  the implementation risk. This is a real scope-reduction, not a hidden
  substitution -- documented here rather than silently swapped.
- **gomoku, hex, y, havannah** -- all "place a stone, win by
  pattern-completion" games, mechanically redundant with tic_tac_toe/
  othello already in the roster (same placement action structure, larger
  boards). Skipped in favor of spending the same implementation budget on
  mechanically distinct games instead of a bigger board for an existing
  mechanic.
- **nine_mens_morris** -- has three distinct phases (place, then move,
  then "fly" once reduced to 3 pieces) needing per-phase action-encoding
  logic; deprioritized given time budget, in the same spirit as the
  task's own "deprioritize games whose action space genuinely can't be
  compressed into something reasonable without destroying the game's
  meaning" guidance -- this one *can* be compressed, just not cheaply
  enough to be worth it alongside six other games.

**Rendering approach, verified before relying on it (not assumed):**
for the four spatial games, `state.observation_tensor(0)` was directly
confirmed (by comparing its argmax-decoded board against each game's own
`board_string`/`str(state)` output) to be a one-hot-per-cell stack of
`(C, H, W)` planes -- `argmax` over the channel axis reconstructs the
board as a small `(H, W)` int grid, `C` values always well under
`NUM_COLORS=16`, with zero per-game-specific parsing needed. `pig` (no
spatial board at all -- its entire state is two running scores plus an
in-progress turn total) and `mancala` (a flat 16-value pit/store
seed-count vector, not planes) each get a small, honestly-synthetic grid
rendering instead (a 4-row scoreboard-bar-chart for pig, a 2x8
pit-count grid for mancala) -- documented as synthetic in
`jepa/data/openspiel_data.py`'s own docstrings, not presented as "real"
boards.

**Action-space budget** (`jepa/models/predictor.py: NUM_ACTIONS=8`,
shared across every source's action embedding -- see CLAUDE.md's Sokoban
gotcha for what an unchecked overflow costs): every game here uses either
a small direct action id or a single fixed `action_id=6` ("click",
matching ARC-3's own ACTION6 -- not load-bearing, just a readable
convention) with the real choice encoded in `(x, y)` instead. `mancala`
needed one real remapping: OpenSpiel assigns player 0's pits ids 1-6 and
player 1's 8-13 (confirmed by direct inspection, not assumed) -- remapped
via `(a - 1) % 7` to a shared 0-5 "my Nth pit from my own store" id,
losing which player's pit in the stored id (the board itself, which the
predictor also conditions on, already encodes whose turn it is) but
fitting the budget without an arbitrary truncation.

**Design choice: single click per move, not a two-step
select-then-destination pattern**, even for checkers (the one `from->to`
move game in the roster). OpenSpiel's checkers exposes moves -- including
multi-jump capture chains -- as one atomic `apply_action` call; there is
no sub-move granularity to actually step through, so a "select" transition
would have to be a fabricated, frame-unchanged, non-environment-driven
step. This project has consistently avoided injecting synthetic
non-environment transitions into training corpora (see e.g. how
`sokoban_data.py` drops the true no-op rather than fabricate one), so
checkers instead gets one real transition per move, clicked at the
*destination* cell (parsed from `action_to_string`'s algebraic notation,
e.g. `"a3b4"` -> destination `b4`, taking the last cell in longer
multi-jump strings like `"a3b4c5"`).

Verified with a real smoke test (`python -m jepa.data.openspiel_data`,
20 episodes x 40 steps per game) before any full-scale generation:
`max(action_id)` per game was 6 (connect_four/tic_tac_toe/othello/
checkers), 1 (pig), 5 (mancala) -- all comfortably under `NUM_ACTIONS=8`.
Frame-changed rate was 100% for every game (every legal move visibly
changes the board -- a real, useful property, mirroring MiniGrid's own
high-changed-rate advantage over ARC-3's sparser dynamics). Also visually
spot-checked several real before/after transitions per game (othello
flips, checkers captures, mancala sowing, pig's scoreboard bars) against
each game's own rules -- all matched expected mechanics.

### Hand-rolled: Snake and Pong

Per this branch's own instructions (no good off-the-shelf source for
either): `jepa/data/arcade_data.py`, two direct, deliberately simple
(a few dozen lines of state-update logic each) implementations --
genuinely different *physics/real-time-feel* mechanics (continuous-ish
motion, growth, self-collision, bounce physics) complementing
`openspiel_data.py`'s turn-based board-game mechanics and the existing
MiniGrid/Sokoban/MinAtar/Procgen sources' navigation/arcade mechanics.

- **Snake** (`game_id="snake"`, 14x14 board, 4 direct actions): random
  policy dies often (wall or self-collision), so episodes are short,
  high-changed-rate bursts. Smoke test: changed-rate 75.4% -- the
  remaining ~25% is the "died against a wall/self" case, where the
  attempted move is rejected before any state change (mirrors MiniGrid's
  own wall-bump no-op convention for invalid moves, an established
  pattern in this project, not a new one).
- **Pong** (`game_id="pong"`, 16x24 court, 3 direct actions --
  paddle up/down/stay): single paddle vs. three walls (a
  "squash/warm-up" simplification, not two-player -- documented as such
  in the module, not a hidden scope cut). Smoke test: changed-rate 100%
  (the ball always moves every step).

Both verified with `python -m jepa.data.arcade_data`: max action ids 3
and 2 respectively, both under `NUM_ACTIONS=8`.

### Final roster: 8 new games, 8 new `game_id`s

`connect_four, tic_tac_toe, othello, checkers, pig, mancala, snake, pong`
-- spanning gravity placement, simple placement, flip-capture, jump-
capture, push-your-luck chance, sowing/distribution, growth/self-collision,
and bounce physics. Combined with the existing MiniGrid (navigation) and
Sokoban (push-with-consequences) sources, that's 10 synthetic pretraining
sources total, a real step up in mechanical diversity from any prior
attempt in this project (which topped out at MiniGrid+Sokoban+one of
MinAtar/Procgen, 3-4 sources, all thematically arcade/navigation).

## Phase 1 infrastructure changes to `jepa/train_moe_predictor.py`

- Wired both new sources in as opt-in pretrain-phase additions
  (`--openspiel-episodes-per-game`, `--openspiel-steps-per-episode`,
  `--arcade-episodes-per-game`; default 0 = off, same convention as
  `--sokoban-episodes-per-config`), added to `synthetic_transitions`
  alongside MiniGrid/Sokoban -- the shared game vocabulary is still built
  generically from the union of every transition's own `game_id`, so no
  separate vocab-wiring code was needed per source.
- Ported `--width-mult` (encoder output channels + MoE expert hidden
  width, base 64 at 1.0x) from the `stage6-capacity-sweep` branch's
  `build_models` -- that branch predates the multi-fold CV infrastructure
  this branch needs, so the mechanism was ported into the current
  `train_moe_predictor.py` rather than merging the whole branch.
- Ported `--resume-from`/`--checkpoint-every` from
  `stage6-selfplay-bootstrap` -- given Phase 3's run here is expected to
  be the longest single training run this project has attempted, periodic
  mid-run checkpointing (bounding how much progress an interruption can
  cost) is worth having from the start rather than discovering the need
  mid-run.
- All three additions (new sources, `--width-mult`, `--resume-from`)
  smoke-tested individually and in combination (tiny episode counts, 1
  pretrain + 1 finetune epoch) before any full-scale run: confirmed
  correct game-vocab size (35 = 25 ARC + minigrid + sokoban + 6 OpenSpiel
  + 2 arcade), correct encoder-width scaling + warm-start-skip behavior
  at `--width-mult 2.0`, and correct checkpoint save/resume round-trip
  (`--resume-from` correctly skips pretrain, reuses the saved vocab, and
  loads both model state dicts).

## Worktree note: gitignored assets

This branch's worktree checkout (per its own standing instructions) has
no `checkpoints/`, `data/`, `logs/`, or `ARC-AGI-3-Agents/recordings/`
-- those are gitignored and only exist in the main checkout. Rather than
hardcode absolute main-checkout paths into every script invocation,
created Windows directory junctions from the worktree root to the main
checkout's copies (`checkpoints`, `data`, `logs`,
`ARC-AGI-3-Agents/recordings`, `ARC-AGI-3-Agents/environment_files`) plus
a direct copy of `ARC-AGI-3-Agents/.env` (small, gitignored, regenerable).
This lets every existing script's relative-path assumptions (`REPO_ROOT =
Path(__file__).resolve().parent.parent`, used throughout `jepa/`) work
unmodified. Training output for this experiment is written to an
explicit, separate `--out` directory, never the bare `checkpoints/`
default, so nothing here risks clobbering the production checkpoint
`checkpoints/moe_predictor.pt` actually used by the Kaggle submission.

(Phase 2, 3, and 4 sections to follow as this experiment progresses.)
