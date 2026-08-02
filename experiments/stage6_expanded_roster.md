# Stage 6: expanded OpenSpiel roster + capacity retest (stage6-expanded-roster)

Direct follow-up to `experiments/stage6_scaled_world_model.md`
(`stage6-scaled-world-model`), which combined ~358k synthetic pretraining
transitions (6 OpenSpiel board games + hand-rolled Snake/Pong + existing
MiniGrid) with up to 2x model capacity, and found neither diversity alone
nor diversity+capacity closed the held-out-ARC-games generalization gap
documented in CLAUDE.md's Stage 6 addendum -- width=1.0 was a clean null,
width=2.0 was unstable across folds (a severe -88.29% regression in fold 1
driven by capacity-enabled overfitting vs. near-neutral -0.03% in fold 2).

This experiment tests the next lever the prior write-up explicitly left
open: **genuinely more OpenSpiel diversity** (the prior attempt used only
6 of OpenSpiel's 123 registered games) **paired with a properly-scaled
pretrain corpus**, before re-testing capacity. Branched from
`stage6-scaled-world-model` (inherits the OpenSpiel/arcade translation
layers, `--width-mult` capacity scaling, and the 5-fold held-out-games
partition from `stage6-multifold-cv`) as `stage6-expanded-roster`.

## Part 1: enumerating and filtering OpenSpiel's full game catalog

`pyspiel.registered_names()` lists 123 registered games (up from the 6
used previously). Programmatically categorized all 123 by `game.get_type()`
(dynamics/information), not by hand-inspection, before deciding anything:

| category | count | disposition |
|---|---|---|
| `SEQUENTIAL` + `PERFECT_INFORMATION` | 42 | the eligible candidate pool |
| `SIMULTANEOUS` dynamics (incl. one-shot matrix games) | 20 | excluded -- this project's translation layer assumes one actor moves at a time, matching how ARC-3's own turns work; a simultaneous-move game has no single well-defined "current player's action" to log as one training transition |
| `IMPERFECT_INFORMATION` (hidden/private state -- card games, hidden-piece games, auctions) | 46 | excluded -- can't render a single, honest, fully-observable grid for a state one player doesn't fully see; rendering "what the current player can see" would silently mix in information-hiding semantics no ARC-3 game has |
| `MEAN_FIELD` dynamics (population-level, not a single agent) | 4 | excluded -- doesn't fit this module's single-episode-single-agent-under-random-policy generation loop at all |
| fails to load without extra required constructor params (`add_noise`, `cached_tree`, `efg_game`, `misere`, `nfg_game`, `normal_form_extensive_game`, `repeated_game`, `restricted_nash_response`, `start_at`, `turn_based_simultaneous_game`, `zerosum`) | 11 | excluded -- these are meta-games/wrappers around another game, not directly playable on their own |

That leaves 42 candidates. The original roster used 6 of them
(`connect_four`, `tic_tac_toe`, `othello`, `checkers`, `pig`, `mancala`).
This experiment adds **20 more**, for **26 OpenSpiel games total** (a
genuine "dozens more" expansion, not just a handful). The remaining 16 of
the 42 were individually excluded, each for a specific, checked reason
(not guessed) -- grouped by why, not restated one-by-one at length:

- **Combined/multi-part action spaces that don't reduce to a single click
  or small direct id without fabricating a sub-step this project has
  consistently avoided** (see `checkers`'/`openspiel_data.py`'s own
  documented reasoning for why a synthetic "select" transition was
  rejected there): `amazons` (piece-move + arrow-placement combined into
  one action), `pentago` (placement + board-quadrant rotation combined),
  `backgammon` (1,352 combined-sub-move actions -- already dropped in the
  original 6-game roster for the same reason, carried over unchanged).
- **No natural spatial board, and a large enough action space that a
  small direct id doesn't fit either** (would need a bespoke synthetic
  renderer, and the mechanic is already reasonably covered by `pig`'s
  chance/push-your-luck or `oware`'s sowing/counting, already in the
  roster): `nim` (29 actions, pile-counting), `yacht` (44 actions,
  dice/scoring categories).
- **Multi-phase rules needing per-phase action-encoding logic**:
  `nine_mens_morris` (place, then move, then "fly" once reduced to 3
  pieces -- excluded in the *original* 6-game roster for exactly this
  reason, carried over unchanged, not re-litigated).
- **Board topology/observation shape that doesn't fit this module's
  square-grid, small-one-hot-plane-count conventions without materially
  more per-game engineering**: `chinese_checkers` (hexagonal board, not a
  square grid), `hive` (irregular tile-adjacency board, large 5,489-action
  space), `ultimate_tic_tac_toe` (its 254-element `observation_tensor`
  doesn't factor into any clean `(C, H, W)` one-hot-plane shape the way
  every other placement game here does).
- **Flagged or unimplemented by OpenSpiel itself**: `quoridor` (prints
  `Warning! The implementation of 'quoridor' has known issues` on load --
  excluded on the grounds that pretraining on a self-flagged-buggy
  implementation isn't worth the risk), `morpion_solitaire`
  (`observation_tensor` is literally unimplemented --
  `ObservationTensorShape unimplemented`, a hard exclusion not a
  judgment call).
- **Lower marginal value for the implementation cost, given time budget**:
  `cursor_go` (a niche cursor-based action encoding on top of Go, redundant
  with `go9` already in the roster), `einstein_wurfelt_nicht` (dice+chess
  hybrid movement, 300 actions, not distinctly different enough from
  `pig`'s chance mechanic + `breakthrough`/`chess`'s movement mechanics
  already covered), `banqi` (Chinese dark chess -- formally
  `PERFECT_INFORMATION` in pyspiel's own typing since flips are chance
  events visible to both players, but the flip/reveal mechanic and
  1,056-action space add real complexity for limited marginal novelty
  over `checkers`/`chess` already included), `maedn` (Ludo-style race
  game -- observed only 1 legal action in a sampled mid-game state, i.e.
  very low branching/"forced move" dynamics under random play, low
  information value), `shogi` (Japanese chess variant whose move-string
  notation is a third, different convention from the "algebraic squares"
  regex the chess-family handler below already reuses across
  `chess`/`antichess`/`crazyhouse`/`xiangqi` -- would need its own parser
  for a family already well-represented).

## Part 2: three generic, reusable handlers instead of 20 bespoke ones

Rather than hand-write a bespoke render+action-pick function pair for
each of the 20 new games (practical for the original 6, not for 20+),
`jepa/data/openspiel_data.py` now has three generic handler families, each
verified against real `get_parameters()`/`observation_tensor` output
before being trusted (not guessed) -- see the module's own docstring for
the full design writeup:

1. **Cell-index placement** (`_make_pick_cell(width, height)`): games
   where the action id already IS a board cell in row-major order
   (`x, y = divmod(a, width)`). Covers `tic_tac_toe`/`othello` (existing,
   unchanged) plus new: `gomoku` (15x15), `hex` (11x11), `y` (19x19),
   `havannah` (15x15), `mnk` (15x15), `twixt` (8x8), `go9` (Go loaded as
   `go(board_size=9)`, not OpenSpiel's 19x19 default, to keep board size
   and episode length comparable to the rest of the roster).
2. **Destination-click via move-string parsing**
   (`_make_pick_destination_algebraic`/`_make_pick_destination_numeric`):
   generalizes `checkers`' existing hardcoded regex (a `from -> to` move
   game with no clean sub-move granularity to step through) into two
   reusable factories. Algebraic (`[a-h]<digit>` squares): `breakthrough`
   (8x8), `clobber` (6x5), `lines_of_action` (8x8), `chess` (8x8),
   `antichess` (8x8), `crazyhouse` (8x8). Numeric-parenthesized
   (`xiangqi`'s own `"(row,col)-(row,col)"` move-string convention, a
   different board shape too -- 9 files x 10 ranks): `xiangqi`. Both take
   the *last* coordinate pair as the destination, matching `checkers`'
   existing multi-jump-chain convention.
3. **Direct small action id** (no xy, action id used as-is): `pig`/
   `mancala` (existing, unchanged) plus new: `oware` (6 actions, no remap
   needed -- verified both players already get ids 0-5 directly, unlike
   `mancala`'s cross-player offset), `2048` (4), `cliff_walking` (4),
   `catch` (3), `stones_and_gems` (5, a Boulder-Dash-style grid game --
   direct id despite having a real spatial board, since its action space
   is small enough to skip the xy detour entirely).

Plus one edge-click variant (`dots_and_boxes`, its own `"P1(h,row,col)"`
move-string format) and one bespoke renderer (`2048`'s
`log2(tile_value) -> color`, since raw tile values like 2048 obviously
exceed `NUM_COLORS=16`).

**`_render_spatial` (the shared one-hot-per-cell `observation_tensor`
reshape) now clips to `NUM_COLORS-1` instead of hard-asserting.** The
original 6 games all happened to have fewer one-hot channels than
`NUM_COLORS` allows; several new games don't (`chess` C=20, `crazyhouse`
C=38) since they encode more piece/state types per cell. Clipping
collapses rarer high-index channels into the top color bucket -- a
documented lossy-but-bounded approximation (same spirit as `mancala`'s
own pre-existing seed-count clipping), not a crash on a game whose
channel count nobody happened to check in advance.

**Verified with a real smoke test (`python -m jepa.data.openspiel_data`,
20 episodes x 40 steps per game) before any full-scale generation, exactly
the discipline the original 6-game module established**: all 26 games
load and generate without error, every game's `max(action_id) < NUM_ACTIONS
= 8` (the hard requirement -- see CLAUDE.md's Sokoban gotcha for what an
unchecked overflow costs), and every game shows a healthy changed-rate
(72%-100%, comparable to the original roster's 100%-across-the-board).
Also spot-checked several games' actual rendered frames directly (not
just the sanity-check's aggregate stats) -- `chess`/`xiangqi` boards show
dense, varied piece-type values (0-14, well within bounds), `2048`'s log2
mapping correctly recovers small integer exponents from real tile values,
`stones_and_gems`' 12x20 board renders with 99.6% non-empty cells (a real,
detailed level layout, not a degenerate near-blank grid).

## Part 3: corpus generation -- sizing and a real infrastructure bug found along the way

**Sizing.** Standalone benchmarking (`generate_all(num_episodes=100, ...)`
across all 26 games, summed) showed generation itself is cheap --
combined generation rate for the whole 26-game roster is on the order of
tens of thousands of transitions/second for most games, down to ~2,000-
3,000/s for the slowest (`stones_and_gems`, `mnk`). A full
`generate_all(num_episodes=3000, steps_per_episode=60)` run (all 26
games) produced **4,586,997 transitions in ~431s (~7.2 minutes)** in a
standalone process -- confirming both that "low-to-mid millions" is easily
reachable computationally, and giving a directly-measured basis (rather
than a guess) for picking a target size.

Chose **`--openspiel-episodes-per-game 1300 --openspiel-steps-per-episode
60`** (~2.03M OpenSpiel transitions, confirmed directly:
**1,987,634 across 26 games** in the real training run) + the existing
MiniGrid recipe unchanged (67,200) + the existing arcade recipe unchanged
(`--arcade-episodes-per-game 450`, 81,000) = **~2.14M total pretrain
transitions**, a genuine ~6x step up from the prior experiment's 358k, and
comfortably within CLAUDE.md's own "low-to-mid millions" target.

**Pretrain epoch count, following the established curriculum-balance
discipline** (CLAUDE.md's diagnosed Procgen curriculum-imbalance mechanism,
and `stage6-scaled-world-model`'s own `--pretrain-epochs 4` reasoning on
its smaller corpus): the established known-good recipe is 20 epochs on
67,200 MiniGrid transitions (~1.34M samples-seen). At ~2.14M transitions,
1 epoch alone is already ~1.6x that reference samples-seen -- close enough
to the established band that `--pretrain-epochs 1` was used rather than
compounding a bigger corpus with unchanged/scaled-up epochs the way the
original Procgen mistake did.

**Real infrastructure bug found and fixed before any full run completed:
Windows `DataLoader` worker spawn cannot handle a corpus this large.**
With the default `num_workers=4` (or even `1`) on CUDA, `_make_loaders`'s
`DataLoader` spawns a subprocess per worker (Windows `spawn`, not `fork`)
that needs to pickle-transfer the *entire* dataset object -- for a
~2.14M-transition list of nested Python lists (each transition holding
two small grids as lists-of-lists), this pickling is catastrophically
slow: a first attempt (`--openspiel-episodes-per-game 1300`,
`JEPA_NUM_WORKERS=1`) showed the worker subprocess consuming 1,200-1,900+
CPU-seconds without ever completing its first batch, confirmed via
`Get-CimInstance Win32_Process` showing the worker's `multiprocessing.spawn`
bootstrap process as a distinct, actively-running child. **Fixed by
setting `JEPA_NUM_WORKERS=0`** (the existing env-var override in
`train_moe_predictor.py`, originally added for a different reason -- a
`MemoryError`-on-a-contended-machine gotcha, see CLAUDE.md) for this run's
pretrain phase, which avoids spawning any worker subprocess at all (no
pickling of the big list ever happens; `__getitem__` conversion runs
directly in the main process instead). This is a new, general lesson
worth adding to CLAUDE.md's own gotcha list: **the existing
`num_workers=4` default (proven fine on this project's prior ~55k-
transition corpora) does not scale to a multi-million-transition Python-
list-of-lists dataset on Windows -- past roughly the hundreds-of-
thousands-of-transitions range, `JEPA_NUM_WORKERS=0` is the safer default
for any future corpus this large, not a slower fallback to reach for only
after something else fails.**

**A second real infrastructure lesson, worth recording for future sessions
at this corpus scale: `num_workers>0` isn't simply "faster," it's
unusable at this scale on Windows, and the safe `num_workers=0`
fallback's own cost is real and must be budgeted for, not assumed away.**
Directly measured (a standalone throughput script instrumenting the exact
training-step code `_run_epochs` runs, 2,000 real forward+backward+
optimizer-step iterations): single-threaded `__getitem__` throughput
stabilizes around **~1,090 samples/s**. At `n_train=1,922,251` for this
corpus, that's **~29 minutes for ONE pretrain epoch alone** (`_run_epochs`
prints nothing until the *entire* epoch, train+val, completes -- so a
process that's silently "not printing" for 20-40 minutes during this
phase is very likely working normally, not stuck; a fixed CPU-time-based
"is it stuck" heuristic checked too early produced two false-positive
"hang" diagnoses during this session before a longer, log-growth-only
patience threshold and a direct throughput measurement resolved it).
`--pretrain-epochs 1` (already the minimum whole-epoch count, chosen for
curriculum-balance reasons in Part 3) keeps this bounded to a single
~29-minute pass rather than compounding it.

## Part 4: capacity retest results

### Fold 1, width=1.0 (diversity alone, no added capacity): real run, real negative result

Full run (`--pretrain-epochs 1 --epochs 60 --width-mult 1.0`,
`JEPA_NUM_WORKERS=0`) completed cleanly end to end -- pretrain phase (1
epoch) then 60 ARC-finetune epochs, `tag=final` checkpoint saved to
`checkpoints_expanded_fold1_w1/`. Total wall-clock ~78 minutes (within
this project's usual per-run budget, despite the much bigger pretrain
corpus -- the curriculum-balanced single pretrain epoch keeps the added
cost bounded as intended). Finetune-phase training log shows normal,
non-collapsed learning throughout (e.g. epoch 60: `train_loss=0.0019`,
`val_pred_mse` and `val_identity_mse` moving together in a healthy
pattern, not a Procgen-style collapse to equal near-zero values).

**Held-out-games result (fold 1, `scripts/eval_scaled_world_model.py`,
2,400 held-out transitions across `r11l, bp35, m0r0, tr87, ka59`):**

| game | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| r11l | 0.000500 | 0.000442 | -13.04% |
| bp35 | 0.024269 | 0.024014 | -1.06% |
| m0r0 | 0.013541 | 0.013405 | -1.02% |
| tr87 | 0.001134 | 0.001111 | -2.04% |
| ka59 | 0.000707 | 0.000697 | -1.47% |
| **overall** | **0.009037** | **0.008928** | **-1.22%** |

**-1.22% overall -- every one of the 5 held-out games individually
negative, not a mixed picture.** `r11l`'s -13.04% looks dramatic in
isolation but is the small-absolute-denominator artifact CLAUDE.md
warns about (checked directly, per this project's standing practice):
identity=0.000442 is tiny, and the absolute gap (0.000058) is small in
its own right -- not a real large-magnitude effect, just a big percentage
on a small base. This result sits at or slightly below the established
5-fold no-diversity baseline band (-0.30% +/- 0.66%) and is *worse* than
the prior, much-less-diverse `stage6-scaled-world-model` fold-1 width=1.0
result (+0.10%, 358k pretrain transitions, 6 OpenSpiel games) -- **~5.3x
more OpenSpiel games and ~6x more total pretrain transitions did not
move this number in a positive direction; if anything it's very slightly
worse, well within noise of "no effect."**

**Trained-games sanity check** (9,600 transitions across the fold's other
20 games): pred=0.001932, identity=0.002366, **+18.34% improvement** --
healthy, clearly positive, comparable in kind to every prior production-
style recipe's own trained-games numbers -- confirms the model learned
real, non-degenerate dynamics from the vastly expanded pretrain corpus.
The gap specifically fails to transfer to genuinely unseen ARC games,
exactly the pattern every one of CLAUDE.md's prior 10 interventions
already established -- this is the 11th consistent with that pattern
(the 10 already documented, now an 11th independent negative result: more
OpenSpiel diversity alone, without added capacity, doesn't close it
either).

### Fold 1, width=2.0 (diversity + capacity together, the main event)

Same recipe as width=1.0 above, `--width-mult 2.0` (128-channel encoder +
128-channel MoE expert hidden width, fresh-initialized -- a width-mult
run always skips the `--encoder` warm-start, confirmed in the log:
`width_mult=2.0 != 1.0: skipping warm-start ... (state dict shape
wouldn't match a 128-channel encoder)`). Ran to completion cleanly
(~80 minutes wall-clock, directly confirmed via a real blocking process-
exit check, not assumed -- no crash, all 60 finetune epochs completed,
`tag=final` checkpoint saved to `checkpoints_expanded_fold1_w2/`).

**Held-out-games result (fold 1, same 2,400 held-out transitions as the
width=1.0 run above):**

| game | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| r11l | 0.000759 | 0.000756 | -0.37% |
| bp35 | 0.031102 | 0.031117 | +0.05% |
| m0r0 | 0.005745 | 0.005744 | -0.03% |
| tr87 | 0.001992 | 0.001989 | -0.14% |
| ka59 | 0.007590 | 0.007589 | -0.02% |
| **overall** | **0.010679** | **0.010681** | **+0.02%** |

**+0.02% overall -- essentially exact parity, no meaningful improvement
but also, critically, no regression.** Every one of the 5 held-out games
individually sits within +/-0.4% of identity, none showing anything
resembling a large swing in either direction.

**This is a materially different, and genuinely important, result
compared to `stage6-scaled-world-model`'s own fold-1 width=2.0 finding
(-88.29%, a severe capacity-enabled-overfitting collapse where every
held-out game got dramatically worse, especially `m0r0` at ~11x its own
identity baseline).** Same fold, same held-out games, same width
multiplier and architecture -- the only thing that changed is the
pretrain corpus: ~6x more transitions (2.15M vs. 358k) spanning 26
OpenSpiel games instead of 6. **The catastrophic instability is gone.**

**Trained-games sanity check**: pred=0.000284, identity=0.000301,
**+5.75%** -- clearly positive (the model learned real dynamics) but
notably *weaker* than this same fold's width=1.0 result (+18.34%) and
far weaker than `stage6-scaled-world-model`'s own width=2.0 result on the
smaller corpus (+69.11%). This is the flip side of the stability finding:
width=2.0 here isn't achieving the dramatic training-distribution overfit
that drove the old experiment's regression, consistent with -- not
contradicting -- the "more data limits capacity's ability to memorize the
training set" interpretation. A model that can't overfit as hard also
doesn't show the corresponding held-out collapse.

**Read together, fold 1's two results tell a coherent, specific story:**
more data didn't make width=2.0 *better* than width=1.0 on held-out
games (+0.02% vs. -1.22% -- both are non-improvements, arguably +0.02%
looks marginally less bad, but neither is a real positive effect on a
metric with this much game-to-game noise) -- but it did make width=2.0
demonstrably *safer*, removing the severe regression risk the smaller-
corpus experiment found. That's a real, useful finding for anyone
considering capacity scaling in this project going forward, even though
it doesn't answer this experiment's headline question (does the gap
close?) with a "yes."

### Fold 2, width=2.0 (second-fold validation, matching this project's standing multi-fold discipline)

Given `stage6-scaled-world-model`'s own fold-1-vs-fold-2 divergence was
the whole reason a second fold mattered there (fold 2 showed near-neutral
results where fold 1 showed catastrophic collapse), and given fold 1's
result here is the *opposite* of that prior experiment's fold-1 finding,
a second fold is the direct way to check whether *this* fold-1 result
(stability, no regression) is itself representative or another
fold-specific artifact -- exactly the discipline `stage6-multifold-cv`
established. Same recipe, fold 2's held-out games (`ar25, cd82, cn04,
dc22, ft09`, matching `stage6-multifold-cv`'s partition), `--width-mult
2.0`.

(TBD -- launching next.)

## Overall verdict

(TBD -- pending fold 2 result.)
