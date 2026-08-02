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

## Phase 2 + 3: capacity x diverse data, fold 1

Trained via `jepa/train_moe_predictor.py`, fold 1's exact held-out games
(`r11l,bp35,m0r0,tr87,ka59`, matching `stage6-multifold-cv`'s partition)
and exact ARC-finetune-phase settings as the established baseline recipe
(`--epochs 60 --external-per-game 2000`, warm-started from
`checkpoints/encoder.pt`) -- the only two things that differ from that
baseline are (a) the pretrain phase now uses the full diverse corpus from
Phase 1 (MiniGrid + 6 OpenSpiel games + Snake/Pong, ~358k transitions,
Sokoban deliberately excluded -- see Phase 1 section) instead of MiniGrid
alone, and (b) capacity, tested at two widths:

```
python -m jepa.train_moe_predictor --pretrain-epochs 4 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --openspiel-episodes-per-game 600 --openspiel-steps-per-episode 60 \
  --arcade-episodes-per-game 450 --checkpoint-every 5 \
  --out checkpoints_scaled_fold1_w<1|2> --width-mult <1.0|2.0>
```

**`--pretrain-epochs 4`, not 20:** the established recipe's 20 epochs on
67,200 MiniGrid-only transitions is ~1.34M pretrain samples-seen. This
run's pretrain corpus is ~358k transitions (~5.3x bigger) -- keeping the
epoch count at 20 would mean ~5.3x more total pretrain gradient updates
than the proven-good recipe, which is exactly the kind of curriculum
imbalance the Procgen rerun (CLAUDE.md's Stage 6 addendum) showed causes
an unrecovered encoder collapse. 4 epochs on the new corpus (~1.43M
samples-seen) keeps total pretrain compute roughly matched to the known-
good recipe instead. The finetune phase is left unchanged (60 epochs) --
that's the phase that does the recovering in Procgen's own diagnosis, so
it wasn't reduced.

**Deliberately excluded Sokoban** (and MinAtar/Procgen, which aren't part
of this branch's scope) from the pretrain mix -- all three had negative
or mixed held-out-games results in earlier testing (see CLAUDE.md), and
including them here would confound whether any effect is attributable to
the *new* diversity under test (OpenSpiel + arcade) versus re-surfacing
an already-known-negative source.

### Width 1.0 result: real run, no collapse, held-out gap NOT closed

Ran to completion (checked via a real blocking wait, not assumed) --
pretrain phase (4 epochs): val_pred_mse and val_identity_mse both shrink
together toward ~0.00001-0.00002 (expected on synthetic data this
different from ARC-3, not evaluated further since the pretrain phase's
own val split isn't the number that matters here). Finetune phase (60
epochs) shows a real, maintained gap the whole way through -- not a
Procgen-style collapse where pred and identity converge to equal, near-
zero, uninformative values: epoch 1 pred=0.08335/identity=0.07774 (pred
briefly worse, expected early), epoch 30 pred=0.00094/identity=0.00122,
epoch 60 pred=0.00032/identity=0.00043 -- pred consistently, increasingly
beats identity as training progresses, ending at a real ~26% gap on the
standard (trained-corpus) validation split. This is the recovery check
Phase 3 asked for, and it passes: no sign of the Procgen collapse
pattern.

**Held-out-games result (fold 1, `scripts/eval_scaled_world_model.py`,
2,400 held-out transitions across the 5 held-out games):**

| game | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| r11l | 0.001322 | 0.001320 | -0.14% |
| bp35 | 0.069121 | 0.069193 | +0.11% |
| m0r0 | 0.020702 | 0.020727 | +0.12% |
| tr87 | 0.004358 | 0.004356 | -0.04% |
| ka59 | 0.000906 | 0.000904 | -0.16% |
| **overall** | **0.022492** | **0.022514** | **+0.10%** |

**+0.10% overall -- essentially unchanged from fold 1's established
baseline (+0.01%, `stage6-multifold-cv`), and squarely inside the -0.30%
+/- 0.66% mean/std band across all 5 folds' worth of prior interventions
documented in CLAUDE.md.** ~5.3x more pretraining data, spanning 8 newly-
added, genuinely different game mechanics (turn-based board games,
push-your-luck chance, growth/collision, bounce physics -- none of them
"more grid navigation," unlike every earlier attempt), at unchanged
(1.0x) model capacity, does not move this number in any meaningful way.

**Trained-games sanity check** (9,600 transitions across the fold's other
20 games): pred=0.000115, identity=0.000133, **+13.33% improvement** --
healthy, positive, comparable to prior production-style recipes' own
trained-games numbers (see CLAUDE.md's various "trained-games" results in
the +8% to +44% range depending on recipe) -- confirms the model learned
real, non-degenerate dynamics on data it *was* trained on. The gap
specifically fails to transfer to genuinely unseen games; it isn't that
the model failed to learn anything at all.

Full checkpoint metadata (`moe_training_meta.json`), per-game numbers,
and raw eval output saved to `logs/stage6_scaled_world_model_eval.json`
(label `scaled-w1-fold1`) and `checkpoints_scaled_fold1_w1/` (gitignored,
main checkout only, not in this worktree's git history).

### Width 2.0 result: capacity made held-out generalization measurably WORSE

Ran to completion cleanly (confirmed via a real blocking process-exit
check, not assumed) -- no crash, all 60 finetune epochs completed. The
epoch-by-epoch log itself looks *healthier* than width=1.0's, not worse:
final epoch (60) val_pred_mse=0.00067 vs val_identity_mse=0.00105 on the
standard trained-corpus validation split, both decreasing together in a
normal, non-collapsed pattern throughout. Nothing in the training log
alone would flag a problem -- this is exactly why Phase 4's actual
held-out-games eval matters and the trained-validation-split number
doesn't answer the real question.

**Held-out-games result (fold 1, same 2,400 held-out transitions as the
width=1.0 run):**

| game | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| r11l | 0.010058 | 0.010029 | -0.28% |
| bp35 | 0.045490 | 0.045325 | -0.36% |
| m0r0 | 0.086388 | 0.007785 | -1009.67% |
| tr87 | 0.001581 | 0.001431 | -10.49% |
| ka59 | 0.002266 | 0.002198 | -3.08% |
| **overall** | **0.030298** | **0.016091** | **-88.29%** |

**-88.29% overall -- a real, substantial regression, not a small-
baseline percentage artifact.** (CLAUDE.md flags this exact failure mode
elsewhere -- a tiny identity-MSE denominator inflating an otherwise-small
absolute gap into a huge percentage -- so it was checked directly here:
m0r0's absolute pred error, 0.086388, is genuinely ~11x its own identity
baseline, 0.007785, and is also the single largest absolute error of any
game in either width's held-out table. This is a real degradation in
kind, not a measurement artifact.) Every one of the 5 held-out games gets
worse, not just the pooled number -- r11l and bp35 are close to parity
(like width=1.0's), but tr87, ka59, and especially m0r0 are substantially
worse in ways width=1.0 never was.

**Trained-games sanity check**: pred=0.000756, identity=0.002448,
**+69.11%** -- clearly *better* than width=1.0's +13.33%. This is the
crux of the finding: **more capacity, on this larger and more diverse
pretraining corpus, produced a model that fits the training distribution
(both the ARC games it trained on and the diverse synthetic sources)
substantially better, while generalizing to genuinely unseen ARC games
substantially worse.** That is the textbook shape of a capacity-enabled
overfitting/memorization tradeoff, not a neutral "no effect" result --
and it is the opposite direction from what the "capacity + diversity"
hypothesis this experiment was built to test predicted. Width=1.0's own
identical-data result (+0.10% held-out, +13.33% trained) sits between
the established baseline and width=2.0 on both axes, consistent with a
real, monotonic-in-capacity trend rather than noise.

Given how large and directionally consistent (every held-out game worse,
not a mixed bag) this swing is, and given this project's own standing
lesson about not trusting a single fold's result (see CLAUDE.md's whole
multi-fold cross-validation section), a second fold was run before
treating this as conclusive -- see below.

### Fold 2 validation: the fold-1 collapse did NOT replicate at the same magnitude

Same recipe, width=2.0, fold 2's held-out games (`ar25,cd82,cn04,dc22,ft09`,
matching `stage6-multifold-cv`'s partition). Ran to completion cleanly (10
polling attempts, ~100 minutes, no crash).

**Held-out-games result (fold 2, 2,400+ held-out transitions):**

| game | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| ar25 | 0.027823 | 0.027802 | -0.08% |
| cd82 | 0.048249 | 0.048257 | +0.02% |
| cn04 | 0.119877 | 0.119868 | -0.01% |
| dc22 | 0.000252 | 0.000246 | -2.68% |
| ft09 | 0.000117 | 0.000095 | -23.07% |
| **overall** | **0.039621** | **0.039610** | **-0.03%** |

**-0.03% overall -- essentially parity, and close to fold 2's own
established baseline (+0.11%). This does NOT replicate fold 1's dramatic
-88.29% collapse.** `ft09`'s -23.07% looks alarming in isolation but is
the small-denominator artifact CLAUDE.md warns about (checked directly,
unlike fold 1's m0r0 case which was real): identity=0.000095 is tiny, and
the absolute gap (0.000022) is small in its own right, nothing like
fold 1's m0r0 (an real ~11x absolute blowup). `dc22`'s -2.68% is a real
but modest absolute effect. No game in fold 2 shows anything close to
fold 1's magnitude of regression.

**Trained-games sanity check**: pred=0.000181, identity=0.000191,
**+5.26%** -- notably lower than fold 1's width=2.0 result (+69.11%) and
even lower than fold 1's width=1.0 result (+13.33%). The pattern that
looked like a clean "capacity trades held-out generalization for
trained-fit" story in fold 1 (higher trained-fit, much worse held-out)
does not hold up as a general rule in fold 2 (lower trained-fit than
either fold-1 result, *and* held-out roughly unchanged from baseline).

### Phase 4 verdict: no evidence "capacity + diversity together" closes the gap -- and one fold shows a real regression risk

Three real, hard-won training runs now on the table, all sharing the
identical diverse pretraining corpus and finetune recipe, differing only
in fold and capacity:

| run | held-out improvement | trained-games improvement |
|---|---|---|
| fold 1, width=1.0 (diverse data, baseline capacity) | +0.10% | +13.33% |
| fold 1, width=2.0 (diverse data, 2x capacity) | **-88.29%** | +69.11% |
| fold 2, width=2.0 (diverse data, 2x capacity) | -0.03% | +5.26% |
| *(reference) established baseline, fold 1, no diverse data* | +0.01% | -- |
| *(reference) established baseline, fold 2, no diverse data* | +0.11% | -- |
| *(reference) 5-fold baseline mean +/- std, no diverse data* | -0.30% +/- 0.66% | -- |

**The core hypothesis this experiment tested -- that genuinely diverse
data at meaningfully larger capacity would let the model learn something
transferable enough to generalize zero-shot to a novel ARC-3 game -- is
not supported by either fold.** Every single held-out-games number here
sits at or below the established no-diversity baseline band; none shows
the kind of positive, meaningful improvement the hypothesis predicted.

**The capacity finding is real but did not replicate cleanly across
folds, which is itself the honest, reportable result, not a reason to
discard fold 1's number.** Fold 1 showed a severe, directionally-
consistent-across-every-held-out-game regression from added capacity (all
5 games worse, one by a large absolute margin) alongside a substantially
better trained-games fit -- the textbook shape of capacity-enabled
overfitting. Fold 2 showed neither effect clearly: held-out stayed near
parity with baseline, and trained-games fit was actually the *weakest*
of all three runs. Two folds are not enough to establish *why* fold 1 and
fold 2 diverge this much (a fold-specific interaction -- e.g. something
about `m0r0` specifically, which alone accounted for most of fold 1's
regression, vs. no single fold-2 game showing a comparable blowup -- is
the most likely explanation, but wasn't isolated further given the
compute cost of another full training run to chase it). What both folds
agree on, without exception, is the thing that actually answers this
experiment's question: **neither fold shows capacity turning this
project's diverse-pretraining data into better held-out generalization.
At best (fold 2) capacity is neutral to the held-out number while
underperforming on trained-games fit relative to the smaller model; at
worst (fold 1) it actively makes held-out generalization dramatically
worse while overfitting harder to the training distribution.** That is a
real regression risk with no compensating upside observed in either fold
-- reason enough on its own not to pursue this direction for anything
Kaggle-submission-relevant without a lot more evidence.

**Decision: not running width=4.0, and not running a third fold, at this
time -- both documented judgment calls, not oversights.** Width=4.0 was
in scope per this task's own Phase 2 framing ("pick 1-2 width
multipliers... document why"). Given neither tested width shows a
positive effect and the *inconsistency* between fold 1 and fold 2's
width=2.0 results already points to high run-to-run/fold-to-fold
variance in this specific regime rather than a clean, extendable
capacity trend, pushing capacity even higher seems more likely to add
noise than signal -- and the original small-data capacity sweep
(`stage6-capacity-sweep`) already found flat-to-worse results at 1x-4x
on a completely different data regime, which is at least weakly
consistent with "more capacity isn't the lever" holding across data
regimes too. A third fold (or a width=1.0 companion run on fold 2, for a
fully matched 2x2) would sharpen confidence in the *magnitude* of the
fold-to-fold inconsistency, but the qualitative verdict -- no fold shows
improvement, one fold shows serious regression -- is already
well-supported by 3 real runs plus the existing 5-fold no-diversity
baseline, and matches this project's own repeated pattern (see CLAUDE.md's
running tally: 10 independent interventions today alone, 9 failures, 1
modest success) of the held-out-games gap being a genuinely hard, likely
data-fundamentals-bound problem that isolated architecture/scale changes
don't crack. Spending further compute confirming the same qualitative
conclusion at diminishing marginal information is a worse use of time
than stating the result plainly now.

## Overall conclusion

**The "capacity + genuinely diverse data together" hypothesis, tested at
a real order-of-magnitude step up from every prior attempt (~358k
pretraining transitions across 10 synthetic sources spanning navigation,
push-mechanics, 6 turn-based board-game genres, chance/dice, and arcade
physics, at up to 2x model capacity), does not close the held-out-ARC-
games generalization gap.** Every held-out number obtained (+0.10%,
-88.29%, -0.03%) is at or below the pre-existing 5-fold no-diversity
baseline band (-0.30% +/- 0.66%), and the one place capacity showed any
directional signal at all was a severe regression in one of two folds
tested, with no compensating held-out improvement anywhere to offset that
risk. This is now the 11th and 12th independent interventions (out of an
accumulating project total, see CLAUDE.md's running tally) to fail to
close this gap, joining 7 conditioning/architecture fixes and 3 prior
data-diversity attempts (MinAtar, MinAtar per-game-id, Procgen) -- all
converging on the same qualitative conclusion this project has reached
several times now (Stage 1's original action-input bug being the one
major exception where a "several independent fixes converge on the same
negative result" pattern *did* turn out to be a fixable bug, not a
fundamental limit): when many different, well-targeted, honestly-
executed interventions all land on the same negative outcome, that
consistency is evidence of a genuine limit in this project's current data
regime, not an unturned dial. Unlike Stage 1's case, this pattern has now
been checked hard for a hidden bug (the action-input-style root cause) via
7+ independent architecture/conditioning audits across this and the
earlier same-day investigation, and none have found one.

**What remains genuinely untested, for a future session:** (a) capacity
paired with diverse data at a fold-matched, fully-symmetric 2x2 grid
(width 1.0 and 2.0, both folds) to isolate whether fold 1's regression
is fold-specific noise or a real if inconsistent effect; (b) capacity
below 1.0 (a smaller, more regularized model) on this same diverse
corpus, which this experiment never tried and which the classic
overfitting-shape of fold 1's result makes newly interesting; (c) this
project's own standing next-step list from CLAUDE.md's Stage 6 addendum,
unchanged by this experiment's result -- test-time adaptation (the one
mechanism that has shown any real, if modest, positive signal all
project) remains the most promising lever, and this experiment's
negative result on "more/different pretraining data + capacity" makes
that relative standing stronger, not weaker.


