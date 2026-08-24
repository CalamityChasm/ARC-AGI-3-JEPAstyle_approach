# bp35/ka59: what these games actually require, found by direct inspection

**Status: COMPLETE for this round -- both games' core mechanic is now
concretely identified, and it directly explains why every exploration
strategy tried this session (three architecturally distinct ones, plus
every world-model-quality intervention including a dramatically improved,
TTA-adapted recurrent predictor) failed identically at 0 level
completions.**

## Motivation

`experiments/stage6_recurrent_exploration.md`'s "Follow-up 3" established
that even the largest, cleanest world-model improvement this project has
ever produced on `bp35`/`ka59` specifically (higher-dose TTA,
changed-patches -9.76% -> +20.74% on `bp35`) still produced 0/16 level
completions -- ruling out world-model *quality* as the bottleneck. Rather
than try more adaptation-method variants (options 4-6 on the ranked
list), this follows the recommended alternative: directly inspect what a
real attempt at each game actually looks like, frame by frame, informed
by tooling already built this session (`scripts/visualize_recording.py`,
extended here with `scripts/render_frames_png.py` and `scripts/
track_sprite_and_actions.py` for direct visual/positional inspection
without needing a live browser).

## `ka59`: a dual-token routing puzzle under a hard, visible 100-action clock

Rendered a `RecurrentSearchTTA` attempt at steps 0, 50, and 98 (`logs/
frames_ka59/`, gitignored, regenerable via `scripts/render_frames_png.py
--game ka59 --agent recurrentsearchtta --steps 0 50 98`). The board is
two blue rooms connected by a corridor, separated by a pink vertical
barrier. Each room contains one small brown "token" and one
yellow-outlined goal square. A yellow bar spans the entire bottom row at
step 0 and shrinks by exactly one segment roughly every ~8 steps,
reaching zero and triggering `GAME_OVER` at step 99/100 -- this is a
literal, on-screen, deterministic countdown, not a hidden mechanism, and
it exactly matches the "`ka59` caps at exactly 100 actions/attempt"
finding `experiments/stage6_action_selection_softmax.md` already
established indirectly (via attempt-length statistics) -- now visually
located and confirmed.

Between step 0 and step 50, both tokens visibly moved toward their own
room's goal square (left token toward the left goal, right token toward
the right goal) -- real, if incomplete, progress. By step 98, the left
token had drifted *away* from its goal again, and the pink barrier never
opened or moved in any frame observed. The win condition is most likely
"route both tokens onto their respective goal squares" (or possibly just
one, or requires the barrier to open via some untested interaction) --
not confirmed further this round, but the structural read is solid: this
is a *goal-directed placement puzzle* with a strict shared turn budget,
not a "keep exploring and something eventually happens" task.

## `bp35`: a vertical escape puzzle with a rising hazard -- and a stark confirmation that exploration never even tried the right direction

Rendered steps 0, 45, 51 (`logs/frames_bp35/`). The board is a gray,
green-speckled vertical maze with white open corridors and brown
obstacle blocks near the top; a pink texture is completely absent at
step 0 and has flooded up to cover roughly the bottom 40% of the board by
step 45, reaching the player sprite's position -- a rising hazard, not a
fixed timer. This explains why `bp35`'s attempt lengths were
*variable* (52, 65, 65, 65, 54 steps observed across four attempts in one
recording) rather than `ka59`'s exact, fixed 100 -- `bp35`'s "up to 64"
finding in `stage6_action_selection_softmax.md` is an upper bound on how
long a given policy survives the flood, not a hard action-count cap.

**The decisive finding**: tracked the player sprite's exact `(row, col)`
centroid every step via `scripts/track_sprite_and_actions.py --game bp35
--agent recurrentsearchtta --color 9` (the sprite is colors 9+11, a small
two-tone diamond). Across the entire 51-step attempt before death, **the
sprite's row coordinate moved by less than 1.5 cells total** (37.0-38.5
throughout) -- it never made real vertical progress at all, only
oscillating horizontally within an ~8-column band (columns 15-23) near
its starting position. Visually, the maze's actual passage upward (a
narrow white vertical shaft connecting the lower corridor to the upper
platform) sits far to the right of this band (column ~44-48 by visual
estimate in the step-0 render) -- **the agent never even approached the
correct horizontal position to find the way up**, let alone climbed
through it, in the entire attempt. `available_actions` was `[3, 4, 6, 7]`
throughout (no vertical-only action id was ever offered or needed
separately -- movement through the shaft is presumably automatic once
correctly aligned, based on the maze's visual structure, though this
wasn't directly confirmed by observing a successful climb).

## Why every exploration strategy this session hit the same wall

Both games require **directed, goal-seeking progress toward a specific,
identifiable target** (a distant vertical shaft in `bp35`; specific goal
cells in `ka59`) **within a hard turn budget** -- not "reach states that
are different from ones already seen." Every strategy tried this session
-- `Hypothesis`'s InfoGain/value blend, `RecurrentCuriosity`'s
retrospective surprise ranking, `RecurrentSearch`'s genuine multi-step
MPC lookahead, all three re-tested with test-time-adapted world models of
varying quality up to a dramatically real +20.74% changed-patches
improvement -- optimizes for *novelty* or *surprise*, which has no
representation of "this direction is progress" or "that specific cell is
the goal." A better-adapted world model makes the agent's *prediction* of
what happens next more accurate; it does nothing to make "climb toward
the far-right shaft" or "route this token onto that yellow square" more
rewarding than any other equally-novel action. This is a full,
mechanistic explanation for the 0/64+ result across every condition
tested against these two games this session, not just a restatement of
the earlier "the per-attempt budget is tight" finding -- the budget isn't
merely tight, the policies tested had no mechanism that would use even a
generous budget correctly, since none of them were ever moving toward the
right place to begin with.

## What this implies for a real fix (not attempted this round)

A genuinely different kind of signal is needed, one that can identify
board structure that looks goal-like or hazard-like and plan directly
toward/away from it -- novelty search cannot be patched into this by
tuning temperature, dose, or search depth further, since the objective
itself (reach unseen states) is structurally indifferent to which
direction is correct. Concretely, this points toward either:

1. **Vision/structure-based goal inference** -- detect salient, distinct
   shapes on the board (the yellow-outlined squares in `ka59`, the white
   corridor structure in `bp35`) via classical CV (connected components,
   color-based segmentation) or a vision-capable LLM prompted on the
   opening frame(s), and use that as an explicit sub-goal for a planner --
   this project already has a parked `stage7-llm-hypothesis` branch that
   may be directly relevant groundwork.
2. **A real value/reward signal grounded in progress, not surprise** --
   e.g. if a countdown bar or similar visible timer can be detected
   generically, an agent could at least learn "time is running out" as a
   pressure signal even without knowing the specific goal; more
   ambitiously, a value head trained to recognize "distance shrinking to
   *some* salient board feature" rather than only sparse
   `levels_completed` events.

Neither was attempted this round -- this experiment's scope was
understanding the mechanic, not building the fix. Worth flagging plainly:
this is a much larger undertaking than anything tried against these two
games so far (every previous fix reused existing infrastructure;
goal-directed planning from visual structure is new machinery), and
should be scoped and confirmed with the user before starting.

## Reproducing this investigation

```
python scripts/render_frames_png.py --game ka59 --agent recurrentsearchtta --steps 0 50 98 --out-dir logs/frames_ka59
python scripts/render_frames_png.py --game bp35 --agent recurrentsearchtta --steps 0 45 51 --out-dir logs/frames_bp35
python scripts/track_sprite_and_actions.py --game bp35 --agent recurrentsearchtta --color 9 --max-steps 55
python scripts/analyze_attempt_structure.py --game bp35 --agent recurrentsearchtta
python scripts/analyze_attempt_structure.py --game ka59 --agent recurrentsearchtta
```
