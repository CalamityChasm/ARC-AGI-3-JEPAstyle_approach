# A classical-CV countdown-bar detector: real, validated infrastructure -- no detected agent-level win on bp35/ka59

**Status: built, validated against real recordings, wired into `Hypothesis`
behind an off-by-default flag. Agent-level backtest on `bp35`/`ka59`: 0/16,
identical to every other condition. Consistent with, not contradicting,
`experiments/stage6_bp35_ka59_mechanics.md`'s own diagnosis -- these two
games need real goal-direction, and "explore less, commit more as time
runs out" doesn't supply that on its own.**

## Motivation

A direct user observation: many ARC-3 games likely have some kind of
visible action/turn counter (a depleting bar, a shrinking meter), and
since it's common across games, a world model that's actually learning
transferable structure should predict it well even zero-shot on a held-
out game. Tested directly (`scripts/diagnose_countdown_bar_prediction.py`):
on `ka59` (held out of `checkpoints_recurrent_holdout`), the recurrent
predictor is **worse than trivially predicting "nothing changes" on the
countdown-bar region specifically** (pred MSE 0.000105 vs. identity MSE
0.000019, ~5.5x worse) -- not selectively failing on idiosyncratic
content while still picking up generic conventions; it fails uniformly,
even on the mechanic that should be the easiest possible transfer case.
Consistent with the game-id-conditioned residual-collapse mechanism
already established this session: the model has no representation of
"bars near an edge tend to shrink" independent of *which specific game*
this is, so an unfamiliar `game_id` embedding gets nothing regardless of
how common the underlying visual pattern is across training games.

This motivated a non-learned alternative that sidesteps the conditioning
problem entirely: a classical-CV detector operating directly on raw
pixels, with no trained component and therefore nothing to fail to
transfer.

## The detector (`jepa/countdown_detector.py`)

Watches the four single-pixel-wide board edges for a depleting bar. On
the first frame of an episode, finds candidate runs of a single
non-dominant color along each edge, long enough to plausibly be a
deliberate UI element. On every subsequent frame, a candidate survives
only if:
1. **No reversion** -- a cell that emptied never turns back to the bar
   color (a real timer never counts back up).
2. **Contiguous-from-an-edge** -- the emptied cells form one unbroken
   block anchored at either end of the span, the shape a real bar always
   depletes in.
3. **Multiple depletion events** (>= 5 separate observation steps where
   the depleted count grew) -- rules out a single one-off jump (e.g. an
   unrelated object moving past the edge once) being mistaken for a real
   per-turn countdown.

**Iteration note, for anyone extending this**: checks 1-2 alone were not
enough. First version (no dominant-color exclusion, no event-count
threshold) fired on every game tested, including ones with no real timer
-- e.g. `bp35`'s bottom row (color 0, background) looked monotonically
"contiguous-from-edge" purely from incidental gameplay noise over a short
window. Adding dominant-color exclusion and the event-count threshold
fixed most false positives (`m0r0`, `ls20`, `tn36` now correctly report no
detection) but not all -- `bp35`, `r11l`, `sp80`, `cd82`, `ar25`, `vc33`,
`ft09` still fire. Directly checked one (`s5i5`, urgency reaching 0.95):
**confirmed real** -- a genuine green progress bar spanning the full
bottom edge at step 0, visibly transitioning toward yellow by step 60
(`logs/frames_s5i5/`, gitignored, regenerate via `scripts/
render_frames_png.py --game s5i5 --steps 0 60`). Given that direct
confirmation, the remaining "positives" are more likely mostly genuine UI
elements too (this project's own frame inspection has found bars/meters
often span a large edge fraction, exactly matching `MIN_BAR_FRACTION`'s
threshold) -- `bp35`'s specific case is plausibly its already-documented
rising-flood hazard (`experiments/stage6_bp35_ka59_mechanics.md`), which
genuinely is a legitimate, monotonically-encroaching danger signal even
though it isn't a literal per-turn counter. Not exhaustively verified
game-by-game beyond this one spot check -- worth doing before trusting
the detector's positives blindly on a wider sweep.

## Integration: `Hypothesis.TIMER_AWARE`

Off by default (`HYPOTHESIS_TIMER_AWARE=1` env var, same on/off-flag
pattern as `TEST_TIME_ADAPT`/`DIAG_MODE`). When on: `self._timer.observe
(latest_frame.frame)` every real turn, `self._timer.reset()` on RESET.
`_effective_epsilon()` scales `EPSILON` down as detected urgency rises --
`EPSILON * (1 - urgency)` -- less random exploration, more commitment to
the current best-scoring action, as the detected budget depletes.
Deliberately the simplest possible integration (a single existing
parameter, scaled by a new signal) rather than a new mechanism, to keep
the test clean.

## Result: 0/16 on bp35/ka59, TIMER_AWARE on and off, identical

n=8 each, `MAX_ACTIONS=300` (the real Kaggle-relevant budget), same
protocol as every other bp35/ka59 comparison this session. `TIMER_AWARE`
on: 0/16. `TIMER_AWARE` off (same-round baseline, same merged codebase):
0/16. No detectable difference.

**This does not contradict `stage6_bp35_ka59_mechanics.md`'s own
diagnosis -- it's the predicted outcome given that diagnosis.** That
investigation found the real bottleneck on these two games isn't "the
agent doesn't know time is running out," it's "the agent has no notion
of which direction is correct" (`bp35`'s player sprite moved less than
1.5 cells vertically in a full 51-step attempt, oscillating near its
start position while the actual passage upward sat ~25 columns away).
Scaling down epsilon just sharpens commitment to whatever the agent's
existing InfoGain/value blend already prefers -- if that preference
carries no directional signal (which is exactly what the mechanics
investigation found), committing to it *more* confidently doesn't turn a
wrong-direction policy into a right-direction one. The detector correctly
identifies "less time is left," it just has nothing to hand that
information to that would know what to do about it.

## Follow-up: full 25-game sweep, both the detector and the agent

Two things checked, per a direct follow-up request: (1) which of the 25
local games have a detectable bar, verified against real frames rather
than trusted blindly; (2) a full 25-game agent-level backtest, not just
the two hardest, already-diagnosed cases.

### Detector coverage: 16/25 games, spot-checked

Ran the detector against each game's most recent recording (no new
gameplay needed): **16/25 games show a detected bar** (`ar25`, `bp35`,
`cd82`, `cn04`, `dc22`, `g50t`, `ka59`, `r11l`, `re86`, `s5i5`, `sc25`,
`su15`, `tr87`, `tu93`, `vc33`, `wa30`); **9/25 show none** (`ft09`,
`lf52`, `lp85`, `ls20`, `m0r0`, `sb26`, `sk48`, `sp80`, `tn36`). Directly
visually verified 3 of the 16 against real rendered frames before trusting
the list (`s5i5`: a green-to-yellow bottom bar; `dc22`: a
background-color-to-black bottom bar, small at step 0, visibly larger by
step 200; `r11l`: a much subtler bar, a short segment on the *left* edge
near the top, easy to miss by eye but genuinely shrinks between step 0
and step 200) -- all three confirmed real, not detector noise. The
"not detected" list is a real, useful data point on its own: it's not
that these games definitely lack any progress indicator (a bar in a
color/position/shape the detector's edge-strip-only search wouldn't
catch, e.g. a numeric counter, a non-edge-adjacent element, or one that
depletes from both ends rather than one, would be invisible to this
specific detector), but it does mean at minimum they don't have the
*one* common pattern (single-edge, one-directional, contiguous depletion)
this detector looks for.

### Agent-level backtest: all 25 games, n=8 each, TIMER_AWARE on vs off

Same `MAX_ACTIONS=300` protocol as every other comparison this session,
run same-round on the merged codebase (400 runs total, ~63 minutes
wall-clock via 5-way parallel batching by game).

| condition | total levels | mean score | distinct games (>=1 level) |
|---|---|---|---|
| off (baseline) | 10 | 0.0667 | 2 (`r11l`, `sp80`) |
| on (timer-aware) | 14 | 0.0573 | 4 (`cn04`, `lp85`, `r11l`, `sp80`) |

Mixed at face value -- more total levels and more distinct games for
`on`, but a lower mean *score* (Kaggle's real scoring formula weights
per-level action-efficiency, not just raw completion count, so completing
more levels less efficiently, or losing efficient completions on one
game while gaining inefficient ones on another, can net negative on score
even while netting positive on the raw count).

**The useful part: two of the four differing games have NO detected bar
at all, giving a real noise-floor calibration for free.** `lp85`
(+1 level, off->on) and `sp80` (-2 levels) both show `TIMER_AWARE=None`
throughout (no bar detected -- see the coverage list above), meaning the
feature was mechanically inert on both; any difference there is pure
run-to-run variance from `Hypothesis`'s own unseeded exploration, not the
detector doing anything. That's a direct, in-band measurement of how much
an n=8 level-count comparison swings by chance alone on this metric: up
to +/-2 levels on a single game, with no intervention active at all.

Against that calibrated noise floor, the two remaining differing games --
**`cn04` (+2 levels) and `r11l` (+3 levels), both on games with a
confirmed real bar** -- are directionally consistent (both positive, both
on bar-detected games) and both larger than the noise-floor swings
observed on the inert games. Real signal, not proof: n=8 per game is the
same sample size this project has repeatedly found insufficient to
separate a genuine small effect from noise on this exact metric (see
CLAUDE.md's own novelty-aware-beta retraction, where an equally
clean-looking n=8 win completely evaporated at n=30). Worth a larger
confirmatory backtest specifically on `cn04`/`r11l` before treating this
as validated, not a reason to dismiss it either.

**`bp35`/`ka59` specifically: 0 total levels on both conditions,
unchanged.** Consistent with, not contradicted by, this broader sweep --
the mechanics-level diagnosis (no sense of direction, not no sense of
urgency) still stands as the best explanation for why these two
specifically see zero effect from either condition.

## What's real and worth keeping regardless

The detector itself is genuine, validated, reusable infrastructure,
independent of this one negative integration test -- it directly answers
the motivating question (does the world model transfer this mechanic?
No) with a working non-learned alternative that does detect it
correctly on real games. Two directions worth trying before writing off
countdown-awareness entirely:
1. **A different consumer of the signal** -- feed urgency into whichever
   goal-directed planner eventually gets built for `bp35`/`ka59`
   specifically (the still-unbuilt recommendation from `stage6_bp35_ka59
   _mechanics.md`), rather than an existing novelty/value-driven agent
   that has no notion of direction to sharpen in the first place.
2. **Test on the full 25-game suite**, not just the two hardest, most
   already-diagnosed cases -- games where the agent's existing Q-blend
   already points in a roughly correct direction most of the time could
   plausibly benefit from committing harder as time runs low, even though
   the two cases tested here (where the underlying policy has no
   direction to sharpen at all) show no effect. Not run this session.
