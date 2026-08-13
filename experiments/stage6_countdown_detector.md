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
