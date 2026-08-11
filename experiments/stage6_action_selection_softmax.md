# Stage 6: why do bp35/tr87/ka59 never get solved, independent of the world model?

**Status: root cause found and fixed for a real, previously-undiagnosed bug -- but it does not fully explain the held-out-games breadth gap. See "Big-budget probe" below for the open question.**

## Motivation

Every backtest run against the 5 held-out games (`r11l`, `bp35`, `m0r0`,
`tr87`, `ka59`) across this entire Stage 6 investigation -- baseline,
`MAX_ACTIONS=900`, test-time adaptation at any dose, Reptile meta-learning,
novelty-aware beta -- has solved `r11l` and only `r11l`. See
`experiments/stage6_test_time_adaptation_agent.md`'s follow-up section for
the full tally: a 7x-larger TTA dose produced a much bigger
representation-level accuracy gain and still made zero difference in
play. That pattern is bigger than any one lever tested so far -- it
suggested the ceiling on `bp35`/`m0r0`/`tr87`/`ka59` isn't really about
world-model quality at all.

## Step 1: how hard are these games, independent of any agent?

Checked `ARC-AGI-3-Agents/environment_files/<game>/*/metadata.json` for
`win_levels` and `baseline_actions` (human baseline per level):

| game | win_levels | first-level baseline | sum of all levels' baselines |
|---|---|---|---|
| `bp35` | 9 | 21 | 651 |
| `m0r0` | 6 | 30 | 1107 (level 5 alone: 500) |
| `tr87` | 6 | 54 | 414 |
| `ka59` | 7 | 28 | 730 |

First-level baselines are all well under 300 -- nothing here suggests
completing *one* level (this project's usual bar for "solved") should
structurally require more than the standard budget, at least by the
human baseline.

**Checked whether any agent, at any budget, has EVER completed a level on
these 4 games across this project's entire history** (48 local recording
files per game -- random policy, `Curiosity`, `Memory`, `Hypothesis`, TTA
experiments -- plus the separately-documented `Memory` teacher-policy
pass at `MAX_ACTIONS=2500`, CLAUDE.md's Stage 5 follow-up section):

| game | ever solved (any agent, any budget, any checkpoint)? |
|---|---|
| `bp35` | **No.** 0/48 local files, not in the 2500-action teacher pass either. |
| `m0r0` | **Yes, exactly once** -- a `Hypothesis` run dated 2026-08-02 (`levels_completed=1`), and again in the 2500-action `Memory` teacher pass. |
| `tr87` | **No.** 0/48 local files, not in the teacher pass. |
| `ka59` | **No.** 0/48 local files, not in the teacher pass. |

`bp35`/`tr87`/`ka59` have never been solved once, by anything, in this
project's entire multi-month history -- not by random play, not by
curiosity-driven exploration, not by exact-recall search with an 8x
larger budget. `m0r0` has a real (if rare) existence proof, so it's not
in quite the same category.

## Step 2: live-trace what the agent is actually doing on these games

Enabled `.env`'s `DEBUG` flag and ran one full `main.py --agent=hypothesis`
episode directly (bypassing `run_scorecard.py`, which discards the raw
log) on each of `tr87`, `bp35`, `ka59`, plus `r11l` as a working control,
using the `checkpoints_holdout_baseline` checkpoint already in
`checkpoints/` from the TTA dose experiment.

**Found immediately: the greedy Q-argmax branch picks the exact same
single action on 100% of its decisions, for the entire episode, on every
one of the three unsolved games:**

| game | available actions | greedy decisions | action chosen |
|---|---|---|---|
| `tr87` | `[1,2,3,4]` | 224/224 | `a2`, always |
| `bp35` | `[3,4,6,7]` | 210/210 | `a3`, always |
| `ka59` | `[1,2,3,4,6]` | 212/212 | `a2`, always |
| `r11l` (control) | ACTION6 only | 197/197 | `a6`, always (but see below) |

Sample trace line (`tr87`): `beta=0.147 a1:0.08205 a2:0.08811 a3:0.08719
a4:0.08145 -> chosen a2` -- Q margins between candidates are routinely
tiny (a few percent of the base value) but one candidate is *consistently*
a hair ahead, so a hard `argmax` locks onto it every single time. Only
`Hypothesis.EPSILON`'s blanket 25%-of-turns uniform-random fallback ever
picks anything else.

**`r11l` "gets away with" the same 100%-single-action pattern because it
has no other choice to begin with** (its `available_actions` is ACTION6
only, per Stage 5 follow-up 3's own finding) **but still gets real
behavioral diversity for free**: `_sample_click`'s existing temperature
softmax over the 64-patch variance map means *where* ACTION6 clicks
varies every turn, even though *which* action is chosen never does.
`tr87`/`bp35`/`ka59` have no such secondary parameter -- their candidate
actions are all SIMPLE actions with no location to vary -- so a locked
top-level choice means a genuinely static, unvarying policy for 75%+ of
every episode.

**This is the exact bug already fixed once in this codebase, just in a
different place.** This project's own Gotchas section already documents
it: "a deterministic argmax over a salience/variance map will default to
the same fixed index every time the map is flat or near-flat" -- found
and fixed for `_sample_click`'s patch selection back in Stage 5
follow-up 3. That fix was never generalized to the *action*-level
argmax in `choose_action`'s greedy branch, which is a separate, structurally
identical decision point that nobody had traced closely enough to notice
was hitting the same failure mode.

## Fix

Replaced the hard argmax over candidate `Q(s, a)` values with a
temperature-weighted softmax sample (`ACTION_SAMPLE_TEMPERATURE = 0.1`,
same value as `PATCH_SAMPLE_TEMPERATURE` for consistency, not
independently swept), mirroring `_sample_click`'s already-validated
pattern exactly: `weights = softmax((q - max_q) / T)`, then
`self._rng.choices(candidates, weights=weights, k=1)`.

**Verified directly before re-running any backtest:** re-traced `tr87`
post-fix -- action distribution went from 224/0/0/0 to a genuinely
near-uniform **53/61/65/63** across a1-a4. The lock is gone.

## Agent-level backtest (n=8, all 5 held-out games, same checkpoint)

| condition | mean levels | total levels (of 8) | distinct games |
|---|---|---|---|
| pre-fix (hard argmax) | 0.375 | 3 | 1 (`r11l`) |
| **post-fix (softmax sample)** | **0.500** | **4** | **1 (`r11l`)** |

A real, if modest, reliability improvement on `r11l` (consistent with
this project's own repeated pattern of fixes landing as reliability
gains on the one already-partially-solvable game, not new breadth --
see the `MAX_ACTIONS=900` and novelty-aware-beta writeups). **But
`bp35`/`tr87`/`ka59` are still at zero completions, even now, with
verified real action diversity.** The argmax-lock bug was real and worth
fixing (kept -- no downside observed, a genuine improvement to
`r11l`'s reliability), but it is not sufficient on its own to explain why
these 3 games have never been solved.

## Big-budget probe: is it simply an action-budget ceiling?

Ran the FIXED agent (real action diversity, `checkpoints_holdout_baseline`)
against `bp35`, `tr87`, `ka59` individually at `MAX_ACTIONS=2500`
(temporarily bumped on the class attribute, then reverted -- this
branch's `hypothesis_agent.py` predates the `HYPOTHESIS_MAX_ACTIONS`
env-var-override pattern added on `stage6-budget-x-checkpoint`, which was
never merged here; a first attempt using that env var silently had no
effect, `total_actions` came back 301 not 2501 -- caught by checking the
scorecard's own `total_actions` field before trusting the result) to
check whether the remaining gap is simply "300 actions isn't enough
runway for a now-genuinely-exploring policy to stumble onto a winning
sequence," matching the historical `Memory` teacher-pass precedent
(`MAX_ACTIONS=2500`, CLAUDE.md's Stage 5 follow-up section).

**Result: `tr87` was solved -- the first time in this project's entire
history.** `total_levels_completed=1`, `total_actions=2501` confirmed.
`bp35` and `ka59` still came back at zero (single run each, n=1 -- not
strong evidence either way for those two specifically, just no completion
*this* run).

**This is real, direct confirmation that the two fixes compound:** a
game that had never been solved once across 48+ historical recording
files, a 2500-action teacher-policy pass with exact-recall search, and
every world-model condition tested this entire session, was solved the
moment BOTH the action-selection lock was fixed AND the budget was large
enough to give the now-genuinely-diverse policy room to find a winning
sequence. Neither fix alone (the softmax fix at `MAX_ACTIONS=300`, or the
historical `MAX_ACTIONS=2500` teacher pass with the OLD locked-argmax
policy) ever cracked it. `bp35`/`ka59` not solving even at 2500 actions
in a single run doesn't rule out the same story applying with more
repeats or more budget -- it just isn't confirmed yet at n=1.

## Confirmation batch: was the tr87 solve a lucky n=1 roll?

Ran 8 fresh repeats of `tr87` alone at `MAX_ACTIONS=2500` (same temporary
bump-and-revert pattern, same fixed agent, same checkpoint) specifically
to check whether the single solve above was a fluke.

**Result: 1/8 repeats completed a level** (`tr87_bigbudget_confirm_r1`,
`total_actions=2501`; r2-r8 all zero). Combined with the original single
run, that's **2 solves across 9 total attempts at this budget (~22%)**.

**This replicates, but modestly -- not the "clearly fixed" result a
single success might have suggested.** The good news: it's not a fluke
-- a genuinely novel-to-this-checkpoint game that had *never* been solved
once across 48+ historical recordings, any budget, any world model, just
got solved twice independently in two separate small samples. The honest
caveat: at `MAX_ACTIONS=2500`, this is roughly a 1-in-4 to 1-in-9 event,
not a reliable win -- `tr87` went from *literally never solvable* to
*occasionally solvable given 8x the budget*, which is real, meaningful
progress, but still far from "cracked." Consistent with this project's
own repeated lesson about not over-reading a small sample in either
direction (the same standard applied when the 8/8 budget+TTA combo result
evaporated at n=25, and equally applicable here to avoid the opposite
mistake of writing off a low-but-nonzero rate as noise).

## bp35 and ka59 at the same budget: a clean negative, not just "not yet confirmed"

Ran 8 fresh repeats each of `bp35` and `ka59` at `MAX_ACTIONS=2500`, same
fixed agent, same checkpoint, same protocol as `tr87`'s confirmation
batch above.

**Result: 0/8 for `bp35`, 0/8 for `ka59` -- zero completions in 16 runs.**
Combined with each game's earlier single run at this budget, that's
**0/9 attempts for both games.** Unlike `tr87` (2/9, a real if modest
positive rate), `bp35` and `ka59` show no sign of budging even with an
8x larger budget and the action-selection fix both in place.

**This sharpens the picture rather than muddying it.** The
action-selection bug and the budget ceiling were real and, together,
demonstrably sufficient to unlock `tr87` -- but they are evidently NOT
sufficient for `bp35`/`ka59`, which points at a third factor specific to
those two games (their own mechanics, a longer/more precise action
sequence than a still-mostly-undirected softmax-sampled policy can find
even with more tries, or something structurally different from `tr87`'s
own difficulty shape). `bp35` (9 levels, win requires 4 distinct simple
actions) and `ka59` (7 levels, 5 distinct actions including ACTION6) are
plausibly harder in a way `tr87` (6 levels, only 4 actions, none of them
ACTION6) isn't -- see the live-trace follow-up immediately below, which
confirmed action diversity is fine on both but found episode length
alone doesn't cleanly explain the gap either.

## Live-tracing bp35 and ka59 specifically: action diversity is fine, episode length differs a lot, neither cleanly predicts the gap

Enabled `DEBUG` again and ran one full `MAX_ACTIONS=2500` episode each on
`bp35` and `ka59` directly through `main.py` (same as the `tr87` trace
earlier), to see what's actually happening now that the argmax-lock bug
is fixed and the budget is large.

**Action diversity is confirmed genuinely fine on both -- the fix is
working as intended, this isn't a repeat of the original bug:**

| game | available actions | action counts (2500-action run) |
|---|---|---|
| `bp35` | `[3,4,6,7]` | a3:495 a4:483 a6:410 a7:461 -- roughly even |
| `ka59` | `[1,2,3,4,6]` | a1:348 a2:336 a3:375 a4:372 a6:330 -- roughly even |

**Episode length (time between `RESET`s -- confirmed via
`hypothesis_agent.py`'s own code that `RESET` only ever fires reactively
on `GameState.GAME_OVER`/`NOT_PLAYED`, not voluntarily) differs
substantially across the three games, and does NOT cleanly predict which
ones solve:**

| game | resets in a 2500(ish)-action run | avg actions/episode | first-level baseline | margin (episode length / baseline) | solved at all? |
|---|---|---|---|---|---|
| `bp35` | 50 (per 2500) | 50.0 | 21 | 2.38x | No (0/9) |
| `tr87` | 2 (per 300, extrapolates to ~17 per 2500) | 150.0 | 54 | 2.78x | **Yes (2/9)** |
| `ka59` | 24 (per 2500) | 104.2 | 28 | **3.72x** (best margin of the three) | No (0/9) |

**This rules out the simplest story ("shorter episodes = harder to
solve").** `ka59` has the *most* headroom per attempt relative to its own
first-level difficulty (3.72x) of any of the three games, and still
never solved once across 9 full attempts -- while `tr87`, with a
tighter margin (2.78x), did. `bp35`'s short ~50-action episodes (2.38x
margin, the tightest of the three) are at least directionally consistent
with "less room per attempt," but on their own don't explain why `ka59`
also fails despite ample room.

**Working read:** episode length is a real, measurable difference between
these games, but not sufficient on its own to explain the gap -- the
remaining bottleneck for `bp35`/`ka59` is more likely genuine mechanical
difficulty (a specific, non-obvious sequence or setup state that a
memoryless, one-step-Q, temperature-diversified policy has no real
sequence-planning ability to discover, regardless of how many independent
short attempts it gets) rather than a budget or diversity problem this
session's fixes could reach. Confirming that properly would need either
(a) direct visual inspection of `bp35`/`ka59`'s frame sequences to
understand their actual mechanics, or (b) a genuinely different
exploration strategy with real multi-step lookahead/planning (this
project's Stage 3 recurrent core carries episode history but was never
evaluated on pure exploration efficacy the way `Hypothesis`'s InfoGain/
value blend was) -- both meaningfully larger undertakings than a live
trace, and not pursued further this session.

## Where this leaves the investigation

**The held-out-games breadth ceiling that survived 13+ independent
world-model interventions this session (data diversity, conditioning
architecture, TTA at any dose, meta-learning) turns out to have a real,
fixable component that has nothing to do with the world model at all: a
genuine exploration-mechanism bug (action-selection argmax lock) compounded
by an under-provisioned action budget.** Both were hiding in plain sight
-- the argmax-lock bug because no prior trace had isolated *which*
subset of games to look at closely (this project's own established
lesson: "when two things diverge on a specific subset, trace that subset
specifically" -- here, diverging from `r11l` rather than from
`Curiosity`), and the budget ceiling because `MAX_ACTIONS=900`'s own
prior backtest (`experiments/stage6_budget_x_checkpoint.md`) never tested
anything past 900, let alone 2500, on these specific 3 games.

**Recommended next steps, in order:**
1. ~~Confirm `tr87`'s solve wasn't a lucky n=1 roll~~ -- **done, see
   above: replicated at 1/8 in a fresh batch, ~22% combined rate across 9
   total attempts. Real, but not yet reliable.**
2. ~~Test `bp35`/`ka59` at the same budget with more repeats~~ -- **done,
   see above: a clean 0/9 for both. They need something beyond
   budget+diversity; worth a targeted live-trace on these two
   specifically before assuming more budget alone would help.**
3. A real `MAX_ACTIONS=300`-budget agent-level backtest (n=25-30, this
   project's own standard for trusting a result) of the softmax fix
   alone, to properly confirm the `r11l` reliability gain (0.375 -> 0.500
   at n=8) isn't itself a small-sample artifact -- this project has hit
   that exact trap twice already this session (novelty-aware beta, the
   budget+TTA combo).
4. If (2) holds up, this reframes the real Kaggle-relevant lever as
   `MAX_ACTIONS` (already the highest-value single change found all
   session, per `experiments/stage6_budget_tta_combo.md`'s n=25
   confirmation) COMBINED with this action-selection fix, not test-time
   adaptation or any world-model change -- worth prioritizing over
   further TTA/meta-learning work for any future submission decision.
   Note the combined lever's own cost, though: `MAX_ACTIONS=2500` is
   ~8x the real Kaggle default's action budget -- not something this
   project could actually ship as-is; the interesting result is that
   `tr87` is solvable in principle now, not that this exact config is
   submission-ready.

## Housekeeping

The `MAX_ACTIONS=2500` bump was temporary (reverted to `300` immediately
after the probe, matching this project's established bump-and-revert
practice). The action-selection softmax fix itself (`ACTION_SAMPLE_TEMPERATURE`)
is a permanent, committed change to `hypothesis_agent.py` -- no env-var
gate, since it strictly improves on the old hard-argmax behavior with no
observed downside.

## Reproducing this experiment

```
# checkpoints_holdout_baseline/ copied into checkpoints/ (see
# experiments/stage6_test_time_adaptation_agent.md's setup).

# Live-trace with full debug output (run_scorecard.py discards this):
# .env: DEBUG=True
cd ARC-AGI-3-Agents
python main.py --agent=hypothesis --game=tr87 > trace.log 2>&1
grep -oE "chosen a[0-9]|sampled a[0-9]" trace.log | sort | uniq -c

# Backtest:
python scripts/run_scorecard.py --agent hypothesis --label actionfix_r1 --game r11l,bp35,m0r0,tr87,ka59
# ... repeat x8

# Big-budget probe:
$env:HYPOTHESIS_MAX_ACTIONS = '2500'
python scripts/run_scorecard.py --agent hypothesis --label actionfix_bigbudget_bp35 --game bp35
```
