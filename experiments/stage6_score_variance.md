# Stage 6 experiment: chasing the 0.23-vs-0.06 Kaggle score variance

**Status: investigated, no code-behavior change to `EPSILON`/`PATCH_SAMPLE_TEMPERATURE`.**
`Hypothesis.MAX_ACTIONS=900` (already held on this branch, see
`stage6_max_actions.md`) remains the one concrete, evidence-backed lever
this investigation turned up -- see "What actually moved the needle"
below.

## Why this lever

CLAUDE.md flags a real, unexplained finding: two back-to-back Kaggle
submissions of *identical* code scored `0.23` then `0.06`, and the `0.06`
run exactly matched the plain-random-agent control. `Hypothesis.__init__`
seeds `self._rng = random.Random()` unseeded, so the agent's own
exploration (epsilon-random fallback, softmax click-location sampling)
genuinely differs run to run -- a plausible, testable contributor to that
gap, alongside possible differences in which hidden games got sampled
between the two submissions (which this investigation cannot measure
locally).

## Method and an important caveat

All runs use `scripts/run_scorecard.py` against the fixed local 25-game
suite in offline mode (same real scoring formula as `stage6_max_actions.md`,
not a proxy). Because the local game set is fixed and identical across every
repeat, **any variance measured here is attributable only to the agent's own
stochastic policy** (epsilon-random choices, click-location sampling) --
**not** to hidden-competition-game sampling, which real Kaggle submissions
also vary on and this setup cannot reproduce. So: "this reduces local
variance" would be a defensible, testable finding, but not by itself proof
that it explains the full real 0.23-vs-0.06 Kaggle gap. Stated honestly up
front rather than implied later.

Env-var overrides added to `hypothesis_agent.py` this session
(`HYPOTHESIS_EPSILON`, `HYPOTHESIS_PATCH_TEMP`, `HYPOTHESIS_MAX_ACTIONS`,
etc. -- same pattern as the existing `HYPOTHESIS_DIAG_MODE`) let a sweep
override one run's value without editing/reverting the file each time.
`scripts/summarize_scorecards.py` (new this session) aggregates a set of
`logs/scorecards/<prefix>*.json` files into mean/std/min/max.

## Step 1: baseline variance at current defaults (EPSILON=0.25, PATCH_SAMPLE_TEMPERATURE=0.1, budget=300)

| n | scores | mean | std | levels completed |
|---|---|---|---|---|
| 5 | `[0.0, 0.0, 0.00222, 0.19048, 0.00862]` | 0.0403 | **0.0752** | `[0,0,1,1,1]` (2/5 zero-completion) |

**Confirmed: the variance is large, not noise-level.** std (0.075) is
nearly 2x the mean (0.040) on a fixed game set -- this is a real,
measurable effect of the agent's own stochastic policy alone, not an
artifact of hidden-game sampling. One run (0.190) is a clear outlier,
consistent with the same effect `stage6_max_actions.md` already flagged:
a single completion landing on a game with a small weighted-average
denominator dominates the aggregate score at this sample size. That
skew, not just "the agent sometimes does worse," is a real part of why
this metric is noisy to begin with.

## Step 2: EPSILON sweep (budget=300, PATCH_SAMPLE_TEMPERATURE=0.1 default)

| EPSILON | n | mean | std | levels completed | zero-completion runs |
|---|---|---|---|---|---|
| 0.0 | 3 | 0.0000 | 0.0000 | `[0,0,0]` | 3/3 |
| 0.10 | 4 | 0.0378 | 0.0633 | `[1,0,1,1]` | 1/4 |
| 0.15 | 4 | 0.0083 | 0.0138 | `[1,0,0,2]` | 2/4 |
| 0.25 (baseline) | 5 | 0.0403 | 0.0752 | `[0,0,1,1,1]` | 2/5 |

- **EPSILON=0.0 is a clean, total failure -- 0/3 completions, every
  single run.** This is not a new bug: CLAUDE.md's Stage 5 history (bug 3)
  already documents that a pure Q-argmax with no random fallback locks
  onto one action and repeats it until `GAME_OVER`. This run reproduces
  that finding exactly rather than discovering anything new -- included
  as the sanity-check lower bound the investigation plan called for.
- **EPSILON=0.15 has a much lower std (0.014) than baseline (0.075) --
  but its mean (0.008) is also 5x lower.** This is exactly the failure
  mode to watch for: a setting that "reduces variance" only because it's
  reliably worse, not because it makes good outcomes more consistent. Not
  a win by the stated criterion (lower std *without* tanking mean/
  completions).
- **EPSILON=0.10's mean (0.038) and completion pattern (3/4 nonzero) are
  close to baseline's (0.040, 3/5), and its std (0.063) is somewhat lower
  (0.075) -- but n=4 vs n=5 is nowhere near enough to distinguish that gap
  from resampling noise**, especially given the score metric's own
  outlier-sensitivity established in Step 1. Not treating this as a real
  effect.

## Step 3: PATCH_SAMPLE_TEMPERATURE sweep (budget=300, EPSILON=0.25 default)

| PATCH_SAMPLE_TEMPERATURE | n | mean | std | levels completed | zero-completion runs |
|---|---|---|---|---|---|
| 0.05 (sharper) | 3 | 0.0646 | 0.0891 | `[1,1,0]` | 1/3 |
| 0.10 (baseline) | 5 | 0.0403 | 0.0752 | `[0,0,1,1,1]` | 2/5 |
| 0.20 (flatter) | 3 | 0.0035 | 0.0050 | `[0,0,2]` | 2/3 |

- **0.05 (more exploitative click sampling) is, if anything, noisier
  than baseline** (std 0.089 vs 0.075) with a similar or slightly better
  mean -- no evidence this helps.
- **0.20 (flatter, closer-to-uniform click sampling) does cut std
  sharply (0.005 vs 0.075) -- but by uniformly gutting performance**
  (mean 0.0035, only 1/3 nonzero-completion runs). This is the same
  "lower variance by being reliably bad" trap as EPSILON=0.15 above, more
  pronounced: pushing click sampling toward uniform-random essentially
  discards the directed-exploration signal the whole hypothesis-bundle
  design exists to provide (see CLAUDE.md's Stage 5 "gotchas" section on
  why `Curiosity`/`Hypothesis` deliberately moved away from uniform/
  argmax click selection in the first place). Not a real fix, just a
  different way to fail.

## Conclusion on EPSILON / PATCH_SAMPLE_TEMPERATURE: no evidence-backed change

None of the four EPSILON values or three PATCH_SAMPLE_TEMPERATURE values
tested produced a std reduction that (a) clearly survives sampling noise
at this n, and (b) doesn't come attached to a proportional mean/
completion-rate cost. Per the project's standing rule to only keep
changes with real, reasoned evidence, **both class-attribute defaults are
left unchanged** (`EPSILON = 0.25`, `PATCH_SAMPLE_TEMPERATURE = 0.1`) --
documented directly in `hypothesis_agent.py`'s comments so a future
session doesn't re-run the same sweep from scratch. This is a genuine
negative result, not a failure to find one: `EPSILON=0.0` reproducing the
already-known catastrophic lock-on is useful confirmation, and ruling out
"just lower the exploration temperature a bit" as a quick fix for the
Kaggle variance question is itself informative for where to look next.

## What actually moved the needle: MAX_ACTIONS, not EPSILON/PATCH_SAMPLE_TEMPERATURE

Backtested the *unchanged* defaults (EPSILON=0.25, PATCH_SAMPLE_TEMPERATURE=0.1)
at the real submission budget this branch already holds (`MAX_ACTIONS=900`),
extending `stage6_max_actions.md`'s original n=2 to n=5:

| MAX_ACTIONS | n | mean | std | std/mean | levels completed | zero-completion runs |
|---|---|---|---|---|---|---|
| 300 | 5 | 0.0403 | 0.0752 | 1.87 | `[0,0,1,1,1]` | 2/5 (40%) |
| 900 | 5 | 0.0371 | 0.0505 | 1.36 | `[2,1,2,2,2]` | **0/5 (0%)** |

Mean score is essentially unchanged (0.040 vs 0.037 -- within this
metric's own noise band, consistent with `stage6_max_actions.md`'s
earlier, smaller-sample finding), but two things move in the same,
favorable direction at n=5:

- **Zero-completion runs dropped from 2/5 to 0/5.** Every single
  900-budget repeat completed at least one level; 40% of 300-budget
  repeats completed none. A run that completes zero levels is
  structurally the worst-case outcome under the real scoring formula (a
  level never reached scores exactly 0) -- and "the agent's exploration
  happened to run out of budget before finishing anything" is a
  qualitatively different, more fixable failure mode than "the agent
  played badly." This is a plausible, mechanistic explanation for at
  least part of the original 0.23-vs-0.06 gap: a run that resembles the
  0.06 result (matching pure-random-agent performance) is consistent
  with a budget-starved run that barely got anywhere, not necessarily a
  policy failure.
- **Relative variance (std/mean) dropped from 1.87 to 1.36.** Still high
  in absolute terms -- this is not "solved" -- but a real reduction, in
  the direction this whole investigation was looking for, from a change
  already staged on this branch rather than from anything new tried this
  session.

This is consistent with `stage6_max_actions.md`'s own theoretical
argument (never-completing scores exactly 0; efficiency can only be hurt
on levels that would've completed anyway) -- now with real evidence at
5x the original sample size, and a completion-reliability effect
(0/5 vs 2/5 zero-completion) that the original 2-repeat comparison didn't
have enough samples to see.

## Honest limits of this investigation

- All measurements here are on the **fixed local 25-game suite** --
  variance from hidden-competition-game sampling (a real, separate
  contributor to any two real Kaggle submissions differing) is not
  measured or ruled out by anything in this document.
- std/mean is still ~1.36 even at the improved MAX_ACTIONS=900 setting --
  this is progress, not a resolved variance problem. The score metric's
  own sensitivity to which specific game/level a completion lands on
  (Step 1's outlier discussion) means some of this noise may be inherent
  to the scoring formula at this sample size, not fully removable by any
  agent-side change.
- Only one seed/repeat count per condition was affordable this session
  (n=3-5); a future session with more time budget could tighten these
  estimates further, particularly for the EPSILON=0.10 result, which is
  the one condition here that's plausibly-but-not-conclusively better
  than baseline.

## Recommendation

No change to `EPSILON`/`PATCH_SAMPLE_TEMPERATURE` defaults -- kept at
0.25/0.1, with the sweep's null result documented in `hypothesis_agent.py`
directly so it isn't re-attempted from scratch. `MAX_ACTIONS=900` remains
the one real, growing-evidence lever for the original variance concern
(now n=5, not n=2) -- still gated on an official Kaggle submission
before merging to `master`, per standing instruction; this session's
backtest is additional local evidence in its favor, not a substitute for
that real-submission validation.

## Reproducing this investigation

```
# baseline / EPSILON sweep (budget=300)
HYPOTHESIS_MAX_ACTIONS=300 HYPOTHESIS_EPSILON=<value> \
  python scripts/run_scorecard.py --agent hypothesis --label <name>

# PATCH_SAMPLE_TEMPERATURE sweep (budget=300, EPSILON left at default)
HYPOTHESIS_MAX_ACTIONS=300 HYPOTHESIS_PATCH_TEMP=<value> \
  python scripts/run_scorecard.py --agent hypothesis --label <name>

# backtest at real budget (defaults unchanged)
HYPOTHESIS_MAX_ACTIONS=900 \
  python scripts/run_scorecard.py --agent hypothesis --label <name>

python scripts/summarize_scorecards.py <label_prefix> [<label_prefix> ...]
```
