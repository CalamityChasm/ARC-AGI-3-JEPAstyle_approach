# Stage 6 -- how much does the plain random agent's own score vary run-to-run?

## Why this exists

The `stage6-object-identity` checkpoint's first real Kaggle submission
scored `0.00` -- the lowest of six real submissions so far, and notably
*below* the previously-assumed floor of `0.06` (the plain official
`random` agent's only-ever real Kaggle submission score; see CLAUDE.md's
"Kaggle competition submission" section). The user pushed back sharply on
an early read of this result: "I've never seen a random agent score 0.0
before -- it would require the agent not moving or making any change on
the game board for every trial." That's a fair objection, and separate
diagnostic work this session already ruled out a model-loading crash on
Kaggle's real environment (checkpoint loading and a forward pass both
verified clean there).

That leaves one clean, checkable question this project has never actually
measured: **how much does the plain random agent's own score vary
run-to-run?** Only one real Kaggle data point for `random` exists
(`0.06`) -- its variance has never been characterized, even locally. If a
working random-ish policy can plausibly land on zero total levels
completed across a full run some meaningful fraction of the time (given
the tight 300-action-per-game budget across a large number of
largely one-shot-per-game episodes), then the object-identity
checkpoint's `0.00` might be well within normal variance rather than a
sign of anything broken. If zero-total-level runs are extremely rare for
`random`, that argues `0.00` is a genuinely surprising result worth more
suspicion.

## Method

- Agent: the plain, unmodified `random` agent
  (`ARC-AGI-3-Agents/agents/templates/random_agent.py`, `Random` class).
  `MAX_ACTIONS` temporarily bumped from its repo default of `80` to
  `300` to match the real Kaggle submission config, run for this sweep,
  then reverted immediately afterward (see "Reverted" section below) --
  same bump-and-revert pattern this project has used for every prior
  matched-budget comparison (Stage 2's `Random` history, Stage 5
  follow-up's `Memory` teacher pass, etc.).
- Ran `python main.py --agent=random` (all 25 local public games in one
  process, no `--game` filter) N times in a row, each a fully independent
  process invocation (`Random.__init__` reseeds off wall-clock time +
  game-id hash, so repeats are not correlated by a shared seed).
- Parsed each repeat's `--- FINAL SCORECARD REPORT ---` JSON block
  (already printed to stdout by `main.py`'s own `cleanup()` handler) for
  `total_levels_completed` and the per-environment breakdown, rather than
  reading the `*.recording.jsonl` files directly -- the scorecard report
  already aggregates exactly the numbers needed and avoids re-parsing large
  recording files.
- Deleted all recording files after each repeat (disk hygiene -- see
  CLAUDE.md's Gotchas section on `recordings/` growing unbounded).
- `main.py` in offline mode reliably exits with a non-zero code
  (documented harmless quirk, see CLAUDE.md's Environment setup step 5) --
  the scorecard JSON is still correctly printed to stdout before that
  exit, so this doesn't affect the parse.

## Results

**30 independent repeats**, each covering all 25 local public games at
`MAX_ACTIONS=300`. Sanity checks confirmed the config was applied
consistently across every repeat: `total_levels` (the sum of each game's
own level count -- a property of the games, not the agent) was exactly
`183` in all 30 repeats, and `total_actions` was exactly `7525` in all 30
repeats (`25 games x 301` -- 300 real actions plus the terminal
RESET-triggering step each `GAME_OVER`/budget-exhaustion cycle counts --
consistent with every game genuinely running the full budget, not
truncating early or double-counting).

Per-repeat `total_levels_completed` (all 30 values, in run order):

```
2, 1, 2, 1, 1, 1, 2, 1, 0, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1, 1, 0, 1
```

| stat | value |
|---|---|
| n | 30 |
| mean | 1.033 |
| std (sample) | 0.615 |
| min | 0 |
| max | 2 |
| **P(zero)** | **5/30 = 0.167 (16.7%)** |

Histogram:

```
0 levels: 5 repeats  #####
1 level:  19 repeats ###################
2 levels: 6 repeats  ######
```

`distinct_games_with_levels` per repeat is identical to
`total_levels_completed` in every single repeat -- across this entire
30-repeat sample, no repeat ever completed more than one level on the
same game. All observed level completions are "first level of some
game," never a second level on top.

**Per-game breakdown (repeats with >=1 level completed on that game, out
of 30; total levels summed across all 30 repeats):**

| game | repeats hit | total levels (of 31 grand total) |
|---|---|---|
| `sp80` | **25 / 30 (83%)** | 25 |
| `cd82` | 2 / 30 | 2 |
| `ar25` | 1 / 30 | 1 |
| `r11l` | 1 / 30 | 1 |
| `m0r0` | 1 / 30 | 1 |
| `tr87` | 1 / 30 | 1 |

**This is the single most important structural finding in this data:**
almost the entire "random-agent floor" here is one game, `sp80`, which a
plain random policy completes its first level of within a 300-action
budget **83% of the time** by chance alone. This matches CLAUDE.md's own
Stage 2 history flagging `sp80` as unusually easy for random ("Random is
actually *more action-efficient* on the one game (`sp80`) both agents
find easy... a game where the fastest path is close to lucky random
search"). The other 5 games each contributed exactly one level across
the entire 30-repeat sample -- essentially noise-level hits, not
reliable sources of score. **The 16.7% zero-rate is, almost mechanically,
close to the complement of `sp80`'s own 83% hit rate** (`1 - 0.83 =
0.167`, an exact match): on the 25 local games, whether a given repeat of
`random` scores zero or not is close to a single coin flip on whether
`sp80` happened to resolve within budget that run, not a broad,
independent accumulation of small chances across many games.

(Wall-clock cost: 30 repeats took 56.6 minutes total, mean 113.2s/repeat,
min 84.1s, max 129.0s -- some slowdown across the sweep from CPU
contention with other processes running concurrently on this machine,
not a code issue.)

## Comparison to CLAUDE.md's existing 8-repeat number

CLAUDE.md's Stage 2 section documents a matched 8-repeat, 25-game,
`MAX_ACTIONS=300` comparison run for `random`: **"Random: 10 total levels
across 8 repeats, 2 distinct games (`sp80`, `cd82`)"** -- mean 1.25
levels/repeat.

This sweep's 30-repeat mean is **1.033** (sample std 0.615, so
SE = 0.615/sqrt(8) = 0.217 at n=8). The old 8-repeat mean of 1.25 sits
about **1 SE above** this larger-sample mean -- a small, unremarkable
deviation, not an unusually good or bad historical draw. The two distinct
games named in that old entry (`sp80`, `cd82`) are also exactly the top
two games in this sweep's per-game breakdown, which is a good
consistency check that both samples are drawing from the same underlying
process rather than measuring something different.

One thing the old 8-repeat entry's aggregate number can't answer on its
own: whether any *individual* one of those 8 repeats was itself a
flat-zero run (only the 8-repeat total is documented, not each repeat's
own value). Using this sweep's directly-measured `P(zero) = 0.167` as an
estimate of the per-repeat zero probability, the probability that **at
least one** of 8 independent repeats lands on zero is
`1 - (1 - 0.167)^8 ~= 0.767` -- roughly a **3-in-4 chance**. So it's
quite likely that a real zero-completion repeat was already sitting
inside that old 8-repeat sample even though its *aggregate* total (10
levels) looked healthy -- pooled totals across repeats can mask individual
zero-runs, another reason not to read a single Kaggle submission's score
as representative of "how `random` typically does."

## Limitation: this measures known-game variance, not novel-game variance

This sweep runs the same 25 local public games that `random` (and every
other agent in this project) has always been tested on. The real Kaggle
evaluation runs on ~110 largely-*novel* hidden games. This experiment
therefore measures "how much does `random`'s score vary run-to-run on
games it has effectively been tuned/observed against," not "how much does
`random`'s score vary on truly unseen games." Those could have
meaningfully different variance -- e.g. if novel games are on average
harder to stumble into a level-up on by pure chance than these 25
(no reason a priori to assume they're easier), the real P(zero) on
Kaggle's hidden set could be higher than what's measured here, not lower.
This experiment is a useful, honest proxy -- not a direct measurement of
the actual Kaggle-scoring distribution.

## Calibrated interpretation for the object-identity checkpoint's `0.00`

**What this does support:** a real, working, non-crashed policy landing
on zero total level completions across a full 25-game sweep is not a
freak, near-impossible event -- on these local games, at this exact
`MAX_ACTIONS=300` budget, it happened in **5 of 30 (16.7%)** independent
plain-`random` repeats. That directly answers the user's specific
objection ("I've never seen a random agent score 0.0 before") at least
for this local proxy: zero-completion runs of a genuinely-functioning
random-ish agent are not rare here, they're roughly a 1-in-6 outcome.
Combined with the separately-run Kaggle diagnostic this session (checkpoint
loading and a forward pass both verified clean on the real Kaggle
environment), a `0.00` score is consistent with "the agent played for
real and got an unlucky roll," not only with "the agent never played a
single action."

**What this does not support, and where the honest uncertainty remains:**

1. **This is a known-game measurement, standing in for an unknown-game
   question.** The local zero-rate is close to mechanically determined by
   one single easy game (`sp80`, an 83% hit rate) that random can stumble
   into within budget. The real Kaggle evaluation runs on ~110 largely
   *novel* hidden games. If none of those hidden games behave like a
   local "freebie" as forgiving as `sp80` -- and there's no principled
   reason to assume one does -- the *true* P(zero) on the hidden set could
   be substantially higher than 16.7%, not lower. Symmetrically, if a few
   hidden games happen to be even easier than `sp80` for a random-ish
   policy, the true P(zero) there could be lower. This experiment cannot
   distinguish those cases; it only establishes that P(zero) > 0 is
   ordinary, not that P(zero) is *specifically* ~17% at Kaggle's actual
   scale.
2. **This measures `random`'s variance, not the object-identity
   checkpoint's variance.** `Hypothesis` (which the object-identity
   checkpoint plugs into) is a directed-exploration agent, not a uniform
   random one -- its own zero-rate could be higher *or* lower than
   `random`'s, depending on whether its exploration strategy is more or
   less likely to stumble onto a hidden-game analog of `sp80` within
   budget. Nothing in this experiment measures that directly.
3. **Kaggle's real score computation may not reduce to a simple
   levels-completed count** the way this local sweep's proxy metric does
   (this project has not independently verified the exact
   scoring/normalization formula the competition uses across ~110 games
   with heterogeneous level counts) -- so even a perfect measurement of
   local `random` variance doesn't mechanically translate into the
   competition's own `0.00`-`1.00` scale.

**Net read:** this result meaningfully *weakens* the case that the
object-identity checkpoint's `0.00` indicates something broken, by
directly demonstrating that a working, unmodified `random` policy
produces a flat-zero outcome at a non-negligible, double-digit-percent
rate under this exact action budget on known games. It does **not**
confirm the object-identity checkpoint itself is fine, and it does not
pin down what its true zero-rate would be on the actual ~110-game hidden
set. Treating the `0.00` result as "explained" would overclaim what a
known-game random-agent variance measurement can show about a novel-game,
directed-exploration-agent result; treating it as "still fully
unexplained/suspicious" would understate what this experiment adds. The
honest position is in between: **`0.00` is no longer a surprising outlier
relative to a demonstrated real baseline distribution, but confirming it
specifically for the object-identity checkpoint (rather than for
`random`) still needs either another same-config Kaggle resubmission or
a novel-game local proxy for the checkpoint itself, neither of which this
experiment provides.**

## Reverted

`random_agent.py`'s `MAX_ACTIONS` was reverted to its repo default (`80`)
after this sweep -- confirmed via `git diff` before committing.
