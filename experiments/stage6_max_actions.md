# Stage 6 experiment: raising `Hypothesis.MAX_ACTIONS` (300 -> 900)

**Status: candidate, not merged.** Kept on `stage6-score-optimization`
per standing instruction -- only merges to `master` after an official
Kaggle submission validates it. **Update (2026-07-17, see bottom of this
doc): a properly matched n=8 backtest refined the story** -- this is a
real reliability improvement (zero-completion runs dropped from 3/8 to
0/7), not a clear mean-score improvement (means are statistically
indistinguishable at this sample size). A real Kaggle submission of this
code scored `0.22`, close to the historical best. Worth keeping, but
described accurately rather than oversold.

## Why this lever

`MAX_ACTIONS = 300` was chosen early in Stage 2/3/5 as a convenient
matched-comparison budget, before `rules.md`'s real scoring formula was
fully internalized:

- **Completion is not partial credit.** A level never reached scores
  exactly 0 for that level -- worse than a slow completion, which still
  scores `min(baseline_actions/agent_actions, 1.0) ** 2 > 0`. Running out
  of action budget mid-level is the single worst outcome the formula can
  produce for that level.
- **Efficiency only penalizes being slower than the human baseline** --
  there's no reward for finishing in fewer actions than baseline (the
  ratio is capped at 1.0 before squaring), so a bigger budget can only
  hurt efficiency on levels that *would have* completed within the old,
  smaller budget anyway. It can't hurt levels that were already fast.
- **The harness runs one Python `Thread` per game** (`agents/swarm.py`),
  all started together and sharing the GIL -- CPU-bound work (our MoE
  forward passes) serializes across every concurrent game's thread. A
  bigger per-game action budget is therefore not free: it could in
  principle starve other games of wall-clock inside the real submission's
  9-hour cap. This is the real risk this experiment needed to check.

## Method

Used `scripts/run_scorecard.py` (new this session) against the local
25-game suite in offline mode, which -- discovered while setting this up
-- already computes the **real** scoring formula locally, including
actual human `level_baseline_actions` per level. This is ground truth,
not an approximation; prefer it over `scripts/score_efficiency.py`'s
hand-built proxy (kept only as a fallback for cases where the real
scorecard isn't available).

Two repeats each of `MAX_ACTIONS=300` (current `master`) and
`MAX_ACTIONS=900` (this branch), all 25 local games per repeat, unchanged
checkpoints. Two repeats, not the project's usual 8, given each 900-budget
repeat costs ~9 minutes wall-clock and this needed to stay scoped.

## Results

| condition | repeat | score | levels completed | total actions | wall-clock |
|---|---|---|---|---|---|
| 300 (baseline) | 1 | 0.00719 | 2 | 7,525 | ~2m46s (r2's time; r1 untimed) |
| 300 (baseline) | 2 | 0.00000 | 0 | 7,525 | 2m46s |
| 900 (this branch) | 1 | 0.01368 | 2 | 22,525 | 8m53s |
| 900 (this branch) | 2 | 0.13638 | 1 | 22,525 | 8m35s |

Baseline average score: **0.0036**. Bumped average score: **0.0750**
(~21x higher) -- but this is driven almost entirely by one repeat
(0.136) where a single completion happened to land on a game with very
few total levels (small weighted-average denominator), which dominates
the aggregate at this sample size. **Not treating this as "900 is 21x
better"** -- treating it as "nothing here suggests 900 is worse, and nothing
about the completion counts (2,0 vs 2,1) suggests the GIL-contention risk
materialized locally."

Wall-clock scaled roughly linearly with the action-budget ratio (3.17x
wall-clock for a 3x budget, not super-linear) -- no sign of GIL
contention causing disproportionate slowdown at this scale. Extrapolating
linearly to the real submission's 110-game, one-thread-per-game harness:
25 games at 900-budget took ~8m50s locally; 110 games (4.4x more
threads) would extrapolate to roughly 39 minutes, well inside the 9-hour
cap -- and that's a pessimistic extrapolation, since real scored runs are
network-bound (HTTP round-trip per action to the gateway) rather than
CPU-bound like this fast local offline-mode test, so true per-action cost
in a real run is dominated by network latency, not our own model's
compute time either way.

## Verdict

Real evidence, honestly reported:
- **No sign this hurts.** Completion counts were comparable across
  conditions; wall-clock scaled linearly, not explosively.
- **Not enough sample size to claim it helps by 21x** -- that number is
  an artifact of one lucky completion on a short-level-count game, not a
  robust effect. A fair statement is "directionally non-negative to
  positive, on far too small a sample to be sure."
- The *theoretical* case (never-completing scores exactly 0; efficiency
  can only be hurt on levels that would've completed anyway) remains the
  strongest argument here, independent of this run's noisy numbers.

**Recommendation: keep 900 as the candidate on this branch.** Don't merge
to `master` until an official Kaggle submission using this branch's
kernel confirms it doesn't regress the real score -- per standing
instruction, local backtest evidence at this sample size isn't sufficient
on its own to promote a change that touches the real submission.

## Reproducing this comparison

```
python scripts/run_scorecard.py --agent hypothesis --label <name>
```
Saves the real scorecard to `logs/scorecards/<name>.json` (gitignored --
local artifact, not committed) and prints the top-line score/completion
summary. Run on `master` (300) vs this branch (900) for a fresh
comparison; more repeats would narrow the noise band further if a future
session has the time budget for it.

## Update (2026-07-17): matched n=8 comparison -- a more nuanced picture, plus a real Kaggle data point

Since the n=2-vs-n=2 comparison above, two more pieces of evidence landed:

**A real Kaggle submission of this branch's code scored `0.22`** (see
CLAUDE.md's Kaggle section) -- close to the best prior score (`0.23`,
`MAX_ACTIONS=300`) and far from the worst (`0.06`, the same 300-budget
code re-run, matching the plain random-agent floor). Encouraging, but
explicitly documented there as not conclusive on its own at n=1.

**A properly matched local backtest, n=8 both sides** (`matched300_r1..8`
vs `matched900_r1..8`, same branch defaults otherwise, run back-to-back
same session -- one `matched900` repeat, r6, was lost mid-run to a
disk-full incident unrelated to this experiment, see CLAUDE.md's Gotchas
section, leaving n=7 for that side):

| condition | n | mean score | std | mean levels completed | zero-completion runs |
|---|---|---|---|---|---|
| 300 | 8 | 0.0296 | 0.0418 | 1.12 | 3/8 |
| 900 | 7 | 0.0250 | 0.0430 | 1.71 | **0/7** |

**This complicates the simple "bigger budget can only help" theoretical
argument made above.** Mean *score* is statistically indistinguishable
between the two conditions (both ~0.025-0.030, overlapping std) -- 900
is not a clear win on raw score at this larger, more reliable sample
size, unlike what the noisy n=2 comparison suggested. But mean *levels
completed* is clearly higher (1.71 vs 1.12) and, more importantly,
**zero-completion runs dropped from 3/8 to 0/7** -- the worst-case outcome
(a level never reached, scoring exactly 0) got meaningfully rarer.

The likely reason score didn't rise to match: this doc's original
efficiency argument ("a bigger budget can only hurt efficiency on levels
that would've completed anyway") assumed a bigger budget just *extends*
the same trajectory prefix unchanged. It doesn't -- `MAX_ACTIONS` is the
per-episode cap the agent's own exploration dynamics (epsilon-random
fallback, exploit-on-level-up repeats, InfoGain scoring) unfold within,
so a longer cap can change *which actions get taken even early in an
episode*, not just whether it eventually times out. A level that
completes under both budgets isn't guaranteed to complete in the same
number of actions in both -- so efficiency-component regressions on
otherwise-fine levels are a real possible mechanism here, not just a
theoretical non-issue as originally framed.

**Revised verdict**: `MAX_ACTIONS=900` is best understood as a
**reliability lever, not a mean-score lever** -- it raises the floor
(fewer catastrophic zero-completion levels) without clearly raising the
average. That's still worth keeping given `rules.md`'s scoring rewards
consistency across 110 hidden games more than any single game's peak, and
it's consistent with the real submission landing near the historical best
rather than the historical worst -- but it should be described accurately
as "more consistent, not clearly higher-scoring on average" going
forward, not oversold as a straightforward improvement.
