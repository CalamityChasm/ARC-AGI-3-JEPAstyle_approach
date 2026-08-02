# Stage 6: a novelty-aware beta cap for held-out games

## Motivation

CLAUDE.md's Stage 6 addendum established two things that, put together,
suggest a concrete, cheap fix worth testing directly:

1. The *gated* MoE prediction (what `changed-patches` measures, what the
   value head's inputs are built on, and what `HypothesisBundle`'s
   confidence-entropy update implicitly tracks via *observed* per-expert
   error against real outcomes) collapses to near-identity on any game
   outside the training vocab.
2. The raw, *ungated* per-expert disagreement (InfoGain) does **not**
   collapse the same way (`scripts/diagnose_infogain_holdout.py`:
   held-out/trained ratio 0.999).

`Hypothesis`'s `Q(s,a) = (1-beta)*InfoGain(a) + beta*V(next_state(a))`
blend derives `beta` from the entropy of the Bayesian confidence
distribution over the 8 experts, built from *observed* prediction error
in the current episode. That signal has no a priori awareness of whether
the current game is one the model was actually trained on -- and there's
a real risk it reports false confidence on an unfamiliar game: if the
gated prediction has collapsed toward "predict no change," and the game
itself doesn't change all that often under exploratory play either, the
observed per-expert error can look small and consistent (i.e.
"confident") purely as an artifact of collapse, handing control to the
value head exactly where its own conditioning is least trustworthy.

`self.game_id not in game_vocab` (the exact lookup `_init_models` already
uses for `self.game_idx`) is a much cheaper, more reliable a priori
signal for "this is a genuinely unfamiliar game" than anything derived
from in-episode observed error -- and it's exactly the condition that
will hold for essentially every real hidden Kaggle game. This experiment
adds a narrowly-scoped override: on such games, cap (never raise) `beta`
toward InfoGain.

## Implementation

`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`:
- `_init_models` now sets `self._is_novel_game = self.game_id not in
  game_vocab` alongside the existing `self.game_idx` lookup.
- A new class attribute, `NOVELTY_BETA_CAP = float(os.getenv(
  "HYPOTHESIS_NOVELTY_BETA_CAP", "0.15"))`, mirroring the project's
  existing `HYPOTHESIS_EPSILON`/`HYPOTHESIS_MAX_ACTIONS`-style env-var
  override pattern.
- In `_choose_action_inner`'s Q-scoring branch: `if self.FORCE_BETA is
  None and self._is_novel_game: beta = min(beta, self.NOVELTY_BETA_CAP)`.
  Applied *after* the real entropy-driven beta is computed, so the
  confidence signal can still push beta lower than the cap if it wants
  to -- it just can never push it above the cap on an unfamiliar game.
  Deliberately skipped when `FORCE_BETA` is set (that mechanism already
  pins beta directly for a different, existing ablation) and has zero
  effect on any game in `game_vocab` -- Stage 5 follow-up 2 already
  validated the full adaptive blend as the right design on familiar
  games, and this change does not touch that path at all.

Verified directly (not just by code inspection) via a live `DEBUG=True`
trace on one held-out game (`r11l`) and one trained game (`ft09`), both
with `HYPOTHESIS_NOVELTY_BETA_CAP=0.15`: on `r11l` every logged beta
after the first few steps sits at exactly `0.150` (the cap binding); on
`ft09` beta climbs freely past the cap (observed up to `0.471` in the
same trace length) -- confirming the cap fires on the unfamiliar game and
is a true no-op on the familiar one, exactly as designed.

## Part 1: quick diagnostic sweep -- what does held-out beta actually look like uncapped?

Before picking a cap value, measured the real, uncapped beta distribution
on held-out vs. trained games directly (`scripts/
diagnose_hypothesis_beta_holdout.py`, replaying real recorded episodes
through `HypothesisBundle`'s confidence update without playing games --
same methodology as the original `scripts/diagnose_hypothesis_beta.py`
that picked the `decay=0.8` value in Stage 5, but split by held-out vs.
trained game and using the `stage6-game-holdout` fold-1 checkpoint
instead of production):

| group | n | mean beta | std | min | max | frac > 0.15 | frac > 0.20 | frac > 0.25 | frac > 0.30 |
|---|---|---|---|---|---|---|---|---|---|
| held-out games (never trained on) | 2384 | 0.2450 | 0.0165 | 0.116 | 0.400 | 0.997 | 0.987 | 0.086 | 0.014 |
| trained games (in vocab) | 1595 | 0.2769 | 0.0537 | 0.135 | 0.971 | 0.998 | 0.989 | 0.604 | 0.278 |

Two things stand out. First, the false-confidence concern is real but not
extreme: held-out beta never reaches the high-confidence band trained
games can reach (max 0.400 vs. 0.971; trained games spend real time above
0.5 in a way held-out games essentially never do) -- so this isn't a
runaway-certainty collapse like the pre-`decay`-fix bug from Stage 5.
Second, held-out beta is also *never particularly low* -- it sits in a
narrow, moderate 0.12-0.40 band, with virtually all of its mass (99.7%)
above 0.15 and 98.7% above 0.20. That's the actual problem this cap
targets: held-out beta doesn't swing toward "very uncertain, trust
InfoGain" the way you'd want on a game the model has no real signal
about -- it sits at a middling, moderately-confident value almost all the
time, handing V roughly 20-30% of the blend on a signal that's already
been shown to be far less trustworthy there than on trained games.

This directly informed the cap choice: a cap of 0.30 would bind on only
~1.4% of held-out transitions (nearly a no-op given where the real
distribution sits); a cap of 0.15 binds on essentially the entire
held-out distribution (99.7%), giving a real, consistent nudge toward
InfoGain without being the maximal all-or-nothing choice (`0.0`, full
override to pure InfoGain on every unfamiliar-game decision, untested at
the agent level by this diagnostic and a larger behavioral change than
the evidence above calls for). **Picked `NOVELTY_BETA_CAP=0.15`** as the
real, evidence-based operating point for the agent-level backtest below,
rather than guessing among the three candidates blind.

## Part 2: real agent-level backtest

The test that actually matters: does capping beta translate into better
real exploration on unfamiliar games, or does it fail to show up at the
agent level -- the same way test-time adaptation's own representation-
level win (`experiments/stage6_test_time_adaptation_agent.md`) didn't
cleanly translate either?

Protocol matches the most recent precedent
(`stage6-test-time-adaptation-agent`): `scripts/run_scorecard.py`
(captures the harness's own `FINAL SCORECARD REPORT`, which already
implements `rules.md`'s real Kaggle scoring formula), `MAX_ACTIONS=300`
(`Hypothesis`'s unmodified default), the `stage6-game-holdout` fold-1
checkpoint (`checkpoints_holdout_baseline/`: 5 games never trained on --
`r11l`, `bp35`, `m0r0`, `tr87`, `ka59` -- plus a matching `value_head.pt`
trained against this checkpoint's own encoder, avoiding Stage 5's
already-documented value-head/encoder latent-space mismatch bug), n=8
repeats per condition on the held-out games. `HYPOTHESIS_NOVELTY_BETA_CAP=
0.15` (cap ON) vs. `HYPOTHESIS_NOVELTY_BETA_CAP=1.0` (cap OFF -- a
mathematical no-op since beta is already bounded in [0, 1], exactly
reproducing pre-this-change behavior on identical code).

### Held-out games (`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`), n=8 each

| condition | mean score | std | mean levels completed | total levels | distinct games solved |
|---|---|---|---|---|---|
| cap ON (0.15) | **0.0603** | 0.1377 | **0.50** | **4** | 1 (`r11l`) |
| cap OFF (1.0, today's default) | 0.0113 | 0.0246 | 0.38 | 3 | 1 (`r11l`) |

Per-run detail: cap ON solved `r11l` in 4 of 8 runs (`r3`, `r5`, `r7`,
`r8`; scores 0.163, 0.029, 0.102, 2.116); cap OFF solved it in 3 of 8
(`r3`, `r7`, `r8`; scores 0.031, 0.379, 0.041). Neither condition ever
completed a level on `bp35`, `m0r0`, `tr87`, or `ka59` -- consistent with
the test-time-adaptation agent-level backtest's own finding that these
four games are essentially never solved by this agent within 300 actions
regardless of checkpoint or mechanism, `r11l` being the one exception both
backtests share.

**Cap ON is ahead on every metric measured here (mean score, mean levels,
total levels), and by a proportionally large margin (~5x mean score,
~1.3x total levels).** At n=8 on a metric this sparse (both conditions
solve a level in under half of all runs, on a single game out of five),
this is a real, honest positive signal but not conclusive proof on its
own -- one extra completion (4 vs. 3) is well within what unseeded
run-to-run variance alone could produce at this sample size (this
project's own `Hypothesis.__init__` seeds `self._rng` with no fixed
seed, and CLAUDE.md's Kaggle section already documents wide same-code
score variance, e.g. `0.06` vs. `0.23` on identical submissions). The
mean-score gap is inflated by cap ON's `r8` run scoring 2.116 (a single
fast, lucky completion, the same "one outlier run drives the mean"
pattern `experiments/stage6_test_time_adaptation_agent.md` already
flagged for this exact metric) -- the levels-completed column (4 vs. 3)
is the more robust comparison, and it still favors cap ON, just less
dramatically.

### Trained games sanity check (same checkpoint, 6 of the fold's other 20 games: `ft09`, `sp80`, `cd82`, `lp85`, `vc33`, `tn36`), n=5 each

| condition | mean score | std | mean levels completed | total levels | distinct games solved |
|---|---|---|---|---|---|
| cap ON (0.15) | 0.0047 | 0.0095 | 0.20 | 1 | 1 (`ft09`) |
| cap OFF (1.0) | 0.4762 | 0.3888 | 0.60 | 3 | 1 (`cd82`) |

At first glance this looks like a large regression on trained games --
but the cap is provably inert here: all 6 games (`ft09-0d8bbf25`,
`sp80-589a99af`, `cd82-fb555c5d`, `lp85-305b61c3`, `vc33-5430563c`,
`tn36-ef4dde99`) are confirmed present in `checkpoints_holdout_baseline/
game_vocab_moe.json`, and `self._is_novel_game` is `False` for all of
them by construction -- `beta = min(beta, NOVELTY_BETA_CAP)` is
structurally never reached on these games regardless of which env-var
value is set. Directly inspected which game produced the swing: `cd82`
was solved in exactly 301 actions with an identical score
(`4.7619...`, i.e. the same fast, low-variance solve path) in 3 of 5
cap-OFF runs and 0 of 5 cap-ON runs -- a single game's stochastic
solve-or-not outcome (driven by `EPSILON=0.25` random exploration, the
opening probes, and softmax click sampling, none of which this change
touches) dominating a 5-run sample, the exact same "one game's binary
outcome swings the whole mean" pattern already seen on the held-out
side and repeatedly documented elsewhere in this project (Stage 5's
`sp80`, the TTA agent backtest's `r11l` outliers). **This is unseeded
run-to-run variance, not a regression caused by this change** -- but it's
a useful reminder of just how noisy this metric is at n=5, and worth
being explicit about rather than glossing over just because the
direction happens to be unfavorable-looking.

## Verdict

**A real, evidence-based cap (chosen from an actual measured beta
distribution, not a guess) shows a positive signal at the agent level on
held-out games -- ahead on every metric at n=8, with the more robust
levels-completed comparison (4 vs. 3) still favoring it after discounting
one outlier-driven score.** This is a more encouraging result than
test-time adaptation's own agent-level backtest, which showed no
detectable benefit (and, on the noisier score metric, slightly favored
being off) at the same n=8 on the same 5 games. It is not, however,
proof at the standard this project holds itself to elsewhere: one extra
level completion out of 8 attempts is exactly the kind of margin that
CLAUDE.md's own Kaggle section (`0.06` vs. `0.23` on identical
resubmitted code) and this project's repeated "raw win/level counts are
noisy at n=8" caveat (Stage 2, Stage 5, the TTA agent backtest) say
needs a larger sample or a higher-resolution metric before being called
conclusive.

The trained-games sanity check is clean on the metric that actually
matters for this change: the cap is structurally inert on all 6 games
tested (confirmed by direct inspection of `game_vocab_moe.json` and the
scorecards' own per-environment ids, not just by code reading), and the
large-looking score gap there is fully attributable to the same kind of
unseeded solve-or-not variance on one game (`cd82`) that this project has
hit repeatedly elsewhere -- not evidence this change regresses familiar-
game behavior.

**Honest bottom line:** this is a real, mechanistically well-motivated
change (grounded directly in the InfoGain-survives-collapse finding, not
speculation) with a positive, non-noise-implausible signal at the agent
level -- the most encouraging held-out-games result of this whole Stage 6
investigation after test-time adaptation's own inconclusive backtest. But
it is one data point at n=8 on an extremely sparse metric, exactly the
regime where this project has previously seen real component-level
improvements (the teacher-policy value head, test-time adaptation itself)
fail to produce a statistically distinguishable agent-level signal. Worth
keeping enabled by default (`NOVELTY_BETA_CAP=0.15`, no regression risk
on familiar games, one extra `if` check's worth of overhead) and worth a
larger-n follow-up (25-30 repeats, per this project's own standing
recommendation for exactly this class of problem) before treating it as
a confirmed win rather than a promising lead.

## Reproducing this experiment

```
# checkpoints_holdout_baseline/ (see experiments/stage6_game_holdout.md's
# training command, plus a value_head.pt trained against its own encoder
# per experiments/stage6_test_time_adaptation_agent.md's "Checkpoint
# setup" section) and the local recordings corpus must be present --
# both gitignored. Swap checkpoints_holdout_baseline/* into checkpoints/
# before running (this experiment used an independent local checkpoints/
# copy, not the shared production directory, specifically to avoid
# touching other concurrent work on the same machine).

python scripts/diagnose_hypothesis_beta_holdout.py

$env:HYPOTHESIS_NOVELTY_BETA_CAP = '0.15'
python scripts/run_scorecard.py --agent hypothesis --label heldout_cap_on_r1 --game r11l,bp35,m0r0,tr87,ka59
# ... repeat x8, then $env:HYPOTHESIS_NOVELTY_BETA_CAP = '1.0' for the OFF condition

# Trained-games sanity check (same checkpoint):
python scripts/run_scorecard.py --agent hypothesis --label trained_cap_on_r1 --game ft09,sp80,cd82,lp85,vc33,tn36
# ... repeat x5 per condition

python scripts/summarize_scorecards.py holdout_cap_on holdout_cap_off trained_cap_on trained_cap_off
```

## Part 3: large-sample rerun (n=30/condition) -- the n=8 result does not replicate

The verdict above was explicit that n=8 on a metric this sparse (a single
game, `r11l`, driving essentially the whole result) wasn't conclusive on
its own, and recommended a 25-30-repeat follow-up before treating the
cap as a confirmed win. This section is that follow-up.

**Protocol:** identical to Part 2's held-out-games backtest -- same
checkpoint (`checkpoints_holdout_baseline/`: `encoder_moe.pt`,
`moe_predictor.pt`, `game_vocab_moe.json`, `value_head.pt`, confirmed
byte-identical via SHA-256 to the exact checkpoint files present in the
worktree that produced Part 2's numbers), same 5 held-out games
(`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`), same `MAX_ACTIONS=300`
(`Hypothesis`'s unmodified default), same `scripts/run_scorecard.py`
tooling, same `HYPOTHESIS_NOVELTY_BETA_CAP=0.15` (cap ON) vs. `=1.0`
(cap OFF, mathematically a no-op) env-var toggle. Only change: n=30 per
condition instead of n=8 (`scripts/run_novelty_backtest_largescale.py`,
a straightforward sequential loop over `run_scorecard.py` -- the
trained-games sanity check from Part 2 was not rerun, since that half
already showed the cap is structurally inert on in-vocab games by direct
inspection of `game_vocab_moe.json`, which a larger sample can't change).
60 total repeats (2 conditions x 30 repeats x 5 games each = 300
individual game-runs), ~34 minutes wall-clock.

### Results (n=30 each)

| condition | mean score | std | median score | mean levels | total levels | distinct games solved |
|---|---|---|---|---|---|---|
| cap ON (0.15) | 0.0294 | 0.0678 | 0.0 | 0.433 | **13** | 1 (`r11l`) |
| cap OFF (1.0) | 0.0937 | 0.2486 | 0.0 | 0.433 | **13** | 1 (`r11l`) |

**Levels completed is now exactly tied: 13 vs. 13, mean 0.433 vs. 0.433,
identical solve rate (13/30 = 43.3%) on the identical single game
(`r11l`) for both conditions.** Neither condition solved `bp35`, `m0r0`,
`tr87`, or `ka59` in any of the 60 runs (30 per condition) -- consistent
with Part 2 and with the test-time-adaptation agent backtest's own
finding that these four games are essentially never solved by this agent
at this budget regardless of checkpoint or mechanism.

**Mean score now favors cap OFF (0.094 vs. 0.029), the opposite direction
from the n=8 result (which favored cap ON, 0.060 vs. 0.011).** This
reversal is fully explained by the same "one outlier run drives the mean"
pattern the Part 2 verdict already flagged as a risk: cap OFF has two
runs that hit the game's max per-level score band (`0.9524` each,
`r18`/`r30`) plus a third at `0.5122` (`r27`), while cap ON's best run
tops out at `0.2742` (`r15`). With levels-completed exactly tied, these
score-magnitude outliers are the entire source of the remaining gap, not
a difference in how often either condition wins.

**Distribution comparison, not just the two means:**
- Solve-rate: 13/30 vs. 13/30 -- identical, no interval needed.
- Per-solved-run `r11l` score, Mann-Whitney U (two-sided): U=439.0,
  **p=0.86** -- no detectable difference in the score distribution
  conditional on solving.
- Per-solved-run action-count-to-solve (level-1 `level_actions`),
  Mann-Whitney U: U=95.5, **p=0.59** -- no detectable difference in
  solve efficiency either (cap ON's solved-run mean was actually slightly
  *higher*, 162 vs. 144 actions, i.e. marginally slower, though not
  significantly so).

Both checks point the same way: nothing about *how* either condition
solves `r11l` differs in a way distinguishable from chance at this
sample size. The pooled mean-score gap is not a hidden second effect
riding under a tied levels-completed count -- it's sampling variance in
which specific runs happened to solve the game quickly (and therefore
score high) versus slowly.

### Verdict: the n=8 result does not replicate. Call it noise.

**At n=8, cap ON led on every metric (mean score, mean levels, total
levels). At n=30, levels-completed is exactly tied and mean score
reverses to favor cap OFF, with neither distribution passing even a loose
significance check.** This is precisely the failure mode the Part 2
write-up warned was possible ("one extra level completion out of 8
attempts is exactly the kind of margin that... needs a larger sample...
before being called conclusive") and precisely the outcome: more data
revealed the original result was closer to noise than the n=8 numbers
suggested, not a confirmation of it.

Worth being explicit about what this does and doesn't mean:
- It does **not** mean the cap is harmful -- levels-completed is exactly
  tied, and the mechanistic argument behind it (InfoGain doesn't collapse
  on held-out games, the gated V/prediction path does -- see Motivation
  above) is still true and still unaffected by this result.
- It does mean there is **no evidence at this sample size that the cap
  changes agent-level outcomes on these 5 held-out games at all**, in
  either direction. The representation-level asymmetry it was built on
  is real; whether biasing beta toward InfoGain converts that into a
  detectable behavioral difference within a 300-action budget remains
  unconfirmed.
- This is now the **third** time this exact pattern has shown up in this
  project's Stage 6 work: a real, well-motivated component-level fix
  (teacher-policy value head; test-time adaptation; now the novelty-aware
  beta cap) fails to produce a statistically distinguishable agent-level
  signal at the sample sizes this project can practically run. The
  common thread across all three is the same: 5 held-out games, a
  300-action budget, and a metric (`levels_completed`) that only ever
  fires on one of the five games (`r11l`) regardless of intervention --
  that specific combination appears to be the actual bottleneck on
  statistical power here, not any one component's design.

**Practical recommendation:** keep `NOVELTY_BETA_CAP=0.15` enabled by
default -- it remains a mechanistically well-motivated, zero-regression-
risk change on trained games (Part 2's sanity check already established
that; nothing here touches it), and there is still no evidence it hurts
held-out-game performance. But retract the "most encouraging held-out-
games result of this whole Stage 6 investigation" framing from Part 2's
verdict -- at real statistical power, it is not distinguishable from a
no-op on the one metric (agent-level `levels_completed`/score on these 5
games) that would actually confirm it helps. If this is revisited again,
a higher-resolution metric that doesn't require an actual game win to
register signal (e.g. directly tracking `Q`/InfoGain component values
across matched action sequences, as `scripts/
diagnose_hypothesis_beta_holdout.py` already does for beta itself) is
more likely to resolve this than a further increase in n on the binary
win/loss metric -- 30 repeats already required ~34 minutes for a fully
tied levels-completed outcome; getting a binomial comparison to resolve
"13/30 vs. 13/30, is that really tied" needs an outcome to differ at all
first, which more of the same repeats on the same games may simply not
produce if the true effect size on this metric is at or near zero.

### Reproducing this rerun

```
# Same checkpoints_holdout_baseline/* swap into checkpoints/ as Part 2.
python scripts/run_novelty_backtest_largescale.py --n 30
python scripts/summarize_scorecards.py ls_holdout_cap_on_ ls_holdout_cap_off_
python scripts/summarize_per_game.py ls_holdout_cap_on_ ls_holdout_cap_off_
```
