# Stage 6 experiment: systematic search-based data harvesting ("Solver")

**Status: the `Solver` agent and harvest are complete and validated; the
MoE retrain comparison is inconclusive due to two documented, external
infrastructure failures (disk exhaustion, twice) that prevented a
full-length (60-epoch) training run within this session. See "Final
verdict" near the bottom for the honest bottom line -- short version: the
search-harvesting idea works and is a real improvement over the prior
self-play attempt; whether it actually produces a better world model is
unresolved and needs a re-run with real disk headroom, not disproven.**

## The problem this addresses

`experiments/stage6_selfplay_bootstrap.md` (branch `stage6-selfplay-bootstrap`)
tried harvesting a bigger, harder training corpus with `Hypothesis` (the
current best *policy*) running a long rollout, then retraining the MoE
world model on random-policy + self-play data combined. That was a real,
verified **regression** (+22.5% changed-patches on the production
checkpoint vs only +5.7% on the self-play-augmented one), traced to an
echo-chamber effect: `Hypothesis`'s own learned Q-ranking narrowed action-
space and click-space coverage on several games (six games collapsed to
exactly one distinct click location under self-play, vs 58-80 under
random) even while it unlocked a few others. A policy that is still
learning reflects its own gaps back into the data it produces.

## The idea this experiment tests

Replace "a learned policy harvests data" with "systematic, policy-free
search harvests data." A new `Solver` agent
(`ARC-AGI-3-Agents/agents/templates/solver_agent.py`) exhausts each
discovered state's action space deterministically (at the same 8x8-patch
click granularity `Hypothesis`/`Curiosity` already use, 64 options rather
than 4,096 raw pixels) before moving on -- there is no learned preference
that can cause it to systematically skip an untried option the way a
policy's Q-ranking can. The only place a learned ranking signal is used
at all is choosing *which already-discovered state to push toward next*
once the current one is exhausted (reusing `Hypothesis`'s own entropy-
gated `Q = (1-beta)*InfoGain + beta*V` blend) -- never *what* to try once
committed to a state. See the module docstring in `solver_agent.py` for
the full design rationale.

**Honesty about scope**: this is systematic, budget-bounded search, not
literal exhaustive/complete search. The true state space is far larger
than any real action budget can cover.

## What was built

- `ARC-AGI-3-Agents/agents/templates/solver_agent.py` -- the `Solver`
  agent. Registered in `ARC-AGI-3-Agents/agents/__init__.py` and
  `scripts/run_stage0.py`'s `--agent` choices, matching this project's
  existing conventions.
- `jepa/memory.py: TransitionGraph` -- reused unchanged (`record`,
  `tried_actions`). No changes needed; it already does exactly the
  per-state (action, xy) -> outcome bookkeeping this agent needs.
- Replay-to-frontier: `Solver` tracks the exact action sequence from the
  post-RESET root to every discovered state (`_path_to_state`). Once a
  state's local action space is exhausted, it issues a *deliberate*
  RESET (rules.md documents RESET as legal at any time, not just on
  GAME_OVER) and replays that exact sequence to reach a chosen frontier
  state before resuming systematic probing there.
- `scripts/harvest_solver.py` -- runs `Solver` across all local games,
  one game at a time (via subprocess, so one misbehaving game can't take
  the whole sweep down), with a per-game action budget and wall-clock
  timeout, and a disk-free-space check before each game.
- `scripts/analyze_corpus_diversity.py` -- corpus-quality/diversity check
  (completion rate, frame-changed rate, distinct actions/clicks per game),
  matching `stage6_selfplay_bootstrap.md`'s own methodology so the two
  approaches are directly comparable.
- Pulled (read-only `git show`, no other changes) from
  `stage6-selfplay-bootstrap`: `train_moe_predictor.py`'s
  `--checkpoint-every`/`--resume-from`/`JEPA_NUM_WORKERS` safety net, and
  `jepa/benchmark.py`'s `--moe` eval flag. From `stage6-score-optimization`:
  `scripts/run_scorecard.py`.

## Budget decision: `SOLVER_MAX_ACTIONS=4000` per game

A smoke test on `vc33` (a game with a small, bounded reachable-state set
under random play) confirmed the replay-to-frontier mechanic fires
correctly and lands on the intended target deterministically: at a
1000-action budget it never triggered (every discovered state still had
untried options left when the budget ran out); at 4000 it triggered 14
times, 13 completed (the 14th was cut off mid-replay by the budget), and
**all 13 completed replays landed exactly on the intended frontier
state** -- direct empirical confirmation that ARC-3's documented
determinism holds under this agent's own replay mechanism, not just an
assumption. On games with much higher branching (`bp35`, `sp80`, `ar25`
-- close to 100% frame-changed rate per this project's own Stage 1
history), the graph instead grows outward through mostly-novel states
without much local exhaustion, so total travels vary a lot by game;
`ar25`'s 4000-action pass triggered 44 travels, all landing correctly.

Chose 4000 (vs. the `Memory`/`Hypothesis` teacher-policy precedent's
2500) because Solver pays real replay overhead a single long rollout
doesn't -- reaching a state at depth d during a travel costs d extra
actions beyond the new-state discovery itself.

## A real bug found during the full harvest, and its fix

The first full 25-game sweep hit a genuine infinite-retry pathology on
`g50t`: `_pick_frontier` picked the same (cheap, depth-1) frontier target
over and over, and every single attempt diverged to the same *different*
state instead of the intended one -- **1560+ consecutive, identical
failed 2-action travel attempts**, burning nearly the entire 4000-action
budget on a target that had already proven unreachable the very first
time. `_pick_frontier` had no memory of past failures, so nothing stopped
it from re-selecting the same doomed target indefinitely. `g50t`'s
subprocess eventually hit the harvest script's 900s per-game timeout
before finishing.

**Fix** (`Solver._unreachable`, a `set[str]` of frontier targets whose
recorded path has failed to reproduce the expected state at least once):
`_pick_frontier` now excludes any target in this set, and a failed
travel adds its target to it immediately. Applied once to
`solver_agent.py`; because `scripts/harvest_solver.py` launches each
game as a fresh subprocess, every game *after* the fix landed on disk
automatically picked it up with no other changes needed -- confirmed
directly: `lp85` (harvested right after the fix) hit a 92% divergence
rate (12/13 travels) but never looped, evicting a *distinct* target each
time instead. `g50t` was re-harvested alone afterward (`python
scripts/harvest_solver.py --games g50t`) and completed cleanly in 186
seconds (vs. hitting the 900s timeout before).

## An unplanned finding: replay divergence is real and highly game-dependent

Because `Solver` is the first agent in this codebase to actually replay a
multi-step action sequence from RESET and *verify* it lands where
expected (`Memory`/`Hypothesis` never do this -- they rely on the exact
graph for lookup, never for replay), this experiment produced the first
direct empirical measurement of how often ARC-3's assumed determinism
("same state + same action -> same next state", `jepa/memory.py`'s own
docstring) actually holds under a real multi-step replay. It's real, but
imperfect, and the imperfection is wildly uneven across games:

| game | travels | landed on target | diverged |
|---|---|---|---|
| `dc22`, `ft09`, `lf52`, `m0r0`, `su15` | 0 | -- | -- (never needed to travel) |
| `ar25`, `ka59`, `ls20`, `re86`, `sb26`, `sk48`, `sp80`, `tn36`, `tr87`, `tu93`, `vc33`, `wa30` | 1-122 | mostly-to-all | 0-7 (low) |
| `bp35` | 122 | 116 | 6 (5%) |
| `cd82` | 2 | 1 | 1 (50%) |
| `cn04` | 16 | 3 | 13 (81%) |
| `lp85` | 13 | 1 | 12 (92%) |
| `sc25` | 47 | 9 | 38 (81%) |
| `g50t` (post-fix rerun) | 119 | 3 | 115 (97%) |

Divergence never crashes anything -- the fix above handles it gracefully
(evict the target, treat the actually-reached state as newly discovered,
keep going) -- but it's a genuine, previously-unmeasured limitation of
this project's exact-hash `TransitionGraph` state identity worth flagging
plainly: **for a meaningful subset of games, the same *visible* frame
does not reliably predict the same future dynamics.** The most likely
explanation is some game-internal state that isn't rendered in the
visible grid (a persistent counter, timer, or seed that isn't reset by
RESET the way the visible board is) -- not investigated further this
session (would need per-game inspection of engine internals this project
doesn't have source access to), but worth remembering: `Memory` and
`Hypothesis`'s own "exact recall" mechanism rests on the same assumption
and is just as exposed to it, only less visibly since neither ever
attempts a multi-step verified replay the way `Solver` does.

## Corpus quality and diversity: `random` vs `solver`

Full 25-game harvest, `SOLVER_MAX_ACTIONS=4000`, `python
scripts/analyze_corpus_diversity.py random solver`:

| corpus | files | transitions | changed rate | total levels completed |
|---|---|---|---|---|
| `random` (existing, 150 files) | 150 | 11,948 | 63.3% | 1 (1 distinct game) |
| `solver` (this experiment, 25 files) | 25 | 98,014 | **78.0%** | 2 (2 distinct games) |

For reference, `stage6_selfplay_bootstrap.md`'s own numbers on the same
kind of comparison: random 63.5% changed rate (consistent with the
63.3% measured freshly here -- a good cross-session sanity check), the
`Hypothesis` self-play harvest 76.2% changed rate on 62,525 transitions.
**`Solver` harvested more data (98k vs 62.5k) at a higher changed rate
(78.0% vs 76.2%) than the policy-driven self-play attempt did**, while
avoiding that attempt's central failure mode -- see below.

**The echo-chamber check, the actual point of this comparison.** The
self-play doc found real, damaging narrowing: six games collapsed to
*exactly one* distinct ACTION6 click location under `Hypothesis`
self-play (vs. 58-80 under random), and several non-click games lost
real action-space coverage (`su15` 7->3, `tu93` 8->5, `ls20` 7->5, etc.)
-- a policy avoiding options its own (possibly wrong) Q-values rated
poorly. **`Solver` cannot exhibit this failure mode by construction**:
it never ranks or skips a legal, available option -- it tries every
untried (action, patch) pair systematically before ever moving on, so
there is no mechanism by which a "belief" about an option's value could
suppress trying it. Directly confirmed in the harvested data: on every
game where ACTION6 was actually available and tried at all, `Solver`
reached **45-64 out of the full 64 possible click patches** (`ar25`,
`bp35`, `cd82`, `cn04`, `ka59` all hit the complete 64/64) -- there is no
single-click-location collapse anywhere in this corpus, the exact
opposite of the self-play doc's six-game collapse.

A different, initially-confusing pattern showed up when comparing
*distinct action ids* per game: `random` shows exactly 7 distinct action
ids on literally every game, while `solver` ranges from 1 to 7 and often
sits below 7. This is **not** an echo-chamber effect -- it's a direct
consequence of a real, useful design difference. `Random.choose_action`
(`ARC-AGI-3-Agents/agents/templates/random_agent.py`) samples uniformly
over *all* 7 non-RESET action ids with **no reference to
`latest_frame.available_actions` at all** -- it happily submits actions
the game itself doesn't currently list as legal, and the environment
apparently just no-ops them rather than rejecting them, which is why
random's own "distinct actions" count is a flat, uninformative 7
everywhere regardless of what's actually offered. `Solver` respects
`available_actions` by construction (its whole option-generation logic
is built from that field), so its distinct-action count is a *faithful*
reading of each game's real, declared action space at the states it
actually visited -- e.g. it independently reproduces CLAUDE.md's Stage 5
finding that `ft09`/`lp85`/`r11l` offer *only* ACTION6, and extends that
same finding to four more games (`s5i5`, `su15`, `tn36`, `vc33`) this
project hadn't previously flagged. Comparing "distinct actions tried"
between the two corpora isn't apples-to-apples for this reason; comparing
click coverage (both corpora *do* respect ACTION6 the same way, since
clicking is legal whenever offered in both) is the fair, and clean, test
-- and `Solver` wins it decisively, as above.

**Read on corpus quality:** unlike the self-play attempt, this comparison
does not need a mixed verdict. More data, a higher changed-rate, and
*no* diversity-collapse anywhere -- proceeding to the retrain gate was
justified by the data itself, not assumed.

## MoE retrain: combined corpus (150 random + 25 solver-harvest files, ~112k transitions)

Same command as production's own build (CLAUDE.md Stage 4 item 6):
`python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60
--num-experts 8 --out checkpoints_search --checkpoint-every 10`.

**Apples-to-apples production baseline, established first, on the exact
combined corpus the new checkpoint would also be evaluated against**
(`jepa.benchmark eval --moe --checkpoint-dir checkpoints`, i.e. the
*existing* production checkpoint, not yet retrained on anything new,
scored against this session's combined 175-file recordings directory):
**+26.4% changed-patches improvement.** This is the number
`checkpoints_search` needs to beat -- not the historical +22.9%
production figure quoted elsewhere in CLAUDE.md, which was measured on a
different (smaller, purely-random) corpus and isn't the right comparator
here (see `jepa/benchmark.py: run_eval_moe`'s own docstring on why the
val split must come from the *same* recordings-directory snapshot for
both checkpoints under comparison).

### A real infrastructure failure mid-run, and how it was handled

The first training attempt crashed partway through the arc-finetune
phase (after the pretrain phase's own 20 epochs completed cleanly) with
a `MemoryError` inside a DataLoader worker's `pickle.load` --
**reproducing CLAUDE.md's own documented full-disk gotcha exactly**,
confirmed directly: `C:` had dropped to **0.19GB free** at the moment of
the crash (from ~21GB free when the harvest started). Per that gotcha's
own precedent, disk pressure on this shared machine can come from a
concurrently-running session in a *different* worktree, not just this
one -- confirmed here too: the *other* (non-worktree) checkout's own
`ARC-AGI-3-Agents/recordings/` held 2.1GB of its own, separate from
anything this session wrote, and this machine had three active
`.claude/worktrees/` directories at the time, any of which could have
been mid-write.

Fixed in two parts, neither requiring the training run to restart from
scratch:
1. **Freed disk on the parts of the problem this session actually
   controls**, without touching the other checkout or other worktrees
   (out of scope to modify per this session's own instructions): moved
   this worktree's `ARC-AGI-3-Agents/recordings/` (3.8GB) to `E:`
   (confirmed real free space earlier in the session) and replaced it
   with an NTFS directory junction at the original path
   (`New-Item -ItemType Junction`), so every existing path reference
   (`jepa/data/trajectories.py`'s hardcoded `ARC-AGI-3-Agents/recordings`
   string, `scripts/harvest_solver.py`, etc.) keeps working unmodified --
   this project's own CLAUDE.md gotcha flagged exactly this junction
   approach as "the durable fix... not yet done" the last time this
   crisis hit; done here. Also killed two lingering worker subprocesses
   left over from the crashed run, which alone recovered several more GB
   (likely memory-mapped/paged temp state tied to those processes) --
   `C:` went from 0.19GB to 15.64GB free after cleanup, without needing
   to identify or touch whatever the other concurrent session was doing.
2. **Resumed rather than restarted**, using the `--resume-from`/
   `--checkpoint-every` safety net ported from `stage6-selfplay-bootstrap`
   for exactly this scenario: `--resume-from checkpoints_search`
   reloaded the completed pretrain-phase weights and game vocab, skipped
   re-running the 20 pretrain epochs, and continued straight into
   arc-finetune. Also set `JEPA_NUM_WORKERS=0` (the other half of that
   same ported safety net) for the resumed run specifically, trading some
   wall-clock speed to avoid the Windows spawn/pickle path that triggered
   the `MemoryError` in the first place, rather than gambling that disk
   headroom stays healthy for the rest of a 60-epoch run on a shared
   machine.

### A second infrastructure setback, and the final scope decision

Disk pressure recurred during the resumed run itself (unrelated to this
session's own data -- confirmed by a live report from the session
coordinator: the actual cause was the host machine's Steam game library
(277GB) and an unemptied Recycle Bin (20GB), both entirely outside this
project and outside what this session could clean up, structurally
capping total headroom at roughly 15GB regardless of how disciplined
this session's own disk hygiene was). `C:` free space was observed
swinging from 15.45GB down to 2.79GB and back up to 15.45GB within about
two minutes during a `JEPA_NUM_WORKERS=2` attempt -- tight enough that
even a moderate worker count risked reproducing the original
`MemoryError`. Per the coordinator's explicit guidance ("if there's
truly no room, pause... rather than losing data silently"), the run was
stopped proactively (not left to crash), the last-good checkpoint was
re-verified intact (`torch.load` round-trip on both `.pt` files,
succeeded), and training was restarted a final time with
`JEPA_NUM_WORKERS=0` for the remainder.

Even at `num_workers=0`, throughput was far below what the reduced
headroom made practical to wait out in this session: the first
post-resume 5-epoch arc-finetune checkpoint took ~48 minutes, and the
next 5 epochs did not complete within a further ~50 minutes of
monitoring (confirmed still actively computing throughout, via steadily
increasing process CPU time -- not hung). Given the full 60-epoch
schedule was no longer feasible within this session at any safe worker
count, training was stopped at **5 completed arc-finetune epochs** (on
top of the full, uninterrupted 20-epoch MiniGrid pretrain phase) and
that checkpoint (`arc-finetune-resumed-epoch5-inprogress`) was evaluated
as-is. **This is a real, honest limitation of this write-up's retrain
result, not a hidden one** -- production's own checkpoint had the full
60 arc-finetune epochs; this one has 5. Flagging this plainly before the
numbers below, not after.

## MoE retrain result: inconclusive, not a validated improvement

`jepa.benchmark eval --moe --checkpoint-dir checkpoints_search`, same
combined-corpus held-out split as the production baseline above:

| checkpoint | arc-finetune epochs | pooled changed-patches improvement |
|---|---|---|
| `checkpoints/` (production) | 60 | **+26.4%** |
| `checkpoints_search/` (this experiment) | 5 | **+46.9%** (pooled) |

**Taken at face value, the pooled number says the search-harvest
checkpoint wins even at 1/12th the training. It doesn't hold up, and the
per-game breakdown shows exactly why -- this is the same "a misleading
pooled metric can hide the real story" lesson this project has hit
before, just in the opposite direction from usual:**

- **24 of 25 games individually got *worse* under the search checkpoint**
  relative to its own identity baseline, several severely so (`s5i5`
  -443.7%, `tr87` -305.8%, `dc22` -107.4%, `tu93` -87.8%). Only `cn04`
  improved, and dramatically (+65.4%, pred=0.00631 vs identity=0.01823)
  -- `cn04` also happens to have by far the largest absolute identity-MSE
  of any game in this corpus, so it dominates the *pooled* average
  (weighted by absolute error magnitude, not by game count) enough to
  flip the overall sign despite being a 1-game exception, not the rule.
- **Directly measured encoder feature variance** (same diagnostic
  methodology `stage6_selfplay_bootstrap.md` used to check for
  collapse): production's per-channel feature std averages **1.75**
  (range 0.89-2.68 across channels); the search checkpoint's averages
  **1.07** (range 0.88-1.45) -- clearly compressed relative to
  production, though not below the `VARIANCE_FLOOR=1.0` threshold this
  project treats as the collapse line (so this is *not* classic full
  collapse, unlike some of what `stage6_selfplay_bootstrap.md` measured
  on its own regressed checkpoint) -- but it's a real, measurable
  reduction in representational dynamic range, consistent with a model
  that hasn't had enough fine-tuning steps yet to develop production's
  full feature scale on ARC-3-specific dynamics, having spent almost all
  of its training budget on MiniGrid instead (20 pretrain epochs vs. only
  5 ARC epochs, a ratio production never ran at -- production's own 20:60
  ratio is inverted here to roughly 20:5).

**Honest read: this is not a validated improvement, and this
experiment's retrain step should be read as inconclusive due to
insufficient training, not as a negative result about the search-
harvested data itself.** The corpus-quality analysis earlier in this
document is real, verified evidence that the *data* is good (more
volume, higher changed-rate, no diversity collapse, unlike the self-play
attempt) -- what's missing is a *fair-length* training run to find out
what a properly-converged model built on it actually looks like. Per
this project's own established practice (e.g. Stage 4's Sokoban ablation,
the self-play-bootstrap's own "if the predictor eval doesn't improve,
stop there" gate), the honest thing to do with an unreliable positive
signal is not to chase it further within this already-overrun session,
and not to launder it into a false "beats production" headline by
quoting the pooled number alone.

## What was skipped, and why

Per this project's own standing gate ("only if [the predictor eval]
shows a real improvement, proceed to retrain the value head and run a
local scorecard backtest"): value-head retraining and the scorecard
backtest were **not run**. The pooled metric nominally cleared the bar,
but the per-game and feature-variance evidence above shows that number
isn't trustworthy enough to build further conclusions on -- retraining a
value head against an undertrained, per-game-mostly-regressed encoder
and then spending real scorecard-backtest compute on it would be
spending effort to (most likely) confirm a foregone "no reliable signal"
conclusion, not testing anything new. `checkpoints/moe_predictor.pt`
(production) is untouched throughout; `checkpoints_search/` (gitignored,
not committed) holds the inconclusive 5-epoch checkpoint for anyone who
wants to inspect it directly or, more usefully, resume training on a
machine with real disk headroom.

## Final verdict

**The search-based harvesting idea itself is validated; the retrain
comparison is not, for a documented, external infrastructure reason --
these are two separate claims, and only the first one has a confident
answer from this session.**

1. **`Solver` (systematic, policy-free search) works as designed and
   avoids the self-play attempt's central failure mode.** Direct,
   verified evidence: full 8x8-patch click coverage (up to 64/64) on
   every game where ACTION6 was exercised, zero single-location
   collapses (vs. six in the self-play corpus), a higher changed-frame
   rate (78.0%) on 57% more data (98k vs. 62.5k transitions) than the
   policy-driven alternative, and a real, previously-undiscovered
   correctness bug (the frontier-target infinite-retry loop) found and
   fixed mid-session with direct before/after evidence. The replay-to-
   frontier mechanic's core claim -- deterministic multi-step replay
   correctly reaches its intended target -- was also directly falsified-
   and-measured rather than assumed: it holds most of the time but not
   universally (a genuinely new, reportable finding about this project's
   `TransitionGraph` state-identity assumption, see above).
2. **Whether a model trained on this better data actually predicts
   better than production remains unanswered**, not because the data
   failed a test, but because two independent, well-documented
   infrastructure failures (a disk-full crash mid-run, then renewed
   external disk pressure from processes entirely outside this project)
   made completing a fair, full-length (60-epoch) retrain infeasible
   within this session. The 5-epoch checkpoint that *was* produced shows
   a misleading-if-quoted-alone pooled improvement that doesn't survive
   a per-game or feature-variance check.

**Recommendation for a future session:** re-run
`python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60
--num-experts 8 --out checkpoints_search --checkpoint-every 10
--resume-from checkpoints_search` (the corpus and all code are already in
place and unchanged; this simply needs to finish the remaining ~55
arc-finetune epochs) on a machine/session with real, stable disk
headroom, then re-run the exact `jepa.benchmark eval --moe` command
above for a genuine apples-to-apples number before deciding whether to
proceed to the value-head/scorecard-backtest gate. Do **not** trust a
pooled changed-patches number on this corpus without also checking the
per-game breakdown -- `cn04`'s outsized absolute-error weight in the
pooled average is a standing distortion risk for any future eval on this
specific combined corpus, not just this session's.

## Reproducing this experiment

```
# 1. Harvest (from repo root; ~4000 actions/game, ~10-15 min/game typical,
#    some games much larger -- see the per-game table above)
python scripts/harvest_solver.py --max-actions 4000 --timeout 900

# 2. Corpus quality check
python scripts/analyze_corpus_diversity.py random solver

# 3. Retrain (same hyperparameters as production; resumable if interrupted,
#    JEPA_NUM_WORKERS=0 to avoid the disk-pressure MemoryError this session hit)
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 \
    --num-experts 8 --out checkpoints_search --checkpoint-every 10
# if killed partway:
JEPA_NUM_WORKERS=0 python -m jepa.train_moe_predictor --epochs <remaining> \
    --num-experts 8 --out checkpoints_search --checkpoint-every 10 \
    --resume-from checkpoints_search

# 4. Apples-to-apples eval (same held-out split, both checkpoints, run
#    back-to-back without changing ARC-AGI-3-Agents/recordings/ in between)
python -m jepa.benchmark eval --moe --checkpoint-dir checkpoints --tag prod
python -m jepa.benchmark eval --moe --checkpoint-dir checkpoints_search --tag search
```
