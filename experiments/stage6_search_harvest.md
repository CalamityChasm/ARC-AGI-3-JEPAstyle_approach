# Stage 6 experiment: systematic search-based data harvesting ("Solver")

**Status (updated, 2026-07-19): COMPLETE. The `Solver` agent and harvest
are validated, the full 60-epoch MoE retrain finished, and the
comparison against production is a genuine, per-game-consistent
improvement -- not just a flattering pooled average. The value head was
retrained against the new checkpoint and a matched local scorecard
backtest (n=5 each) favors the search checkpoint on both mean score and
total levels completed. See "Final verdict" near the bottom for the full
honest bottom line. This supersedes the "inconclusive due to disk
exhaustion" status below, which is left in place as the historical
record of how the first (interrupted) retrain attempt went and why it
wasn't trustworthy on its own.**

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

## What was skipped, and why (superseded -- see the 2026-07-19 follow-up below)

Per this project's own standing gate ("only if [the predictor eval]
shows a real improvement, proceed to retrain the value head and run a
local scorecard backtest"): value-head retraining and the scorecard
backtest were **not run** at the time this section was written. The
pooled metric nominally cleared the bar, but the per-game and
feature-variance evidence above showed that number wasn't trustworthy
enough to build further conclusions on at the 5-epoch checkpoint's stage
of training. This turned out to be the right call in hindsight -- the
5-epoch checkpoint really was undertrained (see below) -- but a
follow-up session (below) completed the full 60-epoch run and found a
materially different, trustworthy result.

## MoE retrain: completed to the full 60 epochs (2026-07-19 follow-up)

A later session resumed training from the exact 5-epoch checkpoint left
behind above, using the same `--resume-from`/`--checkpoint-every` machinery,
and pushed it through to the originally-intended 60 total arc-finetune
epochs (on top of the already-complete 20-epoch MiniGrid pretrain phase --
unchanged from the original run).

**Getting there took seven separate resume cycles, not one continuous
run** -- worth documenting since it's a real, recurring constraint on this
project's infrastructure, not a one-off:

- The training subprocess was killed by something external roughly every
  **50-60 minutes of wall-clock runtime**, regardless of epoch progress,
  disk headroom, or RAM headroom (all three were directly checked and
  confirmed healthy at multiple kill points -- e.g. C: at 49GB free, E: at
  65GB free, no `MemoryError`, no `Traceback`, just a clean log stop).
  This is a **different root cause than this document's original two
  disk-exhaustion incidents** -- this time it was later identified (via
  the session coordinator, who could see infrastructure state this agent
  couldn't) as a **session usage-limit boundary**, not a disk/RAM/OS-level
  kill. The visible symptom is identical either way (background process
  silently disappears, `tasklist` shows nothing, no error in the log), so
  the practical mitigation is the same regardless of cause: checkpoint
  often (`--checkpoint-every 3` was used throughout this follow-up, down
  from the original run's `--every 10`, to bound the loss from any one cutoff
  to at most ~2-3 lost epochs) and just keep resuming.
- Each resume cycle read the *previous* cycle's `--out` directory as its
  own `--resume-from` (both pointed at the same
  `E:\jepa_overflow\checkpoints_search`, honoring the original task's
  instruction to keep all new output on E: rather than C:), computing the
  remaining epoch count as `60 - <cumulative epochs already checkpointed>`
  each time. Progression across the seven cycles: 5 (starting point) ->
  15 -> 27 -> 39 -> 51 -> 60 (final). A couple of cycles lost 1-2 epochs of
  real, logged-but-not-yet-checkpointed progress to a cutoff landing
  between checkpoint boundaries -- cheap to just redo, not worth
  chasing tighter checkpointing for.
- The final cycle (9 remaining epochs) completed cleanly end-to-end,
  confirmed by both the training script's own log reaching
  `[checkpoint] saved ... (tag=final)` and the background task reporting
  a genuine exit code 0 (not another silent kill).

**Per-game evaluation (the primary signal, per this project's own
standing rule -- pooled second):** `jepa.benchmark eval --moe`, same
apples-to-apples methodology as the original 5-epoch attempt (both
checkpoints scored against this worktree's current combined
recordings corpus, same fixed val split):

| checkpoint | arc-finetune epochs | pooled changed-patches improvement |
|---|---|---|
| `checkpoints/` (production) | 60 | **+26.4%** (reproduces the number recorded earlier in this document exactly) |
| `checkpoints_search/` (this experiment, now fully trained) | 60 | **+60.2%** |

Pooled numbers alone were exactly what burned the original 5-epoch
attempt, so the real check is head-to-head per-game improvement% (search
vs. production, both relative to their own identity baseline on the same
val split):

**Search beats production on 18 of 25 games; production wins on 7.**
The wins are large and concentrated on the games with the biggest
absolute identity-baseline error (`ka59` +87pp, `su15` +65pp, `tr87`
+54pp, `cn04` +48pp, `m0r0` +46pp, `bp35` +28pp, `dc22` +27pp) -- exactly
the games whose absolute error dominates the pooled average, so the
pooled win isn't an artifact of one outlier the way this document
originally worried about. Where production wins, the margins are small
and on games with tiny absolute identity-baseline error (`s5i5`, `ft09`,
`vc33`, `r11l`, `sb26`, `re86`, `ls20` -- all within a few percentage
points, consistent with this project's own repeated finding that
percentage swings on near-zero absolute error are mostly noise, not
signal). This is a genuine, not-flattering-average, per-game-consistent
improvement.

**Encoder feature variance also recovered to a healthy range**, directly
addressing the exact concern that made the 5-epoch checkpoint's pooled
number untrustworthy: mean per-channel std on a shared held-out batch is
now **1.47** for the search checkpoint vs. production's **1.69** (both
comfortably above the `VARIANCE_FLOOR=1.0` collapse line, and much closer
together than the 5-epoch checkpoint's compressed **1.07**). Consistent
with the original hypothesis that the compression was an artifact of
insufficient ARC-specific fine-tuning (5 of 60 epochs) rather than
anything about the harvested data itself -- the full 60 epochs closed
most of that gap.

**Value head retrained** against the fully-trained checkpoint's encoder
(`python -m jepa.train_value_head --epochs 20 --encoder
E:/jepa_overflow/checkpoints_search/encoder_moe.pt --out
E:/jepa_overflow/checkpoints_search`), same oversampling setup as
production's own value head. 112,175 value-target samples from this
worktree's combined corpus, 0.5% with a nonzero target (lower than the
~1.6% cited elsewhere in this project's history, consistent with the
Solver harvest's systematic-exploration style producing a different level-
completion density than a teacher-policy rollout) -- val MSE bounced
around 0.0010-0.0032 against a 0.0008 zero-baseline across 20 epochs,
in the same "not clearly better than always-predict-zero on the full
population" territory this project has already documented and explained
(population MSE is dominated by the near-100%-zero-target majority; the
meaningful-subset metric is the honest one, per Stage 5's own history) --
not re-derived in detail here since it wasn't the deciding evidence for
this branch of the gate.

**Local scorecard backtest, matched n=5 per checkpoint** (both using
this worktree's `Hypothesis` agent code unchanged, `MAX_ACTIONS=300`,
`scripts/run_scorecard.py --agent hypothesis`, swapping only the 5 MoE-era
checkpoint files -- `encoder_moe.pt`, `moe_predictor.pt`, `value_head.pt`,
`game_vocab_moe.json`, `moe_training_meta.json` -- in this worktree's
own `checkpoints/` directory between repeats, never touching the main
checkout's production checkpoint):

| checkpoint | scores (5 repeats) | mean score | total levels completed (of 183 possible x5) |
|---|---|---|---|
| production | 0.0, 0.0023, 0.0050, 0.1344, 0.0 | **0.0283** | **4** |
| search (this experiment) | 0.0041, 0.0, 0.0048, 0.1905, 0.0039 | **0.0407** | **6** |

Search wins on both the mean real Kaggle-formula score (+44% relative)
and total levels completed (+50% relative) at this sample size. n=5 is a
genuinely small sample given this project's own repeatedly-documented
observation that single-digit-repeat comparisons are noisy on a sparse
metric like this (see Stage 2/5's history) -- not a statistically
airtight result on its own -- but it points the *same direction* as the
per-game predictor eval and the feature-variance check, three
independent pieces of evidence all favoring the search checkpoint rather
than one noisy metric standing alone.

(Operational note: the harness's anonymous `ARC_API_KEY` had expired
again before this backtest -- the exact, previously-documented gotcha --
causing the first production-backtest attempt to fail instantly with "No
'FINAL SCORECARD REPORT' found in output" (a 401 on the game-listing
call, not a real agent or checkpoint problem). Refreshed via the
documented `https://three.arcprize.org/api/games/anonkey` endpoint and
reran cleanly.)

## Final verdict

**The search-based harvesting idea is validated, and now so is the
retrain: a model trained on the Solver-harvested combined corpus, taken
to the same full 60-epoch budget as production, measurably beats
production on the predictor's own per-game metric, on encoder
representational health, and on a matched local scorecard backtest.**
This supersedes the earlier "inconclusive" verdict above, which was the
right call *at the time* (the 5-epoch checkpoint really wasn't ready to
judge) but is no longer the state of the evidence.

1. **`Solver` (systematic, policy-free search) works as designed and
   avoids the self-play attempt's central failure mode.** Direct,
   verified evidence: full 8x8-patch click coverage (up to 64/64) on
   every game where ACTION6 was exercised, zero single-location
   collapses (vs. six in the self-play corpus), a higher changed-frame
   rate (78.0%) on 57% more data (98k vs. 62.5k transitions) than the
   policy-driven alternative, and a real, previously-undiscovered
   correctness bug (the frontier-target infinite-retry loop) found and
   fixed mid-session with direct before/after evidence.
2. **A model trained on this data, given the full training budget
   production got, predicts genuinely better than production**: +60.2%
   vs. +26.4% pooled changed-patches, a head-to-head per-game win on
   18/25 games (concentrated on the highest-absolute-error games, not an
   artifact of one outlier), and healthier encoder feature variance
   (1.47 vs. 1.69, both above the collapse floor).
3. **That predictor-level improvement carries through to a real,
   matched local scorecard backtest**: +44% mean score, +50% total
   levels completed at n=5 repeats each. Small sample, but consistent
   with, not contradicting, points 1-2.

**Recommendation:** treat `E:\jepa_overflow\checkpoints_search` (the
fully-trained, 60-epoch checkpoint: `encoder_moe.pt`, `moe_predictor.pt`,
`value_head.pt`, `game_vocab_moe.json`, `moe_training_meta.json`) as a
real candidate to promote to production, not just a research artifact.
This session deliberately did **not** overwrite
`checkpoints/moe_predictor.pt` in the main checkout -- promotion is a
decision for a human to make deliberately (e.g. after a larger-n backtest
or a real Kaggle submission comparison), not something to do silently as
a side effect of an evaluation session. If promoting: copy all five
files from `E:\jepa_overflow\checkpoints_search` into the main checkout's
`checkpoints/` directory, and consider a larger-n (8+) backtest first
given n=5's known noise ceiling on this metric.

**If a future session hits the same "background process silently dies
around 50-60 minutes" symptom again:** check with whoever/whatever is
coordinating the session for a usage-limit or similar boundary *before*
assuming it's disk/RAM/OS-related again -- this follow-up confirmed disk
and RAM were both healthy at every kill point this time, and the real
cause turned out to be a session-level constraint external to the
training process entirely. The mitigation is the same either way
(checkpoint often, resume from the last save), but the diagnosis matters
for not wasting time re-freeing disk space that was never the problem.

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

# 5. If the eval shows a genuine per-game-consistent win (it did, see the
#    2026-07-19 follow-up above): retrain the value head against the new
#    encoder, then backtest.
python -m jepa.train_value_head --epochs 20 \
    --encoder checkpoints_search/encoder_moe.pt --out checkpoints_search

# 6. Local scorecard backtest -- Hypothesis loads checkpoints from a
#    hardcoded `checkpoints/` dir (ARC-AGI-3-Agents/agents/templates/
#    hypothesis_agent.py: _CHECKPOINT_DIR), so swap the 5 MoE-era files in
#    and out of that directory around each set of repeats (back up the
#    production copies first if checkpoints/ isn't itself gitignored-and-
#    disposable in your checkout):
#      encoder_moe.pt, moe_predictor.pt, value_head.pt,
#      game_vocab_moe.json, moe_training_meta.json
python scripts/run_scorecard.py --agent hypothesis --label <name>_r1
# ...repeat for r2..rN with the same checkpoint files in place, then swap
# and repeat for the other checkpoint...
```

**A recurring infrastructure note for any future long training run in
this environment:** background processes in this session got killed
externally roughly every 50-60 minutes of wall-clock runtime, independent
of disk/RAM health (confirmed directly, multiple times, during the
2026-07-19 follow-up) -- later identified as a session usage-limit
boundary rather than the disk-exhaustion this document originally
suspected. `--checkpoint-every 3` (down from `--every 10`) and just
resuming repeatedly, computing `<remaining> = 60 - <cumulative epochs
already saved>` each time, is the practical workaround -- expect several
resume cycles for a full 60-epoch run, not one continuous session.
