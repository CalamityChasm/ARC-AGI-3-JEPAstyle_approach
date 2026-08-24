# Stage 6 experiment: harvesting real WIN-episode transitions

**Status: in progress -- draft, being filled in as the harvest campaign runs.**

## The problem this addresses

The rest of this project's Stage 6 work has been about the *world model's*
generalization gap (see CLAUDE.md's Stage 6 addendum: 13+ interventions,
mostly negative, against zero-shot performance on held-out games). This
task is a narrower, more mechanical one: the current production `Hypothesis`
agent already wins some rounds on a handful of the 25 local trained games
(small numbers, but real -- `r11l`, `sp80`, `cn04`, `cd82`, `m0r0` and
others have all seen real completions across this project's many backtests).
That means there is real, mineable signal in the agent's own successful
play: harvest additional real transitions specifically from episodes that
reach a WIN, to build a small, honestly-quantified, success-weighted
corpus for the "most relevant" end of a future training curriculum. This
task produces the data only -- a separate piece of work handles the
curriculum/training side.

## Prior art in this project (read before designing this)

- **CLAUDE.md's "Stage 5 follow-up -- a 'teacher policy' for denser
  value-head data"**: `Memory` (an older, simpler agent) was run with
  `MAX_ACTIONS` temporarily bumped 300 -> 2500 as a "teacher," on the
  reasoning that a much longer per-game budget gives many more
  RESET-to-GAME_OVER attempts, raising the odds of at least one real
  `levels_completed` event per game within a single subprocess call. That
  precedent (bump a class-level `MAX_ACTIONS`, harvest, revert the code
  change afterward) is the direct model for this task's own budget choice
  below.
- **`experiments/stage6_search_harvest.md`** (branch `stage6-search-harvest`,
  not merged): a systematic, policy-free `Solver` agent harvested a much
  larger, higher-changed-rate corpus (98k transitions, 78.0% changed rate)
  than a policy-driven self-play attempt, and retraining the MoE predictor
  on it (full 60-epoch budget) beat production on 18/25 games. That
  experiment optimized for *dynamics diversity/coverage*, deliberately
  avoiding any learned-policy bias in *what gets tried*. This task is a
  different, complementary axis: not "what does the world look like
  everywhere," but "what does a successful play-through of a game this
  agent can already win actually look like" -- i.e. success-weighted, not
  coverage-weighted, and deliberately using the current best *policy*
  (`Hypothesis`) rather than systematic search, since the whole point is
  capturing what a good real playthrough looks like from this project's
  own best available agent.

## Design decisions

**Agent: `Hypothesis`, unmodified scoring logic.** This branch's current
lineage already has the argmax->softmax click-sampling fix (Stage 5
follow-up 3) in place; the novelty-aware beta cap
(`stage6-novelty-aware-beta`) was checked and is *not* present in this
worktree's `hypothesis_agent.py` (only documented in CLAUDE.md, on an
unmerged branch) -- not added here either, since it only changes behavior
on games outside the training vocabulary, which doesn't apply to a
harvest confined to the 25 known local games.

**Budget: `MAX_ACTIONS` temporarily bumped 300 -> 2000.** Calibrated
directly rather than guessed: a default-budget (300-action) run on `r11l`
took 28s wall-clock; a first 2000-action run on `ar25` took 63s. At this
pace, sweeping all 25 games once at 2000 actions is on the order of
25-30 minutes total wall-clock, comfortably harvestable in a handful of
short legs (see "Harvest campaign" below) without needing anywhere near
`Memory`'s 2500 or `Solver`'s 4000-action precedents. 2000 was chosen
over an even larger budget specifically to keep each leg short given this
environment's own documented ~30-45 minute background-process ceiling,
while still giving each game roughly 6-10x the effective number of
RESET-cycle attempts a normal 300-action evaluation run gets (episode
lengths vary by game, so this is an estimate, not an exact multiplier).
Reverted to 300 in `ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`
once the harvest finished (see git history for the diff).

**Storage: routed entirely to `E:`, never `C:`.** `ARC-AGI-3-Agents/.env`'s
`RECORDINGS_DIR` was pointed directly at
`E:/jepa_overflow/winning_harvest/recordings` for the harvest run itself
(not a post-hoc move -- the framework's `Recorder` reads `RECORDINGS_DIR`
straight from the environment, see `ARC-AGI-3-Agents/agents/recorder.py:
get_recordings_dir`), so no raw harvest data ever touched `C:` at all.
`C:` had only ~17GB free at the start of this task; `E:` had ~38GB free.
Disk-free-space was checked before every game via
`scripts/harvest_wins.py`'s own `check_disk()` (aborts below a 5GB
threshold on either drive).

**Harvest campaign structure: many short per-game legs, not one long
blocking sweep.** `scripts/harvest_wins.py` runs one game at a time as a
fresh subprocess (mirrors `stage6-search-harvest`'s own
`scripts/harvest_solver.py` pattern), with a per-game wall-clock timeout
and a disk check before each game -- so one slow/misbehaving game can't
take the whole campaign down, and each leg (a handful of games) stays
well under this environment's own documented background-process ceiling.

## Extraction: filtering for WIN specifically, not "completed episode"

`scripts/extract_winning_transitions.py` is the real filtering step (not
"record everything and hope"). Since `Hypothesis.MAX_ACTIONS` was bumped
well above normal, each raw recording file from this harvest is really a
concatenation of *many* separate episode attempts back to back (segments
separated by RESET actions) -- most of which do not reach WIN. For each
raw file:

1. Scan for a frame with `state == "WIN"` (`arcengine.GameState.WIN`,
   confirmed directly from the installed `arcengine` package: the enum is
   `NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER` -- WIN means the whole
   game, all levels, is complete, not just one level-up).
   `Hypothesis.is_done` returns `True` exactly on `state is GameState.WIN`,
   and the framework's own `Agent.main()` loop
   (`ARC-AGI-3-Agents/agents/agent.py`) stops as soon as `is_done()` is
   true -- so a WIN frame, if present at all, is necessarily the *last*
   line in the file, and at most one WIN can occur per raw file.
2. If no WIN frame exists, the file is skipped entirely -- no
   near-misses, no "completed episode" padding, only real wins.
3. If a WIN frame exists, walk backward from it to the most recent
   RESET action (`action_input.id == 0`) to find where *that specific*
   winning attempt actually started, and extract only that segment
   (frames from the reset through the WIN frame, inclusive) -- not the
   many earlier failed attempts sitting in the same file, and not
   anything after the WIN frame (there is nothing after it; the loop
   already stopped).
4. Each winning segment is written out as its own
   `<original-name>.win.recording.jsonl` file, in the exact same
   `{"timestamp": ..., "data": {...FrameData fields...}}` line format the
   framework's own `Recorder` writes -- byte-for-byte-compatible with an
   ordinary recording file, not a bespoke format.

## Loading the harvested corpus

`jepa/data/trajectories.py` was extended (additively -- default behavior
for the main pipeline is unchanged) with
`load_transitions_from_dir(recordings_dir: Path)`, factored out of the
existing `load_all_transitions(repo_root)` (which now just calls it with
the fixed `<repo_root>/ARC-AGI-3-Agents/recordings` path). The curated
win corpus can be loaded directly with the same parsing logic the rest of
the pipeline already uses, zero re-derivation:

```python
from pathlib import Path
from jepa.data.trajectories import load_transitions_from_dir

win_transitions = load_transitions_from_dir(
    Path("E:/jepa_overflow/winning_harvest/winning_corpus")
)
# same (frame_t, action_id, x, y, frame_t+1, changed, game_id) tuple format
# as jepa.data.trajectories.load_all_transitions -- can be fed straight into
# TransitionDataset(win_transitions, game_vocab) or concatenated with the
# main corpus's own transitions list before building a combined dataset.
```

`E:\jepa_overflow\winning_harvest\winning_corpus\_harvest_summary.json`
(written by the extraction script) has the exact per-game win/transition
counts in machine-readable form, matching the table below.

## Results

**Written by the main session, not the harvesting subagent** -- the
subagent ran the actual harvest (all 25 games, `MAX_ACTIONS=2000`,
completed cleanly except `re86`, see below) but was redirected mid-task
to stop self-resuming per-leg for cost reasons and hand off the final
extraction step; this section covers what that extraction found.

**The WIN-only extraction criterion above found zero winning episodes,
across the entire harvest -- and that's a real finding about the task's
own scoping, not a failed harvest.** `scripts/extract_winning_transitions.py`
scanned all 26 raw files (49,474 frames total) for `state == "WIN"` and
found none. Checked directly rather than assumed: every frame in the
harvest is `NOT_FINISHED` (48,736) or `GAME_OVER` (738) -- `WIN` never
occurs once, on any of the 25 games, even at `MAX_ACTIONS=2000` (up to
~6.7x the normal 300-action budget). Also directly inspected the
`win_levels` field this project had never closely examined before: it's
a **static per-game property** (e.g. `r11l`'s is always `6`, unchanging
across the whole file and across 110 in-file RESETs) -- almost certainly
the *total* level count for that game, not a progress counter. It is
*not* the field this project's own agent-level metrics have ever
tracked; every "total levels completed" number in CLAUDE.md's history
(Stage 2 onward) comes from `levels_completed`, a genuinely different
field. This means the mining task's own premise ("harvest WIN episodes")
targeted a terminal state that essentially never occurs under this
policy/budget -- not a bug in the harvest, a mismatch in what was asked
for versus what this project's agents actually, routinely achieve.

**Corrected extraction, targeting the metric this project has actually
used all along:** built `scripts/extract_level_up_transitions.py`,
mining segments that end at a real `levels_completed` increase (from the
most recent RESET before the increase through the increase frame)
instead of full-game WIN -- the same reward signal Stage 5's value-head
training already targets (`jepa/data/value_targets.py`,
`NONZERO_THRESHOLD`). One more real property confirmed directly before
trusting this: `levels_completed` is **session-cumulative, not reset by
a RESET action** (`r11l` goes `0 -> 1` once at frame 57 and then stays
at `1` across the remaining 110 in-file resets) -- so a RESET-to-increase
segment captures the actual successful attempt, not an artifact of a
counter that resets on its own.

**Result: 4 level-up events found, across 4 of the 25 games, 552
total transitions** -- `ar25` (23 transitions), `ft09` (192), `lp85`
(280), `r11l` (57). Written to
`E:\jepa_overflow\winning_harvest\levelup_corpus\` (also
`_harvest_summary.json`), loadable directly via
`jepa.data.trajectories.load_transitions_from_dir`. This is genuinely
sparse -- not a bug, consistent with this project's own established
history: Stage 5's `Memory`-as-teacher pass (CLAUDE.md, same section)
got 5 level-ups across 5 games in a single 2500-action pass, a
comparable order of magnitude. A single `MAX_ACTIONS=2000` pass per game
was never going to produce a large corpus; a real "mine more of this"
effort would need either many repeated passes per game (this harvest was
one pass each) or a smarter/more-directed harvesting policy than plain
`Hypothesis`, not a criterion fix alone.

**One data-quality flag, not yet resolved:** `re86`'s subprocess exited
with a forced-termination code (`4294967295`, Windows' -1) at 265s during
the harvest, per the mining subagent's own report -- checked directly and
its recording file (14.7MB) ends on a syntactically complete, well-formed
JSON line, not a mid-write truncation, so it's usable as-is; contributed
zero level-up events either way (not among the 4 games above).

**Housekeeping done:** `Hypothesis.MAX_ACTIONS` reverted 2000 -> 300 in
`hypothesis_agent.py` (was left bumped after the harvest, per the
subagent's own flag). `E:\jepa_overflow\winning_harvest\recordings\`
(the raw 331MB harvest) is kept for now in case a different extraction
criterion is worth trying later -- not yet cleaned up.
