# CLAUDE.md

Guidance for Claude Code agents picking up this repo. Read this first --
`architecture.md`, `plan.md`, and `rules.md` are the design/reference docs
(architecture spec, staged build plan, Kaggle competition rules); this file
is the "where things actually stand" doc, kept up to date as work
progresses.

## Project

ARC-AGI-3 Kaggle competition (`arc-prize-2026-arc-agi-3`) entry: a
JEPA-style world-model agent that plays novel grid games via belief
refinement rather than an LLM. Side project / portfolio piece and a JEPA
learning vehicle -- see `plan.md`'s "Framing & goals" for what "success"
means here (not leaderboard-topping).

## CURRENT STATUS (2026-09-07) -- strategic reset; read this before anything below

**Everything from "## Status" down to the Gotchas section is still accurate as a
record of what was tried, but it is no longer the plan.** Three findings
established on 2026-09-07 supersede it. Full analysis:
`experiments/stage7_strategy_reset.md`; backlog is now tracked in GitHub Issues.

### 1. We were optimizing the wrong objective

The competition metric is **RHAE**, not levels-completed:

```
per level:        S_l = min(1.15, h_l / a_l)^2      h = upper-median best-human action count
per environment:  E_e = min( Sum_solved w_l / Sum_all w_n ,  Sum_ALL w_l*S_l / Sum_ALL w_l ),  w_l = l
```

**CORRECTION (2026-09-12): both denominators are over ALL levels, not solved
levels.** The earlier wording (`Sum w_l*S_l / Sum w_l`) did not say which, and
that ambiguity **flips the sign of every "does more play help?" estimate**. Read
from the real scorer in the mounted solver bundle
(`taaf/game.py: GameRun._compute_final_score`), not inferred: the efficiency
term's denominator sums over every level, so **E_e is monotone non-decreasing in
levels solved -- completing another level can never lower a game's score.**

Under the wrong (solved-only) reading, solving an extra level *dilutes* the
efficiency average and a projection of "2x more turns" comes out at a confident
**-29%**. Under the correct reading the same projection is **+66% floor**. A
reimplementation against the real scorer reproduces all 25 of a real run's
`final_score` values bit-exactly, and `benchmark.json` carries
`base_actions_per_level` (the true `h_l` human baselines), so nothing here needs
inferring. See `experiments/stage7_turn_latency.md`.

Efficiency is **squared** (2x human action count => 1/4 score; 10x => ~1%) and
completion is a **hard ceiling** weighted toward deep levels. `GraphExplorerAgent`'s
coverage-first sweep is therefore scored near zero *even on levels it wins* -- which
finally explains the paradox documented at length below: 46 local level-completions
producing a real score of 0.10-0.25.

**A substantial part of the Stage 6 "held-out-game generalization gap" (13+
interventions, 12 failures) was measuring the wrong thing, not a generalization
failure.** Any future local backtest must use RHAE. See GitHub issue #2.
(Formula source: ARC Prize technical report arXiv:2603.24621. The Kaggle Evaluation
page is a JS SPA and could not be read at source -- high confidence, not confirmed.)

### 2. Non-LLM approaches are not competitive at the target rank

Top 10% = **2.99** (rank 287 / 2,870 teams; #1 = 11.04). Verified from the live
leaderboard. The best **non-LLM** notebook in the entire public field scores **0.46**
-- and it is far more elaborate than anything here (beam/IDA*/MCTS against a cloned
simulator, PER buffer, attention CNN). ~40 of the top 50 public notebooks are forks of
one LLM harness (Tufa Labs' "Duck": a 27B model served by vLLM in-kernel, driving the
game through a Python REPL tool).

**Stop JEPA held-out-generalization work and coverage-first exploration as scoring
strategies.** This answers the open question left below in the GraphExplorerAgent
section ("is the held-out gap even the right problem to keep attacking"): no.

### 3. Corrected submission ledger

The record below documents 9 submissions and is wrong in several places -- notably it
states the gateway `retry-max-time` fix was "not yet re-submitted for real scoring"
when it **was submitted and scored 0.18** (`56043778`, 2026-09-06). The competition
API returns **20**:

| date | ref | kernel | score |
|---|---|---|---|
| 2026-07-16 | 54751735 | hypothesis | 0.06 |
| 2026-07-17 | 54771115 | hypothesis | 0.22 |
| 2026-07-20 | 54840911 | hypothesis | 0.16 |
| 2026-07-21 | 54864554 | hypothesis | 0.08 |
| 2026-07-22 | 54889426 | hypothesis | 0.00 |
| 2026-08-02 | 55195099 | hypothesis | 0.09 |
| 2026-08-13 | 55470338 | hypothesis | 0.18 |
| 2026-08-24 | 55728242 | hypothesis | 0.09 |
| **2026-08-25** | **55769792** | **duck fork (not our code)** | **1.77** |
| 2026-08-28 | 55843700 | hypothesis | 0.15 |
| 2026-08-29 | 55858509 | graph-explorer | 0.10 |
| 2026-08-30 | 55901265 | graph-explorer-learned | ERROR |
| 2026-08-30 | 55901513 | graph-explorer-learned | ERROR |
| 2026-08-31 | 55902368 | graph-explorer-learned | ERROR |
| 2026-08-31 | 55921318 | graph-explorer | 0.25 |
| 2026-09-01 | 55926701 | graph-explorer-learned | 0.08 |
| 2026-09-04 | 56003465 | graph-explorer-learned | 0.15 |
| 2026-09-05 | 56022267 | graph-explorer-learned | 0.05 |
| 2026-09-06 | 56043778 | graph-explorer-learned | 0.18 |
| 2026-09-07 | 56084133 | llm-world-engine (CodeWorldAgent) | 0.00 |

**Nineteen submissions of our own code never exceeded 0.25.** One run of a forked LLM
harness scored 1.77 -- our team's best by 7x, and it is not our work (verified
byte-identical to `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b`, 11/11 cells).

*Open discrepancy:* the leaderboard CSV reports 22 submissions for our team; the API
returns 20. Unresolved.

### 4. Two assets that CLAUDE.md did not previously mention

- **`CodeWorldAgent`** (`kaggle_submission_llm_world_engine/`, 1,433 lines): an LLM
  writes the Python source of a world model, an LLM-free beam search plans against it.
  This is exactly the mechanism of published SOTA (arXiv:2605.05138 reports 58.12% mean
  RHAE with it). Scored 0.00 for five diagnosed, mostly-fixable reasons -- GitHub issue
  #3. **It existed only on Kaggle**, in no local copy and no git history, until rescued
  on 2026-09-07.
- **The Duck fork's real ceiling is ~2.0-2.6, not 9.** The "LB 9" in that notebook's
  title is not a measured score: its author is rank 102 with a best of **3.58 over 103
  submissions**, and only 1 of 2,870 teams scores >= 9.0. Hardware, model mount and
  vLLM boot were all verified working; the one real defect found is that the rerun's
  time budget (`concurrency=28` x `max_runtime_s_per_game=7920`) needs ~8.8 h for 110
  games against a 9 h hard cap, with no soft deadline. GitHub issue #4.

**Reaching 2.99 is therefore genuinely hard and not a config fix.** The plan is to run
both tracks: a corrected Duck fork for a rank floor, and `CodeWorldAgent` as the real
deliverable.

### 5. Progress log — 2026-09-07 / 2026-09-08

**All Stage 7 work below is merged to `master`.** PRs #1, #5, #6, #7, #8, #9.

**The repo now has its own test suite: 84 tests, `pytest tests/ -q`.** It had
**zero** before 2026-09-07 (only the vendored framework's). Run them before any
merge. `pytest.ini` sets `pythonpath=.` / `testpaths=tests` and is scoped so the
vendored `ARC-AGI-3-Agents/` suite is unaffected.

**Rescued from permanent loss** (all were in no commit on any branch):
- `CodeWorldAgent` + `llm_engine/` — existed **only on Kaggle**, no local copy.
- `graph_explorer_THIRD_PARTY_LICENSE` — the MIT attribution this file claims is
  "carried forward verbatim". It was an unmet attribution obligation on disk only.
- 58 files total, plus 53 commits across 6 branches that existed nowhere but this
  machine, plus 8 unique checkpoint sets and 83 level-up segments (2,952
  transitions) archived to `E:\arc3_worktree_archive\`. ~13 GB reclaimed.

**Duck fork — three findings:**
1. **The "LB 9" in the notebook title was never a measured score.** Its author is
   rank 102, best **3.58** across **103 submissions**; only 1 of 2,870 teams
   scores >= 9.0. Hardware, model mount and vLLM boot were all verified fine.
   Our 1.77 (n=1) vs their 3.58 (best of 103) is ~2x, not 5x.
2. **Budget fix (merged, PR #7): hygiene, not scoring.** It converts a ~274s
   overrun into a ~900s margin. Wave 4 would have lost only ~3.5% of its playing
   time, not 100% — the original "recover 24% of games" estimate was wrong.
   Confirmed the rerun writes **no** `submission.parquet` (scoring is server-side
   via the gateway), so a hard kill cannot lose already-completed games.
3. **Concurrency raised 28 -> 37 (merged, PR #9), backed by a real measurement.**
   Aggregate throughput *does* rise with concurrency (288.7 -> 310.3 tok/s e2e).
   But picking the raw peak (48, at 322.0) is a **trap**: games run in
   fixed-length waves, so the last wave runs at leftover concurrency, and 48's
   third wave holds only 14 games. Weighted by wave size, 37 gives **+8.0%** and
   48 only +1.7%. **~+8% tokens is not ~+8% score** — RHAE squares efficiency and
   caps on completion.

**`CodeWorldAgent`'s 0.00 has a verified root cause (merged, PR #8):**
`draft_world_model` had an *unconditional* fallback returning the template
skeleton, installed without checking `outcome.ok`. That stub is an **anti-model**
— flat objective, and `self.model is not None` then permanently suppresses
re-drafting, so the repair budget drains on source that cannot pass replay by
construction. Four contributors were reproduced; the sharpest is that **the exec
sandbox's allowlist omitted `super`**, so a genuinely correct world model calling
`super().__init__()` was rejected outright. **Still unfixed and still binding:
throughput (0.02-0.27 actions/sec).** Whether the coder model can write a
replay-passing model for a real 64x64 game is **untested** — the local RTX 2070
cannot host it.

**New gotcha — vLLM reasoning-parser field name.** This build's Qwen3 reasoning
parser streams generated tokens as **`delta.reasoning`** — not `delta.content`,
and not the OpenAI-style `delta.reasoning_content`. A benchmark harness that
counted only `content` reported a healthy server as
`FATAL: server not answering: status=200 err=None` and produced a complete,
plausible-looking, entirely **empty** results table, because `ok` gated token
accounting. **Three free GPU runs were burned on two guessed fixes before anyone
dumped the raw SSE stream and simply read the field name.** Restates this file's
own standing lesson: instrument before theorising.

**Kaggle submissions:** none spent on 2026-09-08 — the slot is deliberately held
while the throughput lever (NVFP4 + MTP speculative decoding) is measured, so
budget fix + concurrency + throughput can ship as one submission rather than
three marginal ones.

## Repo / branch layout

- `master` -- Stage 0 (harness) is complete and stable here. Don't rebase
  or force-push this branch.
- `stage1-jepa` -- Stage 1 (JEPA encoder + dynamics predictor) work in
  progress, branched from master so Stage 1 iteration doesn't destabilize
  the working Stage 0 harness. Merge to master once Stage 1 clears its
  milestone (see Status below) or is deliberately parked as a documented
  limitation.
- Remote: `https://github.com/CalamityChasm/ARC-AGI-3-JEPAstyle_approach`
  (private). Git identity for commits in this repo: `CalamityChasm
  <calcrockett@gmail.com>` (repo-local config, not global -- check `git
  config user.name`/`user.email` if cloning fresh on a new machine, you'll
  need to set these locally again).

## Environment setup (new machine)

1. `python -m venv venv && venv\Scripts\activate` (or `source
   venv/bin/activate` on Linux/Mac).
2. `pip install -r requirements.txt`. Two things to watch:
   - The `torch` pin at the bottom of the file was installed as a **CPU**
     wheel (`--index-url https://download.pytorch.org/whl/cpu`) on the dev
     box this was built on, which had no CUDA GPU. On a machine with an
     RTX 2070 (or any CUDA GPU), install the CUDA build instead. **The
     `cu121` index has no wheel for `torch==2.12.1`** (that index only
     serves older torch versions) -- use `cu126` instead:
     `pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall`
     (check `https://download.pytorch.org/whl/torch/` for which `cuXXX` tags
     exist for the pinned version/your Python version before assuming
     `cu121` -- and note plain `pip install torch --index-url ...` without
     `--force-reinstall` is a no-op if any torch build is already installed,
     since the unpinned requirement is already "satisfied"). Verify with
     `python -c "import torch; print(torch.cuda.is_available())"`. Training
     code in `jepa/` doesn't currently call `.cuda()`/`.to(device)`
     anywhere -- that's the first thing to add before GPU training will
     actually use the GPU (see "Next steps" below).
   - If `pip install -r requirements.txt` hits `resolution-too-deep`, it's
     the langchain/langgraph/smolagents dependency graph -- don't loosen
     the version pins, they're deliberately pinned to
     `ARC-AGI-3-Agents/uv.lock`'s resolved versions because pip's resolver
     can't handle it unbounded.
3. Kaggle API credentials are **not** in git (`.kaggle/` is gitignored).
   Set up `~/.kaggle/credentials.json` (or `kaggle.json`) again on the new
   machine if you need to re-pull competition files -- see `rules.md` for
   the competition ref (`arc-prize-2026-arc-agi-3`).
   - **Correction to an earlier version of this doc:** only the
     `ARC-AGI-3-Agents/` framework code itself is committed to git --
     `ARC-AGI-3-Agents/environment_files/` (the 25 public games' actual
     `metadata.json`/`<game>.py` files, required for the harness to find
     any game at all -- without them `main.py` fails with "Game X not
     found in scanned environments, Available games: []") and
     `arc_agi_3_wheels/` are **not** in git and were never committed; both
     ship only inside the Kaggle competition dataset zip. Re-pull with
     `kaggle competitions download -c arc-prize-2026-arc-agi-3 -p <tmp
     dir>`, then extract just the `environment_files/` and
     `arc_agi_3_wheels/` top-level entries from the zip (it also contains a
     full mirror of `ARC-AGI-3-Agents/` including its `.git/`, which you
     don't need -- our own clone already has that). ~44MB total, downloads
     in seconds.
4. `ARC-AGI-3-Agents/.env` is also gitignored (has an API key in it) --
   copy `.env.example` to `.env` again. It should have
   `OPERATION_MODE=offline` (fully local play, no network needed) and
   `ARC_API_KEY` set to any value (an anonymous key works fine -- see
   `jepa`/harness code or just call
   `requests.get("https://three.arcprize.org/api/games/anonkey")` for a
   fresh one).
5. `ARC-AGI-3-Agents/recordings/` (the trajectory corpus Stage 1 trains on)
   is also gitignored and not transferred -- regenerate before training the
   predictor: `python scripts/run_stage0.py --agent random`, run ~6 times
   from repo root (25 games x 6 passes = 150 files, matching the corpus
   size this doc's Status section describes). Each pass currently exits
   with a non-zero code / `CalledProcessError` from `main.py` in offline
   mode even though every game completes and every recording file is
   written correctly (harmless -- looks like an offline-mode scorecard
   reporting quirk, not a data problem; verify file count/content with
   `python scripts/summarize_recordings.py` rather than trusting the exit
   code).
6. `checkpoints/` (trained model weights) is gitignored -- regenerable via
   `python -m jepa.train_encoder` then `python -m jepa.train_predictor
   --epochs 30` (30 epochs to match the milestone numbers quoted in
   Status below -- the script's own default is 10), not transferred.
   Retrain on the new machine (should be much faster on the RTX 2070 than
   the CPU-only dev box this was built on, once `.to(device)` calls are
   added -- see Next steps -- since neither script uses the GPU yet even
   when it's available). Re-running `python -m jepa.eval_stage1` against
   a freshly regenerated (different-random-seed) recordings corpus won't
   reproduce the exact `-8.7%` figure -- that's expected, the corpus is
   randomly generated each time -- but should reproduce the same
   qualitative result (predictor still fails to beat identity on
   changed patches).

## Status

### Stage 0 -- harness & data collection: DONE

`ARC-AGI-3-Agents/` (vendored framework + 25 public games) runs fully
offline. Two agents: `random` (built into the framework) and `pressonce`
(`ARC-AGI-3-Agents/agents/templates/press_once_agent.py`, a scripted
"press every action once" opening probe). Both log trajectories via the
framework's `Recorder` to `ARC-AGI-3-Agents/recordings/*.jsonl`
(gitignored, regenerate with `python main.py --agent=random` /
`--agent=pressonce` from inside `ARC-AGI-3-Agents/`, or run several passes
of `random` for more data -- that's what generated the 150-file, ~10.2k
transition corpus Stage 1 currently trains on).

### Stage 1 -- JEPA core: MILESTONE PASSED (as of 2026-07-08's bugfix; see item 10)

(Items 1-9 below document a long, genuine debugging journey that
concluded the milestone was *not* clearable with this architecture on
this data. That conclusion turned out to be built on a corrupted local
data pipeline -- see item 10 and the "CRITICAL" gotcha at the bottom of
this file. Left the full history in place rather than deleting it: the
individual fixes in items 1-3 were real and still matter, and the
debugging process in items 7-9 is exactly what you'd want to see before
concluding "not fixable" -- it just happened to be chasing a symptom of a
bug elsewhere, not a true architectural ceiling.)

Built (`jepa/` package):
- `jepa/grid.py` -- shared 64x64, 17-channel (16 ARC colors + 1 pad) grid
  representation used by both ARC-1/2 (pretraining) and ARC-3 (dynamics).
- `jepa/models/encoder.py` -- small CNN encoder, (17, 64, 64) ->
  (64, 8, 8) feature map, + EMA target encoder (momentum 0.996).
- `jepa/models/predictor.py` -- action- and per-game-conditioned one-step
  latent predictor (not Mamba yet -- that's Stage 3). Predicts a residual
  delta on top of the current feature map.
- `jepa/data/arc_static.py` -- loads all ARC-1/2 grids (16,668 of them,
  from `data/ARC-AGI-1` and `data/ARC-AGI-2`, gitignored git clones --
  re-clone from `github.com/fchollet/ARC-AGI` and
  `github.com/arcprize/ARC-AGI-2` if missing) for masked-patch JEPA
  pretraining.
- `jepa/data/trajectories.py` -- loads ARC-3 recording JSONL into
  `(frame_t, action, frame_t+1)` transitions, with a per-transition
  "changed" flag and per-8x8-patch change masks (see below for why these
  matter), plus a `game_id` -> index vocabulary.
- `jepa/train_encoder.py`, `jepa/train_predictor.py` -- training scripts.
- `jepa/eval_stage1.py` -- the milestone check: does the predictor beat a
  "nothing changes" identity baseline on held-out transitions?
- `jepa/device.py` -- shared `get_device()` (CUDA if available, else CPU).
  All three scripts above now do real `.to(device)` transfers (this used
  to not exist at all -- see iteration history below).
- `jepa/data/external_logs.py` -- optional supplementary data loader for
  the Kaggle dataset `calamitychasm/arc-3-logs` (a much larger, lower-
  quality/diversity bulk random-policy scrape across the same 25 games,
  ~105k transitions vs the local corpus's ~12k). Streams directly out of
  `data/arc3_logs.zip` (gitignored, ~2.6GB uncompressed, not extracted to
  disk) with per-game reservoir sampling so it supplements rather than
  drowns out the local corpus. Wired into `train_predictor.py` via
  `--external-per-game N` (opt-in; omitted by default). Notably, this
  corpus's frame-level changed rate is **~64%**, vs. the local corpus's
  much lower rate (see iteration #2 below) -- directly targets the
  changed-transition scarcity that's been the persistent bottleneck.
- `jepa/benchmark.py` -- benchmarking suite: `eval` (per-game breakdown of
  predictor-vs-identity on the held-out local corpus, appended to
  `logs/benchmarks/history.jsonl` so runs are comparable across
  experiments), `throughput` (CPU-vs-GPU training step timing), `history`
  (print past `eval` runs). See Status below for what it's already found.

**Current result:** see the dated entries below -- this section is kept
as a running log rather than a single "current" number, since comparing
across experiments (data mix, epoch count, GPU vs CPU) is the point of
`jepa/benchmark.py`.

**Iteration history (each fixed a real bug/gap, in order):**
1. First training run looked like a pass (predictor beating identity every
   epoch) -- but the eval was comparing the predictor's output against a
   *lagging EMA target* instead of a fair same-encoder baseline, which
   artificially inflated the identity baseline's error. Fixing the
   comparison methodology (both sides must use the same encoder) revealed
   the predictor was actually **24% worse** than identity.
2. Most random-policy ARC-3 transitions are exact no-ops (~37% frame-level,
   and even "changed" frames are usually ~90% static 8x8 patches) --a
   plain mean-MSE loss/eval is dominated by trivially-unchanged content and
   gives almost no signal for real dynamics. Oversampled changed
   transitions (`TransitionDataset.sample_weights`, 3x weight) during
   training: gap narrowed to **-15%**.
3. Added patch-level change-weighting to the *loss itself*
   (`jepa/losses.py: weighted_prediction_loss`, 8x upweight on patches
   whose pixels actually differ) on top of #2: gap narrowed to **-8.7%**.
4. Hypothesized the same action id meaning a different effect in each of
   the 25 games was a major confound for a memoryless shared predictor;
   added per-game embedding conditioning
   (`ActionConditionedPredictor(num_games=...)`,
   `jepa/data/trajectories.py: build_game_vocab`). Result: **no
   improvement** (-8.7% -> -8.7%). This hypothesis was wrong, or at least
   not the dominant factor -- worth knowing so a new agent doesn't
   re-attempt the same fix.
5. (2026-07-08, `local-only-30ep-baseline` in `logs/benchmarks/history.jsonl`)
   Fresh-machine transfer regenerated the local recordings corpus from
   scratch (new random seed, ~12k transitions instead of the original
   ~10.2k) and retrained 30 epochs -- reproduced the same qualitative
   result (**-0.3%** changed-patches, still FAIL) on different data,
   confirming #1-4 above weren't an artifact of one specific corpus draw.
   `jepa/benchmark.py eval`'s new per-game breakdown also showed *why* the
   pooled percentage is a noisy metric on its own: several games
   (`ft09`, `vc33`, `s5i5`) show triple-digit-percent "worse than
   identity" purely because their identity-baseline MSE is tiny
   (~0.00001-0.00003) -- a small absolute error swing there is a huge
   relative swing. Prefer eyeballing absolute MSE alongside the percentage
   for those games, not the percentage alone.
6. (2026-07-08) Added GPU support (`jepa/device.py`) and confirmed via
   `jepa/benchmark.py throughput` that the GPU is genuinely faster for the
   model compute itself (**16x** on this RTX 2070 vs CPU, dummy-data
   forward+backward). But a first real `--external-per-game 2000` run (see
   next entry) ran far slower wall-clock than that 16x would suggest --
   `TransitionDataset.__getitem__` does CPU-side one-hot conversion
   per-sample with `num_workers=0`, so a ~56k-transition corpus (12k local
   + ~44k external) makes single-process data loading the bottleneck, not
   GPU compute. Fixed going forward with `num_workers=4,
   persistent_workers=True, pin_memory=True` on both loaders in
   `train_predictor.py` (only kicks in when `device.type == "cuda"`) --
   but the *first* combined-data run in this history predates that fix,
   so its wall-clock time isn't representative of what a rerun would cost.
7. (2026-07-08, `combined-data-60ep-gpu` in `logs/benchmarks/history.jsonl`)
   Retrained with the `num_workers` fix applied: local (12k) + external
   arc-3-logs capped at 2000/game (42.8k) = ~54.8k transitions, 60 epochs,
   GPU. **Did not clear the milestone, and the changed-patches gap is
   nominally worse (-1.4%) than the -0.3% local-only-30ep baseline** --
   despite the external corpus's ~64% changed-frame rate (vs. the local
   corpus's much lower rate). More data + more epochs did not close the
   gap this time.
   - Checked for the obvious confound first: representation collapse (both
     predictor and identity errors shrinking together toward zero over the
     60 epochs, which the epoch-by-epoch log shows happening -- e.g.
     `val_identity_mse` goes from 0.00220 at epoch 1 to 0.00005 at epoch
     60). Directly measured encoder feature std on a held-out batch after
     training: **~1.1-1.4 per channel, comfortably above the
     `VARIANCE_FLOOR=1.0` in `losses.py`** -- so this is *not* classic
     collapse (features aren't going constant).
   - Best current hypothesis instead: **the encoder's 8x8-patch features
     apparently don't move much in feature space even when the
     corresponding pixels change locally.** If `current_frame` and
     `next_frame` are already close to each other in feature space at the
     patches that changed -- even though the raw pixels there visibly
     differ -- then the identity baseline is "cheating" at the
     representation level regardless of how much training data you throw
     at the predictor. This would explain why more (higher-changed-rate)
     data didn't help: the bottleneck may not be data volume at all, but
     whether the *encoder architecture* preserves enough of a local-change
     signal in its 8x8 feature map for there to be a gap to close. Not yet
     verified further (e.g. by directly inspecting per-patch feature
     deltas vs. pixel deltas) -- flagging as the most promising next
     thing to check, ahead of "more data" approaches.
8. (2026-07-08) Directly verified item 7's encoder-sensitivity hypothesis
   -- and it was **wrong**. Measured per-patch feature-space delta
   (`(f(frame_t) - f(frame_t1))**2`) at changed vs. unchanged patches on
   held-out data: changed patches show **12x larger** feature deltas than
   unchanged ones (mean 2.8e-4 vs 2.3e-5). The encoder *does* register
   local pixel changes fine. So the bottleneck isn't the encoder throwing
   away the signal.
   - Followed up by measuring the *predictor's own residual output*
     (`self.net(x)` before the `feat +` skip-add) against the true target
     delta (`f(frame_t1) - f(frame_t)`) on the same held-out batch: mean
     residual^2 = 2.0e-6 vs. mean true-delta^2 = 3.3e-5 -- the trained
     predictor's residual is **~16x smaller than the actual average
     change**. It has effectively learned to output near-zero and coast
     on the `feat +` skip connection, i.e. it learned to approximate
     identity rather than learning real dynamics, despite `feat` and the
     true target clearly differing at changed patches (item 8's first
     finding). This is the real bottleneck: not data, not the encoder, but
     the one-step predictor's inability (or the training setup's
     inability to make it) commit to a non-trivial residual.
9. (2026-07-08) Ran two targeted experiments to isolate *why* the
   predictor won't commit to real residuals, both **negative results**:
   - **Single-game ablation** (`jepa/train_predictor.py --game`, new flag):
     retrained on `bp35-0a0ad940` alone (2480 transitions, **100%**
     frame-level changed rate, the highest of any of the 25 games) --
     this removes the 25-games-at-once confound *entirely* (not just
     per-game conditioning, which iteration #4 already showed doesn't
     help). If cross-game interference were the cause, an isolated,
     abundant, always-changing single game should be the easiest possible
     case. Result: predictor was **worse than identity at every single
     epoch, all 60 of them** (epoch 60: pred=0.00691 vs identity=0.00623,
     ~11% worse) -- never once caught up, let alone surpassed.
   - **Zero-init residual branch** (`jepa/models/predictor.py`: the last
     `Conv2d` in `ActionConditionedPredictor.net` is now zero-initialized,
     so the model starts as an *exact* identity function and can only earn
     its way to a non-zero residual through training, rather than starting
     with random noise it has to first learn to suppress). Re-ran the same
     single-game ablation: **no meaningful change** (epoch 60:
     pred=0.00591 vs identity=0.00589, still ~0.3% worse, same shape of
     curve throughout). Kept the zero-init anyway as a harmless best
     practice for residual predictors, but it is not the fix.
   - **Conclusion:** this isn't a data-scarcity, cross-game-confound, or
     bad-initialization problem -- three independent interventions (9x
     more data with a much higher changed-rate; complete removal of the
     multi-game setting; zero-init of the residual branch) all failed to
     move the needle, each landing at essentially the same "slightly
     worse than identity" result. The most likely remaining explanation is
     that for single-step, per-frame MSE-optimal prediction, the
     conditional distribution of "what changes given this state+action" is
     genuinely close to i.i.d. noise from this model's point of view (a
     small 2-3-conv-layer network with only action/xy/game conditioning
     has no way to know e.g. exactly where a moving sprite currently sits
     with sub-patch precision, or resolve state that a random-policy
     rollout simply doesn't disambiguate) -- so "predict no change" really
     is close to the MSE optimum available to this architecture on this
     data, not a training failure to escape a local optimum. Closing this
     gap would likely need either (a) a fundamentally more expressive
     dynamics model (Stage 3's Mamba-based sequence model, which has
     *history*, not just one frame, to disambiguate state) or (b) training
     data from a policy that actually progresses through games
     purposefully rather than acting randomly (so state transitions are
     less arbitrary/noisy) -- not more of the same kind of random-policy
     data, regardless of volume.
10. **(2026-07-08) Root cause found: it was never the architecture --
    `ARC-AGI-3-Agents/agents/agent.py`'s `_convert_raw_frame_data` never
    copied `raw.action_input` into the `FrameData` it builds, so every
    recorded local transition had `action_id=0` (RESET, the field's
    default) regardless of what action was actually taken.** Local
    recordings are what `jepa/data/trajectories.py` reads for
    `action_id`, so every local-only signal in items 1-9 above was
    trained/evaluated with a **constant, wrong action label** -- the
    model never had a real chance to learn action-conditioned dynamics
    from the local corpus (external `arc-3-logs` data was unaffected --
    it has its own correctly-populated `action` field, unrelated to this
    framework's recording path). Found while building Stage 3 (a
    "recalling a known winning action" log line never fired despite 9
    resets in one run -- chased that down to *why*, and it led here).
    Fixed with a one-line change (`action_input=raw.action_input` added
    to the `FrameData(...)` call), regenerated the local recordings
    corpus, and reran the exact same combined-data (local + external
    `--external-per-game 2000`, 60 epochs) training as item 7:
    **changed-patches improvement flipped from -1.4% to +29.2% -- a clean
    PASS.** (`action-input-bugfix-60ep` in `logs/benchmarks/history.jsonl`.)
    Per-game breakdown shows real, uneven learning rather than a uniform
    shift: `r11l` +38%, `bp35` +37%, `vc33` +33%, `sp80` +12% (notably,
    exactly the games Stage 2/3's agents were already finding "easy" --
    a good consistency check), while several others (`s5i5`, `tn36`) are
    still negative. Items 1-9's debugging process wasn't wasted -- the
    loss-shaping fixes in items 1-3 are still real improvements baked
    into the current setup, and the single-game/zero-init ablations
    (item 9) were a reasonable, well-executed way to rule out
    architecture before suspecting the data pipeline -- but the final
    "not a fixable training bug" conclusion in the old summary paragraph
    below was wrong. Left for the historical record; superseded by this
    entry.

**Resolution: the milestone is PASSED.** What looked like a fundamental
single-step-predictor ceiling (items 7-9: three independent interventions
-- more data, removing the multi-game setting, zero-init -- all
converging on the same "a few percent worse than identity" result) turned
out to be fully explained by a data-recording bug that made the local
corpus's action-conditioning signal constant/wrong (item 10). With that
fixed, the same architecture that items 7-9 concluded couldn't learn real
action-conditioned dynamics does exactly that: **+29.2% on changed
patches.** The lesson for future debugging sessions: when several
different fixes all converge on the exact same negative result, that
consistency is *more* consistent with a shared upstream data bug than
with "we've exhausted the fixable hypotheses" -- worth auditing the data
pipeline itself (e.g. checking label distributions for suspicious
constants) before concluding a architecture/approach is capped.

## Stage 1 history note: the "pivot to Stage 2" recommendation was superseded

An earlier version of this doc recommended pivoting straight to Stage 2
without further Stage 1 effort, reasoning that three independent fixes
(more data, removing the multi-game setting, zero-init) all converging on
the same "predictor can't beat identity" result meant the single-frame
architecture had hit a genuine ceiling. Item 10 above found the real
cause instead: a data-recording bug, not an architecture limit. Once
fixed, the milestone passed outright.

That said, Stage 2 and Stage 3 got built anyway during this same session
(plan.md's guiding principle #4 -- "add components only when a measured
bottleneck demands it" -- was reasonably satisfied at the time, even
though the specific bottleneck turned out to be misdiagnosed), and both
are still worth keeping: Stage 2's curiosity agent and Stage 3's memory
agent both work *better*, not worse, now that the underlying Stage 1/3
world models are properly trained -- see their Status sections below for
the corrected numbers. Nothing about Stage 2/3's own designs depended on
Stage 1 having failed; they just inherited a broken world model
temporarily, and now don't.

Keep using the honest `changed-patches` metric (via `jepa/benchmark.py
eval`, which also gives the per-game breakdown -- not the naive
whole-grid MSE, which is misleadingly easy to "beat" by mostly predicting
no change) as the real bar for any future Stage 1 work, and append new
experiments to `logs/benchmarks/history.jsonl` via the benchmark tool so
they stay comparable to the items above. `jepa/train_predictor.py --game
<id>` (added this session) is there for fast single-game ablations if a
new hypothesis needs isolating from the 25-games-at-once setting again.
And if a "several different fixes all land on the same negative result"
pattern ever shows up again: audit the data pipeline for a shared
upstream bug before concluding the architecture is capped -- see the
"CRITICAL" gotcha at the bottom of this file for exactly that lesson.

### Stage 2 -- curiosity-driven agent: BUILT, milestone reasonably met (nuanced)

Built `ARC-AGI-3-Agents/agents/templates/curiosity_agent.py` (`Curiosity`
class, registered in `agents/__init__.py`, playable via
`python main.py --agent=curiosity` or `python scripts/run_stage0.py --agent curiosity`).
Uses the Stage 1 encoder + predictor checkpoints purely as an exploration
*ranking* signal (see plan.md: Stage 2 explicitly tolerates a noisy/
imperfect world model) -- reuses them as-is despite Stage 1 never clearing
its own milestone.

**Design (final, after three rounds of finding and fixing real bugs in
this session):**
- Each turn, compare what the predictor expected *last* turn (given the
  action taken) against the *actual* encoded outcome this turn -- that
  discrepancy is the observed "surprise," folded into a per-action running
  EMA (optimistic-initialized, so untried actions get tried first).
- The *next* action is chosen by ranking that per-action EMA (not a
  re-prediction each turn -- cheap dict lookups, no forward pass needed
  for ranking itself). A 25% epsilon-random fallback guards against
  fixating on something that's genuinely, repeatedly surprising but never
  productive (the classic curiosity "noisy TV" problem; a real fix is
  Stage 5's job, not Stage 2's).
- ACTION6 (needs an (x, y) click) competes as *one* option at the
  top-level ranking (mean surprise across all 64 patches) -- only once
  it's chosen does a separate, finer per-8x8-patch surprise map get
  consulted, via weighted-random sampling, to pick which patch, and then a
  uniform-random pixel *within* that patch (not always its exact center).
- On any observed increase in `levels_completed`, immediately repeats the
  last action for up to 2 more steps (plan.md: "exploit immediately on any
  observed score delta") -- kept short since a level-up usually means the
  board just changed underneath you, so blindly repeating the old action
  isn't guaranteed to still make sense.

**Bugs found and fixed, in order (each one first made the agent measurably
worse than random before the fix, on a matched 300-action-budget,
25-game, N-repeat comparison against a temporarily-bumped `Random` -- see
`ARC-AGI-3-Agents/agents/templates/random_agent.py`'s `MAX_ACTIONS`,
reverted to 80 once this comparison was done):**
1. Ranking by the predictor's *predicted* residual (a static function of
   the current frame, no feedback loop) got the agent stuck cycling the
   same ~4 actions forever, 0 levels completed in 300 actions -- a
   consistently-highest-predicted action never got penalized for actually
   producing no change. Fixed by tracking *observed* prediction error
   instead (real RND/ICM-style curiosity, not just "predicted novelty").
2. Ranking all 64 ACTION6 patches as top-level options (tied with simple
   actions at the same optimistic-init value) forced a mandatory ~64-action
   raster-scan of every click location before the agent could do anything
   else, burning ~20% of the budget on reconnaissance whether or not
   ACTION6 even mattered for that game. 8-vs-8 matched-budget comparison
   after fixing bug #1 alone: curiosity 8 total levels vs. random's 10 --
   *worse* than random. Fixed by making ACTION6 compete as a single
   top-level option (see Design above).
3. Even after #2, clicking only ever at a chosen patch's exact center
   throws away 7/8 of the pixel-level precision a fully uniform-random
   click has -- and the exploit-repeat was blindly repeating a
   just-successful action even across a level transition (new board
   layout). Fixed both (random pixel within the chosen patch;
   `EXPLOIT_REPEATS` cut from 5 to 2). Re-ran the 8-vs-8 comparison:
   curiosity 10 total levels vs. random's 10 -- **tied on raw count**, a
   real improvement from -2 to 0, but not yet a clear win.

**Final honest result (8 repeats each, matched 300-action budget, all 25
local games per repeat -- see git history for the exact numbers if this
needs reproducing):** raw total levels completed is *tied* (10 vs. 10
across 8x25-game repeats each) -- not the unambiguous "clearly beats
random" plan.md's Stage 2 milestone asks for, at face value. But per-game
breakdown tells a different story: **curiosity solved 6 distinct games
across its runs (`ft09`, `r11l`, `ar25`, `cd82`, `sp80`, `ls20`) vs.
random's 2 (`sp80`, `cd82`) — 7 of curiosity's 10 successes were on games
random never once solved in the same number of trials.** Random is
actually *more action-efficient* on the one game (`sp80`) both agents find
easy (avg. ~146 actions to first completion vs. curiosity's ~197 there),
but that seems to be a game where the fastest path is close to lucky
random search rather than anything "surprising" -- curiosity trades a bit
of efficiency on that one easy case for reaching several harder games
random's blind search essentially never touches. That's a real, directed-
exploration effect, not sampling noise (7/10 successes concentrated on
random's blind spots is a strong pattern) -- treating the milestone as
reasonably met on that basis, rather than continuing to chase a larger
raw-count margin on such a sparse metric (1-3 level-ups per 25-game sweep
makes the raw count alone noisy either way).

**If revisiting Stage 2 later:** the natural next lever, per plan.md's own
"exploit immediately" framing, would be a smarter exploit phase (right now
it's a blind fixed-length repeat) -- e.g. keep exploiting *only* while the
board keeps changing in the same direction, or track a short/local model
of "what changed the last time this exact action fired" instead of a
global per-action average.

**Update (2026-07-08, after the action-input bugfix -- see Stage 1 item
10 and the "CRITICAL" gotcha below):** re-ran the same 8-repeat matched-
budget comparison with Curiosity now loading the *correctly-trained*
Stage 1 checkpoints (it always loaded `checkpoints/encoder_finetuned.pt`
+ `predictor.pt` -- those files just got much better out from under it).
Result: 9 total levels across 8 repeats (vs. random's 10) -- essentially
unchanged from the pre-fix 10, not the dramatic jump you might expect
given the world model went from failing to passing its own milestone.
Distinct games reached did tick up (4: `ft09`, `m0r0`, `r11l`, `sp80`,
vs. the pre-fix run's overlapping-but-not-identical set) and `m0r0` is
new (a game random never solves) -- so the *directed-exploration* effect
looks a little more robust, even though the raw count didn't move. Best
read: Curiosity's exploration signal was already "good enough to be
useful" even riding on the old, milestone-failing predictor (item 8/9's
finding that residuals were near-zero doesn't mean *zero* signal, just
*weak* signal) -- so fixing the underlying model helped the model's own
metrics far more than it helped this particular agent's raw win count at
this sample size. Not a contradiction, just a reminder that "the world
model got better" and "the agent built on it wins more, at n=8 trials on
a sparse metric" are different claims.

### Stage 3 -- memory (recurrent core + exact transition graph): BUILT, both components pass their own checks

**Mamba substitution, decided upfront:** plan.md calls for "the Mamba
core" here. `mamba-ssm` has no prebuilt wheel for this Windows box, and
building from source needs a local CUDA toolkit exactly matching torch's
build -- this box has CUDA 13.0 via `nvcc` vs. torch's `cu126` build (a
real version mismatch, confirmed via a failed `pip install mamba-ssm
--no-build-isolation` attempt, not just a missing wheel). Used a
`GRUCell`-based recurrent core instead (`jepa/models/recurrent_predictor.py:
RecurrentActionConditionedPredictor`) -- satisfies plan.md's actual stated
requirement ("carries compressed history across the episode") without
that fragile dependency. Revisit real Mamba if this box's CUDA toolkit and
torch's build version ever get aligned.

**Built:**
- `jepa/data/sequences.py` -- loads local recordings as ordered
  *per-episode* sequences (not i.i.d.-shuffled single transitions like
  `trajectories.py`), chunked into fixed-length (16-step) windows for
  truncated BPTT. Local recordings only, not the external `arc-3-logs`
  dataset (that dataset has no clean per-episode boundaries in its
  schema -- see the module docstring).
- `jepa/models/recurrent_predictor.py` -- keeps Stage 1's spatial
  per-patch residual design, adds a `GRUCell` fed a pooled feature
  summary + action/xy/game conditioning each step; its hidden state is
  broadcast back in as one more conditioning channel. Hidden state resets
  at episode start (`init_hidden`), persists across steps within an
  episode/chunk.
- `jepa/train_recurrent_predictor.py` -- trains via truncated BPTT across
  each 16-step chunk (gradients flow across the whole chunk, hidden state
  does *not* persist across chunks of the same episode -- a
  simplification; still enough to learn from recent history).
- `jepa/memory.py: TransitionGraph` -- the "exact visited-transition
  graph" plan.md calls for. A plain dict keyed on hashed exact frame
  content (`blake2b`, 16-byte digest) -> `(action, xy) -> next_state`,
  built up during play, persisted for an agent's *whole lifetime*
  (every RESET within one game, not just one level attempt -- ARC-3
  RESETs return to the same starting frame, so a prior attempt's
  discoveries are exactly recallable). Unit-tested standalone (deterministic
  hashing, correct best-known-action tracking) before integration.
- `ARC-AGI-3-Agents/agents/templates/memory_agent.py: Memory` -- combines
  both: before falling back to curiosity-driven exploration (inherited
  design from Stage 2's `Curiosity`, same bug-fix history applies), checks
  whether the *exact current frame* has a known winning action already
  recorded -- if so, takes it immediately, no re-exploration. Also uses
  the graph to avoid re-trying (action, xy) pairs already tried from this
  exact state while untried ones remain (guaranteed local coverage,
  independent of the global surprise ranking).

**First recurrent-predictor training run hit the same action-input bug as
Stage 1** (see item 10 above and the "CRITICAL" gotcha below) -- since
`jepa/data/sequences.py` reads *only* local recordings (no external-data
fallback available for sequence data), it was 100% exposed to the bug,
more so than Stage 1's combined-data runs. First result: changed-patches
~identity-parity (~-0.5%), which in hindsight was never a real test of
whether memory helps -- the model was trained with every action
mislabeled as RESET. After the fix (regenerate local recordings, retrain):
**changed-patches improvement +21.3%** (pred=0.01727 vs identity=0.02193
at epoch 30) -- a clean, substantial win, consistent with Stage 1's own
post-fix result.

**Memory agent vs. Curiosity vs. random (8 repeats each, matched
300-action budget, all 25 local games per repeat, post-bugfix
checkpoints):** Memory: 10 total levels across 8 repeats, **5 distinct
games** (`ar25`, `cd82`, `m0r0`, `r11l`, `sp80`). Curiosity: 9 total,
4 distinct games. Random: 10 total (from the Stage 2 comparison,
model-independent so still valid), 2 distinct games. Memory reaches the
most distinct games of the three, including `cd82` and `ar25` that
Curiosity's own 8-repeat sample didn't reach in this round -- weak
evidence (still a sparse metric at this sample size) that the added
exact-memory and recurrent-history components help *reach*, more than
raw *count*, matching the pattern already seen going from random to
Curiosity. Did not specifically verify the exact-recall mechanism firing
in a live multi-reset run this session (added a real `logger.info(...)`
call for it in `memory_agent.py`, since `GameAction.reasoning` turned out
to be inert -- see the gotcha below -- but didn't have a confirmed
"recalling a known winning action" hit in the logs checked) -- worth
confirming directly in a future session, e.g. by forcing a
known-winning-state repeat and checking for that specific log line.

**Note on `GameAction.reasoning`:** setting `action.reasoning = "..."`
(done throughout this codebase, including `PressOnce`/`Random` from
before this session, and both `Curiosity` and `Memory`) does **nothing**
-- `GameAction` has no `reasoning` property, and `SimpleAction`/
`ComplexAction` (what `action.action_data.model_dump()` actually
serializes in `do_action_request`) have no `reasoning` field either. It's
a harmless but inert convention already present in the codebase before
this session, not something introduced here -- don't rely on it for
debugging; use real `logging` calls instead (see `memory_agent.py`'s one
example).

### Stage 4 -- mixture-of-gated-experts predictor: BUILT, milestone mostly met after adding the MiniGrid data plan.md originally specified

**Scope deviation, decided upfront (first attempt only):** plan.md's Stage
4 assumes pretraining on diverse generated grid envs (MiniGrid/Sokoban/
Crafter/procgen) on a 96GB box, with K=16-24 experts. The first attempt
(items 1-5 below) skipped that -- neither the data source nor the compute
existed yet for this project -- and trained on the ~55k-transition
combined ARC-3 corpus alone, with a smaller `--num-experts 8`. That
attempt's own conclusion ("closing this gap would most likely need
plan.md's originally-intended diverse multi-environment data") turned out
to be right, and is exactly what item 6 below did about it.

**Built:**
- `jepa/models/moe_predictor.py: MoEPredictor` -- K small pointwise-conv
  experts + a gate (pooled feat + action/xy/game conditioning -> softmax
  over K), weighted-sum combination. `jepa/models/moe_predictor.py:
  load_balance_loss` -- Switch-Transformer-style auxiliary loss (`K * sum_i
  f_i * P_i`, minimized at uniform usage, maximized at total collapse to
  one expert). Combiner network explicitly deferred, per plan.md's Stage 4
  scope.
- `jepa/train_moe_predictor.py` -- data/training setup matching Stage 1's
  `train_predictor.py` (i.i.d.-shuffled single transitions; MoE routing
  doesn't need Stage 3's temporal ordering), plus the load-balance loss
  term and (added for item 6) a two-phase `--pretrain-epochs N --epochs M`
  curriculum: pretrain on MiniGrid, then fine-tune on the ARC-3 corpus,
  sharing one encoder/predictor/optimizer and one game vocabulary (the 25
  ARC games + a single shared `"minigrid"` entry, built from the union of
  both sources up front so the game-embedding table is the same size/
  meaning in both phases).
- `jepa/data/minigrid_data.py` -- translates MiniGrid's native `(object,
  color, state)` grid encoding (`env.unwrapped.grid.encode()`, which
  notably does *not* include the agent -- overlaid separately from
  `agent_pos`/`agent_dir`) into the same flat 0-15 color-code grid
  `jepa/grid.py` already uses for ARC, so the identical encoder and
  training code work on both data sources unchanged. Generates unlimited
  random-policy trajectories across 21 environments spanning plan.md's
  target expert vocabulary (empty-room navigation, `DoorKey`'s key/door
  interaction, `SimpleCrossing`/`LavaCrossing`'s obstacle avoidance,
  `Dynamic-Obstacles`, `Fetch`'s pickup/carry, `Unlock`/`UnlockPickup`/
  `KeyCorridor`'s multi-step puzzles, `RedBlueDoors`, `GoToDoor`) --
  extremely cheap (~2,000 transitions/sec; 67,200 transitions across all
  21 environments in ~31 seconds). All MiniGrid transitions share one
  `game_id="minigrid"` (deliberately *not* one game_id per environment --
  the whole point of this data source is action semantics that are
  consistent across layouts, and a per-env embedding would let the model
  route around learning that instead of through it).

**Milestone is "gate activations show experts specializing; better
generalization than the Stage-3 monolith."** First attempt (items 1-5,
ARC-3 data only) achieved neither. Adding MiniGrid pretraining (item 6)
achieved the generalization half clearly and the specialization half
partially -- see item 6 for the full numbers. Items 1-5 are kept as the
record of *why* more data was the right next lever, not a wasted detour:

1. First run (`--num-experts 8`, `LOAD_BALANCE_WEIGHT=0.01` -- the same
   relative weight commonly used in LLM-scale MoE recipes): gate collapsed
   to an **exact constant** (0.125 for all 8 experts, std ~0.0001 across a
   whole validation batch, regardless of input) -- `load_balance_loss` hit
   its theoretical minimum of 1.0 almost immediately and stayed there.
   Root cause found by inspection, not guesswork: every expert's *last*
   layer was zero-initialized (mirroring `ActionConditionedPredictor`'s
   "start as identity" trick from Stage 1) -- with **all K experts
   producing bit-for-bit identical zero output** at init, and a uniform
   gate, every expert receives the *exact same* gradient every step and
   stays identical to every other expert forever. A genuine symmetry that
   gradient descent cannot break on its own, not a tuning problem. (The
   zero-init trick was correct for Stage 1's single monolithic predictor --
   it's specifically wrong for a multi-expert setup with a shared,
   symmetric initialization.)
2. Fixed the symmetry (small random init, not zero, on each expert's last
   layer -- `jepa/models/moe_predictor.py`) and reduced the load-balance
   weight 10x (0.001): experts *did* start differing from each other
   (measured directly: expert-output std across experts ~0.0038 vs. mean
   abs ~0.0032, i.e. genuinely different per-expert outputs) -- but the
   *gate* still converged to ~uniform (`load_balance_loss` -> 1.000-1.001)
   over 60 epochs, and `changed-patches` improvement was only **+0.3%**
   to **+0.9%** across reruns -- nowhere near Stage 1's fixed-monolith
   result on the *same* corrected data (**+29.2%**, see Stage 1 item 10).
3. Ruled out "the aux loss is still too strong" as the sole explanation by
   testing a **10x and 100x** further reduction (0.0001, then 0.00001,
   30 epochs each): `load_balance_loss` still drifted back toward ~1.0
   (uniform) by the end of training in both cases, just more slowly --
   e.g. at 0.00001 it started at 2.08 (real imbalance) epoch 1 but eroded
   to 1.001 by epoch 30. **Uniform blending is a genuine attractor for
   this main task loss itself**, not purely an artifact of the auxiliary
   term.
4. Tested **`LOAD_BALANCE_WEIGHT=0.0`** (no balancing pressure at all) as
   the other extreme: `load_balance_loss` rose to **~8.0** (out of a
   maximum of `num_experts=8` -- i.e. near-*total* collapse to a single
   dominant expert), the classic opposite MoE failure mode. So the
   trainable range for this loss doesn't sit at "some small positive
   weight" the way it does in typical large-scale MoE recipes -- it's
   bimodal here (uniform-blend or total-collapse), not a smooth dial.
5. Tested fewer experts (**K=3**, weight 0.0001, 30 epochs) in case 8
   experts across ~55k transitions was simply diluting the data too thin
   per architecture.md's own "expert collapse / data dilution" warning:
   same story (`load_balance_loss` -> 1.000, changed-patches ~identity
   parity).

**Working conclusion:** with this data scale (~55k transitions, far short
of plan.md's assumed diverse-multi-env pretraining corpus) and this expert
architecture (small, shallow, per-expert conv nets), a uniform blend of
all experts is a more loss-effective solution than genuine routing --
plausibly because averaging several noisy small experts reduces variance
in a way that helps the MSE objective directly, so the optimizer has no
incentive to commit to sparse routing regardless of how the load-balance
term is tuned. This is consistent with (not contradicting) Stage 1's own
guiding principle #2 ("this is a data-bound problem, not a capacity-bound
one") -- more experts without more/more-diverse data doesn't specialize,
it just re-derives an ensemble average of the same underlying signal.
**Closing this gap would most likely need plan.md's originally-intended
diverse multi-environment data (MiniGrid/Sokoban/Crafter/procgen), not
more tuning of the loss weight on the current ARC-3-only corpus** -- this
is the same "data, not architecture" lesson Stage 1 eventually landed on
too, just for a different symptom.

6. **(2026-07-08) Built the MiniGrid data pipeline (see "Built" above) and
   retrained with `--pretrain-epochs 20 --epochs 60 --num-experts 8`**
   (20 epochs on 67,200 MiniGrid transitions, then 60 on the ~55k ARC-3
   combined corpus, same held-out ARC-3 val split as every prior number in
   this section). Two real bugs found and fixed on the way (both general
   fixes, not MiniGrid-specific hacks):
   - `TransitionDataset.__getitem__` expects `frame_t[0]` to be the actual
     grid (the "list of one layer" convention `arc3_frame_to_tensor`/
     `patch_change_mask` already use for ARC-3 frames and
     `external_logs.py`) -- `minigrid_data.py`'s `_translate_frame` now
     returns `[grid]`, not a bare grid.
   - `jepa/grid.py: patch_change_mask` assumed its input was already
     exactly `(CANVAS, CANVAS)` (always true for ARC-3 frames, never
     checked because no other source existed yet) -- MiniGrid grids are
     smaller and vary in size (5x5 to 25x25). Fixed generally: place the
     diff top-left on a `(CANVAS, CANVAS)` "nothing changed" canvas before
     reshaping, mirroring `grid_to_tensor`'s own top-left placement
     convention, rather than special-casing MiniGrid.
   - Separately, the *first* two-phase training attempt hung indefinitely
     (confirmed via flat process CPU time over 70+ minutes, not a slow
     run) -- both training phases used `persistent_workers=True`
     DataLoaders, and the MiniGrid phase's worker processes weren't fully
     torn down before the ARC phase's new DataLoaders spawned their own,
     some kind of resource contention rather than a clean handoff. Fixed
     by not using `persistent_workers` in this script (it only ever has
     one phase transition, so the per-epoch respawn cost that flag avoids
     barely matters here) plus an explicit `del` of the first phase's
     loaders before building the second phase's.
   - **Result: changed-patches improvement +44.1%** (pred=0.01731 vs.
     identity=0.03094 at the final epoch) -- clearing the "better
     generalization than the Stage-3 monolith" half of the milestone
     decisively (beats Stage 1's fixed monolith at +29.2% *and* Stage 3's
     recurrent monolith at +21.3%, both on the same held-out ARC-3 data).
   - **Gate specialization: real, but a minority behavior, not the norm.**
     `load_balance_loss` still sits at ~1.006 (barely above the K=8
     theoretical-uniform minimum of 1.0) and mean gate entropy across the
     validation set is 2.05 nats vs. 2.08 nats for exactly-uniform (98.6%
     of max) -- so *most* inputs still get a near-uniform blend. But
     unlike every ARC-3-only attempt (items 1-5), where gate weights were
     measured at ~0 variance across an entire batch (literally constant),
     this run shows genuine per-input structure: **~4.4% of validation
     examples have entropy meaningfully below uniform, and ~3.9% have one
     expert clearly dominant (weight > 0.3)** -- with some individual
     examples routing almost entirely to a single expert (max observed
     weight 0.998). This is a qualitatively different, much less
     degenerate result than items 1-5's exactly-flat gate, even though it
     falls short of "most inputs show clear specialization."
   - **Interpretation:** MiniGrid's consistent action semantics and much
     higher changed-frame rate gave the shared encoder/predictor enough
     real signal to learn genuinely better dynamics overall (the
     prediction-quality half of the milestone), and gave the gate enough
     signal to learn *some* real routing rather than none -- but not
     enough to make specialization the dominant behavior across most
     inputs. Plausibly the *rest* of the milestone gap needs either more/
     longer MiniGrid pretraining (only 20 epochs were used here), a larger
     `--pretrain-epochs`-to-`--epochs` ratio, or noisy top-k gating
     (forces hard, sparse per-example routing rather than a soft blend --
     untried this session) rather than more data alone.

**Overall verdict:** the "better generalization than the monolith" half of
Stage 4's milestone is clearly met. The "gate activations show experts
specializing" half is genuinely, measurably better than the pre-MiniGrid
attempts (real per-input variation exists now, up to near-hard routing on
some examples) but isn't the dominant pattern across the validation set --
call this **mostly met**.

7. **(2026-07-08) Tried noisy top-k gating** (`jepa/models/moe_predictor.py`:
   `top_k` param -- per-expert trainable noise added to gate logits during
   training, `torch.topk` + `-inf`-masked softmax forces exactly `k`
   nonzero experts per example instead of a dense blend over all 8;
   `--top-k` CLI flag on `train_moe_predictor.py`). Same MiniGrid-pretrain
   + ARC-finetune curriculum as item 6, `--top-k 2`. **Did not clearly
   improve on item 6, and regressed prediction quality:**
   - changed-patches improvement **+21.9%** (pred=0.00710 vs.
     identity=0.00909 at epoch 60) -- roughly *half* item 6's dense-gate
     result (+44.1%), now behind Stage 1's monolith (+29.2%) again, and
     only narrowly ahead of Stage 3's recurrent monolith (+21.3%).
   - Checked whether the forced sparsity at least bought cleaner
     specialization to compensate: it didn't, really. Which *pair* of
     experts gets selected does vary meaningfully per example (usage
     counts across the 8 experts ranged 176-461, a 2.6x spread, so the
     top-k selection itself isn't degenerate) -- but the *weighting
     between the two selected experts* stays almost perfectly 50/50
     (mean entropy 99.5% of the 2-expert maximum). So top-k gating traded
     "blend all 8 near-uniformly" for "blend 2 (varying) experts
     near-uniformly" -- structurally sparser, but not more *confident*,
     and the forced hard cutoff apparently threw away useful signal the
     dense blend could still use (hence the accuracy drop). The same
     "hedge rather than commit" pull identified in items 1-5 persisted
     even under a harder structural constraint.
   - Not investigated further this session (e.g. `--top-k 1`, longer
     MiniGrid pretraining, a smaller/larger noise scale) -- flagging as
     the natural next things to try if a future session revisits this,
     but per this session's own results, more data (item 6) was a
     clearly better lever than forcing sparser routing (item 7).

**Final status:** item 6 (MiniGrid pretraining, dense gate) is the best
result and the one worth keeping/building on -- **+44.1% changed-patches,
beats both monoliths, partial-but-real gate specialization.** Item 7
(noisy top-k) is documented as a negative result on top of item 6, not a
replacement for it. `checkpoints/moe_predictor.pt` (gitignored, not
committed) was later restored to a dense-gate (non-top-k) checkpoint --
confirmed directly by inspecting its state dict, which has no
`noise_gate.*` keys (those only exist when `MoEPredictor` is constructed
with `top_k` set). The restore-retrain (same command as item 6, on a
freshly regenerated local corpus with a different random seed) landed at
**+22.9% changed-patches** rather than the original +44.1% -- expected
run-to-run variance from a freshly-regenerated corpus (see the "Gotchas"
section), not a regression; the qualitative conclusion (dense gate beats
both monoliths) held. This is the checkpoint Stage 5 below is built on.

8. **(2026-07-09) Tried adding Sokoban as a second synthetic pretraining
   source alongside MiniGrid** (`jepa/data/sokoban_data.py`, built on the
   `gym-sokoban` package -- installs cleanly alongside `gymnasium`/
   `minigrid` since it depends on the legacy `gym` package and
   `sokoban_data.py` instantiates `SokobanEnv` directly rather than going
   through `gym`'s environment registry, so there's no namespace
   collision). Motivation, not "more data" but genuinely *new mechanics*:
   MiniGrid's own pretraining win (item 6) came from exposing the shared
   encoder/experts to consistent-but-different action semantics, not from
   data volume alone -- Sokoban's "push a movable object with persistent,
   potentially irreversible consequences" is a causal pattern neither
   ARC-3 nor MiniGrid reliably exercises, and architecture.md's own
   expert-vocabulary list calls it out specifically. Given `game_id`
   already disambiguates per-source semantics, Sokoban got its own
   `game_id="sokoban"`, distinct from `"minigrid"` (their action spaces
   aren't consistent with *each other*, only internally).
   - **Two real bugs hit before a clean run was possible, both found and
     fixed the same session:**
     1. Sokoban's own action space is 9 actions (0=no-op, 1-4=push,
        5-8=move) -- one more than `jepa/models/predictor.py`'s
        `NUM_ACTIONS=8` (sized for ARC-3's 8 actions; MiniGrid's 7 already
        fit). Storing a raw action id of 8 crashed the shared
        action-embedding lookup with a CUDA "vectorized gather kernel
        index out of bounds" assert, on the very first training batch that
        happened to sample it. Fixed by dropping Sokoban's true no-op
        (redundant anyway -- "nothing changes" is already the trivial
        baseline everywhere else in training) and remapping the remaining
        8 actions (1-8) down to stored ids 0-7; `env.step()` itself still
        gets the *original* unshifted id so push/move semantics stay
        correct.
     2. Separately (a machine/environment issue, not a code bug, but
        directly caused a training run to crash mid-epoch with a
        `MemoryError` inside a DataLoader worker's `pickle.load`): the
        host machine ran completely out of disk space (54MB free out of
        931GB) while this ablation was running. On Windows, a full disk
        can prevent the pagefile from growing, which turns ordinary
        memory-pressure moments into hard `MemoryError`s rather than
        graceful paging -- confirmed by the crash clearing up completely
        after freeing space (pip cache purge, ~23GB; deleting stale
        already-documented `curiosity`/`hypothesis` evaluation recordings,
        ~4.8GB) and rerunning with no code changes. Worth remembering: a
        training crash with a generic `MemoryError` in a worker process is
        worth checking `df -h` / free disk space for, not just RAM --
        especially on Windows.
   - **Controlled comparison** (both runs on the identical local ARC-3
     corpus and identical, seeded MiniGrid corpus -- 67,200 transitions,
     confirmed byte-identical transition counts in both runs -- so the
     *only* difference is Sokoban's presence): `checkpoints_ablation/
     minigrid_only` (baseline, MiniGrid-only pretrain) reached **+29.5%**
     changed-patches (pred=0.00838, identity=0.01189) --
     `checkpoints_ablation/minigrid_sokoban` (treatment, MiniGrid+Sokoban
     pretrain, 33,600 additional Sokoban transitions) reached **+15.7%**
     (pred=0.00656, identity=0.00778) -- **a clear regression, nearly
     halving the improvement over identity, not an improvement.**
   - **Gate specialization: flat to slightly worse, not better.** Directly
     measured mean gate entropy and dominant-expert frequency on the same
     held-out ARC-3 validation split for both checkpoints: mean entropy
     99.3% of the uniform maximum in *both* cases (2.0639 vs 2.0649 nats,
     essentially identical), and the fraction of validation examples with
     one expert clearly dominant (weight > 0.3) was **lower** with Sokoban
     (0.6% vs 2.3%) -- so adding Sokoban didn't move specialization in
     either direction in any meaningful way, and if anything nudged it
     the wrong way on that one metric.
   - **Working hypothesis for the regression (not further verified this
     session, flagged for a future one if Sokoban is revisited):** Sokoban
     is known to be prone to irreversible deadlocks under random play --
     pushing a box into a corner or against a wall away from any target
     makes that box (and often the whole puzzle) permanently unsolvable
     from that point on, after which the remainder of an episode is just
     the player wandering a now-frozen, uninteresting room. The frame-level
     changed-rate measured during pipeline validation (~47%) looked healthy
     and comparable to MiniGrid's own (~43%), but "the frame changed"
     doesn't distinguish meaningful, diverse dynamics from directionless
     post-deadlock wandering -- so a real chunk of the 33,600 Sokoban
     transitions may be low-information noise diluting the pretrain
     signal rather than the intended new-mechanic enrichment, unlike
     MiniGrid where even a random policy still produces reasonably varied
     navigation experience throughout an episode. Not confirmed directly
     (e.g. by measuring how early episodes hit a dead/frozen state) --
     worth checking first if a future session revisits Sokoban, ahead of
     other levers like non-random (e.g. curriculum or reduced-box-count)
     Sokoban data collection.
   - **Per the standing instruction this was scoped under ("if that fails
     to get improvements, begin working on a teacher policy"): this is
     being treated as a clean negative result, not iterated on further
     right now.** `checkpoints/moe_predictor.pt` (the live, in-use
     checkpoint) was *not* overwritten by either ablation run (both wrote
     to `checkpoints_ablation/<variant>/` specifically to avoid clobbering
     the working item-6-lineage checkpoint during this experiment) -- no
     rollback needed. `jepa/train_moe_predictor.py --sokoban-episodes-per-config
     N` remains available (opt-in, default 0) for a future revisit.

### Stage 5 -- hypothesis bundle + directed action selection: BUILT, milestone MET (see the three "follow-up" sections below for the full arc -- this section documents the first three bugs found; the milestone wasn't actually cleared until follow-up 3)

**Design:** builds on Stage 3's `Memory` agent (exact transition-graph
recall, exploit-on-score-delta -- reused unchanged) and replaces Stage
2/3's EMA-based "observed surprise" ranking with plan.md's actual Stage 5
design, scoped to reuse already-trained components:
- **N parallel hypotheses** = Stage 4's K=8 MoE experts (the
  MiniGrid-pretrained checkpoint above), each treated as one hypothesis
  about "what a given action does"
  (`jepa/hypothesis_bundle.py`, `MoEPredictor.predict_all_experts`).
- **Bayesian confidence update** (`p(Hi) *= exp(-error_i / tau)`,
  renormalized) after observing each transition's actual outcome.
- **Entropy of the confidence distribution -> beta**: uncertain -> explore
  (trust InfoGain); confident -> exploit (trust the value head).
- **Q(s,a) = (1-beta)*InfoGain(a) + beta*V(next_state(a))**. InfoGain(a) is
  disagreement across the K experts' raw predictions for candidate action
  a, computed in a single forward pass (the per-patch variance map doubles
  as the ACTION6 click-location salience map). V is a small decoupled
  value head (`jepa/models/value_head.py`) trained via discounted Monte
  Carlo returns from the sparse `levels_completed` signal.
- **Experiment-designer opening probes**: try every simple action once at
  episode start before trusting the bundle's own confidence weights.

**Bug 1 -- value-head/encoder latent-space mismatch.** The value head had
been trained against `encoder_finetuned.pt` (Stage 1's encoder), but
`Hypothesis` loads `encoder_moe.pt` (Stage 4's separately-trained
encoder) -- two independently-trained encoders end up with different,
incompatible latent spaces, so a value head fit to one is close to noise
fed features from the other. Fixed by retraining the value head directly
against `encoder_moe.pt` (`python -m jepa.train_value_head --epochs 20
--encoder checkpoints/encoder_moe.pt`). Worth flagging even after the
fix: the value-head training data is extremely sparse (12,150 samples,
only **1.6% with a nonzero value target**, since `levels_completed`
deltas are rare events under a random-ish policy) -- the retrained head's
val MSE (0.0016-0.0022 across 20 epochs) sits right on top of the
zero-baseline MSE (0.0019), i.e. it's only marginally distinguishable
from "always predict zero." This isn't a bug, just an honest limit of
what a value head can learn from this reward density -- flagged up front
since it explains part of what follows.

**Bug 2 -- ACTION6's InfoGain used a different reduction than every other
action's, an apples-to-oranges comparison that made ACTION6 win almost by
construction.** First full evaluation (8 repeats, matched 300-action
budget, all 25 games, same protocol as Stage 2/3's comparisons):
**0 total levels completed across 200 runs**, versus Curiosity's 11 (5
distinct games) on the same protocol. Tracing an episode's action log
showed the agent spamming `action_id=6` almost every single turn until
`GAME_OVER`, resetting, then spamming it again. Root cause, found by
direct code inspection: in `_score_action`
(`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`), the non-ACTION6
branch computes `ig = info_gain(expert_preds).item()` -- a **mean** of
expert-disagreement variance over every channel and spatial position --
while the ACTION6 branch computed `ig = patch_var.max().item()` -- the
**max** variance over the 64 spatial patches. A max over many patches is
almost always larger than a global mean (especially given that feature
variance is naturally non-uniform across space), so ACTION6 was
structurally near-guaranteed to score highest regardless of what the
experts actually predicted. Fixed by using `patch_var.mean().item()`
instead (mathematically identical to `info_gain()`'s reduction, since
`var(dim=0).mean(dim=0).mean()` over the remaining spatial dims is the
same value as `var(dim=0).mean()` over channels+spatial directly) --
`patch_var.argmax()` is still used separately to pick the click location,
only the cross-action-comparable scalar changed.

**Bug 3 -- no epsilon-random fallback, so the agent locked onto a single
action for an entire episode once bug 2 was fixed.** Re-testing after the
bug 2 fix alone: **3 total levels, 1 distinct game (`sp80`)**, still far
below Curiosity. Tracing episodes again showed a new pattern: the agent
would pick one action and repeat it 25-30 times straight until
`GAME_OVER`, reset, then lock onto a *different* single action for the
next attempt. This is the exact same failure mode Stage 2's `Curiosity`
hit first (see that section's bug 1): `InfoGain(a)` is a *predicted*
disagreement signal recomputed fresh each turn from a near-deterministic
forward pass, with no mechanism to decay once an action's real effect
turns out unsurprising in practice -- unlike Curiosity's own EMA-of-
*observed*-error ranking, which naturally cools down on a repeatedly-tried
action. Fixed the same way Curiosity was fixed: added a 25%
epsilon-random fallback (`Hypothesis.EPSILON = 0.25`) before the greedy
Q-argmax. (Note: sp80's own ~30-action reset cadence, which looked
suspicious while debugging this, turned out to be a property of the game
itself, not a bug -- `Curiosity`'s log on the same game shows an
identical ~30-action reset cycle.)

**Bug 4 -- unbounded confidence accumulation caused runaway certainty,
handing control to the (near-noise) value head for most of every
episode.** After bugs 2-3 were fixed, a targeted diagnostic
(`scripts/diagnose_hypothesis_beta.py`, replaying 20 real local-recording
episodes through the bundle's confidence-update logic without actually
playing games) showed `beta` averaging **0.76** and exceeding 0.5 in
**86%** of transitions -- i.e. the agent was in "trust V" mode for nearly
the whole episode, not the intended entropy-gated blend. Root cause:
`HypothesisBundle.update` accumulated `log_weights += -errors / tau`
every step with no forgetting term, and with `tau=0.01` even tiny,
possibly-spurious per-step differences between experts' errors compound
without bound over a ~300-step episode -- entropy collapses to
"confident" within the first few dozen steps of essentially every
episode and stays there regardless of what happens afterward. Given the
value head is barely distinguishable from a zero baseline (bug 1's
finding), that means action selection was effectively driven by
near-noise for most of an episode. Fixed by adding a geometric forgetting
factor (`log_weights = decay * log_weights + (-errors / tau)`,
`HypothesisBundle.decay`, default now 0.8) so confidence tracks *recent*
reliability rather than accumulating forever. Swept decay in {1.0, 0.95,
0.8, 0.6, 0.4, 0.2} on the same 20 replayed episodes before picking 0.8:
0.95 only partially helped (mean beta 0.61, still >0.5 62% of the time);
0.8 was the best balance (mean beta 0.37, >0.5 only 10% of the time) --
low enough to stop runaway certainty from swamping InfoGain, without
decaying so hard (0.6 and below effectively zeroed beta almost
everywhere) that V never gets to matter at all, which would just make
this equivalent to Curiosity's own ranking with extra steps.

**Final result (8 repeats, matched 300-action budget, all 25 games, all
four bugs fixed):** **1 total level completed, 1 distinct game (`r11l`)**,
versus Curiosity's 11 total / 5 distinct games on the identical protocol.
Milestone ("directed exploration beats the Stage-2 curiosity agent on the
same levels in fewer actions") is **not cleanly met**. A same-game,
matched comparison on `r11l` (the one game both agents solved in this
round) gives a genuinely mixed picture rather than a flat loss, though:
Hypothesis solved it in **53 actions** on its one success, well under
Curiosity's own average of **~175 actions** across its 6/8 successes on
that game -- but Hypothesis only succeeded on **1 of 8** attempts versus
Curiosity's **6 of 8**. So on the one clean head-to-head data point
available, Hypothesis is markedly *faster* when it works, but far less
*reliable* than Curiosity -- a real, honest tradeoff, not a clear win.

**Why reliability is the remaining gap, and why it isn't a quick fix:**
the two most likely structural causes are both already-documented,
data-bound limitations from earlier stages, not something another loss-
tuning pass would fix tonight: (1) the value head's training signal is
~98% zero-target (bug 1) -- meaningfully improving it needs trajectory
data with a much higher rate of real `levels_completed` events, i.e. data
from a policy that actually makes progress rather than more random-policy
volume, echoing Stage 1's own "data, not architecture" lesson; (2) Stage
4's own finding that MoE gate/expert specialization is "real but a
minority behavior, not the norm" caps how informative the Bayesian
hypothesis-confidence mechanism can be in the first place, since it's
built directly on those same experts' disagreement. Both point toward
plan.md's Stage 6 backlog item ("optional public human-trajectory data to
tune which probes are informative") as the well-scoped next lever, rather
than further tuning of `tau`/`decay`/`EPSILON` on the current data --
consistent with plan.md's own guiding principle of adding a component
only when a specific measured bottleneck calls for it, and not chasing
further gains on an already-sparse, already-diagnosed metric.

**Also fixed along the way, unrelated to the bundle's own logic:** the
harness's anonymous API key (`ARC-AGI-3-Agents/.env`'s `ARC_API_KEY`)
had expired between sessions, causing every game-listing request to
return HTTP 401 and every local run to silently produce a zero-action,
zero-level recording (a first attempt at this stage's evaluation looked
like a total agent failure for this reason before the real key-expiry
cause was found) -- refreshed via
`https://three.arcprize.org/api/games/anonkey`, per this doc's own setup
instructions. See the "Gotchas" section below.

### Stage 5 follow-up -- a "teacher policy" for denser value-head data: real component-level win, no clear agent-level win yet

Given the Sokoban ablation (Stage 4 item 8) came back negative, the
originally-scoped fallback was tried instead: rather than building a new
RL-per-game or search-from-scratch teacher, `Memory` (Stage 3) was reused
directly as a "teacher" -- it already does exact-graph-guided, non-
repeating, curiosity-ranked exploration and persists its transition graph
across resets, so running it with a much larger action budget than its
usual milestone-comparison budget approximates directed search without
new, riskier exploration code. `Memory.MAX_ACTIONS` was temporarily
bumped 300 -> 2500 (mirroring this project's established practice of
temporarily bumping an agent's budget for a data-generation/comparison
run, then reverting -- see Stage 2's `Random` history), one pass across
all 25 games was recorded, and the budget was reverted immediately after.

**Teacher pass result:** 5 levels completed across 5 distinct games in a
single 25-game pass (`ar25`, `cd82`, `lp85`, `m0r0`, `r11l`) -- versus
random policy's ~1.6% nonzero-value-target rate over the entire existing
150-file corpus. A genuine density win in absolute terms.

**First retrain on the combined corpus (unweighted): a real negative
result, same failure class as Stage 1's earliest predictor iteration.**
`jepa/train_value_head.py`'s loss is plain unweighted MSE over
per-transition discounted-return targets (`GAMMA=0.95`,
`jepa/data/value_targets.py`). Memory's 2500-step episodes are far longer
than the ~80-step episodes everything else in the corpus produces, and a
discounted return decays to a *technically* nonzero but *practically*
negligible value hundreds of steps before any actual reward event --
meaning most of the ~62.5k newly-added samples are indistinguishable
from zero to an unweighted loss, while inflating the denominator. Result:
`val_mse` matched `zero-baseline_mse` bit-for-bit at **every one of 20
epochs** -- the head learned nothing beyond "always predict ~0," a
strictly worse outcome to observe than the earlier (pre-teacher-corpus)
run, which at least showed epoch-to-epoch movement. This is the exact
"unweighted loss dominated by an overwhelming trivial majority" failure
Stage 1 hit with its first predictor iteration -- same root cause,
different component.

**Fix: oversampling, same pattern as Stage 1's `sample_weights()`.**
Added `ValueDataset.sample_weights()` / `NONZERO_WEIGHT=25.0` /
`NONZERO_THRESHOLD=1e-3` to `jepa/train_value_head.py`, wired through a
`WeightedRandomSampler` on the training split only (validation stays
unweighted/natural-distribution, same rationale as every other honest
eval in this project). Retrained: **`val_mse` on the full validation
population got *worse* than the zero baseline in every epoch (0.0018-
0.0051 vs a flat 0.0011)** -- at first glance a regression, but this is
measuring the wrong thing, exactly the way Stage 1's original whole-grid
MSE was the wrong thing to measure (see that stage's own history) --
98.8% of validation samples are still near-zero-target, so any
population-wide average is dominated by them regardless of what the
*meaningful* cases look like. Evaluating specifically on the meaningful
(`target > 1e-3`) validation subset instead (mirroring Stage 1's own
"changed-patches" pivot away from whole-grid MSE) tells a different,
honest story: **`pred_mse=0.0633` vs `zero-baseline_mse=0.0976` on that
subset -- a real ~35% improvement -- with a genuine positive
correlation (0.284) between predicted and true value.** A small positive
bias exists on typical/near-zero states too (mean predicted ~0.017 vs
true ~0, directly measured on 500 held-out typical samples) but isn't
severe enough to look like miscalibration collapse -- some spread exists
(std 0.049, range roughly -0.05 to +0.18), not a constant output.

**Downstream agent-level result: no clear win at this sample size.**
Re-ran the same matched 8-repeat, 25-game `Hypothesis` vs `Curiosity`
comparison with the improved value head: **Hypothesis 0 total levels / 0
distinct games** (down from the prior best of 1/1), **Curiosity 8 total /
2 distinct games** (down from its own prior 11/5, on unchanged
checkpoints -- expected sampling noise, not a real change). Hypothesis
landing at 0/8 this round is *not* strong evidence of a new regression:
its best-ever observed success rate was already only 1/8 (~12.5%), and
`P(zero successes in 8 trials | p=0.125) ~= 0.34` -- a highly plausible
outcome under pure sampling variance at this rate, not an unusual one.
Consistent with this project's own repeated observation that raw win/
level counts are a genuinely noisy metric at n=8 on a 25-game sweep (see
Stage 2 and Stage 5's own earlier sections) -- a component-level
improvement of this size (a value head that's honestly better, but still
noisy, on a rare event type) isn't guaranteed to be *detectable* in a
full-agent milestone comparison this small, even when the component
itself measurably improved.

**Housekeeping:** `Memory`'s 2500-action teacher pass produced
individual recording files up to 331MB (frame-size-dependent, not a bug
-- some games' per-frame JSON is simply much larger than others, and at
31x the usual episode length that difference compounds) -- 2.3GB total
for 25 files, ballooning `ARC-AGI-3-Agents/recordings/` to 7GB combined
with this round's evaluation recordings. All deleted after extracting
results and saving the retrained `value_head.pt` (fully regenerable via
the exact steps above; nothing here is uniquely irreproducible).
`scripts/compare_agents.py` was fixed to glob per requested agent name
rather than parsing every `*.recording.jsonl` file before filtering --
parsing multiple 300MB+ files just to discard them was the direct cause
of a comparison script silently taking several minutes instead of
seconds.

**Where this leaves Stage 5:** the value head component itself is now
demonstrably better on the metric that matters (ranking/distinguishing
meaningful states, not aggregate population MSE) -- a real, verified
improvement worth keeping. Whether it meaningfully helps the full
`Hypothesis` agent's reliability gap remains genuinely unresolved, not
because the fix didn't work, but because the current evaluation protocol
(8 repeats) doesn't have the statistical power to tell a real
improvement of this size apart from noise on a metric this sparse. A
future session wanting a real answer here should either run substantially
more repeats (e.g. 25-30 instead of 8) or use a less binary,
higher-resolution metric than raw levels-completed count (e.g. the
`InfoGain`/`beta` diagnostics already built in `scripts/
diagnose_hypothesis_beta.py`, or tracking `Q` values / value-head output
directly across a fixed action sequence, rather than requiring an actual
game win to register any signal at all) -- not further changes to the
teacher-data pipeline itself, which already did its job.

### Stage 5 follow-up 2 -- finding the actual bottleneck: a real architectural bug, found and fixed, plus a validated design choice

Rather than guess at further fixes, went looking for direct evidence of
*where* `Hypothesis` was actually failing, in three cheap steps before
touching any code:

1. **Confirmed the exact-recall mechanism isn't silently broken** (a
   loose end flagged back in Stage 3 and never checked). Ran `Hypothesis`
   on 5 games with real budgets: the "recalling known winning action" log
   line never fired, but that's fully explained by zero level completions
   ever occurring in that run to record in the first place -- not a
   broken mechanism, just nothing yet to recall.
2. **Live-traced full episodes with per-step `Q`/`beta` logging**
   (temporary `logger.debug` instrumentation in `_score_action`, gated
   behind `.env`'s existing `DEBUG` flag so it's zero-cost when off --
   left in permanently). Two things jumped out immediately on a `bp35`
   trace (a game this project's own Stage 1 history flags as
   high-activity/click-relevant): candidate-action `Q` margins were
   frequently within 0.0001-0.001 of each other -- close enough to be
   dominated by noise rather than real signal -- and **ACTION6 (the click
   action) scored lowest of the four candidates in nearly every single
   decision across the whole 300-action episode.**
3. **Root-caused the ACTION6 finding.** `_score_action` computed
   ACTION6's score at a neutral center point `(32, 32)` (the same point
   used for every action, on the reasoning that xy conditioning
   broadcasts uniformly and shouldn't bias which patches look
   informative -- true, but beside the point: it meant ACTION6 never got
   scored at *its own best location* before being chosen). A first fix
   attempt -- re-scoring ACTION6 at its own best-variance patch instead
   of the neutral point -- changed *nothing* in the live trace. Digging
   into why revealed the real issue: `MoEPredictor._condition` broadcasts
   the `xy` embedding as a **uniform additive bias across every one of
   the 64 spatial patches equally**, not a spatially-localized signal --
   so telling the model "evaluate as if clicking here" doesn't make it
   attend to that location differently at all, it just shifts the whole
   feature map by the same constant. The re-scoring attempt was reverted
   (confirmed dead weight, not worth the extra forward pass).

   The actual fix: `ig`'s spatial reduction was the real lever, not
   *which* xy to condition on. A flat mean over all 64 patches (the
   correct fix for the earlier apples-to-oranges bug, see Stage 5's first
   set of bugs above) structurally underrates ACTION6, whose true value
   comes from *one* good click location, not an average over 63 mostly-
   irrelevant ones -- while the original flat *max* over 64 patches
   overrated it via extreme-value inflation (a max over many samples is
   statistically larger than a single-shot evaluation for spatially-
   uniform actions, independent of any real signal). Implemented a
   top-k-patch mean instead of a flat mean or flat max
   (`jepa/hypothesis_bundle.py: info_gain(..., top_k_patches=...)`,
   `Hypothesis.TOP_K_PATCHES = 8` out of 64, an unswept starting point
   deliberately between the two failure modes), applied via the *same*
   reduction to every action (not a special case for ACTION6 -- that
   would just reintroduce the original bug in a different shape). This
   also let the redundant second forward pass from the reverted attempt
   be dropped -- one neutral-point pass is enough for every action now.

   **Result: 0 total levels / 0 distinct games -> 6 total levels / 3
   distinct games** (`ft09`, `m0r0`, `r11l`) across a fresh matched
   8x25-game evaluation -- a clean, unambiguous improvement, not noise.
   This was a genuine, previously-undiagnosed architectural bug, not a
   data or tuning problem -- the first concrete evidence this session
   that part of Stage 5's reliability gap was fixable in the model
   itself, not just in the training data feeding it.

4. **Ablated the Q-blend itself** (`Hypothesis.FORCE_BETA`, a class
   attribute defaulting to `None` for the real entropy-driven beta;
   temporarily set to a fixed float to test each extreme, mirroring this
   project's established bump-and-revert pattern for controlled
   comparisons) to check whether combining InfoGain and the value head is
   actually earning its complexity, now that the ACTION6 bug no longer
   confounds the picture. Matched 8x25-game runs for each condition (some
   partial -- an intermittent transient failure in the harness's
   always-hit-the-real-API game-listing call, unrelated to this ablation,
   truncated 3 of 8 repeats for `beta=1` and 1 of 8 for `beta=0`; sample
   sizes of 125-208 are still large enough for a clear qualitative read):

   | condition | total levels | distinct games | runs |
   |---|---|---|---|
   | `beta=0` (pure InfoGain) | 4 | 1 (`m0r0`) | 175 |
   | `beta=1` (pure value-greedy) | 1 | 1 (`r11l`) | 125 |
   | full blend (entropy-driven, default) | 6 | 3 (`ft09`, `m0r0`, `r11l`) | 208 |

   **The full blend clearly beats both extremes on both metrics.**
   Neither InfoGain nor the value head alone comes close to matching the
   combined design -- this validates Stage 5's original entropy-gated
   `Q = (1-beta)*IG + beta*V` design as a real, load-bearing choice, not
   speculative complexity worth stripping out. `FORCE_BETA` is kept as a
   permanent (harmless, defaults to off) hook for any future re-ablation
   rather than a one-off throwaway change.

**Where this actually leaves Stage 5:** the ACTION6 scoring bug was a
genuine, previously-undiscovered piece of the reliability gap, now fixed
with direct, verified evidence of improvement. The Q-blend design itself
is now empirically validated, not just theoretically justified. Both are
real progress beyond the "value head improved in isolation, agent-level
result inconclusive" state from the previous follow-up section -- this
is the first fix in this whole Stage 5 arc with a *directly measured,
unambiguous* full-agent-level improvement behind it, not just noise-
bounded or component-level evidence.

**Correction, immediately after (2026-07-10): the milestone is still not
met.** The 0/0 -> 6/3 result above is valid evidence the ACTION6 fix
helped `Hypothesis` against its *own* prior broken baseline, but it was
never actually re-checked against a *fresh* `Curiosity` baseline in the
same round -- Curiosity's own numbers have ranged 8-11 total / 2-5
distinct games across different rounds all session, so an isolated
Hypothesis-only number doesn't tell you where it stands relative to a
moving target. Ran the missing matched comparison: **Hypothesis 4 total
levels / 1 distinct game (`m0r0`) vs Curiosity 9 total / 5 distinct
games (`ft09`, `lp85`, `m0r0`, `r11l`, `sp80`)**, same round, both
checkpoints/code as currently committed. Curiosity clearly wins on both
metrics, especially breadth (5 distinct games vs 1). (One operational
note along the way: the harness's anonymous API key had expired *again*
before this run -- same symptom as before, all 16 subprocess calls
returning HTTP 401 and silently producing zero real gameplay; refreshed
via the same `https://three.arcprize.org/api/games/anonkey` endpoint and
reran cleanly. These anonymous keys appear to be short-lived enough to
expire within a single working session, not just between sessions --
worth checking first, not last, if a comparison run ever comes back
suspiciously empty.)

The ACTION6 fix and Q-blend validation are still real, worth keeping, and
directly demonstrated to help `Hypothesis` relative to its own prior
broken state -- just not sufficient on their own to close the full gap
against Curiosity. The milestone remains open.

### Stage 5 follow-up 3 -- the milestone is met: a second real bug, found via targeted diagnosis on the games where the gap actually was

Rather than keep re-running the full 25-game sweep hoping for a different
number, focused specifically on the 4 games from the prior comparison
where `Curiosity` won and `Hypothesis` didn't (`ft09`, `lp85`, `r11l`,
`sp80`) -- if there's a fixable bottleneck, it should be visible in a
live trace on exactly these games, not averaged away across 21 others
where both agents already do fine or poorly for unrelated reasons.

**Live-tracing these 4 games surfaced something the earlier trace missed
entirely: on `ft09`, `lp85`, and `r11l`, the trace line only ever printed
a single candidate action (`a6`) -- these games' `available_actions` is
*only* ACTION6.** There's no action-choice decision happening on these
games at all; the *entire* outcome depends on where the click lands. This
reframes the earlier fixes (top-k patch reduction, Q-blend ablation) as
irrelevant to these specific three games -- both operate on the
action-level comparison, which never runs when there's only one action to
begin with.

Checked the actual click coordinates played across a full episode on each
of the three games (pulled directly from the recording files' `action_input.data`
fields): real diversity existed (73-84 distinct locations out of ~300
clicks per game) but one specific point dominated overwhelmingly in each
-- e.g. `(4, 4)` for `r11l`, which is the exact center of patch (0, 0),
the first patch in row-major order. That's the signature of
`patch_var.argmax()`'s tie-breaking behavior: whenever the per-patch
expert-disagreement map is flat or near-flat (no clear spatial signal
that turn -- plausibly the common case, not the exception), `argmax`
deterministically returns the same low index every time, so the agent
defaults to clicking the same likely-uninformative spot over and over
rather than exploring, on exactly the games where click placement is the
*entire* decision.

`Curiosity` (Stage 2) already solved this exact problem for itself, twice
over, in its own bug history: it uses temperature-weighted softmax
sampling over patches instead of a hard argmax, and a uniform-random
pixel *within* the chosen patch instead of always the exact center.
`Hypothesis`'s click-location logic had neither fix, despite being built
after `Curiosity` and able to reuse the same proven approach. Fixed by
adding `Hypothesis._sample_click` (mirrors `Curiosity._sample_click`
directly, `PATCH_SAMPLE_TEMPERATURE=0.1`, same value) and calling it in
place of the `argmax`-based selection.

**Verified before scaling up:** re-ran a single episode on the same 3
click-only games and checked click diversity directly -- distinct
locations jumped from ~75-85 out of ~300 clicks to **282-293 out of
~300** (essentially every click now genuinely new; the previous dominant
repeated point now appears only twice in an entire episode). Then ran 8
repeats on just the 4 divergent games (32 runs, faster than a full
25-game sweep) before committing to a full re-comparison: **3 total
levels / 2 distinct games (`ft09`, `r11l`)**, up from 0/0 on these same 4
games in the prior full-sweep round.

**Full matched 8x25-game re-comparison against a fresh `Curiosity` run,
both fixes in place:**

| agent | total levels | distinct games | avg actions to 1st level |
|---|---|---|---|
| **Hypothesis** | **14** | **5** (`cn04`, `ft09`, `lp85`, `m0r0`, `r11l`) | **158.7** |
| Curiosity | 11 | 4 (`cd82`, `m0r0`, `r11l`, `sp80`) | 180.4 |

**The Stage 5 milestone is met: `Hypothesis` beats `Curiosity` on total
levels, distinct games reached, *and* action-efficiency, all three at
once, in a fresh same-round matched comparison.** This is the first
result in this entire Stage 5 arc where all three numbers point the same
direction simultaneously, not a mixed or noise-plausible picture.

**Why this diagnostic approach succeeded where the broader sweeps
didn't:** every earlier fix in this arc (teacher-policy value head,
ACTION6 top-k reduction) was found either by generic live-tracing or by
addressing an already-documented data-bound limitation -- useful, but not
targeted at *specifically* where `Hypothesis` was losing to `Curiosity`.
Narrowing the trace to the exact games where the two agents diverged
surfaced a bug (argmax tie-breaking on click-only games) that a
whole-sweep trace or an aggregate metric would never isolate, since it
only shows up clearly on a subset of games where ACTION6 is the *only*
option -- diluted across 25 games (most of which have 6-7 available
actions), its effect on the pooled numbers was real but not obviously
attributable to any one mechanism. Worth remembering as a general
debugging strategy: when two things being compared diverge on a specific
subset, trace *that subset specifically*, not the full population.

## Stage 6 addendum -- held-out-game generalization: a real, robust, unfixed gap

**Finding: the world model has no measurable edge over "predict no change" on
any local game it wasn't trained on, and this is now confirmed across all
25 games via 5-fold cross-validation, not one arbitrary split.** This
started from a methodology audit prompted by the object-identity
checkpoint's real `0.00` Kaggle score (see below): every `changed-patches`
number and backtest result in this project's entire history, from Stage 1
onward, was validated via `torch.utils.data.random_split` over individual
*transitions* -- never a held-out *game*. Every local eval has measured
"how well does this generalize to a shuffled-out subset of the same 25
games," never "how well does this generalize to a game it has never seen"
-- which is exactly the axis Kaggle's real evaluation (~110 largely-novel
hidden games) stresses.

**Leave-N-games-out test** (branch `stage6-game-holdout`): held out 5
local games (`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`) entirely from
training, trained baseline and object-identity-recipe checkpoints on the
other 20. `changed-patches` improvement over identity collapsed to ~0% on
the 5 held-out games for *both* checkpoints, despite real improvement
(+2% to +8%) on the 20 trained games. The object-identity checkpoint's
diagnostic-B object-identity gap also collapsed/reversed on held-out
games (+1.18 trained -> -0.02 held-out), confirming its contrastive loss
was learning the *local* games' own color statistics, not a transferable
notion of object identity -- a real, verified contributor to (though not
sole proof of) that checkpoint's `0.00` real score.

**Root-cause elimination, four independent negative results:**
1. **Game-id embedding conditioning** (`stage6-gameid-ablation`): the
   obvious suspect -- any unseen game falls back to a fixed, undertrained
   embedding index. Ablating it entirely (`--ablate-game-id` on
   `jepa/train_moe_predictor.py`) did **not** close the held-out gap
   (stayed at ~0%/-0.2%). A large apparent trained-game win from this
   ablation (+64.9% vs +8.0%, single run each) **did not reproduce** under
   reseeding (`stage6-gameid-reseed`, n=3 each): with-game-id averaged
   +53.5% (std 4.6), no-game-id averaged +42.3% (std 14.7, noisier and
   lower) -- the original +8.0% with-game-id run was a ~10-std outlier,
   not a real effect. **Verdict: keep game-id conditioning on for
   production; this ablation is not an improvement.**
2. **Encoder change-sensitivity** (`stage6-encoder-holdout-diag`): directly
   measured feature-space delta at changed vs. unchanged patches,
   restricted to the 5 held-out games. The encoder is **not** the
   bottleneck -- its change-sensitivity ratio is actually *higher* on
   held-out games (80.2x/48.1x) than on trained games (6.5x/30.3x),
   consistent across all 5 games individually, with feature-norm and
   pixel-magnitude confounds both ruled out. The encoder correctly
   registers that something changed on a game it's never seen.
3. **Predictor residual-commitment** (same agent, follow-up diagnostic):
   the actual localized failure -- the predictor's residual-branch output
   magnitude (pre-skip-connection) collapses to **~0.000** on held-out
   games (vs. 0.235 baseline / 0.010 object-identity on trained games).
   It's coasting entirely on the identity skip-connection specifically
   when conditioned on an unfamiliar game, despite the encoder handing it
   a clear, correctly-detected change signal. Structurally the same
   failure shape as Stage 1's original "predictor learns to approximate
   identity" bug (see Stage 1 item 8 above), reappearing at the
   unseen-game boundary.
4. **Direct fix attempt** (`stage6-residual-commitment-fix`): added an
   explicit anti-collapse hinge loss on residual magnitude at changed
   patches, plus 20% game-id-dropout during training (simulating
   "conditioning is uninformative" on already-familiar games). **Did not
   help**: held-out changed-patches went to -0.1% (statistically the same
   as the unfixed ~0.0%), the residual-commitment ratio was still ~0.000
   on held-out games post-fix, and it cost real accuracy on trained games
   (+1.4% vs. the untouched +8.0% baseline). Likely explanation: game-id
   dropout still shows the model visually-*familiar* content with its id
   zeroed -- a different kind of "unfamiliar" than a truly novel game's
   unseen visual/mechanic content, so there's nothing to transfer from.

**Multi-fold cross-validation** (`stage6-multifold-cv`, run specifically
because relying on one arbitrary 5-game split risked confusing "this
split happened to be hard" with "the model can't generalize" --
partitioned all 25 local games into 5 disjoint folds of 5, covering every
game as a held-out target exactly once): the collapse holds up in
**every single fold**, for both game-id-conditioned and ablated variants
(fold means -0.30% and -0.40% respectively, folds ranging -1.8% to
+0.11%, no fold showing a real edge). This is not an artifact of one
unlucky split.

**Working conclusion:** four independent, well-targeted interventions
(game-id ablation, confirming the encoder is fine, an anti-collapse loss,
simulated training-time unfamiliarity) all converged on the same negative
result, now confirmed robust across 5-fold coverage of all 25 games. Per
this project's own repeated lesson (see the Stage 1 "CRITICAL" gotcha and
Stage 4's MoE gate history): when several different fixes all land on the
same negative outcome, that consistency points at a genuine data-bound
limit, not an unturned architectural knob. **The world model most likely
needs real training-game diversity to generalize zero-shot to a novel
game at all -- the same lesson Stage 4 learned adding MiniGrid pretraining
(more of the *same* ARC-3 data didn't help there either; genuinely
different mechanics did).** This also reframes every real Kaggle score
this project has gotten so far: the likely explanation isn't "checkpoint
X is worse than checkpoint Y," it's that *no* checkpoint trained purely
on these 25 ARC-3 games (± MiniGrid/Sokoban pretraining, which itself
still collapsed to near-uniform gating on ARC-3 specifically) currently
has real signal to offer on a genuinely novel game -- independent of
which one gets submitted. Next steps worth prioritizing over further
loss-shaping on the current corpus: substantially more diverse
pretraining sources (beyond MiniGrid/Sokoban, which were already tried
and only partially helped -- see Stage 4), or a fundamentally different
approach to novel-game adaptation (e.g. test-time few-shot adaptation
from a new game's opening RESET/probe frames, rather than expecting a
frozen zero-shot forward pass to work at all).

**Follow-up: does Stage 5's InfoGain exploration signal collapse too?**
Given InfoGain (`jepa/hypothesis_bundle.py`) is variance across the 8 MoE
experts' *raw, ungated* predictions (`MoEPredictor.predict_all_experts`),
and the *gated* prediction was just shown to collapse to identity on
held-out games, the natural worry: does the Hypothesis agent's whole
exploration signal go dead on exactly the games where it matters most?
Directly measured (`scripts/diagnose_infogain_holdout.py`, reusing
`stage6-game-holdout`'s baseline checkpoint, no retraining needed) mean
InfoGain across 500 held-out-game states x 4 candidate actions vs. the
same on 500 trained-game states: **held-out 3.24e-2 vs. trained 3.24e-2 --
ratio 0.999, no collapse.** The individual experts still genuinely
disagree with each other on unseen games, just as much as on trained
ones. This seemingly contradicts the residual-commitment finding above
(which collapses to ~0.000) but doesn't, once the math is separated: that
number measures the *gated* residual (`(expert_outputs * gate_weights)
.sum(dim=1)`), a weighted blend across experts, while InfoGain measures
*raw per-expert* disagreement before any blending. The most likely
reconciliation: the gate stays close to uniform (Stage 4's own "gate
specialization is a minority behavior" finding), and averaging several
experts whose individual residuals are real and differentiated but
pointing in different directions nets out close to zero -- collapsed
*prediction accuracy*, but not collapsed *underlying disagreement*.
Practical upshot: Stage 5's exploration mechanism (InfoGain-driven action
selection) may retain real signal on genuinely novel Kaggle games even
where the world model's raw *predictive accuracy* doesn't -- an
encouraging, narrower finding than "the whole hypothesis bundle is
useless there," though not yet cross-validated across multiple folds the
way the main finding was, and not yet tested at the *agent* level (does
this signal actually translate into better real exploration on unseen
games, vs. just existing in the numbers).

Separately flagged, not yet tested: whether the entropy-driven adaptive
`beta` (Bayesian confidence -> explore/exploit blend) earns its
complexity over a simple fixed constant. The existing ablation
(`FORCE_BETA`, see Stage 5 follow-up 2) only tested the two *extremes*
(beta=0, beta=1) against the real entropy-driven version -- it's never
been compared against, say, a fixed `beta=0.37` (roughly the observed
mean under `decay=0.8`). If a constant performs just as well, the whole
per-step Bayesian confidence-tracking machinery isn't earning its keep
over a much simpler design.

**Continuous game-embedding investigation (`stage6-context-embedding`):
three more conditioning mechanisms tried, all negative -- 7 independent
interventions now agree.** Given the categorical `game_id -> embedding`
lookup was suspected as the root cause, tested whether replacing it with
a *continuous, observation-derived* representation would close the gap
(the user's own instinct on where to go next):
1. **Stage 3's existing recurrent predictor** (`GRUCell` hidden state
   accumulated from real in-episode transitions, not a category lookup)
   -- extended with `--exclude-games`, evaluated on folds 1-2 with a real
   accumulated hidden state (not zeroed): **-1.7%** and **+0.5%**, both in
   the same near-zero noise band as the MoE predictor. Whether the hidden
   state was "warmed up" or fresh made no difference.
2. **Single-frame content conditioning** (`FrameContextEncoder`, a
   `context_mode="frame"` option on `MoEPredictor` deriving the
   conditioning vector from the current frame's own pooled features
   instead of a game_id lookup): folds 1-2, **-0.1%/-0.0%** vs. the
   categorical baseline's **-0.5%/-0.0%** -- statistically indistinguishable.
3. **Multi-transition episode context** (`EpisodeContextEncoder`,
   Deep-Sets-style pooling over 8 other same-episode transitions --
   architecturally the closest to a real meta-learning/context-inference
   approach): fold 1, **-0.6%**, every one of the 5 held-out games
   negative individually.

**Combined with the 4 negative results earlier in this section (game-id
ablation, encoder audit, anti-collapse residual loss, simulated
unfamiliarity), that's 7 independently-designed interventions --
including three genuinely different continuous-conditioning mechanisms,
not just variations on one idea -- all converging on the same null
result.** This is stronger evidence than before that the ceiling is
data-bound, not a fixable conditioning architecture: no way of
*representing* which game this is, categorical or continuous, helps when
the training corpus itself only contains 20-25 games' worth of mechanics
to learn from. The natural reading: this reframes "replace the game
embedding" as very unlikely to be the fix on its own, regardless of how
cleverly the embedding is built -- the two directions worth prioritizing
next are (a) genuinely more diverse pretraining sources (the one lever
that *did* work before, for Stage 4's MoE gate specialization via
MiniGrid), or (b) test-time adaptation -- actually updating on a hidden
game's own observed opening frames *during play*, rather than expecting
any frozen zero-shot forward pass, however conditioned, to already
generalize. All three new mechanisms and their eval infrastructure are
implemented and committed on `stage6-context-embedding` (not merged) if a
future session wants to revisit with substantially more diverse training
data behind them -- the code is real and reusable even though this
round's conclusion was negative. Full numbers and per-game breakdown in
`experiments/stage6_continuous_game_embedding.md`.

**First data-diversity attempt against this specific gap: MinAtar
pretraining, also negative (`stage6-diverse-pretraining`).** Given all 7
interventions above targeted conditioning/architecture, not data volume,
tried the other lever that actually worked once before (Stage 4's
MiniGrid win): added MinAtar (`jepa/data/minatar_data.py`, a clean-room,
no-ROM reimplementation of 5 classic Atari-style games -- `breakout`,
`asterix`, `freeway`, `seaquest`, `space_invaders` -- as small grid-based
multi-channel environments, much closer to this project's own
representation than real Atari's raw RGB frames would be; all 5 share one
`game_id="minatar"`, mirroring MiniGrid's own reasoning) to the
pretraining curriculum. Controlled ablation on fold 1's exact recipe
(byte-identical MiniGrid+ARC corpora, differing only in MinAtar's
presence): **worse on both metrics** -- standard trained-games
changed-patches +2.1% vs. baseline's +4.0%, and held-out fold-1
generalization **-1.4%** vs. baseline's -0.1% (`r11l` individually dropped
to -10.5% with MinAtar added). Procgen (the other originally-planned
source, still untested) was skipped this round since the task was
explicitly gated on MinAtar showing promise first.

**Open question, not yet investigated:** MinAtar's 5 sub-games (paddle
physics, maze-chasing, lane-crossing, submarine survival, shooting) don't
share nearly as much underlying structure with each other as MiniGrid's
21 navigation-themed environments did -- pooling them under one shared
`game_id` may be forcing the model to fit several mutually-inconsistent
action->effect mappings at once, exactly the confound Stage 1 originally
worried about (and found *not* to be the dominant issue) for the 25 ARC-3
games themselves. Untested whether per-sub-game MinAtar ids, or a
genuinely closer-shaped source (Procgen's more puzzle/maze-like
environments, e.g. `maze`/`heist`, rather than reflex-heavy arcade games),
would fare differently -- this round only tested one specific choice
(shared id, MinAtar specifically), not the full space of "diverse data"
options. Combined with Sokoban's own earlier negative result (Stage 4
item 8, a different diagnosed cause -- deadlock-polluted data), the
working pattern so far is that *borrowed game-engine* diversity doesn't
reliably transfer to ARC-3's puzzle-logic mechanics merely by being
"another grid game" -- shape/genre match may matter as much as diversity
itself.

**First genuinely positive (if modest) result of the whole investigation:
test-time adaptation (`stage6-test-time-adaptation`).** Every fix tried
above -- 8 interventions, all negative -- was still a *frozen, zero-shot
forward pass* at evaluation time. Tested a mechanistically different
question: if the model takes a few real gradient steps using a hidden
game's own observed transitions, *as they're seen during play*, does its
prediction quality on that game actually improve? Used the
`stage6-game-holdout` fold-1 baseline checkpoint, streamed a held-out
game's recorded transitions in order, and every K transitions ran a few
AdamW steps on a deliberately restricted, ANIL-style subset of the MoE
predictor (each expert's last `Conv2d` + the gate's last `Linear`, ~33.8K
params -- encoder and embeddings frozen, chosen specifically to avoid
retesting the already-failed continuous-embedding mechanisms under a
different name).

**Result: real, monotonic improvement in 4/5 held-out games as adaptation
data accumulates** (K=10, 3 steps, lr=5e-5; changed-patches at 0 -> 200
adaptation transitions): `r11l` -1.2% -> +0.5%, `ka59` -1.4% -> +0.2%,
`bp35` +0.0% -> +0.3%, `m0r0` -0.3% -> -0.1%, `tr87` flat. A step-count
sweep (1/3/5 steps) on `r11l` showed a clean monotonic dial -- more
adaptation steps produce more held-out improvement *and* proportionally
more trained-game interference -- exactly the pattern you'd expect from
real learning, not noise (noise wouldn't dial linearly with an
intervention's magnitude in both directions at once).

**Real, honest limits, not swept under the rug:** the magnitude is small
(roughly 1-2 percentage points over 200 transitions / ~60 gradient
steps), nowhere near production's own +8% to +30% improvement on trained
games -- this is not a standalone fix. It also isn't free: re-evaluating
the same adapted checkpoints on an 8-game trained-games probe showed real
(non-catastrophic) interference, pooled improvement dropping from +9.8%
to +6.6%-10.0% depending on adaptation intensity, scaling with how much
adaptation was applied. Practical integration would need per-game resets
(don't carry adapted weights across different games/episodes) and a
larger adaptation budget than a single realistic Kaggle episode likely
provides at `MAX_ACTIONS=300`.

**Working read:** unlike every conditioning/architecture fix and the
first data-diversity attempt, this is a real, mechanistically distinct
lever -- the model demonstrably *can* extract signal from a novel game's
own data given real gradient access, which nothing else today could
show. At its current small magnitude it's best framed as a complementary
tool to layer on top of better pretraining data (still being tested in
parallel), not a replacement for it. Full methodology, per-game tables,
and integration notes in `experiments/stage6_test_time_adaptation.md`.

**Two more parallel follow-ups: a real methodological fix that still
doesn't close the gap, and a genre-matched retry undermined by its own
confound.**

*MinAtar retry with per-sub-game ids* (`stage6-minatar-pergame-id`):
tested whether pooling MinAtar's 5 mechanically-dissimilar games under
one shared `game_id` (mirroring MiniGrid's successful pattern) was the
cause of MinAtar's original failure. Splitting into 5 separate ids
(`minatar_breakout`, etc., no other changes) swung standard trained-games
changed-patches from **+2.1% to +55.8%** -- confirming the pooling
confound was real and large, a genuinely useful lesson for any future
synthetic-source integration (default to per-sub-environment ids unless
there's a MiniGrid-style shared-semantics reason to pool). **But held-out
fold-1 generalization stayed at ~0.0%, unchanged** -- structurally
unsurprising in hindsight, since a never-seen ARC-3 game still falls back
to the same untrained embedding index regardless of how MinAtar's own
internal games are identified. This fix targeted the wrong axis for the
held-out-games question specifically.

*Procgen* (`stage6-procgen-pretraining`, `maze`+`heist` -- chosen
specifically as a better genre match to ARC-3's puzzle-logic character
than MinAtar's reflex-arcade games, with a real RGB->16-color k-means
quantization layer since Procgen renders raw pixels): standard-corpus
changed-patches **regressed** (+48.2% baseline -> -2.0%), and held-out
fold-1 stayed near-collapsed (-1.5% -> -0.2%). **Diagnosed the actual
mechanism, not just the symptom, before accepting the result at face
value**: encoder batch-level feature variance is fine (Procgen's is
*higher* than baseline's), but frame-to-frame temporal-change
sensitivity collapses ~150x during pretrain and -- unlike the MiniGrid
baseline, which recovers within 1 finetune epoch -- never recovers across
60 ARC-3 finetune epochs. Traced to doubling the pretrain-phase corpus
size (67,200 -> 134,400 transitions) at unchanged epoch counts -- a
curriculum-balance confound, not clean evidence against genre-matched
data itself. **This result is honestly inconclusive on the genre-matching
hypothesis specifically** -- it tests "more pretrain data without
adjusting the schedule" more than it tests "does puzzle-genre-matched
data help," and a properly rebalanced rerun (matched pretrain epochs to
corpus size, or fewer Procgen episodes) would be needed before drawing a
clean conclusion either way.

**Running tally: 9 independent interventions against the held-out-games
gap today, 9 failures to close it** (7 conditioning/architecture fixes,
MinAtar shared-id, MinAtar per-game-id, Procgen) **plus 1 genuinely
positive but modest result (test-time adaptation) using a mechanistically
different approach (real gradient updates, not a frozen forward pass).**
Recommended next steps, in order: (1) a properly curriculum-balanced
Procgen rerun, since the current negative result doesn't cleanly test
what it was meant to; (2) build out test-time adaptation further (larger
adaptation budgets, per-game weight resets) since it's the only lever
that's shown any real, positive, dialable signal all day; (3) treat
"more of the same kind of external grid-game data" with real skepticism
going forward given the accumulating pattern, and consider whether
ARC-3-*shaped* synthetic puzzle generation (rather than borrowed game
engines from other genres) might be a better-matched data source than
anything tried so far.

**Curriculum-balanced Procgen rerun (`stage6-procgen-rebalanced`): the
imbalance hypothesis didn't hold up either.** Subsampled MiniGrid and
Procgen to ~33,600 transitions each (67,200 total, matching the original
MiniGrid-only baseline's corpus size, pretrain epochs unchanged at 20) to
isolate diversity from volume. Standard-corpus changed-patches was still
a regression (**+67.6% baseline -> -0.3%**, an ~811x absolute-MSE
collapse -- worse than the original unbalanced attempt's ~150x), and
held-out fold-1 stayed flat (**-0.2% -> -0.1%**, no improvement).
Rebalancing *did* give a measurably better encoder starting point at
finetune epoch 1 (val_identity_mse 5.7x better than the original run),
confirming the diagnosed pretrain/epoch imbalance was real -- but the
collapse reasserted itself and progressed across the entire 60-epoch
finetune phase regardless, ending at essentially the same terminal
magnitude as before. **Working read: the curriculum imbalance was a real
contributing factor but not the dominant cause.** The more likely
remaining driver, flagged for a future session rather than claimed
proven: Procgen's RGB->16-color k-means quantization is lossy/noisy in a
way none of this project's other pretraining sources are (MiniGrid,
Sokoban, MinAtar are all natively categorical/discrete, no quantization
step needed) -- the translation layer itself, not the game genre, may be
the actual problem.

**Running tally, end of today's diverse-pretraining-data investigation:
10 independent interventions against the held-out-games gap, 9 failures,
1 genuine (if modest) success.** The 9 failures span conditioning fixes,
architecture changes, and now three different data-diversity attempts
(MinAtar shared-id, MinAtar per-game-id, Procgen original, Procgen
rebalanced -- 4 data attempts, if counted separately, all negative). The
1 success, test-time adaptation, remains the most promising lever
identified today and the natural next thing to build out further.

**Test-time adaptation built out into the real agent (`stage6-test-time-
adaptation-agent`): the prediction-quality win did not translate into a
real gameplay win.** Widened the earlier diagnostic's sweep across all 5
held-out games (K in {5..200}, steps in {1..12}, LR in {1e-5..4e-4}) and
confirmed the monotonic tradeoff dial holds up broadly, not just on
`r11l` -- picked a deliberately conservative operating point (K=5,
STEPS=8, LR=5e-5: +0.84% mean held-out changed-patches, -1.6pp
trained-game interference). Built `jepa/test_time_adapter.py`
(`TestTimeAdapter`): snapshots the ~33.8K-param ANIL-style subset,
restorable via `reset()`, persists across RESETs of the same game (not
each RESET -- mirrors `TransitionGraph`'s own persistence choice) and
only resets on a genuinely new game (already enforced for free -- one
fresh `Hypothesis` instance per `game_id`). Wired into
`hypothesis_agent.py` behind `HYPOTHESIS_TEST_TIME_ADAPT=1` (default
off); per-turn latency measured directly at ~17ms, negligible next to a
real network round-trip.

**Agent-level backtest -- the test that actually matters, run for the
first time today:** n=8 on the 5 held-out games, n=4 on a 25-game trained
sweep. **No detectable benefit on held-out games** (TTA on: 0.375 mean
levels/3 total; off: 0.500/4 total, both only ever solving `r11l`; the
raw score gap was driven by two outlier fast completions, not a
systematic effect) **and no detectable regression on trained games**
(ON/OFF within noise of each other). A real, independently-verified
representation-level improvement did not show up in actual play at this
sample size -- the same "world model got measurably better, agent win
count didn't move" pattern this project has already hit twice before
(Stage 2's post-bugfix Curiosity re-test, Stage 5's teacher-policy value
head) -- not a contradiction, just a reminder that small-sample
agent-level metrics need much more power to detect a real but modest
effect than a direct representation-level measurement does. **Verdict:
not a submission candidate on its own merits yet, but safe to ship
disabled-by-default** (zero regression, negligible latency, unit-tested
reset logic) -- worth revisiting with a larger backtest sample or a
larger adaptation budget if a future session wants a more decisive
answer, rather than concluding the mechanism doesn't work at all.

**The biggest single test of the day: scaling model capacity and
pretraining diversity together (`stage6-scaled-world-model`) -- a
real, well-reasoned hypothesis, tested properly, still negative.** Every
data-diversity attempt above added one modest source (33-67k
transitions) at unchanged model capacity. A genuinely different, larger
test: build a real roster of dozens of distinct game mechanics --
6 OpenSpiel board/strategy games (`connect_four`, `tic_tac_toe`,
`othello`, `checkers`, `pig`, `mancala` -- board-game moves represented
as (x,y) click-style transitions, reusing the same mechanism ARC-3's own
ACTION6 already has xy-conditioning for; `backgammon` was evaluated and
dropped, its 1,352-action combined-sub-move space wasn't practical to
decompose in the time available) plus hand-rolled Snake and Pong for
real-time physics -- each game given its own distinct `game_id`
(applying the per-sub-game-id lesson from the MinAtar retry above from
the start, not re-discovering it), totaling **~358k synthetic pretrain
transitions, a genuine order-of-magnitude step up** from any single prior
source. Paired this with re-testing model width (1x vs. 2x,
`--width-mult`) -- the actual untested combination, since the earlier
`stage6-capacity-sweep` ablation tested capacity alone on the small
original data and found no benefit there.

**Width=1.0 (diversity alone, no extra capacity): a clean, well-powered
negative.** Held-out fold-1 changed-patches: **+0.10%**, statistically
indistinguishable from the established baseline (+0.01%) and well inside
the existing 5-fold noise band. No Procgen-style curriculum collapse this
time (pretrain epochs were deliberately sized to hold total
samples-seen roughly constant relative to the proven recipe, learning
directly from that earlier mistake) -- a trustworthy null result, not a
confounded one. ~5.3x more pretraining data spanning genuinely different
mechanics, by itself, does not move the held-out-games number.

**Width=2.0 (diversity + capacity together): inconsistent, and where it
moved, it moved the wrong way.** Fold 1: **-88.29%** -- a dramatic
regression, every one of the 5 held-out games individually worse, paired
with the *best* trained-games result of the whole day (+69.11%) --
textbook capacity-enabled overfitting, not generalization. Validated on
fold 2 before treating fold 1 as conclusive (this project's own standing
lesson about not trusting one fold): fold 2 came back **-0.03%**, near
parity with fold 2's own baseline (+0.11%) and the *weakest* trained-fit
of the three runs (+5.26%) -- fold 1's collapse did not replicate. Both
folds agree on the answer that actually matters, though: neither shows
capacity turning this diverse data into better held-out generalization,
and the inconsistency between folds (severe regression vs. near-neutral)
is itself a real finding -- capacity scaling on top of this data regime
is unpredictable, not a reliable lever, with a real downside risk and no
observed upside.

**Working read:** the reasoning behind this hypothesis was sound and
targeted the right thing (capacity and data need to scale together;
today's other diversity attempts genuinely were "light" pretraining at
unchanged capacity) -- but at the data scale actually achievable on this
project's hardware (~358k transitions, nowhere near real foundation-model
scale), adding capacity seems to widen the model's ability to fit its
*whole* training manifold (including the ARC-finetune games) more
precisely, which sometimes bleeds into worse generalization rather than
better -- the classic capacity-without-proportionally-more-data risk,
not evidence the underlying idea is wrong at a truly large scale, just
evidence it doesn't close the gap at the scale this project can actually
reach. **This is the 11th independent intervention against the
held-out-games gap today, and the 10th failure** -- test-time adaptation
remains the only lever that showed any real, positive, dialable signal.
Full roster reasoning, per-fold tables, and methodology in
`experiments/stage6_scaled_world_model.md` on branch
`stage6-scaled-world-model` (not merged to master).

**Checked whether finetuning an existing open-source world model beats
training from scratch: no viable candidate exists.** A real, current web
search (not relying on stale training-data knowledge) checked Genie/
Genie 2/3 (proprietary, no public weights), DIAMOND and IRIS (public
weights, but each checkpoint is a *single-Atari-game specialist* trained
on ~100k frames -- fine-tuning one buys a differently-pretrained
single-game model, not cross-game generalization, which doesn't address
the actual problem), WHAM/Microsoft Muse (single-game, non-commercial
research license), Oasis (Minecraft-only, MIT, but inference-only --
no training/fine-tune code was ever released), and V-JEPA2/Matrix-Game
(genuinely large-scale and diverse, but either wrong-domain -- realistic
video vs. this project's flat 16-color grids -- or requires 24GB+ VRAM
neither the local RTX 2070 nor Kaggle's free-tier GPUs have). Notably,
this reconfirms a decision `architecture.md`'s own "Discarded-for-
Complexity Ideas" footnote already made before this session started --
V-JEPA-family and Atari-benchmark models were considered and shelved for
the same reasons found again today. The from-scratch, diverse-synthetic-
pretraining approach this project is already running remains the more
defensible path; no shortcut via an existing checkpoint is available.

**MinAtar retry with per-sub-game ids and Procgen's own scale-up attempts
weren't the last word on data diversity -- a genuine roster expansion via
OpenSpiel, still negative (`stage6-expanded-roster`).** OpenSpiel actually
has **123 registered games**, not the 6 used in `stage6-scaled-world-model`
-- programmatically categorized all 123 by dynamics/information type
(not hand-inspected) before deciding anything: 42 are `SEQUENTIAL` +
`PERFECT_INFORMATION` (the eligible pool, matching how ARC-3's own turns
work), 20 are simultaneous-move (excluded -- no single well-defined
"current player's action" to log), 46 are imperfect-information (excluded
-- can't render an honest fully-observable grid for a state a player
doesn't fully see), 4 are mean-field/population-level (excluded, wrong
paradigm entirely), 11 fail to load without extra wrapper params
(excluded as meta-games, not directly playable). **Went from 6 to 26
OpenSpiel games** -- a genuine "dozens more" expansion, not a token
increase -- for **~2.14M total pretrain transitions** (~6x the prior
attempt's 358k), sized via directly-measured generation throughput
(4.59M transitions in ~7.2 minutes standalone) rather than a guess, with
pretrain epochs deliberately kept to 1 to hold total samples-seen in the
established curriculum-balance band (learning directly from the Procgen
imbalance mistake, not re-discovering it).

**Real infrastructure lesson found along the way**: Windows' `DataLoader`
worker-spawn (`num_workers>0`) pickles the *entire* dataset object to
each spawned subprocess -- fine at the ~55k-transition scale this
project's `num_workers=4` default was tuned for, but catastrophically
slow past roughly the hundreds-of-thousands range (a worker burned
1,200-1,900+ CPU-seconds without completing one batch on the 2.14M
corpus). Fixed via the existing `JEPA_NUM_WORKERS=0` override (originally
added for an unrelated memory gotcha) -- worth defaulting to for any
future corpus at this scale, not reaching for only after something else
fails first.

**Fold-1, width=1.0 result (real, clean run, ~78 min wall-clock):
still negative, and if anything very slightly worse than the smaller
attempt.** Held-out changed-patches: **-1.22% overall, every one of the
5 held-out games individually negative** (not a mixed picture -- `r11l`'s
-13.04% is the small-absolute-denominator artifact this doc already
warns about, checked directly rather than assumed). This sits at or
slightly below the established 5-fold no-diversity baseline band
(-0.30% +/- 0.66%), and is *worse* than the prior 6-game/358k attempt's
+0.10% -- **~4.3x more OpenSpiel games and ~6x more total pretrain
transitions did not move this number in a positive direction.**
Trained-games sanity check is healthy (+18.34%), confirming real,
non-degenerate learning happened from the much larger corpus -- it
simply doesn't transfer to genuinely unseen ARC games, the same pattern
every one of the 10 prior interventions already established. **This is
the 11th consistent negative result against the held-out-games gap.**

**Width=2.0 retest on this same expanded corpus -- the direct test of
whether proportional capacity fares differently on genuinely more data:
still no benefit, but a real, useful secondary finding.** Fold 1:
**+0.02%** (essentially exact parity, not a real effect either way).
Fold 2 (validation): **-0.05%**, confirming fold 1's story rather than
contradicting it. **12th-13th consistent negative results.** The
secondary finding: `stage6-scaled-world-model`'s severe fold-1
capacity-instability (-88.29%, textbook capacity-enabled overfitting on
the smaller 358k corpus) **did not replicate at this larger data scale,
in either fold** -- width=2.0 is safe here (no regression risk), it
simply still isn't beneficial. That's a genuine, useful data point on
its own: more data does appear to remove the *downside* risk of scaling
capacity, even though it hasn't yet produced an *upside*. Real,
reusable infrastructure outlasts this negative result regardless:
`jepa/data/openspiel_data.py` now has three generic, reusable handler
families (cell-index placement, destination-click parsing, direct-
action-id) instead of bespoke per-game code, so adding further OpenSpiel
games in a future session is mostly config work, not new engineering.
Full per-game exclusion reasoning and per-fold tables in
`experiments/stage6_expanded_roster.md` on branch `stage6-expanded-roster`
(not merged to master).

**Running tally after this whole diverse-data investigation: 13
independent interventions against the held-out-games gap, 12 failures,
1 modest success (test-time adaptation).** Across conditioning fixes,
architecture changes, and now five separate data-diversity attempts at
increasing scale (MinAtar x2, Procgen x2, a 358k-transition 29-game mix,
and a 2.14M-transition 26-OpenSpiel-game roster), nothing has closed the
gap. The pattern is now consistent enough across enough independently-
designed interventions that it's reasonable to treat this as this
project's hardware/data ceiling for zero-shot generalization via
pretraining alone, not a specific unfound bug -- test-time adaptation
(real gradient updates during play, not a frozen forward pass) remains
the only mechanism that has shown any real, positive, dialable signal
across this entire investigation.

**Color-permutation augmentation (`stage6-augmentation`): the 14th
intervention, and the 13th failure -- with a real cost this time, not
just a null result.** Every prior attempt targeted the predictor's
conditioning or the training data's diversity; this targeted something
different -- whether the model was overfitting to the *specific* color
statistics of the 25 local games (directly evidenced by the
object-identity checkpoint's contrastive-loss collapse earlier in this
section). Added a `--color-augment` flag (`jepa/data/trajectories.py`)
applying a fresh random permutation of all 16 ARC colors per training
example, identically to `frame_t` and `frame_t1` so the causal
action-effect relationship stays truthful -- color 0 deliberately
included in the permutation (no cross-game convention establishes it as
background in ARC-3 specifically, unlike classic ARC puzzles).

**Result: no improvement on held-out games (+0.01% baseline -> -0.19%,
both inside the established 5-fold noise band) and a real regression on
trained games (+7.97% -> -1.76%),** corroborated by the training run's
own validation curve eroding steadily from +2.4% at epoch 1 to -0.5% at
epoch 60 -- not a fluke. This rules out "stop the model from relying on
specific color ids" as sufficient on its own, and the trained-game
regression is itself informative: color identity was a real, exploitable
shortcut the model was using to do well on familiar games, and removing
access to it didn't buy back any transfer to unfamiliar ones -- just cost
accuracy on the games it already handled.

**Spatial (rotation/flip) augmentation was investigated and deliberately
not attempted**, a good example of catching a correctness risk before it
could produce a misleading result: `rules.md`'s own Action Space section
states that simple-action semantics "vary per game and must be
discovered through exploration -- not documented in advance." Since
there's no reliable way to know whether a given game's action ids carry
a fixed spatial meaning (e.g. "move up") that a rotation/flip would need
to relabel to keep training examples truthful, implementing it risked
planting subtly incorrect training signal rather than genuine
augmentation -- correctly judged not worth the risk without a way to
verify it first.

**Novelty-aware beta override (`stage6-novelty-aware-beta`): the most
encouraging agent-level result of the whole investigation, though still
n=8.** Every fix above targeted the world model itself; this instead
changes the *agent's* strategy to work around a known weakness rather
than fix it. The agent already has a direct, deterministic signal for
"this game is genuinely unfamiliar" -- `self.game_id not in game_vocab`,
the same lookup already used for the fallback embedding index -- which is
more reliable than the existing Bayesian confidence-entropy `beta`
signal, since that confidence is built from *observed* per-expert error
and could look artificially "confident" on a collapsed, unfamiliar-game
prediction (consistently predicting no-change looks like agreement even
when it's not informative). Added `NOVELTY_BETA_CAP`
(`hypothesis_agent.py`): when the current game is outside the trained
vocabulary, `beta = min(beta, NOVELTY_BETA_CAP)`, biasing the Q-blend
toward InfoGain (the signal already shown not to collapse on held-out
games) instead of the value head. Familiar-game behavior is untouched --
the existing adaptive blend was already validated as the right design
there (Stage 5 follow-up 2).

**Cap value (0.15) chosen from real evidence**: replaying held-out-game
episodes through the confidence-tracking logic showed beta sitting in a
narrow 0.12-0.40 band there (mean 0.245, 99.7% of mass above 0.15) --
0.15 gives a real, consistent nudge toward InfoGain rather than a
near-no-op.

**Agent-level backtest (n=8, `MAX_ACTIONS=300`, the 5 held-out games):
cap ON leads on every metric.** Mean score 0.0603 vs. 0.0113, mean
levels 0.50 vs. 0.38, total levels 4 vs. 3 (`r11l` solved 4/8 vs. 3/8) --
unlike test-time adaptation's more mixed agent-level picture, this one
favors the change across the board, including the outlier-resistant
levels-completed comparison. A trained-games sanity check initially
looked alarming (mean score 0.0047 vs. 0.4762) but was traced directly to
one game's identical fast-solve path landing 3/5 times under the old
default vs. 0/5 under the cap -- unseeded-RNG variance, not a real
effect, confirmed by the cap being structurally inert on all 6 trained
games checked (each present in the training vocabulary, so the override
never fires there).

**Honest caveat, same standard as every other result today:** n=8 on a
sparse binary metric is real, encouraging evidence -- not proof. Worth a
larger backtest before treating this as validated, but it's the first
result all day that's positive on every agent-level metric at once,
rather than a representation-level win that didn't clearly show up in
play. Full write-up in `experiments/stage6_novelty_aware_beta.md` on
branch `stage6-novelty-aware-beta` (not merged to master).

**Correction (`stage6-novelty-beta-largescale`): the n=8 result did not
replicate at n=30 -- it was noise, and the "most encouraging result"
framing above is retracted.** Rerunning the identical comparison at n=30
per condition (60 total runs): levels-completed, the more robust metric,
came back **exactly tied** (13 vs. 13, an identical 43.3% solve rate on
`r11l`, the only game either version ever solved across all 60 runs).
The mean-score gap didn't just shrink, it **reversed direction** (n=8
favored cap ON 0.060 vs. 0.011; n=30 favors cap OFF 0.094 vs. 0.029),
fully explained by two high-scoring outlier runs on the cap-OFF side --
exactly the kind of small-sample artifact this project has hit before
(see the earlier `+64.9%` game-id-ablation result that also evaporated
on reseeding). Mann-Whitney U tests found no significant difference in
per-run score (p=0.86) or solve efficiency (p=0.59). This is the third
time this session a real, mechanistically well-motivated component-level
idea (after the teacher-policy value head and test-time adaptation) has
failed to produce a statistically detectable agent-level effect at
practical sample sizes on these 5 held-out games -- not necessarily
because the mechanism is wrong, but because this evaluation protocol
doesn't have the power to tell a real small effect apart from noise.
**Recommendation: keep the cap enabled by default** (it's still
structurally inert on trained games -- zero regression risk -- and there
is still no evidence it hurts), **but do not treat it as a validated
improvement.** This retraction is consistent with, not contradicted by,
the real Kaggle submission below (`0.09`, itself unremarkable against the
established noise floor) -- two independent tests, the larger local
backtest and the one real submission, now agree there's no detectable
benefit, where a single small local sample had briefly suggested one.

**MAX_ACTIONS=900 + test-time adaptation, combined (`stage6-budget-tta-
combo`): a structurally interesting n=8 signal, explicitly not trusted
yet given the lesson just above.** Two levers that each showed modest
individual promise -- a longer action budget (helps every checkpoint
tested) and test-time adaptation (real but small representation-level
gain) -- had never been tested together. Four-condition backtest (n=8,
5 held-out games, `stage6-game-holdout` fold-1 checkpoint): baseline
0.500 mean levels (4/8 total), budget-900-alone 0.750 (6/8), TTA-alone
0.375 (3/8), **combo 1.000 mean levels -- every one of 8 repeats
completed at least one level, the first zero-zero-completion-run
condition in this project's entire Stage 6 backtest history.** Naive
addition of the two individual effects predicts ~5/8; observed is 8/8,
suggesting real compounding rather than two independent small effects.
**Important limits, not glossed over**: mean *score* (as opposed to
levels) is still outlier-driven and not trusted on its own; breadth did
not improve -- all four conditions, including the combo, only ever
solved `r11l`, so this is a reliability gain on a game already
partially solvable, not new generalization to a harder game. And given
the novelty-aware beta override *just* showed an equally clean-looking
n=8 win across every metric that completely evaporated at n=30, **this
result is explicitly flagged as preliminary, not validated** -- a
25-30-repeat confirmatory backtest is the recommended next step before
it influences any submission decision, not an immediate green light.

**A Reptile meta-learning objective (`stage6-meta-learning`): the
biggest build attempted today, and it produced this investigation's
first *dosage-confirmed* representation-level improvement -- amplifying
test-time adaptation itself, not just matching it -- though still no
detected agent-level effect.** Every fix so far started from a normally-
trained checkpoint and hoped it happened to be adaptable. This instead
built a first-order (Reptile) meta-training objective explicitly
optimizing the predictor for *post-adaptation* performance: the same
~33.8K-param ANIL-style head `TestTimeAdapter` uses at real eval time,
periodically nudged during training via real inner-loop adaptation steps
on sampled training-pool games (`jepa/train_meta_predictor.py`).

**A real bug found and fixed first**: the textbook ANIL design (freeze
the head from ordinary training, update it only via the Reptile nudge)
caused catastrophic representation collapse -- with the head unable to
produce meaningful residuals during normal training, the encoder learned
to make transitions look trivially identical rather than learn real
dynamics. Fixed by keeping the head in ordinary joint training *and*
layering the Reptile nudge on top, verified via a smoke test before any
full run.

**Standard-dose Reptile was a clean negative** -- post-adaptation
held-out improvement (+0.57%/+0.42%) was *worse* than the plain
baseline checkpoint's own post-adaptation improvement (+0.78%/+0.66%,
`stage6-test-time-adaptation-agent`'s own number). **High-dose Reptile
(3x more updates/epoch, no epsilon annealing) reversed this: +0.98%/
+1.28% vs. the same +0.78%/+0.66% baseline** -- a real, directionally
consistent gain concentrated on 3 of the 5 held-out games, confirming
the standard-dose result was a dosing artifact, not a ceiling on the
approach itself. This is the first result in this whole investigation
that measurably improves on test-time adaptation's own already-real
effect, not just matches or fails to match it.

**Preliminary agent-level backtest (n=8, since the high-dose result
looked promising): levels-completed came back exactly tied with the
already-published baseline+TTA-on numbers** (3 total, 0.375 mean, all on
`r11l`) -- the same "component measurably improved, small real-play
sample couldn't detect it" pattern this project has now hit three times
today (teacher-policy value head, test-time adaptation itself, and now
this). Full write-up in `experiments/stage6_meta_learning.md` on branch
`stage6-meta-learning` (not merged to master) -- worth a larger backtest
of the high-dose checkpoint specifically before drawing an agent-level
conclusion either way.

## GraphExplorerAgent: a training-free, ported reproduction of external work (separate from JEPA)

**Decision, stated upfront:** given the difficulty closing Stage 6's held-
out-game generalization gap (13 independent interventions, 12 failures --
see above) and given a directly relevant, independently-scored external
result exists on this exact benchmark, this project paused JEPA-track work
to reproduce that external result first, on its own, before considering
any integration with the JEPA world model. This section covers that
reproduction; it is deliberately self-contained (no shared code, no
shared checkpoints with `jepa/`) so it can be evaluated and submitted on
its own merits.

**Source and attribution.** Ported from Evgenii Rudakov, Ryan Shock, and
Nathan Cowley, *"Graph-Based Exploration for ARC-AGI-3 Interactive
Reasoning Tasks"* (AAAI 2026 Workshop on AI for Scientific Research,
[arXiv:2512.24156](https://arxiv.org/abs/2512.24156)), original code at
[github.com/dolphin-in-a-coma/arc-agi-3-just-explore](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore),
MIT-licensed (compatible with this project's own MIT license; the
required upstream copyright notice/license text is carried forward
verbatim in `ARC-AGI-3-Agents/agents/templates/graph_explorer_THIRD_PARTY_LICENSE`).
Their system placed **3rd on the ARC-AGI-3 Preview Challenge private
leaderboard** using training-free, exact-state graph exploration --
no learned model, no gradient descent, at all. The method has two parts:
a **Frame Processor** (segments each frame into connected same-color
components via flood-fill, detects and masks probable status bars via a
rule-based edge/ratio/twin-count heuristic, and buckets click-target
candidates into 5 visual-salience priority tiers) and a **Level Graph
Explorer** (exact-state hashing + a directed graph over observed
transitions, with BFS-maintained shortest-path distances from every
explored node to the nearest node still holding an untested edge --
"frontier-distance routing" -- so action selection always has a concrete
coverage target, not just a reactive win-recall check). This is
mechanistically the single biggest difference from this project's own
`jepa/memory.py: TransitionGraph.lookahead_best_path` (see
`experiments/stage6_graph_lookahead.md`): that mechanism is gated on a
win having already happened (`cum_delta > 0`), so it measurably never
fired across 90,000 real decisions in this project's own backtest, while
frontier-distance routing needs zero reward signal to be useful at all --
it drives systematic coverage from the very first action.

**Port scope**: `ARC-AGI-3-Agents/agents/templates/graph_explorer_core.py`
(`GraphExplorer`/`NodeInfo` -- the exploration graph engine, algorithm
logic unchanged from upstream) and `graph_explorer_agent.py`
(`FrameProcessor`/`GraphExplorerAgent` -- frame segmentation, status-bar
detection, and the agent itself). Registered as `graphexploreragent`.
Adaptations from upstream (all documented in the module docstrings, not
just here): this repo's vendored `arcengine` has 8 actions (RESET +
ACTION1-7) vs. upstream's local framework snapshot's 7 (RESET +
ACTION1-6) -- `SIMPLE_ACTION_ID2GAME_ACTION` is built dynamically from
`GameAction` so ACTION7 isn't silently dropped; `FrameData` here uses
`levels_completed`/`win_levels` fields, not upstream's `score` field;
upstream's `HeuristicAgent.main()` override (which did level-up detection
inline in its own main loop) was restructured into `choose_action` itself
since this project's other agents all rely on the shared base `Agent.main()`
rather than per-agent overrides -- behaviorally identical, just relocated;
a top-level try/except with a safe fallback was added to `choose_action`
(no `main()` override to put it in, and matches this project's own
established "heartbeat" pattern from `hypothesis_agent.py`); matplotlib-
dependent debug/visualization methods (upstream wrote PNGs for the paper's
own figures) were dropped -- not needed for headless offline/Kaggle runs
and would add a dependency this project doesn't otherwise need.

**A real port bug found and fixed, not just upstream's.** First full local
sweep hit a real crash rate (`falling back` warnings on ~11% of decisions,
one game -- `bp35` -- specifically showing 76 occurrences, 59 of them a
hard `AssertionError: Edge result must be untested before recording a
test`). Root-caused by direct traceback inspection, not guesswork: my own
port's defensive fallback branch (`if hashed_frame not in
self.graph_explorer._nodes: record_test(...)`) hardcoded
`suspicious_transition=False`, while upstream's equivalent line passes the
*real* `suspicious_transition` value through. That value is what gates
upstream's own "ignore until 3 consistent observations" protection
(`GraphExplorer.suspicious_transitions`/`suspicious_transitions_threshold`
-- the exact mechanism upstream's own README credits with fixing their
pre-evaluation RESET-loop bug). Hardcoding `False` bypassed that
protection specifically for GAME_OVER-adjacent transitions (the
`GameState.GAME_OVER` branch short-circuits *before* recording the edge
that led there, by design, matching upstream -- `last_transition_suspicious`
exists precisely to flag the next real transition as unconfirmed), letting
a single, possibly-wrong transition get permanently recorded on its first,
unconfirmed observation -- which later crashed when the same edge was
genuinely retested and produced a different, correct result conflicting
with the bad cached one. Fixed by threading the real `suspicious_transition`
value through instead of a hardcoded default (`graph_explorer_agent.py`,
`_choose_action_inner`). Verified directly on the same game (`bp35`):
AssertionErrors 59 -> 0; total fallbacks 76 -> 16. A residual ~6-7%
fallback rate remains (almost entirely a `KeyError` from `choose_edge`
being called on a still-"suspicious"/unconfirmed node) -- traced this to
code that is *structurally identical* to upstream's own (same call site,
same guard, same outer try/except-and-repeat-last-action pattern) rather
than a further divergence, so it was left as-is rather than "improved"
past what the reproduction is meant to test.

**Local test results (no repeats yet beyond a handful of single 25-game
passes -- see "Next steps" below):**

| run | pooled score | total levels | distinct games | AssertionErrors | notes |
|---|---|---|---|---|---|
| pre-fix sweep | 0.179 | 6 | 5 (`r11l`,`lp85`,`tu93`,`vc33`,`sp80`) | 59 (concentrated on `bp35`) | `suspicious_transition` bug present, ~11% fallback rate |
| post-fix sweep A | 0.255 | 6 | 5 (`r11l`,`lp85`,`m0r0`,`vc33`,`sp80`) | 28 (spread across 25 games) | bug fixed, ~9% fallback rate |
| post-fix sweep B | 0.162 | 7 | 5 (`r11l`,`lp85`,`tu93`,`vc33`,`sp80`) | 0 | bug fixed, ~3.6% fallback rate |

Both single-pass, `MAX_ACTIONS=300`, all 25 local games, matching this
project's own established protocol elsewhere (pooled score is the same
real Kaggle-scoring-formula metric `main.py`'s own scorecard already
computes, not an approximation). `tu93` (2 of 3 runs) is a game none of
this project's own JEPA-based agents (`Hypothesis`/`Memory`/`Curiosity`/
`Random`) have been documented solving anywhere in this file -- a
genuinely different exploration profile, consistent with this being a
structurally different (coverage-driven, not reward-driven) algorithm.
`r11l`, `lp85`, `vc33`, `sp80` recur in all three runs; `tu93`/`m0r0`
swap in/out as the 5th -- expected run-to-run variance (unseeded RNG,
same pattern already documented throughout this project's own agent
comparisons), not yet enough samples for a precise pooled-score estimate.
What *is* clear across all three: the post-fix runs' AssertionError count
(28, then 0) trends toward zero as expected, and pooled score/total
levels stay in a tight, plausible band (0.162-0.255, 6-7 levels) with no
recurrence of the pre-fix crash pattern -- consistent with "the bug fix
produced a real, directionally consistent improvement," not proof of an
exact score by itself.

**First real Kaggle submission: resolved, `0.10`.** Ref `55858509`,
submitted 2026-08-29, `MAX_ACTIONS=300`, standalone dataset/kernel (no
JEPA involvement), submitted before the n=8x25 backtest below existed
(only 3 local single-pass samples behind it at submission time).
`SubmissionStatus.COMPLETE`, public score **0.10**. This sits inside
production `Hypothesis`'s own established real-score range (0.00-0.23,
`0.10` close to that range's median) -- **not a clear win over
`Hypothesis` on real hidden games**, despite dramatically outscoring
every JEPA-based agent's *local* 25-game backtest (see immediately
below). This is a real, honest, non-trivial finding, not a contradiction
to explain away: GraphExplorerAgent's coverage-first exploration
thoroughly dominates the *local, in-distribution* 25-game roster, but
that local dominance did not (at n=1) translate into a real edge on
Kaggle's mostly-novel hidden games -- consistent with this project's own
Stage 6 finding that the local 25 games and genuinely novel games are a
different regime entirely, just demonstrated here from an unexpected
angle (a training-free agent, not a checkpoint-generalization problem).
Per this project's own established discipline for every first submission,
treat `0.10` as n=1 -- informative, not conclusive -- but it does mean
the very large local margin over `Hypothesis` should NOT be read as "this
agent will also dominate the real leaderboard" without more real
submissions to check.

**Follow-up: "is `0.10` a submission-mechanism crash, or genuine?" --
directly checked, not assumed.** A real user catch: given the pure agent
scored 3-4x lower on the real submission than its local mean (0.242), it
was worth directly ruling out a silent setup/mechanism failure before
accepting "held-out-game generalization" as the explanation. Built a
standalone diagnostic kernel (`kaggle_submission_graph_explorer/
notebook_diag/`, reusable going forward, same free-unconditional-cell
trick this doc already documents for the Hypothesis submission) that
mirrors the *real* submission notebook's exact setup order (copy harness
-> copy agent files -> overwrite `agents/__init__.py` with the minimal
version -> import) on Kaggle's actual environment, then goes one step
further than the original Hypothesis diagnostic ever did: constructs a
real `GraphExplorerAgent` instance (via `__new__` + manual attribute
setup, since a diagnostic push has no live `arc_env`) and calls its real
`choose_action()` against a synthetic 64x64 frame end-to-end. **Result:
`ALL FRAMEPROCESSOR DIAGNOSTICS PASSED` and `END-TO-END DIAGNOSTIC
PASSED`** -- `segment_frame`, `identify_status_bars`,
`frame_segments_to_action_groups`, `hash_frame`, and a full
`choose_action()` call all ran cleanly on Kaggle's real `numpy==2.0.2`,
`python==3.12.13` (vs. this project's own dev box: `numpy==2.3.2`,
`python==3.13.14`) and returned a genuine, sensible action
(`GameAction.ACTION6`), not a crash.

**Why this specifically mattered to check:** `choose_action`'s own
top-level try/except means that if the classical-CV pipeline threw on
*every* real decision (e.g. some numpy-version behavior difference), the
agent would silently fall back to `self.last_action_object` every time --
which starts at `GameAction.RESET` and is only ever updated on a
*successful* decision, so a total, silent failure here would look exactly
like "the agent just spams RESET forever," landing near or even below the
random-agent floor without ever showing up as a Kaggle `ERROR` status
(the run still "plays," just does nothing useful). This diagnostic
directly rules that specific failure mode out for the setup/classical-CV
layer. What it can *not* rule out: the real live gateway interaction
(network timing, the real multi-game harness) -- a free test push has no
gateway access, so that's the one thing only a real scored submission
exercises, and this remains unverified.

**Net read on `0.10`:** most likely genuine, not a crash artifact --
three independent, concrete reasons, not just "generalization gaps are a
thing": (1) the diagnostic above directly rules out the specific silent-
failure mode that would produce a near-random score for an unrelated
reason; (2) **even on the familiar local 25-game roster, this agent never
solved 17 of 25 games (68%) across all 8 backtest repeats** -- pooling
across Kaggle's much larger, mostly-novel hidden set (this project's own
Stage 6 work references ~110 games) would plausibly dilute the average
further by simple arithmetic, no mechanism failure required; (3) the
paper's own "3rd place" result is from a *different* evaluation (their
ARC-AGI-3 Preview Challenge, 6 games/52 levels) -- their real-world
success doesn't guarantee it transfers to this specific competition's
hidden set. `0.10` also sits clearly above the established random-agent
floor (`0.06`), consistent with real, functioning exploration having
happened. Not 100% certain without a second same-config submission (this
project's standard way of separating a real effect from a lucky/unlucky
single roll) -- but the mechanism itself is now directly verified clean,
not assumed.

**Kaggle submission.** Given this is a from-scratch reproduction being
tested on its own merits, it gets its own dataset/kernel, entirely
separate from the JEPA `jepa-hypothesis-agent` dataset -- no checkpoints,
no `jepa/` package, no torch/GPU dependency at all (`enable_gpu: false`),
just the two ported agent files. Staged in
`kaggle_submission_graph_explorer/` (`dataset_stage/`,
`notebook/arc3-graph-explorer-submission.ipynb`,
`notebook/kernel-metadata.json`) -- follows the exact same
reproduction/submission steps this file's own Kaggle section documents
below, pointed at the new dataset/kernel ids
(`calamitychasm/graph-explorer-agent`,
`calamitychasm/arc3-graph-explorer-submission`).

**Real n=8x25 backtest (done, this project's own established protocol --
`MAX_ACTIONS=300`, all 25 local games, 8 repeats): a decisive result.**

| repeat | pooled score | levels | distinct games |
|---|---|---|---|
| 1 | 0.328 | 6 | 5 |
| 2 | 0.227 | 7 | 6 |
| 3 | 0.328 | 5 | 4 |
| 4 | 0.348 | 7 | 6 |
| 5 | 0.245 | 6 | 5 |
| 6 | 0.124 | 5 | 4 |
| 7 | 0.128 | 5 | 5 |
| 8 | 0.210 | 5 | 4 |

**Cumulative: 46 total levels completed, 8 distinct games ever solved**
(`bp35`, `cd82`, `lp85`, `r11l`, `s5i5`, `sp80`, `tu93`, `vc33`), **mean
pooled score 0.242** (range 0.124-0.348 across repeats -- real run-to-run
variance from unseeded RNG, same as every other agent in this project,
but every single repeat still clears 0.12).

**This dramatically exceeds every JEPA-based agent's own 8x25-repeat
numbers documented anywhere in this file**: `Hypothesis` (the current
production agent, after every fix -- ACTION6 top-k, dead-end filtering,
argmax-softmax) peaked at 14 total levels / 5 distinct games; `Memory` 10
total / 5 distinct; `Curiosity` 8-11 total / 2-5 distinct across
different rounds; `Random` 10 total / 2 distinct. GraphExplorerAgent's 46
total levels is **more than 3x** the best any learned-model agent this
project has built has ever reached on this same protocol, and 8 distinct
games beats every one of them too. Zero training, zero GPU, zero
checkpoints. This is the single clearest local result in this entire
project's history for "does this approach reach more of the local game
roster than what we've built" -- not close enough to attribute to noise
the way several other n=8 results in this file have turned out to be
(compare to the novelty-aware-beta cap's n=8 result, which evaporated at
n=30 -- this result's margin over every other agent's own n=8 total is
far too large for that same story to apply here).

**What this does and doesn't say:** it says GraphExplorerAgent's
coverage-first, training-free approach explores the *local* 25-game
roster far more thoroughly than any world-model-driven agent this project
has built. It says nothing yet about the held-out/novel-game
generalization question Stage 6 spent so much effort on (`tu93`
specifically -- and every other game here -- is a *trained* game for the
JEPA agents' checkpoints in the sense that the whole local corpus is
in-distribution for them; GraphExplorerAgent has no training at all, so
"in-distribution" doesn't even apply the same way). The real Kaggle
submission (below) is the only test that touches genuinely novel games --
it has since resolved at `0.10`, inside `Hypothesis`'s own established
range and not a clear win, confirming this local dominance did not (at
n=1) carry over to real hidden games.

**Next steps, not yet done:** (1) once the real submission below
resolves, decide whether a second same-config submission is worth a
future day's quota (n=1 cannot separate a real effect from noise, same
standard as every other first submission in this file); (2) consider
whether/how anything from this agent's design (frontier-distance routing
especially) is worth folding into the JEPA track's own `TransitionGraph`
-- tracked separately in `experiments/stage6_graph_lookahead.md`'s own
"next steps" -- rather than merging the two agents' code; (3) given how
large this local margin is, worth asking directly whether the *held-out-
game* generalization gap Stage 6 fought all session (13 interventions, 12
failures) is even the right problem to keep attacking with a learned
world model at all, vs. leaning further into training-free, coverage-
first exploration for whatever fraction of Kaggle's hidden games turn out
to resemble the local roster closely enough for graph exploration alone
to matter.

### MAX_ACTIONS=300 was never a real Kaggle constraint -- and raising it surfaced a real, fixed bug

**Why 300 was wrong for this agent specifically.** 300 matched this
project's own established local-testing convention (`Hypothesis`/
`Curiosity`/`Memory` all use it), not a real competition limit. Checked
directly: `rules.md` documents a **9-hour run-time cap for the whole
notebook** (not per-game), and `agents/swarm.py: Swarm.main()` -- read
directly, not assumed -- spins up **one thread per game and starts them
all together**, joining only once every thread finishes. Every one of the
up to 110 real hidden games therefore gets close to the *entire* 9-hour
window concurrently, not `9h / 110 games` divided. Upstream's own real
default was `MAX_ACTIONS = 1000000` (effectively unbounded, gated by
`TOTAL_TIME_ALLOWED = 7.9 hours`) -- not a small fixed count. `MAX_ACTIONS`
is now `GRAPH_EXPLORER_MAX_ACTIONS`-overridable (default still 300, so
existing comparisons elsewhere in this file are unaffected unless
explicitly overridden).

**Calibration run (`r11l`, `GRAPH_EXPLORER_MAX_ACTIONS=5000`, isolated --
no 25-way CPU contention) found two things, one reassuring, one a real
bug.** Reassuring: **~49 actions/sec sustained**, 5000 actions in ~102
seconds, with fps *increasing* slightly and stabilizing over the run --
no evidence the graph's own bookkeeping (`_rebuild_distances`'s BFS,
`NodeInfo.edge_data`) gets meaningfully slower as it grows across
thousands of actions. The bug: the agent completed its one level at
**action 88**, then got stuck for the remaining **4912 of 5000 actions**
-- 95 resets, ~90% of decisions hitting the graph's own "choose_edge
called on a still-unconfirmed node" fallback, zero further progress.
Root cause, found by reading `choose_action`'s own fallback path: on any
exception it returns `self.last_action_object`, which only updates on a
*successful* decision -- so once stuck, the agent replays the exact same
action against the exact same stuck state forever, since repeating a
failing action can never change whatever caused it to fail. Invisible at
300 actions (just looked like "somewhat elevated fallback rate"); at
5000 it's revealed as a real trap, not noise.

**Fix**: track consecutive `choose_action` exceptions
(`self._consecutive_fallbacks`); past `CONSECUTIVE_FALLBACK_ESCAPE_
THRESHOLD` (3), stop repeating `last_action_object` and escape with a
genuinely fresh random action (`_random_escape_action`, sampling from
`latest_frame.available_actions` including a real random `(x, y)` for
ACTION6) instead. Re-ran the identical `r11l` calibration: fallback count
dropped **4484 -> 3141**, resets rose **95 -> 235** (more erratic, but no
longer perpetually stuck on the exact same action) -- **but levels
completed on this one game still capped at 1.** Read honestly: the fix is
real and verified to change behavior (the literal infinite-repeat pattern
is gone), but it did not, by itself, unlock further progress on this
specific game in this one run -- whatever caused progress to stall after
level 1 on `r11l` is evidently a separate, deeper issue than the
repeat-loop mechanism alone. Not yet investigated further (n=1, one
game) -- proceeding to the full 25-game sweep at the larger budget is the
next real test of whether the fix (and the larger budget itself) helps
broadly, rather than reading too much into one game's result.

### "Urgency": exact multi-hop win-recall, reusing jepa/memory.py's TransitionGraph

Motivated directly by the MAX_ACTIONS calibration finding above: a
coverage-first algorithm with no notion of "I already know how to make
progress here" pays a real, measured cost (the "coupon collector" effect)
every time it has to rediscover a known-good move via undirected random
exploration instead of just taking it. `GraphExplorerAgent` had no
exact-recall mechanism at all (unlike Stage 3/5's `TransitionGraph`-based
JEPA agents) -- this section adds one.

**First attempt (single-hop) was cleanly negative, verified by direct
measurement, not assumption.** Recorded `hashed_frame -> (action_id, xy)`
for every transition observed to increase `levels_completed`, checked it
on every decision. Result on a 5000-action `r11l` calibration: **zero
recalls fired.** Root cause, found by instrumenting rather than guessing:
`RESET` returns to the *level's own starting frame*, not the specific
(often several-steps-downstream) frame where the winning action was
actually taken -- a single remembered hop can't bridge that gap. This
needed a real fix, not a bigger lookahead constant.

**Second attempt: reuse `jepa/memory.py: TransitionGraph.lookahead_best_path`
directly** (already built, already tested, already used by
`Hypothesis`/`Memory` for exactly this kind of multi-hop exact recall).
It's pure Python (`hashlib`/`dataclasses`/`collections.deque`, zero torch)
-- reusing it doesn't compromise this agent's "no JEPA world model"
isolation, it's just a proven, non-learned data structure. Records
*every* observed transition (not just wins) so a full path can be
reconstructed, not just the final hop; `self.transition_memory` is a
separate, never-reset structure from `self.graph_explorer` (which still
gets `reset()` on every level_up, unchanged).

**A second bug, also found by instrumenting rather than assuming the fix
worked:** still zero recalls after switching to multi-hop lookahead.
Traced directly: `TransitionGraph`'s own default hashing works on the
*raw, unmasked* frame (matching how `Hypothesis`/`Memory` already use it),
but `GraphExplorerAgent` hashes the *status-bar-masked* frame specifically
so repeat visits to the same logical state hash identically. Using two
different hashing schemes for the same lookup meant states essentially
never matched. Fixed by adding an optional `next_state_key` override to
`TransitionGraph.record()` (backward-compatible -- existing positional
callers in `Hypothesis`/`Memory` are unaffected, verified directly) and
keying everything in `GraphExplorerAgent` off its own existing
`hashed_frame` consistently.

**With both fixes in place, direct instrumentation confirmed the
mechanism works exactly as designed** -- states *are* recognized as
revisits (`transition_memory` grows, `seen()` returns `True` on repeat
visits), edges accumulate correctly, and the graph-size/fps profile from
the earlier calibration still holds (no slowdown). But real recalls on
`r11l` specifically *still* never fired, and this time the reason is not
a bug: direct inspection of `levels_completed` across dozens of `RESET`
events showed it **never reverts** -- it stays at `1` across the entire
rest of the run. **ARC-3 checkpoints progress at the level boundary; a
`RESET` returns you to the *current* (not-yet-won) level's own start, not
the game's absolute beginning.** `r11l` reached level 2 once (around
action 35) and has been stuck there -- never won even once, across every
test run this session -- ever since. There is no second win anywhere in
`transition_memory` for the mechanism to route back to; it isn't broken,
it simply hasn't had an opportunity yet on this specific game. This also
means the mechanism's real value is narrower than first framed: it helps
when a GAME_OVER/RESET cycle returns to a state *within the current,
not-yet-won level* that a past attempt already knows a productive
continuation from -- not "re-crossing an already-permanently-passed
earlier level," which apparently doesn't happen given how progress is
checkpointed.

**Full 25-game sweep at the same MAX_ACTIONS=5000 budget: the run itself
crashed (disk-full, unrelated to the urgency logic), but the real result
was still recoverable, and it's a clean negative -- the mechanism never
fired once across the whole roster.**

The sweep (all 25 games, started together at 17:15:06) hit
`OSError: [Errno 28] No space left on device` partway through -- first in
`Swarm.main()`'s own `logger.info(json.dumps(scorecard.model_dump()))`
call, then in several agents' `recorder.record()` cleanup writes. The 13
games whose "Finishing" line did make it into the log all show the exact
same elapsed time (2964.2-2964.6s) despite very different action counts
(1787-4930) -- strong evidence a shared I/O failure killed them together
partway through, not that they legitimately hit a stopping condition
independently. **This is the same "full disk" failure class this doc's
own Gotchas section already warns about for `recordings/`** -- old
recordings from this session's own single-hop/hash-mismatch debugging
runs (13:0x-17:10 timestamps, several hundred MB each) were still sitting
uncleaned when the final sweep started, and a 25-game x 5000-action sweep
alone generates multiple GB more (individual recording files up to
~600MB, frame-size-dependent, consistent with this doc's own
already-documented per-file sizes at this scale).

**Real per-game results were still recoverable directly from the 25
recording files** (each one accumulates real `levels_completed`/
`win_levels` per frame regardless of whether the final aggregate
scorecard JSON ever got logged) -- parsed directly rather than trusting
the crashed log:

**17 total levels completed, 12 distinct games** (`ar25, cd82, dc22,
ft09, lf52, lp85, m0r0, r11l, s5i5, sp80, tu93, vc33`) -- essentially
flat against the pre-urgency 5000-action baseline (17 levels, 13 distinct
games) elsewhere in this section. No detectable improvement, no
detectable regression.

**The urgency mechanism never fired once across all 25 games in this
run** -- confirmed by grepping the full log for its own
`logger.info(f"{self.game_id} - {reasoning}")` call (the same line
verified to work correctly in the earlier isolated r11l diagnosis), not
assumed from a null result alone. Consistent with, and now generalizing,
the r11l-specific finding above: the condition it needs (revisiting an
*exact* already-explored state that has a recorded, still-relevant
productive path within `URGENCY_LOOKAHEAD_DEPTH=8` hops) is apparently
rare across a single pass at this game roster's scale, not just
specific to r11l's own stuck-at-level-2 situation. The mechanism itself
was directly verified correct via instrumentation before this run (states
recognized as revisits, edges accumulating, a real level-up recorded) --
this is a "rarely gets a chance to matter in one pass," not "broken,"
result, but it's still a clean, honest null on the actual question asked
("does this help on the full roster") at this budget and this graph
depth.

**Working read:** exact multi-hop win-recall remains a theoretically
sound mechanism (ARC-3's determinism means every recorded edge is a
verified fact, not a prediction) but its practical payoff at
`MAX_ACTIONS=5000`/depth=8 on a *single* pass through 25 games is
essentially zero, because there's rarely enough repeat-visit structure
within one pass for a multi-hop path to have already been recorded before
it would matter. It would more plausibly pay off on a much longer-lived
graph (e.g. `Memory`'s own persist-across-resets pattern given a much
larger action budget, or a `TransitionGraph` seeded from a prior
session's harvest rather than built from scratch each run) -- not
verified this session, flagged as the natural next test if this is
revisited, rather than iterated on further right now given the clear
null result at this scale.

**Housekeeping from this run**: `scripts/extract_level_up_transitions.py`
crashed on a truncated final JSON line in one of the recording files (a
direct consequence of the disk-full crash cutting a write off mid-line)
-- fixed by skipping a malformed line instead of aborting the whole
extraction (`json.JSONDecodeError` now caught in `_load_lines`, harmless
for well-formed files). Ran the fixed extractor across the full
`recordings/` directory (207 files, both this run's and the session's
earlier stale debugging runs) before deleting anything: 45 level-up
events, 2043 transitions preserved into `data/graph_explorer_harvest/`.
Then deleted all raw recordings (freed ~8.9GB, C: went from ~5.9GB to
~15GB free) -- this project's disk margin was already thin (documented
elsewhere in this file) and a single large sweep plus leftover debug
recordings was enough to exhaust it outright this time, not just come
close.

### GraphExplorerStructuralAgent: a purpose-built graph win-distance model, a clean negative result

Follow-up to a design discussion about generalizing "the graph" beyond
exact-hash lookup: does a small model trained on structural graph features
(out-degree, in-degree, visit count, BFS depth from episode start, node
candidate-richness) transfer across games to predict "hops to a known
win from this exact state"? `jepa/graph_features.py:
StructuralGraphTracker` + `scripts/train_graph_win_distance_model.py`
(a small `GradientBoostingRegressor`, deliberately not a torch model).

**Result: a clean, honest negative on the model's own validity check,
before ever reaching a backtest.** Leave-games-out cross-validation (5
folds, 13 games, 8096 labeled decisions) scored *worse* than a trivial
constant baseline -- mean MAE 5.61 hops vs. 4.27 for predicting the
training mean everywhere, correlation 0.137. Feature importances showed
why: `depth` (48%) and `num_available_actions` (37%) dominated, both
absolute, game-specific scales -- the model was mostly learning "which
game's rough size is this," the same "memorize local statistics instead
of a transferable feature" failure Stage 6's object-identity checkpoint
already hit, just for graph structure instead of encoder features. Given
the model failed its own honest validation, no agent-level backtest or
submission was attempted for this design -- see the immediately following
section for the design that replaced it.

### GraphExplorerLearnedAgent: local click-effect prediction, a decisive local win, real submission still unresolved

**Origin: a sharper version of "does the graph have exploitable local
structure," reframed from graph topology to local pixel/object content.**
A user-proposed diagnostic (`scripts/diagnose_state_similarity.py`) asked
a narrower, safer question than global frame symmetry (deliberately
rejected -- a puzzle could use symmetry as its actual mechanic, not
decoration): do two *different* exact graph states that happen to share
the same small local neighborhood around an ACTION6 click point show the
same click *outcome*? Run against the win-adjacent harvest corpus (3,586
click transitions, 12 games): **624 cross-state local-patch matches, with
~100% frame-changed-outcome consistency in 9 of 12 games** (one group in
`r11l` recurred across 14 distinct states, always with the same outcome).
A weaker, position-invariant "same object, different location" version
(120 matches) was much less consistent (0-100% depending on game) and not
pursued further. Full method and numbers in that script's own output
(`data/state_similarity_report.json`).

**Design: pretrain across games, then test-time-adapt per game --
deliberately the "higher-power" option over a cheap non-parametric
nearest-neighbor upgrade, at explicit user request regardless of what the
cheap version would have shown.** `jepa/click_effect_model.py:
ClickEffectModel` -- a small CNN (7x7 local pixel patch, color-embedded)
+ MLP (5 segment shape/color/size features) fused into a shared trunk with
two heads (P(frame changes), P(this click wins)). Pretrained on the same
harvest corpus (`scripts/harvest_click_effect_data.py`,
`scripts/train_click_effect_model.py`): leave-games-out CV mean AUC on
`frame_changed` = **0.77** (3 of 4 folds clearly above chance, one
near-chance) -- a real, if imperfect, signal, and a qualitatively
different (much better) result than the structural model's above. Raw
accuracy looked bad (0.573 vs. 0.861 majority-baseline) purely because
per-game "does clicking change anything" base rates range 0%-100% across
games -- AUC is the honest metric since the live agent recalibrates per
game anyway via `jepa/click_effect_adapter.py: ClickEffectAdapter`
(same ANIL-style restricted-subset test-time-adaptation recipe as
`jepa/test_time_adapter.py`, the one lever in this project's whole Stage 6
investigation that showed real positive cross-game signal -- see that
section's own history): a ~600-param subset (trunk + both heads) gets a
few real gradient steps every ~10 observed clicks using the *current*
game's own data, persists across resets, resets fresh per new game.

**Live integration is scoped to explore-mode only, on purpose, unlike
`GraphExplorerJepaAgent`'s regression.** That earlier class biased both
explore- and travel-mode using a signal later shown to be near-flat
(InfoGain) and was a clear loss (34 vs 46 levels). This class only biases
*which untested click to try next* (temperature-weighted softmax over
predicted P(frame_changed), same z-normalized pattern as every other
tie-break in this file) -- travel-mode stays untouched. The structural
risk that any explore-mode bias can front-load GAME_OVER-triggering
actions is real regardless of signal quality and was directly observed in
a single-game smoke test (r11l, 150 actions): fallback rate 58% vs. the
pure agent's 35% on identical budget -- flagged before the backtest, not
after.

**Real n=8x25 backtest result: decisive, not marginal.**

| metric | pure GraphExplorerAgent | GraphExplorerLearnedAgent |
|---|---|---|
| total levels | 48 | **55** (+15%) |
| distinct games ever | 8 | **9** |
| mean pooled score | 0.179 | **0.674** (3.76x) |

Every one of the 8 learned-agent repeats (0.51-0.77) scored higher than
every one of the 8 pure-agent repeats (0.07-0.37) -- complete separation
between the two distributions, not an overlapping-noise result. The
elevated single-game fallback rate did not show up as a pooled-level
cost. Mechanistically coherent, not just statistically clean: levels
completed only rose modestly while pooled score (efficiency-weighted,
squared) nearly quadrupled -- consistent with the model successfully
skipping dead clicks rather than unlocking dramatically more content.

**Real Kaggle submission: a real setup bug found and fixed, but still
unresolved after two attempts.** Staged as
`kaggle_submission_graph_explorer_learned/` (dataset
`calamitychasm/graph-explorer-learned-agent`, kernel
`arc3-graph-explorer-learned-submission`), following the same
proven-pattern reuse as the pure GraphExplorerAgent's own submission
pipeline. First submission (ref `55901265`) was made *before* waiting for
an unconditional diagnostic push to finish (explicit user instruction:
speed over caution) -- the diagnostic then found a real bug missed during
staging: `graph_explorer_agent.py` imports `from jepa.memory import
TransitionGraph` (for its own urgency/win-recall mechanism) but
`jepa/memory.py` was never copied into the dataset. Fixed (dataset
version 2), diagnostic re-run and confirmed clean (`MODEL DIAGNOSTICS
PASSED`, `END-TO-END DIAGNOSTIC PASSED`, including a full simulated
`choose_action` call exercising both the tie-break and
`adapter.observe`/adaptation paths). **The first submission still
resolved to `SubmissionStatus.ERROR`** (only Kaggle's generic "system
error" message, no further detail via the API). A resubmission (ref
`55901513`), made *after* the dataset fix landed, **also errored
identically** -- same generic message, same underlying kernel version.
Since the diagnostic (import + model load + a full simulated decision)
now passes cleanly against the exact same fixed dataset, the remaining
unverified surface is specifically the real gateway/multi-game harness
execution during an actual scored rerun -- something this project's own
free-diagnostic-push technique has never been able to reach (documented
limitation, see the original Kaggle debugging saga earlier in this file).
Per user instruction, retries are now spaced ~50 minutes apart rather
than immediate, to avoid mistaking a transient platform issue for a real
code bug (or vice versa).

**A concrete but ultimately wrong hypothesis, tested and ruled out
directly: `enable_gpu`.** Neither proven-working torch-based pattern in
this project actually tests torch-with-`enable_gpu:false` -- the pure
`GraphExplorerAgent` has no torch dependency at all (the flag is moot),
and `Hypothesis` uses torch with `enable_gpu:true`. Switching this
class's kernel to `enable_gpu:true` to match the one proven torch+GPU
combination seemed like a well-reasoned next step. **It was wrong, and
revealed a different real bug instead**: Kaggle's GPU pool for this
kernel provisioned a Tesla P100 (CUDA capability sm_60), which the
bundled PyTorch build's compiled kernels don't support (minimum sm_70) --
every real tensor op on GPU threw `torch.AcceleratorError: CUDA error: no
kernel image is available for execution on the device`, confirmed
directly by toggling the flag on identical code via the diagnostic
kernel. `torch.cuda.is_available()` returns `True` regardless (a GPU is
physically present, just incompatible), so `jepa/device.py: get_device()`
has no way to detect this ahead of time -- see this file's own Gotchas
section for the durable version of this finding. Reverted to
`enable_gpu:false` (the only config ever confirmed clean) for both the
submission and diagnostic kernels.

**Three submissions, three identical failures, still unresolved.** A
third attempt (ref `55902368`) on the confirmed-CPU config, made with no
new code change (there was nothing left to fix locally), also resolved to
`SubmissionStatus.ERROR` with the same generic message and the same
underlying kernel version as the first two -- the only notable difference
was staying `PENDING` for ~64 minutes before erroring, versus a much
faster resolution on the first two attempts, a real if inconclusive data
point. Per direct user instruction, no fourth attempt was made
automatically. **Everything a diagnostic push can reach is now confirmed
clean**: file/path staging (audited byte-for-byte against two
proven-working submission notebooks), model loading and a full simulated
`choose_action` call, and 110-thread concurrent model construction +
inference + live adaptation (both GPU and CPU-forced, mirroring Kaggle's
real per-game-thread Swarm scale) all pass with zero errors. Three
consistent, identical failures on an otherwise fully-verified setup is
more consistent with something in the real online-mode gateway/multi-game
harness execution -- the one path no free push can reach -- than with a
remaining code bug, though this isn't provable from here without either
Kaggle Support's own visibility or another submission attempt.
**Resolved: a real score, on the 5th attempt -- and a real, sobering
generalization gap, not just a submission-mechanism saga.** A CONTROL
TEST (resubmitting the unchanged, already-proven pure `GraphExplorerAgent`
kernel while the 3rd learned-agent error was still fresh) came back
`SubmissionStatus.COMPLETE`, score **0.25** -- conclusively ruling out a
platform-wide outage as the explanation for the 3 identical errors, since
Kaggle's real infrastructure was demonstrably working normally at the
same time. That refocused the search onto what's actually different
between this project's two proven torch-based configs (pure
`GraphExplorerAgent`: no torch, `enable_gpu:false`; `Hypothesis`: torch,
`enable_gpu:true`) and this class's own original config (torch,
`enable_gpu:false`) -- the one combination never proven anywhere in this
project's history. A 4th attempt hit Kaggle's real daily submission quota
(confirmed via the API's own `FAILED_PRECONDITION` error body, not
assumed) -- errored submissions evidently still count against it, contra
this project's earlier assumption. After waiting out the ~6-hour reset
(confirmed precisely via the quota error's own countdown), a 5th attempt
switched to `enable_gpu:true` (matching `Hypothesis`'s proven profile)
while adding `GRAPH_EXPLORER_LEARNED_FORCE_CPU=1` (a new env-var lever,
`graph_explorer_learned_agent.py: _init_model`) to keep this agent's own
compute on CPU regardless -- avoiding the confirmed P100 incompatibility
while still running on the one proven infrastructure profile. Diagnostic-
verified clean first (`device: cpu`, no CUDA errors, both diagnostics
passed), then submitted (ref `55926701`): **`SubmissionStatus.COMPLETE`,
public score 0.08.**

**Honest read: this does not look like the local backtest's 3.76x
advantage transferring to real hidden games.** `0.08` sits *below* the
pure agent's own two real scores from this exact investigation --
`0.10` (an earlier submission) and `0.25` (today's own control test, on
the same infrastructure, same day) -- despite the learned agent
dominating the pure agent completely in every local backtest repeat
(0.51-0.77 vs 0.07-0.37, zero overlap). Same standard as every other
first-real-submission result in this project's history: this is n=1 for
`GraphExplorerLearnedAgent` and cannot on its own distinguish "the local
signal genuinely doesn't transfer" from "this one draw landed unlucky" --
but the gap here is large, and it lands in the same direction as this
project's own dominant, repeated finding elsewhere (Stage 6 addendum: 13+
interventions on the JEPA world model, only test-time adaptation showed
any real cross-game signal). A component-level win (patch-match
consistency, AUC 0.77 cross-validated by game) and a massive local
agent-level win did not clearly survive contact with genuinely novel
games. Worth a second same-config submission before treating this as
settled, per this project's own standard practice -- but the honest prior,
given everything else this project has found about local-to-hidden-game
transfer, should lean toward "another instance of the pattern," not
"probably just unlucky."

**Second data point (2026-09-04, ref `56003465`, identical config, no
code changes): `SubmissionStatus.COMPLETE`, public score `0.15`.** Two
real scores now: `0.08`, `0.15` (mean `0.115`). This does *not* confirm
the first score was a stable low floor -- `0.15` is noticeably higher --
but it also doesn't come close to reproducing the local backtest's 3.76x
advantage: the pure agent's own two real scores from this same
investigation (`0.10`, `0.25`, mean `0.175`) are still higher on average
than the learned agent's two (`0.115`). At n=2 per side this is not a
statistically clean comparison, and both learned-agent scores sit
squarely inside this project's own long-established `0.00`-`0.25` real-
score noise band -- but there is no signal yet that this agent's local
dominance is showing up on hidden games. Verdict unchanged from the first
data point: consistent with, not a refutation of, this project's dominant
finding that local wins from a learned component don't reliably transfer
to novel games. Before checking prior probabilities, checked directly
whether a bug could explain the low scores (revert-to-random or similar)
rather than assuming: a fresh local debug run (r11l, 300 actions, DEBUG
logging) showed the tie-break scoring path firing 40 times with zero
exceptions and varied, non-degenerate output; the Kaggle diagnostic
already separately confirmed the same on the actual deployed checkpoint;
and the real scores (`0.08`, `0.15`) sit above the documented random-
agent floor (`0.06`), not collapsed onto it -- no positive evidence of a
silent-failure bug, though real hidden-game content and real scored-run
logs remain permanently unavailable to verify further from here.

**Where this leaves the design, if revisited:** the most defensible next
lever, per this project's own strongest cross-cutting finding, is pushing
harder on `ClickEffectAdapter`'s live test-time adaptation (the untuned
`k=10`/`lr=5e-4`/`n_steps=3` defaults, or seeding it with deliberate
opening probe clicks) rather than trying to improve the frozen pretrained
prior -- adaptation is the one mechanism that's shown real cross-game
signal anywhere in this project's entire Stage 6 history. A broader
pretraining corpus drawing on the same synthetic sources already built
for the JEPA predictor (MiniGrid, Sokoban click-transitions, same
patch/segment-feature framing) is untried for this specific narrower
task and worth a shot, though "more of the same kind of data" has failed
repeatedly elsewhere in this project. Capacity/architecture/augmentation
changes are the least promising avenues -- all three shapes of that idea
were tried on the JEPA side and were uniformly negative or harmful.

**Follow-up: swept `ClickEffectAdapter`'s dose, the recommended next lever
-- and it was a real, substantial win, component-level.**
`scripts/sweep_click_effect_tta.py` (leave-games-out: pretrain on all-but-
one game, adapt on that held-out game's own first ~70% of transitions in
order, evaluate AUC on its last ~30%, never adapted on -- same discipline
as every other cross-game check in this project). The original untuned
defaults (`k=10, n_steps=3, lr=5e-4`) were never actually validated --
just a reasonable-looking starting guess. Frozen (no adaptation) pooled
AUC on held-out games: **~0.51, barely above chance.** The original
defaults only reached **~0.54** -- adaptation was firing, but far too
weakly to matter. A real, clean dose-response curve (two independent
sweep runs): jumps sharply to **~0.77** at `k=5, n_steps=8, lr=2e-3`
(one game, `ft09`, went from 0.07 -- worse than random -- to 0.99), then
plateaus/wobbles in the 0.73-0.76 band for every higher dose tried (up to
`k=1, n_steps=40, lr=2e-2`) -- confirming this isn't "more is always
better," there's a real ceiling around the "aggressive" tier. One game
(`m0r0`) never improved and slightly degraded at higher doses in both
runs -- consistent enough to be a real per-game characteristic, not
noise. Adopted `k=5, n_steps=8, lr=2e-3` as the new default. Verified
locally afterward (fresh single-game run, r11l, 300 actions): stable,
zero exceptions, ~24.7 fps sustained -- the heavier dose (n_steps 3->8,
firing twice as often) doesn't introduce any latency or stability
regression at this model's tiny size.

**Same honest caveat as everywhere else in this file applies here too:**
this component-level validation (leave-games-out AUC on the *local* 25
games) is exactly the kind of local signal that has NOT reliably
predicted real Kaggle scores for this agent so far (see the two real
submissions above, 0.08 and 0.15, against a local backtest that predicted
a 3.76x win). Confirmed via a fresh n=3 local backtest with the new dose
(0.678 mean pooled score, consistent with the old dose's 0.674 at n=8 --
no operational regression), then submitted for real scoring (ref
`56022267`): **`SubmissionStatus.COMPLETE`, public score `0.05`.**

**This is the lowest real score this agent has ever gotten, and it's
actually below the documented random-agent floor (`0.06`).** Three real
scores now: `0.08`, `0.15`, `0.05` (mean `0.093`) -- the tuned dose,
despite a dramatically stronger *component-level* signal (pooled held-out
AUC 0.54 -> 0.77, one game going from worse-than-random to near-perfect),
produced this agent's worst real-world result yet, not its best. This
directly answers the question this section's prior caveat posed: "was the
adaptation just too weak before" is **not** the explanation for the
disappointing real scores -- making adaptation dramatically stronger by
every local measure available made the real score *worse*, not better.
That rules out "insufficient dose" as the bottleneck and leaves the same
explanation this project has landed on repeatedly elsewhere: a real
local-to-hidden-game generalization gap, not a tunable hyperparameter.

**Worth flagging as a real, if speculative, possibility rather than
dismissing:** a more aggressive adaptation dose adapts *faster* to
whatever local pattern it happens to observe early in an episode on an
unfamiliar game -- if a hidden game's own click semantics don't resemble
the 12 local games this model was pretrained on, faster/heavier
adaptation could mean committing harder, and earlier, to a wrong read of
that specific game, rather than staying appropriately uncertain. This
would be a case where local-to-hidden-game transfer failure and stronger
adaptation actively compound instead of the latter compensating for the
former -- consistent with, but not proven by, this single real data
point. Given three real scores now sit at `0.05`-`0.15` against the pure
agent's own `0.10`/`0.25`, and the local backtest has now been shown
twice in a row to not predict the real outcome (once for the base dose,
once for the tuned dose, in opposite directions of "surprise"), further
tuning of this specific mechanism is not recommended without first
finding a way to validate against something more like real hidden-game
conditions than the local 25-game roster -- the same conclusion Stage 6's
JEPA-side investigation reached after 13+ attempts, now independently
reproduced by a completely different, much simpler component.

### GraphExplorerJepaAgent: built, backtested, and a clear negative result

`graph_explorer_jepa_agent.py: GraphExplorerJepaAgent` subclasses
`GraphExplorerAgent` unchanged and adds informed tie-breaking at the two
places the pure algorithm has no basis for choice beyond uniform random:
`mode="explore"` (which untested action to try next within the current
priority tier) scores candidates via `jepa/hypothesis_bundle.py:
info_gain` (the same MoE-expert-disagreement signal `Hypothesis` already
uses); `mode="travel"` (navigating toward a known frontier once the
current node has nothing left to explore) scores candidates via the
value head against a frame cache (`self._frame_cache: dict[hash ->
pixel array]`, populated as a side effect of every tie-break call, since
`GraphExplorer` itself only ever stores hashes). `graph_explorer_core.py:
GraphExplorer` grew an optional `tie_break_fn(node, mode, candidates) ->
edge_idx` hook for this, called from `choose_edge` in place of
`random.choice` when set; default `None` reproduces the pure agent's
exact behavior (verified: existing unit tests still pass unchanged).
Registered as `graphexplorerjepaagent` -- needed a manual
`AVAILABLE_AGENTS` entry (`agents/__init__.py`), same as `ReasoningAgent`,
since it subclasses `GraphExplorerAgent` rather than `Agent` directly.

**A real bug found and fixed before backtesting, via direct evidence, not
left as speculation.** The first draft's `r11l`-specific anomaly (179/301
fallback actions vs. 0 for the pure agent on the same game) was
originally written up as "consistent with InfoGain seeking disruptive
actions" -- an unverified hypothesis. Checked directly instead of left
that way: added DEBUG-gated per-decision logging (`.env`'s `DEBUG=True`),
replayed `r11l`, and found the real story is different. Across 89 real
explore-mode decisions, candidate InfoGain scores clustered within
roughly 0.0002-0.0006 of each other on a ~0.017 baseline (near-flat, not
meaningfully differentiated), and the chosen click location landed on
just 2 of 39 distinct spots 22/89 times (~25%). This is the exact
"deterministic argmax over a flat/near-flat map defaults to the same
handful of indices" failure this project already hit and fixed twice
before (`Curiosity._sample_click`, `Hypothesis`'s
`PATCH_SAMPLE_TEMPERATURE`) -- newly discovered in a third context.
Fixed the same way: replaced the hard `argmax` with temperature-weighted
softmax sampling (`_weighted_sample`, `TIE_BREAK_TEMPERATURE = 0.5`,
z-normalized so the temperature is scale-invariant across info_gain's
and the value head's differing raw magnitudes).

**Real n=8x25 backtest (this project's own established protocol,
`MAX_ACTIONS=300`, all 25 local games, 8 repeats): a clear, decisive
regression relative to the pure agent.**

| repeat | pooled score | levels | distinct games |
|---|---|---|---|
| 1 | 0.156 | 4 | 4 |
| 2 | 0.106 | 5 | 5 |
| 3 | 0.114 | 3 | 3 |
| 4 | 0.059 | 5 | 4 |
| 5 | 0.115 | 4 | 4 |
| 6 | 0.137 | 3 | 3 |
| 7 | 0.114 | 5 | 5 |
| 8 | 0.167 | 5 | 4 |

**Cumulative: 34 total levels, 6 distinct games** (`bp35`, `lp85`, `r11l`,
`sp80`, `tu93`, `vc33`), **mean pooled score 0.121** -- vs. the pure
agent's own same-protocol **46 levels, 8 distinct games, mean 0.242** on
identical code otherwise. Every single JEPA-augmented repeat's score sits
below the pure agent's *worst* repeat (0.124). This is not noise: the
margin (34 vs 46 levels, roughly half the mean pooled score) is far too
large and far too consistent across all 8 repeats to be sampling
variance -- unlike, say, the novelty-aware-beta cap's n=8 result
elsewhere in this file, which evaporated at n=30 because its own n=8
margin was already small and inconsistent.

**Directly checked and ruled out the obvious explanation before accepting
the result.** Given the first draft's `r11l` finding involved elevated
exception/fallback rates, the natural hypothesis was "the JEPA variant
just crashes into more fallback (i.e. effectively-random) decisions,
wasting budget." Checked directly: total `falling back` events summed
across all 8 repeats were **2796 for the JEPA-augmented agent vs. 5025
for the pure agent** -- the JEPA variant actually has *fewer* exceptions
overall, not more. So the regression is not an artifact of instability;
it's the informed signal itself, working as designed (softmax-sampled,
not degenerate), producing a worse policy than uniform random for this
specific coverage-first algorithm.

**Working read, not yet directly verified further:** the pure algorithm's
whole mechanism of action is broad, roughly-uniform coverage of untested
actions within a fixed budget -- any systematic bias in *which* untested
action gets tried first, even a well-motivated one, changes *when* within
an episode a given action (including any that happen to trigger
GAME_OVER) gets tried, not just whether it eventually does. If informed
selection front-loads certain actions earlier in an episode than uniform
random would, and a meaningful fraction of untested actions are
GAME_OVER-triggering, that could mean more, earlier resets and less depth
reached per episode -- a real structural cost that a "smarter" per-step
choice doesn't show up in until measured at the whole-episode level. This
would explain why fallback *count* went down (fewer suspicious/unconfirmed-
node hits, since informed choices may correlate with edges the graph
already partially understands) while overall *outcome* still got worse
(the something-lost is in exploration breadth/depth, not caught by the
exception-rate metric at all). Consistent with, not proof of -- not
verified with targeted per-episode instrumentation the way the argmax bug
was.

**Recommended next steps, in order, if this is picked back up:** (1)
directly test the "front-loaded resets" hypothesis above by instrumenting
episode-level "actions before first RESET" and "unique nodes discovered
before first RESET" for both variants, matched-seed if possible; (2) if
confirmed, the fix is likely to bias tie-breaking *toward* safer-looking
candidates rather than toward high-uncertainty ones (the opposite of the
current design) -- try scoring by *negative* value-head-predicted risk
or by a fixed small epsilon-random floor to cap how much any one signal
can dominate; (3) given how large and consistent this regression is,
seriously consider whether *any* per-step informed bias helps this
particular algorithm at all, vs. whether the JEPA world model's real
contribution to this agent family should be something orthogonal to
per-step action choice entirely (e.g. exact-recall/exploit-on-win layered
on top once a win is found, which this class still doesn't have, rather
than biasing exploration order itself).

**Bottom line:** the pure, training-free `GraphExplorerAgent` remains the
stronger local performer by a wide, now double-backtested margin (46 vs.
34 levels; 0.242 vs. 0.121 mean score). This JEPA integration, as built,
should not be treated as an improvement or submitted -- it's a real,
useful negative result (a specific, well-understood algorithm where
informed per-step bias hurts rather than helps), not a dead end for the
broader idea of combining the two approaches.

## Kaggle competition submission: root cause found, real score obtained

**Current status: the Stage 5 Hypothesis agent has four real, scored,
`SubmissionStatus.COMPLETE` entries in the actual
`arc-prize-2026-arc-agi-3` Kaggle competition:** public score `0.23` on
the first submission (`MAX_ACTIONS=300`), `0.06` on an immediate resubmit
of the exact same kernel version (`MAX_ACTIONS=300`, identical code),
`0.22` on `stage6-score-optimization`'s candidate (`MAX_ACTIONS=900`,
`EPSILON`/`PATCH_SAMPLE_TEMPERATURE` left unchanged after a sweep found
no config improvement -- see `experiments/stage6_score_variance.md`),
submitted 2026-07-17, and `0.16` on `stage6-search-harvest`'s candidate
(the production checkpoint swapped for one retrained on a systematic,
policy-free search-harvested corpus, `MAX_ACTIONS` left at the original
`300` -- deliberately not combined with the `stage6-score-optimization`
budget change, to keep this one variable isolated), submitted
2026-07-20. That first pair is a real finding, not noise to explain
away: `Hypothesis.__init__` seeds `self._rng = random.Random()` with no
fixed seed, so exploration genuinely differs run to run, and the
private/public leaderboard game sampling may also differ between
submissions -- both plausible contributors. The `0.06` run also exactly
matches the unmodified official random-agent control's own score,
meaning on that particular run the full hypothesis-bundle machinery did
no better than picking actions uniformly at random. **This is the
clearest concrete lead for future improvement work**: reducing variance
and improving worst-case reliability likely matters at least as much as
improving best-case peak score.

**Honest read on the `0.22` result: encouraging, but not conclusive on
its own.** It's close to the best prior score (`0.23`) and far from the
worst (`0.06`, the random-agent floor) -- consistent with the local
backtest evidence in `experiments/stage6_max_actions.md`/
`stage6_score_variance.md` (budget=900 dropped zero-completion runs from
2/5 to 0/5 locally). But with only one submission at this config, it's
equally consistent with "MAX_ACTIONS=900 doesn't change much and this
run simply landed in the already-known good part of the 0.06-0.23
variance range" -- the *same* code already demonstrated that range on a
single unchanged config, so a single new data point at a different
config can't cleanly separate "the change helped" from "got a good roll
again." Telling those apart for real would need multiple submissions per
config, which at Kaggle's real limit of **1 submission/day** (see this
doc's step 4 below -- corrected from an earlier, wrong "5/day" claim)
means several more days of one-a-day submissions, not something to
resolve in a single session. Treat `0.22` as "no evidence of regression,
some evidence of improvement" -- a real, honest data point, not a
declared win.

**Honest read on the `0.16` search-harvest result: the same
noise-floor problem, and it cuts the other way this time -- don't
overcorrect into reading it as a regression.** The local evidence behind
this checkpoint was real and carefully verified (`experiments/
stage6_search_harvest.md`: beats production on 18/25 games individually
on changed-patches, not just pooled; wins a matched local scorecard
backtest at n=8, mean score 0.0586 vs 0.0180, 10 vs 5 levels completed).
`0.16` is lower than both `0.23` and `0.22`, which could look like a
regression at a glance -- but it sits almost exactly at the midpoint of
the *old* checkpoint's own already-documented `0.06`-`0.23` range at this
exact `MAX_ACTIONS=300` config, using nothing but run-to-run variance
already proven to exist on identical code. A single submission cannot
distinguish "this checkpoint is worse," "this checkpoint is the same,"
and "this checkpoint is better but landed a below-average roll" from
each other when the noise floor is this wide. Symmetric with the `0.22`
case: treat `0.16` as "no strong evidence either way," not as a
disproof of the local backtest -- the local evidence and the real score
aren't in conflict, they're just not resolvable against each other at
n=1. Getting a real answer needs more same-config submissions on both
checkpoints, which at 1/day is a multi-day undertaking, not a
same-session one.

**Update (2026-07-21): a 5th submission, `0.08`, gives the search-harvest
checkpoint its own n=2 at this same `MAX_ACTIONS=300` config (the first
was `0.16`).** Honest read, same standard applied to every prior point in
this section rather than a double standard now that the number is lower:
the two search-harvest points (`0.16`, `0.08`) span a narrower range than
the *old* checkpoint's own established `0.06`-`0.23` spread, but both
land comfortably *inside* that old range -- so at n=2 per side, this still
does not separate "the search-harvest checkpoint performs differently
(better or worse) than the old one on real scored runs" from "both
checkpoints are drawing from a similar wide noise distribution and we
just haven't sampled enough to tell." The mean of the two search-harvest
points (`0.12`) is, if anything, slightly below the old checkpoint's own
two-point mean (`0.145`) -- **not evidence of improvement on real
submissions yet, despite strong, carefully-verified local evidence**
(18/25 games individually on changed-patches, a clean n=8 local scorecard
win). This is worth sitting with rather than explaining away: the local
backtest signal for this checkpoint has been consistently strong, and
real-world scores so far haven't confirmed it either way -- which is
exactly the kind of gap more same-config submissions exist to resolve,
not a reason to distrust either source on its own. (Separately, and even
more starkly: `stage6-object-identity`'s new checkpoint -- see that
experiment's own section -- has dramatically stronger local evidence
still (object-identity gap +1.20 vs. this checkpoint's +0.079, backtest
win where every run beat every run) and has **zero** real submissions yet.
Worth keeping in mind when deciding where to spend the next scarce
daily submission.)

**Update (2026-07-22): the object-identity checkpoint's first real
submission scored `0.00`** -- `SubmissionStatus.COMPLETE` (not `ERROR`,
so the run wasn't rejected outright), `MAX_ACTIONS=300` unchanged, ref
`54889426`. This is the lowest score of all six real submissions so far,
and notably sits *below* the previously-established floor: the `0.06`
result earlier in this section exactly matched the plain random-agent
control, but `0.00` is lower than that -- worth being honest that this is
a real outlier, not just "another point inside the known noise range,"
without jumping to "something is broken" on n=1 alone.

What argues against a hidden crash/regression: the free kernel test push
(same version, v13) ran with zero tracebacks before this was submitted;
this exact checkpoint produced plenty of nonzero real scores across ~20
local backtest runs in the same-day `EPSILON` resweep (see
`experiments/stage6_object_identity.md`'s follow-up section); and Kaggle
marked the run `COMPLETE`, meaning its own harness didn't flag a fatal
error. What can't be ruled out from here: Kaggle exposes no execution
detail for a `COMPLETE`-but-low-scoring run (the `errorDescriptionNullable`
trick only helps for submissions that errored, not ones that finished and
scored 0), so there's no way to directly confirm *why* from the API alone.
Plausible innocent explanation: 300 actions across up to 110 largely-novel
hidden games is already a tight budget, the established variance floor
was already wide (`0.06`-`0.23` on identical code), and this checkpoint's
much sharper, more confident representations (`+1.20` object-identity
gap, all trained on *local* games' specific color statistics) could
plausibly make it less robust -- not more -- to hidden games whose visual
patterns differ from what it specialized on. That's a real, honest
hypothesis, not a dismissal -- and exactly the kind of thing a second
same-config submission would help distinguish from a genuine outlier
roll. **Treating this as inconclusive at n=1, same as every other new-
checkpoint submission this session, not as a disproof of the strong local
evidence** -- but it's the most concerning single data point yet, and
worth prioritizing a same-config resubmission before spending further
quota on new candidates.

**Follow-up (2026-07-22), a real user catch: "I've never seen a random
agent score 0.0 before -- it would require the agent not moving or
making any change on the game board for every trial."** That's a sharper
objection than "treat it as noise." The heartbeat pattern means any
runtime exception during play falls back to `_safe_fallback_action` --
a genuinely random legal action -- not to a no-op or a crash; if
`_init_models` had failed on every single game, the score should land
near the random floor (`~0.06`), not below it. A literal `0.00` is more
consistent with **zero actions ever being recorded at all** (the
placeholder `submission.parquet` shipping because `main.py` never played
a single game) than with "the agent played and did unusually poorly." So
this needed direct checking, not another hedge. Also worth retracting
outright: the "free kernel test push (v13) ran with zero tracebacks"
point made above is weaker evidence than it sounds -- that push only
ever exercises the notebook's *non-rerun* branch (writes a dummy
`submission.parquet`, never touches the `KAGGLE_IS_COMPETITION_RERUN`-
gated setup/copy/checkpoint-load code), exactly the blind spot that let
the original path-mounting bug survive multiple submissions undetected
earlier in this project. It was never real evidence the gated path
worked for this dataset version.

Built a standalone diagnostic kernel (`kaggle_submission/notebook_diag/`,
reusable going forward, same free-unconditional-cell trick this doc
already documents) that runs *unconditionally* -- no rerun gate -- and
replicates `hypothesis_agent.py: _init_models`'s exact logic against the
live `calamitychasm/jepa-hypothesis-agent` dataset: walks
`/kaggle/input`, copies `jepa/` + `checkpoints/` from the nested dataset
mount path, loads all three checkpoints (`encoder_moe.pt`,
`moe_predictor.pt`, `value_head.pt`) via `load_state_dict`, and runs one
real forward pass through encoder -> predictor -> value head. **Result:
`ALL DIAGNOSTICS PASSED`** -- dataset mount correct, all three checkpoints
load with zero shape/key errors on Kaggle's actual `torch 2.10.0+cpu`
image, game vocab loads with the expected 26 entries, forward pass
produces correctly-shaped output. Also directly diffed this checkpoint's
`encoder_moe.pt` state dict against the production checkpoint's (same
keys, same shapes, zero mismatches) and confirmed `game_vocab_moe.json`
is byte-identical between the two -- ruling out a shape/vocab-size
mismatch as well. This is the same checkpoint files that were live for
the real `0.00` submission (not re-versioned since), so a corrupted
checkpoint or broken dataset mount is now ruled out directly, not just
argued around.

**This retracts the "sharper representations, less robust" hypothesis
above -- it was speculation floated without actually checking the more
basic failure mode first, and it doesn't survive this diagnostic.** What
remains genuinely open, now that setup/checkpoint-loading is confirmed
fine: (1) this diagnostic ran CPU-only and never exercises the real
`main.py`/gateway game loop, so it cannot rule out a GPU-specific or
live-harness-specific failure in the actual scored path; (2) the
project has only ever tested the plain random-agent control *once*
(`0.06`) -- its own run-to-run variance on this same tight 300-action,
~110-largely-novel-game budget has never been measured, so "even a
working random-ish agent can occasionally land on zero completions
across every one of ~110 games" hasn't been ruled out either. Neither of
these is confirmed -- both are just what's left after ruling out the
setup/checkpoint layer directly. A same-config resubmission (or, cheaper
and non-quota-consuming, another instrumented free diagnostic push that
also drives a real `main.py --agent hypothesis` run against a local
mock gateway if one can be improvised) remains the most direct way to
actually resolve this, not further hypothesis generation.

**Update (2026-08-02): the novelty-aware beta override's first real
submission scored `0.09`** -- `SubmissionStatus.COMPLETE`, ref
`55195099`, production checkpoint (the same lineage behind the `0.23`/
`0.06` pair, not a new checkpoint) with `NOVELTY_BETA_CAP=0.15` added and
`MAX_ACTIONS=300` deliberately left at default -- the one variable
isolated, matching every other submission's discipline this session.
This is the sixth real submission at this exact `MAX_ACTIONS=300`
config, joining production's own `0.23`/`0.06`, search-harvest's
`0.16`/`0.08`, and object-identity's `0.00` -- `0.09` sits squarely
inside that established `0.00`-`0.23` noise band, close to
search-harvest's own `0.08`. **Same standard as every prior first
submission: this is n=1 for this specific change, and cannot be told
apart from ordinary run-to-run variance on a single data point** --
it neither confirms nor refutes the local backtest's own strong signal
(n=8, favored the change on every metric: mean score, mean levels, total
levels, and the outlier-resistant games-completed comparison). A larger
local backtest (20-30 repeats) was already in progress at the time of
this submission, independent of it -- that result will sharpen the local
evidence regardless of what one real score does or doesn't show. Telling
whether this specific change moves real scores would need more
same-config submissions, which at 1/day is a multi-day undertaking, not
something this single data point can resolve on its own.

**Update (2026-08-28): 9th real submission, ref `55843700`, testing the
`tried_actions` dead-end-avoidance fix (see
`experiments/stage6_transition_graph_health.md`) -- `MAX_ACTIONS=300`
unchanged, production checkpoint lineage unchanged, TTA/novelty-beta both
off (neither is on master), the dead-end fix is the one isolated
variable. Resolved `SubmissionStatus.COMPLETE`, public score `0.15`.**
Local evidence behind this one was honestly mixed, not a clean local win
being validated: a 5-fold, n=30 agent-level backtest found the fix's
early folds (+3, +3 levels) decayed to noise by fold 5 (final cumulative
+3 levels, 29->32, and a *worse* avg-actions-to-first-completion) -- the
same decay-to-noise shape as the novelty-aware-beta result already
documented above. What's genuinely solid, independent of that agent-level
noise, is the component-level measurement: `Hypothesis` never inherited
`Memory`'s "avoid re-trying a known-unproductive action from this exact
state" behavior when it was built in Stage 5, and restoring it cut
exact-repeat waste 5-10x on every game tested (direct replay measurement,
not agent-level noise). Submitted anyway on the same reasoning as prior
ambiguous-local-result submissions in this project's history: it's
low-risk (falls back to the full candidate set when nothing untried
remains, doesn't touch ACTION6), and real submission data is the only
thing that's ever actually resolved this project's noisy-small-sample-
backtest problem before.

**Real-score read: `0.15` sits squarely inside the established
`0.00`-`0.23` noise band** -- above the floor (`0.00`, `0.06`, `0.09`x2),
below the ceiling (`0.23`), close to search-harvest's own `0.16`. Same
standard as every prior first-submission-for-a-new-candidate result in
this project: n=1 cannot be told apart from ordinary run-to-run variance
on identical code. This neither confirms nor refutes the fix helps real
play -- but it's a coherent result, not a contradictory one: the local
5-fold backtest already said "inconclusive," and this real submission
says the same thing independently, rather than the two disagreeing.
Given the fix is low-risk and the component-level improvement is real and
directly measured, keep it regardless -- there's no evidence here or
locally that it hurts.

**Correction (2026-08-28, same day): the `55843700` submission above did
NOT use the true production checkpoint -- it used a `stage6-game-holdout`
fold-1 experimental checkpoint that had silently been sitting in
`checkpoints/` in its place.** Found while a sibling investigation
(`experiments/stage6_rollout_compounding_error.md`) loaded
`checkpoints/moe_training_meta.json` and noticed `"exclude_games":
["r11l","bp35","m0r0","tr87","ka59"]` and a 21-entry `game_vocab_moe.json`
(20 ARC games + minigrid) -- the true production lineage has **26**
entries (all 25 local games + minigrid, `n_games: 26`, no `exclude_games`
field, dated 2026-07-08/09, matching the original Stage 4 item 6
training). Since `checkpoints/` is gitignored (never version-controlled),
there's no direct history of *when* this substitution happened or what
overwrote it -- the wrong files are dated 2026-08-11, well before this
session started, so it predates this conversation, not something done in
it. Recovered the true production checkpoint from a still-intact sibling
worktree (`.claude/worktrees/agent-a490cf638b9f81638/checkpoints/`,
verified by exact file match on the 2026-07-08/09 dates and the 26-entry
vocab) and restored it into `checkpoints/`; the wrong fold-1 files are
preserved at `checkpoints_holdout_fold1_backup/` rather than deleted, in
case anything needs to reference that specific experiment later.

**What this means for the `0.15` score: it wasn't testing "dead-end fix
on production," it was testing "dead-end fix on a checkpoint trained on
20 of the 25 local games instead of 25"** -- two variables conflated in
one submission, not the clean isolated-variable test the write-up above
assumed. This doesn't retroactively make `0.15` meaningless (it's still
a real score for the code that actually ran), but it can no longer be
read as "the dead-end fix, holding the checkpoint constant at production"
-- that comparison hasn't actually been made yet. **A clean resubmission
(dead-end fix on the now-restored true production checkpoint, properly
isolated) is the right next use of a daily submission slot**, not
assumed to reproduce `0.15`.

**Worth doing before trusting any local eval, benchmark, or diagnostic
number produced anywhere in this project between whenever the
substitution happened and 2026-08-28**: anything that loaded
`checkpoints/encoder_moe.pt`/`moe_predictor.pt`/`value_head.pt`/
`game_vocab_moe.json` by their default path during that window (not a
`checkpoints_ablation/`-style explicit alternate path) was silently
evaluating the fold-1 holdout checkpoint instead of production. No
systematic audit of which specific results in this document were affected
has been done -- flagging the risk rather than claiming it's been ruled
out.

Everything needed to reproduce the submission from scratch on a new
machine is in `kaggle_submission/` (checked into git) plus the steps
below. This section is the reproduction guide; the dated blow-by-blow
(useful if the submission mechanism breaks again and needs
re-diagnosing) follows it.

### What actually broke, and what didn't

Every real scored submission attempt failed identically (Kaggle's own
generic `"A system error. Please try resubmitting..."` message, no
further detail available via the API or CLI) across multiple, materially
different fixes: hardened subprocess-based setup, a gateway-wait timing
overlap, `enable_gpu` toggling, a top-level `try/except` "heartbeat"
around `choose_action`/`is_done`, and writing a placeholder
`submission.parquet` before any risky setup ran. None of those fixed it,
because none of them addressed the actual bug -- but all are real,
worth-keeping hardening and are described below.

**The root cause: `kernel-metadata.json`'s `dataset_sources` mounts a
Kaggle dataset at `/kaggle/input/datasets/<owner>/<dataset-slug>/`, not
`/kaggle/input/<dataset-slug>/`.** Every version of the submission
notebook's setup cell referenced the un-nested path
(`/kaggle/input/jepa-hypothesis-agent/...`) for our own dataset, while
correctly using the *nested* form for the competition's own attached data
(`/kaggle/input/competitions/arc-prize-2026-arc-agi-3/...` -- that one was
right from the start, copied from the official sample notebook). Every
real scored run was hitting an uncaught `FileNotFoundError` on the very
first `shutil.copytree` call in cell 1 -- before `main.py` ever opened a
single game -- which explains why none of the other fixes ever had a
chance to matter: the notebook was crashing before reaching any of that
code.

**How this was actually found:** a *free* (non-scored) `kaggle kernels
push` with a diagnostic cell that ran unconditionally -- outside the
`if os.getenv('KAGGLE_IS_COMPETITION_RERUN')` gate that hides all real
setup logic during an ordinary test push -- printing `torch.__version__`
and `os.walk("/kaggle/input")`, then attempting `torch.load(...)` on each
checkpoint directly. Kaggle does not expose the actual execution log from
a real scored competition rerun (`kaggle kernels output` only ever
returns the *last test push's* non-rerun output, confirmed by checking it
against a known-different real submission's result) -- so this
"unconditional diagnostic cell in an otherwise-gated setup script" pattern
is the only way found this session to get real signal out of the scored
environment without spending a submission. Worth reusing directly if this
ever needs debugging again: temporarily hoist a print/probe above the
`KAGGLE_IS_COMPETITION_RERUN` check, push (free), pull the log, revert.

**Also checked and ruled out, despite looking suspicious at first:** a
torch version mismatch. Kaggle's notebook image ships `torch==2.10.0`;
these checkpoints were trained locally under the `torch==2.12.1` pin (see
this doc's own Environment setup section). Confirmed directly via the
same diagnostic cell that `torch==2.10.0` loads all three
`torch==2.12.1`-saved state dicts (`encoder_moe.pt`, `moe_predictor.pt`,
`value_head.pt`) with zero errors -- state dicts are plain tensor
`OrderedDict`s and are forward/backward compatible across this version
gap. Not the bug.

### Other real hardening added along the way (kept, not the root cause)

- **Heartbeat pattern** (`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`):
  `choose_action` and `is_done` each wrap their real logic in a top-level
  `try/except`, falling back to a new `_safe_fallback_action` (a random
  legal action) on any exception instead of letting it propagate and kill
  `main.py` for every remaining game. Directly motivated by a real,
  previously-unguarded crash path: `jepa/grid.py`'s `grid_to_tensor`/
  `patch_change_mask` hard-`raise ValueError` if any frame exceeds the
  hardcoded 64x64 `CANVAS` -- our 25 local public games are all exactly
  64x64 by coincidence, so this could never have been caught by any local
  testing, only by a hidden competition game with a different shape.
- **`__init__` hardening**: model construction and checkpoint loading
  (`_init_models`) now run inside a `try/except` in `__init__`; on any
  failure, `self._init_failed = True` and `choose_action` immediately
  routes to `_safe_fallback_action` for that game rather than crashing
  agent construction itself (which happens before `choose_action`'s own
  try/except ever gets a chance to run).
- **`DIAG_MODE`** (`HYPOTHESIS_DIAG_MODE=1` env var): makes
  `choose_action` always return a safe random action while still running
  `__init__` in full. Built specifically to isolate "does setup/checkpoint
  loading work" from "does the real Q-scoring/MoE inference path work" --
  kept as a permanent, zero-cost-when-off debugging lever for next time.
- **Placeholder `submission.parquet`**, written before any risky setup
  runs in the rerun branch. rules.md: submissions are "auto-generated as
  long as the agent acts on the games" -- meaning a total setup crash
  *before* `main.py` plays a single game would otherwise leave no
  submission file at all. This is pure insurance, not a fix for anything
  specific; kept because it's free and strictly safer.
- **Gateway-wait/local-setup overlap**: the gateway-readiness `curl`
  check now runs as a background `subprocess.Popen` (retry window shrunk
  600s -> 90s) while file-copying/importing happens concurrently, joining
  on it only right before `main.py` needs the gateway up. Motivated by
  Kaggle's own guidance that a rerun errors out if the agent doesn't make
  its first move within roughly 15 minutes of container start -- not the
  actual bug this time, but a real latency improvement worth keeping.

### Full reproduction steps (new machine, from scratch)

**1. Stage the Kaggle dataset contents.** Needs: the whole `jepa/`
package (small, ~50KB zipped -- `jepa/__init__.py` and
`jepa/data/__init__.py` are both empty, so no eager imports of
unavailable deps like `minigrid`/`gym_sokoban` happen just from copying
the package), exactly four checkpoint files (`checkpoints/encoder_moe.pt`,
`checkpoints/moe_predictor.pt`, `checkpoints/value_head.pt`,
`checkpoints/game_vocab_moe.json` -- *not* the full `checkpoints/` dir,
which also has non-MoE-era files from earlier stages),
`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py`, and
`kaggle_submission/dataset-metadata.json` (already in this repo). Copy
all of these into one staging directory matching that layout (`jepa/`,
`checkpoints/`, `hypothesis_agent.py`, `dataset-metadata.json` all as
siblings).

**2. Push the dataset:**
```
kaggle datasets create -p <staging-dir>              # first time only
kaggle datasets version -p <staging-dir> --dir-mode zip -m "<message>"   # every update after
```
`--dir-mode zip` is required -- without it, `kaggle datasets create/version`
silently *skips* folder arguments (`jepa/`, `checkpoints/`) with a
"Skipping folder: X; use '--dir-mode'" message and only uploads loose
files, which is a second, separate way to end up with missing files at
the mount path (different from this section's main bug, but easy to
conflate with it -- always double check the upload log names
`jepa.zip`/`checkpoints.zip` as separate uploaded files, not raw folder
skips). Wait for `kaggle datasets status <owner>/<slug>` to report
`ready` before pushing a kernel that depends on it -- versioning is
asynchronous.

**3. Push the kernel** (`kaggle_submission/notebook/`, already in this
repo -- `kernel-metadata.json` + `arc3-hypothesis-agent-submission.ipynb`):
```
kaggle kernels push -p kaggle_submission/notebook
```
This is free and safe to run repeatedly. It only exercises the notebook's
*non-rerun* branch (writes a dummy `submission.parquet`, doesn't touch
any of the `KAGGLE_IS_COMPETITION_RERUN`-gated setup/agent code) -- so it
validates dependency installation and catches Python syntax errors, but
cannot by itself catch a bug like this section's root cause. Check
`kaggle kernels status <owner>/<kernel-slug>` until `COMPLETE`/`ERROR`,
then `kaggle kernels output <owner>/<kernel-slug> -p <dir>` to pull
`<kernel-slug>.log` and confirm no errors before spending a real
submission.

**4. Submit for real scoring** (consumes the daily submission quota --
**correction (2026-07-16): the real limit is 1/day, not "5/day" as an
earlier version of this doc claimed.** That "5/day" figure was wrong (or
the limit changed) -- confirmed directly this session by hitting a 400 on
a second same-day submission attempt with the real server error body
(the plain `kaggle competitions submit` CLI swallows this detail --
see the API-body-extraction trick two paragraphs down, which is how this
was found): `"Submission not allowed:  Your team has used its daily
Submission allowance (1) today, please try again tomorrow UTC (2.6 hours
from now)."` The reset is a fixed daily UTC boundary (reads as UTC
midnight from the "2.6 hours" phrasing observed at ~21:22 UTC) rather
than a rolling 24h window from your last submission -- budget one real
submission per UTC day, and treat every one as scarce, not five):
```
kaggle competitions submit -c arc-prize-2026-arc-agi-3 -k <owner>/<kernel-slug> -v <version> -f submission.parquet -m "<message>"
```
Check status with `kaggle competitions submissions -c arc-prize-2026-arc-agi-3 --csv`
(`SubmissionStatus.PENDING` can take hours to resolve to `COMPLETE`/
`ERROR` -- this is real queued compute, not an instant check). For more
detail than the CLI table shows on an error (though still not much --
Kaggle's own generic message is usually all that's available), hit the
API directly:
```
curl -s -u <username>:<key> "https://www.kaggle.com/api/v1/competitions/submissions/list/arc-prize-2026-arc-agi-3"
```
and read `errorDescriptionNullable`. **This only helps for a submission
that was actually created and then failed/errored later** (e.g. the DIAG
run) -- it's useless for a `400 Client Error: Bad Request` that happens
at the `CreateCodeSubmission` call itself (like the daily-quota case
above), since no submission record ever gets created for those. For
*that* case, the plain `kaggle competitions submit` CLI prints only the
bare `400 Client Error: ...` line with zero body -- call the Python API
directly instead to get the real JSON error message:
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
try:
    api.competition_submit_cli(file_name='submission.parquet', message='...',
                                competition='arc-prize-2026-arc-agi-3',
                                kernel='<owner>/<kernel-slug>', version='<version>')
except Exception as e:
    print(e.response.status_code, e.response.text)  # the real error body
```
(Note the kwarg is `version`, not `kernel_version` -- `competition_submit_cli`'s
actual signature differs from what you might guess from the CLI's own
`-v/--version` flag name.)

**5. If it errors again with no useful detail:** don't guess-and-resubmit
blindly (burns quota fast). First run a control test -- push the
*unmodified* official sample notebook (if the competition provides one)
as a completely separate kernel and submit it too. If the control also
fails, the problem is platform/account-side, not this codebase. If the
control succeeds (as it did this session, `SubmissionStatus.COMPLETE`,
public score `0.06`), the bug is confirmed to be in this notebook/agent
specifically -- use the free unconditional-diagnostic-cell trick described
above to narrow it down without spending more of the daily quota.

## Gotchas learned the hard way (don't re-discover these)

- **(2026-09-05) All three real submission notebooks had silently shrunk the
  gateway-readiness wait from the official reference's proven 600s down to
  90s -- found only by diffing our notebook against `arcprize/
  ARC-AGI-3-Kaggle-Starter` (the official starter kit) and Kaggle's own
  `inversion/arc3-sample-submission-stochastic-goose` sample notebook,
  something never done before across this whole submission debugging saga.**
  Every one of `kaggle_submission/`, `kaggle_submission_graph_explorer/`, and
  `kaggle_submission_graph_explorer_learned/`'s notebooks ran the
  gateway-sidecar readiness check (`curl ... http://gateway:8001/api/games`)
  with `--retry-max-time 90`, reasoned about at the time as "if the gateway
  isn't reachable in 90s it's very unlikely to become reachable in the time
  remaining anyway" (see git history / this file's own prior wording). The
  official reference notebook uses `--retry-max-time 600` -- a full 10
  minutes -- confirmed by fetching its `scripts/build_notebook.py` directly.
  This matters because **main.py's own initial `/api/games` check has only a
  10s timeout and no retry loop of its own** (confirmed directly by reading
  `main.py`) -- if the real gateway sidecar ever takes longer than 90s to
  come up under real competition load (up to 110 games' worth of gateway
  infrastructure spinning up, concurrently with every other team's own
  kernel), our shortened window gives up, logs a warning, and launches
  `main.py` against a gateway that may still not be ready -- with nothing
  downstream able to recover, silently producing a near-empty run. This is
  **exactly the kind of path no free diagnostic push can ever exercise or
  disprove** (the `KAGGLE_IS_COMPETITION_RERUN` gate that runs this code
  only fires during a real scored rerun), so it sat undiscovered through
  every one of this project's real submissions across all three agent
  families -- and would plausibly explain a real, if hard-to-quantify,
  downward bias/added variance across every one of them (Hypothesis's own
  0.00-0.23 spread; GraphExplorerLearnedAgent's disappointing 0.05-0.15
  against a 3.76x local backtest edge), on top of the separately-confirmed,
  well-documented ARC-AGI-3 public/private generalization gap (see the
  "Kaggle competition submission" section's own external-research
  discussion). **Fixed in all three real submission notebooks and the
  `_build_learned_agent_notebook.py` builder script**: `--retry-max-time`
  restored to `600`, keeping the background-`Popen`-overlap-with-setup
  technique (strictly faster than the official's sequential wait at the
  same worst-case budget, never slower) -- pushed as free kernel updates
  (hypothesis v18, pure graph-explorer v3, graph-explorer-learned v3).
  **Correction (2026-09-07): it WAS re-submitted -- ref `56043778`,
  2026-09-06, `SubmissionStatus.COMPLETE`, public score `0.18`**, the
  graph-explorer-learned agent's best real score by a clear margin (prior:
  0.08, 0.15, 0.05). At n=1 against this project's documented 0.00-0.25
  noise band that is not proof the fix helped -- the case for it remains
  mechanistic (matches the official reference; removes a silent-failure
  path no free diagnostic push can reach), not statistical. **Worth doing again if this project's
  submissions keep underperforming their local backtests**: diff against
  the official starter kit early, not as a last resort -- it caught a real,
  concrete, previously-unconsidered discrepancy in under 30 minutes that a
  much longer independent debugging saga (GPU/P100, quota, control tests)
  never surfaced, because that saga was entirely internally-generated
  hypotheses rather than a reference-implementation diff.
- **Kaggle's `enable_gpu: true` can provision a Tesla P100 (CUDA capability
  sm_60), which the bundled PyTorch build's compiled kernels don't support
  (minimum sm_70)** -- confirmed directly via a diagnostic push
  (`kaggle_submission_graph_explorer_learned/notebook_diag/`) while
  debugging `GraphExplorerLearnedAgent`'s submission errors: with
  `enable_gpu: true`, every real tensor op on the GPU throws
  `torch.AcceleratorError: CUDA error: no kernel image is available for
  execution on the device`, even though `torch.cuda.is_available()`
  returns `True` (a GPU is physically present, just unsupported) --
  `jepa/device.py: get_device()`'s own "CUDA if available" logic has no
  way to detect this ahead of time. This was directly, empirically tested
  by TOGGLING `enable_gpu` between `true`/`false` on the exact same code
  and checkpoint -- `false` (CPU-only) loads and runs cleanly every time;
  `true` fails deterministically on this hardware. Hypothesis's own
  `enable_gpu: true` submissions have apparently never hit this (different
  GPU pool assignment, or their larger model/checkpoint happens not to
  trigger it at the exact op that fails) -- don't assume `enable_gpu: true`
  is safe just because another submission in this project uses it
  successfully elsewhere. If a future submission needs real GPU compute,
  check the actual provisioned card's compute capability first (or design
  the fallback path to be genuinely acceptable, not just "shouldn't
  happen") rather than assuming the competition's advertised RTX 6000-
  class hardware pool is what any given kernel actually gets.
- **`checkpoints/` is gitignored and can silently hold a stale
  experimental checkpoint in place of production, with zero warning and
  no version history to catch it.** Discovered 2026-08-28: `checkpoints/
  encoder_moe.pt`/`moe_predictor.pt`/`value_head.pt`/`game_vocab_moe.json`
  had a `stage6-game-holdout` fold-1 checkpoint sitting in them (21-game
  vocab, `exclude_games` for 5 local games) instead of the true 26-game
  production lineage, apparently since around 2026-08-11 -- long enough
  that a real Kaggle submission (`55843700`) and at least one local
  diagnostic ran against it believing it was production, before anyone
  noticed. Nothing enforces that `checkpoints/`'s contents match what
  CLAUDE.md documents as current -- it's just whatever file was last
  copied there, from any experiment, by any session. **Before trusting
  any eval/diagnostic/submission that loads checkpoints by their default
  `checkpoints/` path, spot-check `game_vocab_moe.json`'s entry count
  (26 = true production; anything else is a different experiment's
  artifact) and `moe_training_meta.json` for an unexpected `exclude_games`
  field** -- this takes seconds and would have caught the issue
  immediately. If this recurs and the true checkpoint isn't obviously
  sitting in the current worktree, check sibling worktrees under
  `.claude/worktrees/*/checkpoints/` before assuming it's lost -- that's
  where the real one was recovered from this time.
- **`kaggle datasets version -p <dir> --dir-mode zip` (CLI 2.2.3, Windows)
  fails on plain top-level *files* in the staging dir with `[Errno 2] No
  such file or directory` pointing at a mangled path like
  `...\.kaggle/uploads\C_/kagstage_hypothesis_agent.py.json`** --
  directory args (which get zipped first) upload fine; a bare file like
  `hypothesis_agent.py` or `dataset-metadata.json` sitting directly in the
  staging dir hits a bug in the CLI's per-file upload-tracking path
  construction on Windows (mixes `/` and `\`, produces a path whose
  parent directory doesn't exist). Not a staging-path-length issue --
  reproduced identically from both a long nested scratchpad path and a
  short one (`C:\kagstage`). **Workaround: `mkdir` the missing directory
  the error names** (in this case `%TEMP%\.kaggle\uploads\C_\`) before
  retrying -- the upload then succeeds normally. Check the exact failing
  path in the error message first since the mangled directory name (`C_`
  here) may differ by drive letter/staging path.
- **`ARC-AGI-3-Agents/recordings/` (gitignored, fully regenerable) grows
  without bound and will eventually fill the disk if nothing ever cleans
  it up.** Hit this directly (2026-07-17): a backtest sweep and a
  self-play data-harvest running concurrently pushed it to **20GB across
  1,300 files**, dropping the C: drive to 0.3GB free and causing a real
  mid-run failure (`OSError: [Errno 28] No space left on device` inside
  `run_scorecard.py`'s own JSON write, silently dropping one repeat's
  result -- the file existed but was 0 bytes, not merely missing, so a
  naive glob-and-parse over `logs/scorecards/*.json` will crash rather
  than skip it; check for and delete zero-byte scorecard files before
  summarizing). Fixed by moving everything older than 15 minutes (to
  avoid touching files an active run still had open) to a secondary
  drive: `Get-ChildItem <recordings-dir> -File | Where-Object
  {$_.LastWriteTime -lt (Get-Date).AddMinutes(-15)} | Move-Item
  -Destination <archive-dir>` -- freed 18.8GB in one pass. If this
  recurs, the durable fix (not yet done) would be a directory junction
  pointing `recordings/` at the secondary drive so future writes land
  there automatically, but that needs no active process holding a handle
  into the directory when you convert it -- safest done between runs, not
  mid-sweep. On this dev box specifically, `E:` is a second drive with
  real free space (`Get-PSDrive` to check current free space on any
  drive) -- worth checking before assuming C: is the only option.
- **The harness's anonymous `ARC_API_KEY` expires within roughly a day,
  not just between machines/sessions.** Hit this repeatedly across this
  project's later sessions -- every game-listing call returns HTTP 401,
  `main.py` silently proceeds with an empty game list, and every agent
  run in that state produces a technically-valid recording file with zero
  real actions and zero levels (which can look exactly like a real agent
  regression if you're mid-comparison, not an infrastructure issue).
  Refresh via `curl https://three.arcprize.org/api/games/anonkey` and
  update `.env`'s `ARC_API_KEY` -- and if a comparison run ever comes back
  suspiciously empty (0 runs found, or every agent scoring 0 across the
  board), check for this *first*, before assuming a code change broke
  something.
- **A deterministic `argmax` over a salience/variance map will default to
  the same fixed index every time the map is flat or near-flat, not a
  "no preference" no-op.** `Hypothesis`'s original click-location
  selection (`patch_var.argmax()`) looked reasonable in isolation, but
  produced a strong, consistent bias toward one specific repeated click
  location on every game tested, because ties (or near-ties) resolve to
  the same low patch index every single time rather than exploring.
  `Curiosity` had already solved this exact problem for itself
  (temperature-weighted softmax sampling over patches, plus a uniform-
  random pixel within the chosen patch, not always dead-center) --
  worth checking whether an existing, already-debugged agent in the same
  codebase already solved a given problem before re-deriving a
  from-scratch solution that reintroduces it.
- **When two things being compared diverge on a specific subset of cases,
  trace *that subset specifically* -- an aggregate trace or pooled metric
  can fail to isolate a bug that's fully explanatory on the subset where
  it actually matters.** A live trace across a full 25-game sweep missed
  a real bug that became immediately obvious once narrowed to just the
  handful of games where two agents' results actually diverged (see
  Stage 5's "follow-up 3" section above) -- the bug's effect was real in
  the pooled numbers but not attributable to any one mechanism until
  isolated to the right subset.
- **A new synthetic data source's action space must fit inside
  `jepa/models/predictor.py`'s `NUM_ACTIONS=8`** (shared across every
  data source's action embedding, sized for ARC-3's 8 actions -- MiniGrid's
  7 fit without changes). Sokoban's native action space is 9 (see Stage 4
  item 8) and silently produced a raw id of 8 the first time a random
  rollout happened to sample it -- the resulting out-of-bounds embedding
  lookup doesn't fail at data-generation time (plain Python list indexing
  never checks against `NUM_ACTIONS`), it fails later, deep inside a CUDA
  kernel, on whatever training batch first happens to include that sample
  (`CUDA error: device-side assert triggered`, `vectorized gather kernel
  index out of bounds`). If you add another data source, sanity-check
  `max(action_ids) < NUM_ACTIONS` on its generated transitions *before*
  a training run, not after a confusing CUDA crash.
- **A `MemoryError` inside a DataLoader worker's `pickle.load` can be a
  full-disk symptom, not a real memory shortage.** On Windows, an
  almost-full disk can prevent the pagefile from growing, turning
  ordinary memory-pressure moments into hard failures instead of graceful
  paging. Hit this mid-training during the Stage 4 Sokoban ablation (see
  that section) with 54MB free out of 931GB -- the crash (and a
  crash-loop of repeatedly respawning, dying workers) cleared up
  completely after freeing disk space, with zero code changes. Check
  `df -h` before assuming a `MemoryError` means RAM.

- **CRITICAL (2026-07-08): `ARC-AGI-3-Agents/agents/agent.py`'s
  `_convert_raw_frame_data` never copied `raw.action_input` into the
  `FrameData` it constructs** -- every frame recorded via `append_frame`
  (i.e. every line of every `*.recording.jsonl` file) therefore had
  `action_input.id == 0` (`ActionInput()`'s default, which happens to be
  `GameAction.RESET`) *regardless of what action was actually taken*.
  `FrameDataRaw` (the object returned by `arc_env.step()`) has the correct
  `action_input` all along -- it was just dropped in the conversion.
  Confirmed by regenerating a recording before vs. after the fix: action
  id distribution went from 100% `0`s to a real, roughly-even spread
  across all 8 action ids. One-line fix: add `action_input=raw.action_input`
  to the `FrameData(...)` call in `_convert_raw_frame_data`.
  - **This silently corrupted every local-recording-derived training
    signal that depends on `action_input`** -- `jepa/data/trajectories.py`
    (Stage 1's local transitions) and `jepa/data/sequences.py` (Stage 3's
    episode sequences, which use *only* local recordings, no external
    data) both read `action_input.id` to get the action taken. Before this
    fix, every such transition looked like "action=RESET" to the training
    code, no matter what was actually pressed -- a predictor trained on
    that data structurally cannot learn real action-conditioned dynamics
    from the local corpus, because the action signal was constant noise.
  - **This directly explains Stage 1's original "can't beat identity"
    result and this session's first Stage 3 recurrent-predictor attempt.**
    After the fix (regenerate local recordings, retrain both), Stage 1's
    milestone check went from FAIL to a clean **PASS (+29.2% on changed
    patches)**, and the recurrent predictor went from ~identity-parity to
    **+21.3%**. See the Status sections below for the corrected numbers --
    this fix is the single highest-impact change made in this session,
    and it means the original Stage 1 "plateaued at -8.7%, likely an
    architecture/data-scarcity limit" framing (this doc's own prior
    iteration history, and the "pivot to Stage 2" recommendation built on
    it) was **built on partially/wholly corrupted local data, not a
    genuine ceiling.** The external-data experiments (item 7 in the old
    history) were *not* affected by this bug (the Kaggle `arc-3-logs`
    dataset has its own correct `action` field, unrelated to this
    framework's recording path) -- only the local-recordings-derived
    portions of training/eval were.
  - If you ever add a new local-data-dependent pipeline, sanity-check the
    action distribution first (`Counter(t[1] for t in transitions)` from
    `trajectories.py`, or equivalent) -- an all-one-value distribution is
    exactly this bug (or a reintroduction of it) and will silently produce
    a garbage-in-garbage-out training run that still runs to completion
    without erroring.

- **The anonymous `ARC_API_KEY` in `ARC-AGI-3-Agents/.env` can expire
  between sessions**, even in fully offline mode -- `main.py` always hits
  the *real* `{ROOT_URL}/api/games` endpoint to get the game list before
  anything else happens (offline mode only affects gameplay stepping, not
  this initial listing call), and an expired/invalid key makes that
  request return HTTP 401. When it does, `main.py` logs the 401, gets an
  empty game list, and exits immediately -- every agent run in that state
  silently produces a fully-valid-looking recording file with zero
  actions and zero levels, which can look exactly like a real agent
  failure if you're mid-debugging something else (this happened while
  evaluating Stage 5 -- a first "0 levels completed across 200 runs"
  result briefly looked like a hypothesis-bundle bug before the real
  cause turned out to be an expired key). Get a fresh one with
  `curl https://three.arcprize.org/api/games/anonkey` and update
  `ARC_API_KEY` in `.env`; a quick single-game run is enough to confirm
  it's fixed (watch for a nonzero total action count instead of an
  immediate "No games available to play" error in the log).
- **A Kaggle `dataset_sources` attachment mounts at
  `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`**
  -- unlike a competition attachment, which *does* mount at the
  un-nested-looking `/kaggle/input/competitions/<comp>/` (easy to
  pattern-match from and assume datasets work the same simpler way, which
  is exactly the mistake that cost this project every real competition
  submission for a full debugging session -- see "Kaggle competition
  submission" above for the full story and the free-diagnostic-cell trick
  that found it).
- **`.gitignore` `data/` pattern (no leading slash) matches any directory
  named `data` anywhere in the tree**, including `jepa/data/` (real source
  code, not the gitignored top-level Kaggle/ARC-1/2 download cache). Fixed
  by scoping to `/data/`, `/checkpoints/`, `/logs/`, `/runs/` with a
  leading slash. If you add new top-level dirs to gitignore, scope them
  the same way or check `git check-ignore -v <path>` before assuming a new
  file got tracked.
- **`pip install -r requirements.txt` without version pins fails** on the
  langchain/langgraph/smolagents/openai stack with `resolution-too-deep`
  (pip's resolver gives up). The current pins were copied from
  `ARC-AGI-3-Agents/uv.lock`'s already-resolved graph -- don't loosen them.
- **`arc-agi` (PyPI package) requires `pillow>=12.1.1`**, but
  `ARC-AGI-3-Agents/uv.lock` pins pillow 11.3.0 for a different reason
  (that repo doesn't depend on `arc-agi`'s pillow floor the same way once
  extra deps resolve) -- current pin is `pillow==12.3.0`, keep it there.
- **A vendored template had a real bug**:
  `ARC-AGI-3-Agents/agents/templates/langgraph_thinking/vision.py`
  referenced `PIL.ImageDraw.Coords`, which doesn't exist in any Pillow
  version. Fixed via `from __future__ import annotations` (defers
  annotation evaluation) rather than editing the type away -- if this
  breaks again after a Pillow upgrade, that's why.
- **Piping a background command through `tee` to a bad path silently kills
  the pipeline** (tee's permission-denied error can propagate and end the
  whole command with exit 1, even though the actual Python training
  process sometimes kept running to completion anyway in observed cases --
  don't rely on that). Just let `run_in_background` capture stdout
  directly; don't pipe through `tee` to a path outside the scratchpad dir.
- **EMA-target-vs-online-encoder asymmetry will bias any "identity
  baseline" comparison you compute *during* training** if you encode
  `frame_t` with the online encoder and `frame_t+1` with the lagging EMA
  target -- the resulting gap is partly real signal, partly just
  online/EMA weight drift. For a fair comparison (predictor vs identity),
  encode both sides of the comparison with the *same* encoder weights.
  This is exactly the bug in iteration 1 above.
- **`train_predictor.py`'s `num_workers=4` DataLoader workers each fork a
  full copy of the in-memory transitions list** (Windows uses spawn, not
  fork, so each worker process re-imports and re-pickles the dataset
  rather than sharing memory) -- with both the train and val loaders using
  `persistent_workers=True`, that's 8 worker processes, each holding its
  own ~1.9GB copy of a ~55k-transition combined corpus (~15GB total, on
  top of the ~2-3GB main process). Fine on a 32GB box with a few GB to
  spare, but watch free RAM (`Get-CimInstance Win32_OperatingSystem`) if
  training on a smaller machine or mixing in a much larger external corpus
  -- drop `num_workers`/`persistent_workers` on the val loader specifically
  first (it only iterates once per epoch, so the parallelism matters far
  less there) if memory gets tight.
- **A CPU-bound `num_workers=0` DataLoader can quietly dominate wall-clock
  time even with a working GPU.** A first `--external-per-game 2000` run
  (56k transitions, 60 epochs) ran for 45+ minutes with a healthy-looking
  GPU (`jepa.benchmark throughput` measured 16x GPU vs CPU for the model
  compute alone) because `TransitionDataset.__getitem__` builds the
  one-hot tensors per-sample in the single main process. Confirmed via
  `Get-Process | Select CPU` showing one process pegged at several
  CPU-seconds per wall-second (BLAS/OpenMP threading inside a single
  Python process, not real multiprocessing) rather than GPU utilization.
  Killed and restarted after adding `num_workers=4` -- watch for this
  symptom (long runtime, `nvidia-smi` showing low utilization, one bloated
  Python process) as the tell that data loading, not the model, is the
  bottleneck.
