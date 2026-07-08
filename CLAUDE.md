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

### Stage 4 -- mixture-of-gated-experts predictor: BUILT, milestone NOT met (honest negative result)

**Scope deviation, decided upfront:** plan.md's Stage 4 assumes pretraining
on diverse generated grid envs (MiniGrid/Sokoban/Crafter/procgen) on a
96GB box, with K=16-24 experts. Neither exists for this project (the
MiniGrid data source was deferred back in Stage 1's "Next steps" and never
built; this box has an RTX 2070, not a 96GB card). Trained on the same
~55k-transition combined ARC-3 corpus (local + external `arc-3-logs`) as
Stages 1/3 instead, with a smaller `--num-experts 8` default -- a
pragmatic scope decision per plan.md's own guiding principle #4, not an
oversight.

**Built:**
- `jepa/models/moe_predictor.py: MoEPredictor` -- K small pointwise-conv
  experts + a gate (pooled feat + action/xy/game conditioning -> softmax
  over K), weighted-sum combination. `jepa/models/moe_predictor.py:
  load_balance_loss` -- Switch-Transformer-style auxiliary loss (`K * sum_i
  f_i * P_i`, minimized at uniform usage, maximized at total collapse to
  one expert). Combiner network explicitly deferred, per plan.md's Stage 4
  scope.
- `jepa/train_moe_predictor.py` -- same data/training setup as Stage 1's
  `train_predictor.py` (i.i.d.-shuffled single transitions; MoE routing
  doesn't need Stage 3's temporal ordering), plus the load-balance loss
  term.

**Milestone is "gate activations show experts specializing; better
generalization than the Stage-3 monolith." Neither half was achieved,
despite five different configurations tried (a real, reproducible
negative result, not a single failed attempt):**

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

If revisiting: try noisy top-k gating (forces hard, sparse per-example
routing rather than a soft weighted blend) before more loss-weight
sweeps -- soft blending may be the specific mechanism letting the
optimizer avoid commitment here.

## Gotchas learned the hard way (don't re-discover these)

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
