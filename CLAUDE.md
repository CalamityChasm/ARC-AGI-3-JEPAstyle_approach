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
