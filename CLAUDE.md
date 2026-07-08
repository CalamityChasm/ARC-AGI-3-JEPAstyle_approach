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

### Stage 1 -- JEPA core: MILESTONE NOT MET, recommend pivoting to Stage 2

(See "Stage 1 recommendation: pivot to Stage 2" below for the reasoning --
three independent negative experiments, not just one plateau.)

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

**Gap-closing trend plateaued on loss-shaping alone (items 1-4: -24% to
-8.7%, then flat). Three further independent interventions (item 7's 9x
more/higher-changed-rate data; item 9's complete removal of the
25-games-at-once setting; item 9's zero-init residual branch) all also
failed to close it**, each landing at essentially the same "a few percent
worse than identity" result regardless of the intervention. That
consistency is itself the signal: it's not a data-volume problem
(item 7), not a cross-game-confound problem (item 9's single-game
ablation), not a bad-initialization problem (item 9's zero-init), and not
an encoder-sensitivity problem (item 8 directly measured the encoder
registering local changes fine -- 12x larger feature deltas at changed
vs. unchanged patches). What's left is the predictor's actual job: item 8
showed it has learned to output a near-zero residual (~16x smaller than
the true average change) and coast on the `feat +` skip connection. The
working conclusion is that **"predict no change" is close to the true
MSE-optimal prediction available to this specific architecture (single
frame in, action/xy/game conditioning, 2-3 conv layers, no history) on
this specific data (random-policy rollouts) -- not a fixable training
bug.**

## Stage 1 recommendation: pivot to Stage 2

Given the above, sinking further effort into this exact setup (more data,
more epochs, more loss-shaping) is unlikely to clear the milestone --
three different categories of fix were tried and none worked, which is
stronger evidence than a single negative result would be. Two things
would plausibly still move it, in order of expected leverage:

1. **A model with history, not just one frame** (Stage 3's Mamba-based
   sequence predictor) -- a single-frame predictor structurally cannot
   disambiguate state that depends on anything before the current frame
   (e.g. an object's velocity/direction, a level-specific rule learned
   from earlier in the episode). This is exactly what Stage 3 was already
   scoped for; the negative result here is evidence *for* needing it, not
   a sign Stage 3 will have the same problem.
2. **Trajectory data from a policy that actually progresses** (per this
   session's discussion) -- random-policy rollouts make many transitions
   close to genuinely unpredictable from the model's point of view (an
   action's effect depends on game state a random policy never
   deliberately sets up). A policy that plays *purposefully* (even a
   simple heuristic/curiosity-driven one) would produce transitions where
   action -> effect is more consistent and learnable, which is a data-
   *quality* argument distinct from item 7's data-*volume* experiment
   (volume alone, even at a high changed-rate, didn't help -- so this
   isn't "try the same kind of data again," it's "try structurally
   different data").

Both point toward **Stage 2** (per plan.md: a curiosity-driven agent using
prediction error for exploration, which is explicitly tolerant of an
imperfect world model) as the right next move -- and Stage 2 play
naturally generates exactly the kind of "runs that actually progress"
trajectory data that could make a future revisit of Stage 1 dynamics
fitting more productive. This is a legitimate scope decision per plan.md's
guiding principle #4 ("add components only when a measured bottleneck
demands it") backed by three converging negative experiments, not a
cop-out.

If a future session does revisit single-frame Stage 1 prediction before
Stage 3: keep using the honest `changed-patches` metric (via
`jepa/benchmark.py eval`, which also gives the per-game breakdown -- not
the naive whole-grid MSE, which is misleadingly easy to "beat" by mostly
predicting no change) as the real bar, and append new experiments to
`logs/benchmarks/history.jsonl` via the benchmark tool so they stay
comparable to the items above. `jepa/train_predictor.py --game <id>` (added
this session) is there for fast single-game ablations if a new hypothesis
needs isolating from the 25-games-at-once setting again.

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

## Gotchas learned the hard way (don't re-discover these)

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
