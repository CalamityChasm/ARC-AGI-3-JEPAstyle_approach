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

### Stage 1 -- JEPA core: IN PROGRESS, milestone not yet met

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

**Gap-closing trend plateaued on loss-shaping alone (items 1-4: -24% to
-8.7%, then flat), and item 7's ~9x-larger, much-higher-changed-rate
combined corpus *also* didn't close it (-0.3% -> -1.4%).** That's a
meaningful negative result: it weakens "just needs more data" as the
explanation and strengthens "the encoder's own feature map may not be
sensitive enough to local pixel changes" as the leading hypothesis (see
item 7's detail above) -- worth verifying directly before sinking more
effort into new data sources (MiniGrid, more Kaggle logs, etc.), since if
the encoder is the bottleneck, more/better trajectory data alone won't
fix it either.

## Next steps (pick up here)

In priority order, given item 7's finding above:

1. **Verify the encoder-sensitivity hypothesis directly.** Before trying
   another data source, check whether `CNNEncoder`'s 8x8 output actually
   moves when a patch's pixels change: take matched (frame_t, frame_t1)
   pairs with a known-changed patch, encode both with the trained
   `encoder_finetuned.pt`, and compare the per-patch feature-space delta
   at changed vs. unchanged patches. If changed-patch feature deltas
   aren't meaningfully larger than unchanged-patch deltas, that confirms
   the encoder itself (not the predictor, not the data) is the
   bottleneck -- e.g. `GroupNorm` + stride-2 downsampling may be smoothing
   out small local changes before the predictor ever sees them. Candidate
   fixes if confirmed: reduce/remove normalization strength, add a loss
   term that explicitly rewards feature-space sensitivity to pixel
   changes, or reduce the encoder's downsampling stride.
2. **If the encoder checks out fine:** MiniGrid/Sokoban trajectories are
   still on the table per plan.md's original data recipe (consistent
   action semantics across episodes, unlike ARC-3's per-game-meaning
   action ids) -- install `minigrid`, generate random-policy trajectories,
   translate to the shared 17-channel grid format via `jepa/grid.py`.
3. **Or:** accept the current state as a documented limitation and move to
   Stage 2 (curiosity-driven agent) per plan.md -- it explicitly doesn't
   require a perfect world model, just prediction-error-based exploration,
   which is somewhat robust to a noisy/imperfect predictor. This is a
   legitimate scope decision, not a cop-out; plan.md's guiding principle
   #4 is "add components only when a measured bottleneck demands it."

Whichever direction: keep using the honest `changed-patches` metric (now
via `jepa/benchmark.py eval`, which also gives the per-game breakdown --
not the naive whole-grid MSE, which is misleadingly easy to "beat" by
mostly predicting no change) as the real bar, and append new experiments
to `logs/benchmarks/history.jsonl` via the benchmark tool so they stay
comparable to items 5 and 7 above.

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
