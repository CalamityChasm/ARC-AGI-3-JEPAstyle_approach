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

**Current result (`python -m jepa.eval_stage1` after 30 epochs on the
current checkpoints):** predictor is **~8.7% worse than identity**, even
isolated to spatial patches that actually changed pixel-wise. Does not yet
clear Stage 1's milestone ("predicts next latents meaningfully better than
an identity baseline").

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

**Gap-closing trend has plateaued.** Three real fixes took it from -24% to
-8.7%; the fourth did nothing. Diminishing returns on loss-shaping alone.

## Next steps (pick up here)

In priority order, per plan.md's own data-scarcity diagnosis:

1. **Most likely next lever: MiniGrid/Sokoban trajectories.** plan.md's
   original Stage 1 data recipe called for generated turn-based grid-env
   trajectories (MiniGrid, Sokoban, Crafter, procgen) specifically because
   they have *consistent* action semantics across all episodes/levels
   (unlike ARC-3, where the same action id means something different in
   every one of the 25 games). This was deferred to keep Stage 1 scoped;
   given the current plateau, it's probably the right next experiment --
   pretrain/co-train the dynamics predictor on MiniGrid rollouts (install
   `minigrid`, generate random-policy trajectories, render to the same
   17-channel grid format via `jepa/grid.py`, though MiniGrid's grid
   representation will need a translation layer since it's
   object/color/state channels rather than ARC's flat color code) before
   fine-tuning on the scarce ARC-3 data.
2. **Now that there's a GPU:** the training code has no `.to(device)`
   calls anywhere -- add device handling to `jepa/train_encoder.py` and
   `jepa/train_predictor.py` first, since none of the current CPU-bound
   30-epoch/~15min runs will speed up on the RTX 2070 without it. Once
   GPU'd, more epochs / a larger encoder (`out_channels` in `CNNEncoder`)
   become cheap to try.
3. **Or:** accept the current state as a documented limitation and move to
   Stage 2 (curiosity-driven agent) per plan.md -- it explicitly doesn't
   require a perfect world model, just prediction-error-based exploration,
   which is somewhat robust to a noisy/imperfect predictor. This is a
   legitimate scope decision, not a cop-out; plan.md's guiding principle
   #4 is "add components only when a measured bottleneck demands it."

Whichever direction: keep using the honest `changed-patches` metric in
`jepa/eval_stage1.py` (not the naive whole-grid MSE, which is misleadingly
easy to "beat" by mostly predicting no change) as the real bar.

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
