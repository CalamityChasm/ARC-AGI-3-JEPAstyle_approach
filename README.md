# ARC-AGI-3 JEPA-style Agent

A world-model agent for the [ARC Prize 2026 (ARC-AGI-3)](https://arcprize.org/competitions/2026/arc-agi-3)
Kaggle competition. Instead of prompting a language model to reason about
the board directly, the agent learns a small JEPA-style latent world
model (encoder + dynamics predictor) from played trajectories, then
plans over it: a mixture-of-experts predictor supplies a Bayesian
"hypothesis bundle" over what each action does, an InfoGain signal drives
exploration, a decoupled value head drives exploitation, and an exact
transition-graph memory recalls known winning moves instantly on repeat
visits to an exact game state.

This is a side project and a JEPA learning vehicle, not a leaderboard
attempt — see [`plan.md`](plan.md)'s "Framing & goals" for what "success"
means here.

## Where to look

| File | What it's for |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | **Read this first.** Current state of the project: what's built, what passed its milestone, what didn't, open questions, and a running log of real bugs found and fixed (including the full Kaggle submission debugging story). |
| [`plan.md`](plan.md) | The original staged build plan (Stage 0 → 6) and design rationale. |
| [`architecture.md`](architecture.md) | The full target architecture spec. |
| [`rules.md`](rules.md) | Summarized ARC-AGI-3 competition rules — scoring formula, action space, hardware/time limits. |
| [`notes.md`](notes.md) | A narrative development log, written for a reader who wants the story rather than the reference doc. |
| [`experiments/`](experiments) | Write-ups for in-progress experiments living on other branches (not yet merged to `master`). |

## Repo layout

- `jepa/` — the world model: grid representation, CNN encoder, MoE dynamics
  predictor, GRU-based recurrent core, value head, hypothesis bundle,
  exact transition-graph memory, and training/eval scripts for all of the
  above.
- `ARC-AGI-3-Agents/` — a vendored copy of the competition's agent
  framework, plus this project's own agents in
  `ARC-AGI-3-Agents/agents/templates/` (`hypothesis_agent.py` is the
  current one; `curiosity_agent.py` and `memory_agent.py` are earlier,
  simpler stages kept for comparison).
- `scripts/` — local tooling: run an agent across all local games
  (`run_stage0.py`), get the framework's own real scorecard (which
  computes the actual Kaggle scoring formula offline — `run_scorecard.py`),
  compare agents' recorded runs (`compare_agents.py`), summarize
  recordings (`summarize_recordings.py`).
- `kaggle_submission/` — everything needed to reproduce the real Kaggle
  submission from a clean checkout (dataset metadata, kernel metadata,
  the submission notebook). See `CLAUDE.md`'s "Kaggle competition
  submission" section for the full step-by-step.
- `checkpoints/` (gitignored) — trained model weights, regenerable via
  the training scripts below.
- `data/` (gitignored) — downloaded ARC-1/2 grids and the external
  trajectory-logs dataset, used for pretraining.

## Setup

1. `python -m venv venv && venv\Scripts\activate` (or `source
   venv/bin/activate` on Linux/Mac).
2. `pip install -r requirements.txt`. If you have a CUDA GPU, replace the
   CPU torch wheel afterward — see `CLAUDE.md`'s "Environment setup"
   section for the exact index URL to use (it's version-sensitive).
3. Set up Kaggle API credentials (`~/.kaggle/kaggle.json`) if you need to
   pull competition files, then grab `environment_files/` and
   `arc_agi_3_wheels/` from the competition dataset zip — these aren't in
   git and the harness won't find any games without them.
4. `cp ARC-AGI-3-Agents/.env.example ARC-AGI-3-Agents/.env` and fill in
   `ARC_API_KEY` (any value works offline; get a real anonymous one from
   `curl https://three.arcprize.org/api/games/anonkey` — these expire in
   roughly a day, a recurring gotcha documented in `CLAUDE.md`).

`CLAUDE.md`'s "Environment setup (new machine)" section has the full
version with every gotcha already hit and fixed — follow that if
anything here doesn't just work.

## Running an agent locally

```
python scripts/run_stage0.py --agent hypothesis          # all 25 local games
python scripts/run_stage0.py --agent hypothesis --game r11l   # one game
```

Other agents available the same way: `random`, `pressonce`, `curiosity`,
`memory`. Recordings land in `ARC-AGI-3-Agents/recordings/` (gitignored).

### Local backtesting

```
python scripts/run_scorecard.py --agent hypothesis --label my_run
```

Runs the agent across all local games and captures the framework's own
`FINAL SCORECARD REPORT`, which — in offline mode — already computes the
real Kaggle scoring formula (completion + human-baseline-relative
efficiency, level-weighted) using the real per-level human baseline
action counts. This is the actual metric, not an approximation; prefer
it over ad hoc level-count comparisons. Results save to
`logs/scorecards/<label>.json` (gitignored).

```
python scripts/compare_agents.py hypothesis curiosity
```

Compares two agents' most recent recorded runs on total levels
completed, distinct games solved, and average actions-to-first-completion.

## Training / regenerating checkpoints

```
python -m jepa.train_encoder
python -m jepa.train_predictor --epochs 30
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8
python -m jepa.train_value_head --epochs 20 --encoder checkpoints/encoder_moe.pt
```

See `CLAUDE.md`'s Stage 1/3/4/5 sections for what each script needs as
input data and what milestone each checkpoint is expected to clear.
`jepa/benchmark.py eval` gives a per-game breakdown of predictor-vs-
identity performance (the honest "changed-patches" metric, not naive
whole-grid MSE — see `CLAUDE.md` for why that distinction matters).

## Reproducing the Kaggle submission

Short version: stage `jepa/`, four checkpoint files, and
`hypothesis_agent.py` into a Kaggle dataset; push
`kaggle_submission/notebook/`; submit. Full step-by-step, including the
exact `kaggle` CLI commands and the one non-obvious gotcha that cost a
full debugging session (a Kaggle dataset attachment mounts at
`/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`),
is in `CLAUDE.md`'s "Kaggle competition submission" section.

## Branches

- `master` — stable; everything through the Stage 5 milestone and the
  first real, scored Kaggle submission lives here.
- `stage6-score-optimization` — in-progress experiment (see
  `experiments/`), not yet merged; held pending validation against a
  real submission before merging to `master`.
- `stage1-jepa` — superseded; Stage 1's work was folded into `master`
  once its milestone passed. Kept around for history, not active.
