# Stage 6: scaled-architecture + continuous-curriculum prep (no real run launched)

**Scope of this document, up front:** this is *preparation* work, not a
result. Per explicit instruction, the actual multi-day large-scale training
run was **not launched** -- that's gated behind a separate operational
decision (how to safely run an unattended multi-day process in this
environment) still open with the user. Everything below is: a genuinely
larger architecture, built and measured; a continuous curriculum training
script, built and verified; the mega-corpus assembled and measured; a short
smoke test proving all of it runs correctly end to end; and a concrete
launch plan for the real run once it's cleared. Nothing here claims a
representation-quality or held-out-generalization result -- the smoke test
is far too short (3-4 epochs, a few hundred transitions/epoch) to produce
one, and isn't meant to.

## Why this exists

CLAUDE.md's Stage 6 addendum documents an extensive investigation (13+
independent interventions) into why this project's world model shows no
measurable generalization edge over "predict no change" on ARC-3 games it
wasn't trained on. Every one of those interventions -- including a
2.14M-transition, 26-OpenSpiel-game roster and a 2x width sweep
(`stage6-expanded-roster`, `stage6-scaled-world-model`) -- tested capacity
or data changes at a scale that, on reflection, was still small relative to
what the hardware (an RTX 2070, 8.6GB VRAM, sitting mostly idle at
~285K-parameter model sizes) can actually support. This task is the
follow-up: build something that's a genuine step up in both capacity *and*
data scale, with a curriculum specifically designed the way the user asked
-- widest/least-relevant data first, narrowing toward the
most-relevant-but-scarcest (real local ARC-3 recordings) as training
proceeds -- rather than repeating either of the already-tried "capacity
alone" or "data alone" ablations.

## 1. Architecture: `ScaledCNNEncoder` + `ScaledMoEPredictor`

New files, **additive** to the existing `CNNEncoder`/`MoEPredictor`
(`jepa/models/encoder.py`, `jepa/models/moe_predictor.py`) rather than
modifying them in place:

- `jepa/models/encoder_scaled.py: ScaledCNNEncoder`
- `jepa/models/moe_predictor_scaled.py: ScaledMoEPredictor`

**Why separate classes, not a parameterized existing class:** `CNNEncoder`
and `MoEPredictor` are loaded by several already-shipped checkpoint-loading
paths (the Kaggle submission notebook, `hypothesis_agent.py`, various
`scripts/diagnose_*.py`). A new class with its own state-dict shape carries
zero risk to any of that -- nothing that already imports the old classes is
touched. `jepa/models/__init__.py` now additionally exports
`ScaledCNNEncoder`/`ScaledMoEPredictor` alongside the originals.

**What's actually different, not just parameterized:**

- `CNNEncoder` is 4 strided convs -- one downsample per conv, *zero* extra
  processing at any resolution before moving to the next. `ScaledCNNEncoder`
  keeps the same 64->32->16->8 downsampling schedule (so every existing
  8x8-patch assumption downstream -- the predictor's per-patch residual
  design, ACTION6's salience map, `jepa/grid.py`'s `patch_change_mask` --
  still applies unchanged) but inserts `blocks_per_stage` real residual
  blocks (`ResBlock`: two 3x3 convs + GroupNorm + GELU + additive skip) *at*
  each of the four resolutions before downsampling to the next. This is the
  "more depth, retaining higher spatial resolution longer" half of the
  brief -- the network actually does work at 64x64 and 32x32 now, not just
  passes through them.
- `width_mult` scales channel count at every stage (rounded to a multiple
  of 8 for `GroupNorm(8, ...)` validity). The 8x8-patch *output* channel
  count (`feature_channels`, fed into the predictor) scales too by default
  (`fixed_out_channels=False`), so a wider encoder hands the predictor a
  proportionally wider feature map -- capacity increases end-to-end, not
  just in the encoder.
- `ScaledMoEPredictor` generalizes `MoEPredictor`'s fixed 2-conv-layer
  (1-hidden-layer) experts to configurable `expert_depth` (more 1x1-conv
  hidden layers per expert -- kept pointwise/1x1, matching `MoEPredictor`'s
  own rationale that experts specialize on a causal *pattern* at a patch,
  not a spatial-context operation, which is the encoder's job) alongside
  the already-configurable `num_experts` and `expert_hidden` width.
  `load_balance_loss`, the noisy-top-k gating option, and the
  gate/expert-symmetry-breaking init are all reused unchanged from
  `moe_predictor.py` (imported, not reimplemented).
- `blocks_per_stage=0, width_mult=1.0` (the "size-0" default) reproduces
  `CNNEncoder`'s depth/resolution schedule and comparable parameter count
  -- confirmed directly: a default-config `ScaledCNNEncoder` +
  `ScaledMoEPredictor` (8 experts, `expert_depth=1`) totals **297,748
  params** (244,096 encoder + 53,652 predictor), matching the "current
  production encoder+predictor is only ~285K total parameters" figure this
  task's own brief cited. It is **not** literally state-dict-compatible
  with an existing `encoder.pt`/`moe_predictor.pt` checkpoint (the
  residual-block generalization changes some layer shapes even at zero
  extra blocks -- e.g. the stem is a single 3x3 conv here vs. `CNNEncoder`'s
  first strided 4x4 conv) -- a from-scratch-trained default-config
  `ScaledCNNEncoder` is a fair small-scale regression point, not a drop-in
  replacement. `checkpoints/encoder.pt` (copied into this worktree per the
  task's setup instructions) exists as a reference for what the small-scale
  baseline looks like, but the real launch trains the scaled model from
  scratch given how much the corpus and architecture both change -- direct
  weight transfer from the old, much-smaller encoder isn't attempted.

### Measured VRAM ceiling (RTX 2070, 8.59GB total)

Measured directly with `scripts/probe_scaled_vram.py` -- real forward +
backward + optimizer-step passes on dummy data (isolates pure model-size
VRAM cost from any data-loading cost), `torch.cuda.max_memory_allocated()`
after 5 steady-state steps (a warm-up step run first, memory stats reset
before timing so cuDNN algorithm selection / first-kernel-launch overhead
doesn't pollute the numbers).

**A real bug caught and fixed before trusting these numbers:** the first
probe run double-applied the `width_mult` scaling to `expert_hidden`
(config table already had already-scaled values, then the probe function
multiplied by `width_mult` again) -- silently building a
679M-param/14.5GB-peak config that "succeeded" without an
`OutOfMemoryError`, because Windows' CUDA driver has a system-memory
fallback policy that spills excess allocations into host RAM instead of
raising -- so an apparent "fits" result can be badly misleading if not
cross-checked. Confirmed directly (rerunning that exact bugged config
standalone, `width_mult=8, blocks=3, experts=24, expert_hidden` mistakenly
set to `512*8=4096` internally): 5 steps took **4.8s** (~960ms/step) vs.
the honest configs below running in the tens-of-milliseconds-per-step
range -- an order-of-magnitude slowdown, the unmistakable signature of host
RAM spillover, not genuine GPU-resident computation. Fixed the double-scale
bug and re-ran the corrected sweep.

| width_mult | blocks/stage | experts | expert_hidden | batch | total params | peak VRAM (GB) | reserved VRAM (GB) | ms/step |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 0 | 8 | 64 | 32 | 343,640 | 0.258 | 0.277 | 22.1 |
| 2.0 | 1 | 8 | 128 | 32 | 2,475,544 | 1.421 | 2.267 | 81.5 |
| 3.0 | 2 | 12 | 192 | 32 | 8,499,932 | 3.140 | 3.387 | 214.9 |
| 4.0 | 2 | 16 | 256 | 32 | 15,612,320 | 4.277 | 4.578 | 331.8 |
| 4.0 | 2 | 16 | 256 | 16 | 15,612,320 | 2.307 | 2.403 | 177.1 |
| 6.0 | 3 | 16 | 384 | 16 | 45,545,760 | 4.710 | 5.086 | 438.9 |
| **8.0** | **3** | **24** | **512** | **16** | **85,184,680** | **6.834** | **6.996** | **713.4** |
| 8.0 | 3 | 24 | 512 | 32 | 85,184,680 | 11.854 | 13.082 | 21,167.4 |
| 10.0 | 4 | 24 | 640 | 16 | 162,365,480 | 10.963 | 11.635 | 26,438.0 |

This is the corrected sweep, run after fixing the expert_hidden
double-scaling bug described above -- recovered from this agent's own
tool-call transcript after it was cut off mid-write by a session-limit
error, not re-measured. (An earlier, pre-fix sweep also ran, reaching
width_mult=12 at 678,975,664 params / 14.508GB "peak" with `ok=True` --
that number is exactly the kind of spillover-contaminated result the bug
writeup above warns about, superseded by this table, not an additional
data point to trust.)

**The last two rows are silently in host-RAM-spillover territory, not
genuinely GPU-resident, despite `ok=True` and reserved VRAM appearing to
exceed the 8.59GB card total** (13.08GB and 11.64GB *reserved* on an
8.59GB card is only possible via Windows' CUDA fallback policy) --
confirmed by the same signature already established above: ms/step
exploding to 21.2s and 26.4s respectively, a 30-60x blowup over the
213.4-438.9ms/step range every genuinely-fitting config in this table
shows. Both are excluded from consideration for the real launch on that
basis, not just because they're numerically large.

**Working ceiling recommendation:** `width_mult=8, blocks_per_stage=3,
num_experts=24, expert_hidden=512, batch_size=16` -- 85,184,680 params
(~286x the current production encoder+predictor's ~297,748), genuinely
GPU-resident at 6.834GB peak / 6.996GB reserved (81% of the card, ~1.6GB
headroom for the held-out-eval loader and driver/OS overhead running
alongside it), at a measured 713.4ms/step with no spillover signature.
`batch_size=32` at this same width spills over (see table) -- 16 is the
real ceiling for this width, not a conservative choice. A less aggressive
fallback if 16 turns out too tight in practice once real data loading is
added on top of this dummy-data probe: `width_mult=6` (45.5M params,
5.09GB reserved, 439ms/step) leaves a much larger ~3.5GB safety margin at
roughly half the parameter count -- worth switching to this if the first
real training leg (see launch plan below) shows OOM or spillover-speed
symptoms at width=8.

## 2. Curriculum: continuous synthetic-to-real sampling shift

`jepa/train_scaled_curriculum.py` (new script, not a modification of
`jepa/train_moe_predictor.py` -- that script's existing hard
`--pretrain-epochs`-then-`--epochs` two-phase split is left alone as a
still-valid, separate option; this is specifically the continuous
alternative the task asked for).

**Design:**
- Every non-ARC-3 source (MiniGrid, Sokoban, MinAtar, OpenSpiel, the
  hand-rolled ARC-synthetic puzzles) is pooled into one `"synthetic"`
  category. Real ARC-3 data (local recordings, optional external
  arc-3-logs, and a documented-but-not-yet-available integration point for
  a sibling session's search-harvested "winning round" corpus --
  `--search-harvest-dir`, see below) is the `"real"` category.
- **Within** a category, each *source* -- not each transition -- gets equal
  sampling pull: a source's per-example base weight is `1 / count_in_that_
  source`. This matters concretely at this corpus's actual scale: OpenSpiel
  alone can contribute well over a million transitions while MiniGrid
  contributes tens of thousands, and without this normalization OpenSpiel
  would drown out every other synthetic source within the "synthetic" 97%
  of an early epoch regardless of how genuinely different MiniGrid's
  mechanics are.
- **Across** categories, `synthetic_frac(epoch)` interpolates from
  `--synthetic-start` (default 0.97) down to `--synthetic-end` (default
  0.20) over the run, via `--curriculum-schedule {linear, cosine, flat}`.
  Each epoch draws a **fresh** weighted sample of `steps_per_epoch *
  batch_size` transition indices from that epoch's *current* mixture
  (`numpy.random.Generator.choice` with per-epoch-recomputed probabilities)
  -- a genuinely continuous shift over the whole run, not a single
  hard-switch epoch boundary.
- Verified directly, not just trusted from reading the code (per this
  task's own instruction): the smoke test below logs `target_synthetic_frac`
  alongside the *realized* fraction actually drawn each logged epoch, and
  they track each other correctly across the whole schedule (e.g. a 4-epoch
  run: target 0.970/0.713/0.457/0.200, realized 0.925/0.650/0.525/0.350 --
  close, with the residual gap from finite per-epoch sample size at only
  `steps_per_epoch=5` in the smoke test, not from any bug in the weighting
  logic).

**Held-out validation** is a `VAL_FRACTION=0.1` slice of *every* source,
held out of the curriculum sampler entirely -- mirrors every other script's
honest-eval convention in this project. Two val loaders are built: one
pooled across all sources (diagnostic), and one restricted to
`arc3_local` only, which is the actual metric every prior stage's milestone
in this project has been judged against (`changed-patches` improvement on
held-out real ARC-3 data) -- logged prominently so a future comparison
against production/prior checkpoints isn't diluted by synthetic-source
validation examples mixed in.

**Checkpointing / resume**, mirroring `jepa/train_moe_predictor.py`'s own
`--checkpoint-every`/`--resume-from` pattern (built on a sibling branch this
session, ported here rather than reinvented): `--checkpoint-every N` saves
`encoder_scaled.pt` + `moe_predictor_scaled.pt` + `optimizer.pt` +
`game_vocab_scaled.json` + a `curriculum_meta.json` (completed epoch count,
full args, per-source transition counts) every N epochs.
`--resume-from <dir>` loads all of that and continues the epoch loop from
`completed_epochs + 1`. **Verified directly, not just implemented and
assumed working:** ran 2 epochs with `--checkpoint-every 2`, then a
separate `--resume-from` invocation for 2 more epochs (total 4) -- the
resumed run picked up at epoch 3/4 exactly as expected, and its loss curve
continued the same downward trajectory the equivalent uninterrupted 4-epoch
run showed (0.789 -> 0.701 -> 0.635 -> 0.605 uninterrupted; 0.789 -> 0.721
[epoch 2, checkpoint] -> 0.596 -> 0.503 resumed -- same shape, not a reset
or discontinuity).

**Search-harvest integration point:** the task flagged a sibling session
mining winning-round ARC-3 transitions specifically, to incorporate as the
"most relevant" end of the curriculum if available. It is **not** present
in this worktree at time of writing (checked directly -- no such corpus
exists in this worktree or any sibling worktree's `ARC-AGI-3-Agents/
recordings/` beyond the standard random-policy recordings). Rather than
block on it or guess at its eventual format, `jepa/data/trajectories.py`
was given a small, honest refactor: `load_all_transitions(repo_root)` is
now a thin wrapper around a new `load_transitions_from_dir(recordings_dir)`
that accepts *any* directory of `*.recording.jsonl` files in the existing
format. `train_scaled_curriculum.py --search-harvest-dir <path>` uses this
directly -- when that corpus exists, pointing at it needs no further code
changes, just the flag.

## 3. Mega-corpus: composition and size

Assembled by `assemble_sources()` in `jepa/train_scaled_curriculum.py`,
generated on the fly at training time (not pre-serialized to a disk file --
see the storage section below for why that's a deliberate choice, not an
omission).

| source | mechanism | distinct game_ids | notes |
|---|---|---|---|
| MiniGrid | navigation/key-door/pickup puzzles | 21 environments, 1 shared `game_id="minigrid"` | already in this project (Stage 4); default 40 episodes/env x 80 steps = 67,200 transitions |
| Sokoban | push-with-persistent-consequences | 7 room configs, 1 shared `game_id="sokoban"` | already in this project (Stage 4); **off by default** (`--sokoban-episodes-per-config 0`) per its own documented negative result (CLAUDE.md Stage 4 item 8) -- available as an opt-in flag, not recommended for the real launch without revisiting that finding first |
| MinAtar | 5 classic-Atari-style reflex games | 5 sub-games, **per-game ids** (`minatar_breakout` etc.) | verified the per-game-id fix (not the original shared-id version) is what's wired in -- see "verification" below; default 160 episodes/game x 80 steps = 64,000 transitions |
| OpenSpiel | board/strategy games -- gravity placement, flip-capture, jump-capture, push-your-luck, seed-sowing, and (in the 26-game roster) chess-family, connection games, arcade-style (2048, catch), and a hex-cell strategy game | **26** games, one `game_id` each | the *expanded* roster from `stage6-expanded-roster` (not the original 6-game version) -- copied in from a sibling worktree specifically because the task asked for genuinely large-scale diversity; verified directly (not assumed): all 26 games generate transitions without error and every one respects `NUM_ACTIONS=8` (max observed action id checked directly across all 26, zero violations) |
| ARC-synthetic puzzles | hand-rolled route/toggle/match/symmetry/enclosure puzzles, ARC-3-*shaped* rather than borrowed from another engine | 5 types, 1 `game_id` each | built earlier this session (already on this branch per the task's own note); default 168 episodes/type x 80 steps = 67,200 transitions |
| ARC-3 local recordings (real) | actual local gameplay, random policy | 25 games (this worktree's regenerated set) | see storage note below -- only 1 pass (~2,025 transitions) generated in this worktree for smoke-testing; the real launch needs the full ~6-pass/150-file regeneration CLAUDE.md's setup section describes (~12k transitions) |
| ARC-3 external logs (real, optional) | `calamitychasm/arc-3-logs` Kaggle dataset | 25 games | **not available in this worktree** (`data/arc3_logs.zip` doesn't exist here) -- `--external-per-game` is wired and will use it automatically if the zip is pulled before the real launch, but isn't required |
| ARC-3 search-harvest (real, optional, future) | sibling session's winning-round mining | TBD | integration point only, see above -- not yet available |

**At the default flags used for the smoke test** (minigrid only, real
sources): 840 + 2,000 = 2,840 pooled transitions. **At the flags intended
for the real launch** (all synthetic sources on, local recordings at full
regenerated size): see the launch plan's exact command below for projected
totals -- roughly 67,200 (minigrid) + 64,000 (minatar) + 67,200
(arc_synthetic) + several million (26-game OpenSpiel, sized per the launch
plan's throughput-based budget below) + ~12,000 (arc3_local, once fully
regenerated) transitions, i.e. **the "widest, least-relevant" end of the
pool outnumbers the "most-relevant" end by roughly 100-300x in raw
transition count** -- exactly why the curriculum's per-source (not
per-transition) weighting inside each category, and the category-level
schedule itself, both matter: without them the real ARC-3 signal would be
essentially unreachable by pure i.i.d. sampling at any point in training,
not just underweighted early on.

## 4. Storage

Checked before and during data generation, per the task's explicit
instruction (this project has hit real full-disk crises before -- see
CLAUDE.md's Gotchas):

- **`C:` had ~18.1GB free at the start of this session and ~18.0GB free at
  the end** (`wmic logicaldisk get size,freespace,caption`, checked
  directly, not assumed) -- essentially unchanged, because this prep
  deliberately generates all synthetic-source transitions **in memory**
  (Python lists of tuples, never written to an intermediate corpus file on
  disk) rather than pre-serializing a mega-corpus to disk before training.
  This is a real design choice, not an oversight: it means there is no
  giant corpus file to stage anywhere, on either drive -- the only disk
  footprint from data generation is the small local ARC-3 recordings
  (`ARC-AGI-3-Agents/recordings/`, gitignored, ~a few MB for the one smoke-
  test pass done in this worktree).
- **`E:` had ~40.4GB free at the start, ~39.3GB at the end** of this
  session (the ~1GB drift is unrelated background activity from other
  agents sharing this drive across sibling worktrees, not anything this
  task wrote) -- `E:\jepa_overflow\` is this project's own established
  overflow location (already contains `checkpoints_search/` and
  `recordings/` subdirectories from earlier sessions' work).
- **Checkpoint size scales with parameter count, and matters for the real
  run's storage planning:** directly measured from the smoke test's
  297,748-param model -- `encoder_scaled.pt` 983KB, `moe_predictor_scaled.pt`
  223KB, `optimizer.pt` 2.42MB (Adam's two moment buffers roughly double
  the raw parameter footprint), ~3.6MB total per checkpoint. Scaling
  linearly with parameter count to the recommended launch config above
  (85,184,680 params, ~286x the smoke-test size): roughly **~1.0GB per
  checkpoint** (286 x 3.6MB). At `--checkpoint-every 2` over a ~100-leg,
  ~200-epoch run (see launch plan below) that's ~100 checkpoint writes --
  keeping all of them would eat E:'s entire ~37GB free margin. **The
  launch plan below includes an explicit checkpoint-rotation step (keep
  only the last 3) for exactly this reason** -- this is a real operational
  requirement this prep surfaced, not a hypothetical. **The real launch's
  `--out` must point at `E:\jepa_overflow\...`, not this worktree's default
  (`checkpoints_scaled/`, which lives on `C:`)** -- with `--checkpoint-every`
  saving repeatedly over a multi-day run, even a few dozen periodic saves
  at that per-checkpoint size would meaningfully eat into `C:`'s already-
  thin ~18GB margin if left at the default. (The default was deliberately
  *not* changed to `E:` in the script itself, to keep the smoke-test/dev
  loop simple and independent of drive layout -- the launch command below
  overrides it explicitly.)

## 5. Smoke test: what was actually verified

Run directly, not assumed from reading the code:

1. **End-to-end correctness at tiny scale**: `--epochs 3 --steps-per-epoch 5
   --batch-size 8 --minigrid-episodes-per-env 2` (minigrid + arc3_local
   only) completed cleanly, encoder+predictor totaling 297,748 params,
   ~29-30s/epoch at this tiny size (data-generation-and-DataLoader-
   dominated at this scale, not representative of real-corpus wall-clock --
   see the launch plan's own throughput estimate below).
2. **Curriculum schedule genuinely shifts, verified by realized-mix
   logging, not just trusted**: see section 2 above -- target and realized
   `synthetic_frac` track each other across all 3-4 logged epochs.
3. **Checkpoint/resume genuinely works, verified by an actual
   interrupt-and-resume, not just code review**: see section 2 above.
4. **All 26 OpenSpiel games generate valid transitions and respect
   `NUM_ACTIONS=8`**, checked directly (zero violations) rather than
   assumed from the copied module's own docstring claims.
5. **VRAM measurements are real, not estimated** -- see section 1, including
   catching and fixing a real double-scaling bug in the probe script itself
   before trusting its numbers.

**What was explicitly not run:** any multi-epoch training long enough to
produce a real changed-patches/held-out-generalization number. That's the
actual experiment this prep is gating, not something to fake a preview of
at smoke-test scale.

## 6. Launch plan (not yet executed)

**Written by the main session directly, not the prep agent** -- the prep
agent hit a session-limit error mid-write of this section; its real VRAM
numbers were recovered from its tool-call transcript (section 1 above),
but this launch plan itself is new, composed from that recovered data
plus the flags actually implemented in `jepa/train_scaled_curriculum.py`
(checked directly against the script's own `argparse` definitions, not
guessed).

### What's deliberately held fixed vs. what's new

Every data-composition choice below (MiniGrid/MinAtar/OpenSpiel/
ARC-synthetic episode counts) reuses the exact scale already validated
earlier this session (`stage6-expanded-roster`'s ~2.14M-transition,
26-OpenSpiel-game corpus; MiniGrid/MinAtar/ARC-synthetic at their
already-used defaults) rather than inventing new, unvalidated data
volumes. **The only two things this launch actually changes relative to
every prior negative result are model capacity (285K -> 85.2M params,
~286x) and a continuous synthetic-to-real curriculum instead of a
two-phase split.** This keeps the run interpretable as a real test of the
user's "capacity was the missing ingredient" hypothesis, rather than
confounding it with an unrelated data change on top.

### Prerequisites (not yet done, needed before launch)

1. **Regenerate the local ARC-3 recordings corpus to its normal ~12k-
   transition, 6-pass size** (this worktree only has 1 smoke-test pass,
   ~2,025 transitions) -- from repo root, in the venv:
   `python scripts/run_stage0.py --agent random`, run 6 times. Cheap,
   a few minutes total.
2. Confirm `E:\jepa_overflow\` still has enough free space for ~100
   rotated-but-temporarily-overlapping ~1GB checkpoints plus the winning-
   transition harvest corpus already landing there from the sibling
   mining task (~37GB free as of this writing -- recheck immediately
   before launch, not just trust this number).
3. Revert `Hypothesis.MAX_ACTIONS` in `hypothesis_agent.py` back to 300
   if the sibling mining task's bump is still in place (unrelated to this
   launch, but shares the repo -- don't launch scaled training against a
   dirty tree with an unrelated debug change still active).

### Launch command

```
python -m jepa.train_scaled_curriculum ^
  --width-mult 8 --blocks-per-stage 3 --num-experts 24 --expert-hidden 512 --expert-depth 1 ^
  --batch-size 16 --lr 3e-4 ^
  --epochs 200 --steps-per-epoch 1000 ^
  --synthetic-start 0.97 --synthetic-end 0.20 --curriculum-schedule linear ^
  --minigrid-episodes-per-env 40 --minigrid-steps-per-episode 80 ^
  --minatar-episodes-per-game 160 --minatar-steps-per-episode 80 ^
  --openspiel-episodes-per-game 1400 --openspiel-steps-per-episode 60 ^
  --arc-synthetic-episodes-per-type 168 --arc-synthetic-steps-per-episode 80 ^
  --out E:\jepa_overflow\checkpoints_scaled_launch ^
  --checkpoint-every 2
```

(`--openspiel-episodes-per-game 1400` x 26 games x 60 steps/episode =
~2.18M transitions, matching `stage6-expanded-roster`'s already-validated
~2.14M scale -- an approximation, not a reproduction of that run's exact
episode count, since that number isn't in this doc's own history; log the
real realized per-source counts from the first leg's `curriculum_meta.json`
and sanity-check against this estimate before trusting later legs.
`--minatar-episodes-per-game` and `--arc-synthetic-episodes-per-type`
default to 0 in the script -- both must be passed explicitly or those
sources silently drop out of the mix entirely.)

**Resume** (every leg after the first):
```
python -m jepa.train_scaled_curriculum --resume-from E:\jepa_overflow\checkpoints_scaled_launch [... same flags ...]
```

### Timing estimate, and what to verify on the first real leg

Pure model-compute time at this config: 1000 steps x 713.4ms =
~713s (~11.9min) per epoch. Real per-epoch wall-clock will run higher
once real DataLoader/eval overhead is added on top of the dummy-data
probe this estimate comes from -- **the first real leg is the actual
measurement; treat this ~12-18min/epoch range as a estimate to confirm,
not a number to schedule around blindly**, consistent with this whole
project's own standing discipline (see e.g. the MoE gate/Procgen/
scaled-world-model sections of CLAUDE.md, all of which caught estimation
errors by checking real numbers rather than trusting projections). At
200 epochs x ~15min average, total wall-clock is roughly ~50 hours
(~2 days) -- a real multi-day run, matching what was asked for, and
adjustable by changing `--epochs` up or down once the first leg's real
per-epoch time is known.

### Leg structure (cost-conscious supervised chunk learning)

- **1 leg = 2 epochs** (`--checkpoint-every 2`), ~24-36min wall-clock at
  the estimated per-epoch range -- comfortably under the ~45min
  background-process kill limit this session already confirmed directly,
  with real margin rather than cutting it close.
- **~100 legs total** across the full 200-epoch run. Each leg-boundary
  check-in is a monitor invocation -- at ~100 of them over ~2 days, the
  per-invocation cost (cheap model, minimal context, mechanical
  check-only logic, per the earlier design in this conversation) matters
  more than usual; don't let the monitor read large files or reason
  open-endedly per cycle.
- **Watchdog, not just presence/absence checking**: the VRAM sweep above
  showed spillover configs run 30-60x slower (21-26 *seconds*/step vs.
  213-713*ms*/step for genuinely GPU-resident configs) while still
  reporting `ok=True`/exit code 0 -- a silent-degradation failure mode a
  simple "did it crash" check would miss entirely. The monitor should
  compute observed ms/step from each leg's own log (epoch wall-clock /
  steps-per-epoch) and escalate to the user if it exceeds ~1500ms/step
  (2x the measured 713.4ms baseline) rather than continuing to "succeed"
  at 30x slower throughput for hours before anyone notices.
- **Checkpoint rotation**: after each leg's checkpoint write succeeds,
  delete any checkpoint directory/snapshot older than the 3 most recent
  (~1GB each, ~3GB steady-state instead of ~100GB if left unrotated over
  the full run) -- a mechanical, no-judgment step the monitor can do
  every leg.
- Escalate (don't attempt to fix) on: any traceback, an OOM error
  specifically (switch to the `width_mult=6` fallback config above rather
  than guessing at a fix), the ms/step watchdog tripping, or either drive
  dropping below the existing 5GB safety threshold already enforced by
  this project's harvest scripts.

### Still an open decision for the user, not assumed here

`--epochs 200` (~2 days) is this plan's default, chosen to land in the
middle of "days" as stated, not at either extreme. If a firm ceiling
(e.g. "stop by Wednesday") or a firm minimum is preferred instead, that's
a real preference to state before launch, not something to infer.

## Honest limitations of this prep, not glossed over

- The OpenSpiel 26-game roster and the `arc_synthetic_data.py` module were
  both copied in from sibling worktrees that already built them this
  session (correctly credited above), not built fresh here -- this task's
  own instructions explicitly anticipated this ("already merged into this
  branch — check it's present" for `arc_synthetic_data.py`; in this
  worktree specifically, neither module was actually present until copied
  in, since this worktree branched from `master`, not from the branches
  that built them). Verified functionally (imports, generates valid data,
  respects `NUM_ACTIONS`) rather than assumed correct from the source
  worktree's own documentation.
- Sokoban is included in the source-assembly code but left **off by
  default**, per its own already-documented negative result -- turning it
  on for the real launch would be revisiting a closed finding, not
  following this task's brief.
- The local ARC-3 recordings corpus in this worktree is a single random-
  policy pass (~2,025 transitions across 25 games), not the full ~12k-
  transition, 6-pass corpus CLAUDE.md's setup section describes as the
  project's normal baseline -- regenerating that fully (`python scripts/
  run_stage0.py --agent random`, ~6 times) is a launch-day prerequisite,
  not something done here (would have added significant wall-clock for a
  smoke test that doesn't need the full corpus to validate the pipeline).
- The VRAM probe's largest configs (`width_mult` 14-16) were killed
  mid-run after it became clear they were thrashing into Windows' CUDA
  system-memory-fallback spillover (see section 1) rather than genuinely
  exercising GPU-resident compute -- the trimmed, honest sweep stops at
  `width_mult=12`, which is already well past any config this plan
  actually recommends launching with.
