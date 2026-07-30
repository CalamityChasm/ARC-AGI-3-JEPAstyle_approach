# Stage 6 experiment: is the game-id-ablation win reproducible across seeds?

**Status: DONE. The headline +64.9% (no-game-id) vs +8.0% (with-game-id) ~8x
gap does NOT reproduce.** Across 3 independently-seeded reruns per
condition on the identical 20-game corpus/recipe: **with-game-id
mean +53.5% (std 4.6, range 48.2-57.1%) vs no-game-id mean +42.3% (std
14.7, range 26.1-55.0%)** -- overlapping ranges, and if anything
with-game-id now looks slightly *better* on average and substantially
*more stable* (3x lower std) than no-game-id, the opposite of the
single-run finding's implied conclusion. The held-out-games result (both
conditions ~0% regardless of seed) reproduces cleanly and is not in
question. **Per the task's own gating condition ("if the effect
reproducibly holds up, proceed to the full-25-game candidate
comparison"), that step was skipped** -- there is no reproducible effect
to carry forward into a production candidate. See "Results" and "Honest
read" below for the full numbers and a specific finding about *why* the
original single run was so misleading (not just generic seed noise --
the original with-game-id run looks like a genuine outlier even relative
to this session's own seed-to-seed spread).

## Motivation

`experiments/stage6_gameid_ablation.md` found, from a single training run
per condition, that removing per-game embedding conditioning from the MoE
predictor produced a huge changed-patches improvement on the 20 games both
checkpoints trained on: **+64.9% (no-game-id) vs. +8.0% (with-game-id)** --
roughly an 8x gap. That experiment's own "honest read" flagged this as a
single-run result needing at least one more independently-seeded rerun
before treating the magnitude as load-bearing (this project's own standing
norm -- see CLAUDE.md's Stage 4 MoE history for why single-run MoE results
have been distrusted before, e.g. the 44.1% -> 22.9% swing from a corpus
reseed alone).

This experiment reruns both conditions across 3 independent seeds each (6
training runs total) on the identical 20-game held-out corpus/recipe, to
find out whether the ~8x gap is a robust, reproducible effect or a lucky/
unlucky single draw.

## Method

Identical recipe to `stage6_gameid_ablation.md` and `stage6_game_holdout.md`
(same branch lineage, same `--exclude-games r11l,bp35,m0r0,tr87,ka59`,
same `--pretrain-epochs 20 --epochs 60 --num-experts 8 --external-per-game
2000 --contrast-weight 0.0 --checkpoint-every 5`), varying only two things
between the 6 runs: `--seed {0,1,2}` and presence/absence of
`--ablate-game-id`.

**New: `--seed` flag added to `jepa/train_moe_predictor.py`** (this branch).
Previously the script had no explicit global seeding at all outside a fixed
`torch.Generator().manual_seed(0)` for the train/val split (deliberately
fixed across runs so the held-out validation population stays identical
for comparison) -- everything else (model weight init, `WeightedRandomSampler`
batch order) rode on whatever per-process entropy torch's default RNG
happened to have. Different process launches already differed by default
(torch seeds its default generator from OS entropy per-process), so the
prior single-run-per-condition results were not literally deterministic
duplicates of each other -- but there was no explicit, checkable seed
control. `--seed N` now calls `torch.manual_seed`/`random.seed`/
`np.random.seed` up front. Does NOT touch MiniGrid's own independently-
seeded (seed=0 default) synthetic pretrain corpus generation -- that stays
byte-identical across all 6 runs, isolating the variable under test to
model init + ARC/MiniGrid DataLoader sampling order, not also the pretrain
data itself.

Commands (repeated for seed in {0, 1, 2}):
```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 \
  --out checkpoints_reseed/seed{N}_gameid --seed {N}

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --ablate-game-id --checkpoint-every 5 \
  --out checkpoints_reseed/seed{N}_nogameid --seed {N}
```
(`JEPA_NUM_WORKERS=0`, same rationale as the prior experiments -- shared/
contended GPU box, this session running alongside another concurrent
diagnostic agent.)

**Operational note on launching long training runs.** Confirmed the same
silent-death risk `stage6-gameid-ablation` documented for long
(80-90-minute) `run_in_background` bash-tracked training processes, and
used the same fix (fully OS-detached `Start-Process`, no parent job-object
tying the process to any tracked shell task) for the real runs. One
additional wrinkle found this session: a first detached-process attempt
appeared to hang (`Get-Process`/`tasklist` showed 0 CPU time and a static
~7.5MB working set for several minutes) -- turned out to be a red herring
caused by Python's stdout being block-buffered (not line-buffered) once
redirected to a file, combined with this sandboxed environment's process
introspection tools apparently not reporting CPU/working-set accurately
for fully-detached cross-session processes. Confirmed the process was
actually healthy by relaunching the identical command in the foreground
with `python -u` (unbuffered) via a short-lived tracked background bash
call: real progress lines (`loaded 9600 local ARC-3 transitions`, `loaded
33998 external ARC-3 transitions`, ...) appeared exactly as expected within
a couple of minutes. All 6 real runs used `python -u` plus the detached-
process launch pattern together, so log files show genuine live progress
AND the process survives past a tracked shell task's lifetime.

Evaluation: `scripts/eval_gameid_reseed.py` (new, generalizes
`scripts/eval_gameid_ablation.py`'s fixed 3-checkpoint comparison to an
arbitrary dict of checkpoints) -- same changed-patches-over-identity
methodology, on both the 20 trained games and the 5 held-out games, with
the ablated checkpoints' held-out/trained slices both evaluated with
`game_idx` forced to 0 (matching how they were trained), exactly as in
`stage6_gameid_ablation.md`.

## Results

All 6 training runs (seeds 0, 1, 2 x {with-game-id, no-game-id}) trained
cleanly to completion (20 MiniGrid-pretrain + 60 ARC-finetune epochs each)
on the identical 20-game corpus (9,600 local + 33,998 external + 67,200
MiniGrid transitions, 21-entry game vocab, `exclude_games=[r11l, bp35,
m0r0, tr87, ka59]` -- confirmed via each run's `moe_training_meta.json`,
differing only in `seed` and `ablate_game_id`). `scripts/
eval_gameid_reseed.py` evaluated all 6 against the identical held-out-game
(n=1881 changed-patch transitions) and trained-game (n=5441) populations
used by the original ablation experiment.

### Per-run changed-patches improvement over identity

| run | seed | trained games (20) | held-out games (5) |
|---|---|---|---|
| with-game-id | 0 | +55.1% | -2.4% |
| with-game-id | 1 | +57.1% | +1.9% |
| with-game-id | 2 | +48.2% | -0.3% |
| no-game-id | 0 | +45.8% | -0.3% |
| no-game-id | 1 | +55.0% | -0.3% |
| no-game-id | 2 | +26.1% | +0.3% |

### Distribution summary

**Trained games (20), n=3 seeds per condition:**

| condition | mean | std | min | max |
|---|---|---|---|---|
| with-game-id | **+53.5%** | 4.6 | +48.2% | +57.1% |
| no-game-id | **+42.3%** | 14.7 | +26.1% | +55.0% |

**Held-out games (5), n=3 seeds per condition:**

| condition | mean | std | min | max |
|---|---|---|---|---|
| with-game-id | -0.2% | 2.1 | -2.4% | +1.9% |
| no-game-id | -0.1% | 0.4 | -0.3% | +0.3% |

Full per-checkpoint, per-game breakdown: `logs/gameid_reseed_results.json`.

## Honest read

**The headline 8x gap from `stage6_gameid_ablation.md` (+64.9% no-game-id
vs +8.0% with-game-id, n=1 each) does not reproduce.** With 3 independently
seeded runs per condition:

1. **The two conditions' ranges overlap substantially** (with-game-id:
   48.2-57.1%; no-game-id: 26.1-55.0%) -- not the clean, large separation
   the single-run result suggested.
2. **With-game-id's mean (+53.5%) is actually slightly *higher* than
   no-game-id's (+42.3%)** -- the opposite direction from what the
   original single-run comparison implied.
3. **No-game-id has ~3x higher variance (std 14.7 vs 4.6)** -- across
   these 3 seeds, removing per-game conditioning didn't just fail to help
   on average, it made the trained-games result noticeably less
   *reliable* run to run.
4. **The held-out-games result reproduces cleanly and is not in
   question**: both conditions stay tightly clustered near 0% regardless
   of seed (with-game-id: -2.4% to +1.9%, mean -0.2%; no-game-id: -0.3%
   to +0.3%, mean -0.1%) -- consistent with `stage6_gameid_ablation.md`'s
   own finding that this ablation does nothing for hidden-game
   generalization. That part of the prior experiment holds up fine; it's
   specifically the trained-games "large win" that doesn't.

**A specific, worth-flagging finding: the original with-game-id run's
+8.0% looks like a genuine outlier, not just typical seed noise.** This
session's 3 with-game-id seeds are tightly clustered (48.2%, 55.1%,
57.1%, std 4.6) -- +8.0% sits roughly **10 standard deviations** below
that mean, for a metric that isn't behaving in an obviously heavy-tailed
way across the 3 seeds actually measured. By contrast, the original
no-game-id run's +64.9% is only about 1.5 std above this session's
no-game-id mean (42.3%, std 14.7) -- unusual but not wildly implausible
given how much more variance that condition already shows. This asymmetry
suggests the original 8x gap was driven mostly by the **with-game-id run
landing somewhere unusually bad** (a poor init trapped in a worse local
optimum, or some other run-specific issue not captured by `--seed`
alone, e.g. before this branch's `--seed` flag existed the run relied on
whatever implicit per-process entropy torch happened to draw) rather than
by no-game-id being genuinely, reliably ~8x better. Not independently
root-caused here (the original run's exact process/environment state
isn't reproducible after the fact), but worth remembering as a concrete
illustration of why this project's standing "don't trust n=1" norm exists
-- a single unlucky draw in *either* condition, not just the "interesting"
one, can manufacture a dramatic-looking effect.

**Per the task's own gating condition, this closes the loop here rather
than continuing to a production candidate.** Step 3 ("if the effect
reproducibly holds up... retrain a no-game-id MoE predictor on the full
25-game corpus and compare against production") was explicitly
conditioned on the effect holding up under reseeding. It did not: there
is no reproducible trained-games improvement from ablating game-id
conditioning to carry forward, and the modest evidence that exists (lower
mean, higher variance for no-game-id) points mildly *against* making this
change to production, not for it. Training and benchmarking a full-25-game
no-game-id candidate would be spending compute to promote a checkpoint
built on a hypothesis this experiment just falsified -- skipped
deliberately, not from lack of time.

**Recommendation: keep the current production recipe (per-game
conditioning on).** This does not resurrect Stage 1's old "no improvement"
finding for per-game conditioning (CLAUDE.md item 4) as a positive
endorsement -- this experiment's own data is also consistent with
game-id conditioning being genuinely neutral-to-mildly-positive on this
architecture/data scale, not a clear win either. But there is no
evidence here that removing it helps, and real (if weak) evidence that it
makes trained-games prediction quality less consistent run-to-run. The
one thing this ablation was originally motivated by -- closing the
held-out-game generalization gap -- remains firmly unsolved by this
change, exactly as `stage6_gameid_ablation.md` already found and this
session's reseeding reconfirms.

**What this means for the broader "which lever helps held-out-game
generalization" question:** still open. Per-game conditioning is neither
the cause of nor the fix for that gap (both `stage6_gameid_ablation.md`
and this reseed agree on that). Whatever the actual bottleneck is, it's
somewhere else in the pipeline -- consistent with the parallel encoder
investigation this task description flagged as already in progress on
another branch.

## Operational notes (concurrency and buffering gotchas)

Two mechanical findings worth keeping for future long-training sessions
in this repo, neither of which is about the model or data:

- **Launching all 6 seeded runs fully concurrently caused severe I/O
  contention, not a proportional slowdown.** All 6 processes hit
  `load_external_transitions`'s per-line JSON-parse-and-reservoir-sample
  pass over the same junctioned `data/arc3_logs.zip` at nearly the same
  moment; 10+ minutes in, every process still showed ~0 accumulated CPU
  time (`Get-Process`/`tasklist`) despite `nvidia-smi`/overall CPU load
  showing plenty of headroom -- a classic "resources look free but
  nothing is actually progressing" trap. Fixed by killing 4 of the 6 and
  running only 2 concurrent at a time (paired by seed) -- confirmed
  healthy immediately (real epoch-by-epoch log lines within seconds once
  contention dropped). "GPU/CPU/RAM all show headroom" is not sufficient
  evidence that N concurrent training processes are actually making
  progress, especially when several of them share one large input file.
- **A `python` process with default (non-`-u`) stdout redirected to a
  file can look completely stalled for many minutes even when it's fine**
  -- stdout block-buffers once it's not a tty, so a handful of early
  `print()` calls (data loading, MiniGrid generation) can sit unflushed
  for a long time. Every real run in this experiment was launched with
  `python -u` specifically so live polling of the log file would be
  meaningful; without it, "no new log lines" is not reliable evidence of
  a hang, as this session initially discovered the hard way with a
  detached process that looked stuck but wasn't.

## Reproducing this experiment

```
# Corpus setup (once): copy the verified 150-file *.random.80.* corpus into
# ARC-AGI-3-Agents/recordings/, and data/arc3_logs.zip into data/ -- both
# gitignored, see CLAUDE.md's environment-setup section.

for seed in 0 1 2; do
  python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
    --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
    --contrast-weight 0.0 --checkpoint-every 5 \
    --out checkpoints_reseed/seed${seed}_gameid --seed ${seed}

  python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
    --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
    --contrast-weight 0.0 --ablate-game-id --checkpoint-every 5 \
    --out checkpoints_reseed/seed${seed}_nogameid --seed ${seed}
done

python scripts/eval_gameid_reseed.py
```
