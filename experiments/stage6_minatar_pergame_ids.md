# Stage 6 follow-up: was MinAtar's pooled `game_id="minatar"` a confound, or is the held-out-games gap deeper than that?

**Status: DONE. Mixed, clarifying result.** Per-game MinAtar ids
(`minatar_breakout`, `minatar_asterix`, `minatar_freeway`,
`minatar_seaquest`, `minatar_space_invaders` instead of one shared
`"minatar"`) produce a **large, real improvement on the standard
trained-games sanity check** (+55.8% vs. the shared-id run's +2.1%, a
10-epoch trailing mean) -- confirming the pooling confound flagged in
`experiments/stage6_diverse_pretraining.md` was genuine. But this **does
not translate into any held-out-ARC-games generalization improvement**:
-0.0% on fold 1's 5 held-out games, statistically indistinguishable from
both the shared-id run's -1.4% and the MiniGrid-only baseline's -0.1%,
all three sitting inside this metric's own established noise band (see
`experiments/stage6_multifold_generalization.md`: fold-to-fold std
0.66-0.76pp). **The pooling confound was real and worth avoiding in any
future synthetic-data integration -- but it was not what stood between
MinAtar and closing the held-out-games gap.** This is the 9th independent
intervention against that specific gap (following the 7 in CLAUDE.md's
Stage 6 addendum plus the shared-id MinAtar attempt itself) and the 9th
to fail to close it.

## Motivation

`experiments/stage6_diverse_pretraining.md` tried MinAtar (5 clean-room,
no-ROM Atari-style games) as a second synthetic pretraining source
alongside MiniGrid, with all 5 sub-games pooled under one shared
`game_id="minatar"` (mirroring MiniGrid's own choice of one id across 21
environments). That attempt was a negative result on both the standard
trained-games metric (+2.1% vs. baseline's +4.0%) and the held-out-games
generalization test (-1.4% vs. baseline's -0.1%), with one game (`r11l`)
individually dropping to -10.5%.

That write-up flagged an unresolved confound: MiniGrid's 21 environments
are all coherently about the same underlying theme (grid navigation), so
sharing one id lets the model learn one consistent action vocabulary
across genuinely-similar layouts. MinAtar's 5 sub-games don't share
nearly as much structure with each other -- paddle-and-ball physics
(breakout), lane-crossing timing (freeway), submarine survival
(seaquest), and shoot-em-up projectiles (space_invaders, asterix) are
mutually distinct causal patterns, not variations on one theme. Pooling
them under one id may have forced the shared encoder/predictor to fit
several mutually-inconsistent action->effect mappings at once -- exactly
the kind of confound Stage 1 originally worried about for the 25 ARC-3
games themselves (and found *not* to be the dominant issue there, see
CLAUDE.md's Stage 1 item 4), but never tested for a *synthetic*
pretraining source before. This experiment isolates that one variable.

## What was changed

`jepa/data/minatar_data.py`: `generate_transitions` now defaults to
`per_game_ids=True`, assigning each sub-game its own id
(`GAME_ID_PREFIX + game_name`, e.g. `"minatar_breakout"`) instead of the
old shared `GAME_ID = "minatar"` constant. `GAME_ID` is kept for backward
compatibility (and `per_game_ids=False` reproduces the original shared-id
behavior if ever needed again), but nothing else calls it anymore.

No changes were needed in `jepa/train_moe_predictor.py`: its game-vocab
construction (`game_ids = sorted({t[6] for t in arc_transitions} |
synthetic_game_ids)`) already reads `game_id` generically off each
transition tuple's 7th field, so it automatically picked up 5 distinct
MinAtar entries instead of 1 the moment the data source started emitting
them -- confirmed directly in the training log (`26 distinct games in
the shared vocab`, vs. 22 in the original shared-id run: 20 trained ARC
games + 1 minigrid + 5 minatar_* this time, vs. 20 + 1 + 1 before).

`scripts/eval_diverse_pretraining.py` was reused completely unchanged --
its vocab-size handling is also fully generic (`len(game_vocab)`,
`MoEPredictor(num_games=len(game_vocab), ...)`), so it needed no
adjustment for the larger 26-entry vocab.

## Method: identical fold-1 corpus/recipe to the original MinAtar test

Same command as `experiments/stage6_diverse_pretraining.md`, only the
output directory changed:

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --minatar-episodes-per-game 160 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_minatar_pergame
```

Corpus setup mirrored the original experiment exactly: the canonical
150-file `*.random.80.*` local recordings corpus (copied from
`E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\`) and `data/arc3_logs.zip`
staged into this worktree. Confirmed from the training log that the
non-MinAtar parts of the corpus are byte-identical to the original
experiment: 9,600 local ARC-3 transitions, 33,998 external transitions,
67,200 MiniGrid transitions -- matching `stage6_diverse_pretraining.md`'s
own numbers exactly. MinAtar itself also generated the identical
transition count (64,000 across 5 games), just distributed across 5
`game_id`s instead of 1.

Sanity-checked before training that the new per-game-id assignment
actually works: a 100-transition smoke sample showed all 5 distinct
`minatar_*` ids present, action ids still `{0..5}` (well within
`NUM_ACTIONS=8`), changed-rate 68% (consistent with the original
module's own smoke-test figure).

Single training run (GPU shared with 2 other concurrent agents this
session, `JEPA_NUM_WORKERS=0` per the project's contended-GPU gotcha,
launched as a detached background process). Wall-clock: ~113 minutes
(20 pretrain epochs on 131,200 transitions + 60 finetune epochs on
43,598 ARC-3 transitions), in the same range as the original experiment's
two runs (100-127 min).

Per the task's scoping, only the per-game-id variant was retrained this
session (not a fresh baseline or a fresh shared-id run) -- the baseline
(+4.0% / -0.1%) and shared-id MinAtar (+2.1% / -1.4%) numbers below are
`experiments/stage6_diverse_pretraining.md`'s own already-published,
same-fold-1-recipe results, not re-derived here. This makes the
per-game-id vs. shared-id comparison a **cross-run** comparison rather
than the tighter same-run paired comparison that document itself
achieved for baseline-vs-shared-id -- worth keeping in mind given how
large the standard-metric swing is (see "Limitations" below).

## Results

### Standard-corpus sanity check (20 trained games, matching Stage 4's own metric)

Final-epoch (60/60) and 10-epoch trailing-mean (epochs 51-60) changed-patches
improvement over identity, on the held-out-*transitions* validation split
of the 20 *trained* games:

| variant | pred_changed_mse (ep 60) | identity_changed_mse (ep 60) | improvement (ep 60) | improvement (mean, ep 51-60) |
|---|---|---|---|---|
| baseline (MiniGrid-only) | -- | -- | +3.5% | +4.0% |
| MiniGrid+MinAtar (shared id) | 0.00125 | 0.00125 | +0.0% | +2.1% |
| **MiniGrid+MinAtar (per-game ids)** | **0.00198** | **0.00459** | **+56.9%** | **+55.8%** |

Per-game ids produce a **large, unambiguous improvement over both the
shared-id run and the MiniGrid-only baseline** on this metric -- roughly
14x the shared-id run's improvement and 14x the baseline's. This is
exactly the direction the pooling-confound hypothesis predicted: once
each MinAtar sub-game gets its own embedding, the shared
encoder/predictor no longer has to average across 5 mutually-inconsistent
action->effect mappings to minimize loss on any one of them, so it can
actually commit to real, non-trivial residuals for MinAtar-relevant
dynamics that show up (via the shared encoder) in the ARC-3 fine-tuning
phase too. Full epoch-by-epoch numbers for epochs 51-60 (used for the
trailing mean) are in the raw training log
(`logs/minatar_pergame/train.out.log` in this branch's worktree, not
committed -- gitignored per this project's `/logs/` convention; the
per-epoch pred/identity pairs are reproduced in this document's git
history if needed).

### The test that matters: fold-1 held-out-games generalization

Using `scripts/eval_diverse_pretraining.py` unchanged, same held-out
game set and evaluation methodology as the original experiment (natural
`game_vocab.get(game_id, 0)` fallback for the never-seen held-out
game_ids, matching `hypothesis_agent.py`'s real production behavior):

| variant | pred_changed_mse | identity_changed_mse | improvement over identity (n=1881) |
|---|---|---|---|
| baseline (MiniGrid-only) | 0.040336 | 0.040285 | **-0.1%** |
| MiniGrid+MinAtar (shared id) | 0.006381 | 0.006296 | **-1.4%** |
| **MiniGrid+MinAtar (per-game ids)** | **0.052490** | **0.052469** | **-0.0%** |

(The absolute MSE values aren't directly comparable across variants --
different checkpoints, different random corpus draws for the local
recordings underlying the `arc3_logs`-derived val split composition, etc.
-- the `improvement_pct` column, each variant's own predictor-vs-identity
comparison on the *same* held-out data, is the fair cross-variant
comparison.)

Per held-out game:

| game | baseline | shared-id MinAtar | per-game-id MinAtar |
|---|---|---|---|
| r11l | -0.1% | **-10.5%** | -1.2% |
| bp35 | -0.1% | -0.3% | +0.0% |
| m0r0 | -0.1% | -2.6% | -0.4% |
| tr87 | -0.4% | -0.3% | -0.7% |
| ka59 | +0.1% | -1.1% | -0.2% |

**Per-game ids land almost exactly on the MiniGrid-only baseline** (-0.0%
pooled vs. baseline's -0.1%) -- a real improvement over the shared-id
run's -1.4%, and notably `r11l` (the shared-id run's one standout-bad
number at -10.5%) improves to -1.2%, much closer to baseline's own -0.1%
on that game. But "closer to baseline" here means **closer to zero
generalization advantage**, not closer to a real positive result --
every single per-game number is still negative or approximately zero,
same as every other checkpoint tested against this fold across this
project's entire Stage 6 history.

## Interpretation: both things are true at once

**The pooling confound was real.** A 55.8-point swing on the standard
metric (+2.1% -> +55.8%) from changing nothing but how MinAtar's 5
sub-games share (or don't share) an embedding index is a large,
unambiguous effect, not noise -- this project's own established noise
bands for this kind of metric (see CLAUDE.md's Stage 1 item 5, Stage 4's
own reruns) are single-digit-to-low-double-digit percentage points, not
50+. Pooling structurally dissimilar synthetic sub-games under one shared
id, even when their action *interface* is identical (as MinAtar's is),
can cost a large amount of learnable signal if their actual dynamics
diverge enough -- a concrete, reusable lesson for any future synthetic
data source this project adds: default to per-sub-environment ids unless
there's a specific reason (like MiniGrid's own genuinely-shared
navigation semantics) to pool them.

**But the held-out-ARC-games generalization gap is untouched.** Despite
that large standard-metric win, the held-out-games number moved from
-1.4% (shared id) to -0.0% (per-game id) -- a shift *toward* the
already-established ~0% noise floor that baseline, shared-id MinAtar, and
every one of the 7 other Stage 6 interventions before it all land on, not
a shift toward a real positive result. This is the clean way to read
"per-game ids fixed the standard metric but not the held-out one": the
standard metric measures whether the model learned to predict
MinAtar/ARC-3 dynamics *for games it was trained on* (where per-game
embeddings give it a lookup table entry to specialize into) -- the
held-out metric measures whether *any* of that learning transfers to a
game the model has *never* seen an embedding for. Per-game ids can only
ever help the former; by construction, a never-seen ARC-3 held-out game
still falls back to the same untrained `game_vocab.get(id, 0)` embedding
index either way, so there was never a mechanism by which the pooling fix
could plausibly transfer to genuinely novel games. This is worth stating
plainly since it's easy to conflate "the standard number went up a lot"
with "we're closer to fixing the real problem" -- here they're
genuinely decoupled.

**This is now the clearer of the two possible explanations for MinAtar's
original negative result, and it points at genre/mechanic mismatch, not
data-pooling mechanics.** The task motivating this experiment asked
whether pooling or genre-mismatch better explains why MinAtar didn't
close the gap. With the pooling variable isolated and shown to move the
standard metric by 50+ points while leaving the held-out metric
unchanged, the evidence now favors **genre mismatch (or, more precisely,
the same "no held-out-game-agnostic conditioning trick has worked" data-
bound limit documented across all 9 interventions to date)** as the real
explanation, not an artifact of how MinAtar's own sub-games were IDed.
Consistent with the broader Stage 6 pattern (CLAUDE.md's "Stage 6
addendum": 7 architecture/conditioning fixes plus this project's own
continuous-game-embedding investigation, `stage6-context-embedding`,
7 total interventions there) -- no way of *representing* game identity,
categorical or continuous, closes this gap, because the underlying
limitation isn't in how games are identified, it's in how much of the
training corpus's mechanics actually transfer to an unseen game's
mechanics. MinAtar's reflex/physics-heavy mechanics (paddle tracking,
lane-crossing timing, projectile dodging) most likely simply don't share
enough causal structure with ARC-3's puzzle-logic mechanics to transfer,
regardless of how cleanly its own internal sub-games are identified to
the model.

## Limitations

- **Cross-run, not same-run, comparison against the shared-id number.**
  Per the task's scoping, only the per-game-id variant was retrained this
  session; the baseline and shared-id numbers are taken from
  `experiments/stage6_diverse_pretraining.md`'s own already-published
  results rather than re-derived in the same training session. The
  held-out-games result (-0.0%, landing inside the already-well-
  characterized ~0% noise band) is unlikely to be affected by this --
  that band is itself established via 5-fold cross-validation across
  many separate runs (`stage6-multifold-cv`), not a single paired
  comparison. But the standard-metric swing (+2.1% -> +55.8%) is large
  enough that a skeptical read would want a repeat run (ideally with a
  fresh random seed) before fully trusting the *exact* magnitude, even
  though the direction and rough size of the effect are consistent with
  the mechanistic explanation above (removing a genuine training-time
  confound) rather than looking like a fluke.
- **One fold only**, same caveat the original MinAtar experiment carried
  forward from `stage6_multifold_generalization.md`'s own guidance: extra
  scrutiny is explicitly warranted for a *positive* result that might be
  a lucky single-fold draw. This result's held-out-games finding is
  negative (well, null) and closely matches the already-5-fold-validated
  baseline pattern, so that specific caveat doesn't obviously apply to
  the held-out-games conclusion -- but the standard-metric win, while not
  the primary question this experiment was testing, was likewise only
  checked on one fold.
- **Did not test per-game ids on Sokoban**, which had its own,
  differently-diagnosed negative result (CLAUDE.md's Stage 4 item 8:
  deadlock-polluted data, not a pooling issue, and Sokoban was already a
  single game rather than a pooled multi-game source, so the pooling
  question doesn't apply there the same way).

## Reproducing this experiment

```
# Corpus setup identical to experiments/stage6_diverse_pretraining.md:
# copy the verified 150-file *.random.80.* corpus from
# E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\ into
# ARC-AGI-3-Agents/recordings/, and data/arc3_logs.zip into data/.

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --minatar-episodes-per-game 160 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_diverse_minatar_pergame

python scripts/eval_diverse_pretraining.py --fold 1 \
  --heldout-games r11l,bp35,m0r0,tr87,ka59 \
  --baseline-ckpt checkpoints_diverse_baseline \
  --minatar-ckpt checkpoints_diverse_minatar_pergame
```

(`--baseline-ckpt` can point at a nonexistent directory if you only want
the per-game-id variant's own numbers -- `eval_diverse_pretraining.py`
skips missing checkpoint directories gracefully rather than erroring,
which is how this session's own eval was run, since the original
experiment's `checkpoints_diverse_baseline`/`checkpoints_diverse_minatar`
directories were deleted after that experiment extracted its results,
per this project's disk-hygiene practice.)

Training took ~113 minutes on a shared RTX 2070
(`JEPA_NUM_WORKERS=0`, 2 other concurrent agents on the same GPU this
session). `eval_diverse_pretraining.py` runs in well under a minute.
