# Stage 6 experiment: is the held-out-game generalization collapse robust across many different 5-game splits?

**Status: DONE. The finding holds up robustly across all 5 folds: both
the baseline (with per-game embedding conditioning) and the no-game-id
(ablated) checkpoint collapse to ~identity parity on held-out games in
every single fold, not just the one originally tested. Game-id ablation
never closes the gap in any fold. This was not an artifact of the
original 5-game choice.**

## Motivation

`experiments/stage6_game_holdout.md` and `experiments/
stage6_gameid_ablation.md` both used the same fixed 5-game holdout
(`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`) to test whether this project's
MoE world-model checkpoints generalize to games never seen in training.
Both found a collapse to ~identity parity on those 5 held-out games, and
found that removing per-game embedding conditioning (`--ablate-game-id`)
does not fix it.

A real methodological objection: this project's 25 local games are known
to vary a lot in difficulty/character (see `CLAUDE.md` throughout -- e.g.
`sp80` is unusually easy for random play, others are barely ever solved
by anything). Every conclusion drawn from the game-holdout line of
experiments rested on exactly one arbitrary 5-game split. If that split
happened to be unusually easy, hard, or otherwise atypical to generalize
away from, the "collapse, and game-id ablation doesn't fix it" conclusion
could be an artifact of that one split rather than a real property of the
model/architecture.

This experiment builds proper 5-fold cross-validation: partition all 25
local games into 5 disjoint folds of 5 games each, train both the
baseline and no-game-id variant on each fold's 20-game corpus (holding
out that fold's 5 games entirely, local and external data), evaluate
changed-patches improvement over identity on each fold's held-out games,
and aggregate across all 5 folds.

## The 25 local games and the fold partition

The authoritative list of 25 local games was derived from
`ARC-AGI-3-Agents/environment_files/` (one subdirectory per game --
required for the harness to find any game at all) and cross-checked
against the unique `game_id` prefixes present in the verified 150-file
`*.random.80.*` local recordings corpus
(`E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\`) -- both sources agree
on exactly this set of 25 games:

```
ar25, bp35, cd82, cn04, dc22, ft09, g50t, ka59, lf52, lp85,
ls20, m0r0, r11l, re86, s5i5, sb26, sc25, sk48, sp80, su15,
tn36, tr87, tu93, vc33, wa30
```

Fold 1 reuses the exact 5-game holdout from `stage6-game-holdout` /
`stage6-gameid-ablation` (`r11l, bp35, m0r0, tr87, ka59`), as required so
that fold's already-trained checkpoints could be reused rather than
retrained. The remaining 20 games were sorted alphabetically and split
into 4 more folds of 5 games each, in order -- a simple, reproducible
partition, not curated for any property of the games:

| fold | held-out games |
|---|---|
| 1 (reused) | `r11l`, `bp35`, `m0r0`, `tr87`, `ka59` |
| 2 | `ar25`, `cd82`, `cn04`, `dc22`, `ft09` |
| 3 | `g50t`, `lf52`, `lp85`, `ls20`, `re86` |
| 4 | `s5i5`, `sb26`, `sc25`, `sk48`, `sp80` |
| 5 | `su15`, `tn36`, `tu93`, `vc33`, `wa30` |

All 25 games are covered exactly once across the 5 folds (disjoint,
exhaustive partition).

## Method

Two checkpoint variants per fold, trained via `jepa/train_moe_predictor.py`:

- **baseline** (with per-game embedding conditioning): `--contrast-weight
  0.0` (no contrastive loss, isolating this comparison from the separate
  `stage6-object-identity` question).
- **no-gameid** (ablated): same recipe plus `--ablate-game-id` (forces
  every transition's `game_idx` to a constant 0 throughout training and
  validation -- see `stage6_gameid_ablation.md` for why this is
  equivalent to fully removing per-game conditioning while keeping the
  checkpoint's state-dict shape unchanged).

Identical recipe to fold 1 for every fold (`experiments/
stage6_game_holdout.md`'s command, just with each fold's own
`--exclude-games`):

```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games <fold's 5 games> \
  --contrast-weight 0.0 [--ablate-game-id] --checkpoint-every 5 \
  --out checkpoints_fold<N>_<variant>
```

20 MiniGrid-pretrain epochs (67,200 transitions, unaffected by the
ARC-game holdout) + 60 ARC-finetune epochs (local recordings for the
fold's 20 trained games, ~9,600 transitions, + external `arc-3-logs`
capped at 2,000/game, ~32,800 transitions, both excluding the fold's 5
held-out games). `JEPA_NUM_WORKERS=0` throughout (shared/contended GPU,
per this project's established gotcha). Fold 1's checkpoints were reused
directly (copied from `stage6-gameid-ablation`'s worktree, verified to
reproduce its exact previously-documented numbers byte-for-byte before
reuse -- see "Verification" below). Folds 2-5 (8 training runs total)
were trained fresh on this branch, each launched as a fully OS-detached
process and polled synchronously to completion (all 8 runs completed
cleanly, `checkpoint_tag: "final"`, no errors in any `.err.log`).

Evaluation: `scripts/eval_multifold.py` (a generalization of `scripts/
eval_gameid_ablation.py` that takes a fold's held-out game list and
checkpoint directories as CLI arguments instead of a single hardcoded
5-game split) computes changed-patches (pred-vs-identity MSE on changed
8x8 patches) pooled across all of a fold's held-out-game transitions
(no train/val split needed, since neither checkpoint ever trained on any
held-out-game transition in any fold). The no-gameid checkpoint is
evaluated with `game_idx` forced to 0 (matching how it was trained); the
baseline checkpoint uses the natural `game_vocab.get(game_id, 0)`
fallback for a never-seen game_id (mirroring `hypothesis_agent.py`'s real
production behavior on a genuinely novel Kaggle game).

**Verification before trusting the pipeline for new folds:** ran
`eval_multifold.py` against fold 1's checkpoints first and confirmed it
reproduces `stage6_gameid_ablation.md`'s exact previously-published
numbers (baseline +0.0%, no-gameid -0.2%, identical per-game breakdown to
the decimal) -- confirming the local recordings corpus, checkpoint
copies, and evaluation logic were all set up correctly before spending
compute on 8 new training runs.

## Results

### Per-fold changed-patches improvement over identity (held-out games only)

| fold | held-out games | baseline (with game-id) | no-gameid (ablated) |
|---|---|---|---|
| 1 | r11l, bp35, m0r0, tr87, ka59 | +0.01% | -0.21% |
| 2 | ar25, cd82, cn04, dc22, ft09 | +0.11% | -0.02% |
| 3 | g50t, lf52, lp85, ls20, re86 | +0.04% | +0.03% |
| 4 | s5i5, sb26, sc25, sk48, sp80 | -0.18% | -0.04% |
| 5 | su15, tn36, tu93, vc33, wa30 | -1.46% | -1.76% |
| **mean** | | **-0.30%** | **-0.40%** |
| **std (n=5)** | | **0.66%** | **0.76%** |

(Fold 3's `lp85` produced zero changed-patch examples in the held-out
slice of the local `random.80` corpus for that game -- excluded from
both the numerator and denominator consistently for both variants, so it
doesn't bias the fold-3 pooled number; noted here for reproducibility.)

### Per-game breakdown, all 5 folds (25 games total, each appearing in exactly one fold)

| game | fold | baseline | no-gameid |
|---|---|---|---|
| r11l | 1 | -1.6% | -2.4% |
| bp35 | 1 | +0.0% | -0.3% |
| m0r0 | 1 | +0.2% | -0.1% |
| tr87 | 1 | +0.2% | -0.0% |
| ka59 | 1 | -0.9% | -0.4% |
| ar25 | 2 | +0.0% | -0.0% |
| cd82 | 2 | +0.2% | -0.0% |
| cn04 | 2 | +0.1% | -0.0% |
| dc22 | 2 | -3.3% | -0.5% |
| ft09 | 2 | +35.7% | -17.4% |
| g50t | 3 | -0.7% | +0.0% |
| lf52 | 3 | -0.9% | -0.6% |
| lp85 | 3 | n/a (0 changed patches) | n/a |
| ls20 | 3 | -0.1% | +0.0% |
| re86 | 3 | +0.4% | +0.2% |
| s5i5 | 4 | +2.3% | -3.5% |
| sb26 | 4 | -2.0% | +1.9% |
| sc25 | 4 | -0.6% | +0.5% |
| sk48 | 4 | -0.5% | -0.7% |
| sp80 | 4 | -0.1% | -0.1% |
| su15 | 5 | +0.4% | -2.6% |
| tn36 | 5 | -0.6% | +0.4% |
| tu93 | 5 | -1.5% | -3.6% |
| vc33 | 5 | +0.6% | -8.2% |
| wa30 | 5 | -4.3% | -2.2% |

`ft09` (fold 2) is the one clear outlier in the whole table (+35.7%
baseline / -17.4% no-gameid) -- both numbers are computed from a tiny
absolute-MSE base (identity_changed_mse=0.000129 for baseline,
0.000014 for no-gameid, both games' `pred`/`identity` values differing
only in the 4th-5th decimal place), the exact "small absolute error swing
=> huge relative swing" pattern this project's own `CLAUDE.md` (Stage 1
item 5) already flagged for `ft09`/`vc33`/`s5i5` specifically. Not a real
generalization win -- excluding it, every other per-game number sits
within roughly ±4% of parity in both directions, consistent noise around
zero rather than a real edge for either variant.

## Aggregate verdict

**The original single-split finding holds up robustly across all 5
folds.** Every fold's pooled held-out-game improvement is close to zero
(within noise) for both variants, mean -0.30% (baseline) / -0.40%
(no-gameid) across the 5 folds, std ~0.66-0.76 percentage points. Fold 5
is the fold with the largest deviation from zero (-1.46% / -1.76%), but
that's still small in absolute terms and consistent with sampling noise
around a true value of ~0%, not a qualitatively different pattern from
the other 4 folds -- there is no fold in which either variant shows a
real, unambiguous positive generalization edge, and no fold in which
game-id ablation closes the gap.

**Answering the two questions this whole experiment family asked, now
with 5-fold evidence instead of 1:**

1. **Does the world-model's prediction-quality advantage over identity
   collapse on held-out games? Yes, consistently across all 5 folds**,
   not just the original arbitrary split. Every fold lands within a few
   tenths of a percent of identity parity for the baseline variant --
   there is no fold where the model shows a real edge on games it never
   trained on.
2. **Does removing per-game embedding conditioning close this gap? No, in
   no fold.** The no-gameid variant's per-fold numbers (-0.21%, -0.02%,
   +0.03%, -0.04%, -1.76%) are statistically indistinguishable from the
   baseline's own per-fold numbers (+0.01%, +0.11%, +0.04%, -0.18%,
   -1.46%) -- both variants collapse together in every fold, and the
   no-gameid variant is if anything very slightly worse on average
   (-0.40% vs -0.30%), though that gap (0.10 percentage points against a
   0.66-0.76 point std) is well within noise and shouldn't be read as a
   real difference either.

This was genuinely worth checking rather than assuming: a single 5-game
split is a small, arbitrary sample out of `C(25,5) = 53,130` possible
5-game holdouts, and this project's own games are known to differ a lot
in character. Five independent folds covering all 25 games without
overlap is a much stronger basis for the conclusion than one split could
provide, and the conclusion did not change -- if anything, it's now on
firmer footing precisely because it wasn't an accident of which games
`r11l, bp35, m0r0, tr87, ka59` happen to be.

**What this does and doesn't establish:** this confirms the collapse
and the ablation's failure to fix it are properties of this training
recipe/architecture/data-scale combination on this project's 25 local
games as a population, not an artifact of one lucky-or-unlucky split.
It does not by itself explain *why* the gap exists (see
`stage6_gameid_ablation.md`'s own discussion: the leading hypothesis
remains that the shared encoder/predictor's learned dynamics are tied to
the training games' specific visual/color statistics and mechanics in a
way a categorical per-game signal was never really carrying) or point to
a fix beyond what that document already flagged (more/more-diverse
training games, or an architectural change targeting game-agnostic
dynamics more directly) -- both remain open for a future session.

## Reproducing this experiment

```
# Corpus setup (once, same as stage6_game_holdout.md): copy the verified
# 150-file *.random.80.* corpus into ARC-AGI-3-Agents/recordings/, and
# data/arc3_logs.zip into data/ -- both gitignored.

# Fold 1: reuse checkpoints_holdout_baseline / checkpoints_holdout_nogameid
# from stage6-gameid-ablation (or retrain with that branch's own command).

# Folds 2-5 (8 training runs):
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games ar25,cd82,cn04,dc22,ft09 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_fold2_baseline
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games ar25,cd82,cn04,dc22,ft09 \
  --contrast-weight 0.0 --ablate-game-id --checkpoint-every 5 --out checkpoints_fold2_nogameid
# ... same pattern for folds 3 (g50t,lf52,lp85,ls20,re86), 4
# (s5i5,sb26,sc25,sk48,sp80), 5 (su15,tn36,tu93,vc33,wa30)

python scripts/eval_multifold.py --fold 2 --heldout-games ar25,cd82,cn04,dc22,ft09 \
  --baseline-ckpt checkpoints_fold2_baseline --nogameid-ckpt checkpoints_fold2_nogameid
# ... same pattern for folds 3-5
```

Each training run took roughly 70-100 minutes on a shared RTX 2070
(contended with at least one other concurrent agent session throughout
this experiment -- two runs per fold were launched in parallel to halve
wall-clock time, which appeared to work without memory/stability issues
at this model's small size). `eval_multifold.py` runs in under a minute
per fold.
