# Stage 6 experiment: does the object-identity checkpoint's advantage survive on unseen games?

**Status: DONE. The object-identity checkpoint's representation advantage
does NOT survive on held-out games -- it doesn't just shrink, it reverses
sign, landing at a smaller magnitude than the baseline's own (also
negative) held-out gap.** This is a genuine negative result for the
contrastive loss's generalization, and a plausible (though not provable
from local data alone) explanation for the checkpoint's real Kaggle
score of 0.00.

## Motivation

`stage6-object-identity`'s same-color contrastive loss produced a huge,
carefully-verified local win (object-identity cosine-similarity gap
+0.0114 -> +1.1564; beats production on changed-patches on 18+/25
games) but its first real Kaggle submission scored **0.00** -- the
worst of six real submissions, below the previously-assumed floor
(0.06, the plain random-agent control's own score). A setup/checkpoint-
loading crash was already ruled out directly (a diagnostic Kaggle kernel
loads this exact checkpoint set and runs a full forward pass with zero
errors).

A real methodology gap was found while investigating: **every local
evaluation in this project -- `train_moe_predictor.py`'s train/val
split, `jepa/benchmark.py eval`, every backtest scorecard -- splits by
individual TRANSITION, never by GAME.** Every number ever computed for
any checkpoint has been validated on transitions drawn from the *same 25
local games* the model trained on. There has never been a true test of
"how does this checkpoint do on a game it's never seen" -- exactly the
axis Kaggle's ~110-largely-novel-hidden-game evaluation stresses. The
specific worry: the same-color contrastive loss might be learning the
*local* 25 games' own specific color statistics/usage patterns rather
than a genuinely transferable "same color = same object" notion.

This experiment builds a leave-N-games-out generalization test: train a
"production-style" (no contrastive loss) and an "object-identity-style"
(contrastive loss on) checkpoint on the identical 20-game corpus (5
games held out entirely, never seen by either checkpoint, local or
external), then evaluate both ONLY on the 5 held-out games -- the best
local proxy available for "hidden Kaggle games."

## Held-out games: selection and reasoning

Held out: **`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`** (5 of the 25 local
games), chosen for a representative mix rather than all-easy or
all-hard:

- **`r11l`, `bp35`, `m0r0`** -- three of the most frequently-recurring
  "commonly solved" games across this project's entire history (Stage
  2/3/5's agent comparisons, Stage 1's single-game ablation target for
  `bp35` specifically for its 100% frame-level changed rate). If the
  contrastive loss's advantage is real and general, it should still show
  up on these -- they're not exotic, just excluded from training.
- **`tr87`, `ka59`** -- two games that **never once appear** in any
  "solved"/"commonly touched" list anywhere in CLAUDE.md's entire
  history (checked via a full grep of every game code across every
  Stage 1-5 section) -- the closest local proxy to a genuinely
  under-characterized game, closer in spirit to a novel hidden Kaggle
  game than any of the well-studied five above.

This selection also matches `scripts/diagnose_encoder_vs_predictor.py`'s
own `PROBE_GAMES` list (`["ft09", "m0r0", "r11l", "bp35", "s5i5", "tr87",
"ka59", "vc33"]`), so diagnostic B's existing per-game structure could be
reused directly by restricting to these 5, rather than re-deriving a
fresh probe-game list.

## Corpus and training recipe

Both checkpoints trained via `jepa/train_moe_predictor.py` (this
branch's modifications: `--exclude-games`, `--contrast-weight`, ported
`--checkpoint-every`/`JEPA_NUM_WORKERS` resilience infra) on the
**identical** corpus:

- Local recordings: the verified 150-file `*.random.80.*` corpus
  (`E:\ARC-AGI-3-JEPAstyle_data\recordings_archive\`, the same corpus
  `scripts/diagnose_encoder_vs_predictor.py` treats as ground truth),
  minus the 5 held-out games' 30 files (6 files/game) -- **9,600
  transitions across the remaining 20 games** (verified: 150 - 30 = 120
  files x 80 transitions/file = 9,600).
- External `arc-3-logs` augmentation: `--external-per-game 2000`,
  matching production's own real recipe (`checkpoints/
  moe_training_meta.json`: `external_per_game: 2000`) -- also excluding
  the 5 held-out games from this source (it covers the same 25 ARC
  games, so a leave-games-out experiment must exclude it there too, not
  just from local recordings).
- MiniGrid pretrain: `--pretrain-epochs 20`, 67,200 transitions across
  21 environments -- unaffected by the game-holdout question (MiniGrid
  has no overlap with any ARC-3 game).
- `--epochs 60 --num-experts 8`, warm-started from `checkpoints/
  encoder.pt` -- matching production's own real curriculum (Stage 4
  item 6) exactly, so the *only* deliberate difference between the two
  checkpoints trained here is `--contrast-weight` (0.0 vs 0.05, the
  object-identity experiment's own value).

Game vocabulary: 20 ARC games + `"minigrid"` = 21 entries. The 5
held-out games are **not** in either checkpoint's vocabulary at all.

Commands:
```
python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_holdout_baseline

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.05 --checkpoint-every 5 --out checkpoints_holdout_objid
```
(`JEPA_NUM_WORKERS=0` set for both -- this box's GPU and RAM are shared
with two other concurrent agent sessions, and CLAUDE.md's own gotcha
section documents a real `MemoryError`/full-disk failure mode from
multi-worker DataLoaders under memory pressure; single-process loading
is slower but doesn't depend on pagefile/RAM headroom staying free.)

## Evaluation methodology

`scripts/eval_game_holdout.py` (new): loads transitions from the 5
held-out games ONLY (via `load_all_transitions`'s existing
`name_substrings` filter, reused as an include-list), then for each
checkpoint:

1. **changed-patches** (pred-vs-identity MSE on changed 8x8 patches),
   overall (pooled across all 5 held-out games) and per-game. No
   train/val split needed -- since neither checkpoint ever trained on
   any held-out-game transition, the *entire* held-out population is
   valid eval data, not just a 10% slice.
2. **Diagnostic B** (same-color vs different-color patch cosine
   similarity gap, from `scripts/diagnose_encoder_vs_predictor.py`),
   restricted to the 5 held-out games' frames only -- tests whether the
   *representation* advantage itself (not just prediction accuracy)
   generalizes to unseen games.

Held-out games' `game_id` is not in either checkpoint's `game_vocab`, so
lookups fall back to index 0 (`defaultdict(int, game_vocab)`) --
**deliberately mirroring** `ARC-AGI-3-Agents/agents/templates/
hypothesis_agent.py`'s real production behavior on a genuinely novel
Kaggle game (`self.game_idx = game_vocab.get(self.game_id, 0)`), so this
eval faithfully simulates the real hidden-game situation rather than
silently cheating with a held-out game's own (nonexistent) trained
embedding.

## Results

Both checkpoints trained cleanly to completion (60/60 arc-finetune
epochs, no crashes, `JEPA_NUM_WORKERS=0` throughout on a shared GPU with
two other concurrent agent sessions). Confirmed via `moe_training_meta.json`
that the two checkpoints are identical in every recorded setting except
`contrast_weight` (0.0 vs 0.05): same 9,600 local + 33,998 external
transitions, same 67,200 MiniGrid pretrain transitions, same 21-entry
game vocab, same `exclude_games` list. Confirmed via `scripts/
eval_game_holdout.py`'s own vocab check that **0/5 held-out games are
present in either checkpoint's `game_vocab`** -- a genuine holdout, not
an accidental leak.

### 1. Changed-patches (prediction quality) on the 5 held-out games

Pooled across all 2,400 held-out-game transitions (480/480/338/281/302
per game respectively):

| checkpoint | pred_changed_mse | identity_changed_mse | improvement |
|---|---|---|---|
| baseline (no contrastive loss) | 0.065562 | 0.065571 | **+0.0%** |
| object-identity (contrast_weight=0.05) | 0.042601 | 0.042602 | **+0.0%** |

Per-game breakdown (both checkpoints land at essentially the same
near-zero improvement on every single held-out game):

| game | baseline improvement | object-identity improvement |
|---|---|---|
| `r11l` | -1.6% | +0.0% |
| `bp35` | +0.0% | +0.1% |
| `m0r0` | +0.2% | +0.0% |
| `tr87` | +0.2% | +0.0% |
| `ka59` | -0.9% | -0.3% |

**Neither checkpoint beats identity on any held-out game, and neither has
a meaningful edge over the other here.** Both collapse to ~identity
parity the moment the game embedding falls back to index 0 (the same
`game_vocab.get(self.game_id, 0)` fallback `hypothesis_agent.py` uses on
a real novel Kaggle game) -- this specific metric doesn't distinguish the
two checkpoints on unseen games at all; it shows the *entire* MoE
predictor's prediction-quality edge (not just the contrastive-loss part
of it) evaporating on games neither ever trained on. That's a broader
finding about this project's per-game-embedding approach's own
generalization limits, not something specific to the contrastive loss.

### 2. Diagnostic B (object-identity cosine-similarity gap)

This is where the two checkpoints diverge sharply.

**On the 5 held-out games** (never trained on by either checkpoint):

| checkpoint | same-color cos sim | diff-color cos sim | gap |
|---|---|---|---|
| baseline | 0.3437 (n=9,857) | 0.4090 (n=124) | **-0.0654** |
| object-identity | 0.5608 (n=9,857) | 0.5770 (n=124) | **-0.0163** |

**On 3 of the 20 *trained* games** (`ft09`, `s5i5`, `vc33` -- sampled as
a same-checkpoint, same-methodology control to see what each
checkpoint's gap looks like on data it actually saw during training):

| checkpoint | same-color cos sim | diff-color cos sim | gap |
|---|---|---|---|
| baseline | 0.5228 (n=2,588) | 0.4754 (n=42) | **+0.0474** |
| object-identity | 0.9969 (n=2,588) | -0.1837 (n=42) | **+1.1805** |

The object-identity checkpoint's trained-game gap (**+1.1805**) closely
reproduces the original stage6-object-identity experiment's own
headline number (**+1.2003**, measured on a different -- full 25-game,
local-only -- corpus) -- a good consistency check that this branch's
independently-retrained checkpoint reproduces the same real effect, not
an artifact of that experiment's specific data mix.

**On held-out games, that same +1.1805 gap does not shrink toward zero
-- it flips sign, landing at -0.0163.** That's a smaller-magnitude
negative gap than the baseline's own held-out gap (-0.0654), so the
contrastive loss did leave *some* residual trace on unseen games (the
gap is less negative than the untrained-on-object-identity baseline) --
but the intended effect (same-color patches looking *more* alike than
different-color ones) is entirely absent on games neither checkpoint
ever saw. The baseline's own held-out gap being negative too (patches of
the same color look *less* alike than different-color patches on unseen
games) suggests the vanilla MoE encoder's representations are already
somewhat incoherent on totally novel games regardless of the contrastive
loss -- consistent with the changed-patches result above showing the
same "everything degrades toward parity on unseen games" pattern.

Sample-size caveat, stated plainly: the diff-color pool is much smaller
(n=42-124) than the same-color pool (n=2,588-9,857) in every condition
-- this mirrors `diagnose_encoder_vs_predictor.py`'s own diagnostic B
methodology (diff-color pairs are capped at 10 samples per frame by
design, since same-color positions are typically far more numerous per
frame than cross-color pairs). The *direction* is nonetheless consistent
and the magnitudes are well outside the ~±0.01 "noise level" this
project's own earlier measurements established for production's
original (near-zero) gap -- so this reversal reads as a real effect, not
sampling noise, even with the smaller diff-color sample.

Full raw numbers: `logs/game_holdout_results.json` (per-checkpoint
changed-patches + diagnostic B on held-out games) plus the
trained-games control measurement reproduced in this section.

## Honest read

**The core question this experiment was built to answer: does the
object-identity checkpoint's advantage over production survive on
genuinely unseen games? No -- it collapses, and on the representation
metric that mattered most (diagnostic B), it doesn't just shrink, it
reverses sign.** A checkpoint that shows same-color patches as *nearly
identical* (cos sim 0.997) and different-color patches as *strongly
dissimilar* (cos sim -0.18) on the games it trained on shows almost the
opposite pattern on games it never saw (same-color 0.56, diff-color
0.58, same-color patches marginally *less* similar). This is consistent
with -- though doesn't by itself prove -- the hypothesis stated in this
experiment's motivation: **the same-color contrastive loss, as currently
implemented, appears to be learning the training games' own specific
color usage/layout statistics rather than a genuinely transferable
"same color implies same object" abstraction.** A loss that pulls
same-color patches together using only within-frame supervision has no
explicit pressure to generalize what "same object" means across
visually different game boards -- and this result is exactly what that
theoretical gap would predict if it were biting in practice.

On the changed-patches (prediction-quality) metric, there's no
meaningful difference between the two checkpoints on held-out games --
both are at ~identity parity, a separate and broader finding that the
whole per-game-embedding MoE approach doesn't have any prediction edge
left once a game's embedding is a fallback rather than something it
actually learned. This isn't a contrastive-loss-specific problem; it's
a limitation of the game-conditioning design itself on truly novel
games, worth flagging for any future stage that wants to improve hidden-
game generalization more broadly (e.g. game-agnostic conditioning, or a
few-shot in-context adaptation mechanism, rather than a fixed
per-game-id embedding table that's meaningless for an id the model has
never seen).

**Does this "explain" the Kaggle 0.00?** Consistent with it, not proof
of it. This experiment shows the *representation* advantage the
checkpoint was promoted on doesn't transfer to unseen games in the
direction hoped for -- which is exactly the kind of gap that could turn
a checkpoint that looks decisively better locally into one that performs
no better (or, per this held-out evidence, arguably with *less*
consistent representations than an untouched baseline) on Kaggle's
~110-largely-novel hidden games. But a single real submission (0.00) sits
inside a documented 0.06-0.23 noise floor for *this project's other*
checkpoints at the same config, and this experiment doesn't run the
object-identity checkpoint itself through a real or simulated hidden-game
episode -- it measures the encoder's representation and the predictor's
one-step accuracy, not full-episode agent behavior. Treating this as
"the found root cause" would overclaim; treating it as "a genuine,
now-confirmed generalization failure in the specific mechanism that
checkpoint was promoted on, and a plausible contributing factor to its
weak real score" is the honest, supported claim this experiment actually
provides evidence for.

**Follow-up worth flagging (not done here, stretch goal per this
experiment's own scope):** a matched local `Hypothesis`-agent backtest
scorecard restricted to just the 5 held-out games (baseline-holdout vs.
object-identity-holdout checkpoints) would test whether this
representation-level finding translates into full-episode behavioral
differences -- skipped here to keep this experiment's core deliverable
(the transition-level generalization test) the priority within the
available time budget, per the task's own stretch-goal framing.

## Reproducing this experiment

```
# Corpus setup (once): copy the verified 150-file *.random.80.* corpus into
# ARC-AGI-3-Agents/recordings/, and data/arc3_logs.zip into data/ -- both
# gitignored, see CLAUDE.md's environment-setup section for where to get them.

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.0 --checkpoint-every 5 --out checkpoints_holdout_baseline

python -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 \
  --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
  --contrast-weight 0.05 --checkpoint-every 5 --out checkpoints_holdout_objid

python scripts/eval_game_holdout.py
```
(`JEPA_NUM_WORKERS=0` recommended on a shared/contended GPU box, per
CLAUDE.md's own gotcha.) Each training run took roughly 80-90 minutes on
a shared RTX 2070 (contended with two other concurrent agent sessions);
`eval_game_holdout.py` runs in under a minute.
