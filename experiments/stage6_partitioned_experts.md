# Stage 6 experiment: does hard-partitioning MoE experts by training game close the held-out-games gap?

**Status: COMPLETE. Clean negative on the central question (held-out generalization is unchanged), but a real, unambiguous positive on the narrower mechanism question (does forced partitioning spread genuine competence across more of the 8 experts) -- at a severe, unacceptable cost to trained-game prediction quality.**

## Motivation

CLAUDE.md's Stage 6 addendum ("held-out-game generalization") established
that the MoE predictor's gated prediction has no measurable edge over
"predict no change" on any local game it wasn't trained on -- confirmed
robust across 5-fold cross-validation (fold means around -0.30% +/-
0.66%). By the time this experiment started, 13 independent
interventions (game-id ablation, encoder audits, anti-collapse losses,
three genuinely different continuous game-embedding schemes, five
diverse-pretraining-data attempts up to 2.14M synthetic transitions) had
all failed to close this gap; only test-time adaptation (real gradient
updates during play) showed any real, if modest, positive signal.

Two directly relevant diagnostic facts, both already established before
this experiment:

1. **Raw per-expert disagreement does NOT collapse on held-out games.**
   `scripts/diagnose_infogain_holdout.py` measured variance across the 8
   experts' raw (pre-gate) predictions on held-out vs trained-game
   states: ratio 0.999 -- essentially identical. The 8 experts keep
   producing genuinely different raw outputs on unseen games.
2. **What collapses is the GATED blend.** The gated residual magnitude
   collapses to ~0.000 on held-out games (vs 0.235 baseline on trained
   games) -- the model coasts on the identity skip-connection
   specifically when the game-id conditioning is unfamiliar. Separately
   (Stage 4's own finding), gate specialization is "a minority behavior,
   not the norm" even on TRAINED games -- mean gate entropy sits at
   ~98-99% of the exactly-uniform maximum.

**Working theory tested here:** near-uniform blending of genuinely
diverse-but-uncoordinated expert opinions nets out close to zero
(regression to the mean) -- a different failure than "the experts went
dead." The gate has no structural incentive to ever prefer one expert
over another, because nothing in training ties any expert to a
distinguishable competence region. If each expert is instead forced to
specialize on a fixed subset of games during training, the gate has
something concrete to learn to route on, and might learn a genuinely
content-based (not just game-id-lookup-based) routing rule that
generalizes to a held-out game whose id it's never seen.

**A directly relevant diagnostic run mid-task, by the coordinator**
(`scripts/diagnose_expert_specialization.py`, on the *production*
checkpoint, all 25 trained games): for 119 (game, action) contexts with
>=16 changed-transition samples, split-half best-expert agreement was
88.2% vs a 13.0% chance baseline -- a real, stable best-expert-per-context
signal exists, not noise. But the winner distribution is heavily
concentrated: expert 3 wins 61.3% of contexts, expert 2 wins 32.8%, and
the other 6 experts combined win only ~6%. So the untouched 8-expert
ensemble isn't 8 distinct causal hypotheses -- it's closer to 2
generally-competent experts plus 6 that are almost never the best
predictor anywhere. This sharpens the question this experiment asks:
even if hard-partitioning doesn't close the held-out-games gap, does it
at least spread real predictive competence across more of the 8 experts
(a more even winner distribution), or does it just relocate the same
"2 dominate, 6 are dead weight" pattern onto different game clusters?
Both the headline held-out number and this split-half diagnostic are
reported below.

## Mechanism

Builds `jepa/train_partitioned_moe.py`, based on `jepa/
train_moe_predictor.py`, changing ONLY the ARC-3 finetune phase (the
MiniGrid pretrain phase is left exactly as-is -- MiniGrid's single shared
`game_id="minigrid"` can't be meaningfully hard-partitioned, and changing
pretrain too would confound the one variable this experiment tests):

1. **Fixed game -> expert assignment.** The 20 fold-1 training games
   (5 held out: `r11l`, `bp35`, `m0r0`, `tr87`, `ka59`, exactly matching
   `stage6-game-holdout`) are deterministically round-robin-assigned to
   8 expert clusters (`jepa/train_partitioned_moe.py:
   build_game_to_expert` -- sorted game-id list, `expert = i % 8`).
   `num_experts=8` kept fixed to match the established architecture for
   gate-entropy comparability against Stage 4/Stage 6's existing numbers.
2. **Hard-routed main task loss.** For each ARC finetune example, the
   main prediction loss uses ONLY the assigned expert's raw
   (skip-connected) output (`MoEPredictor.predict_all_experts`, gathered
   per-example by assigned-expert index) -- not the gate-blended
   combination. Only that expert's parameters (+ the shared encoder)
   receive gradient from the main task loss for a given example.
3. **Routing-supervision auxiliary loss.** Since the main loss no longer
   touches the gate, a separate cross-entropy loss between the gate's
   logits (`MoEPredictor.gate_logits`, a new lightweight method that
   computes gate logits without paying for expert computation) and the
   known one-hot assignment trains the gate directly.
4. **Game-id dropout on the routing loss only.** 25% of routing-loss
   forward passes see `game_idx` zeroed out (the same fallback index a
   genuinely unseen game gets in production) before computing gate
   logits for the auxiliary loss -- forces the gate to predict the
   correct expert from pooled visual features alone on a fraction of
   steps, discouraging a pure `game_id -> expert` lookup-table shortcut
   that would be structurally useless on a held-out game's untrained
   embedding index. (Note: CLAUDE.md documents an earlier, different
   game-id-dropout attempt, `stage6-residual-commitment-fix`'s
   anti-collapse hinge loss, that didn't help in a different context --
   this is a different mechanism, routing supervision rather than a
   residual-magnitude hinge, tried here on its own merits.)
5. **Soft gate at eval/inference time, no hard override.** `evaluate()`
   is unchanged from `train_moe_predictor.py` -- uses the normal learned
   soft gate. Hard routing is a training-time-only mechanism, matching
   the realistic deployment path.

Loss-magnitude sanity check (per this project's own established practice
of checking auxiliary-loss scale before committing to a full run, given
Stage 4's load-balance-weight history): a 1-epoch, local-only smoke test
showed `main_loss=0.0623`, raw `routing_loss=1.7357` (weighted
contribution `0.01 * 1.7357 = 0.017`, the same order of magnitude as the
main loss, not swamping it) and `routing_acc=0.461` after just one epoch
(vs. a random baseline of 1/8=12.5%) -- confirmed the routing signal is
learnable and the `ROUTING_LOSS_WEIGHT=0.01` calibration was reasonable
before launching the full run.

## Corpus and training recipe

Identical fold-1 recipe to `stage6-game-holdout`'s own baseline command,
for direct comparability:

```
python -m jepa.train_partitioned_moe --pretrain-epochs 20 --epochs 60 \
    --num-experts 8 --external-per-game 2000 \
    --exclude-games r11l,bp35,m0r0,tr87,ka59 \
    --checkpoint-every 5 --out checkpoints_partitioned_experts
```

- Local recordings: the verified 150-file `*.random.80.*` corpus, minus
  the 5 held-out games' 30 files -- 9,600 transitions across the
  remaining 20 games.
- External `arc-3-logs` augmentation: `--external-per-game 2000`,
  excluding the same 5 held-out games.
- MiniGrid pretrain: `--pretrain-epochs 20`, 67,200 transitions across 21
  environments, unchanged mechanism (dense gate, no hard-partitioning).
- `--epochs 60 --num-experts 8`, warm-started from `checkpoints/
  encoder.pt` (base pretrained-on-ARC-1/2 encoder, same starting point as
  every other MoE checkpoint in this project's history).

Round-robin game -> expert assignment (20 training games / 8 experts):

```
expert 0: ar25-0c556536, ls20-9607627b, tn36-ef4dde99
expert 1: cd82-fb555c5d, re86-8af5384d, tu93-0768757b
expert 2: cn04-2fe56bfb, s5i5-18d95033, vc33-5430563c
expert 3: dc22-fdcac232, sb26-7fbdac44, wa30-ee6fef47
expert 4: ft09-0d8bbf25, sc25-635fd71a
expert 5: g50t-5849a774, sk48-d8078629
expert 6: lf52-271a04aa, sp80-589a99af
expert 7: lp85-305b61c3, su15-1944f8ab
```

Training itself hit a real disruption worth recording: the run crashed or
stalled partway through the 60-epoch ARC-finetune phase (exact cause not
captured -- discovered only via a stale checkpoint several hours later,
not a live crash trace) and had to be resumed from an epoch-35 backup
with a fresh optimizer for 25 more epochs to reach `checkpoint_tag=final`.
The coordinator ran the actual evaluation directly after finding training
had completed but the eval step had never been run -- see this repo's own
session history for the full story if it matters; not relevant to the
result itself, since both checkpoints below are genuinely fully trained.

## Evaluation methodology

`scripts/eval_partitioned_experts.py`, comparing this experiment's
checkpoint (`checkpoints_partitioned_experts/`) against the
already-existing `stage6-game-holdout` fold-1 baseline checkpoint
(`checkpoints_holdout_baseline/`, the same checkpoint reused directly by
several other stage6-* experiments this session rather than retrained
from scratch -- established project convention), both evaluated on the
identical verified 150-file corpus (all 25 games):

1. **changed-patches**, pooled over the 20 trained games and pooled over
   the 5 held-out games separately, plus a held-out per-game breakdown --
   the standard metric every stage6-* experiment reports.
2. **Gate entropy / dominant-expert frequency** on held-out games and a
   2000-transition trained-games sample (methodology ported from
   `scripts/diagnose_gate_entropy_holdout.py`) -- does the *deployed*
   soft-gated blend actually use more of the 8 experts on held-out
   games, not just cosmetically during training?
3. **InfoGain** on held-out vs trained games (methodology ported from
   `scripts/diagnose_infogain_holdout.py`) -- does raw per-expert
   disagreement (the Stage 5 hypothesis-bundle exploration signal) still
   survive under hard-partitioning?
4. **Split-half best-expert agreement** on trained-game (game, action)
   contexts (methodology ported from `scripts/
   diagnose_expert_specialization.py`, the coordinator's own mid-task
   diagnostic) -- does hard-partitioning spread real predictive
   competence across more of the 8 experts, or relocate the same
   "2 dominate, 6 dead weight" pattern onto different clusters?

## Results

`scripts/eval_partitioned_experts.py`, both checkpoints on the identical
verified 150-file corpus (raw output in `logs/partitioned_experts_eval.json`):

| metric | baseline (dense gate) | partitioned-experts (hard-routed) |
|---|---|---|
| trained-games changed-patches | **+49.89%** | **-0.08%** |
| held-out changed-patches (pooled) | -0.19% | -0.35% |
| held-out `r11l` | -1.4% | **-31.6%** |
| held-out `bp35` | -0.3% | +0.2% |
| held-out `m0r0` | +0.3% | -0.0% |
| held-out `tr87` | -0.3% | -0.7% |
| held-out `ka59` | -0.2% | -3.8% |
| gate entropy, held-out (% of max) | 99.82% | **7.60%** |
| gate entropy, trained (% of max) | 99.00% | **0.00%** |
| dominant-expert frac, held-out | 0.00% | **100.00%** |
| dominant-expert frac, trained | 1.85% | **100.00%** |
| InfoGain held-out / trained ratio | 0.520 | 0.157 |
| split-half agreement (trained) | 71.9% (chance 13.2%) | **95.5%** (chance 13.2%) |
| mean winner mode-share (trained) | 54.4% | 76.0% |
| max single-expert winner share (trained) | 37.1% (expert 6) | **18.0%** (expert 1) |
| experts that ever win at least once | 7/8 | **8/8** |

## Honest read

**The narrow mechanism question has a clean, unambiguous YES: hard
game-cluster partitioning does spread genuine predictive competence
across more of the 8 experts, and makes the gate far more decisive.**
Split-half best-expert agreement on trained games jumped from 71.9% to
95.5% (both far above the 13.2% chance floor, so this isn't newly-created
noise), the winner distribution flattened (max single-expert share
37.1% -> 18.0%, meaning no expert dominates the way baseline's experts 6/3/7
did), and for the first time in this whole investigation *every* expert
wins at least one (game, action) context outright, not just 5-7 of 8.
This is exactly the outcome coordinator's earlier `diagnose_expert_
specialization.py` diagnostic (on the full 25-game production checkpoint)
was asking whether hard-partitioning could produce, and on the trained
games it clearly did.

**But this "successful" specialization came bundled with a catastrophic,
unacceptable cost: trained-games prediction quality collapsed from
+49.89% to -0.08% -- essentially total loss of the baseline's real
predictive edge.** The mechanism is directly visible in the gate-entropy
numbers: the gate didn't just become *more decisive*, it became
**completely deterministic** (0.00% of max entropy, 100% dominant-expert
frequency, on *both* trained and held-out games). Bypassing the gate for
the main task loss (hard-routing to the assigned expert) plus the
routing-supervision auxiliary loss taught the gate to always commit
absolutely to one expert per input -- which is exactly what "genuine
specialization" should look like in principle, but it also destroyed the
soft-blend ensemble-averaging effect that Stage 4's own history already
established was doing real work at this data scale (`CLAUDE.md` Stage 4:
"a uniform blend of all experts is a more loss-effective solution than
genuine routing... averaging several noisy small experts reduces
variance"). Forcing hard commitment traded that away and did not get
enough single-expert accuracy back to compensate -- a real, measured
example of exactly the risk that finding warned about, not a new
discovery so much as living proof of it.

**On the actual central question -- does this close the held-out-games
gap -- the answer is no, and if anything it's a slightly worse
picture than baseline, not better.** Held-out pooled changed-patches went
from -0.19% (baseline, inside the established 5-fold noise band) to
-0.35% (partitioned) -- both are noise-band-consistent zeros, but
`r11l` specifically collapsed from -1.4% to -31.6%, a real, large,
game-specific regression. The gate's 100.00% dominant-expert-frequency on
held-out games shows *why*: forced to commit to exactly one expert with
no ambiguity, and with no game-id signal to route by on a truly unseen
game (same untrained fallback embedding index every prior intervention
has hit), the gate confidently routes to *some* expert regardless -- just
as confidently wrong as it is confidently right on trained games. Sharp,
decisive routing is not the same thing as *correct* routing when the
input is genuinely outside the training distribution; the routing
signal it learned was tied to game identity, and game identity is
exactly the one signal a held-out game structurally cannot supply.

**InfoGain also degraged under hard-partitioning** (held-out/trained
ratio 0.157 vs. baseline's 0.520, itself already lower than the 0.999
ratio the production 25-game checkpoint showed in `scripts/
diagnose_infogain_holdout.py` -- a reminder that InfoGain robustness is
checkpoint-specific, not a fixed property of the architecture). Forcing
each expert to narrowly specialize on 2-3 specific games' content may
make the experts *more* similar to each other, not less, when facing
genuinely novel input none of them were shaped for -- the opposite of
what Stage 5's exploration signal needs from a held-out game.

**Verdict: this specific hard-routing recipe is not a viable fix and
should not be pursued further as-is.** It's the 15th-ish independent
intervention against the held-out-games gap this project has tried, and
it joins the negative column -- but it's a more informative negative than
most, because it directly demonstrates *why* the "spread specialization
across experts" idea alone doesn't help: the held-out bottleneck isn't
"the gate can't commit," it's "the gate has no game-independent signal to
commit *correctly* to for a game it's never seen," and forcing commitment
without giving it that signal just makes the wrong answer more confident.
If a future session wants to revisit forced specialization, the fix would
need to come with a genuinely content-derived (not game-id-derived)
routing signal that could plausibly transfer -- but `stage6-context-
embedding`'s three continuous-conditioning attempts (frame-content,
episode-context, recurrent hidden state) already tried exactly that
substitution and all three failed identically. The more defensible
reading, consistent with this project's accumulated pattern: specialization
mechanics (partitioning, routing losses) are not the bottleneck here at
all; the bottleneck is that nothing in a novel game's own first few
observed frames currently gives the model *any* reliable signal about
which of its trained-game "modes" (however they're organized) actually
applies -- which is exactly the gap test-time adaptation (real gradient
updates from the new game's own data) is the one mechanism shown to
address at all, elsewhere in this investigation.

**User-proposed alternatives not tested here** (bootstrap experts on
different transition subsets, per-expert dropout/data resampling, an
explicit diversity loss on predicted deltas rather than gate load-balancing):
given this result, these would likely hit the same ceiling for the same
reason -- they'd plausibly still produce genuine specialization (this
experiment already shows that part is achievable) without solving the
"no transferable routing signal for a truly novel game" problem that
turned out to be the actual blocker. Not recommended as the next lever
to pull; test-time adaptation and further content-derived conditioning
(if a genuinely new mechanism is found, not a variant of the three
already tried) remain better-motivated next steps.

## Reproducing this experiment

```
python -m jepa.train_partitioned_moe --pretrain-epochs 20 --epochs 60 \
    --num-experts 8 --external-per-game 2000 \
    --exclude-games r11l,bp35,m0r0,tr87,ka59 \
    --checkpoint-every 5 --out checkpoints_partitioned_experts

python scripts/eval_partitioned_experts.py
```
(`JEPA_NUM_WORKERS=0` recommended on a shared/contended GPU box, per
CLAUDE.md's own gotcha -- used for this run since another background
agent session was training concurrently on the same GPU.)
