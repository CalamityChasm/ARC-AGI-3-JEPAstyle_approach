# ARC-AGI-3 Agent: Architecture (as actually built)

This describes what's actually implemented and running on `master` today,
not the original plan. The original planning spec (Mamba recurrent core,
16-24 experts, a combiner network, separate reward/termination heads,
test-time-training at level transitions, a symbolic representation bank)
is preserved in git history and in `notes.md`'s narrative account of how
and why each piece diverged from that plan during implementation --
mostly "smaller/simpler worked as well or better and was cheaper to get
right," per `plan.md`'s own guiding principle of not adding a component
until a measured bottleneck demands it. See `CLAUDE.md` for the current,
continuously-updated status and experiment log; this file is the static
"what is the shape of the system" reference.

---

## Design philosophy

Every ARC-3 game is treated as a belief-refinement problem: a small,
per-game-unknown set of discrete actions (≤8, one with a grid (x, y)
coordinate) whose effects have to be discovered through play, sparse
score/WIN/GAME_OVER feedback, and no natural-language instructions. The
agent is a small JEPA-style world model (encode → predict-next-latent)
combined with a Bayesian bundle of hypotheses about "what the actions do"
and an exact memory of states already seen, rather than an LLM or a
hand-coded heuristic engine. Belief-refinement, not language reasoning,
is the core mechanism.

---

## High-level pipeline

```
Raw frame (64x64 grid, 16 ARC colors + pad) ──► CNN encoder ──► feat (C, 8, 8)
                                                                      │
                        ┌─────────────────────────────────────────────┤
                        │                                             │
              exact-memory lookup                          MoE predictor
        TransitionGraph: has this exact                (8 experts, action/xy/
        frame been seen before, with a               game-conditioned gate)
        known winning action?                                       │
                        │                          ┌──────────────────┴───────────────┐
                 yes ──►│ take it, skip            │                                   │
                 the rest of this turn      predict_all_experts             gated forward
                 entirely                   (ungated, per-expert)         (real next-feat
                        │                            │                     prediction)
                        no                    InfoGain(a) = variance             │
                        │                     across experts, per candidate   ValueHead
                        │                     action -- also doubles as the      │
                        │                     ACTION6 click-location map         │
                        │                            │                           │
                        │                    HypothesisBundle: Bayesian     V(next_state)
                        │                    confidence over the 8 experts        │
                        │                    as "hypotheses," entropy → β         │
                        │                            │                           │
                        └──────────────► Q(s,a) = (1-β)·InfoGain(a) + β·V(a) ◄────┘
                                                       │
                                          ε-random fallback (25%),
                                          temperature-weighted click
                                          sampling for ACTION6
                                                       │
                                                  chosen action
                                                       │
                                                  Environment
                                                       │
                              observed outcome ──► TransitionGraph.record(),
                                                    HypothesisBundle.update()
```

---

## Component detail

### 1. Encoder (`jepa/models/encoder.py: CNNEncoder`)

Deliberately small: 4 stride-2/stride-1 convolutions (GroupNorm + GELU),
`(17, 64, 64)` one-hot grid → `(C, 8, 8)` feature map, `C = 64` by
default (a `width_mult` scaling knob exists and has been tested up to
2x-4x; wider consistently failed to improve held-out-game generalization
and once caused real capacity-enabled overfitting, so `C = 64` remains
the deployed default -- see `CLAUDE.md`'s Stage 6 addendum). An EMA
target copy (`make_ema_target`, momentum 0.996) exists only for computing
a fair, non-drifting training/eval signal, never used at inference.

No connected-component/object-tokenization subsystem was built; object
identity is left entirely to the encoder's learned representation. A
same-color contrastive loss was tried as an explicit push toward
object-identity representations (`--contrast-weight`) -- a real, large
local win that was later shown not to generalize past the games it was
trained on (see `CLAUDE.md`'s object-identity checkpoint history).

### 2. Dynamics predictor (`jepa/models/moe_predictor.py: MoEPredictor`)

`s_hat_{t+1} = feat + Σ_k g_k(feat, action, xy, game) · E_k(feat, action, xy, game)`

- `K = 8` experts by default (the original plan's 16-24 assumed a much
  larger, more diverse pretraining corpus than this project has had at
  any point; higher K without that data risks the "expert collapse"
  failure mode observed directly in early attempts -- see `CLAUDE.md`
  Stage 4).
- Each expert is a small per-patch pointwise-conv network. The gate is a
  dense softmax over all K experts by default; a noisy top-k gating
  variant exists (`top_k` param) but was a documented regression, not
  adopted.
- Conditioning: action id (embedding), normalized `(x, y)` (only
  meaningful for the click action), and a per-game id (embedding,
  `game_vocab.get(game_id, 0)` fallback for any game outside the trained
  vocabulary -- this fallback, and whether to condition on game identity
  at all, was extensively re-investigated in Stage 6; ablating it
  entirely does not fix the held-out-game generalization gap and is not
  the deployed configuration).
- The combiner network the original plan called for (blending pairs of
  experts for compound effects) was explicitly deferred and never built
  -- the dense gate's own soft blend already covers this informally.
- `predict_all_experts` exposes the *ungated* per-expert predictions
  separately -- this is what `InfoGain` and the Bayesian hypothesis
  bundle below consume; it is a different, and empirically more
  robust-to-novelty, signal than the gated forward pass (see Stage 6:
  gated prediction collapses to near-identity on unseen games, but raw
  per-expert disagreement does not).

### 3. Value head (`jepa/models/value_head.py: ValueHead`)

One small decoupled MLP off a pooled summary of `feat`, predicting
expected discounted future progress (`GAMMA = 0.95` Monte Carlo return
target, see `jepa/data/value_targets.py`). The original plan's separate
reward and termination heads were never built -- one combined value
signal proved sufficient for what the action-selection formula needs.
Trained on a corpus that is ~98% zero-target (`levels_completed` deltas
are rare under any policy), with oversampling (`NONZERO_WEIGHT = 25`) to
keep the rare positive examples from being drowned out.

### 4. Hypothesis bundle (`jepa/hypothesis_bundle.py: HypothesisBundle`)

The "N parallel hypotheses" from the original plan are, concretely, the
predictor's own 8 MoE experts -- each treated as one hypothesis about
"what a given action does," reusing the already-trained experts rather
than a separate hypothesis-search structure.

- Bayesian confidence: `p(H_i) *= exp(-error_i / τ)` per observed
  transition, `τ = 0.01`, renormalized (softmax over log-weights).
- Geometric forgetting (`decay = 0.8`) prevents runaway certainty from
  accumulating over a long episode -- an earlier undecayed version left
  the agent "confident" (and therefore trusting the value head) for
  most of every episode regardless of whether that confidence was still
  earned.
- `entropy() / max_entropy()` → `β`: high entropy (experts still
  disagree about recent accuracy) → low `β` → trust `InfoGain`; low
  entropy (one expert has clearly been right) → high `β` → trust `V`.
- `InfoGain(a)` = variance across the 8 experts' raw predicted
  next-features for candidate action `a`. For the click action, a
  top-k-patch mean (`TOP_K_PATCHES = 8` of 64) is used instead of a flat
  mean or flat max over all patches -- both flat reductions were found to
  systematically mis-rank the click action relative to simple actions.

### 5. Exact memory (`jepa/memory.py: TransitionGraph`)

A plain hash-keyed dict (`blake2b` digest of exact frame content) →
`(action, xy) → next_state`, built up during play and persisted for an
agent's whole lifetime on one game (every RESET, not just one level
attempt -- ARC-3 resets return to the same starting frame, so prior
discoveries are exactly replayable). If the current exact frame has a
previously-recorded winning action, the agent takes it immediately, no
re-exploration. Also tracks which `(action, xy)` pairs have already been
tried from the current exact state, to guarantee local coverage
independent of the Bayesian/InfoGain ranking.

### 6. Action selection (`ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py: Hypothesis`)

`Q(s, a) = (1 - β)·InfoGain(a) + β·V(next_state(a))`, greedy argmax, with:

- Exact-recall short-circuit (above) checked first, every turn.
- `EPSILON = 0.25` uniform-random fallback before the greedy argmax --
  without it the agent locks onto a single action or click location for
  a whole episode once it stops looking surprising, even if it was never
  productive.
- Click-location sampling: temperature-weighted softmax over the 64
  per-patch InfoGain values (`PATCH_SAMPLE_TEMPERATURE = 0.1`), then a
  uniform-random pixel within the chosen patch -- a hard argmax here was
  found to default to the same low-index patch whenever the map is flat,
  which is common, not rare.
- Opening-probe plan: try every simple action once at episode start
  before trusting the bundle's own confidence weights.
- `MAX_ACTIONS = 300` per game (the real Kaggle default; a 900-action
  variant was locally validated as broadly helpful across checkpoints
  but has not been adopted as the default pending further confirmation).
- A top-level `try/except` around `choose_action`/`is_done`
  ("heartbeat") falls back to a genuinely random legal action on any
  unexpected exception (e.g. a hidden game with an unexpected frame
  shape), rather than crashing the whole scored run.

### 7. Training pipeline

Two-phase, per checkpoint:

1. **Synthetic pretrain**: encoder + MoE predictor trained on a growing
   roster of non-ARC data sources for mechanic diversity -- MiniGrid
   (navigation-themed, 21 environments, one shared `game_id`), and an
   evolving set of further sources tried across Stage 6 (MinAtar,
   Procgen, OpenSpiel board/strategy games, hand-rolled Snake/Pong) at
   various scales up to ~2.14M synthetic transitions. Each genuinely
   distinct game/environment gets its own `game_id` -- pooling
   mechanically-dissimilar sources under one shared id was measured to
   actively hurt. Pretrain epoch count is sized to hold total
   samples-seen roughly constant relative to the corpus size, learned
   the hard way after a curriculum-imbalance bug caused an unrecovered
   representation collapse on one over-sized, under-epoched attempt.
2. **ARC-3 finetune**: the same encoder/predictor continue training on
   local ARC-3 recordings (`~12k` local + up to `2000/game` from a larger
   external random-policy corpus), patch-level change-weighted loss
   (8x upweight on patches that actually changed -- a plain mean-MSE
   loss is dominated by the mostly-static majority otherwise) plus a
   Switch-Transformer-style load-balance auxiliary loss on the gate.

The value head is trained separately, after the predictor, against
whichever encoder is being deployed (encoder/value-head latent-space
mismatch was a real, once-diagnosed bug -- retrain the value head
whenever the encoder changes).

---

## What was planned but is not in production

- **No recurrent (Mamba or otherwise) core carrying history across the
  episode.** A GRU-based recurrent predictor was built (`jepa/models/
  recurrent_predictor.py`) as the Mamba substitute the original plan
  called for (a local CUDA-toolkit/torch-build mismatch made real Mamba
  impractical on this hardware), and its own Stage 3 milestone was met
  -- but the deployed `Hypothesis` agent uses the plain MoE predictor,
  not the recurrent one. When tested for held-out-game generalization in
  Stage 6, the recurrent hidden state did not generalize better than the
  stateless predictor, so there was no reason to add the complexity back
  in for the currently deployed agent.
- **No combiner network** for blending pairs of experts (deferred at
  Stage 4, never revisited).
- **No test-time-training at level transitions** consolidating confirmed
  hypotheses into base weights mid-episode, as the original plan
  described. A related but distinct idea -- real gradient-step adaptation
  using a hidden game's own observed transitions during play
  (`TestTimeAdapter`, an ANIL-style ~33.8K-param restricted parameter
  subset) -- was built and validated as the one mechanism in Stage 6's
  entire generalization investigation that shows genuine, dialable
  positive signal. It is not yet merged to `master` (see below).
- **No symbolic representation bank** (persistent object IDs, semantic
  role inference) -- left entirely to the encoder's learned features.
- **No LLM or language-conditioning component anywhere** -- deliberate,
  for full offline/no-internet eval compatibility.

## Active R&D, not yet in production (see `CLAUDE.md`'s Stage 6 addendum)

A large, still-unmerged body of Stage 6 work exists on `stage6-*`
branches, investigating why the deployed model has no measurable
zero-shot prediction advantage over identity on any game it wasn't
trained on -- the central open problem for real Kaggle performance, since
the ~110 hidden competition games are almost entirely of this kind.
Fifteen-plus independent interventions (conditioning changes,
architecture changes, five separate data-diversity attempts up to
dozens of new games and millions of synthetic transitions, data
augmentation, capacity scaling) have failed to close this gap and are
treated as having established it as a real data/hardware ceiling, not a
specific unfound bug. Two mechanisms show real, if still modest,
promise: `TestTimeAdapter` (real gradient-step adaptation during play)
and a Reptile-style meta-learning training objective explicitly
optimizing for post-adaptation performance (`jepa/train_meta_predictor.py`,
branch `stage6-meta-learning`) -- neither has yet cleared a real
agent-level backtest at a trustworthy sample size. `CLAUDE.md` is the
authoritative, continuously-updated source for exactly what's been tried
and what the current best next step is.
