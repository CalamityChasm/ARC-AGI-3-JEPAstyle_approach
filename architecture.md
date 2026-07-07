# ARC-AGI-3 Agent: Architecture Spec
 
## Design Philosophy
 
The agent treats every game as a **belief-refinement problem under a 9-hour training budget and a fixed action interface**: a small set of discrete actions (≤7, some with grid coordinates) whose effects are unknown at the start of each game, sparse score/WIN/GAME_OVER feedback, and levels that compound earlier mechanics. The core bet is that a JEPA-style latent world model, combined with a small population of structured hypotheses about *what the buttons and on-screen elements do*, can out-perform pure exploration baselines by being directed rather than random — without requiring an LLM or internet access at eval time.
 
---
 
## High-Level Pipeline
 
```
Raw frame (grid) ──► CNN/patch encoder ──► latent state s_t
                                              │
                          ┌───────────────────┴────────────────────┐
                          │         MAMBA RECURRENT CORE            │
                          │   (carries episode + level history)     │
                          └───────────────────┬────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
            MoE PREDICTOR              SALIENCE MAP              DECOUPLED HEADS
       (gated experts + combiner)   (per-region pred. error)   (value / reward / done)
                    │                         │                         │
                    └───────────► HYPOTHESIS BUNDLE ◄───────────────────┘
                          (Bayesian confidence, entropy → β_t)
                                       │
                               ACTION EXPERT
                  Q(s,a) = (1-β)·InfoGain(a) + β·V(ŝ_{t+1}(a))
                                       │
                                  Environment
                                       │
                          feedback ──► all components above
```
 
---
 
## Component Detail
 
### 1. Perception / Encoder
- Small CNN (or patch-based ViT) — frames are small, low-color grids, so this is lightweight.
- Object-level tokenization via connected-component analysis kept as a thin auxiliary signal (position/color/shape/size per object), feeding the salience and attribution mechanisms — **not** a full parallel symbolic subsystem.
- EMA target encoder (standard JEPA) used only for computing prediction error after the fact.
### 2. JEPA World Model — Predictor
- **Mamba recurrent core**: carries compressed history across the episode (and across level transitions, modulo TTT consolidation — see below). This replaces frame-stacking-style memory.
- **Mixture-of-gated-experts**: `ŝ_{t+1} = Σ_k g_k(s_t, a_t) · E_k(s_t, a_t)`
  - Each expert `E_k` is a small MLP representing one atomic causal pattern (translate, rotate/reflect, recolor, increment/decrement-a-tracked-value, trigger-on-spatial-overlap, etc.).
  - Each gate `g_k` is a small MLP learning **what context activates** expert k — action identity, counter state, spatial overlap, or any combination — discovered during pretraining rather than hand-categorized.
  - **K = 16–24 experts** as the planning default, with a load-balancing loss that penalizes total collapse without forcing perfectly even utilization. K is treated as a cheap empirical sweep (12/24/48) early in the pipeline, not a fixed decision.
- **Combiner network**: small MLP that blends pairs of expert/gate outputs to represent compound, per-game mechanics (e.g., "moves right AND recolors") as learned interpolations rather than requiring a dedicated expert per compound effect.
### 3. Hypothesis Bundle (Belief State)
- N parallel hypotheses = candidate (action/meter/tile → expert+gate) assignments, each predicting a full trajectory, not just one step.
- Bayesian confidence update: `p(Hᵢ) ∝ p(Hᵢ) · exp(-prediction_error_i / τ)`.
- Entropy monitor over the hypothesis distribution → `β_t` (learned, not scheduled), driving the explore/exploit blend.
- **Core vs. level-local split**:
  - *Core mechanics* (confidently confirmed) consolidate into the base model via lightweight test-time training (TTT) at level transitions, and stop being actively tested.
  - *Level-local* hypotheses reset at each level transition; only the **diff** from the previous level (new objects, changed tiles, new salience) is re-probed.
### 4. Salience Map / Areas of Interest
- Per-region (per-token) prediction error between predictor output and target encoder output.
- Serves two roles: (a) **attribution** — gradient sensitivity of expert gates to action/counter/position inputs, used to localize *why* a region surprised the model; (b) **prioritization** — persistently-surprising regions seed new hypotheses and become experiment-designer targets.
### 5. Action Expert
- **Phase blending**: `Q(s_t,a) = (1-β_t)·InfoGain(a) + β_t·V(ŝ_{t+1}(a))`
  - `InfoGain(a)` = disagreement across hypothesis-conditioned predictions (KL divergence between predicted next-latents).
  - `V(·)` = learned critic (decoupled small head).
- **Experiment designer**: opening probes are "press each action once," "Wait and observe" (for meters), "step on each visually-distinct tile type" (for spatial triggers, seeded by salience). Later probes follow attribution: ablate one factor at a time (repeat action / vary position / vary timing) when attribution is ambiguous.
- **Cost-weighted hypotheses (EFE-style)**: hypotheses whose effect-code is "global reset/fail" (life-counter-type) are down-weighted for *active* testing — confirmed opportunistically from data collected anyway, not deliberately triggered.
- **Wait/no-op** is a first-class action, especially valuable for detecting time-based (meter) dynamics.
### 6. Decoupled Heads
- Small separate MLPs off the shared latent for: value (critic), reward (score delta), and termination/death prediction. Kept separate from the main predictor by design — same rationale as DIAMOND's decoupled reward/termination networks, just applied to a latent rather than pixels.
### 7. Training Regimen
- **Phase 1**: self-play pretraining of encoder + MoE predictor + gates + combiner across the ~25 public ARC-AGI-3 games (no synthetic puzzle generator — relies on real game diversity).
- **Phase 2**: belief state calibration (confidence should match empirical accuracy) — frozen world model.
- **Phase 3**: experiment-designer + action-ranker tuning, using the public human trajectory dataset for "what probes were informative" signal.
- **Phase 4**: joint fine-tune with `R_total = R_epistemic + R_instrumental + R_calibration`.
- **At eval time**: TTT between levels consolidates confirmed core hypotheses into base weights; per-game hypothesis bundles re-initialize at game start.
- All pretrained components (and any external open-weight checkpoints) are attached as Kaggle datasets — allowed under "no internet, but freely available pretrained models OK."
---
 
## Known Weak Points and Mitigations
 
| Weakness | Why it happens | Mitigation already designed in |
|---|---|---|
| **Cold start** | First few actions of a new game have a near-uniform hypothesis bundle and high InfoGain everywhere — close to undirected exploration regardless of sophistication. | Experiment designer's fixed opening sequence (probe every action, then Wait) shrinks this window as fast as possible; unavoidable in principle. |
| **Genuinely novel mechanics** | A mechanic with no good fit in the expert/gate/combiner space produces uniformly high error with no clean attribution. | Falls back to RND-style novelty-seeking exploration — graceful degradation to "as good as the simple baselines," not a hard failure. |
| **Expert collapse / data dilution** | Too many experts (e.g., 64) spreads ~25 games' worth of pretraining data too thin; a few popular experts dominate, rest are undertrained noise. | K = 16–24 default, load-balancing loss tolerant of unevenness but penalizing total collapse, with an early empirical sweep across K values. |
| **Compounding rollout error** | Any multi-step planner compounds per-step prediction error; long-horizon plans become unreliable. | Effectively caps useful planning horizon — favors shorter lookahead / more frequent re-planning over long open-loop plans. No silver bullet; treat as a known ceiling. |
| **Compound/overlapping triggers** | If one action fires multiple gates at once (movement + counter decrement + tile effect simultaneously), salience lights up everywhere and attribution becomes ambiguous. | Experiment designer's "ablate one factor at a time" probes (repeat vs. vary action/position/timing) are the main lever; full disambiguation isn't guaranteed for highly entangled mechanics. |
| **Risk calibration needs prior data** | Cost-weighting life-counter hypotheses requires *some* basis for "this looks bad," which doesn't exist before any death has occurred. | First death is treated as informative-by-default (opportunistic confirmation) rather than something to avoid pre-emptively; acceptable one-time cost. |
| **9-hour training budget** | Tighter than originally assumed (was scoped around 12h). | Architecture kept to small CNN + Mamba + small MLP experts/gates/heads — no component individually expensive; K-sweep and phase structure designed to be interruptible/prioritizable if time runs short. |
| **~50% test data on current leaderboard** | Current standings have real variance; final ranking may shift. | No architectural mitigation — just a reminder not to over-fit design decisions to current leaderboard gaps. |
 
---
 
## Footnote: Discarded-for-Complexity Ideas (kept for future reference)
 
These were evaluated and set aside **specifically because of implementation complexity relative to benefit**, not because they're bad ideas. Worth revisiting if early results show a specific bottleneck matching one of these.
 
- **V-JEPA 2 / V-JEPA 2-AC pretrained encoder**: strong action-conditioned world-model recipe (frozen encoder + small AC predictor, ~300M params), but encoder pretrained on natural internet video — large domain gap to flat-colored sprite grids. Revisit if from-scratch encoder pretraining underperforms badly and a natural-video prior turns out to transfer better than expected.
- **Atari-pretrained world models (DreamerV3 / IRIS / STORM / DIAMOND) as the base**, rather than training JEPA from scratch: better visual/action-structure match than V-JEPA 2, smaller (DIAMOND's dynamics model is ~4.4M params), and "no internet, pretrained models OK" makes this legal. Revisit if from-scratch pretraining on ~25 public games proves too data-starved.
- **DIAMOND's EDM-based diffusion predictor**: the *lesson* (action-determined components should be unambiguous; few-step iterative generation can be stable with the right formulation) is already folded into the gated-expert design. The diffusion machinery itself — full pixel-space generation — was discarded as unnecessary overhead given JEPA operates in latent space already.
- **OpenVLA / Pi0-style VLA**: language-conditioning machinery is dead weight (ARC-AGI-3 gives no instructions), and 3-7B backbones eat the compute budget. The one transferable idea — a small specialized "action expert" network — is already present in your action expert/ranker.
- **SIMA 2's two-tier cognitive/control split**: conceptually validated the "lightweight planner over a learned world model" shape, but SIMA 2 itself (Gemini-based, closed weights) isn't usable. Revisit if a small local open-weight model (a few hundred M params) becomes worth adding as an explicit high-level planner above the world model.
- **Full LLM-driven executable world models** (GPT-5.5-style refactoring loop, 15/25 public games solved): strongest published result in the ARC family, but requires a frontier LLM at inference — incompatible with no-internet eval. The general "refinement loop: explore → verify → iterate" shape is already present via the hypothesis bundle's falsification/recombination cycle.
- **Synthetic puzzle generators for pretraining/adversarial hardening (original Phases 1 & 5)**: building a generator capable of producing arbitrarily novel-but-coherent interactive rule systems is itself close to "solve the underlying problem." Replaced with self-play on real public games. Revisit only if self-play pretraining data proves insufficient *and* a scoped-down generator (e.g., simple parameterized toy grids with known action semantics) seems tractable as a side project.
- **Full symbolic Representation Bank** (persistent object IDs, semantic role inference like "red has moved twice → red is agent color", explicit static/mobile classification): risks duplicating what the JEPA encoder should learn implicitly, and is a second major subsystem competing for engineering time. Kept only the lightweight per-object position/color/shape/size tokens needed for salience/attribution. Revisit if post-hoc inspection shows the latent space *isn't* capturing object identity/roles well.
- **Free-form mid-episode hypothesis generation** (vs. codebook search + combination): generating genuinely novel structured hypotheses on the fly is close to circular (it's the capability ARC-AGI-3 tests for). Replaced with search-and-combine over a pretrained finite codebook + combiner. This is a fundamental ceiling, not just a complexity tradeoff — flagged here mainly as a reminder of *why* the codebook approach was chosen, not as something to "add back" later.