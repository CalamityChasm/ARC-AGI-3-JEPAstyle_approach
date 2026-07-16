# Development Notes

A chronological, detailed record of how this project was built: every
design decision, every experiment and its result, every bug found and how
it was fixed. `CLAUDE.md` is the living "current state" doc; this file is
the fuller narrative behind it, kept as a single reference for anyone
picking up the project later (including future-me) who wants the reasoning
and not just the conclusions.

Companion docs: `architecture.md` (system design spec), `plan.md` (staged
build plan and its guiding principles), `rules.md` (competition rules
reference).

---

## 0. Project framing

This is an entry for the ARC-AGI-3 Kaggle competition
(`arc-prize-2026-arc-agi-3`): a world-model-based agent that plays novel,
procedurally-varied grid games by building an internal latent
representation of "what this game is doing" and refining beliefs about it
turn by turn, rather than searching over hand-coded heuristics or
prompting a language model to reason about the board directly. The
approach is JEPA-style (Joint Embedding Predictive Architecture): an
encoder maps a game frame to a compact latent, a predictor learns to
forecast the *next* latent given the current latent and the action taken,
and everything downstream (exploration, memory, planning) operates on
that learned latent space instead of raw pixels.

The guiding principles carried through the whole build (from `plan.md`):
work in stages, each with its own falsifiable milestone; don't add a
component until a measured bottleneck actually demands it; prefer a
data-side fix over an architecture-side one when the two are hard to tell
apart; keep an honest, hard-to-game evaluation metric front and center
rather than one that's flattered by trivial solutions (e.g. "predict no
change").

---

## 1. Environment and data setup

The project runs on a local Windows machine with an RTX 2070. Setup
involved:

- A pinned `requirements.txt` — the langchain/langgraph/smolagents/openai
  dependency stack needs pinned versions to resolve at all; without pins,
  pip's resolver gives up with `resolution-too-deep`. The pins were taken
  from the vendored `ARC-AGI-3-Agents/uv.lock`'s already-resolved graph.
  One follow-on pin conflict: the `arc-agi` PyPI package wants
  `pillow>=12.1.1`, while the vendored framework's own lock pins an older
  pillow for unrelated reasons — settled on `pillow==12.3.0`, which
  satisfies both.
- PyTorch installed as a CUDA build. The `cu121` wheel index doesn't carry
  a build for the pinned torch version at all (only older torch versions
  exist there) — `cu126` does, and that's what's installed. Verified with
  a `torch.cuda.is_available()` check.
- `ARC-AGI-3-Agents/` — a vendored copy of the competition's harness
  framework plus the 25 public practice games — runs fully offline
  (`OPERATION_MODE=offline` in its `.env`, an arbitrary/anonymous API key
  is sufficient since no network calls are made in this mode).
- `environment_files/` (the actual per-game metadata/logic files the
  harness needs to find any game at all) and `arc_agi_3_wheels/` are not
  part of the framework's own git history — they ship only inside the
  Kaggle competition dataset zip and were pulled from there directly.
- Large data assets (ARC-AGI-1/2 puzzle corpora for encoder pretraining,
  local gameplay recordings, trained checkpoints) are all regenerable and
  kept out of version control by design — checked in would bloat the repo
  for no benefit since every one of them can be reproduced with a
  documented command.
- One recurring operational note: running the harness's `main.py` in
  offline mode reliably exits with a non-zero status even when every game
  completes correctly and every recording file is written — an offline
  scorecard-reporting quirk in the framework, not a real failure. Verify
  results by inspecting the recording files directly rather than trusting
  the process exit code.

---

## 2. Stage 0 — harness and data collection

Goal: get the competition harness running locally, fully offline, with a
way to log gameplay trajectories for later training.

Built:
- Confirmed the vendored framework runs all 25 public games offline.
- A `PressOnce` agent (`ARC-AGI-3-Agents/agents/templates/press_once_agent.py`):
  a scripted opener that presses every available action exactly once
  before doing anything else — a cheap way to surface a game's basic
  action space before any learned policy runs.
- Reused the framework's built-in `Random` agent and its `Recorder`
  utility, which logs every `(frame, action, next_frame)` step to a
  `*.recording.jsonl` file per episode.
- `scripts/run_stage0.py` — a thin wrapper to run any registered agent
  across all local games from the repo root and collect its recordings in
  one place.

This stage is complete and stable; the corpus it produces (run
`--agent random` several times over to get more passes; ~150 recording
files, ~10-12k transitions was the typical corpus size used downstream)
is what every later stage's local-data training draws on.

---

## 3. Stage 1 — JEPA core (encoder + one-step dynamics predictor)

### 3.1 Design

- `jepa/grid.py` — a shared grid representation used everywhere
  downstream: a 64x64 canvas, one-hot over 17 channels (16 ARC colors + 1
  padding channel), so puzzles/frames smaller than 64x64 get placed
  top-left and padded.
- `jepa/models/encoder.py` — a small CNN encoder mapping a (17, 64, 64)
  frame down to a (64, 8, 8) feature map, plus an EMA-updated ("target")
  copy of the same encoder (momentum 0.996), following standard JEPA
  practice of using a slowly-updated target network to produce prediction
  targets and avoid representational collapse.
- `jepa/models/predictor.py` — a one-step latent predictor conditioned on
  the action taken (and, for ACTION6, the click coordinate), predicting a
  *residual* on top of the current feature map rather than the next
  feature map directly (`next_feat = feat + predictor(feat, action)`).
- `jepa/data/arc_static.py` — loads the full ARC-AGI-1 and ARC-AGI-2
  puzzle corpora (16,668 grids combined) for masked-patch pretraining of
  the encoder, independent of any specific game's dynamics.
- `jepa/data/trajectories.py` — parses recorded gameplay JSONL into
  `(frame_t, action, frame_t+1)` transitions for the predictor, along with
  a per-transition changed/unchanged flag and per-8x8-patch change masks,
  plus a game-id vocabulary for optional per-game conditioning.
- `jepa/eval_stage1.py` — the actual milestone check: does the trained
  predictor beat a trivial "nothing changes, next frame equals this frame"
  identity baseline on held-out transitions, using the same encoder for
  both sides of the comparison? This "changed-patches" metric (isolating
  the comparison to patches that actually changed pixel-wise, rather than
  a whole-grid average that's dominated by static background) became the
  standing bar for this stage, since a naive whole-grid MSE is trivially
  "beaten" by mostly predicting no change at all.

### 3.2 Iteration history

**Iteration 1 — comparison methodology bug.** The first training run
looked like a clear pass: the predictor beat the identity baseline every
epoch. That comparison turned out to be unfair — it compared the
predictor's output against a *lagging EMA target* encoding rather than a
same-encoder identity baseline, which artificially inflated the identity
baseline's error (the EMA target's weights had drifted from the online
encoder's, so "identity" under the EMA encoder wasn't really identity
under the online encoder). Fixing the comparison so both sides use the
same encoder weights revealed the predictor was actually **24% worse**
than identity. Lesson: any "beats a baseline" claim needs both sides
computed under matched conditions, or the comparison isn't measuring what
it claims to.

**Iteration 2 — class imbalance in transitions.** Most random-policy
transitions in this data are exact no-ops: roughly 37% of frames are
completely unchanged frame-to-frame, and even "changed" frames are
usually ~90% static 8x8 patches with only a small region actually
different. A plain mean-MSE loss is dominated by the trivially-unchanged
majority and gives the model almost no gradient signal about real
dynamics. Oversampled changed transitions during training (3x sample
weight via a `WeightedRandomSampler`): gap narrowed to **-15%**.

**Iteration 3 — loss-level change weighting.** On top of iteration 2,
added patch-level weighting directly into the loss function itself (an
8x upweight on any 8x8 patch whose pixels actually differ between frames):
gap narrowed further to **-8.7%**.

**Iteration 4 — per-game conditioning (negative result).** Hypothesized
that the same action id meaning a physically different effect in each of
the 25 different games was a major confound for a single shared
predictor with no notion of "which game is this." Added a learned
per-game embedding, concatenated into the predictor's conditioning
vector. Result: no improvement at all (-8.7% before and after). This
ruled out cross-game action-semantics confusion as the dominant issue —
useful to know so this fix isn't re-attempted later under the impression
it hasn't been tried.

**Iteration 5 — reproducibility check.** Regenerated the local recording
corpus from scratch (a fresh random-policy pass, different random seed,
slightly different transition count) and reran 30 epochs of training:
reproduced essentially the same qualitative result (-0.3% changed-patches,
still failing to beat identity) on an independently-drawn corpus,
confirming iterations 1-4 weren't an artifact of one specific data draw.
A per-game breakdown also showed the pooled percentage can be a
misleadingly noisy summary on its own: a few games have such a tiny
identity-baseline error to begin with that a small absolute error swing
reads as a triple-digit percentage change — worth eyeballing absolute MSE
alongside the percentage for those cases rather than trusting the
percentage in isolation.

**GPU support.** Added a shared `get_device()` helper and wired real
`.to(device)` calls into training and eval — none of the code had
actually used the GPU up to this point despite one being available.
Benchmarked model compute alone at roughly 16x faster on GPU than CPU on
this hardware.

**Iteration 6 — external data, first pass.** With GPU support in place,
tried supplementing the local corpus with a much larger external
random-policy recording dataset (~105k transitions across the same 25
games, with a considerably higher changed-frame rate, ~64% vs. the local
corpus's much lower rate) via a streaming loader that reads directly out
of the dataset's zip archive without extracting it to disk (per-game
reservoir sampling so the much larger external corpus supplements rather
than drowns out the local one). The first such run ran far slower
wall-clock than the GPU speed-up would suggest — traced to the data
loader doing CPU-side one-hot tensor conversion per sample in a single
process with `num_workers=0`, making data loading (not GPU compute) the
actual bottleneck once the combined corpus grew past ~55k transitions.
Fixed with `num_workers=4, persistent_workers=True, pin_memory=True` on
both loaders (only enabled when running on CUDA).

**Iteration 7 — external data, with the fix.** Retrained on the combined
corpus (~54.8k transitions) for 60 epochs with the data-loading fix in
place. Result: did *not* clear the milestone, and the gap was nominally
worse (-1.4%) than the local-only baseline, despite the external corpus's
much higher changed-rate. More data and more epochs did not close the gap
this time — a genuinely surprising result that prompted deeper
investigation rather than just tuning further.

First checked the obvious confound: representational collapse (encoder
features drifting toward a constant, trivially easy-to-predict output).
Both predictor and identity losses were shrinking together across the 60
epochs, which is consistent with collapse — but directly measuring
encoder feature standard deviation on a held-out batch showed ~1.1-1.4
per channel, comfortably above the variance floor used in the loss
function, ruling out classic collapse.

Next hypothesis: the encoder's 8x8 feature map might simply not move much
in feature space even when the corresponding raw pixels change locally —
i.e., the "identity is hard to beat" result might be a representation
problem, not a predictor problem.

**Iteration 8 — testing the representation hypothesis (refuted).**
Directly measured per-patch feature-space delta between `frame_t` and
`frame_t+1` at changed vs. unchanged patches. Changed patches showed
**12x larger** feature deltas than unchanged ones (mean 2.8e-4 vs 2.3e-5)
— the encoder clearly does register local pixel changes. So the
bottleneck wasn't the encoder discarding signal.

Followed up by directly inspecting what the *predictor's own residual
output* looked like (before the `feat +` skip-connection is added) versus
the true target delta on the same held-out data: mean squared residual
was 2.0e-6 versus a true average squared delta of 3.3e-5 — the trained
predictor's residual was roughly **16x smaller** than the actual average
change. In other words, the predictor had learned to output something
close to zero and simply coast on the skip connection — an approximation
of "predict no change" rather than learning real dynamics, even though
the encoder's own features clearly showed the two frames differing at
changed patches. This pinpointed the actual bottleneck: not data volume,
not the encoder, but the predictor's unwillingness (or the training
setup's inability to push it) to commit to a non-trivial residual.

**Iteration 9 — isolating the cause (two negative results).** Ran two
targeted, independent interventions to find out *why* the predictor
wouldn't commit to real residuals:

- Single-game ablation: retrained on one game alone (the highest-activity
  game in the corpus, 100% frame-level changed rate, 2480 transitions),
  removing the 25-games-at-once setting entirely (a stronger test than
  iteration 4's per-game conditioning, which only added a game signal
  without removing the multi-game training mix). If cross-game
  interference were the true cause, an isolated, abundant, always-changing
  single game should be the easiest possible case to succeed on. Result:
  the predictor was worse than identity at every single one of 60 epochs
  — never once caught up.
- Zero-initialized residual branch: changed the predictor's last layer to
  start at exactly zero, so the model begins training as an exact
  identity function and has to earn its way to a non-zero residual
  through gradient signal, rather than starting from random noise it
  first has to learn to suppress. Re-ran the same single-game ablation:
  no meaningful change (still ~0.3% worse than identity, same overall
  curve). Kept this initialization as a reasonable default for residual
  predictors going forward, but it wasn't the fix.

At this point, three independent, well-targeted interventions — 9x more
data with a much higher changed-rate, complete removal of the multi-game
setting, and zero-initialization to remove a bad-starting-point
confound — had all converged on essentially the same "a few percent
worse than identity" result. The working conclusion at the time was that
this looked like a genuine architectural ceiling for a memoryless
single-frame predictor: given only the current frame and an action id, a
small shallow network may have no way to disambiguate, e.g., exactly
where a moving element currently sits with sub-patch precision, so
"predict no change" is close to the true MSE optimum available to it —
not a fixable training failure, but a case for either a model with real
history (a recurrent predictor, which was already planned for the next
stage) or training data from a policy that behaves purposefully rather
than randomly.

**Iteration 10 — the actual root cause, found afterward.** While building
the next stage's memory component, a specific diagnostic log line (meant
to fire when a known winning action was recalled from an exact previously
seen state) never fired despite the relevant state clearly recurring
across multiple resets in a test run. Chasing down *why* it never fired
led to inspecting exactly how actions were being recorded to the local
recording files — and surfaced the real bug: the harness's raw-frame
conversion routine (`_convert_raw_frame_data` in the vendored framework's
`agent.py`) never copied the actual action taken (`raw.action_input`)
into the `FrameData` object it constructs before recording it. Every
recorded frame therefore carried an action id of 0 — which happens to be
the default value, and also happens to be the RESET action — regardless
of what action was actually performed. The underlying object returned by
the environment step itself (`FrameDataRaw`) had the correct action all
along; it was simply dropped during this one conversion step.

This meant every piece of local-recording-derived training signal used in
iterations 1-9 above was trained and evaluated against a constant, wrong
action label. A predictor trained on that data structurally cannot learn
real action-conditioned dynamics from the local corpus, because the
action signal it's conditioned on carries no information at all — it's
constant noise. (The external dataset was unaffected, since it has its
own independently correct action field from a different recording path —
which is also *why* iteration 7's addition of external data didn't fix
things: it was diluted by, not replacing, the broken local signal, and
the eval set itself was drawn from the broken local corpus.)

The fix was a single line: add `action_input=raw.action_input` to the
`FrameData(...)` constructor call. After fixing it, regenerating the
local recording corpus, and rerunning the exact same combined-data,
60-epoch training setup as iteration 7: **changed-patches improvement
flipped from -1.4% to +29.2% — a clean pass.** A per-game breakdown
showed genuinely uneven, non-uniform learning (some games improved 30-40%,
a few remained negative) rather than a flat global shift, which is itself
a good sanity check that real per-game dynamics were being learned rather
than some trivial global artifact.

### 3.3 Resolution and the lesson

The milestone is passed. What had looked like a fundamental
single-step-predictor ceiling — three independent fixes all converging on
the same negative result — turned out to be fully explained by a single
upstream data-recording bug that made the local corpus's action-condition
signal constant and uninformative. With that fixed, the same architecture
that iterations 7-9 had concluded couldn't learn real action-conditioned
dynamics did exactly that.

The general lesson worth carrying forward: when several genuinely
different fixes all converge on the exact same negative result, that
pattern of consistency is *more* consistent with a shared upstream data
bug than with "every fixable hypothesis has been exhausted." It's worth
auditing the data pipeline itself — e.g., checking whether a supposedly
varied label is suspiciously constant — before concluding an architecture
or approach has hit a real ceiling. None of the earlier debugging work
was wasted, though: the loss-shaping fixes from iterations 1-3 were real,
independent improvements that remained part of the final setup, and the
single-game/zero-init ablations in iteration 9 were exactly the right
kind of test to run before suspecting the data pipeline — they just
happened to be chasing a symptom of a bug elsewhere.

---

## 4. Stage 2 — curiosity-driven exploration agent

### 4.1 Design goal

Stage 2's premise (per the staged build plan) is that even an imperfect
world model can still drive useful exploration, since exploration only
needs a *ranking* signal over candidate actions, not perfect prediction.
Built `Curiosity` (`ARC-AGI-3-Agents/agents/templates/curiosity_agent.py`),
reusing the Stage 1 encoder/predictor checkpoints as-is.

### 4.2 Bugs found and fixed, each measured against a matched-budget
comparison against the framework's `Random` agent

1. **Static-prediction ranking got the agent stuck.** The first design
   ranked candidate actions by the predictor's predicted residual
   magnitude — a function of the current frame alone, with no feedback
   loop from whether that prediction was actually right. This caused the
   agent to cycle the same ~4 actions forever (0 levels completed across
   a 300-action matched-budget run), since a consistently
   highest-predicted action never got penalized for actually producing no
   real change. Fixed by switching to *observed* prediction error instead
   — real curiosity in the RND/ICM sense, tracked as a per-action running
   exponential moving average, optimistically initialized so untried
   actions get tried first.
2. **Treating every click location as a separate top-level option was a
   reconnaissance tax.** ACTION6 (a click action needing an x, y
   coordinate) was initially expanded into 64 separate top-level ranking
   options, one per 8x8 patch, tied at the same optimistic starting value
   as every simple action — forcing roughly a 64-action mandatory raster
   scan of every click location before the agent could do anything else,
   burning about 20% of the action budget on reconnaissance regardless of
   whether clicking even mattered for a given game. An 8-repeat
   matched-budget comparison after fixing bug 1 alone showed the agent
   completing 8 total levels versus random's 10 — worse than random.
   Fixed by making ACTION6 compete as a single top-level option (mean
   surprise pooled across all patches), only consulting the finer
   per-patch surprise map, via weighted-random sampling, once ACTION6 is
   actually chosen.
3. **Clicking always at a patch's exact center threw away most of the
   available precision, and the exploit phase blindly repeated actions
   across level transitions.** Fixed both: sample a uniformly random
   pixel within the chosen patch rather than always the center, and
   reduced the number of automatic repeats after a score increase
   ("exploit") from 5 to 2, since a level-up typically means the board
   just changed underneath the agent, so blindly repeating the same
   action for many more steps isn't necessarily still sensible. Re-ran
   the same comparison: 10 total levels versus random's 10 — a real
   improvement over bug 2's -2 gap, though only tied on raw count.

### 4.3 Final design

- Rank candidate actions by a per-action EMA of *observed* surprise
  (predicted vs. actual encoded outcome), optimistically initialized.
- A 25% epsilon-random fallback to avoid fixating on something that's
  genuinely, repeatedly surprising but never actually productive (the
  classic "noisy TV" problem — a proper fix for this was deliberately
  deferred to a later stage with a real information-gain signal).
- ACTION6 competes as one option at the top level; the finer per-patch
  map is only consulted once it's chosen, with a uniformly sampled pixel
  within the winning patch.
- On any observed increase in levels completed, immediately repeat the
  last action for up to 2 more steps.

### 4.4 Results

Raw total levels completed across an 8-repeat, 25-game, matched
300-action-budget comparison came out tied with random (10 vs. 10) — not
an unambiguous win by that metric alone. But the per-game breakdown told
a more useful story: the curiosity-driven agent solved 6 distinct games
across its runs versus random's 2, and 7 of its 10 total successes were
on games random never solved once across the same number of trials.
Random was actually more action-efficient on the one game both agents
found easy (suggesting that game's fastest path is close to lucky random
search rather than anything genuinely "surprising"), but curiosity traded
a little efficiency there for reaching several harder games random's
blind search essentially never touched. Given how concentrated that
pattern was (7 of 10 successes on random's blind spots, not scattered),
this was treated as a real directed-exploration effect and the milestone
was considered reasonably met, rather than continuing to chase a larger
raw-count margin on a metric this sparse (1-3 level-ups per full 25-game
sweep) at this sample size.

After the Stage 1 action-recording bugfix (which the curiosity agent
inherited automatically, since it loads the same checkpoints), the same
8-repeat comparison was re-run out of due diligence. Raw count barely
moved (9 vs. random's 10), but distinct games reached ticked up slightly
and included a game random never solves — suggesting the exploration
signal was already "good enough to be useful" even riding on the
previously under-trained predictor, and that fixing the world model
helped the model's own internal metrics far more than it moved this
particular agent's win count at this small a sample size. A reminder that
"the world model got measurably better" and "the agent built on it wins
more, at n=8 trials on a sparse metric" are two different claims and
shouldn't be conflated.

---

## 5. Stage 3 — memory: recurrent core and exact transition recall

### 5.1 Mamba substitution

The staged plan calls for a Mamba-based sequence model as this stage's
recurrent core. `mamba-ssm` has no prebuilt wheel for this Windows
machine, and building it from source requires a local CUDA toolkit that
exactly matches the installed torch build — this machine has a newer CUDA
toolkit (13.0 via `nvcc`) than the torch build in use (`cu126`), a real
version mismatch, not just a missing prebuilt wheel (confirmed via a
failed from-source build attempt). Substituted a `GRUCell`-based recurrent
core instead, which satisfies the actual functional requirement — carry
compressed history across an episode — without the fragile native-build
dependency. Worth revisiting real Mamba if the toolkit and torch versions
are ever aligned on this machine.

### 5.2 Built

- `jepa/data/sequences.py` — loads local recordings as ordered
  *per-episode* sequences rather than shuffled independent transitions,
  chunked into fixed 16-step windows for truncated backpropagation through
  time. Local recordings only — the external bulk dataset has no clean
  per-episode boundaries in its schema, so it can't be sequenced the same
  way.
- `jepa/models/recurrent_predictor.py` — keeps the Stage 1 predictor's
  spatial per-patch residual design, adds a `GRUCell` fed a pooled feature
  summary plus the existing action/coordinate/game conditioning at each
  step; the resulting hidden state is broadcast back in as an additional
  conditioning channel on the next step. Hidden state resets at the start
  of each episode and persists across steps within an episode/chunk.
- `jepa/train_recurrent_predictor.py` — trains via truncated BPTT across
  each 16-step chunk (gradients flow across the full chunk; the hidden
  state does not persist across separate chunks of the same episode — a
  deliberate simplification that still lets the model learn from recent
  history).
- `jepa/memory.py` (`TransitionGraph`) — an exact, lossless memory: a
  dictionary keyed on a hash of exact frame content, mapping
  `(action, click location) -> next state` for every transition actually
  observed, persisted for an agent's entire lifetime across a game
  (including every reset, not just one level attempt — since this
  competition's resets return to an identical starting frame, discoveries
  from an earlier attempt are exactly recallable on a later one). Unit
  tested standalone — deterministic hashing and correct best-known-action
  tracking — before wiring it into an agent.
- `Memory` agent — combines both components: before falling back to the
  curiosity-style exploration ranking (inherited unchanged from Stage 2,
  including its bug-fix history), it checks whether the exact current
  frame already has a known winning action recorded, and takes it
  immediately if so, with no re-exploration. It also uses the transition
  graph to avoid re-trying action/location pairs already attempted from
  the exact current state while untried ones remain — a form of
  guaranteed local coverage independent of the global surprise ranking.

### 5.3 A framework quirk discovered along the way

Setting a `reasoning` attribute on an action object (a convention already
present in the codebase before this stage, used throughout) turns out to
do nothing — the underlying action type has no such field, and nothing in
the framework serializes or logs it anywhere. It's a harmless but inert
convention, not a bug introduced at this stage, but worth knowing: it's
not a usable debugging channel. Replaced it with a real logging call for
one specific event worth being able to confirm (recalling a known winning
action from the transition graph).

### 5.4 Results

The first recurrent-predictor training run was exposed to the exact same
action-recording bug documented in Stage 1 — and more severely, since the
per-episode sequence loader reads *only* local recordings, with no
external-data fallback to dilute the effect. That first result showed
roughly identity-parity, which in hindsight was never a meaningful test
of whether the recurrent core helped at all, since the model was trained
with every action mislabeled as a reset. After the same fix (regenerate
local recordings, retrain): **changed-patches improvement of +21.3%** — a
clean, substantial win consistent with Stage 1's own post-fix result.

An 8-repeat, matched-300-action-budget, 25-game comparison across random,
curiosity, and memory agents (post-fix checkpoints throughout) showed
`Memory` reaching the most distinct games of the three (5, versus
curiosity's 4 and random's 2), including two games curiosity's own sample
hadn't reached in the same round. At this sample size that's still weak
evidence rather than a decisive result, but it's consistent with the
same pattern already observed going from random to curiosity: the added
components help *reach* (diversity of games solved) more clearly than
they move the raw *count*. The exact-recall mechanism's live firing in a
genuine multi-reset run wasn't specifically confirmed via logs this
round — worth verifying directly in a future session by forcing a
known-winning-state repeat and checking for the corresponding log line.

---

## 6. Stage 4 — mixture-of-gated-experts predictor

### 6.1 Scope decision, made upfront

The architecture spec's version of this stage assumes pretraining across
diverse generated grid environments (MiniGrid, Sokoban, Crafter, procgen)
on much larger compute than what's available here, with 16-24 experts.
The first attempt at this stage deliberately skipped that data-diversity
requirement — neither the data pipeline nor the compute budget for it
existed yet — and trained on the existing ~55k-transition combined
ARC-3 corpus alone, with a smaller 8-expert configuration. That attempt's
own conclusion (that closing the remaining gap most likely needed the
originally-specified diverse multi-environment data) turned out to be
correct, and is exactly what the later MiniGrid pipeline addressed.

### 6.2 Built

- `MoEPredictor` — K small pointwise-convolution expert networks plus a
  gate (pooled features and conditioning vector, softmax over K experts),
  combined as a weighted sum: `next_feat = feat + sum_k gate_k * expert_k(...)`.
  Each expert is deliberately small and shallow — the intent is many
  specialized small functions discovered during training, rather than one
  large general one (which is what the monolithic Stage 1/3 predictors
  already tried). A "combiner network" for blending pairs of experts into
  compound effects, described in the architecture spec, was explicitly
  deferred as out of scope for this stage.
- A Switch-Transformer-style load-balancing auxiliary loss
  (`N * sum_i(f_i * P_i)`, minimized at uniform expert usage, maximized at
  total collapse onto one expert).
- `train_moe_predictor.py` — training setup matching Stage 1's approach
  (shuffled independent transitions; expert routing doesn't need Stage
  3's temporal ordering), later extended with a two-phase
  pretrain-then-finetune curriculum: pretrain on MiniGrid data, then
  fine-tune on the ARC-3 corpus, sharing one encoder, one predictor, one
  optimizer, and one game vocabulary built from the union of both data
  sources up front (so the game-embedding table has consistent size and
  meaning across both phases).

### 6.3 First attempt: ARC-3 data only

1. **Expert symmetry bug.** The first training run (8 experts, a
   load-balancing weight matching typical large-scale MoE recipes) showed
   the gate collapsing to an exact constant — 0.125 for every expert,
   effectively zero variance across an entire validation batch, regardless
   of input. Root cause, found by direct inspection rather than
   guesswork: every expert's last layer had been zero-initialized,
   mirroring the "start as an identity function" trick that had worked
   well for the single monolithic predictor in Stage 1. With all K
   experts producing bit-for-bit identical (zero) output at
   initialization, and a uniform gate, every expert receives the exact
   same gradient every step and stays identical to every other expert
   forever — a genuine symmetry that gradient descent cannot break on its
   own, not a tuning problem. The zero-init trick was correct for a
   single predictor; it's specifically wrong for a multi-expert setup
   with shared, symmetric initialization. Fixed with small random
   initialization instead of zero.
2. With the symmetry fixed and the load-balancing weight reduced 10x,
   experts did start producing genuinely different outputs from each
   other (verified directly by measuring output variance across experts),
   but the gate itself still converged toward uniform blending over 60
   epochs, and changed-patches improvement was only in the 0.3-0.9% range
   — far short of the fixed-monolith result on the same corrected data
   (+29.2%, see Stage 1).
3. Ruled out "the auxiliary loss weight is still too strong" as the sole
   explanation by testing a further 10x and 100x reduction: the
   load-balance loss still drifted back toward uniform by the end of
   training in both cases, just more slowly — uniform blending appears to
   be a genuine attractor of the main prediction loss itself, not purely
   an artifact of the auxiliary term.
4. Tested the opposite extreme — zero load-balancing weight, no
   balancing pressure at all — and got the classic opposite MoE failure
   mode instead: near-total collapse onto a single dominant expert. So
   the useful range for this loss weight isn't a smooth dial the way it
   often is in large-scale recipes; it behaves more like a bimodal switch
   between uniform-blend and total-collapse on this data scale.
5. Tried fewer experts (3 instead of 8) in case 8 experts was simply
   diluting a fixed amount of data too thin per expert: same story
   (load-balance loss converges to uniform, changed-patches at roughly
   identity parity).

Working conclusion at this point: with this amount of data (~55k
transitions, far short of the diverse multi-environment corpus the
architecture spec assumes) and this expert architecture (small, shallow
per-expert networks), a near-uniform blend of all experts is simply a
more loss-effective solution than genuine sparse routing — plausibly
because averaging several small, individually noisy experts reduces
variance in a way that helps a mean-squared-error objective directly, so
the optimizer has no real incentive to commit to sparse routing no matter
how the balancing term is tuned. This lines up with the same "data-bound,
not capacity-bound" conclusion Stage 1 eventually reached, just showing up
as a different symptom: closing the gap looked like it needed the
originally-specified diverse data source, not more loss-weight tuning on
the existing ARC-3-only corpus.

### 6.4 MiniGrid data pipeline

Built a translation layer and trajectory generator for MiniGrid
(a lightweight, gymnasium-based suite of grid environments) specifically
because its action semantics are *consistent* across all its
environments and episodes — unlike ARC-3, where the same action id can
mean a completely different thing in each of the 25 different games,
which is exactly the kind of confound Stage 1's iteration 4 had already
shown a per-game embedding alone couldn't paper over.

- `jepa/data/minigrid_data.py` — translates MiniGrid's native grid
  encoding (an `(object, color, state)` triple per cell, which notably
  does not include the agent itself — the agent's position and facing
  direction are overlaid separately) into the same flat 0-15 color-code
  grid representation already used for ARC frames, so the same encoder
  and training code work unmodified on both data sources.
- Covers 21 environments spanning a useful range of dynamics: empty-room
  navigation, key/door interaction, obstacle avoidance (static and
  dynamic), pickup/carry mechanics, and several multi-step puzzle
  environments combining these. All share a single game id
  (deliberately *not* one id per environment) — the entire point of this
  data source is action semantics that stay consistent across many
  different layouts, and giving each environment its own embedding would
  let the model route around learning that consistency rather than
  learning through it.
- Cheap to generate: roughly 2,000 transitions per second under a random
  policy; the full 21-environment sweep produces on the order of 67,000
  transitions in about half a minute.

Two real bugs surfaced while wiring this in, both general fixes rather
than MiniGrid-specific hacks:
- The transition dataset expected a frame's grid to be wrapped in a
  single-element list (matching the convention already used by the ARC-3
  and external-data loaders) — the MiniGrid translator initially returned
  a bare grid array instead.
- The patch-change-mask utility assumed its input was always exactly the
  full 64x64 canvas size (always true for ARC-3 frames, so this had never
  been exercised otherwise) — MiniGrid grids are smaller and vary in size
  from 5x5 up to 25x25. Fixed generally by placing the diff on a
  full-canvas zero array, top-left-aligned (matching the existing
  top-left placement convention used elsewhere for smaller grids), before
  reshaping into patches, rather than special-casing MiniGrid.

A separate, unrelated issue: the first two-phase training run (MiniGrid
pretrain, then ARC fine-tune) hung indefinitely — confirmed by watching
process CPU time stay completely flat for over an hour rather than
merely running slowly. Both phases had been using `persistent_workers`
on their data loaders, and the first phase's worker processes weren't
being fully torn down before the second phase span up its own new set —
some kind of resource contention rather than a clean handoff between
phases. Fixed by turning off `persistent_workers` for this script (its
per-epoch respawn cost barely matters given there's only one phase
transition total) and explicitly releasing the first phase's loaders
before constructing the second phase's.

### 6.5 Results with MiniGrid pretraining

Retrained with a 20-epoch MiniGrid pretraining phase followed by 60
epochs of ARC-3 fine-tuning, 8 experts, evaluated on the same held-out
ARC-3 validation split used throughout this stage.

- **Changed-patches improvement: +44.1%** — clearing the "better
  generalization than the recurrent monolith" half of the milestone
  decisively (beating both the Stage 1 fixed-monolith result of +29.2%
  and the Stage 3 recurrent-monolith result of +21.3% on the same held-out
  data).
- **Gate specialization: real, but still a minority behavior rather than
  the norm.** The load-balance loss sat at roughly 1.006, barely above
  the theoretical uniform minimum of 1.0 for 8 experts, and mean gate
  entropy across the validation set was about 98.6% of the maximum
  (fully uniform) entropy — so most individual inputs still received a
  near-uniform blend across all experts. But unlike every ARC-3-only
  attempt above (where gate weights were essentially constant across an
  entire batch), this run showed genuine per-input structure: roughly
  4.4% of validation examples had entropy meaningfully below uniform, and
  about 3.9% had one expert clearly dominant (weight above 0.3), with
  some individual examples routing almost entirely to a single expert
  (observed weights as high as 0.998). A qualitatively different, much
  less degenerate result than the exactly-flat gate from the ARC-only
  attempts, even though it falls short of specialization being the
  dominant pattern.

Interpretation: MiniGrid's consistent action semantics and much higher
changed-frame rate gave the shared encoder and predictor enough real
signal to learn genuinely better dynamics overall, and gave the gate
enough signal to learn some real routing rather than none at all — but
not enough to make specialization the dominant behavior across most
inputs. The remaining gap plausibly needs more or longer MiniGrid
pretraining, a different pretrain-to-finetune epoch ratio, or a gating
mechanism that forces harder, sparser routing rather than a soft blend.

Overall verdict for this stage: the generalization half of the milestone
is clearly met; the specialization half is measurably, qualitatively
better than the pre-MiniGrid attempts but not yet the dominant pattern —
called "mostly met."

### 6.6 Noisy top-k gating experiment

As a direct attempt at forcing harder, sparser routing rather than a soft
blend, added Shazeer-style noisy top-k gating to the expert model:
per-expert trainable noise added to the gate logits during training
(encouraging exploration across experts early in training), followed by
keeping only the top-k highest-scoring experts per example (softmax over
just those, with the rest masked to zero) instead of a dense blend over
all K. Retrained with the identical MiniGrid-pretrain-then-ARC-finetune
curriculum as the previous result, using k=2.

Result: did not clearly improve on the dense-gate result, and in fact
regressed prediction quality — changed-patches improvement fell to
roughly +21.9%, about half the dense-gate result's +44.1%, landing behind
even the Stage 1 fixed monolith's +29.2% and only narrowly ahead of the
Stage 3 recurrent monolith's +21.3%.

Checked whether the forced sparsity at least bought cleaner, more
confident specialization in exchange for the accuracy drop — it didn't,
really. Which *pair* of experts got selected did vary meaningfully across
examples (per-expert usage counts spread over more than a 2.5x range, so
the top-k selection itself wasn't degenerate), but the *weighting between
the two selected experts* stayed almost perfectly 50/50 on average
(mean entropy about 99.5% of the two-expert maximum). So top-k gating
effectively traded "blend all 8 experts near-uniformly" for "blend 2
(varying) experts near-uniformly" — structurally sparser in which experts
participate, but not more confident about how much each one should
count, and the forced hard cutoff appears to have discarded useful signal
the dense blend could still make use of, which is the most likely
explanation for the accuracy regression. The same "hedge rather than
commit" tendency identified in the earlier ARC-only attempts persisted
even under this harder structural constraint.

This was left as a documented negative result rather than pursued
further (e.g. k=1, longer pretraining, different noise scales) — on the
evidence gathered this session, more/better data (the MiniGrid pipeline)
was a clearly more effective lever than forcing sparser routing on top of
an already-good dense-gate result. The dense-gate, MiniGrid-pretrained
checkpoint (the +44.1% result) is the one carried forward into later
stages, not the top-k variant.

### 6.7 A second synthetic pretraining source (Sokoban): a real, controlled negative result

A later pass at this stage tried adding a second synthetic pretraining
source alongside the grid-navigation environment already in use, with a
specific and different motivation than "more data": the earlier addition
had helped by exposing the shared model to genuinely new, consistently-
labeled action semantics, not just by increasing data volume — so the
natural next question was whether a *third*, mechanically distinct
environment family (this one built around pushing movable objects, with
persistent and sometimes irreversible consequences — a causal pattern the
existing two sources don't reliably exercise) would extend that same
benefit further. A translation layer for this environment family was
built following the same pattern as before: map its native grid
representation onto the shared flat color-code format, generate
unlimited random-policy trajectories cheaply, and give it its own
distinct identifier in the game vocabulary (its action semantics aren't
consistent with the *other* synthetic source's, only internally
consistent with themselves, so sharing an identifier would have been
inappropriate).

Two real, unrelated bugs surfaced before a clean comparison was possible:

- The new environment's action space turned out to be one action larger
  than what the shared model's action-conditioning mechanism was sized
  for (an embedding table originally sized to match the primary game
  suite's own action count, which every other data source added so far
  had happened to fit inside without needing to check). This didn't fail
  at data-generation time — it's just an integer that gets stored — it
  failed much later and much more confusingly, as a low-level index-out-
  of-bounds crash deep inside a GPU kernel, on whatever training batch
  first happened to include the one action id that exceeded the table's
  size. Fixed by dropping that environment's true "do nothing" action
  (already redundant with the trivial "nothing changed" baseline used
  everywhere else) and remapping its remaining actions to fit within the
  existing bound.
- Separately, and unrelated to any code defect: the machine this was
  running on ran completely out of disk space partway through a training
  run, which surfaced as a generic out-of-memory crash inside a data-
  loading worker process, repeatedly, in a crash-and-respawn loop. This
  is a known operating-system-level interaction on the platform in
  use — an almost-full disk can prevent virtual memory from being able to
  grow, turning what would normally be a graceful slowdown into a hard
  failure. The fix was entirely outside the code: freeing real disk space
  (clearing a large, fully regenerable package-manager cache, and
  removing already-fully-documented evaluation output that no longer
  needed to exist on disk) resolved it completely, with no changes to any
  training script at all. Worth remembering as a general debugging
  instinct: a generic out-of-memory error during a long-running job is
  worth checking available disk space for, not just available RAM —
  especially on this platform, where the two are more entangled than they
  might appear.

With both fixed, a controlled comparison was run: the same held-out
validation data, the same primary-game training corpus, and the same
first synthetic source's data (confirmed identical transition counts
across both runs), with the *only* difference being whether the second
synthetic source's data was included in the pretraining mix or not. The
result was a clear regression, not an improvement: changed-patches
improvement dropped from roughly +29.5% (first-source-only) to roughly
+15.7% (both sources combined) on identical evaluation data — nearly
half the improvement lost, not gained. A direct measurement of gate
specialization (the other half of this stage's own milestone) told the
same story: mean routing entropy was effectively identical between the
two checkpoints (both essentially at the fully-uniform-blend ceiling),
and the fraction of examples showing one clearly dominant expert was, if
anything, slightly lower with the second source included than without
it — so the addition didn't help specialization either, and nudged it
mildly in the wrong direction on that one measure.

The most likely explanation, not directly verified this round but worth
checking first if this is revisited: the specific mechanic this second
source was chosen for is known to produce irreversible dead ends under
truly random play — a single careless move can permanently make part or
all of a puzzle unsolvable, after which the remainder of that episode is
just aimless wandering in a now-frozen, uninteresting arrangement. A
basic frame-level "did anything visibly change" check looked healthy and
comparable to the first synthetic source's own rate, but that check
can't distinguish meaningfully varied dynamics from directionless
wandering after an irrecoverable mistake — so a real portion of this
second source's generated data may have been low-information noise
diluting the pretraining signal rather than the intended enrichment,
unlike the first source, where even a random policy still produces
reasonably varied experience throughout an episode. A non-random,
curriculum-style, or reduced-difficulty data collection approach for
this specific source would be the natural thing to try first before
concluding the mechanic itself isn't useful, if a future pass revisits
it — but per the explicit scope this was undertaken under (try it, and
if it doesn't clearly help, move on to a different lever entirely), this
was treated as a clean, honest negative result and not iterated on
further in the moment. The previously-working checkpoint was left
completely untouched throughout this experiment (the comparison runs
wrote to separate, clearly-labeled output locations specifically to
avoid any risk of overwriting it), so there was nothing to roll back.

---

## 7. Stage 5 — hypothesis bundle and directed action selection: milestone met

### 7.1 Design

Builds on Stage 3's `Memory` agent, reusing its exact transition-graph
recall and exploit-on-score-delta behavior unchanged, and replaces the
Stage 2/3 exponential-moving-average "observed surprise" ranking with a
more principled design:

- **N parallel hypotheses.** Rather than building a separate
  hypothesis-search structure from scratch, Stage 4's K=8 trained mixture
  experts are reused directly as the hypothesis pool: each expert already
  represents one learned candidate function for "what does this action
  do," so each one is treated as one hypothesis. `MoEPredictor` gained a
  `predict_all_experts` method returning the raw, ungated per-expert
  predictions for a given state/action, bypassing the gate entirely.
- **Bayesian confidence tracking** (`jepa/hypothesis_bundle.py`,
  `HypothesisBundle`): after each observed transition, every hypothesis's
  (expert's) prediction error against the actual outcome updates a
  running confidence weight via `p(H_i) *= exp(-error_i / tau)`,
  renormalized. This tracks which experts have actually been reliable
  *in this specific game, so far* — independent of how the gate itself
  was trained to route.
- **Entropy-driven explore/exploit blend.** The entropy of the confidence
  distribution maps to a blend parameter beta: high entropy (hypotheses
  still disagree about what to trust) favors exploration; low entropy
  (one or a few hypotheses have clearly proven reliable) favors
  exploitation.
- **Action scoring:** `Q(s, a) = (1 - beta) * InfoGain(a) + beta * V(next_state(a))`.
  InfoGain(a) is the variance across the K hypotheses' raw predictions for
  candidate action a — how much they currently disagree about its effect,
  i.e., how informative actually taking it would be. This falls out of
  the same single forward pass used for scoring, and for ACTION6 the
  same per-patch variance map doubles as a click-location salience map,
  avoiding a separate 64-patch scan. V is a small decoupled value head
  trained separately (see below), confidence-weighted across the K
  hypotheses' predicted next states.
- **Opening probes.** At the start of each episode, every simple action is
  tried once (mirroring the Stage 0 `PressOnce` agent's pattern) before
  switching over to the Q-driven policy, so the hypothesis bundle has real
  per-action signal before it starts trusting its own confidence weights.

### 7.2 Value head

A small MLP sitting on top of pooled encoder features, trained separately
via discounted Monte Carlo returns computed from the sparse
`levels_completed` reward signal available in gameplay recordings
(`jepa/models/value_head.py`, `jepa/train_value_head.py`).

A latent-space mismatch was found and corrected: the value head had
originally been trained against the Stage 1 single-predictor encoder's
checkpoint, but the hypothesis-bundle agent loads the separately-trained
Stage 4 mixture-expert encoder checkpoint to produce features for both
the experts and the value head — two encoders trained independently end
up with different, incompatible latent spaces, so a value head fit to
one produces close to noise when fed features from the other. Fixed by
retraining the value head directly against the correct encoder checkpoint.
Worth flagging even after the fix: the value head's training data is
extremely sparse (roughly 12k samples, only about 1.6% with a nonzero
value target, since level-completion events are rare under a
mostly-random policy) — its validation error after retraining sits right
on top of a trivial "always predict zero" baseline, i.e. it's only
marginally distinguishable from that baseline. Not a bug, just an honest
limit of what this component can learn from how sparse the reward signal
currently is — and, as it turns out, a significant factor in what
follows.

### 7.3 Evaluation: three more real bugs found, milestone not cleanly met

The first full evaluation (matched 8-repeat, 300-action-budget, 25-game
comparison against the curiosity agent, the same protocol used for every
earlier stage comparison) came back with the hypothesis-bundle agent
completing zero levels across 200 runs, against the curiosity agent's
eleven. Tracing through an actual episode's action log showed the agent
repeatedly selecting the click action almost every single turn until the
game ended, resetting, and repeating the same pattern. Direct code
inspection of the action-scoring function found the cause: the click
action's information-gain score was computed as a *maximum* over the 64
spatial patches, while every other action's score was computed as a
*mean* over all patches and channels — an apples-to-oranges comparison
where a maximum over many values is almost always larger than a global
mean, so the click action was structurally near-guaranteed to score
highest regardless of what the underlying hypotheses actually predicted.
Fixed by using the same mean-based reduction for both, while still using
the patch-level maximum separately to choose *where* to click once that
action is actually selected.

Re-evaluating after that one fix improved things (three total levels
completed, one distinct game) but still fell far short of the curiosity
agent's eleven. Tracing episodes again revealed a new pattern: the agent
would settle on a single action and repeat it for the rest of an entire
episode's budget, then settle on a different single action after the
next reset. This is the same failure mode encountered and fixed earlier,
in the curiosity agent's very first bug (see section 4.2 above): an
information-gain signal computed fresh each turn from a near-deterministic
prediction has no built-in mechanism to "cool down" once an action's real
effect turns out unsurprising in practice — unlike an exponential moving
average of *observed* prediction error, which naturally does. Fixed the
same way that earlier bug was fixed: added a 25% chance of taking a
uniformly random action instead of the greedy best-scoring one.

A remaining diagnostic question was whether the entropy-driven
explore/exploit blend was actually behaving as intended. A standalone
script that replays real recorded episodes through the confidence-update
logic (without actually playing any games) found that the blend
parameter averaged around 0.76 and exceeded the explore/exploit midpoint
in 86% of transitions — meaning the agent was spending nearly the whole
episode trusting the value estimate rather than the information-gain
signal, not the intended balanced blend. The root cause was that the
underlying confidence-accumulation update had no forgetting term: it
summed evidence across every single transition since the start of an
episode, and with the update's temperature parameter small relative to
typical prediction-error differences between hypotheses, even tiny or
spurious differences compounded without bound over a few hundred steps —
confidence collapsed toward "certain" within the first few dozen actions
of essentially every episode and then stayed there regardless of later
evidence. Given the value head's own signal is close to a zero baseline
(as noted above), this meant action selection was effectively driven by
near-noise for most of an episode. Fixed by adding a geometric forgetting
factor to the confidence update, so it tracks which hypothesis has been
reliable *recently* rather than accumulating certainty forever. A sweep
across several forgetting-factor values on the same replayed episodes
found a clear middle ground: strong enough to prevent runaway certainty,
without decaying so aggressively that the value estimate never gets to
matter at all (which would reduce the whole design to the earlier
information-gain-only ranking with extra steps).

With all three fixes applied, a final matched evaluation completed one
total level across the full 200-run sweep, on one distinct game — still
behind the curiosity agent's eleven total, five distinct games, on the
identical protocol. The milestone (directed exploration should beat
curiosity on the same levels in fewer actions) is not cleanly met. A
same-game, matched comparison on the one game both agents solved in this
round gives a more nuanced picture than a flat loss, though: the
hypothesis-bundle agent solved it in 53 actions on its one success,
well under the curiosity agent's own average of roughly 175 actions
across its six successful attempts on that same game out of eight tries
— but the hypothesis-bundle agent only succeeded on one of its eight
attempts, against curiosity's six of eight. So on the one clean
head-to-head data point available, the newer design is markedly faster
when it succeeds but considerably less reliable overall — a genuine,
honest tradeoff rather than an unambiguous win either way.

The most likely explanations for the remaining reliability gap are both
already-documented, data-bound limitations from earlier work rather than
something another round of parameter tuning would resolve: the value
head's training signal is roughly 98% zero-valued, so meaningfully
improving it would need trajectory data with a much higher rate of real
level-completion events — data from a policy that actually makes
progress, not more volume of the same largely-random policy; and the
underlying mixture-of-experts model's own gate specialization was
already found, in the previous stage, to be real but a minority behavior
rather the norm, which caps how informative a confidence mechanism built
directly on those experts' disagreement can be in the first place. Both
point toward better or more purposeful training data as the well-scoped
next lever, consistent with a recurring theme across this whole project:
add a new component or tune a parameter only when a specific, measured
bottleneck calls for it, and prefer a data-side explanation over further
tuning when the two are hard to tell apart.

### 7.4 A "teacher" data source for the value head: a real component-level win, still unproven at the full-agent level

Following up on the exact conclusion above, and after the second
synthetic-pretraining attempt came back negative, the originally-scoped
fallback plan was tried: generate denser, more purposeful training data
by reusing an already-built, already-validated exploration policy as a
data-collection tool, rather than building an entirely new search or
reinforcement-learning system from scratch. The existing memory-and-
recall agent from an earlier stage already does non-repeating, curiosity-
ranked exploration and keeps its exact state-transition memory across
resets — running it with a much larger action budget than its usual
comparison budget was a low-risk way to approximate directed search
without new exploration logic. The budget was temporarily increased
roughly eightfold for a single data-collection pass across all 25 games,
then immediately reverted afterward, following the same
temporarily-adjust-and-revert pattern already used elsewhere in this
project for fair comparisons.

That single pass produced real completion events on five of the twenty-
five games — a meaningful density improvement in absolute terms over the
existing corpus, where such events were rare enough that a value-
estimating component had previously found almost nothing to learn from.

The first attempt to actually use this new data, however, was a genuine
negative result, and an instructive one: retraining the value-estimating
component on the combined corpus, with an ordinary unweighted loss,
produced a component that had learned nothing at all — its error on
held-out data matched a trivial "always predict nothing happens"
baseline exactly, epoch after epoch. The cause traced back to episode
length: the newly added data came from episodes roughly thirty times
longer than the rest of the corpus, and a discounted-return target decays
to a technically-nonzero but practically meaningless value hundreds of
steps before any actual reward event. The vast majority of the newly
added samples were, for training purposes, indistinguishable from the
already-overwhelming majority of true-zero samples — diluting the
signal rather than enriching it, purely because of how the loss weighted
them. This is precisely the same class of problem encountered very early
in this project's very first model, when an unweighted loss dominated by
a trivial majority gave almost no usable signal for the rare, important
minority case — and it was fixed the same way here as it was fixed
there: oversampling the meaningful examples during training.

With that fix applied, the picture became more interesting rather than
simply "fixed." Evaluated the naive way — average error across the
entire held-out validation population — the retrained component now
looked *worse* than the trivial baseline, not better. But this is exactly
the same trap as evaluating a whole grid instead of just the parts that
changed, encountered and corrected very early in this project: the vast
majority of validation examples are still trivially uninteresting, so an
average over all of them mostly just measures how well a component
handles the boring majority, not whether it has learned anything useful
about the rare cases that actually matter. Restricting evaluation
specifically to the meaningful examples told a different, more honest
story: a real, roughly one-third reduction in error relative to the
trivial baseline on exactly those examples, alongside a genuine (if
modest) positive correlation between predicted and true values. A
follow-up check for a systematic bias on ordinary, uninteresting states
found only a small one — not the kind of collapse that would make the
component useless in practice.

Whether this translated into a better-performing full agent, though,
remains genuinely unresolved rather than confirmed either way. Re-running
the same matched comparison against the curiosity-only baseline agent
produced zero total level completions this round for the hypothesis-
bundle agent, down from a single completion in the prior best run —
on the surface, a regression. But this agent's best-ever observed success
rate across any prior round was already only about one in eight
attempts, and under that same rate, seeing zero successes in eight fresh
attempts would be expected to happen roughly a third of the time purely
by chance — not an unusual outcome at all, and nowhere near strong enough
evidence to conclude the change made things worse. This is the same
sparse-metric caution that has come up repeatedly throughout this
project: a raw win count at this sample size simply doesn't have enough
resolution to distinguish a real, moderate improvement in one internal
component from ordinary run-to-run noise in the final outcome.

One piece of housekeeping worth recording: the long data-collection
episodes produced individual recording files far larger than anything
else generated in this project (some in the hundreds of megabytes,
apparently reflecting genuine per-game differences in how much
information a single frame carries, magnified thirty-fold by the much
longer episodes) — briefly ballooning total disk usage for this kind of
data to several gigabytes, on a machine that had already run out of
storage once earlier in the same broader effort. All of it was deleted
once results were extracted and the retrained component itself was
safely saved, since none of it is uniquely irreproducible — the exact
steps to regenerate it are documented. A comparison utility written
earlier was also found to be needlessly slow because of this — it parsed
every recording file before filtering by which agent produced it, paying
the cost of reading multi-hundred-megabyte files even when they weren't
the ones being asked about — and was fixed to filter before parsing
instead.

The honest summary: the underlying component genuinely, measurably
improved on the metric that actually reflects what it's supposed to do.
Whether that improvement is large enough to show up in a full end-to-end
comparison is still an open question, not because the fix failed, but
because eight attempts per condition isn't enough resolution to tell a
real moderate improvement apart from noise on a metric this coarse and
this rare an event. Resolving that would need either substantially more
repeated trials, or a finer-grained way of measuring progress that
doesn't require an actual full game win just to register any signal —
not further changes to the data-generation approach itself, which
already did what it was meant to do.

### 7.5 Finding the actual bottleneck: a real bug, found and fixed with direct evidence

Rather than guessing at further fixes, the next step was to go looking
for direct evidence of exactly where the hypothesis-driven agent was
failing, in a few cheap steps before changing any code.

First, confirmed that the exact-recall mechanism carried over from the
memory-based agent wasn't silently broken — a loose end left unconfirmed
much earlier in this project. Running the agent on a handful of games
showed the "recalling a known winning action" log line never firing, but
that turned out to be fully explained by zero level completions ever
happening in that run to record in the first place, not a broken
mechanism with nothing to show for it.

Second, added temporary step-by-step logging of the internal scoring
values behind every decision and watched a full real episode play out.
Two things stood out immediately on a game already flagged elsewhere in
this project as involving a lot of on-screen activity: the numeric gap
between candidate actions was frequently tiny enough to be indistinguishable
from noise, and — more strikingly — the click-based action scored lowest
of the available options in nearly every single decision across an
entire episode, on a game where clicking plausibly matters.

Tracing why revealed a real, previously undiagnosed architectural issue.
The click action was being scored at a fixed, neutral point on the grid
— the same point used for every other action — under the reasoning that
the coordinate information shouldn't bias which regions looked
interesting, only the action type should. That reasoning held, but it
missed something: the click action's own score was never being evaluated
at the specific location it would actually pick if chosen. A first
attempt to fix this — re-scoring it at its own best-looking location
instead of the neutral point — changed nothing at all when tested. Digging
into why revealed that the underlying model's coordinate conditioning
is applied as a single uniform shift across every region of the grid
simultaneously, not as a way of directing attention to one specific
region — so telling it "evaluate as if acting here" doesn't actually make
it look at "here" any differently than anywhere else. That attempted fix
was reverted once confirmed to be dead weight.

The real lever turned out to be how the click action's informativeness
score was being *summarized* across the grid, not which coordinate it
was conditioned on. Averaging across the entire grid (the correct fix for
an earlier, different bug where the click action had been unfairly
favored) structurally underrates an action whose true value comes from
one good location out of many, not an average across mostly-irrelevant
ones — while the original approach of taking the single highest value
across the grid overrated it instead, for an unrelated statistical
reason: the maximum of many samples tends to look larger than a single
evaluation purely from having more chances, regardless of whether any of
them are genuinely meaningful. The fix settled on a middle ground —
averaging across only the top handful of regions instead of either the
single best one or the entire grid — applied identically to every
action's scoring, not as a special case for the click action alone,
which would have just reintroduced a version of the original problem.

The result was a clean, unambiguous improvement: from zero level
completions across an entire matched evaluation sweep to six completions
across three distinct games, on a fresh identical sweep. This was the
first fix in this whole line of investigation with directly measured,
unambiguous improvement behind it — not something bounded by sampling
noise or only verified in isolation from actual play.

With that confounding bug out of the way, one more question remained
worth answering directly rather than assuming: was the strategy of
blending an exploration signal with a value estimate actually earning
its complexity, or would either half alone have done just as well? Ran
the same matched evaluation three times, each with the blend fixed to
one extreme or left at its normal adaptive setting: pure exploration
signal alone reached one distinct game, pure value-estimate greediness
alone also reached only one distinct game, and the normal adaptive blend
reached three distinct games with more than double the total completions
of either extreme (some of these comparison runs were shortened by an
unrelated, intermittent connectivity hiccup during game setup, but the
remaining sample sizes were still large enough for a clear read).
Neither ingredient alone matched the combination — a real, validated
confirmation that the original design decision to blend the two signals,
rather than relying on just one, was sound.

**Correction, checked immediately afterward: the milestone is still not
met.** The zero-to-six-completions result above is genuine, valid
evidence that the fix helped the hypothesis-driven agent relative to its
own earlier, broken state — but that comparison was only ever run against
the agent's own prior baseline, never re-checked against a *fresh* run of
the comparison agent in the same round. The comparison agent's own raw
numbers had already been observed to swing considerably from round to
round earlier in this project, so a number measured in isolation doesn't
say where it stands relative to a moving target. Running the missing,
properly matched, same-round comparison gave a clearly different
picture: the comparison agent reached five distinct games this round
against the hypothesis-driven agent's one, and led on total completions
too. The fix was real and worth keeping, but on its own it wasn't enough
to close the gap — the milestone remains unmet, and it would have been
misleading to let the earlier, incomplete comparison stand as the final
word without going back and checking it properly.

### 7.6 The milestone is met: a second real bug, found by tracing exactly where the two agents diverged

Rather than continuing to re-run the full sweep hoping for a different
outcome, the next round of investigation deliberately narrowed to just
the handful of games where the comparison agent had succeeded and the
hypothesis-driven one hadn't. The reasoning: if there's a genuinely
fixable bottleneck, it should show up clearly on exactly those games, not
get averaged away across the majority of games where both agents already
behave similarly for unrelated reasons.

Live-tracing those specific games surfaced something the earlier,
broader trace had completely missed: on three of the four, only a single
type of action was ever available at all. There was no "which action"
decision happening on these games whatsoever — the entire outcome
depended purely on *where* on the grid a click landed. This meant the
earlier fixes (the scoring-reduction fix, the exploration-blend
validation) were structurally irrelevant to these specific games, since
both operated on a comparison between different action types that never
actually occurred here.

Pulling the actual click coordinates played across a full episode on
each of these games revealed real underlying diversity (dozens of
distinct locations attempted), but with one specific point dominating
overwhelmingly in each case — and that dominant point turned out to sit
exactly at the center of the very first grid region in reading order.
That's the signature of a deterministic "pick the single best-looking
option" rule defaulting to the same fixed choice whenever its input is
flat or nearly flat, rather than a genuine, confident preference — on
turns where nothing about the current state stood out spatially
(plausibly the common case, not a rare edge case), the mechanism fell
back to clicking the same likely-uninformative spot again and again
instead of continuing to explore.

The telling part: an earlier stage of this same project had already
solved this *exact* problem for its own click-selection logic — using a
softened, temperature-weighted random choice among candidate regions
instead of a hard "pick the single best one" rule, plus a randomly
chosen point within the chosen region rather than always its exact
center. The hypothesis-driven agent, built later and reusing much of
that earlier agent's design in other respects, had never actually
inherited this specific, already-proven fix — its own click-selection
logic was written fresh and reintroduced the identical problem the
earlier agent had already moved past. Fixed by directly reusing the same
approach.

Verified narrowly before scaling back up: replaying a single episode on
the same three games and counting distinct click locations showed the
fix working exactly as expected — from roughly a quarter of clicks being
genuinely distinct locations to nearly every single click landing
somewhere new. A focused, faster comparison on just the four diverging
games (skipping the other twenty-one, which weren't where the gap was)
showed real completions appearing on two of the four where there had
been none before. Only then was it worth spending the time on a full,
matched, same-round comparison again.

That full comparison finally gave an unambiguous result: the hypothesis-
driven agent reached more total level completions, more distinct games,
and did so in fewer actions on average than the comparison agent, all
three at once, in the same round. This is the first point across this
entire line of investigation where every relevant number pointed the
same direction simultaneously, rather than a mixed or noise-plausible
picture. The milestone this whole effort was built around is met.

The most useful general lesson from this last round: every earlier fix
in this investigation had been found either through broad, generic
tracing or by addressing an already-documented, general limitation —
useful, but not specifically targeted at the actual, narrow place where
the two agents' results diverged. Deliberately narrowing the
investigation to just the cases where a comparison actually disagreed,
rather than continuing to search across the full, averaged population,
surfaced a bug that had been real and present all along but invisible in
pooled results — because it only mattered decisively on a subset of
cases where one particular option was the *only* option, diluted into
near-invisibility everywhere else. When two things being compared
diverge, tracing the specific cases where they diverge is far more
informative than re-examining the whole population again.

---

## 8. Lessons and gotchas worth remembering

- **A shared upstream data bug can produce a pattern that looks exactly
  like "we've exhausted every fixable hypothesis."** When several
  genuinely different, well-targeted fixes all converge on the same
  negative result, that's a signal to go audit the data pipeline itself
  before concluding an architecture is capped — see Stage 1's action-id
  recording bug, which invalidated most of that stage's early conclusions
  and was found only while debugging an unrelated later stage.
- **Any "does X beat baseline Y" comparison needs both sides computed
  under strictly matched conditions.** Comparing a predictor's output
  against a lagging EMA target instead of the same encoder used for the
  predictor's own input silently inflates the baseline's apparent error
  and can flip a real failure into an apparent pass.
- **A trivial "predict no change" baseline is dangerously easy to beat on
  a naive whole-grid metric.** Isolating comparisons to patches or frames
  that actually changed is what actually stress-tests a dynamics model;
  whole-grid MSE mostly measures how well a model reproduces a mostly
  static background.
- **An initialization trick that's correct for one architecture can be
  actively harmful for a structurally different one.** Zero-initializing
  a residual branch is a good default for a single predictor (starts as
  an identity function, has to earn any deviation); zero-initializing
  *every* expert in a multi-expert model creates an unbreakable symmetry
  where all experts receive identical gradients forever, since they all
  start bit-for-bit identical.
- **A loss weight can be a bimodal switch rather than a smooth dial.**
  The mixture-of-experts load-balancing loss weight didn't trade off
  smoothly between "uniform blend" and "genuine routing" as it was
  tuned down across several orders of magnitude — it stayed near-uniform
  across a wide range and then flipped to near-total collapse at zero,
  with no useful middle ground found on this data scale.
- **Forcing structural sparsity (top-k routing) doesn't automatically buy
  confidence.** A model can satisfy a hard sparsity constraint (few
  experts selected) while still hedging almost equally between the
  selected ones — sparser is not the same property as more decisive, and
  chasing the former didn't produce the latter here.
- **A silently-swallowed field in a data conversion step will run to
  completion without ever raising an error.** The single missing
  `action_input=raw.action_input` assignment produced a fully-valid,
  fully-parseable recording file with a constant, wrong value in one
  field — nothing about the pipeline downstream had any way to notice.
  Sanity-checking a label's distribution (is this supposedly varied field
  actually constant?) before trusting it as a training signal is cheap
  insurance against exactly this kind of bug.
- **Attributes that look like they should do something in a framework
  sometimes don't.** A `reasoning` field set on action objects throughout
  this codebase turned out to be completely inert — never read, never
  serialized, never logged by anything. Worth confirming a debugging
  channel actually surfaces anywhere before relying on it.
- **CPU-bound data loading can fully hide a fast GPU.** A training run can
  look GPU-accelerated (a healthy device, a real speed-up measured on the
  model compute in isolation) while wall-clock time is actually dominated
  by single-process, single-threaded per-sample data preparation — the
  tell is a bloated single Python process burning CPU time roughly
  proportional to wall-clock time, rather than GPU utilization.
- **Persistent data-loader workers need to be explicitly torn down across
  a phase transition within one script**, or a second phase's newly
  spawned workers can contend with the first phase's still-alive ones in
  a way that hangs indefinitely rather than erroring — the process
  keeps running, doing nothing, which looks identical to "just running
  slowly" until process-level CPU usage is checked directly.
- **Results drawn from freshly regenerated random-policy data won't
  reproduce exact numbers run to run**, since each generation pass uses a
  different random seed — expect the same qualitative conclusion but not
  the identical percentage on a rerun; this is expected behavior
  throughout this project, not a bug to chase.
- **Two scores meant to be directly compared need the same statistical
  reduction, not just the same underlying quantity.** A per-candidate
  information-gain score computed as a maximum over many spatial
  locations for one candidate, and as a mean over the same locations for
  every other candidate, will structurally favor the maximum-based one
  almost regardless of the underlying signal — a maximum over many values
  is almost always larger than a mean over the same values. This kind of
  bug is easy to miss because each branch looks individually reasonable
  in isolation; it only shows up when the two are actually pitted against
  each other in a ranking.
- **A predicted-disagreement or predicted-novelty signal, recomputed
  fresh from a near-deterministic model each time, has no reason to decay
  just because an action has already been tried many times.** This
  produces the same "locks onto one action forever" failure mode whether
  the underlying signal is a raw predicted residual or raw inter-model
  disagreement — the fix in both cases observed in this project was the
  same: either track something based on *observed* outcomes (which
  naturally cools down once an action's real effect is known), or add an
  explicit random fallback so the loop can't fully close.
- **A confidence-accumulation update with no forgetting term will
  eventually become artificially certain, even when the underlying
  evidence doesn't actually support that certainty.** Summing evidence
  across an unbounded number of sequential observations, with a
  sensitivity parameter that's small relative to typical per-observation
  differences, compounds tiny or even spurious per-step differences into
  large, sticky confidence gaps within a surprisingly small number of
  steps. A geometric forgetting factor — weighting recent evidence more
  than old evidence — is a simple, standard fix, but the right amount of
  forgetting isn't obvious a priori and is worth sweeping empirically
  rather than guessing: too little forgetting reproduces the runaway
  problem, too much effectively discards the mechanism entirely.
- **A value or reward-prediction component trained on a very sparse
  positive-signal rate can end up only marginally better than a trivial
  "always predict the common case" baseline**, even after fixing an
  unrelated bug (like a mismatched input representation) that looked like
  it should have been the main blocker. Worth checking directly, since a
  component that's technically "trained" and "wired in correctly" can
  still be contributing close to noise to whatever it feeds into.
- **A component that looks like a total failure at first (zero successes
  across every test run) can sometimes be a cheap infrastructure problem
  in disguise, not a fault in the component's own logic.** An expired
  credential in one project's harness caused every game to be immediately
  unavailable, producing recording files that were technically valid but
  contained no real gameplay at all — worth ruling out an environment- or
  infrastructure-level explanation before assuming a fresh, complex
  design has failed outright.
- **A shared conditioning table (like an action-embedding lookup) needs
  every data source's value range checked against it explicitly, not
  assumed.** A new data source with one more category than an existing
  shared table's size produced no error at data-generation time — the
  out-of-range value just got stored as a plain integer — and only
  surfaced much later, as a confusing low-level crash deep inside GPU
  code, on whichever training batch first happened to sample it. A quick
  bounds check on newly generated data, before ever starting a training
  run, would have caught this immediately instead of after a much more
  confusing failure.
- **Adding a second instance of "the same kind of fix that worked before"
  is not guaranteed to produce the same result, and a second, differently
  -sourced batch of synthetic data can measurably hurt rather than help,
  even when the reasoning for trying it was sound.** The most likely
  explanation found here was specific to the new source's own mechanic
  (irreversible dead ends under random play degrading much of its data
  into low-information noise), not a flaw in the general strategy of
  adding diverse pretraining sources — a reminder that a strategy proven
  once needs to be re-verified each time it's applied again, not assumed.
- **A generic out-of-memory error during a long-running job can be a
  full-disk symptom on some platforms, not an actual memory shortage.**
  Confirmed directly: a training crash inside a data-loading worker
  process, in a repeating crash-and-respawn loop, cleared up completely
  after freeing real disk space, with zero changes to any code. Worth
  checking available disk space, not just available memory, when a
  memory-related error shows up unexpectedly mid-run.
- **A discounted-return target computed over episodes of very different
  lengths can silently dilute a training signal rather than enrich it,**
  even when the new data genuinely contains more of the rare events being
  targeted. A target that decays smoothly with distance from a reward
  event stays *technically* nonzero for a very long stretch before a much
  longer episode's reward, but becomes practically indistinguishable from
  the already-dominant "nothing happened" majority long before that — so
  simply mixing in longer, richer episodes without accounting for the
  loss's sensitivity to episode length can make an unweighted average
  worse, not better, purely as an artifact of how much low-information
  padding came along with the genuinely useful examples.
- **A component that got measurably better on the metric that reflects
  its actual job can still show a flat or even worse result in an
  end-to-end comparison, without that meaning the improvement was fake.**
  When the end-to-end outcome depends on a rare, effectively binary event
  (win or no win, within a small handful of attempts), the outcome-level
  measurement simply doesn't have the resolution to reliably detect a
  moderate upstream improvement — the right response is to either gather
  substantially more trials or measure something with finer resolution
  than the binary outcome itself, not to conclude the improvement wasn't
  real.
- **A comparison or analysis utility that reads every matching file
  before filtering by what was actually asked for pays the full cost of
  the largest, least-relevant files too.** A utility that had run
  instantly against a corpus of small files became unexpectedly slow once
  much larger files (unrelated to what was being compared) were added to
  the same directory — worth filtering by filename pattern *before*
  opening and parsing file contents whenever files of very different
  sizes might coexist in the same location.
- **Live, step-by-step tracing of real decisions can surface a bug that
  purely offline, aggregate evaluation completely hides.** A pattern
  invisible in summary statistics — one specific action type losing a
  ranking comparison in nearly every single decision across an entire
  episode — was obvious within seconds of watching the actual per-step
  values scroll by. Worth reaching for direct, granular tracing earlier
  when an aggregate result is flat or poor and the cause isn't obvious,
  rather than only ever looking at final summary numbers.
- **A plausible-sounding fix can be directly, cheaply falsified by
  testing it in isolation before committing to it.** Re-scoring one
  option at what seemed like a more representative input, instead of a
  generic default one, was a reasonable-sounding hypothesis — and
  produced no measurable change at all once actually tested, which
  itself was the useful signal: it revealed that the underlying
  mechanism didn't work the way it was assumed to (a supposedly
  location-specific input was actually being applied as a uniform,
  non-specific shift everywhere at once). Testing a fix's actual effect
  before keeping it, even when the fix seems obviously correct, catches
  wrong mental models of how a component works, not just wrong code.
- **When two things being compared aren't naturally on the same footing
  (one option's value depends on choosing the single best spot out of
  many, another's doesn't depend on location at all), neither of the two
  "obvious" ways of summarizing across those many spots — plain averaging
  or taking the single highest value — treats them fairly, for two
  different and opposite reasons.** Averaging washes out a genuinely
  strong single spot; taking the maximum inflates a result purely from
  having more chances to sample a high value, independent of whether
  anything meaningful is actually there. A middle-ground summary
  (averaging over just the strongest few, not one or all) can resolve
  this kind of asymmetry when the two "obvious" extremes each fail in
  opposite directions.
- **Isolating each ingredient of a combined design, one at a time, is the
  most direct way to check whether that combination is actually earning
  its complexity — and the answer isn't always assumable in advance.**
  Testing each half of a blended decision rule alone, against the same
  matched evaluation used for the combined version, gave a clean,
  unambiguous answer (the combination clearly beat either half) rather
  than leaving it as an assumption. This is a general, reusable technique
  for validating any design that blends multiple signals: force each
  input to an extreme, one at a time, and compare against the blend.
- **"Improved against its own prior baseline" and "now beats a specific
  competing approach" are two different claims, and confirming the first
  is not evidence for the second when the competing approach's own
  results move around from run to run.** A before/after comparison run
  only against one thing's own earlier numbers, without a freshly
  re-measured comparison point from the same round, can look like a
  milestone was reached when the real, most relevant comparison hasn't
  actually been run yet. Worth explicitly re-checking against a fresh,
  same-round baseline before declaring a comparative milestone met, not
  just a self-comparison.
- **A short-lived credential can expire mid-session, not just between
  sessions, and the failure mode can look exactly like a real regression
  rather than an infrastructure problem.** An authentication token used
  for this project's harness turned out to expire within roughly a day —
  short enough to lapse in the middle of active work, not just when
  picking the project back up later. Every symptom (an empty result set,
  every agent scoring zero across the board) looked identical to a real
  behavioral regression until checked directly. Worth ruling out an
  expired credential first, specifically, whenever a comparison
  unexpectedly comes back completely empty rather than just different.
- **A deterministic "pick the single best option" rule will default to
  the exact same choice every time its input is flat or nearly flat, not
  behave like a neutral non-preference.** A selection mechanism that
  looked reasonable in isolation produced a strong, consistent bias
  toward one specific repeated choice in practice, because ties (or
  near-ties, which may be the common case rather than the exception) resolve
  to the same low-indexed option every single time instead of exploring
  among them. Worth checking whether an existing, already-debugged
  component elsewhere in the same project already solved the identical
  problem (a softened, weighted-random choice instead of a hard maximum)
  before re-deriving a fresh solution that quietly reintroduces it.
- **When two things being compared diverge, tracing the specific cases
  where they diverge is far more informative than re-examining the whole
  population again.** A trace across an entire broad sweep missed a real,
  fully explanatory bug that became obvious within moments once narrowed
  to just the handful of cases where the two things being compared
  actually disagreed — the bug's effect was real in the pooled results
  all along, just diluted into near-invisibility by everything else in
  the average.

---

## 9. Kaggle competition submission: a real scored entry obtained

With Stage 5's milestone met, the agent was submitted to the actual
competition this project targets. The submission mechanism itself turned
out to be its own debugging project, worth documenting in full since the
failure mode gave almost no useful information back and the eventual
cause was easy to miss.

**The symptom:** every real scored submission attempt came back with the
platform's own generic system-error message and no further detail, no
matter what changed between attempts — a hardened, more defensive setup
script; a timing fix aimed at a documented "must make a first move within
roughly fifteen minutes" constraint; toggling hardware acceleration on;
wrapping the agent's per-turn decision logic in a safety net that falls
back to a random legal move on any unexpected exception rather than
letting the whole run die; even writing a placeholder result file before
any risky setup ran, in case setup itself was the failure point. None of
it changed the outcome. That pattern — several genuinely different fixes
all landing on the identical failure — echoes the exact lesson from
Stage 1's action-recording bug and Stage 4's Sokoban ablation: it's a
sign to go looking for one shared upstream cause, not to keep iterating
on downstream guesses.

**The turning point** was running a control: submitting the platform's
own unmodified example submission, byte-for-byte, as a completely
separate entry. It succeeded, with a real score. That single result
reframed everything — the platform, the account, and the basic submission
mechanism all demonstrably worked; whatever was failing had to be
specific to this project's own code or setup, not the environment around
it.

**The actual cause, once found, was almost embarrassingly simple: an
attached data source doesn't mount where its name would suggest.** The
platform nests an attached personal dataset under an extra directory
layer beyond what the dataset's own name implies — a convention that had
already been followed correctly, by coincidence, for the competition's
own attached files (copied straight from the working example), but never
applied to this project's own attached checkpoint-and-code bundle. Every
real scored run was failing on the very first file-copy operation, before
the agent ever played a single move — which is exactly why none of the
other fixes ever mattered: they were all downstream of a line that never
successfully ran.

Finding this required a specific technique, since the platform gives no
access to a real scored run's actual execution log: a free, non-scored
test push with a diagnostic step placed *outside* the normal
scored-run-only code path, so it runs during an ordinary harmless test
rather than only during a real (and quota-limited) submission. That
diagnostic printed the runtime's library versions, walked the attached
data's actual mount location, and attempted to load each checkpoint file
directly — surfacing both the wrong path and, as a secondary finding,
confirming that a library-version difference between where the
checkpoints were produced and where they were being loaded (a real,
independently-worth-checking suspicion at the time) was not actually a
problem at all. Once the path was corrected, the very next real
submission succeeded outright.

**Final result:** a real, completed, scored submission using the Stage 5
hypothesis-bundle agent, meaningfully outscoring the platform's own
unmodified random-policy example on the same scoring run. All of the
defensive changes made while chasing the wrong hypotheses were kept
regardless — a decision-logic safety net, a setup-time safety net, and a
same-purpose diagnostic toggle are all genuinely good practice
independent of this specific bug, even though none of them were the fix.

**The general lesson, worth carrying into any future platform-integration
work:** when an error message carries zero diagnostic content, don't
keep guessing against the expensive, quota-limited, slow feedback loop.
Find or construct a cheap, fast, repeatable one instead — here, a free
test push with a probe planted outside the normally-gated code path —
and use *that* to actually observe what's happening before spending
another real attempt.

## 10. Current status (see `CLAUDE.md` for the live version of this section)

- Stage 0: done.
- Stage 1: milestone passed (+29.2% changed-patches over identity,
  post-bugfix).
- Stage 2: built, milestone reasonably met (directed exploration reaches
  more distinct games than random, even though raw level-completion count
  alone doesn't show a clean margin at this sample size).
- Stage 3: built, both components (recurrent predictor, exact transition
  graph) pass their own checks; +21.3% changed-patches for the recurrent
  predictor, and the combined memory agent reaches the most distinct
  games of any agent tested so far.
- Stage 4: built, milestone mostly met — +44.1% changed-patches (beats
  both single-predictor variants), with real but not yet dominant gate
  specialization. Noisy top-k gating was tried as a further push on
  specialization and documented as a negative result; the dense-gate,
  MiniGrid-pretrained checkpoint is the one used going forward. A later
  attempt to add a second synthetic pretraining source (a push-mechanic
  environment family, alongside the existing navigation-focused one) was
  a clean, controlled negative result — changed-patches improvement fell
  from +29.5% to +15.7% on an otherwise-identical comparison, and gate
  specialization didn't improve either — most likely because that
  source's random-policy data is prone to irreversible dead ends that
  degrade much of it into low-information noise; not pursued further
  per the explicit scope this was tried under, and the working checkpoint
  was left untouched throughout.
- Stage 5: hypothesis-bundle agent designed, built, and evaluated. Four
  real bugs were found and fixed along the way (a value-head/encoder
  latent-space mismatch, an apples-to-oranges information-gain comparison
  that made one action structurally dominate, a missing random-exploration
  fallback that let the agent lock onto a single action for a whole
  episode, and an unbounded confidence-accumulation update that produced
  runaway certainty). After all four fixes, the milestone (beating the
  curiosity agent on the same levels in fewer actions) is not cleanly
  met: total levels and distinct games reached both remain behind the
  curiosity agent's numbers on a matched comparison, though the one
  same-game head-to-head data point available shows the newer agent
  succeeding much faster when it does succeed, just far less reliably.
  The most likely remaining causes are both data-bound limitations
  already documented in earlier stages (a value head trained on an
  extremely sparse positive-signal rate, and a mixture-of-experts model
  whose specialization was already found to be a minority behavior) —
  the natural next lever is better or more purposeful training data, not
  further parameter tuning on the current setup. A follow-up attempt to
  address the sparse-signal cause directly — reusing an existing
  exploration agent as a data-collection "teacher" with a much larger
  action budget — produced a real, verified improvement in the value-
  estimating component itself (roughly a third lower error than a trivial
  baseline on the examples that actually matter, after also fixing a
  genuine unweighted-loss dilution bug along the way), but whether that
  translates into a better full agent remains an open question: a repeat
  of the same end-to-end comparison came back essentially flat, which is
  itself expected and uninformative at this sample size given how rare
  and binary the outcome being measured is, not evidence the improvement
  wasn't real. A subsequent, more direct investigation — live-tracing
  real decisions step by step rather than only looking at outcome-level
  summaries — found and fixed a genuine architectural bug: the
  click-based action was being scored in a way that structurally
  undervalued it relative to other actions, unrelated to any training
  data question. Fixing it produced a clean, unambiguous improvement
  against the agent's own prior broken state (from zero level completions
  across a full matched evaluation to six completions across three
  distinct games) — the first fix in this whole line of investigation
  with directly measured, noise-proof evidence behind it. A follow-up
  check confirmed the underlying design decision to blend two different
  decision-making signals together, rather than relying on either alone,
  was itself sound: isolating each signal individually performed clearly
  worse than the combination. However, a properly matched, same-round
  re-comparison against the exploration-only agent immediately afterward
  showed the milestone was still not met at that point — the exploration-
  only agent reached five distinct games that round against the
  hypothesis-driven agent's one, and led on total completions too.

  A final round of investigation deliberately narrowed to just the
  handful of games where the two agents' results actually diverged,
  rather than continuing to re-run the full sweep. This surfaced a
  second real bug, invisible in the broader trace: on the games where
  only one type of action was ever available (so the entire outcome
  depended purely on click placement), a deterministic "pick the single
  best-looking option" rule was defaulting to the same repeated click
  location whenever its input was flat or near-flat, rather than
  exploring — the same problem an earlier stage of this project had
  already solved for its own click logic, but which hadn't been carried
  over when this later agent's click-selection was written fresh. Fixed
  by reusing that already-proven approach directly. A final matched,
  same-round comparison confirmed the fix worked: the hypothesis-driven
  agent reached more total completions, more distinct games, and did so
  in fewer actions on average than the exploration-only agent, all three
  at once — the milestone this whole stage was built around is now met.
