# Stage 7 — Strategy reset: gap analysis and prioritized plan

**Status:** proposed, awaiting human sign-off. No compute spent, no submission
quota spent, nothing merged to `master`.
**Date:** 2026-09-07. **Deadline:** 2026-11-02 (~56 days ⇒ ~56 submissions max).
**Goal set by the human:** finish in the **top 10%** of the leaderboard.

---

## 0. The one number that reframes everything

Pulled live from the Kaggle API today, not from memory:

| | score | rank |
|---|---|---|
| #1 Tufa Labs | 11.04 | 1 |
| #10 | 4.99 | 10 |
| #20 | 4.32 | 20 |
| **Top 10% cutoff** | **2.99** | **287 / 2,870** |
| **Our team "How bad can it go?"** | **1.77** | **531** |
| Median | 0.28 | 1,435 |

**The target is 2.99.** Two facts about our 1.77 matter more than the number itself:

1. **It is not this project's work.** It came from submission `55769792`, kernel
   `calamitychasm/lb-9-arc3-duck-v12-with-qwen-3-8-27b` — verified by cell-by-cell
   diff to be a **byte-identical copy** of the public notebook
   `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b` (11/11 cells identical;
   metadata differs only in `id`, `id_no`, `is_private`). Zero code changes,
   blank description.
2. **Everything this repo actually built scores 0.00–0.25.** Best own-work results:
   `GraphExplorerAgent` 0.25, `Hypothesis` 0.23, `GraphExplorerLearnedAgent` 0.18.
   At 0.25 we would rank ~1,600th.

So the honest framing is not "we are 1.7x from the goal." It is: **our own agents are
~12x from the goal, and our current rank is borrowed.**

---

## 1. Why our agent class cannot reach 2.99 — mechanism, not defeatism

### 1.1 The metric structurally punishes what our agents do

The scoring metric is **RHAE** (Relative Human Action Efficiency), from the official
ARC Prize technical report (arXiv:2603.24621):

```
per level:        S_l = min(1.15, h_l / a_l)^2         h = upper-median best-human action count
per environment:  E_e = min( Σ_solved w_l / Σ_all w_n ,  Σ w_l·S_l / Σ w_l ),   w_l = l
final:            T   = mean over environments, reported as a PERCENTAGE (0–100)
```

Two consequences we had never accounted for:

- **Efficiency is squared.** 2× the human action count ⇒ ¼ score. 10× ⇒ ~1%.
  `GraphExplorerAgent`'s coverage-first sweep spends thousands of actions to clear a
  level a human clears in tens. It is *structurally* scored near zero **even on the
  levels it wins.** This finally explains the paradox CLAUDE.md documents at length:
  46 local level-completions collapsing to a real score of 0.10–0.25.
- **Completion is a hard ceiling.** The left term caps a game's score at the
  level-index-weighted fraction of levels actually finished. Depth across many games
  dominates; scraping level 1 on a few games is worth almost nothing.

This is a **new, mechanistic explanation** for the local-to-hidden-game gap that
CLAUDE.md spent 13+ interventions chasing. The gap was never purely generalization —
a large part of it is that our local backtest metric (levels completed) is not the
competition metric (squared efficiency, weighted by depth). We were optimizing the
wrong objective.

### 1.2 The leaderboard is an LLM monoculture

Survey of public notebooks (verified by pulling source):

| Approach | Best public score |
|---|---|
| LLM harness (Tufa "Duck" family) | **4.32 – 11.04** — ~40 of the top 50 notebooks are forks |
| Best **non-LLM** public notebook (`nihilisticneuralnet`, beam/IDA*/MCTS + cloned simulator + CNN) | **0.46** |
| Our best own work | **0.25** |

The best non-LLM notebook in the entire public field is far more elaborate than
anything we built — and it still only reaches 0.46, well below the 2.99 bar.
**No non-LLM approach anywhere in the public field is competitive at the rank we need.**

### 1.3 Verdict on the existing tracks

**Stop JEPA world-model generalization work, and stop coverage-first graph exploration
as a scoring strategy.** Not because the work was bad — the debugging in CLAUDE.md is
genuinely rigorous — but because (a) the metric penalizes their action profile
quadratically, and (b) the public field demonstrates the whole class caps out ~6x
below target. This supersedes CLAUDE.md's open "should we keep attacking the held-out
gap" question with a clear answer: no.

---

## 2. What the Duck harness actually is

Verified by pulling the source (public dataset `jeroencottaar/taaf-kaggle-source-share`;
also open-sourced at `github.com/Tufalabs/duck-harness`):

A **27B LLM served by vLLM inside the Kaggle kernel** on an RTX PRO 6000 (96 GB,
`machine_shape: NvidiaRtxPro6000`), driving the game through a **Python REPL tool**:

1. Observations are injected into a Python sandbox **as variables** (`current_frame.segmentation`
   with connected components/containment/adjacency, `transitions`, `last_action_result`,
   `valid_actions`; the raw numeric grid is deliberately hidden).
2. The model writes Python snippets to inspect them and calls `action(...)` from inside Python.
3. Oldest messages are evicted, so it plays indefinitely in a 32K window.
4. It also **sees the grid as an image** (multimodal, 4× upscale).
5. ~110 games at 28-way concurrency inside a 9-hour cap.

**Tufa's own stated lesson:** the gains came from *multimodality and better base models*;
**hand-crafted tools actively hurt**. The 1.17 → 9 jump across forks was achieved with
prompts, loop, and policy **unchanged** — only the base model version and inference
throughput changed. At 28-way concurrency over 110 games in 9 h, **decode throughput is
the binding constraint**, which is why every top fork is an inference-optimization fork.

---

## 3. The crux: why did our byte-identical copy score 1.77?

The original notebook's title claims **LB 9**. We ran it unmodified and got **1.77**.
That 5x gap is the single highest-value unknown in this analysis, and diagnosing it
costs **zero submission quota**. Candidate causes, ranked, each with a concrete check:

| # | Hypothesis | How to check (all free) |
|---|---|---|
| 1 | **Model mount inaccessible to our account.** `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` returned **403** to our credentials on the dataset endpoint during research. If the weights never mounted, vLLM fails or falls back. | Attempt the mount in a free `Save & Run All` push; read the log for the model path and vLLM boot. |
| 2 | **RTX PRO 6000 not actually provisioned.** Tufa's notebook explicitly warns the GPU must be *manually selected* on a fork; requesting it in metadata is not sufficient. On a weaker/absent GPU, throughput collapses. | Free push; print `nvidia-smi` and the resolved machine shape unconditionally. |
| 3 | **vLLM boot failure on Blackwell sm_120** (known FlashInfer sampler JIT crash; fix `VLLM_USE_FLASHINFER_SAMPLER=0`). | Free push; read boot log. |
| 4 | **The "LB 9" claim is inflated.** Per-notebook scores are self-reported in titles; the API returned `bestPublicScore = None` for every kernel surveyed. | Check whether any *fork author* (not Tufa) is verifiably in the 4–9 band on the leaderboard. |
| 5 | **Run-to-run variance.** Real, but implausible as the whole story: Tufa's own dev-set spread is ±0.45, not ±7. | n=2 at the same config, only if 1–4 come back clean. |

**Hypotheses 1–3 are all silent-failure modes with exactly the signature CLAUDE.md has
been burned by before** (a run that completes, produces a submission, and scores near
the floor). They are also all free to test. This is step 1 of the plan.

---

## 4. The other asset: `CodeWorldAgent` is aimed at the right target

Undocumented in CLAUDE.md, and **it existed nowhere on this machine or in git** — the
only copies were on Kaggle. It is now rescued into
`kaggle_submission_llm_world_engine/` (commit on this branch, preservation only).

What it does (1,433 lines, genuinely our own work): opening probes build a transcript →
**Qwen3-Coder-30B-A3B writes the Python source of a `WorldModel` class** → validated by
replaying the transcript → each step chosen by an **LLM-free beam search** (depth 2,
beam 4) over that model → the coder repairs the source when predictions diverge →
Gemma-3-12B consulted only on planner stalls.

**This is precisely the shape of published SOTA.** "Executable World Models for ARC-AGI-3"
(arXiv:2605.05138) reports **58.12% mean RHAE** with GPT-5.5 by having a coding agent
write an executable model of the game and plan against it; OPINE-World (arXiv:2607.01531)
solves 20/25 games the same way. The design is well-aimed; the implementation is broken.

**Why it scored 0.00 — concrete evidence, not speculation** (from the retrievable
`llm-world-engine-test-run` output):

1. **The persisted world models are the untouched template stub** — `predict()` returns
   `state, 0, False` ("nothing ever changes"), `goal_hint()` returns `0.0`. The beam
   search had *literally zero signal*. This alone fully explains 0.00.
2. `action_head` failed on **every** call: `UndefinedError: 'str object' has no attribute
   'text'` (Gemma chat-template). *Fixed* in the version that actually shipped.
3. `repair failed after 3 attempts`, repeatedly — every transition diverged.
4. **Unhandled `TypeError: object of type 'int' has no len()`** at `llm_engine/diff.py:18`
   killed an entire game thread. `code_world_agent.py` contains **zero `try`/`except`** —
   none of the heartbeat hardening `hypothesis_agent.py` already has.
5. **Throughput: 0.27 fps on `ar25`, 0.02 fps on `bp35`** — both 0 levels. Against ~110
   games in 9 hours, coverage would be negligible even with a working model.

Items 1, 3, 4, 5 all shipped into the scored run unchanged (only `llm_client.py` was
patched, 22 minutes before submission).

---

## 5. The decision the human needs to make

Two routes reach 2.99. They are not mutually exclusive, but they mean different things.

**Route A — run a properly-configured Duck fork.**
Permitted: `rules.md` states public sharing on Kaggle is treated as open-sourced under
an OSI license, and forking is normal Kaggle practice. Expected outcome if hypotheses
1–3 in §3 are the cause: 4–9, comfortably past 2.99. Effort: days, mostly operational.
**But it is running someone else's system, and it is not this project's contribution.**

**Route B — fix and ship `CodeWorldAgent`.**
Genuinely our own, aimed at the published SOTA mechanism, higher ceiling. But its
current state is 0.00 with a first-order throughput problem, and getting it to 2.99 is
a multi-week engineering program with real risk of not landing before 2026-11-02.

**Recommendation: run both, in this order** — A to establish a rank floor and to learn
the operational path (GPU selection, model mounts, vLLM boot) that **Route B needs
anyway**, and B as the real deliverable. The §3 diagnostics serve both routes: Route B
also needs a working large-model serving path in-kernel, and it currently loads models
via `transformers` rather than vLLM, which is very likely a major cause of its
0.02–0.27 fps.

**What I will not decide unilaterally:** whether "top 10%" means *team rank* (Route A
satisfies it) or *our agent in the top 10%* (only Route B does). That changes
everything downstream and is the human's call.

---

## 6. Prioritized plan

### Phase 0 — free, immediate, no quota (this week)

| # | Task | Tier | Done =|
|---|---|---|---|
| 0.1 | ✅ **Rescue `CodeWorldAgent` into git** — done, commit on this branch | — | committed |
| 0.2 | **Diagnose the 1.77 gap** (§3 hypotheses 1–4) via free `Save & Run All` pushes with unconditional diagnostics | capable | a written verdict naming the cause, or ruling out 1–4 |
| 0.3 | **Correct CLAUDE.md's submission record** — it documents 9 of at least 20; the 2026-09-06 gateway fix is recorded as "not yet submitted" when it **scored 0.18** | orchestrator | table matches the API |
| 0.4 | **Repo hygiene** — commit the uncommitted `retry-max-time=600` gateway fix (currently a 1-line diff existing in no commit), push `master` (3 commits ahead), reclaim 11.7 GB from 8 stale worktrees (C: at 98%, 22.6 GB free) | cheap | clean tree, `master` pushed, >30 GB free |
| 0.5 | **Unblock GitHub workflow** — `gh` is not installed and there is no token, so PRs/Issues per the operating model are currently impossible | human | decision below |

### Phase 1 — establish a floor (gated on 0.2)

- If §3 finds a fixable config cause: fix it, validate on a free push, then **one**
  scored submission. Expected 4–9; clears the 2.99 target.
- If §3 finds the "LB 9" claim was inflated: re-plan Route A around a verifiable fork
  before spending quota.

### Phase 2 — make `CodeWorldAgent` real (the actual deliverable)

Ordered by what the evidence says is binding:

1. **Throughput first.** 0.02–0.27 fps is fatal regardless of model quality. Move
   serving from `transformers` to **vLLM** (the Duck harness proves this path works
   in-kernel and we will have just debugged it in Phase 1).
2. **Fix drafting.** World models persisting as untouched stubs is the direct cause of
   0.00. Needs a local reproduction and a test asserting a drafted model is *not* the
   template.
3. **Harden.** Add the heartbeat `try`/`except` pattern `hypothesis_agent.py` already
   uses; fix `diff.py:18`.
4. **Test the data path.** Per the mission's testing discipline and this repo's history
   of silent data-corruption bugs — note the project currently has **no unit tests of
   its own code at all**, only the vendored framework's.
5. Only then consider borrowing from Duck: segmentation-as-variables, multimodal grid
   view, message eviction. Tufa's finding that *hand-crafted tools hurt* is a real prior
   against porting our `TransitionGraph`/`ClickEffectModel` into the loop.

### Explicitly not doing

- Further JEPA held-out-generalization work (§1.3).
- Any change that raises action counts to raise completions — the square term eats it.
- Trusting n≤8 local backtests. And note: **our local metric must change to RHAE**, or
  local results will keep failing to predict real scores exactly as they have all along.

---

## 7. Open risks

- **Route A is not our work.** If the goal is a portfolio piece, a fork raises rank but
  proves nothing. Flagged for the human, not resolved here.
- **`CodeWorldAgent` has no local runtime.** Both models are Kaggle Model mounts; the
  local box is an RTX 2070 (8 GB) and cannot host a 30B model. Iteration will be
  slow (free pushes only) unless a smaller local proxy model is used for logic testing.
- **Metric variance is large** (Tufa's own dev-set spread ±0.45 at n=20). One submission
  will not distinguish a real 0.5-point gain from noise. Budget accordingly against
  1/day and ~56 days.
- **Unverified:** the RHAE formula comes from the ARC Prize technical report; the Kaggle
  Evaluation page is a JS SPA and could not be read directly. High confidence, not
  confirmed at source.

---

## 8. What I need from the human before starting

1. **Route A, B, or both?** (§5) — specifically, does "top 10%" mean team rank or our
   own agent?
2. **GitHub tooling** (0.5): install `gh`, or provide a PAT, or accept that PRs/Issues
   are tracked as branch-local markdown until then.
3. **Confirm Phase 0 may proceed** — all of it is free and reversible; only Phase 1
   spends quota.
