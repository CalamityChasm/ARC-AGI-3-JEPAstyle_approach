# Stage 7 — SOTA research: the paper track, Tufa, the public field, and the biggest lever we own

Date: 2026-09-15. Branch: `stage7-sota-research`.

Labels follow this repo's convention. **[VERIFIED]** = read directly from a
primary artifact (one of our own run artifacts, a Kaggle API response, bundle or
notebook source, or a locally-run computation over them). **[INFERRED]** =
reasoning over verified facts, not directly observed. Secondary sources (papers,
blogs) are cited and labelled as such — they are other people's measurements,
not ours.

**No competition submission was made by this investigation, and none will be.**
Everything here is read-only analysis of already-completed runs, read-only
Kaggle API calls, and public source. The daily slot was untouched.

Reproduce the two new measurements with:

```
venv/Scripts/python.exe scripts/analyze_tokens_per_action.py \
    fp8=<dir> nvfp4=<dir> ctx16k=<dir> dedupe=<dir> anim=<dir> \
    --json experiments/stage7_tokens_per_action.json
venv/Scripts/python.exe scripts/analyze_rhae_binding.py <nvfp4-baseline-dir> \
    --json experiments/stage7_rhae_binding.json
```

`<dir>` is an unpacked `kaggle kernels output` directory for, respectively,
`lb-9-arc3-duck-v12-with-qwen-3-8-27b`, `arc3-duck-nvfp4-baseline`,
`arc3-duck-nvfp4-ctx16k`, `arc3-duck-nvfp4-dedupe`, `arc3-duck-nvfp4-anim`.

---

## TL;DR — the five things that changed the picture

1. **`tokens_per_action` is explained, and it is not a token problem.** 36–54%
   of every LLM call we pay for takes **no game action at all**. Every one of
   them stops with the same message — `Yielded control to solver:
   turn_time_budget` — because `LOCAL_ANALYZER_YIELD_SECONDS = 60` is smaller
   than our median ~153 s LLM round-trip. The analyzer therefore gets *exactly
   one* model call per turn, and if that call was an inspect-only `python` call,
   the turn produces nothing. [VERIFIED, §1]

2. **The system prompt actively causes this.** It tells the model *"You can call
   the `python` tool as many times as you want per step"* and *"Do not ration
   tool calls… Spend extra tool calls to confirm what changed."* Both statements
   are **false on our serving stack**, where the budget permits one call.
   [VERIFIED, §1.4]

3. **Completion, not efficiency, is our binding term — by about 10:1.** On the
   nvfp4 baseline, completion binds in **16 of 25** games and efficiency in 9.
   The entire efficiency cap costs **1.25 points of an 11.94 completion-only
   ceiling** (reported 10.69). Chasing action efficiency cannot get us to the
   top-10% bar; solving more levels can. [VERIFIED, §2]

4. **The team at 3.74 has already built, and is right now running, a controlled
   lever program against exactly these pathologies** — an *action floor* for the
   no-op turns, a *world-model wipe guard*, a *death blacklist* that infers
   hidden per-level action budgets, and a *restart-at-stall*. Each ships with a
   matched `-ctl` control arm. All are ~15–60 lines in the notebook's
   customization cell. Source pulled and quoted. [VERIFIED, §4]

5. **Tufa's 18.81 is not publicly documented anywhere.** Everything they have
   published — GitHub, the research page, the ARC Prize milestone write-up —
   describes the milestone-1 Duck that scores ~1.2–1.6 on the public 25. The gap
   is 138 submissions of private iteration, not a published technique we are
   failing to copy. [VERIFIED by absence, §3]

The anim-solver graft's free run **completed while this was being written**:
public-25 mean **9.97** on **2,615 actions**, against the baseline's 10.69 /
3,633. It has already been submitted by the orchestrator (ref `56264462`,
pending). §5.1 reads that result.

---

## 1. Task 5 — the unexplained internal lever, explained

### 1.1 The decomposition

`stage7_model_search.md` §2 established the identity
`turns = agg_gen_tok_s × T / (tokens_per_action × N_games)` and showed
`tokens_per_action` spanning 515 → 1,782, "larger than any throughput factor any
lever in this whole investigation has moved". It did not say why.

`benchmark.json` records `generated_tokens` for **every individual action**, and
most of them are **zero** — one LLM call can emit a Python program that issues
several game actions, and only the first carries the cost. So the quantity
factors. `scripts/analyze_tokens_per_action.py` over five completed runs
[VERIFIED]:

| run | public-25 | actions | costed turns | tok/act | tok/turn | act/turn | free actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp8 | 3.37 | 1,629 | 559 | 1,782 | 5,192 | 2.91 | 65.7% |
| **nvfp4 (ours)** | **10.69** | **3,633** | **695** | **515** | **2,692** | **5.23** | **80.9%** |
| ctx16k | 2.01 | 3,435 | 921 | 632 | 2,355 | 3.73 | 73.2% |
| dedupe | 7.09 | 1,885 | 371 | 959 | 4,875 | 5.08 | 80.3% |
| anim | 9.97 | 2,615 | 586 | 670 | 2,989 | 4.46 | 77.6% |

The 3.46× spread in `tok/act` is **1.93× in tokens-per-turn × 1.80× in
actions-per-turn**. Neither factor dominates; both move.

### 1.2 The real master identity

Cross-checking `benchmark.json` against the server's own counters closes it
properly. For the nvfp4 baseline [VERIFIED, `vllm-metrics-final.prom`]:
`generation_tokens_total = 1,919,110` over
`request_generation_tokens_count = 1,339` requests — and the transcripts contain
**1,312 `[ANALYZER STATUS]` blocks**. That is **1.02 LLM requests per analyzer
turn**: there is no retry storm and no multi-request repair loop. One turn, one
call.

So the quantity that actually governs play is:

```
actions_per_game = (LLM calls per game) × (actions per LLM call)
```

| run | calls/game | actions/call | = actions/game | P(call acts) | actions per acting call |
|---|---:|---:|---:|---:|---:|
| **nvfp4** | **53.6** | **2.71** | **145.3** | **54.0%** | **5.23** |
| anim | 54.3 | 1.93 | 104.6 | 63.6% | 4.46 |
| ctx16k | 87.6 | 1.57 | 137.5 | 45.3% | 3.73 |
| dedupe | 32.8 | 2.30 | 75.4 | 48.4% | 5.08 |
| fp8 | — | — | 65.2 | 46.7% | 2.91 |

Calls per game is set by wall clock ÷ latency: 7,920 s ÷ ~153 s ≈ 52.
`stage7_analyzer_timeouts.md` measured the same thing from the other side
("~52 analyzer turns per game, median 153 s per turn, all 25 games hit the
wall").

### 1.3 Where the 46% goes — one message, every time

Counting `step_executed:` in every transcript [VERIFIED]:

| run | `yield_seconds` | analyzer turns | executed | **no-op** | no-op share |
|---|---:|---:|---:|---:|---:|
| **nvfp4 (ours)** | **60.0** | 1,287 | 695 | **592** | **46.0%** |
| anim | 180.0 | 922 | 586 | 336 | 36.4% |
| fp8 | 60.0 | 1,198 | 559 | 639 | 53.4% |
| dedupe | 60.0 | 766 | 371 | 395 | 51.6% |
| ctx16k | 60.0 | 2,032 | 921 | 1,111 | 54.7% |

**Every single no-op turn in all five runs carries the same stop reason:**

```
step_executed: False
message: Yielded control to solver: turn_time_budget.
```

Not a traceback, not a malformed tool call, not a timeout. Across 25 transcripts
there are only 52 `Traceback` occurrences and 25 read timeouts (the latter
already root-caused as the per-game deadline in `stage7_analyzer_timeouts.md`).

The mechanism, read out of the bundle
(`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`,
`src/ARC3-Inference/inference/agent/tool_agent.py:1770-1790, 1945-1978`)
[VERIFIED]:

```python
def control_yield_reason() -> str | None:
    ...
    if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:
        return "turn_time_budget"
    return None

while self._tool_steps is None or turn_count < self._tool_steps:
    yielded_control_reason = control_yield_reason()
    if yielded_control_reason is not None:
        break
    ...
    dispatch = self._dispatch_tool(state_path, tool_name, arguments)
    if dispatch.step_executed:
        break                      # acted -> turn done
    yielded_control_reason = control_yield_reason()
    if yielded_control_reason is not None:
        break                      # 153s elapsed >= 60s budget -> turn dead
```

The loop is allowed up to `LOCAL_ANALYZER_TOOL_STEPS = 12` tool calls. It never
gets a second one, because the first response arrives at t≈153 s against a 60 s
budget. **The 12-step allowance is dead code on our stack.**

This also explains the anim run's 1.44 requests per turn: `yield_seconds = 180`
permits a second call (153 < 180), and a third is cut off (306 > 180).

### 1.4 The prompt is telling the model something false

`inference/agent/prompts.py: COMPACT_TOOL_SESSION_ADDENDUM` (lines 103–114)
[VERIFIED]:

> - You can call the `python` tool as many times as you want per step.
>   Investigate until your code has a clear probe or plan.
> - **Do not ration tool calls when the state is unclear. Spend extra tool calls**
>   to confirm what changed between frames and whether the last action affected
>   gameplay state or only HUD elements such as countdown bars.

Both lines describe a regime where tool calls are cheap. On our stack a tool
call costs ~153 s out of a 7,920 s per-game budget, i.e. about **1/52nd of the
entire game**, and the model is explicitly told not to ration them. The
resulting behaviour — inspect first, act next turn — is exactly what the prompt
asks for, and it halves our effective play.

This is not a bug in the Duck. It is a prompt written for Tufa's own latency
regime, running unmodified on a much slower one.

### 1.5 What this is worth

At 54 calls/game and 54% acting, we get ~29 productive calls per game. Removing
the no-ops entirely would give ~54 — an upper bound of **+85% productive model
turns at zero hardware cost**. [INFERRED — upper bound, not a forecast: some
inspection is genuinely load-bearing, and §5.2 explains why the one run at
yield 180 did not realise it.]

For calibration against everything Stage 7 has already mined: the best measured
serving win in this project's history is **+16.5%** (`stage7_duck_throughput.md`),
and concurrency 28→37 was **+8.0%**. This lever is a different order of
magnitude, and it is upstream of all of them.

---

## 2. What the score is actually made of — completion vs efficiency

The task brief frames efficiency as central ("RHAE squares it"). Measured on our
own baseline, that is **not** where our score is lost.

`scripts/analyze_rhae_binding.py` recovers per-level action counts from
`artifacts/*_events.jsonl` (every action is stamped with its `level`) and
per-level human baselines from `benchmark.json`'s `base_actions_per_level`.
Those baselines total **17,135 actions over the 183 levels of the 25 public
games** — **exactly** the human baseline quoted in arXiv:2607.15439, which
confirms they are the official `h_l`, not a proxy. [VERIFIED]

`term1 = Σ_solved w_l / Σ_all w_l` is exactly computable; `reported_E` is ground
truth from `score.json`. Comparing them decides the binding term without needing
to reconstruct `term2`:

```
levels solved: 48 / 183 (26.2%)
actions on the never-completed level: 2,557 / 4,970 (51.4%)

COMPLETION binds (reported == term1): 16 / 25
  ar25 cd82 cn04 g50t ka59 lf52 lp85 m0r0 r11l re86 s5i5 sb26 sk48 su15 tn36 tr87
EFFICIENCY binds (reported <  term1):  9 / 25
  bp35 dc22 ft09 ls20 sc25 sp80 tu93 vc33 wa30

points lost to the efficiency cap: 1.25 of an 11.94 completion-only ceiling
                                   (reported mean 10.69)

on the levels we engaged: ours 4,970 vs human 3,623 = 1.37x human
across ALL levels incl. the 135 never reached: 4,970 vs 17,135 = 0.29x
```

Three consequences:

- **Perfect efficiency is worth at most +1.25 on a public-25 scale of 10.69**
  (+11.7% relative). Even that is unreachable, since it requires never exceeding
  the human action count on any solved level.
- **The completion-only ceiling at our current level count is 11.94.** Scaling
  by the observed public-25 → hidden ratio (10.69 → 2.83), reaching the 3.41
  top-10% bar needs roughly public-25 ≈ 12.9 — **above the ceiling that our
  current 48 solved levels can produce even at perfect efficiency**. [INFERRED
  from a single linear ratio; the mapping is not measured, but the direction is
  robust.] **Only solving more levels gets there.**
- Where efficiency *does* bind, it binds hard and locally: `sp80` has
  `term1 = 4.76` and reported **0.17** — we spent ~233 actions on a level 1 the
  human clears in far fewer. §4.3 has the probable reason, and it is not
  "the model is sloppy".

**The 51.4%-of-actions-on-the-never-completed-level figure independently
reproduces the brief's 50.2%.** But note what RHAE does with those actions:
*nothing*. Only solved levels enter either sum. They are pure opportunity cost —
time that could have been spent solving — not a scored penalty.

---

## 3. Task 2 — Tufa Labs at 18.81

**Finding: there is no public description of the system that scores 18.81.**
[VERIFIED by absence — searched and fetched their GitHub, research page, X, the
ARC Prize milestone write-up, and coverage.]

What *is* public, all of it describing the milestone-1 system:

| source | content |
|---|---|
| [github.com/Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness) | the harness we run, MIT-ish open source, milestone-1 state |
| [tufalabs.ai/research/duck-harness](https://tufalabs.ai/research/duck-harness/) | design write-up: REPL over Python variables, image+ASCII+segmentation, "infinite play via eviction", **Qwen 3.6 27B FP8**, mean public-25 **1.6002 ± 0.4475** over 20 tries |
| [arcprize.org/blog/arc-prize-2026-milestone-1](https://arcprize.org/blog/arc-prize-2026-milestone-1) | 1st of 3; code-writing vs the runners-up's "vision-LLM-as-policy" JSON-action designs |
| [x.com/tufalabs](https://x.com/tufalabs/status/2072336849465417747) | "We hit 1.21% with our lightweight harness" |

Their two stated design principles are worth recording because they contradict
things this project has repeatedly been tempted by:

- *"keep the harness lightweight and generic and let the model drive"* — gains
  came from **multimodality and better base models**, not hand-built tools.
- *"hand-crafted tools actually hurt the model; letting it improvise worked
  better."* This is the same conclusion our own `GraphExplorerJepaAgent`
  reached the expensive way (46 → 34 levels when an informed per-step bias was
  added to a working algorithm).

**Reading the 18.81 honestly.** Tufa has **138 submissions**. The public fork
scores ~2.8 (our measured mean on byte-identical code) to 4.33 (best of 62 for
`wuliao0`). Getting from there to 18.81 is a 4–6× multiple that no public
artifact explains. Two non-exclusive readings, neither verifiable from here:
(a) they have a materially better *model* — they are the only team with a reason
to hold one back; (b) they have solver-level work of the kind §4 shows other
teams building in the open. **[INFERRED.]** Either way, **there is no Tufa
technique sitting in public that we are failing to copy.** That is a useful
negative: the copy-the-leader strategy has no target, and the realistic
comparison class is the 3.4–4.3 band of public-fork teams, not the 18.81.

---

## 4. Task 4 — what the field is actually shipping right now

### 4.1 The leaderboard

Pulled live [VERIFIED, `competition_leaderboard_view`, 2026-09-15]: 1 Tufa Labs
**18.81**, 2 Ebi 8.68, 3 Lord Han Solo 8.44, 4 NVARC3 8.40, 5 Third Intelligence
8.21, 6 Daniel Franzen 7.63, 7 mostik.ai 7.51, 8 Tong Hui Kang 7.38, 9 Mark
Slavin 7.29, 10 Kyutai 7.19. Our 3.11 is rank 422 / 3,065.

### 4.2 Thuitanium / Knowless Crew (3.74) — a controlled lever program, in public

This is the most valuable thing in the whole survey. `sahasawatt` and
`yocybercode` are publishing **matched treatment/control pairs**, one lever each,
on the same NVFP4-serving + anim-solver chassis we now run. Every one is
implemented as a wrapper installed in the notebook's customization cell, after
`serving_setup` and **before the `inference` import** (the harness reads its
knobs at import time). Source pulled with `kaggle kernels pull` [VERIFIED].

All of them also apply the same two knob overrides:
`LOCAL_ANALYZER_SEED = 20260825`, `LOCAL_ANALYZER_YIELD_SECONDS = 180`
— and assert the upstream persisted value is still `"60"`, so a silent upstream
change fails loudly.

| lever | slug | mechanism (their own words, verbatim where quoted) |
|---|---|---|
| **world-model wipe guard** | `sahasawatt/thui-wm-v0` / `-ctl` | wraps `ToolAgent._update_summarized_knowledge_from_step_summary` to keep the six summarized-knowledge fields "across an in-level `game_over` (the harness erases them before the auto-RESET replays the same level)"; level transitions and run completion wipe as upstream |
| **action floor** | `sahasawatt/thui-af-v0` | wraps `ToolAgent.analyze`: "after 2 consecutive turn ends with no executed action on the same level, the harness commits one exploratory action through the solver's own `step_env` (a MOUSE click on a rare-colour cell not used before on this level, else a random non-mouse valid action; cap 40 per level). No prompt text is added." |
| **death blacklist** | `sahasawatt/thui-db-v1` | wraps `_summarize_step_sequence` to record, per level, the action that ended the game, and `_build_user_prompt` to list them. Goes further: infers hidden **per-level action budgets** from repeated deaths — "≥ 3 lives whose totals agree within ±1 → the median total (**sp80 L1 = 30, tn36 = 61, sp80 L2 = 45**)" and per-type budgets ("**sp80 L2 = 5 SPACE**") |
| **restart-at-stall** | `sahasawatt/thui-rs-v0` | wraps `ToolAgent.analyze`: "after 20 analysis turns without a level change the agent's history and world model are reset and the sampling seed bumped (at most 2 restarts)" |
| serving arms | `thui-m0-s20`, `thui-l4-s16/s28` | MTP-0 / KV-7-GiB profile; `max_num_seqs` sweeps |

`thui-af` is dated **2026-09-15** — today. An independent team, working from the
same artifacts, identified the same no-op pathology this document root-causes in
§1 and built a floor against it. That is the strongest possible corroboration
short of a matched run.

### 4.3 The `sp80` disaster, explained by someone else's finding

Our `sp80` scored **0.17** against a `term1` of 4.76 — the worst
efficiency collapse in the run. Thuitanium's death-blacklist code names `sp80`
explicitly: **level 1 has a hidden 30-action budget, level 2 a 45-action budget
and a 5-`SPACE` type budget.** We spent ~233 actions on `sp80` level 1.

So the collapse is not sloppiness — the game *kills you* at action 30, and the
only way to learn that is to die repeatedly and count. **And the harness wipes
the world model on every one of those deaths** (§4.4), so the agent can never
accumulate the count. The wipe guard and the death blacklist are two halves of
one fix.

### 4.4 The wipe, in our own harness source

`tool_agent.py:1113-1126` [VERIFIED]:

```python
def _update_summarized_knowledge_from_step_summary(self) -> None:
    summary = self._last_step_summary
    if not summary:
        return
    if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
        for key in ("world_model", "goal_model", "action_model",
                    "recent_findings", "open_questions", "current_plan"):
            self._summarized_knowledge[key] = ""
```

Six of the seven persistent knowledge fields are erased on **every** level
transition, run completion **and game over**. Only `cross_level_notes` survives.
The system prompt meanwhile tells the model *"Levels often build on earlier
mechanics"*.

**Sized on our own run** [VERIFIED, from `artifacts/*_events.jsonl`]: the nvfp4
baseline logged **74 `GAME_OVER` events and 48 level changes** — about **122
wipes across 720 analysis steps, one every ~6 steps.** The 74 game-over wipes
are the pointless ones: the level replays unchanged, and the agent has just
thrown away everything it learned about it.

### 4.5 Others sweeping the same knobs

`juliancamilovilla` is running, in public, exactly the two analyzer-knob
experiments §1 implies [VERIFIED, pulled source]:

- `arc-agi3-nvfp4-carry-yield-long-2`:
  `os.environ["LOCAL_ANALYZER_YIELD_SECONDS"]="180"` **and**
  `_tay._LOCAL_ANALYZER_YIELD_SECONDS = 180.0` — patching both the env and the
  already-bound module global, which is the correct way.
- `arc-agi3-nvfp4-carry-nothink-long`:
  `LOCAL_ANALYZER_ENABLE_THINKING = False`.

Plus `carry2/carry3/mtpshare/manual` variants. `dantelok` (3.49) has moved on to
`lb-9-arc3-duck-v17-with-gpt-oss-120b` — a different base model.

---

## 5. Task 1 & 3 — the paper track, and what survives our budget

### 5.1 First: the anim graft result, recovered

`calamitychasm/arc3-duck-nvfp4-anim` v1 is `COMPLETE`. `stage7_model_search.md`
§6.2 left the result blank and set the gate: *"ship it if public-25 ≥ ~10.69
with actions not collapsed."* [VERIFIED, pulled output]:

| | baseline | anim graft |
|---|---:|---:|
| public-25 mean | **10.69** | **9.97** (−6.7%) |
| actions | 3,633 | 2,615 (−28%) |
| LLM calls | 1,339 | 1,358 (+1.4%) |
| P(call acts) | 54.0% | **63.6%** |
| games scoring 0 | 4 | 5 |

**It did not clear its own gate.** Score is slightly down and actions are down
28%. But read against §1: the graft's `yield 180` did what it was supposed to —
the acting fraction rose 54% → 64% — and *actions still fell*, because
actions-per-acting-call fell further (5.23 → 4.46) and the calls-per-game budget
is fixed by latency. The knob moved the right quantity and the solver swap gave
it back. **This is a confounded two-variable run, as `stage7_model_search.md`
§2.0 warned**, and it is the single best argument for testing `yield_seconds`
*alone* on the baseline chassis (§6, intervention 2).

It has already been submitted (ref `56264462`, 2026-09-16, `PENDING`). Nothing
here argues for spending another slot on it.

### 5.2 The 2026 paper track, in one table

All secondary sources — their numbers, not ours.

| paper | mechanism | headline | feasible for us? |
|---|---|---|---|
| [arXiv:2603.24621](https://arxiv.org/abs/2603.24621) ARC-AGI-3 tech report | the benchmark + RHAE definition | humans 100%, frontier < 1% as of Mar 2026 | reference |
| [arXiv:2605.05138](https://arxiv.org/abs/2605.05138) Executable World Models | coding agent writes a Python world model, verifies it against observed transitions, refactors, plans through it | 58.12% mean RHAE | **no** — this is our own `CodeWorldAgent` (scored 0.00); needs many model calls per action |
| [arXiv:2605.25931](https://arxiv.org/abs/2605.25931) AERA | explicit **EXPLORE → VERIFY → PLAN** phases; formalises a speed/depth Pareto frontier that RHAE's square penalises deviation from | 0.2116 public-25, 0.30 private | **partially** — the *phase discipline* is cheap; see intervention 5 |
| [arXiv:2607.15439](https://arxiv.org/html/2607.15439) Ablation: do coding agents need EWM / simplification / verification? | nested ablation of the three mechanisms | verification best in all 4 settings but **1.82–3.26× the tokens**; textual **beat** executable at both GPT-5.5 settings; simplification "can be harmful when an agent has not yet formed a stable model"; best run: 183/183 levels in **7,758 actions vs 17,135 human** | **the negative results are the usable part** |
| [arXiv:2607.28287](https://arxiv.org/abs/2607.28287) Tycho | "active abstraction": actor decides *when building/using a model is worth its action cost*, delegating to a model-builder | 88.49 RHAE (Opus 4.8); 100.00 (GPT-5.6 Sol / Opus 5) | **no** — frontier models, matched large inference budgets |
| [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531) OPINE-World | ontology-error-prioritised interactive exploration | — | no |

**The single most useful thing in the paper track is a negative result.** The
ablation paper finds that at the highest capability the *textual* world model —
no executable requirement, no simplification, no verification — completed all
183 levels action-efficiently, and that the two "sophisticated" mechanisms are
either expensive (verification: 1.8–3.3× tokens) or actively harmful when the
model has not yet stabilised (simplification). Our own history says the same
thing twice over: `CodeWorldAgent` (the executable-world-model design) scored
**0.00**, and Tufa's own conclusion is that hand-crafted tools hurt.

**Every paper-track mechanism costs more model calls per action. We have ~54
calls per game, hard-capped by latency.** That is the feasibility filter, and it
excludes essentially all of them *as designed*. The one thing that buys the
budget to afford any of them is §1: recovering the 46% of calls we currently
throw away.

The AERA framing survives on its own terms, though, and is worth stating because
it maps onto §2: the metric rewards *knowing when to stop exploring*. Our agent
never stops — 51.4% of actions go into a level that never completes, and the
harness erases its accumulated knowledge every ~6 steps so it cannot tell that
it is looping.

---

## 6. Ranked shortlist of interventions

Ranked by (expected effect × confidence) ÷ cost. Every one is testable on a
**free** `kaggle kernels push` against the public-25 path, with the baseline to
beat being **10.69 mean / 3,633 actions**.

Per `stage7_model_search.md` §2, judge every candidate on **score and actions
together**: a score gain with an actions collapse is how a capability regression
disguises itself.

---

### 1. World-model wipe guard — keep knowledge across an in-level `game_over`

- **Mechanism.** `tool_agent.py:1113` erases six of seven summarized-knowledge
  fields on `game_over`, but a game over triggers an auto-RESET that **replays
  the same level with the same mechanics**. Wrap the method to skip the wipe
  when `game_over and not level_transition and not run_complete`.
- **Evidence.** [VERIFIED] the wipe code, read from the bundle we run.
  [VERIFIED] **74 `GAME_OVER` events in our baseline run**, ~3 per game, one
  wipe every ~6 analysis steps. [VERIFIED] a team at **3.74** ships exactly this
  with a matched control arm (`sahasawatt/thui-wm-v0` / `thui-wm-ctl`).
  [VERIFIED] the `sp80` mechanism in §4.3 makes the loss concrete: the only way
  to learn a hidden 30-action budget is to die and count, and the wipe deletes
  the count.
- **Harness change.** ~15 lines in the notebook customization cell, after
  `serving_setup`, before the `inference` import. Pure wrapper, exception-guarded,
  falls through to the original on any error.
- **Effect size.** [INFERRED] modest but broad — a few percent of public-25,
  concentrated on the death-heavy games (`sp80`, `bp35`, `dc22`, `sk48`). Not a
  multiplier.
- **Risk.** Very low. It only *retains* state; it cannot spend actions or tokens.
- **Cheapest test.** One free push; count `THUI_WM_KEPT`-equivalent log lines to
  confirm it fires ~74 times, then compare score+actions.

---

### 2. `LOCAL_ANALYZER_YIELD_SECONDS` 60 → 180, **alone**, on the baseline chassis

- **Mechanism.** §1.3. At 60 s against a 153 s round-trip the analyzer gets
  exactly one model call per turn, so any inspect-first turn is dead. 180 s
  permits inspect→act inside one turn, with the tool result already in context.
- **Evidence.** [VERIFIED] the loop code and the 100%-`turn_time_budget` stop
  reason. [VERIFIED] the one run at 180 (anim) has the highest acting fraction
  of all five (63.6% vs 46–54%). [VERIFIED] every Thuitanium lever arm and
  `juliancamilovilla/arc-agi3-nvfp4-carry-yield-long-2` set it to 180.
- **Harness change.** Two lines — patch `SETUP_ENV_PATH` json + `os.environ`
  before the `inference` import, then assert
  `_tool_agent._LOCAL_ANALYZER_YIELD_SECONDS == 180.0`. Copy the assert pattern
  from `thui-rs-v0` verbatim; it fails loudly if upstream changes.
- **Effect size.** [INFERRED] uncertain and possibly **zero or negative**.
  Calls-per-game is fixed by latency, so this does not buy calls — it regroups
  them. The one datapoint we have (anim) raised the acting fraction and still
  lost actions and score, but it changed the solver at the same time. **This
  test exists to de-confound that**, which is worth a free push on its own.
- **Cheapest test.** One free push: baseline chassis, yield the only change.

---

### 3. Fix the prompt's false claim about the tool-call budget

- **Mechanism.** §1.4. `COMPACT_TOOL_SESSION_ADDENDUM` tells the model it may
  call `python` "as many times as you want per step" and to "spend extra tool
  calls". On our stack it gets one. Rebind
  `tool_agent.COMPACT_TOOL_SESSION_ADDENDUM` before the analyzer is constructed
  (it is imported *into* `tool_agent`'s namespace at line 18, so patching
  `prompts.*` does nothing — patch `tool_agent.*`), replacing those two lines
  with an accurate statement: this turn affords one `python` call, so inspect
  and act in the same call, ending with `action(...)`.
- **Evidence.** [VERIFIED] the prompt text, the import binding, and the
  one-call-per-turn measurement. [VERIFIED] precedent for this exact override
  technique in our own repo (`scripts/_patch_duck_notebook_context.py` overrides
  `tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW` the same way). **[INFERRED]** that
  correcting it changes behaviour — no one has measured this.
- **Harness change.** ~10 lines. Must assert the original string is present
  before replacing, so an upstream reword fails loudly rather than silently
  no-op'ing.
- **Effect size.** [INFERRED] potentially the largest single lever here — the
  no-op fraction is 46% and this addresses its stated cause rather than its
  timing. Upper bound +85% productive turns (§1.5); a realistic guess is far
  less.
- **Risk.** Real, and the direction is known from our own history: `dedupe` and
  `ctx16k` both degraded the model's ability to decide and **both raised tokens
  per turn and lost score**. Forcing premature action could do the same. Prefer
  the honest correction ("you get one call, so inspect *and* act in it") over a
  prohibition ("never inspect").
- **Cheapest test.** One free push. The diagnostic is cheap and decisive: the
  acting fraction is printed in every `[ANALYZER STATUS]` block, so a single
  3-game smoke run at 1800 s tells you whether it moved before you spend a
  full 25-game run.

---

### 4. Action floor — commit an exploratory action after 2 dead turns

- **Mechanism.** Wrap `ToolAgent.analyze`: after 2 consecutive no-action turn
  ends on the same level, push one exploratory action through `step_env` — a
  MOUSE click on a rare-colour cell not previously used on this level, else a
  random non-MOUSE valid action; cap 40 per level.
- **Evidence.** [VERIFIED] shipped by the 3.74 team today
  (`sahasawatt/thui-af-v0`, 2026-09-15), reached independently from the same
  artifacts. [VERIFIED] our own no-op measurement is what it targets.
- **Harness change.** ~60 lines, the largest of these. Needs access to the
  solver's `step_env` and per-level bookkeeping.
- **Effect size.** [INFERRED] converts dead turns into environment information
  rather than into model turns — a different, weaker fix than #3, which tries to
  make the *model* act. Guarantees a floor on actions per level.
- **Risk.** Moderate, and specifically interacts with §4.3: forced random
  actions **spend the hidden per-level action budget**. On `sp80` (budget 30)
  a floor could actively cause deaths. Their 40-per-level cap is not obviously
  safe on budget-limited levels.
- **Cheapest test.** One free push, but do #3 first — they address the same
  40–50% of turns and the interaction is not additive.

---

### 5. Restart-at-stall — reset history + world model + seed after N stalled turns

- **Mechanism.** After 20 analysis turns with no level change, clear history and
  world model and bump the sampling seed; at most 2 restarts.
  (`sahasawatt/thui-rs-v0`.)
- **Evidence.** [VERIFIED] shipped by the 3.74 team. [VERIFIED] the pathology in
  our data: **51.4% of all actions go into the level that never completes**, and
  every one of the 25 games ends `gave_up`. [SECONDARY] AERA (arXiv:2605.25931)
  formalises exactly this "know when to stop" decision as the thing RHAE's
  square rewards.
- **Harness change.** ~30 lines.
- **Effect size.** [INFERRED] targets the largest single pool of wasted work,
  but note it is **opportunity cost, not a scored penalty** (§2) — the gain only
  materialises if the re-draw actually solves the level. Also note the tension
  with #1: this deliberately wipes the world model, which #1 deliberately
  preserves. They are not contradictory (different triggers) but must not be
  merged carelessly.
- **Cheapest test.** One free push on the stall-heavy games (`sk48` scored 0.00,
  `sp80` 0.17, `sc25` 0.80, `bp35` 0.96).

---

### 6. Death blacklist / per-level action-budget inference

- **Mechanism.** Record the action that ended each life, per level; surface them
  in the user prompt; infer hidden per-level action budgets and per-type budgets
  from ≥3 agreeing deaths. (`sahasawatt/thui-db-v1`.)
- **Evidence.** [VERIFIED] shipped by the 3.74 team, with concrete inferred
  budgets for games in our own roster (`sp80` L1 = 30, L2 = 45 + 5 `SPACE`;
  `tn36` = 61). [VERIFIED] our `sp80` result (0.17 against a 4.76 completion
  term) is the predicted failure.
- **Harness change.** The most code of any item here (~100 lines, two wrappers
  plus an inference routine) and it adds prompt tokens.
- **Effect size.** [INFERRED] potentially large on the ~9 efficiency-bound
  games, but §2 caps the *entire* efficiency term at +1.25 of 11.94. Its real
  value is more likely in completion: not dying means reaching deeper levels.
- **Cheapest test.** Depends on #1 — without the wipe guard the agent has no
  memory to accumulate budgets into.

---

### 7. `LOCAL_ANALYZER_ENABLE_THINKING = False`

- **Mechanism.** Disables the reasoning block; tokens per turn should fall
  sharply, cutting latency and raising calls/game.
- **Evidence.** [VERIFIED] the knob exists (`tool_agent.py:144`) and
  `juliancamilovilla/arc-agi3-nvfp4-carry-nothink-long` is running it.
- **Effect size / risk.** [INFERRED] **high risk, listed for completeness, not
  recommended first.** Reasoning is 32.8% of the prompt
  (`stage7_context_budget.md`) and is the mechanism by which this model plans.
  Our own precedent is discouraging: every intervention that reduced what the
  model could think with (`dedupe`, `ctx16k`) **raised** tokens per turn and lost
  score. It is cheap to test free, so it is worth *a* push — just not the first.

---

### What is out of reach, and why

- **A better model.** Closed by `stage7_model_search.md` §3–4 and unchanged by
  anything found here: the NVFP4 serving bundle verifies 419 per-file SHA-256s
  plus its own hash, and of 409 enumerated Kaggle models everything fitting
  95 GiB is a weaker coder.
- **Tycho-style active abstraction / model-builder delegation
  (arXiv:2607.28287), and verification-by-replay (arXiv:2607.15439).** Both need
  many model calls per action. We have ~54 calls **per game**. Verification
  alone costs 1.82–3.26× the tokens in the paper's own measurements. Not
  reachable until §1's 46% is recovered, and probably not then.
- **Executable world models (arXiv:2605.05138).** Already tried in this repo as
  `CodeWorldAgent`: scored **0.00**. The 2026 ablation paper now independently
  reports that the *textual* variant beat the executable one at both GPT-5.5
  settings. Do not revive it.
- **More serving throughput.** Seven dead levers already
  (`stage7_kv_residency_levers.md`, `stage7_duck_throughput.md`,
  `stage7_model_search.md`), best win +16.5%. §1.2 shows why the ceiling is
  low: throughput buys calls, and 46% of calls are discarded. Fix the denominator
  first.

---

## 7. Corrections to earlier Stage 7 documents

- `stage7_model_search.md` §2 — "`tokens_per_action` varies 3.5× … whether that
  variation is a property of the model or of the solver is *not* settled".
  It is neither, primarily: it is a property of **`yield_seconds` against turn
  latency**, which determines what fraction of LLM calls take an action at all
  (§1.3). The fp8/nvfp4 confound remains real but is no longer the main question.
- `stage7_model_search.md` §6.2 — the anim free-run result, left pending, is
  **9.97 / 2,615 actions** and did **not** clear its own gate (§5.1).
- `stage7_analyzer_timeouts.md` TL;DR — "the harness is 100% latency-bound" is
  right, and its recommendation not to chase the 25 read timeouts still stands.
  But the same measurement contains a much larger finding it did not draw:
  **46% of the ~52 turns per game take no action.** The timeouts are 1.9% of
  turns; the yields are 46%.
- The task brief's framing that RHAE's squared efficiency term is what punishes
  us is **not supported by our own data**: completion binds in 16 of 25 games
  and the whole efficiency cap is worth 1.25 of 11.94 (§2).
