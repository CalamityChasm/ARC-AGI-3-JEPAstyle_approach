# Stage 7 — Component search: what whole subsystems are still swappable

Date: 2026-09-20. Branch: `stage7-components`.

**No competition submission was made by this investigation, and none will be.**
Everything here is read-only Kaggle API calls, public bundle source, public
kernel logs, and local computation over them. The daily slot was untouched.

Labels follow this repo's convention. **[VERIFIED]** = read directly from a
primary artifact (a Kaggle API response, a downloaded bundle's own source or
pickle, a public kernel's own log, or a local computation over them).
**[INFERRED]** = reasoning over verified facts. Secondary sources (papers,
vendor blogs) are cited and labelled as such.

Artifacts are in `experiments/stage7_components_artifacts/`:

| file | what it is |
|---|---|
| `bundle_provenance.json` | `git_status.txt` + `taaf-kaggle-bundle.json` from 62 candidate datasets |
| `bundle_solver_flags.json` | solver flags read out of each bundle's own `benchmark_initial.pkl` |
| `bundle_tree_diffs.txt` | `diff -rq` of every downloaded solver tree against our anim incumbent |
| `candidate_dataset_files.json` | full file listing + accessibility for all 62 |
| `leaderboard_20260920.csv` | full 3,175-team leaderboard |
| `competition_notebooks_20260920.json` | 1,074 public competition notebooks |
| `public_kernel_telemetry.json` | per-game `[finished]` lines parsed out of public kernel logs |
| `avo_*.py.txt` | the AVO module's own source |

`scripts/summarize_public_kernel_log.py` reproduces the telemetry table.

---

## 0. TL;DR — the answer is yes, and it is the same shape as the last win

**There are more solver bundles, they are all accessible with our credentials,
and one of them is our current winning solver plus a whole new subsystem,
published by the same author whose previous bundle was worth +25%.**

`jakobbrggen/taaf-kaggle-source` (v27, `experiment/avo-v2`, 2026-09-01) ships a
`benchmark_initial.pkl` whose solver reads [VERIFIED, §2.2]:

```
avo_agent            = True     <-- the new subsystem
animation_awareness  = True     <-- our +25% win, retained
animation_retrieval  = False
hard_noop_guard      = True
max_runtime_s_per_game = 7920.0 <-- identical to ours
concurrency          = 28       <-- our build overrides to 37
```

It is a **strict superset** of `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`:
`animation.py` and `noop_guard.py` are both still present, and
`inference/avo/` (733 lines) is added [VERIFIED, §2.3].

AVO is durable cross-level memory + a phased inspect/plan/implement/evaluate
loop + a **stagnation supervisor**. The supervisor costs **zero extra model
calls** — it appends a paragraph to the prompt the turn already sends. That
matters more than anything else in this document, because §5 shows every
*hypothetical* deliberation component (verifier, self-consistency, ensemble)
is excluded by clock arithmetic and this one is not.

### Ranked shortlist

| # | candidate | kind | what it replaces/adds | fits? | cheapest test |
|---|---|---|---|:--:|---|
| **1** | **`raist321/taaf-avo-v27-bundle`** (pinned re-host of `jakobbrggen/taaf-kaggle-source` v27) | **whole solver-bundle swap** | anim solver → anim + durable memory + phased loop + stagnation supervisor | **yes** | 4-line edit to the existing anim notebook; §3.4 |
| **2** | **`iseesmth/taaf-kaggle-source-prolong-eval`** | **whole solver-bundle swap** | anim solver → anim + PROLONG lossless programmatic memory in the sandbox | **yes** | same 4-line edit, different mount/label |
| 3 | `MULTIMODAL_GRID_LINES=1` (ships inside candidate 1) | perception component | flat PNG render → grid-lined render | yes, but see §4.1 | rides free on candidate 1 as a second arm |
| 4 | `sonphamorg/arc3-flashnext-serving-part-{a,b,c}-v1` | **whole serving-stack swap** | the sealed keithtyser appliance → an unsealed, pre-converted build of the *same* model revision | maybe; 135 GB in 3 mounts | free push, boot-only smoke; §4.2 |
| 5 | `ataraxian/arc3-duck-prompt-v36a` prompt subsystem | prompt-subsystem graft | anim's `prompts.py` → a rank-19 team's rewrite | yes | graft one file; weakest of the five, §4.3 |
| — | verifier pass / self-consistency / prompt ensemble | **rejected on arithmetic** | — | **no** | §5 |
| — | model swap, KV/serving tuning, retrieval over prior games | **rejected, already closed** | — | no | §6 |

**Candidate 1 is the recommendation.** It is the only candidate that is
(a) a whole-subsystem swap, (b) a strict superset of the configuration that
currently scores 3.79, (c) from the author whose last bundle was worth +25%,
and (d) free of extra model calls.

**The honest counterweight, stated before the enthusiasm:** the one public run
of this bundle scored **4.32** on public-25 — but it ran on the *old FP8-27B*
chassis, and its matched control's log has expired, so it is not an A/B
[VERIFIED, §3.3]. There is also a design-level risk that AVO's own
`INSPECT` phase *tells the model not to act*, which is the opposite of what
our measured 37.7% dead-turn rate needs (§3.5). Both are real and both are
addressable by which arm is run first.

---

## 1. Method, and what was actually enumerated

### 1.1 Datasets

Sixteen search terms against `datasets/list`, five pages each, deduplicated:
**551 distinct datasets**, 206 matching an ARC-3/TAAF/Duck pattern, of which
**62 were probed individually** for file listings [VERIFIED,
`candidate_dataset_files.json`].

**All 62 returned a full file listing on our account. Zero were inaccessible.**
Accessibility is not the constraint; knowing which one is worth mounting is.

For each, `git_status.txt` and `taaf-kaggle-bundle.json` were pulled
individually over the raw download API (the CLI's own single-file path is
unreliable on Windows — see this repo's own gotcha about mangled upload paths;
the same helper is used in reverse here).

### 1.2 Notebooks and the leaderboard

`kernels/list` across four sort orders: **1,074 distinct public competition
notebooks** [VERIFIED]. Full leaderboard CSV: **3,175 teams** [VERIFIED].

Public kernel *outputs* are readable for public notebooks, and they contain
the harness's own `[finished] <game> state=… level=n/m score=… actions=…`
lines. That is **counted telemetry, not a sampled score** — the distinction
`stage7_noise_floor.md` insists on — and it is how §3.3's numbers were got.

### 1.3 Where we stand, refreshed

[VERIFIED, `leaderboard_20260920.csv`]

| | |
|---|---|
| us (`How bad can it go?`, `calamitychasm`) | **rank 204 / 3,175, score 3.79, 33 submissions** |
| top 10% | rank 318, score **3.51** |
| top 5% | rank 159, score **3.93** |
| #1 Tufa Labs | 18.81 (141 submissions) |

The brief's "bar ~3.47" is now **3.51**, and we are in the **top 6.4%**.
**Top 5% is +0.14 away** — inside one good draw of the anim arm's own measured
range (3.37–3.79).

---

## 2. The bundle landscape: thirteen distinct solver lineages

### 2.1 The lineage table

Every TAAF bundle carries a `git_status.txt` naming the upstream
`ARC3-Inference` revision and branch. Deduplicating 62 datasets by that
revision gives **13 distinct lineages** [VERIFIED,
`bundle_provenance.json`]:

| rev | branch | date | what it is | example mount | author's LB rank/score |
|---|---|---|---|---|---|
| `e567133` | `main` | 2026-06-03 | pre-share `main` | `phuongncn/arc3-tufa-v21-source-v001` | 10 / 7.37 |
| `e442e01` | `submission-share-mode` | 2026-06-10 | older share | `jeroencottaar/taaf-kaggle-source` | 1 / 18.81 |
| **`aa69123`** | `add-kaggle-share-flag` | 2026-06-12 | **the June duck the sealed NVFP4 bundle itself pins** | `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` | 99 / 4.17 |
| `3090308` | `feat/deepinfra-and-goal-hints` | 2026-08-06 | goal-inference toggles, `loop_engineering.py`, `trace_index.py` | `toprakg/taaf-kaggle-source` | 947 / 1.28 |
| `a8a6ddc` | `submission/v1` | 2026-08-07 | `compaction.py`, `external_history.py`, `framework/budget.py`, `inference/meta/` | `gktrkakman/taaf-src-subv1` | 843 / 1.53 |
| **`9158303`** | `feature/animation-awareness` | 2026-08-07 | **OUR INCUMBENT** | `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` | 168 / 3.91 |
| `25d4085` | `prolong` | 2026-08-11 | PROLONG programmatic memory, no anim | `iseesmth/duck-harness-prolong-source-20260811` | 492 / 3.10 |
| `85d0141` | `nca-ppt` | 2026-08-11 | NCA/SFT branch | `iseesmth/duck-harness-nca-source-20260811` | 492 / 3.10 |
| **`2f2c948`** | `main` | 2026-08-11 | **anim + PROLONG memory** | `iseesmth/taaf-kaggle-source-prolong-eval` | 492 / 3.10 |
| `4823629` | `experiment/polyphony` | 2026-08-26 | `inference/agent` replaced by `inference/core` + `inference/polyphony` | `raist321/taaf-polyphony-v25-bundle` | 536 / 3.00 |
| **`74ff3df`** | `experiment/avo-v2` | 2026-09-01 | **anim + AVO** | `jakobbrggen/taaf-kaggle-source` v27 / `raist321/taaf-avo-v27-bundle` | 168 / 3.91 |
| `cf7d448` | `v10-persistence` | 2026-09-12 | "leave a digest behind when history is dropped"; no anim, no noop guard | `tanapatriewruja/taaf-src-v10-persistence` | 602 / 2.79 |
| (local) | `relator-v11-overlay` | 2026-08-08 | hand-written overlay on the June `main` | `brentwoodard/relator-duck-v11-hybrid-source` | 1890 / 0.23 |

Author rank is a **weak** proxy — a published bundle is usually an experiment,
not the author's best submission — but it is directional and it is free.

### 2.2 Feature matrix, read from each tree

[VERIFIED, `bundle_tree_diffs.txt` + direct file-existence check]

| bundle | `animation.py` | `noop_guard.py` | `programmatic_memory.py` | `inference/avo/` | `polyphony/` | `compaction.py` |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **anim (incumbent)** | Y | Y | – | – | – | – |
| **`jakobbrggen/taaf-kaggle-source` (avo-v2)** | **Y** | **Y** | – | **Y** | – | – |
| **`iseesmth/…-prolong-eval`** | **Y** | **Y** | **Y** | – | – | – |
| `taaf-src-v10-persistence` | – | – | – | – | – | – |
| `taaf-polyphony-v25-bundle` | – | – | – | – | Y | – |
| `duck-harness-prolong-…` | – | – | Y | – | – | – |
| `toprakg/taaf-kaggle-source` | – | – | Y | – | – | – |
| `gktrkakman/taaf-src-subv1` | – | – | – | – | – | Y |
| `jeroencottaar/…-share` (June) | – | – | – | – | – | – |

**Only two bundles are strict supersets of what currently scores 3.79.** They
are candidates 1 and 2. Everything else in the table would *lose* the
animation awareness that this project measured at +25% (Welch t=3.55, p<0.05),
which makes them net-negative-by-construction unless what they add is larger
than what they remove — and none has evidence that strong.

### 2.3 Solver flags, read from each bundle's own pickle

`benchmark_initial.pkl` carries the solver dataclass. Disassembling it
(`pickletools.dis`, no unpickling, no imports) gives the flags without running
anything [VERIFIED, `bundle_solver_flags.json`]:

| bundle | `avo_agent` | `animation_awareness` | `hard_noop_guard` | `max_runtime_s_per_game` | `concurrency` |
|---|:--:|:--:|:--:|---:|---:|
| **anim (incumbent)** | absent | True | True | 7920.0 | 28 |
| **avo-v2** | **True** | **True** | **True** | **7920.0** | 28 |
| prolong-eval | absent | True | True | 7920.0 | 28 |
| polyphony-v25 | absent | absent | absent | 7920.0 | 28 |
| v10-persistence | absent | absent | absent | **3600.0** | **5** |
| toprakg goal-hints | absent | absent | absent | **2700.0** | **6** |
| gktrkakman subv1 | absent | absent | absent | **5400.0** | 28 |

The bottom three are configured for someone else's much shorter runs and would
need their budgets rewritten as well as their solver — more surface, less
reason.

### 2.4 One reproducibility hazard, and its fix

`jakobbrggen/taaf-kaggle-source` is a **rolling** dataset, currently at
**version 27**, last updated 2026-09-01 [VERIFIED, `datasets/view`]. If he
pushes v28, a mount by slug silently changes what we ran. This project has
already been burned by exactly this class of drift (`checkpoints/` holding a
fold-1 experimental checkpoint through a real submission).

`raist321/taaf-avo-v27-bundle` is a **frozen v1 pin** whose subtitle says
"Byte-identical re-host of jakobbrggen/taaf-kaggle-source v27
(experiment/avo-v2)". **Verified, not taken on trust** [VERIFIED]:

```
diff -rq b/b01/src b3/b00/src   ->  no differences
benchmark_initial.pkl           ->  identical
taaf-kaggle-bundle.json         ->  identical
preamble.txt / setup_commands.json -> identical
deploy_target.pkl               ->  identical
```

**Mount the pinned re-host, not the rolling slug.**

---

## 3. Candidate 1 — the AVO solver bundle

### 3.1 What it is

`inference/avo/` — 733 lines across six files, subclassing `ToolAgent` rather
than reimplementing it. Its own `__init__.py` says [VERIFIED, source]:

> NVIDIA's AVO … reached 100.00 RHAE on the ARC-AGI-3 public set by wrapping a
> frontier model in three things the model does not have on its own: a memory
> that survives the context window, an explicit loop that separates forming a
> hypothesis from testing it, and a supervisor that notices when the loop stops
> making progress and says so. None of that is model weights, which is the
> point — the same underlying model scores roughly 30% unwrapped.

The NVIDIA claim is real and independently confirmed: AVO with Claude Opus 5
completed all 183 public-set levels for **100.00 RHAE in 6,624 environment
actions**, against ~30% for the bare model
([NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/))
— **[secondary source, not our measurement]**, and with a different, far
stronger model than ours. NVIDIA released **no code and no ablations**, so
Tufa's `inference/avo/` is a *reimplementation from the published description*,
not NVIDIA's own system.

Four components:

1. **`AvoMemory`** (159 lines) — `ToolAgent._summarized_knowledge` is cleared
   whenever the session's runtime dir changes, i.e. **at every level boundary,
   which is exactly where the knowledge is worth the most**. AVO persists the
   same content to disk keyed by game and reloads it, with atomic writes
   because the Kaggle kernel is killed by wall clock, not a clean shutdown.
2. **Phased loop** (`prompts.py`, 76 lines) — INSPECT / PLAN / IMPLEMENT /
   EVALUATE rotate **on turn index, not on model self-report**, "because a
   model that is stuck is exactly the one that will claim it is evaluating."
3. **`Supervisor`** (163 lines) — see §3.2.
4. **Exploit deadline** — `exploit_deadline = 0.6` of
   `max_runtime_s_per_game`, i.e. **AVO stops building a world model at
   4,752 s of each game's 7,920 s and plays its best policy**. The settings
   file's own comment: "Without this the arm's most likely failure is an
   elegant world model and no score."

### 3.2 Why the supervisor is the interesting part

It is a strictly better-engineered version of the `commit floor` arm
`stage7_depth.md` §7.1 built, and it was tuned against a real 25-game run whose
numbers are in the source comments [VERIFIED, `avo_supervisor.py.txt`]:

> That definition alone turned out to be too loose to detect anything. In the
> first 25-game run almost every turn produced a pixel-wise novel frame on the
> 64×64 grid, so `barren_turns` was reset before it could ever reach the
> threshold: the twelve games that finished on a score of zero drew 30
> interventions across 767 turns, and the thirteen games that scored drew 33
> across 747 — the supervisor could not tell the two apart, and four games that
> ran 59 to 96 turns without a point were never nudged once.

So it has **two** triggers: `barren_turns >= 3` (yields to frame novelty) and
`unrewarded_turns >= 12` (ignores novelty entirely), with a 3-step escalation
to a HARD REDIRECT that then restarts the ladder rather than repeating itself.

This is the same lesson our own depth work reached from the other side, and the
mechanism already has one debugging iteration behind it that ours does not.
**It costs zero extra model calls**: `directive()` returns a string that is
appended to the user prompt the turn already sends.

All of it is env-tunable without touching source:
`ARC3_AVO_STAGNATION_TURNS`, `ARC3_AVO_UNREWARDED_TURNS`,
`ARC3_AVO_ESCALATION_INTERVENTIONS`, `ARC3_AVO_EXPLOIT_DEADLINE`,
`ARC3_AVO_PHASED_LOOP`, `ARC3_AVO_MAX_{FACTS,RULES,FAILURES}`
[VERIFIED, `avo_settings.py.txt`].

### 3.3 The only public run of it, and what it does and does not say

`yocybercode/thui-avo-v0` (Thuitanium / Knowless Crew, rank 58, 4.50), pushed
2026-09-05. Its own log confirms the arm was live [VERIFIED, parsed from the
public kernel log]:

```
avo_agent=True, animation_awareness=True, animation_retrieval=False,
hard_noop_guard=True, max_runtime_s_per_game=7920.0, concurrency=28
```

and `artifacts/avo_memory.json` is in its output, so the memory really wrote.

Public-25 telemetry, from its own `[finished]` lines
[VERIFIED, `public_kernel_telemetry.json`]:

| kernel | serving stack | solver | mean | actions | levels | games scoring |
|---|---|---|---:|---:|---:|---:|
| `yocybercode/thui-avo-v0` | **FP8 Qwen3.8-27B** + stock wheelhouse | anim + **AVO** | **4.32** | 2,572 | 21 | 15 |
| `lucifer19/blackcat-avo-variation-c10` | FP8 Qwen3.8-27B | anim + *own* memory layer | 5.15 | — | 25 | — |
| `yocybercode/thui-animfast-b71-full25-r1` | **NVFP4** | anim | 9.56 | 2,008 | 39 | 20 |
| our `arc3-duck-nvfp4-anim` | NVFP4 | anim | 9.97 | 2,615 | 42 | 20 |

**Read this carefully, because the naive reading is wrong in both directions.**

- It is **not** evidence AVO is bad. The AVO run is on the FP8-27B chassis,
  which `stage7_model_search.md` §1 measured at 3.37 local against NVFP4's
  10.69 — a ~3× local handicap. 4.32 on that chassis is *above* our own FP8
  arm's 3.37, not below it.
- It is **not** evidence AVO is good either. The matched control
  (`yocybercode/thui-v1-1`, the kernel the AVO notebook says it swaps one thing
  against) has an **expired log** — 800 characters, no `[finished]` lines
  [VERIFIED] — so the A/B cannot be recovered. And a single public-25 mean
  carries SE ±2.46 (`stage7_noise_floor.md`), which swallows the whole gap.
- The AVO run's `[finished]` notes read `tokens≈119,000` on 20 of 25 games,
  i.e. that chassis was **token-capped**, a constraint our stack does not have.
  One game (`m0r0`) took **0 actions**.

**Weak revealed-preference counter-evidence, recorded rather than buried:**
Thuitanium ran AVO on 2026-09-05 and every subsequent full-25 kernel of theirs
(`thui-l1-*` 09-10/09-12, `thui-anim-full25-r2` 09-15, `thui-af-*` 09-15) is
**anim-based, not AVO** [VERIFIED, their kernel list]. They appear to have
moved on. That is a reason for caution, not a disproof — they never ran it on
the NVFP4 stack either.

**The gap nobody has filled: AVO has never been run on the sealed NVFP4
appliance.** Every public AVO kernel mounts the FP8 wheelhouse. That is exactly
the gap the anim graft filled when it was worth +25%.

### 3.4 Does it fit, and what is the cheapest test

**Hardware: unchanged.** The bundle is 1.6 MB of Python. Nothing is loaded onto
the GPU that was not before; the sealed serving appliance and the model mount
are untouched. Memory is a small JSON on `/kaggle/working`.

**Clock: unchanged in the sense that matters.** `_avo_game_budget_seconds()`
returns `self.max_runtime_s_per_game`, which the bundle already sets to
**7920.0** — identical to ours [VERIFIED, `solver.py:1102` + pickle]. No extra
model calls: the supervisor writes into the existing prompt, and the phase
directive replaces prompt text rather than adding a turn.

**Cheapest test — a 4-line edit to the notebook we already ship**
(`kaggle_submission_duck_nvfp4_anim/`, built by
`scripts/_build_duck_nvfp4_anim.py`, which asserts all 18 inherited cells by
per-cell sha256):

1. `kernel-metadata.json` `dataset_sources[2]`:
   `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`
   → **`raist321/taaf-avo-v27-bundle`**
2. cell 7: `_find_bundle_dir("anim-20260807-anim")` →
   `_find_bundle_dir("avo-kaggle")` (the label in the new bundle's
   `taaf-kaggle-bundle.json`), and the `animation.py` existence assert still
   passes unchanged.
3. cell 11: `assert bm.label == "anim-20260807-anim"` → `"avo-kaggle"`, and
   **add** `assert bm.solver.avo_agent is True`. The existing
   `animation_awareness and hard_noop_guard` assert passes as-is.
4. cell 13: **this one is a real trap, not cosmetic.** Cell 13 asserts
   `target.max_runtime_s == 32400.0`. The AVO bundle's `deploy_target.pkl`
   carries **54000.0** (15 h — the AVO settings docstring's "15h Kaggle run")
   [VERIFIED, pickle]. Kaggle's hard cap is 9 h, so the notebook must **set**
   32400 rather than assert it, or the run overruns. `max_runtime_s_per_game`
   is unaffected at 7920.0, so AVO's exploit deadline still lands at 4,752 s
   where it should.

Everything else — cell 9's `sys.path` substitution, `LOCAL_ANALYZER_SEED=20260825`,
`LOCAL_ANALYZER_YIELD_SECONDS=180`, the serving profile, `concurrency=37` —
carries over untouched.

Then: one free `kaggle kernels push` for **catastrophe detection and
telemetry only** (does it boot, does `avo_agent=True` print, do
`avo_interventions`/`avo_hard_redirects`/`avo_exploit_turns` appear in the
experiment event, is the actions count in the normal band). **Do not rank it
on that run's score** — ±2.46 cannot. Ranking is n≥3 real submissions.

### 3.5 The risk that actually matters, and the arm that removes it

`INSPECT` (phase 1 of 4) opens with **"Do not act this turn unless the
situation is already unambiguous."** One turn in four is *instructed* not to
act. Our measured dead-turn rate is already **37.7%** (354 of 940 turns,
46.1% of the clock), and the depth work concluded that dead turns are the
dominant sink [VERIFIED, `stage7_depth.md` §2]. AVO's phased loop could
plausibly push that number up, and the supervisor's benefit would have to beat
its own loop's cost.

**This is directly separable, and the bundle ships the separation:**
`ARC3_AVO_PHASED_LOOP=0` gives "memory + supervisor only, same base agent",
which the settings file states is its purpose [VERIFIED, `avo_settings.py.txt`].

**Recommended arm order:**

- **Arm A: `ARC3_AVO_PHASED_LOOP=0`** — memory + supervisor + exploit deadline,
  no anti-action instruction. This is the arm whose mechanism matches our
  measured bottleneck with no mechanism pointing the other way.
- **Arm B: default (phased loop on)** — only if Arm A clears, or as a second
  slot.

A **counted falsifier registered in advance**, in the style
`stage7_depth.md` §7.3 established: the untreated chassis's own
P(act | ≥2 dead turns before) is **32.0% (164/513)**. If the supervisor works,
the armed-turn conversion rate should exceed that; if it does not, the
mechanism is inert regardless of what the score does. This is readable off a
free run, because it is counted, not sampled.

---

## 4. Candidates 2–5

### 4.1 Candidate 2 — PROLONG programmatic memory
**`iseesmth/taaf-kaggle-source-prolong-eval`** (rev `2f2c948`, label
`PROLONG-eval`, 2026-08-11, CC0, v2)

The second strict superset: `animation_awareness=True`, `hard_noop_guard=True`,
`max_runtime_s_per_game=7920.0`, plus
`inference/agent/programmatic_memory.py`. Its own docstring [VERIFIED]:

> Lossless, append-only game memory for programmatic retrieval. **The model
> never receives this file in its prompt.** Instead the sandbox exposes a
> read-only `memory` object so the model can search and parse the complete
> trajectory with Python **without consuming active-context tokens**.

It is wired **unconditionally** — imported by both `solver.py` and
`tool_agent.py`, no flag [VERIFIED].

**Why this is a serious candidate and not a footnote.** This repo has measured
twice that context is load-bearing and cannot be cut (history dedupe −34%,
ctx16k −81%), and Tufa's own stated weak areas are "context management and
perception". PROLONG is the only mechanism found that *adds* recall without
adding context: the trajectory lives on disk and is queried with the `python`
tool the turn already has. It costs **zero extra model calls** and zero prompt
tokens.

**Against it:** the author's team is rank 492 / 3.10, below us; no public run of
this specific bundle was found; and the graft is a full solver swap, so it
carries the same integration surface as candidate 1 with less evidence behind
it. Same 4-line test recipe (label `PROLONG-eval`).

### 4.2 Candidate 3 — `MULTIMODAL_GRID_LINES`
Ships **inside** candidate 1. `vision_context.py` gains
`_render_grid_lined_image()`: the canvas is filled with a mid-gray
`(128,128,128)` — deliberately midway between ARC grays 2 and 3 so it reads as
neither — and each cell is painted `scale-1` px, leaving a 1 px line
[VERIFIED, source]. It is off by default, gated on
`MULTIMODAL_GRID_LINES=1`, and requires `scale >= 4`.

This is a **perception** change, which is different from the upscale lever
CLAUDE.md recorded as inert (66 vision tokens at every setting): grid lines
change *what the image shows*, not its resolution. **Caveat to check before
believing it:** if the model really only ever sees 66 vision tokens, a 1 px
line may not survive tokenisation at all — that is checkable on a free run from
the token counts, and should be checked before spending a slot. Our stack
already runs `MULTIMODAL_UPSCALE=4`, which meets the `scale >= 4` gate
[VERIFIED, cell 9's own assert].

Cheapest test: it is a second, independent arm on the same mount as candidate 1
— one env var, no code change.

### 4.3 Candidate 4 — an unsealed serving stack for the *same* model
**`sonphamorg/arc3-flashnext-serving-part-{a,b,c}-v1`** (rank 16, **6.49**)

The one real answer to "other vLLM runtimes / serving stacks". Its README
[VERIFIED]:

> Part A of a three-dataset, fully offline serving package … Expanded payload:
> 293 files, 62,039,290,777 bytes … **Model revision:
> `7b719225242aacd3dbd3f9407468c2ee9a9d2594`** … The ZIP is a transport/file-count
> wrapper only. Routed experts remain NVFP4, and only ten PLE FP8 tensors are
> deterministic BF16 for pinned-vLLM compatibility. **No model reconstruction,
> decompression, copying, or PLE conversion occurs inside the notebook.**

That revision hash is **identical** to the one the sealed keithtyser bundle
pins (`MODEL_HF_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"`,
`stage7_model_search.md` §3). So this is **the same model, outside the seal**,
with its own pinned vLLM runtime archive and its own source bundle, published
by a team at 6.49.

**What it would buy:** it dissolves the constraint "the serving configuration
cannot be changed because `serving_setup.py` verifies its own sha256". Every
closed serving lever (KV pool, dtype, MTP depth, prefix caching) becomes open
again — though note this repo has already measured most of those as dead ends
*on the sealed stack*, and nothing says they revive.

**Against it, honestly:** ~135 GB across three mounts plus a fourth runtime
mount; the PLE-offload path that lets 126 GB of weights live on a 95 GiB card
is the sealed bundle's own, and whether sonphamorg's package reproduces it is
unverified from here; their bundled solver is of unknown lineage and would have
to be replaced by ours anyway; and we would be rebuilding a working stack from
scratch for no measured gain. **Rank 4 because the upside is "unlock a set of
levers already measured dead", not "a new subsystem".** Cheapest test is a free
push that does nothing but boot the server and print `/v1/models`.

Also enumerated and rejected for task 2: `saltb0x/arc3-vllm-wheelhouse-v0271-cu129`
and `codywhatleymd/arc3-vllm-0271-sm120-wheelhouse` (vLLM 0.27.1 wheelhouses) —
the sealed bundle verifies its runtime image's **layer digests**
(`serving_setup.py:848–891`), so a wheelhouse cannot be substituted under it;
they are only usable with a stack that is already unsealed, i.e. only as part
of candidate 4 or the FP8 stack we have already abandoned.

### 4.4 Candidate 5 — the prompt subsystem of a rank-19 team
**`ataraxian/arc3-duck-prompt-v36a`** (`ataraxian` = *Ya Xu*, rank 19, **6.23**)

Sixteen dataset versions named `arc3-duck-prompt-v24a … v36a`, i.e. a
rank-19 team publishing its prompt iterations in public. The tree is a fork of
the **June `aa69123` base**, so it is *not* a superset of anim — swapping it
wholesale would drop animation awareness.

What they actually changed, isolated by diffing against the June base rather
than against anim [VERIFIED]: `prompts.py` shrinks 114 → 89 lines. They
**removed** the "no player avatar", HUD/timer-bar and segmentation paragraphs,
and **added** two things:

- an explicit per-level tried-list: *"Keep an explicit record of what you have
  already tried on the CURRENT level … Before you declare a 'completely
  different approach', read that checklist."*
- and, notably: *"**Refusing to call `action(...)` for more than one
  consecutive turn is itself a failure mode.**"*

That second line is the commit floor, as a standing prompt rule, from a team
at 6.23. It is the cheapest possible expression of the mechanism our depth
work built a whole wrapper for.

**Ranked last because it is a graft, not a swap** — taking two sentences out of
someone's prompt and pasting them into ours is precisely the parameter-tuning
shape that has failed every time in this project, and the removals that come
with their prompt are not separable from the additions without judgement calls.
Listed because the evidence (rank 19) is the strongest of any author here, and
because if candidate 1's supervisor is what works, this is the same idea for
free.

---

## 5. Deliberation components that were evaluated and **rejected on arithmetic**

The brief asks specifically about a verifier/critic pass, self-consistency over
candidate actions, retrieval over prior games, and a prompt ensemble. Each was
costed against the measured clock rather than argued about.

**The budget** [VERIFIED, `stage7_depth.md` §2, anim run]: 25 games,
7,920 s each, **all 25 end `gave_up` on wall clock**; 940 `analyze()` turns
(37.6/game); 1,358 model responses (54.3/game); **586 executing turns
(23.4/game)**; 198,233 s total, so **146 s per model call**. Levels that clear
take a median of **8** turns; levels that never clear take **20**.

Because every game already consumes its entire clock, **any added call does not
extend the run — it displaces turns.**

| component | extra calls/game | extra seconds/game | share of the 7,920 s clock | turns/game after | verdict |
|---|---:|---:|---:|---:|---|
| verifier/critic on each executing turn | +23.4 | +3,416 | **43%** | 37.6 → ~21 | **rejected** |
| self-consistency, k=3, on executing turns | +46.8 | +6,832 | **86%** | 37.6 → ~5 | **rejected** |
| prompt ensemble, 2 prompts, every turn | +37.6 | +5,490 | **69%** | 37.6 → ~12 | **rejected** |
| **AVO supervisor** | **0** | **0** | **0%** | 37.6 | **viable** |
| **PROLONG programmatic memory** | **0** | **0** | **0%** | 37.6 | **viable** |

Losing 16 turns per game is **two whole median cleared-level budgets**, on a
metric where `w_l = l` makes depth worth 3.8× breadth. A verifier would have to
raise the per-turn success rate by more than 40% just to break even, and this
project has no evidence any prompt-level change moves anything by 40%.

**Retrieval over prior games is rejected for a second, independent reason.**
The scored set is ~110 games none of which we have ever seen, and this repo's
single most replicated finding — 13 interventions, 12 failures across Stage 6 —
is that signal learned on the 25 public games does not transfer to novel ones.
A retrieval corpus built from the public 25 (and such corpora exist and are
accessible: `johnlussier/arc-agi-3-replay-bank-payload`,
`jihangli1121/arc-agi-3-replays-v1`, `justforgags/arc3-sft-trajectories`,
`magicsword001/arc-agi-3-game-rules`,
`karnakbaevarthur/arc-agi-3-all-tasks-explanation`) would be retrieving from
the one distribution already shown not to generalise. **PROLONG is the
within-game form of the same idea, and that is the form that has a mechanism.**

---

## 6. Leaderboard and field intelligence (tasks 4 and 5)

### 6.1 The top of the board [VERIFIED, 2026-09-20]

| rank | score | team |
|---:|---:|---|
| 1 | **18.81** | Tufa Labs (141 subs) |
| 2 | 13.12 | Daniel Franzen |
| 3 | 11.64 | Matija Ludvig & Zhongwei Wang |
| 4 | 11.54 | Lord Han Solo |
| 5 | **11.04** | **NVARC3** |
| 6 | 8.72 | Tong Hui Kang |
| 7 | 8.68 | Ebi |
| 8 | 8.21 | Third Intelligence |
| 9 | 7.51 | mostik.ai |
| 10 | 7.37 | Fususu (`phuongncn`) |

**NVARC3 at rank 5 is NVIDIA's AVO team.** Their 100.00 is on the *public* 25
with Claude Opus 5; on the hidden set, inside Kaggle's 9 h / one-GPU envelope
with a 125 B open model, the same architecture is at **11.04**. That is the
honest calibration for how much of AVO's headline transfers, and it is still
~3× us.

### 6.2 What the 6–9 band is doing that we are not

Two things are visible, and neither is a knob:

1. **They publish their own experiment series and iterate on components.**
   `ataraxian` (rank 19, 6.23) has 16 published prompt-subsystem versions;
   `sonphamorg` (rank 16, 6.49) rebuilt the entire serving package from the
   model revision up; `phuongncn` (rank 10, 7.37) publishes source
   reproductions. They are changing subsystems, repeatedly, exactly as this
   project's own history says works.
2. **Nothing in the 6–9 band is a public notebook we could fork.** Of 1,074
   public competition notebooks, the highest-voted current-generation ones
   (`wuliao0/duck-qwen3-8-anim-base` 250 votes,
   `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp` 169,
   `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b` 284) are the lineage
   we already run. **Forking has been exhausted; component grafting has not.**

### 6.3 Nothing new has shipped since 2026-09-15

A dedicated sweep of `datasets/list` sorted by `updated` across 14 terms finds
exactly **three** ARC-3-relevant datasets updated on or after 2026-09-10
[VERIFIED]:

- `tanapatriewruja/taaf-src-v10-persistence` (2026-09-12) — lineage `cf7d448`,
  no anim, no noop guard, budgets configured for a 3,600 s/5-way run; author
  rank 602 / 2.79. Low prior.
- `samrishb/arc-agi-3-samrish-skills-bundle` (2026-09-17) — **not Duck-family
  at all**: a separate hand-written solver (`arc3x` with `maze_solver`,
  `combat_solver`, `patrol_solver`, `debate`, `context_compactor`) plus a
  `world_model_lab`. Nothing in it grafts onto our harness; author rank 271 /
  3.62. Its `ARCHITECTURE.md` is a genuinely good read on the public/hidden
  asymmetry but is not a component.
- one irrelevant match.

**No new solver bundle, serving stack or notebook of consequence has appeared
since 2026-09-12.** The AVO bundle (2026-09-01) is the newest thing worth
having and it is nineteen days old and unused by anyone on the NVFP4 stack.

### 6.4 The paper track since 2026-09-15 — nothing actionable

[secondary sources]

- **NVIDIA AVO** ([blog](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/))
  — 100.00 RHAE / 183 levels / 6,624 actions with Claude Opus 5. **No code, no
  ablations.** Tufa's `inference/avo/` is the only implementation we can run,
  and it is a reimplementation.
- **Symbolica AI "Arcgentica"** ([blog](https://www.symbolica.ai/blog/arc-agi-3),
  [code](https://github.com/symbolica-ai/arcgentica)) — orchestrator/subagent
  harness where subagents return compressed summaries to constrain context
  growth; 36.08% public-25 at $1,005 of API spend. **Not runnable here**: it is
  built on the Agentica SDK against hosted API models, and a Kaggle rerun has
  no internet. The *idea* (compressed summaries instead of raw history) is what
  PROLONG implements locally.
- **"Explore Before You Solve: The Speed–Depth Trade-off in Epistemic Agents
  for ARC-AGI-3"** ([arXiv 2605.25931](https://arxiv.org/pdf/2605.25931), AERA)
  — an ARC Prize 2026 paper-track entry, EXPLORE-before-PLAN. Its most useful
  claim for us is negative and matches our own record: it argues all 25 public
  games are reachable by non-intelligent strategies and the public set "cannot
  discriminate intelligent exploration from trivial heuristics" — an
  independent restatement of `stage7_noise_floor.md`.
- `arxiv.org/list/cs.AI/recent` on 2026-09-20 shows **no** new ARC-AGI-3 paper
  [VERIFIED].
- `naylinnaunghood/htba-arc-agi3-source` (a paper-track writeup + notebook) is
  accessible but its author does not appear in the scoring band of interest.

---

## 7. What is *not* available — stated plainly

- **No stronger model.** Reconfirmed, not re-derived: the sealed bundle pins
  419 per-file SHA-256s to revision `7b719225…`, and of 409 enumerated Kaggle
  models nothing under 95 GiB beats Qwen3.8-Flash-Next on coding capability
  (`stage7_model_search.md` §3–4). Candidate 4 unseals the *serving*, not the
  *model choice*.
- **No further solver bundle worth having.** Thirteen lineages exist, two are
  supersets of what we run, and the other eleven each *remove* animation
  awareness. That list is complete as of 2026-09-20 across 551 enumerated
  datasets.
- **No runnable frontier harness.** Arcgentica and AVO-proper both require
  hosted frontier models; the rerun has no internet.
- **No in-budget verifier, self-consistency, or ensemble.** §5 — the clock, not
  taste, excludes them.

**If candidates 1–3 all fail, the honest conclusion is that the component well
is dry** and the remaining moves are (a) resubmitting the anim arm, which is
worth roughly the difference between its mean 3.53 and its observed max 3.79 as
a max-of-n statistic, and (b) candidate 4's much larger rebuild. Recording that
in advance so a future session does not rediscover it as a surprise.

---

## 8. Recommended order

1. **Build the AVO graft** off `kaggle_submission_duck_nvfp4_anim/` with the
   §3.4 four-line change and the pinned mount `raist321/taaf-avo-v27-bundle`.
   Keep the per-cell sha256 assertions; add
   `assert bm.solver.avo_agent is True`. **Fix the 54000 → 32400 budget trap.**
2. **Free push, Arm A (`ARC3_AVO_PHASED_LOOP=0`)** — catastrophe detection and
   *counted* telemetry only: does it boot, does the run produce
   `avo_interventions > 0`, and what is the armed-turn conversion rate against
   the pre-registered null of **32.0% (164/513)**. **Ignore the score.**
3. **Submit Arm A** if the telemetry is sane. n≥3 before any claim, per the
   measurement rule.
4. **Arm B (phased loop on)** and **`MULTIMODAL_GRID_LINES=1`** as the next two
   slots — both are one env var on the same mount, so they are nearly free to
   prepare.
5. **PROLONG-eval** as the fourth arm if AVO does not clear.
6. Candidates 4 and 5 only after that.

---

## Appendix A — integration preconditions, checked rather than assumed

All [VERIFIED] against the downloaded `raist321/taaf-avo-v27-bundle` /
`jakobbrggen/taaf-kaggle-source` v27 tree:

| precondition | required by | result |
|---|---|---|
| `taaf-kaggle-bundle.json` → `benchmark_label` | cell 7 `_find_bundle_dir` | **`"avo-kaggle"`** (distinct from the serving bundle's `"duck-harness-kaggle"`, so `assert BUNDLE_DIR != ANIM_BUNDLE_DIR` still holds) |
| `src/ARC3-Inference/inference/utils/animation.py` exists | cell 7 assert | **present** |
| no `serving_setup.py` in the solver bundle | cell 7 label disambiguation | **absent**, as with anim |
| `bm.solver.animation_awareness is True and hard_noop_guard is True` | cell 11 assert | **both True** |
| `bm.solver.max_runtime_s_per_game` | cell 13 pin (7920.0) | **already 7920.0** |
| `target.max_runtime_s` | cell 13 assert (32400.0) | **54000.0 — MUST BE SET, NOT ASSERTED** |
| the bundle's own `setup_commands.json` | ignored by our notebook | points at `driessmit1/arc3-vllm-h100-wheelhouse-v3` + `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`. **The anim bundle carries an equivalent FP8 setup command and our graft already ignores it** — cell 7 resolves the serving bundle by the `duck-harness-kaggle` label and runs keithtyser's command instead. No new risk. |
| `inference/avo/` present, `avo_agent=True` in the pickle | the point of the swap | **both** |

The one item in that table that would break a run is the `max_runtime_s`
54000 → 32400 line. Everything else passes as written.
