# Stage 7 — Grafting the AVO solver bundle onto the sealed NVFP4 chassis

Date: 2026-09-19. Branch: `stage7-avo`. Kernel
`calamitychasm/arc3-duck-nvfp4-avo` (free push, no submission).

**No competition submission was made by this investigation, and none will be.**
Everything here is read-only Kaggle API calls, public bundle source, local
computation over both, and one free `kaggle kernels push`. The daily slot was
untouched.

Labels follow this repo's convention. **[VERIFIED]** = read directly from a
primary artifact (a Kaggle API response, a downloaded bundle's own source or
pickle, a run's own `benchmark.json` / `transcripts/` / `artifacts/`, or a local
computation over them). **[INFERRED]** = reasoning over verified facts.

Reproduce every number here with:

```
venv/Scripts/python.exe scripts/_build_duck_nvfp4_avo.py --both --check
venv/Scripts/python.exe scripts/diff_solver_bundles.py <anim-bundle> <avo-bundle>
venv/Scripts/python.exe scripts/summarize_avo_run.py avo=<dir> anim=<dir> \
    --json experiments/stage7_avo_data.json
venv/Scripts/python.exe scripts/analyze_dead_turns.py anim=<dir> avo=<dir>
```

`<dir>` is an unpacked `kaggle kernels output` directory; `<*-bundle>` an
unpacked solver dataset.

Artifacts in `experiments/stage7_avo_artifacts/`:

| file | what it is |
|---|---|
| `bundle_drift_anim_to_avo.txt` | full source/pickle/target diff of the two bundles |
| `avo_arm1_telemetry.txt` | `summarize_avo_run.py` output, arm 1 against the incumbent |

---

## 0. TL;DR

*(filled in from the free run — see §5)*

---

## 1. What AVO is, and why it was the one component that fitted

`stage7_components.md` enumerated 551 Kaggle datasets, resolved them into **13
distinct TAAF solver lineages**, and found that exactly **two** are strict
supersets of the configuration that currently scores 3.79. The AVO bundle
(`ARC3-Inference 74ff3df`, branch `experiment/avo-v2`, 2026-09-01) is the first.

It adds `inference/avo/` — 733 lines across six files, subclassing `ToolAgent`
rather than replacing it [VERIFIED, source]:

1. **`AvoMemory`** (159 lines). `ToolAgent._summarized_knowledge` is cleared
   whenever the session's runtime dir changes — i.e. **at every level boundary,
   which is exactly where the knowledge is worth the most**. AVO persists the
   same content to disk keyed by game and reloads it, with atomic writes
   (`os.replace`) because a Kaggle kernel is killed by wall clock, not a clean
   shutdown. Both `load` and `save` are exception-guarded.
2. **`Supervisor`** (163 lines). Two triggers: `barren_turns >= 3`, which yields
   to frame novelty, and `unrewarded_turns >= 12`, which ignores novelty
   entirely — with a 3-step escalation to a HARD REDIRECT that then restarts the
   ladder. The second trigger exists because the first alone did not work, and
   the source says so in numbers from their own 25-game run: *"the twelve games
   that finished on a score of zero drew 30 interventions across 767 turns, and
   the thirteen games that scored drew 33 across 747 — the supervisor could not
   tell the two apart"*. **It costs zero extra model calls**: `directive()`
   returns a string appended to the prompt the turn already sends.
3. **Phased loop** (`prompts.py`, 76 lines). INSPECT / PLAN / IMPLEMENT /
   EVALUATE rotate **on turn index, not on model self-report**, *"because a model
   that is stuck is exactly the one that will claim it is evaluating"*.
4. **Exploit deadline.** At `0.6 × max_runtime_s_per_game` = **4,752 s** of each
   game's 7,920 s, AVO stops building a world model and plays its best policy.
   The settings file's own comment: *"Without this the arm's most likely failure
   is an elegant world model and no score."*

**Zero extra model calls is the whole reason this component and not another.**
`stage7_components.md` §5 costed the alternatives against the measured clock —
146 s per model call, every game already consuming its entire 7,920 s, so an
added call does not extend a run, it *displaces a turn*:

| component | extra calls/game | share of the clock | verdict |
|---|---:|---:|---|
| verifier/critic on each executing turn | +23.4 | 43% | rejected |
| self-consistency, k=3 | +46.8 | 86% | rejected |
| prompt ensemble, 2 prompts | +37.6 | 69% | rejected |
| **AVO supervisor** | **0** | **0%** | viable |

### Honest prior, stated before any result

The only public run of this bundle — `yocybercode/thui-avo-v0` — scored **4.32**
on public-25. That is **not** evidence for or against it here:

- it ran on the **FP8-27B** chassis, which `stage7_model_search.md` measured at a
  ~3× local handicap against NVFP4;
- it was **token-capped** (`tokens≈119,000` on 20 of 25 games), a constraint our
  stack does not have, and one game took **0 actions**;
- its matched control's log has **expired** (800 characters, no `[finished]`
  lines), so the A/B cannot be recovered [VERIFIED, `stage7_components.md` §3.3];
- and a single public-25 mean carries **SE ±2.46**, which swallows the gap.

Weak revealed-preference counter-evidence, recorded rather than buried: the team
that ran it has published five subsequent full-25 kernels and **all five are
anim-based, not AVO** [VERIFIED, their kernel list]. They appear to have moved
on. They never ran it on NVFP4 either.

**Nobody has run AVO on the sealed NVFP4 appliance.** That is precisely the gap
the anim graft filled when it was worth +25%. This is a genuine unknown, not a
predicted win.

---

## 2. The two traps, and how each was handled

### Trap 1 — `deploy_target.pkl` carries a 15-hour budget

[VERIFIED, by unpickling both bundles' targets]

| | anim (incumbent) | AVO |
|---|---:|---:|
| `target.max_runtime_s` | **32,400.0** (9 h) | **54,000.0** (15 h) |
| `bm.solver.max_runtime_s_per_game` | 7,920.0 | 7,920.0 |

The inherited cell 13 **asserts** 32,400.0:

```python
if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
    raise RuntimeError(...)
```

Left alone, that raises and the run dies at setup. Simply deleting the check is
worse: the notebook's soft deadline is computed from this value
(`soft_end = start + budget - 600`, cell 17), so a 54,000 inherited budget means
the graceful-exit deadline lands **six hours past Kaggle's 9 h hard cap** and the
kernel is killed mid-run instead of finishing and writing `score.json`.

**Handled by SETTING, not asserting** — and still failing loudly on drift:

```python
_inherited_max_runtime_s = float(getattr(target, 'max_runtime_s', 0.0) or 0.0)
if _inherited_max_runtime_s not in (32400.0, 54000.0):
    raise RuntimeError(...)          # a third value means the pinned bundle moved
target.max_runtime_s = 32400.0
print(f'AVO_BUDGET_OVERRIDE inherited_s={...} applied_s={...} '
      f'per_game_s={...} exploit_deadline_s={...}', flush=True)
```

`max_runtime_s_per_game` is untouched at 7,920.0, so AVO's exploit deadline still
lands at 4,752 s where the settings file intends it.

### Trap 2 — the phased loop is a second variable

AVO's `INSPECT` — phase 1 of 4 — opens [VERIFIED, `inference/avo/prompts.py`]:

> Do not act this turn unless the situation is already unambiguous.

One turn in four is *instructed not to act*, on top of a measured **37.7%**
dead-turn rate that `stage7_depth.md` identified as the dominant depth sink
(46.1% of the entire wall clock). Our own dose–response says removing
deliberation hurts (commit floor +48% actions / −10% score; no-thinking +203%
actions / 0.64) — but that does **not** establish that adding more helps. There
may be an optimum, and this is a second variable.

**Handled by pinning `ARC3_AVO_PHASED_LOOP=0` for arm 1**, which the bundle
ships precisely as its memory+supervisor-only ablation [VERIFIED,
`avo/settings.py`: *"Off gives a memory+supervisor-only ablation against the same
base agent"*]. Verified in the notebook through the solver's own call rather
than through `os.environ`:

```python
_avo = _avo_settings_mod.AvoSettings.from_env()   # the call _make_analyzer makes
assert _avo.phased_loop is False, _avo
assert (_avo.stagnation_turns, _avo.unrewarded_turns,
        _avo.escalation_interventions) == (3, 12, 3), _avo
assert _avo.exploit_deadline == 0.6, _avo
assert (_avo.max_facts, _avo.max_rules, _avo.max_failures) == (24, 16, 16), _avo
```

The last three lines are what make "one variable" a checked claim rather than an
intention: every other AVO knob is asserted to be the shipped default.

### Every other Appendix A precondition, re-checked rather than assumed

Verified locally by unpickling the real downloaded bundle, not inferred
[VERIFIED]:

| precondition | required by | result |
|---|---|---|
| `benchmark_label` | cell 7 `_find_bundle_dir` | `avo-kaggle` ✓ (≠ `duck-harness-kaggle`, so `BUNDLE_DIR != ANIM_BUNDLE_DIR` holds) |
| `inference/utils/animation.py` exists | cell 7 assert | present ✓ |
| no `serving_setup.py` in the solver bundle | cell 7 disambiguation | absent ✓ |
| `animation_awareness is True` | cell 11 | True ✓ |
| `hard_noop_guard is True` | cell 11 | True ✓ |
| `type(bm.solver).__module__` | cell 11 | `inference.framework.solver` ✓ |
| `avo_agent is True` | cell 11 (**new** assert) | True ✓ |
| `max_runtime_s_per_game` | cell 13 | 7,920.0 ✓ |
| `target.max_runtime_s` | cell 13 | **54,000.0 — trap 1** |

### The pin, re-verified today

`jakobbrggen/taaf-kaggle-source` is a **rolling** dataset. We mount
`raist321/taaf-avo-v27-bundle`, a frozen v1 re-host. This project has already
lost a real submission to exactly this class of drift (`checkpoints/` silently
holding a fold-1 experimental checkpoint through submission `55843700`), so the
claim was re-checked rather than inherited: a fresh download of **both** on
2026-09-19 diffs clean — **no differences other than raist321's own added
`README.md`** [VERIFIED].

---

## 3. A third thing, which the research did not find: the swap is not one variable

`stage7_components.md` §2.2 established "strict superset" by checking **which
files exist**. That is necessary and not sufficient. The two bundles are **3.5
weeks of upstream commits apart** (`9158303` 2026-08-07 → `74ff3df` 2026-09-01),
and `scripts/diff_solver_bundles.py` shows six shared `.py` files changed
underneath [VERIFIED, `bundle_drift_anim_to_avo.txt`]:

| file | changed lines | what it is |
|---|---:|---|
| `inference/framework/solver.py` | 242 | AVO wiring, `animation_retrieval` field, `effective_flags()` |
| `inference/agent/tool_agent.py` | **198** | **see below — this one matters** |
| `inference/utils/animation.py` | **85** | **see below** |
| `inference/agent/vision_context.py` | 64 | additive: `MULTIMODAL_GRID_LINES`, vision-token telemetry |
| `inference/agent/prompts.py` | **9** | **see below** |
| `inference/framework/kaggle.py` | 8 | deploy-template plumbing we never execute |

### The AVO arm's model cannot call `animation()`

`tool_agent.py:1913` [VERIFIED, source]:

```python
animation_handler=_handle_animation if self._animation_retrieval_enabled else None,
```

`animation_retrieval` is a **new field, default False**, and the pickle carries
it explicitly False. In the **incumbent**, the field does not exist at all —
`grep -c animation_retrieval` on the anim bundle's `solver.py` returns **0** —
and the `animation()` sandbox function is installed unconditionally.

`animation.py` additionally **deletes** `should_suggest_animation()`,
`animation_hint_text()` and the five `ANIMATION_HINT_*` thresholds: the
incumbent's *"proactively suggest animation retrieval when stuck"* behaviour,
which is literally the anim branch's own HEAD commit message. It is replaced by
a static `worth_inspecting` flag (≥200 transient pixels or ≥25% bbox area) in
metadata the model already receives.

And `prompts.py` compresses the animation block from **seven bullets to two**.

**Why this is not a footnote.** Animation awareness is this project's single
largest measured win (+25%, Welch t=3.55, p<0.05), and our incumbent measured it
with retrieval **on**. Upstream's own comment says retrieval *"works and the
model reaches for it unprompted in 64% of calls, but across Experiments 3 and 4
it bought no score, so we do not pay for it by default"* — that is **their**
measurement on **their** chassis, and this project's whole recent history is
that another team's measurement does not transfer to the NVFP4 appliance.

**So arm 1 is honestly "AVO added *and* animation retrieval removed", not
"anim + AVO".** A bundle swap is a bundle-level comparison; that is the shape
the component search chose, and it is still the right first test. But the
telemetry below has to be read with the removal in mind, and a negative result
cannot be attributed to AVO alone.

**The separation is one env var**, recorded here and not run:
`ARC3_ANIMATION_RETRIEVAL=1` on the same mount restores the tool (though not the
deleted hint system). That is the obvious arm 3 if arm 1 underperforms.

### What was checked and found inert

Not everything that differs is a confound. These were checked rather than
assumed [VERIFIED]:

| pickled difference | why it does not matter |
|---|---|
| `n_passes` 4 → 1 | cell 17 sets `bm.n_passes = 1` unconditionally |
| `bm.games` differs | cell 17 **replaces** the list entirely and asserts the 25 public ids match `PUBLIC_GAME_IDS` exactly, in order |
| `game_weights` | cell 17 sets `bm.game_weights = None` |
| `kaggle_model_dataset_source`, `kaggle_served_model_name` (both FP8) | our notebook runs **keithtyser's** `setup_commands.json` from `BUNDLE_DIR`; the solver bundle's is never read. The incumbent carries an equally wrong FP8 string and is equally unaffected |
| `local_server_repo_dir`, deploy-target `kernel_*` / `dataset_ref` | jakob's own deployment tooling; unused here |
| `vision_context.py`'s 64 new lines | gated on `MULTIMODAL_GRID_LINES`, which defaults off |

---

## 4. What was built

`scripts/_build_duck_nvfp4_avo.py`, patterned on the same patch discipline as
the anim graft and the depth arms: the 18 inherited cells are sha256-recorded
before the patch and re-checked after, and the build **refuses** if any cell
other than the four named ones differs.

```
built kaggle_submission_duck_nvfp4_avo/notebook/arc3-duck-nvfp4-avo.ipynb
  base      arc3-duck-nvfp4-anim.ipynb sha256=4fdd395c6974240d (18 cells, all asserted)
  patched   cells [0, 7, 11, 13]; 14 inherited cells byte-identical
  inserted  1 markdown + 1 code cell after index 13 (PUBLIC25_SETTINGS)
  mount     jakobbrggen/taaf-kaggle-source-anim-20260807-anim
            -> raist321/taaf-avo-v27-bundle
  arm       ARC3_AVO_PHASED_LOOP=0
```

Changed, exhaustively:

* **cell 0** — attribution header (ours already; rewritten for AVO).
* **cell 7** — the mount ref, and `_find_bundle_dir("anim-20260807-anim")` →
  `_find_bundle_dir("avo-kaggle")`. `ANIM_BUNDLE_DIR` **keeps its name** so that
  cell 9 — the graft's teeth, holding the `sys.path` substitution and the
  `LOCAL_ANALYZER_*` overrides — stays byte-identical. The inherited
  `animation.py` assert two lines below is what proves the new bundle is still
  the anim superset, and it passes untouched.
* **cell 11** — `bm.label` assert, plus the new `avo_agent is True` assert.
* **cell 13** — trap 1.
* **two inserted cells** — trap 2.

Cells 1–6, 8, 9, 10, 12, 14–17 are asserted byte-identical. That includes cell 9
and cell 17, so the serving profile, `LOCAL_ANALYZER_SEED=20260825`,
`LOCAL_ANALYZER_YIELD_SECONDS=180`, `concurrency=28`, the 600 s gateway wait, the
per-game 7,920 s budget and the run loop are the incumbent's exactly.

**Arm 2** (`kaggle_submission_duck_nvfp4_avo_phased/`,
`calamitychasm/arc3-duck-nvfp4-avo-pl`) is built and **not pushed**. Its only
difference from arm 1 is one env var and its assertion [VERIFIED, by diffing the
two built notebooks]:

```
-os.environ["ARC3_AVO_PHASED_LOOP"] = "0"        -assert _avo.phased_loop is False
+os.environ["ARC3_AVO_PHASED_LOOP"] = "1"        +assert _avo.phased_loop is True
```

### Rehearsed before spending the run

Every cell-7/11/13/15 assertion was executed locally against the real downloaded
bundle — the pickles were genuinely unpickled, not inspected by opcode — and all
pass [VERIFIED]. `AvoSettings.from_env()` reads `phased_loop=False` under
`ARC3_AVO_PHASED_LOOP=0` and `True` under `=1`;
`bm.solver._avo_game_budget_seconds()` returns 7,920.0, so the exploit deadline
is 4,752.0 s.

---

## 5. Arm 1 telemetry

*(pending the free run)*

### The measurement rule, restated before any number

**The public-25 score cannot rank arms.** `stage7_noise_floor.md` puts the SE of
a single public-25 mean at **±2.46** — wider than any effect this project has
ever measured, and wider than the entire gap between our best and worst real
submissions. The mean is reported below for comparability only.
`summarize_avo_run.py` prints it labelled `CANNOT RANK` in the output itself.

What *can* be compared from one run is **counts**. Two runs of the same 25 games
in the same order either take different numbers of turns, execute different
numbers of actions, and clear different levels, or they do not. Those are
noise-free in the sense that matters: they are not estimates.

### Incumbent baseline, recomputed rather than quoted

Recomputed from `arc3-duck-nvfp4-anim`'s own kernel output with the tooling on
this branch, and it reproduces the published figures exactly [VERIFIED]:

| | anim (incumbent) |
|---|---:|
| turns (`analyze()` calls) | 940 |
| executed / dead | 586 / **354 (37.7%)** |
| model calls | 1,358 (54.3/game) |
| actions | 2,615 (104.6/game) |
| levels | 42 / 183 |
| games scoring ≥ 1 | 20 of 25 |
| public-25 mean | 9.97 |
| P(act \| ≥2 dead turns before) | **51/152 = 33.6%** |

One correction carried forward: `stage7_components.md` §3.5 pre-registered the
falsifier's null as "32.0% (164/513)". That figure **pools three sibling runs**.
The like-for-like comparator for a single AVO run is the incumbent's own
**33.6% (51/152)**, and that is what is used below.

---

## 6. Recommendation

*(pending the free run)*
