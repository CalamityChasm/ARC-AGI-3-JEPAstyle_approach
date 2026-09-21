# Stage 7 — Advisory HUD annotation on the model's primary board view

Date: 2026-09-21. Branch: `stage7-hud-perception`. Kernel
`calamitychasm/arc3-duck-nvfp4-hud`, dataset `calamitychasm/taaf-anim-hud-v1`
(free push, no submission).

**No competition submission was made by this investigation, and none will be.**
Everything here is local computation over the champion run's own artifacts, a
patched copy of an MIT-licensed public bundle, one `kaggle datasets create` and
one free `kaggle kernels push`. The daily submission slot was untouched.

Labels follow this repo's convention. **[VERIFIED]** = read directly from a
primary artifact (a Kaggle API response, the downloaded bundle's own source, a
run's own `benchmark.json` / `transcripts/` / `artifacts/`, or a local
computation over them). **[INFERRED]** = reasoning over verified facts.

Reproduce every number here with:

```
venv/Scripts/python.exe scripts/_build_hud_bundle.py --bundle <pristine> --out <staging>
venv/Scripts/python.exe scripts/_build_duck_nvfp4_hud.py --check
venv/Scripts/python.exe scripts/diff_solver_bundles.py <pristine> <staging>
venv/Scripts/python.exe scripts/count_hud_mentions.py anim=<dir> hud=<dir> \
    --json experiments/stage7_hud_perception_data.json
venv/Scripts/python.exe scripts/summarize_avo_run.py hud=<dir> anim=<dir>
venv/Scripts/python.exe -m pytest tests/test_hud_arm.py -q
```

Artifacts in `experiments/stage7_hud_perception_artifacts/`:

| file | what it is |
|---|---|
| `bundle_source_diff.patch` | the complete unified diff of the four patched files |
| `bundle_drift_anim_to_hud.txt` | `diff_solver_bundles.py` over source, pickles and deploy target |
| `bundle_patch_manifest.json` | sha256 before/after per file, and the unchanged-file count |
| `hud_baseline_anim.txt` | `count_hud_mentions.py` on the champion run |

---

## 0. TL;DR

**Verdict: PENDING.** The free run is queued. Section 6 registers the
falsifiers; nothing in this document is a result.

What is already settled:

* The detector fires on **22 of 25** public games, flags **0.78–3.12%** of
  cells, and **provably cannot reach the board interior** — every flagged cell
  is within 2 of a frame edge, by construction and checked on 55 real boards
  [VERIFIED].
* The gate is real: with `ARC3_HUD_ANNOTATION` unset, `segment_layer`'s return
  value is equal to the unpatched bundle's key for key on every one of those
  boards, and every prompt string is byte-identical [VERIFIED].
* **The brief's baseline of "4,608 HUD mentions across all 25 games" is ~85%
  harness boilerplate.** Only **718** are model-generated, in **24 of 25**
  games. §5 has the split. The falsifier is registered on 718, not 4,608.
* **A fifth file had to be patched that nothing in the design anticipated.**
  The Python-tool sandbox is a subprocess with an explicit *allowlist*
  environment, and it is the only caller of `segment_layer`. Without one line
  of passthrough this arm would have shipped as a **silent no-op with the whole
  prompt change in place** — the exact failure shape that is hardest to detect
  after the fact. §4.2.

---

## 1. Why this arm, and why now

`stage7_avo.md` closed the previous lead and did it cleanly: the AVO arm passed
its pre-registered falsifier decisively (dead-turn rate 37.7% → 27.7%,
P(act | ≥2 dead before) 33.6% → 54.5%) and **still lost**, 42 levels → 35,
because the extra actions went into levels that were never cleared (59.5% of
actions vs 48.6%). Three mechanisms have now bought actions — commit floor,
no-thinking, AVO — and none bought score.

The conclusion recorded there: **action count is not the bottleneck. The next
lever must target whether a level is solvable at all — perception.** That is
also where Tufa's own writeup puts their weakness (*"context management and
perception"*), and where this project's measured 9–155 s analyzer timeouts live.

### The specific perception cost this arm attacks

Across the champion run's transcripts, the model repeatedly

* re-derives HUD geometry from scratch, turn after turn,
* misattributes HUD pixels to game objects, and
* spends whole turns establishing "did only the HUD change?".

Verbatim, from the run [VERIFIED]:

> "My white-object tracker accidentally matched the growing row-0 HUD bar
> (rows 0-0), so the 'RIGHT/DOWN moved it' reading was HUD noise."

> "that's the HUD, not a cell. I should exclude it: the HUD is at cols 52-63
> rows 0-11"

> "DOWN did nothing (only HUD changed)"

### What the base prompt already does, and why that matters

The incumbent's `VISUAL_GAME_ADDENDUM` **already warns about this, in prose**
[VERIFIED, `prompts.py`]:

> "In many games, a long horizontal or vertical line near an edge is a timer or
> remaining-steps bar... A common failure mode is to mistake a segmented edge
> bar for clickable puzzle pieces... DON'T DO THIS!"

That paragraph accounts for 2,820 of the raw "HUD" matches by itself (3 per
system prompt × 940 prompts). So this arm is **not** "tell the model HUD
exists". It is: *the advice is already there and is not enough, because
following it requires the model to redo a geometric computation every turn.*
The arm converts the prose into a computed field on the object the model
already reads.

### Why annotate and not mask

Masking is the obvious version and it is the wrong one. The detector catches
edge **bars**, not HUD **blocks** set away from the edges — a known, stated
limitation. A mask makes that limitation destructive: a wrong flag deletes a
playable object and the model cannot recover, because it never sees what was
removed. An annotation makes the same wrong flag cheap: the model sees
everything it saw before, plus a prior it can contradict from behaviour. The
prompt says so explicitly, and `tests/test_hud_arm.py::test_annotation_is_purely_additive`
is the mechanical proof that nothing is lost.

---

## 2. The change

`current_frame.segmentation` is the model's primary view of the board
[VERIFIED, `prompts.py`: *"Use `current_frame.segmentation` as your primary view
of the board"*; the raw numeric grid is deliberately not exposed]. It is
produced by `segment_layer()` in `inference/utils/segmentation.py`.

With `ARC3_HUD_ANNOTATION=1`, and only then:

* each node gains **`hud: bool`**;
* the returned dict gains **`hud_node_ids: [...]`**, the sorted flagged ids.

A node is flagged when it sits fully against a frame edge **and** is either

1. a **line** — aspect ratio ≥ 5:1 in the orientation that edge implies, or
2. one of **≥ 3 identical shapes** ("dots") on that same edge.

**Nothing is removed, masked, hidden, reordered or altered.**

### Provenance of the rules

Ported from `identify_status_bars_with_rule` and its helpers in Rudakov, Shock &
Cowley, [arXiv:2512.24156](https://arxiv.org/abs/2512.24156), MIT-licensed —
already in this repo at `ARC-AGI-3-Agents/agents/templates/graph_explorer_agent.py`
with its notice at `graph_explorer_THIRD_PARTY_LICENSE`. All three thresholds
(edge distance 3, aspect ratio 5, twin count 3) are theirs, unchanged. The
notice is carried into the published dataset as
`GRAPH_EXPLORER_THIRD_PARTY_LICENSE`.

### Two deviations from upstream, both recorded because they change behaviour

1. **Twins are matched by the bundle's translation-invariant `hash`**, not by
   upstream's `(area, is_rectangle, color)` triple. Equal hash implies equal
   normalized cell set and equal color, hence equal area and equal
   rectangle-ness — so hash-twins are a strict **subset** of upstream twins and
   the detector fires *less* often, never more. Safe direction for an advisory
   flag.
2. **Bounding boxes come from a node's `boundary`** (traced contour reduced to
   corner points) rather than from the cell set, because `boundary` is what a
   node exposes. Every bbox-extreme cell lies on the outer perimeter and on a
   straight run whose endpoints are corners, so this is exact —
   `test_boundary_bbox_matches_the_true_cell_bbox` checks it against an
   independent flood-fill bbox on **every node of all 55 fixture boards**,
   rather than leaving it as an argument.

One upstream quirk is **preserved rather than fixed**: the `left`/`top` tests
(`max < 3`) admit depths 0–2 while `right`/`bottom` (`min > extent - 3`) admit
only 0–1. That asymmetry is what was validated against real frames, and
"correcting" it would only make the detector fire more.

---

## 3. The safety argument, and how it is checked

The claim that makes an advisory flag acceptable is narrow: **the detector
cannot reach the board interior.**

It holds by construction. A node is flagged only if one of the four edge tests
passes, and each test constrains *every* cell of that node:

| test | condition | consequence for every cell |
|---|---|---|
| left | `max_col < 3` | `col ≤ 2` → depth ≤ 2 |
| right | `min_col > W-3` | `col ≥ W-2` → depth ≤ 1 |
| top | `max_row < 3` | `row ≤ 2` → depth ≤ 2 |
| bottom | `min_row > H-3` | `row ≥ H-2` → depth ≤ 1 |

A node flagged as a *twin* also has to be on one of the same edges, so the bound
is uniform over flagged nodes, not just group leaders.

Measured on 55 real boards from the champion run, all 25 games [VERIFIED]:

| quantity | value |
|---|---|
| games where the detector fires | **22 / 25** (`ls20`, `sb26`, `sk48` never fire) |
| boards where it fires | 50 / 55 |
| share of cells flagged | **0.78% – 3.12%** |
| max depth of any flagged cell from the nearest edge | **2** |

This **independently reproduces the go/no-go validation** that motivated the
arm (22/25, 0.8–3.1%, depth 0–2) through a completely different code path — the
bundle's own `segment_layer` with hash-based twins, rather than the repo's
`graph_explorer_agent.FrameProcessor` with area-based twins. Two
implementations agreeing on the same 25 games is worth more than either alone.

---

## 4. What was built

### 4.1 Four files patched, 71 byte-identical

The patch cannot be expressed in the notebook — it edits solver source that the
notebook only mounts. So the bundle is republished as
`calamitychasm/taaf-anim-hud-v1` (MIT permits it; attribution shipped inside).

`scripts/_build_hud_bundle.py` pins the **pre-patch** sha256 of each of the four
files, applies anchored replacements that must match **exactly once** or the
build refuses, `compile()`s every result, and sha256-checks that all other files
copied through unchanged.

`diff_solver_bundles.py` over the pristine and patched bundles [VERIFIED,
`bundle_drift_anim_to_hud.txt`]:

```
=== source files that differ (.py only) ===
  inference/utils/segmentation.py             143 changed lines
  inference/agent/prompts.py                   49 changed lines
  inference/agent/python_tool_sandbox.py       10 changed lines
  inference/agent/tool_agent.py                 5 changed lines

=== pickled solver fields ===      (only a fresh threading.Event identity)
=== deploy target fields ===       (none)
=== benchmark-level fields ===     (none)
```

The pickles are untouched, so **AVO's trap 1 does not apply here**: we inherit
the anim bundle's own `deploy_target.pkl` with `max_runtime_s = 32400.0`, cell
13 is byte-identical, and its assert passes unmodified. There is no 54,000 to
override — that was the AVO bundle's. Verified by diffing the deploy target
rather than assumed.

The detector lives in `arc3_hud/hud_detect.py` in this repo and is **spliced
verbatim** into `segmentation.py` between markers, so the code the tests
exercise and the code that ships cannot drift; `test_detector_source_is_spliced_verbatim`
asserts the exact substring relation.

### 4.2 The thing the design did not anticipate: the sandbox is a subprocess

`segment_layer` does not run in the notebook's process. `python_tool_sandbox.py`
splices its source into a bootstrap that runs as a **child process**, launched
with an explicit allowlist environment [VERIFIED, `python_tool_sandbox.py:427`]:

```python
def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp", "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }
```

`ARC3_HUD_ANNOTATION` is not on that list, so it can never reach the detector.
**The failure mode this would have produced is the worst available one**: every
prompt would describe `hud` and `hud_node_ids`, the notebook's own checks would
look green, the model would go looking for a field that is never there, and the
arm would read as "annotation tested, no effect" when nothing had been tested.
One line fixes it; the allowlist stays an allowlist.
`test_kernel_sets_the_flag_before_the_solver_import` and the notebook's own
`_sandbox_env()` assert both guard it.

### 4.3 Ordering is load-bearing

`prompts.py`'s constants and `tool_agent.py`'s `_PYTHON_TOOL_DESCRIPTION` are
built at **import** time, and cell 9 imports `inference`. So the flag is
exported in **cell 7**, not in a cell after the graft the way AVO's was — AVO's
flag is read per game inside the solver and could be set late; this one cannot.
A flag set after cell 9 would annotate the board while leaving the prompt
silent about a field the model was never told exists. That ordering is asserted
in the notebook (`_prompts_mod._HUD_ANNOTATION is True`) and in the suite.

### 4.4 One variable

`scripts/_build_duck_nvfp4_hud.py` records the sha256 of all 18 inherited cells
and refuses if any cell other than **0 (attribution header)** and **7 (mount +
flag)** differs. Cells 1–6, 8–17 are byte-identical, which includes **cell 9**
(the graft's teeth: the `sys.path` substitution, `LOCAL_ANALYZER_SEED=20260825`,
`LOCAL_ANALYZER_YIELD_SECONDS=180`), **cell 13** (`concurrency=28`,
`max_runtime_s_per_game=7920`, `analyzer_timeout=900`, the 32,400 s budget
check) and **cell 15** (the run loop, the 600 s gateway wait). `kernel-metadata.json`
changes only `id`, `title` and `dataset_sources[2]`; `model_sources`,
`docker_image`, `machine_shape` and mounts 0 and 1 are carried over.

**To be confirmed after the run** from `vllm-server-identity.json`, which is the
authoritative record of what was actually served — not `effective_flags.json`,
which carries inert bundle defaults that read like a model swap and are not.

---

## 5. The baseline, re-measured — and a correction to it

The brief for this arm put the HUD signal at **"4,608 mentions across all 25 of
25 games"**. Re-measuring produced a number close to that and a **materially
different meaning** [VERIFIED, `scripts/count_hud_mentions.py` over
`anim_out/transcripts/`]:

| transcript section | raw matches | blocks | model-generated? |
|---|---:|---:|---|
| SYSTEM PROMPT | **2,820** | 940 | no — the harness's own HUD paragraph |
| USER PROMPT | **1,051** | 943 | no |
| THINKING | 563 | 1,358 | **yes** |
| ASSISTANT | 92 | 737 | **yes** |
| TOOL CALL | 63 | 1,342 | **yes** |
| TOOL RESULT | 38 | 1,342 | sandbox stdout — excluded |
| MODEL RESPONSE META | 62 | 1,358 | duplicates TOOL CALL — excluded |
| ANALYZER STATUS | 0 | 940 | no |
| **raw total** | **4,689** | | |
| **model-generated** | **718** | | |

**Only 15.3% of the raw count is the model.** The rest is the harness quoting
its own prompt back into the transcript 940 times.

This matters for the falsifier and not just for tidiness: **this arm adds HUD
words to the prompt**, so a raw count would rise mechanically whatever the model
did. Registering the falsifier on 4,608 would have made it unfalsifiable in the
direction that matters.

Corrected baseline, and the numbers §6 registers against [VERIFIED]:

| quantity | anim (champion) |
|---|---:|
| HUD mentions, **model-generated only** | **718** |
| of which **derive geometry** (a row/col index or range on the same line) | **286** |
| mentions per model block | 0.209 |
| games with a model-generated mention | **24 of 25** (not 25/25) |
| references to the new `hud` field | 0 (it did not exist) |

"24 of 25" rather than "25 of 25" is the same correction: every game mentions
HUD *somewhere* because the system prompt does, in every game, by construction.

Outcome and cost baselines, unchanged from `stage7_avo.md` [VERIFIED,
`benchmark.json` + transcripts]:

| quantity | anim |
|---|---:|
| turns (`analyze()` calls) | 940 |
| dead turns | 354 → **dead-turn rate 37.7%** |
| model calls | 1,358 |
| actions | 2,615 |
| share of actions in never-cleared levels | 48.6% |
| **levels cleared** | **42 / 183** |
| games scoring ≥ 1 | 20 of 25 |
| public-25 mean | 9.97 |

---

## 6. Pre-registered falsifiers

**Registered before the run produced anything.** The kernel was pushed at
2026-09-21 and this section was written while it was still `QUEUED`.

### 6.1 Mechanism — did the annotation change how the model reasons?

Counts, from `scripts/count_hud_mentions.py`, over **model-generated blocks
only**:

| # | prediction | falsified if |
|---|---|---|
| M1 | **model-generated HUD mentions fall** from 718 | ≥ 718 |
| M2 | **geometry-deriving mentions fall** from 286 — the sharper one, since this is the specific work the field replaces | ≥ 286 |
| M3 | the model **actually reads the field**: ≥ 100 references to `hud_node_ids` / `n['hud']` across the run (baseline 0) | < 100 |

**M3 is the one that decides whether anything was tested at all.** If the model
never touches the field, M1/M2 movement is noise and the arm is untested rather
than refuted. M1 and M2 are *directional expectations, not hard falsifiers*:
stated honestly, mentions could rise simply because the model now has a named
thing to discuss. That is why M2 (deriving geometry) is separated from M1
(mentioning HUD at all), and why M3 is reported alongside both.

### 6.2 Catastrophe — did anything break?

| # | prediction | falsified if |
|---|---|---|
| C1 | the arm runs to completion | no `score.json`, or the run is killed on the 9 h cap |
| C2 | deliberation budget is unchanged | turns outside 940 ± 15%, or model calls outside 1,358 ± 15% |
| C3 | the served stack is identical | `vllm-server-identity.json` differs from the champion's in model, seqs, KV bytes or ctx |
| C4 | the annotation is live in the run | the log has no `HUD_ARM flag=1 ...` line |

### 6.3 Outcome — supporting evidence only

| # | prediction | note |
|---|---|---|
| O1 | **levels cleared ≥ 42** | the primary outcome, and it carries real run-to-run variance |
| O2 | share of actions in never-cleared levels ≤ 48.6% | the quantity AVO moved the wrong way |
| O3 | games scoring ≥ 1 above 20 of 25 | |

**The public-25 mean cannot rank arms.** A single free run's mean carries
**SE ±2.46** (`stage7_noise_floor.md`), measured from an accidental null arm in
which 16 games a mechanism provably never touched still moved sd 12.30 per game.
It is reported for comparability and nothing else.

**Levels cleared is an outcome, not proof.** It is a count rather than a sampled
score, so it is far better behaved than the mean — but 42 vs 35 in the AVO arm
was still one draw against one draw. Ranking happens on the hidden set, n ≥ 3
per arm, and this document does not licence a submission.

### 6.4 What a pass and a fail each mean

* **M3 fails** → the arm is untested. Fix the surfacing (the field exists and is
  documented; the model ignoring it is a prompt problem, not a detector
  problem), do not conclude anything about perception.
* **M3 passes, M2 passes, O1 fails** → the same shape as AVO: the mechanism
  worked and did not convert. That would be the second independent mechanism to
  do so, and it would be real evidence that *the model's reasoning cost is not
  what limits which levels are solvable* — a more valuable finding than a win.
* **M3 and O1 both pass** → a candidate worth hidden-set replication at n ≥ 3.

---

## 7. Where reality differed from the plan

Recorded plainly rather than worked around silently.

1. **The env gate could not reach the detector.** §4.2. A fifth file
   (`python_tool_sandbox.py`) had to be patched. Without it the arm was a silent
   no-op with a full prompt change in place.
2. **The baseline was ~85% boilerplate.** §5. 4,608 raw → **718**
   model-generated; "25 of 25 games" → **24 of 25**.
3. **The champion's own notebook is not on `master`.** `kaggle_submission_duck_nvfp4_anim/`,
   `scripts/summarize_avo_run.py`, `scripts/analyze_dead_turns.py` and
   `scripts/diff_solver_bundles.py` all live on `stage7-avo`. This branch is off
   `master` as instructed and brings those four in unchanged
   (`git checkout stage7-avo -- ...`), which is why the diff is larger than the
   arm itself.
4. **The test suite on `master` is 134, not 147.** 147 is `master` + the AVO
   branch's own `tests/test_avo_arm.py`. This branch is **167** (134 + 33 new).
5. **`summarize_avo_run.py` needed no generalization** — it already takes
   arbitrary `label=<dir>` pairs. Its AVO-specific counter rows simply print
   zero for a HUD run.
6. **`kaggle kernels push` needs `PYTHONUTF8=1` on this box.** CLI 2.2.3 reads
   the notebook with the system codec (cp1252) and the inherited anim notebook
   contains 12 non-ASCII characters, so the push dies with
   `'charmap' codec can't decode byte 0x9d`. New gotcha; it applies to every
   kernel in this lineage, not just this one.
7. **`kaggle datasets create` hit the documented Windows upload-path bug**
   (`[Errno 2]` on a mangled temp path for bare top-level files). The fix is the
   one already in `CLAUDE.md`: `mkdir` the exact directory the error names —
   which here was the whole staging path nested under
   `%LOCALAPPDATA%\Temp\.kaggle\uploads\C_\`, not just `C_\`.
8. **`__pycache__` nearly shipped.** Importing the staged tree for a smoke test
   left bytecode behind and grew the dataset 1.7 MB → 2.5 MB. A stale `.pyc`
   inside a mounted dataset is a silent way to run source that is not the source
   in the diff. The build script now excludes it and the staging dir was rebuilt
   clean (79 files, 0 `.pyc`).

---

## 8. Tests

`pytest tests/ -q` → **167 passed** (134 inherited + 33 new in
`tests/test_hud_arm.py`), ~55 s. The new tests run against **real boards from
the champion run** (`tests/fixtures/hud_frames.json.gz`, 55 boards, all 25
games, extracted by `scripts/_extract_hud_frames.py`), not synthetic grids.

What they pin:

* **the gate**: output identical to upstream when off, on every board; five
  truthy and six falsey spellings;
* **additivity**: strip `hud` / `hud_node_ids` and the result equals upstream
  exactly — same nodes, ids, order and adjacency;
* **firing**: exactly 22 of 25 games, pinned;
* **the safety invariant**: no flagged cell deeper than 2 from an edge, using an
  independent flood fill rather than the bundle's own components;
* **not just aspect**: a 40×1 bar with the exact HUD shape placed mid-board is
  not flagged;
* **bbox exactness**: `boundary`-derived bbox equals the true cell bbox on every
  node of every board;
* **the splice**: `hud_detect.py`'s body is a verbatim substring of the shipped
  file, and the patcher raises on a missing *or ambiguous* anchor;
* **the arm**: only cells 0 and 7 differ from the champion notebook, cells 9 and
  13 are byte-identical, the flag is set before the solver import, and the
  kernel mounts our bundle and no longer mounts upstream's;
* **prompt honesty**: the wording contains `HEURISTIC`, `MISSES`,
  `wrongly flag`, `Verify` and `Nothing is hidden`.

---

## 9. Result

*(to be filled in from the free run)*
