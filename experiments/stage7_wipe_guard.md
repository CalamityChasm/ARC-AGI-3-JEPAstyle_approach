# Stage 7 — The world-model wipe guard: the first additive mechanism

Date: 2026-09-16. Branch: `stage7-wipe-guard`. Kernel
`calamitychasm/arc3-duck-nvfp4-anim-wg` v1 (free push, no submission).

**No competition submission was made by this investigation, and none will be.**
Everything here is read-only analysis of completed runs, plus one free
`kaggle kernels push`. The daily slot was untouched.

Labels follow this repo's convention. **[VERIFIED]** = read directly from a
primary artifact (bundle source, a run's own artifacts, a Kaggle API response,
or a locally-run computation over them). **[INFERRED]** = reasoning over
verified facts, not directly observed.

Reproduce every number here with:

```
venv/Scripts/python.exe scripts/analyze_wipe_guard.py \
    baseline=<nvfp4-dir> anim=<anim-dir> wg=<wipeguard-dir> \
    --json experiments/stage7_wipe_guard_runs.json
venv/Scripts/python.exe scripts/measure_wipe_cost.py <anim-dir> <nvfp4-dir> \
    --json experiments/stage7_wipe_guard_cost.json
venv/Scripts/python.exe -m pytest tests/test_wipe_guard.py -q
venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim_wipeguard.py --check
```

---

## 0. The falsifier, stated before the result

Per the task brief, and fixed before the run was pushed:

> **If wipes are intercepted but levels-completed and the wasted-action
> fraction are unchanged, the mechanism does not work here and the line
> closes.**

Two additions made *before* seeing any outcome, both forced by measurements in
§2 that the brief could not have had:

- **The interception count to expect is 16, not ~74.** §2 shows why. A run that
  reports ~16 `WIPE_GUARD_KEPT` lines has fired correctly; one that reports ~74
  has a bug.
- **Because the target is 16 interceptions preserving ~2.7k characters across
  8 of 25 games, a null result on public-25 is only weak evidence against the
  mechanism** — it is also consistent with the mechanism working and being too
  small to see through this metric's noise. The write-up says which of those it
  is at the end, rather than collapsing them.

---

## 1. The wipe, with file:line evidence

### 1.1 Where it is

`jakobbrggen/taaf-kaggle-source-anim-20260807-anim`,
`src/ARC3-Inference/inference/agent/tool_agent.py` **lines 1343–1356**
[VERIFIED — file pulled from the dataset we mount, sha256
`856bf9b895d0ad8b959c8f828c7132b0e09eaa47f4c5cc6173785354090f8be7`]:

```python
    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""
```

### 1.2 Does the anim bundle wipe identically? Yes — verified, not assumed

The brief flagged that the anim bundle might differ. It does not, for this
method. The 14-line block above is **byte-identical** to `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`'s
own `tool_agent.py` lines **1113–1126** — the file the SOTA research quoted.
[VERIFIED — both files pulled and compared programmatically.]

The two files are *not* otherwise identical (anim 2,448 lines vs keith 2,063;
different sha256s; anim adds `noop_guard.py`, animation awareness, and pins
`ARC3-Inference 9158303` on `feature/animation-awareness` against keith's
`aa69123` on `add-kaggle-share-flag`). The wipe is simply one of the parts that
did not change.

### 1.3 Which fields die, and which one survives

Six of the seven fields in `_empty_world_model()` (`tool_agent.py:448`)
[VERIFIED]:

| field | on `game_over` |
|---|---|
| `world_model` | erased |
| `goal_model` | erased |
| `action_model` | erased |
| `recent_findings` | erased |
| `open_questions` | erased |
| `current_plan` | erased |
| **`cross_level_notes`** | **survives** |

`cross_level_notes` is the one field the prompt frames as durable across levels;
everything the agent has worked out about *this* level's mechanics is in the six
that die.

### 1.4 Why the game-over branch specifically is wrong

Three of the flags share one wipe, but only two of them mean the level changed:

- `level_transition` — new level, new mechanics. Wiping is right.
- `run_complete` — game finished. Wiping is right.
- `game_over` — the agent died. The harness **auto-RESETs into the same level**.

That last claim is verified directly in our own event streams, not inferred from
the code: across every `GAME_OVER` in both runs, the `level` field does **not**
revert, and the next action is a `RESET` at the same level. [VERIFIED]

```
29 action  Action 21  act#21 step 5  lvl 2  state NOT_FINISHED  disp RIGHT
30 action  Action 22  act#22 step 5  lvl 2  state GAME_OVER     disp RIGHT
32 action  Action 23  act#23 step 5  lvl 2  state NOT_FINISHED  disp RESET
```

So on a game over the six fields describe a level the agent is about to replay
with identical mechanics, and they are erased anyway. §4.3 of
`stage7_sota_research.md` makes the cost concrete: Thuitanium's published
death-blacklist table infers a hidden **30-action budget on `sp80` level 1**,
and the only way to learn a number like that is to die repeatedly and count —
which is exactly what this wipe makes impossible.

### 1.5 Where the wipe is called from

`tool_agent.py:1965-1968` [VERIFIED] — once per *executed step summary*, inside
`_run_python_tool`:

```python
        step_executed = any(bool(item.get("executed")) for item in action_results)
        if step_executed:
            self._last_step_summary = self._summarize_step_sequence(action_results)
            self._update_summarized_knowledge_from_step_summary()
```

`_summarize_step_sequence` (`tool_agent.py:1250`) ORs the three flags across
**every action in the sequence** the model issued in that one `python` call.
This matters for the guard's condition: a sequence that both completed a level
and died gets `level_transition=True`, and must wipe as upstream.

---

## 2. Sizing it — the "74 wipes" figure is a 2× over-count, and the wrong chassis

`scripts/analyze_wipe_guard.py`. Two corrections, both [VERIFIED]:

**(a) Events are not wipes.** The wipe fires once per executed *step summary*,
and one `python` call can execute several game actions. The right unit is
distinct `(game, analysis_step)` groups containing a qualifying event.

**(b) The event stream double-counts.** Every `type == "action"` row is mirrored
by a `type == "analysis"` row carrying the same `action_num` and the same
`state`. Counting rows by `action_num is not None` — which is what
`scripts/analyze_rhae_binding.py` does — therefore counts **4,970 rows against
`benchmark.json`'s 3,633 real actions**, and counts every game over twice:

```
baseline GAME_OVER by type: {'action': 37, 'analysis': 37}
anim     GAME_OVER by type: {'action': 16, 'analysis': 16}
```

**The 74 in `stage7_sota_research.md` §4.4 and in the task brief is 37 real game
overs, mirrored.** With the row filter applied, action totals reproduce
`benchmark.json` exactly (3,633 / 2,615), which is the check that this fix is
the right one.

### 2.1 What that leaves

| | nvfp4 baseline | **anim (incumbent)** |
|---|---:|---:|
| public-25 mean | 10.69 | 9.97 |
| actions | 3,633 | 2,615 |
| LLM calls | 1,339 | 1,358 |
| levels solved | 48 / 183 | 42 / 183 |
| wasted-action fraction | 50.9% | 49.3% |
| real game overs | 37 | **16** |
| level transitions | 48 | 42 |
| total wipes | 85 | 58 |
| **guardable wipes** | **37** | **16** |
| games with ≥1 guardable wipe | 10 / 25 | **8 / 25** |

**The guard's headroom on the chassis we actually run is 16 interceptions, not
~74 — 4.6× smaller than the brief assumed.** The anim solver ships
`hard_noop_guard=True` and dies less than half as often as the June duck.

Anim, per game [VERIFIED]:

| game | score | game overs | guardable wipes | levels | actions |
|---|---:|---:|---:|---:|---:|
| `sp80` | 0.00 | 6 | 6 | 0/6 | 215 |
| `tu93` | 21.46 | 3 | 3 | 4/9 | 100 |
| `bp35` | 0.85 | 2 | 2 | 1/9 | 45 |
| `dc22` | 0.00 | 1 | 1 | 0/6 | 97 |
| `sc25` | 0.00 | 1 | 1 | 0/6 | 94 |
| `su15` | 2.03 | 1 | 1 | 1/9 | 55 |
| `vc33` | 21.22 | 1 | 1 | 3/7 | 111 |
| `wa30` | 1.89 | 1 | 1 | 1/9 | 148 |
| other 17 | — | 0 | 0 | — | — |

Six of the sixteen are on `sp80` alone, the game whose score is 0.00.

### 2.2 What each wipe actually destroys

A wipe costs nothing if the fields were empty. `scripts/measure_wipe_cost.py`
reads the rendered `Working world model carried from earlier turns:` block out
of every turn's own prompt in the transcripts and measures it at each in-level
game over. [VERIFIED]

*(Anchoring note, because it changes the answer: `_summarized_knowledge_lines`
returns `[]` when every field is empty, so the block is **absent** on those
turns. Counting blocks positionally silently misaligns after the first empty
one — it resolved only 6 of anim's 16 wipes. Anchoring on the transcript's
`--- analysis_step=N |` turn header resolves 16/16 and 37/37.)*

| | anim | baseline |
|---|---:|---:|
| in-level game-over wipes | 16 | 37 |
| …landing on an already-empty world model | 6 | 22 |
| characters destroyed, total | **2,726** | 5,527 |
| median | 130 | 0 |
| max | 720 | 1,078 |

Per game on anim: `sp80` 0, 257, 117, 372, 0, 0 · `tu93` 0, 0, 720 · `bp35`
144, 57 · `dc22` 342 · `sc25` 290 · `su15` 263 · `vc33` 164 · `wa30` 0.

**The empty ones are ambiguous, not harmless.** Look at the baseline's `r11l`:
`251, 0, 223, 0, 0, 0, 0, 0, 0`. That is a death cascade, and the emptiness of
wipes 4–9 is *downstream of* wipes 1 and 3 — with the guard, there would have
been something to preserve each time. So the honest reading is: the guard's
*direct* first-order saving on the incumbent is ~2.7k characters, and its
second-order saving (breaking cascades) is unmeasurable from a control run. It
cannot be larger than the 16 interceptions allow either way.

**[INFERRED] This is a small intervention.** ~2.7k characters — roughly 700
tokens — spread across 16 moments in 8 of 25 games, against ~2,989 tokens per
turn and 940 turns. That is the honest prior going into the run, and it is
weaker than the brief's framing.

---

## 3. The guard

`scripts/wipe_guard_cell.py`, installed as one added cell. The behavioural
change is one early return:

```python
        if game_over and not level_tx and not run_done:
            n = _wg_bump("kept")
            print(f"WIPE_GUARD_KEPT n={n} level={...} action={...} session={...}", flush=True)
            return None
```

Level transitions and run completion fall through to the original method
unchanged; `cross_level_notes` is untouched by either path.

### 3.1 Safety

- **It can never kill a game.** The flag read is in a `try/except`, and so is
  the call to the original — so neither a malformed summary nor a broken
  knowledge dict can propagate out of bookkeeping into the dispatch path, which
  upstream has no guard for at all.
- **It fails fast rather than degrading silently.** At install time it reads
  `inspect.getsource` of the real upstream method and asserts all three flag
  reads and all six field names are present, and that `cross_level_notes` is
  *not* in the wipe list. An upstream reword raises at cell-execution time
  instead of silently guarding nothing. It also refuses to install twice.
- **A 9-case synthetic probe runs before any real game.** It exercises the
  in-level game over, game-over-plus-level-up, game-over-plus-run-complete,
  bare level-up, bare run-complete, an ordinary step, `None`, `{}`, and a
  summary object whose `.get` raises — asserting the resulting field state of
  each, that `cross_level_notes` survives all of them, and that the branch
  counters moved exactly as expected. Probe traffic is then zeroed out of the
  counters so it cannot pollute the run's own tally.

### 3.2 One variable, enforced mechanically

`scripts/_build_duck_nvfp4_anim_wipeguard.py` inserts the guard cell after the
anim notebook's customization hook and then **asserts, by per-cell sha256, that
all 18 inherited cells are present, in order, unchanged**. It also refuses to
build if the guard cell text contains `os.environ`, `LOCAL_ANALYZER` or
`bm.solver`. `stage7_config_locality.md`'s warning — a prior attempt carried an
unrelated `seqs=16` through three runs and confounded the only one that ran — is
the reason this is a mechanical check rather than an intention.

The guard is installed at the customization hook rather than beside cell 9's
knob overrides on purpose: those must precede the `inference` import because
`tool_agent` reads them into module globals at import time, whereas a method
wrapper is resolved per call and `ToolAgent` instances are constructed later,
per game, inside the solver.

### 3.3 Tests

`tests/test_wipe_guard.py` — 24 tests, all passing (repo suite: 157 → 181, none broken). It stands up a stub
`inference.agent.tool_agent` whose wipe method is **byte-for-byte the anim
bundle's**, execs the real cell source against it, and checks: the one case that
changes; a control proving unguarded upstream wipes that same case; all five
combinations that must still wipe; four that must stay untouched;
`cross_level_notes` across all of them; the two no-raise paths; three fail-loud
paths (upstream drops the `game_over` flag / renames a field / starts wiping
`cross_level_notes`); double-install; the counters; that **exactly one** method
on `ToolAgent` is rebound; that the cell mentions no knob or env var; and that
the notebook carries this exact cell rather than a drifted copy.

---

## 4. Free-run result

Kernel `calamitychasm/arc3-duck-nvfp4-anim-wg` v1, free push, public-25 path.
Compared against the **anim** arm (9.97 / 2,615 actions), which is the
incumbent — *not* the keithtyser baseline.

<!-- RESULT -->

---

## 5. Recommendation

<!-- RECOMMENDATION -->

---

## 6. Corrections to earlier Stage 7 documents

- `stage7_sota_research.md` §4.4 and §6.1 — "**74 `GAME_OVER` events** in our
  baseline run … about **122 wipes across 720 analysis steps**". The 74 is 37
  real game overs mirrored by the event stream's `type == "analysis"` rows
  (§2). The wipe count is 85, not 122, because wipes fire per step summary and
  several qualifying events share one. And the figure that matters for this
  intervention is the anim chassis's **16**, not the baseline's 37.
- `scripts/analyze_rhae_binding.py` — `level_actions()` counts every row with a
  non-null `action_num`, which includes the mirrored `analysis` rows and the
  per-game `initial` row. Its action totals are inflated ~37% (4,970 vs the
  real 3,633) and its published "51.4% of actions on the never-completed level"
  is 50.9% with the row filter applied. The *conclusion* — completion binds,
  not efficiency — is unaffected; the proportion is computed from inflated
  numerator and denominator alike.
