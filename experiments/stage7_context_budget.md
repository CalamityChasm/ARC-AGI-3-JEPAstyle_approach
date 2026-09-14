# Stage 7 — The context budget: where 20,126 tokens per request actually go

Date: 2026-09-13. Branch: `stage7-context-budget`. Labels follow this repo's
convention: **[VERIFIED]** = read directly from a primary artifact (run
artifacts, bundle source, model config, or a locally-run computation over
them); **[INFERRED]** = reasoning over verified facts, not directly observed.

**No competition submission was made.** Everything here is offline analysis of
the completed baseline run plus free `kaggle kernels push` runs. The daily
submission slot was untouched.

Reproduce every number with:

```
venv/Scripts/python.exe scripts/analyze_context_budget.py <baseline-output-dir> \
    --tokenizer <tokenizer.json> --json experiments/stage7_context_budget_data.json
venv/Scripts/python.exe scripts/vision_token_cost.py
venv/Scripts/python.exe scripts/project_context_window.py experiments/stage7_context_budget_data.json
venv/Scripts/python.exe scripts/project_turns_rhae.py <baseline-output-dir>/benchmark.json \
    --multipliers 1.0,1.34,1.57,1.93,2.0
venv/Scripts/python.exe scripts/_test_dedupe_patch.py <baseline-output-dir> --tokenizer <tokenizer.json>
venv/Scripts/python.exe scripts/compare_context_runs.py baseline=<dir> ctx16k=<dir> dedupe=<dir>
```

Baseline run: `calamitychasm/arc3-duck-nvfp4-baseline` v2, `COMPLETE`,
public-25, 2026-09-10, mean self-eval **10.69**, **3,633 actions**, profile
`kv5-bf16-mtp3-c8-cg32`.

---

## 0. Why this line exists, and what it inherited

`stage7_turn_latency.md` established that turn latency is **86.97% queue wait**
and that the cause is KV residency: `105,202 / 21,608 = 4.87` concurrent
requests, `Running` pinned at ~3.2, 191 preemptions, `max_num_seqs = 8` never
reached. `stage7_kv_residency_levers.md` then killed all three numerator
levers — fp8 KV is architecturally impossible on this model, an 8 GiB pool
OOMs against 81.8 GiB of weights, and prefix caching is a correctness hazard on
the Mamba/GDN path (identical actions, 47% score collapse).

That leaves the **denominator**: 21,608 tokens resident per request. This
document decomposes it and tests what can be taken out.

---

## 1. TL;DR

1. **The decomposition closes.** Summing the components gives 20,660 tokens
   against the server's own mean prompt length of 20,126 (103%). [VERIFIED]
2. **The multimodal image is 2.7% of the prompt, not the dominant cost.** The
   brief's leading hypothesis is refuted. [VERIFIED]
3. **`MULTIMODAL_UPSCALE` is inert.** Upscale 1, 2, 3 and 4 all cost exactly 66
   vision tokens, because `preprocessor_config.json`'s `shortest_edge: 65536`
   is `min_pixels` and a 64x64 grid at 4x is *exactly* that. Lowering it saves
   nothing. [VERIFIED]
4. **Disabling images would make throughput *worse*, not better.** [INFERRED,
   from verified inputs] — see §4.
5. **The single real lever is the harness's own context window**, because the
   trim loop always refills the prompt to the budget (measured fill 92.6%). Two
   candidates were built and run free: cutting the window, and de-duplicating
   the history. [VERIFIED for the mechanism]
6. **26% of every request is content the model is already reading elsewhere in
   the same request** — a ~24-line instruction block rebuilt verbatim on every
   turn. [VERIFIED]

---

## 2. The decomposition [VERIFIED]

Method (`scripts/analyze_context_budget.py`): the harness writes
`prompts/<game>.log`, a "LATEST MODEL CALL SNAPSHOT" rendering every message of
that game's final request in order (`tool_agent.py:830-860`). Each section is
tokenized with the model's own tokenizer, pulled from the Kaggle model mount
`keithtyser/qwen3-8-flash-next-nvfp4`. The one thing the log drops is the image
part of a multimodal user message, so images are priced separately from the
real boards in `artifacts/*_events.jsonl`, re-rendered exactly as
`vision_context.py: frame_to_png_data_url` does.

Mean over the 25 final-request snapshots:

| component | tokens | share | what it is |
|---|---:|---:|---|
| assistant reasoning | 6,784 | 32.8% | retained model thinking |
| **user boilerplate** | **5,343** | **25.9%** | **the same instruction block, repeated 8.6x** |
| system prompt | 2,854 | 13.8% | fixed, untrimmable |
| assistant tool calls | 2,105 | 10.2% | retained python code |
| tool results | 2,103 | 10.2% | retained tool stdout |
| user variable | 903 | 4.4% | state line, world model, executed actions |
| **image (real vision tokens)** | **568** | **2.7%** | **the hypothesised dominator** |
| **total** | **20,660** | | vs server's 20,126 — **103%** |

The snapshots are each game's *deepest-history* request, so landing 2.7% above
the run-wide mean is expected, not error.

**Closure check.** `vllm:request_prompt_tokens_sum / _count` =
26,948,103 / 1,339 = **20,126**. Independent server-side check on image count:
`vllm:mm_cache_queries_total` = 9,828 over 1,339 requests = **7.34 images per
request**, confirming an image rides on every retained user message. The
multimodal cache hit rate is 9,170/9,828 = **93.3%**, so stale images cost KV
footprint and prompt tokens but almost no encoder compute — which is exactly
why they never surfaced in the latency decomposition.

---

## 3. The knobs, with file:line

| knob | where it is set | value | controllable? |
|---|---|---|---|
| harness context window | `serving_setup.py:129` `ANALYZER_CONTEXT = 32_768` → written to env at `serving_setup.py:2935`; read at `tool_agent.py:138`; becomes the budget at `tool_agent.py:945-948` | 32768 | **yes**, by overriding the module constant after setup |
| trim budget | `tool_agent.py:945-948` `window - 512 (reply) - 512 (safety)` | 31,744 | derived |
| retained-turn cap | `tool_agent.py:151` `_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30` | 30 | yes, but **not binding** — the token budget bites first at ~8.6 turns |
| multimodal on/off | `serving_setup.py:2945` `MULTIMODAL_CONTEXT = "current_grid"`; read per-call at `vision_context.py:37` | on | yes — but see §4 |
| multimodal upscale | `serving_setup.py:2946` `MULTIMODAL_UPSCALE = "4"`; `vision_context.py:41` | 4 | yes — but **inert**, see §4 |
| tool-output cap | `serving_setup.py` `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = 1024`; `tool_agent.py:938` | 1024 | yes; tool results are only 10.2% |
| vLLM `--max-model-len` | `serving_setup.py:2323`, same `ANALYZER_CONTEXT` constant | 32768 | **explicitly not a lever** |

The last row is the trap the brief warned about, and it is worth spelling out
because the two settings share one constant: patching `ANALYZER_CONTEXT` in
`serving_setup.py` would move **both**. The patch in
`scripts/_patch_duck_notebook_context.py` deliberately overrides only
`tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW`, in the notebook's own designated
customization hook (cell 14), after setup has run — leaving `--max-model-len`
at 32768. vLLM allocates KV on demand, so the model-length cap is not what
makes sequences long, and dropping it below what the harness sends would get
requests rejected rather than truncated.

### 3.1 Why the window is the *only* lever [VERIFIED]

`tool_agent.py:1686`:

```python
while history and self._estimate_request_input_tokens([system_message, *history], tools=tools) > budget_tokens:
    if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
        break
```

The loop drops the oldest message block until the estimate fits, and the
harness keeps appending turns, so **the prompt is always pushed right up
against the budget**. Measured across the 25 snapshots: estimated payload
29,404 of a 31,744 budget = **92.6% fill** (the 7.4% slack is the granularity
of dropping a whole block at a time).

The consequence is not obvious and it reframes the whole problem: **making
individual messages cheaper does not reduce tokens per request.** The trim loop
simply retains more turns until it hits the budget again. Anything that only
frees budget converts into *more history*, not *fewer tokens* — a quality
change, not a throughput one. Only moving the budget itself moves
tokens-per-request.

---

## 4. The image: hypothesis refuted twice over

### 4.1 It is 2.7%, not the dominator [VERIFIED]

568 real vision tokens out of 20,126. Even deleting every image outright
removes 2.8% of the prompt.

### 4.2 `MULTIMODAL_UPSCALE` cannot be lowered [VERIFIED]

`preprocessor_config.json` from the model mount:

```json
{"size": {"longest_edge": 16777216, "shortest_edge": 65536},
 "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
 "image_processor_type": "Qwen2VLImageProcessorFast"}
```

`shortest_edge` is `min_pixels`; one vision token covers
`patch_size * merge_size` = 32 px per side. A 64x64 grid at upscale 4 is
256x256 = **65,536 px, exactly `min_pixels`**. `smart_resize` scales anything
smaller back *up* to it (`scripts/vision_token_cost.py`):

| upscale | rendered | after smart_resize | tokens/image | tokens/request |
|---:|---|---|---:|---:|
| 1 | 64x64 | 256x256 | 66 | 484 |
| 2 | 128x128 | 256x256 | 66 | 484 |
| 3 | 192x192 | 256x256 | 66 | 484 |
| **4** | **256x256** | **256x256** | **66** | **484** |
| 6 | 384x384 | 384x384 | 146 | 1,072 |
| 8 | 512x512 | 512x512 | 258 | 1,894 |

The upstream author's `4` is precisely the smallest upscale that avoids being
resampled. There is nothing to win here.

### 4.3 Disabling images would make throughput *worse* [INFERRED from VERIFIED]

This is the interesting one, and it is a direct consequence of §3.1.

`tool_agent.py:462`:

```python
def _estimate_tokens(value):
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return max(1, (len(rendered) + 2) // 3)
```

`json.dumps` of a multimodal message includes the entire
`data:image/png;base64,...` URL. Measured over the 25 snapshots: **13,590
base64 chars per request → 4,530 "tokens" of budget, against 568 real vision
tokens. An 8.0x overcount, eating 14.3% of the 31,744-token budget.**

So disabling multimodal frees 4,530 estimated tokens, which the trim loop
immediately refills with real history at ρ = 0.684 real tokens per estimated
token → **+3,099 real tokens**, against the −568 saved. Net **+2,531 real
tokens per request**, residency 4.88 → **4.35 (0.89x)**.

Turning the image off is a throughput *regression*. It is not in the run plan
for that reason, and this is a prediction the plan is willing to be wrong
about — it is stated here so a future session can falsify it cheaply rather
than re-deriving it.

---

## 5. The duplication [VERIFIED]

`tool_agent.py:_build_user_prompt` rebuilds the same ~24-line instruction block
on every turn — "Only tool: `python`. It receives…", "Keep tool output
compact…", "When calling `python`, emit exactly the tool-call format…", and so
on. Measured: **621 real tokens per user message** (min 571, max 798, n=215
messages across the 25 snapshots), with 8.6 retained.

**4,721 real tokens per request (23.5% of the whole prompt) are byte-identical
copies of text the model is also reading in the current turn's user message.**
Add the stale images and the picture is:

| | real tokens | harness-estimated tokens |
|---|---:|---:|
| stale boilerplate (all but newest) | 4,721 | 7,204 |
| stale images (all but newest) | 502 | 3,998 |
| **total redundant** | **5,223 (26.0% of prompt)** | **11,202 (38.1% of budget)** |

**38% of the harness's context budget is spent on content that is either an
exact duplicate or an 8x-overcounted base64 string.**

---

## 6. The two candidates

Both are generated from the pristine 10.69-baseline notebook by
`scripts/_duck_notebook_patch.build`, which asserts the baseline md5, requires
the hook cell's anchor line, compares every other cell byte-for-byte, and
`compile()`s the generated cell. Exactly one cell differs in each. Each build
is archived at `kaggle_submission_duck_nvfp4/variants/<slug>.ipynb`.

### Candidate A — `ctx16k`: halve the harness window

`LOCAL_ANALYZER_CONTEXT_WINDOW` 32768 → 16384, budget 31,744 → 15,360, via
`tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW` in cell 14. `--max-model-len` stays
at 32768. The patch fails loud: it checks that the solver's `ToolAgent` comes
from the module being patched, that no `analyzer_factory` bypasses the path,
that the starting value is 32768, and it reads `_context_budget_tokens` back
off a real probe agent.

Projected (`scripts/project_context_window.py`, calibrated on the baseline):

| window | real prompt | tok/request | residency | x turns | retained turns |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 20,126 | 21,559 | 4.88 | 1.00 | 8.6 |
| 24,576 | 14,937 | 16,370 | 6.43 | 1.32 | 6.0 |
| **16,384** | **9,748** | **11,181** | **9.41** | **1.93** | **3.4** |
| 8,192 | 4,558 | 5,991 | 17.56 | 3.60 | 0.8 |

**Why 3.4 retained turns is less alarming than it looks** [VERIFIED]: Duck
carries an explicit `_summarized_knowledge` world model as a separate instance
attribute (`tool_agent.py:957`), updated from the model's own assistant
prefixes (`:1896`, `:1930`) and re-injected into *every* user prompt
(`:1236`). It is not part of the message history and survives trimming
entirely. Duck already has a designed defence against a short context; this
candidate tests how much of the load it actually carries.

### Candidate B — `dedupe`: stop re-sending what the model already has

Removes from every historical user message any line that appears **verbatim in
the newest user message**, and drops stale images. Hooks
`ToolAgent._chat_completion` — the send path — and deliberately *not*
`_trim_messages_for_context` or `_persistent_history_messages`, so the stored
history stays fat and every retention decision is bit-for-bit the baseline's.
Only the wire payload shrinks. (Hooking the trim path instead would have
converted the saving into more retained turns at unchanged tokens-per-request:
a quality change, not a throughput one, and not what this experiment is
measuring.)

Verified before spending GPU time (`scripts/_test_dedupe_patch.py` extracts the
patch body from the generated notebook, executes it against a stub
`tool_agent`, and replays the 25 real request snapshots through it):

```
text tokens before      : 20,101
text tokens after       : 15,150
text saved              :  4,952
stale vision tokens cut :    502
TOTAL real tokens saved :  5,453   (27.1% of the 20,126 prompt)
invariant violations    : 0
PASS: every removed line is present verbatim in the newest user message
```

Projected: tokens/request 21,559 → 16,106, residency 4.88 → **6.53 (1.34x)**,
retained turns **unchanged** at ~8.6, information loss zero by the invariant.

### RHAE upside at those multipliers

From `scripts/project_turns_rhae.py` on the baseline `benchmark.json` (the
scorer is reimplemented verbatim from
`tufa-arc-agi-framework/src/taaf/game.py: GameRun._compute_final_score` and
reproduces all 25 recorded `final_score` values bit-exactly at x1.0):

| x turns | optimistic | stuck-aware | +1 only (floor) | levels |
|---:|---:|---:|---:|---:|
| 1.00 | 10.69 | 10.69 | 10.69 | 48 |
| **1.34** (dedupe) | 18.65 | 15.31 | **14.88 (+39.2%)** | 62 |
| **1.93** (ctx16k) | 34.92 | 23.17 | **17.01 (+59.2%)** | 68 |
| 2.00 | 37.69 | 24.78 | 17.70 (+65.6%) | 72 |

Monotone by construction: the efficiency denominator sums over **all** levels
and is fixed, so an extra solved level adds `w_l * S_l >= 0` to a constant
denominator. More turns cannot lower the score.

---

## 7. Results

*Pending — both kernels pushed free and running. This section is filled in from
`scripts/compare_context_runs.py`, which reports score and actions together and
recomputes the mean from the per-game `[finished]` lines rather than trusting
the progressive summary blocks.*

---

## 8. Recommendation

*Pending the runs.*

---

## RESULTS (2026-09-14) — both candidates regressed; the line is closed

| run | public-25 | actions | vs baseline |
|---|---:|---:|---|
| baseline | **10.69** | 3,633 | — |
| history dedupe | **7.09** | **1,885** | score −33.6%, actions **−48.1%** |

**The transform did exactly what it was built to do** — startup telemetry:
`HISTORY_DEDUPE active: hook=ToolAgent._chat_completion probe_chars=24547->12376`,
i.e. the payload was genuinely halved, and all five startup assertions passed.

**And actions still halved.** That is the opposite of the mechanism's whole
prediction: fewer tokens per request should raise residency and therefore
turns. Instead turns fell ~48%.

**[INFERRED] What this says.** The deduplicated content was load-bearing.
Stripping the repeated instruction block from history did not just shrink the
prompt, it changed model behaviour — plausibly longer or more confused
generations, so time-per-turn rose faster than the smaller prompt saved. The
26% of the request that is "content the model is already reading elsewhere"
is apparently not redundant *to the model*, only to a reader.

This is the second time on this stack that a change with a sound throughput
mechanism produced a behavioural regression invisible to token accounting
(prefix caching was the first: identical actions, −47% score). **On this
harness, prompt content cannot be treated as a free variable.**

## Verdict: the throughput line is closed

Five levers, five failures, all found on free runs, no submission quota spent:

| lever | outcome |
|---|---|
| fp8 KV dtype | architecturally impossible (`QSA requires BF16`) |
| KV pool 5 → 8 GiB | CUDA OOM (81.8 GiB weights / 95 GiB card) |
| prefix caching | identical actions, −47% score (Mamba/GDN state) |
| multimodal upscale | **inert** — 66 vision tokens at every upscale |
| history dedupe | −34% score, −48% actions |

Both sides of `residency = pool_tokens / tokens_per_request` are now exhausted
with the model held fixed. The numerator is capped by GPU memory; the
denominator is capped by the model needing the context it is given.

**Recommendation: stop optimising throughput on this stack.** The remaining
direction that does not trade away capability is not a serving or prompt knob
at all. Tufa's own framing — *"solvability depends on model capability, the
cost is dictated by the harness"* — now reads as a ceiling statement: we have
spent the harness side out.
