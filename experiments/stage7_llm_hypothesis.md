# Stage 7 exploration: an LLM as a low-frequency hypothesis generator

**Status: exploratory research spike, not a Stage.** This isn't in
`plan.md` -- it's a one-off evaluation of an idea floated in response to
seeing the current Kaggle-leaderboard-leading team reportedly using a
fine-tuned LLM with strong tool-wrapping. Scoped explicitly as "no
pressure if it's a stupid idea" (see the framing this branch was spawned
under). **Verdict up front: interesting capability, not worth building
into the real submission right now.** Full reasoning below.

## The idea being evaluated

Not per-action LLM control (a multi-second call on every one of
thousands of actions across up to 110 games, under the harness's
one-Python-thread-per-game-under-the-GIL model, inside a 9-hour cap, is a
real throughput risk regardless of GPU size -- and this codebase already
has a per-action LLM template, see "What already exists" below, which
would hit exactly this problem). Instead: a **low-frequency hypothesis
generator**, called once per episode/reset, that reads a compact
serialization of the frame(s)/diffs seen so far in a game and produces a
natural-language guess at the game's mechanics -- feeding either smarter
opening probes (replacing `Hypothesis._probe_plan`'s current "try every
simple action once" logic) or a prior over which MoE expert / action to
trust, feeding into the existing `beta`-blend in
`jepa/hypothesis_bundle.py`.

## Constraint check: no internet during scored runs (confirmed)

`rules.md`: *"Internet access disabled during scored runs."* -- and
concretely, this project's own already-working submission kernel has it
hard-coded: `kaggle_submission/notebook/kernel-metadata.json` has
`"enable_internet": false`. This is not a maybe -- any real, submittable
version of this idea **cannot call an external LLM API** (OpenAI,
Anthropic, etc.) during the actual scored run. It would need a fully
local model bundled the same way `checkpoints/*.pt` already is (via
`dataset_sources`, or Kaggle's separate `model_sources` mechanism --
`kernel-metadata.json` already has an empty `"model_sources": []` slot,
suggesting that path was scaffolded but never used).

`rules.md` also confirms: *"External data/pretrained models: freely &
publicly available ones are allowed"* -- so a small open-weight local LLM
(e.g. a 1-3B quantized model) would be permissible under the rules, it's
purely an engineering/reliability question, not a rules question.

**Hardware note:** the user has H100 access for *prototyping*, but
that's not what the scored run gets. `rules.md`'s actual competition
hardware pool is **RTX 6000 (g4-standard-48)**, no internet, and this
project's own `CLAUDE.md` already documents `enable_gpu` as having
"known unresolved flakiness" on the current Kaggle kernel (one of
several hardening attempts tried while chasing the real submission bug --
see CLAUDE.md's Kaggle section). A heavier model dependency directly
inherits that risk: more GPU memory pressure, more to go wrong in exactly
the subsystem already flagged as unreliable, on top of an already
GPU-optional design (nothing in `jepa/` currently *requires* GPU to run
inference-only, and `Hypothesis`'s existing checkpoints total **~3.9MB**
combined across all four `.pt` files -- trivially CPU-inference-fast).

## What already exists in this repo (don't rebuild it)

`ARC-AGI-3-Agents/agents/templates/llm_agents.py` already has a full
per-action LLM agent (`LLM`, plus `ReasoningLLM`, `FastLLM`, `GuidedLLM`,
and a `MyCustomLLM` template) built on `openai` + function-calling, one
API call per action via `client.chat.completions.create(...)`. This is
exactly the design this idea deliberately avoids (per-action, not
per-episode) -- but it's useful prior art: the message-history management
(`push_message`, FIFO-trimmed), the function/tool schema for the 8 ARC-3
actions, and the frame pretty-printer (`pretty_print_3d`, a naive
`str(row)` dump, much more token-heavy than the hex-digit format used in
this prototype) are all reusable if a hypothesis-generator agent gets
built for real. `requirements.txt`'s langchain/langgraph/smolagents/
openai pins exist to support this template and its siblings
(`langgraph_thinking/`, `reasoning_agent.py`, `smolagents.py`) --
confirms this project already anticipated wanting LLM-agent tooling, just
never wired one into the JEPA-based agents.

**Also directly relevant:** `ARC-AGI-3-Agents/.env`'s `OPENAI_API_KEY` is
a placeholder (`your_openai_api_key_here`), not a real key, in both the
main checkout and this worktree, and no `ANTHROPIC_API_KEY` exists
anywhere in the repo's env files or this sandbox's environment variables.
**No external LLM API was actually callable from a standalone script in
this environment for prototyping purposes.**

## Which documented weakness this could plausibly target

Two candidates from `CLAUDE.md`'s Stage 5 history:

1. **The value head is trained on ~98% zero-target data**, barely
   distinguishable from a zero baseline on typical states (Stage 5's
   "teacher policy" follow-up improved this meaningfully on the
   *meaningful*-subset metric, but the full-population signal is still
   thin). An LLM can't manufacture reward density that isn't there in the
   data -- this is a data-volume problem, not a reasoning problem, so an
   LLM hypothesis generator doesn't obviously help here.
2. **The opening-probe logic is a dumb fixed action-by-action scan**
   (`Hypothesis._probe_plan`), with no understanding of game mechanics.
   This is the more plausible target -- *if* an LLM could look at a
   game's early frames and produce a better-than-uniform prior over which
   actions/regions are worth probing first, it could shrink the
   "reconnaissance tax" the same way Stage 2's `Curiosity` bug #2 fix
   (making ACTION6 compete as one option instead of a mandatory 64-patch
   raster-scan) did.

But there's a subtlety worth being honest about: the *numeric* mechanism
already built (MoE expert disagreement = `InfoGain`) is already a
formalized version of "which actions are surprising/informative" --
that's the whole point of Stage 5's design. What an LLM's *semantic*
reading could add on top isn't "detect surprise" (already covered
numerically) but **distinguish surprising-and-meaningful from
surprising-but-cosmetic** (e.g. a HUD countdown bar draining every single
turn regardless of action is maximally "different from predicted" in
pixel/feature space, but semantically irrelevant to action choice) -- see
the prototype results below, which found exactly this pattern in real
data. That's the one concretely non-redundant value-add this idea has
over the existing numeric machinery, not "form hypotheses" in general.

## Prototype: can an LLM produce a specific, plausible hypothesis from a compact diff serialization?

### Setup

Built `scripts/serialize_episode_for_llm.py` (this branch, isolated --
doesn't touch `hypothesis_agent.py` or any live integration). Reads a
recorded episode's `*.recording.jsonl` (local, gitignored, present in the
main checkout's `ARC-AGI-3-Agents/recordings/` but not this worktree --
referenced by absolute path, not copied), and serializes: the full first
grid as a compact hex-digit-per-cell grid (64x64 chars, one char per
cell, values 0-15 as `0-9a-f`), then for each subsequent step the action
taken and only the *changed cells* (`(row,col): old -> new`), capped at
40 per step to bound length on very busy frames. A 16-step episode
serializes to roughly **8,000-20,000 characters** (~2,000-5,000 tokens)
depending on how change-heavy the game is -- small enough that a
once-per-episode call is cheap even on a modest local model, and small
enough to not be the bottleneck if this were ever built for real.

### No external API available -- used Claude (this agent) directly as the LLM under test

As noted above, no usable `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` was
available in this environment for a standalone script to call out to.
Rather than skip the test or fake a result, this agent (Claude,
Sonnet 5) read the serialized text directly and produced a hypothesis,
exactly the operation a bundled LLM would need to perform -- this tests
the actual capability question ("can an LLM extract a specific,
plausible hypothesis from this compact representation") even though it
doesn't test the *packaging* question (a small local model bundled into
a Kaggle dataset is a separate, unverified concern -- see Verdict).
There's no local ground-truth label for "the game's real mechanic," so
judgment of specificity/plausibility is this agent's own, same
limitation the task was scoped under.

### Results (3 real recorded episodes, random-policy local recordings)

**`r11l` (documented in `CLAUDE.md` as a click-only game --
`available_actions=[6]`):** Serialized 16 steps. Step 0's grid shows a
clearly maze/path-like structure (a large uniform "floor" region, several
small distinct clustered shapes at different coordinates using an outlier
color not used elsewhere in the grid). After the first `ACTION6` click,
step 2 (a non-ACTION6 action, since `available_actions` only *lists*
ACTION6 but the random policy sends others anyway) shows a **100+ cell
diagonal-trail diff** -- consistent with a delayed click-triggered
movement/reveal animation resolving on the following frame, not an
instant one-frame effect. From step 3 onward, nearly every action
(regardless of which of `ACTION1-7` was sent) produces an identical
**1-cell diff at `(row=N, col=0)`** where N increments by exactly 1 each
step -- a HUD/counter strip in column 0 that advances on every tick
independent of the action, not part of gameplay. **Hypothesis produced:**
*"top-down maze/point-and-click game; column 0 is a step counter or
similar HUD element unrelated to the action taken; the distinct small
colored clusters scattered on the floor are landmarks/objects (key,
door, target); ACTION6 clicks trigger a movement or reveal effect that
resolves on the following frame, not instantly."* Specific and
falsifiable, not a generic non-answer.

**`bp35`** (`CLAUDE.md` flags this as the local corpus's
highest-activity, "100% frame-level changed rate" game):
`available_actions=[3, 4, 6, 7]` at reset. Step 0's grid shows a
repeating tiled background of small 4-cell icon clusters plus several
large uniform-color rectangular regions, one containing a distinct
3x2 icon pattern (colors `9`/`14`/`3` in a recognizable box shape).
Steps 3-15 show a small ~5x7-cell multi-color block consistently present
around rows 37-41, shifting column position by a few cells almost every
step regardless of which action fired (`ACTION1`, `ACTION2`, `ACTION4`,
`ACTION7` all moved it) -- consistent with a single sprite/cursor sliding
along a horizontal track. Separately, `ACTION6` clicks toggle a *fixed*
region's colors (revealing the `9/14/3` icon pattern underneath a
different color, i.e. flipping a switch/door state) and also append to a
bottom-row counter strip identical in structure to `r11l`'s. Around steps
11-14, diff sizes balloon (102 -> 343 -> 280 -> 492 changed cells) with a
segmented bar pattern visibly shifting rows (row 61 -> row 55 over two
steps) -- read as a scrolling background layer or multi-element
animation triggered once enough of something else has happened, not a
per-action-caused effect. **Hypothesis produced:** *"puzzle/platformer
with a movable sprite on a horizontal track (moved by the 4 simple
actions), switch/door toggles via ACTION6 clicks on fixed regions, a
step-counter HUD strip, and a larger background reconfiguration event
partway through the episode consistent with a scrolling layer or gate
opening."* Again specific, with concrete cited evidence for each claim,
not verified against ground truth but not a vague answer either.

**`sp80`** (`CLAUDE.md`: a game both `Curiosity` and `Hypothesis` find
"easy," ~30-action reset cadence): Step 0 shows a top row (row 0) of a
distinct color (`14`) shrinking by ~2-3 cells per step, and (from step 3)
a rectangular region changing from color `9` to color `8` a row at a
time, cleanly matching a "fill" or "progress" visual. **Hypothesis
produced:** *"row 0 is a countdown/moves-remaining bar that drains every
turn regardless of action (consistent with the documented ~30-action
reset cadence -- 30ish cells roughly matches a full drain); the 9->8
region is a fill/progress indicator, possibly the actual win condition to
race the timer toward."* This one is the most directly checkable against
the project's own prior documented behavior (the ~30-action cadence) and
the inferred mechanism is consistent with it.

### Judgment

All three produced **specific, structured, falsifiable claims** referencing
concrete evidence in the serialized diffs (exact rows/columns/colors),
not generic "this is a grid game" non-answers. This directly answers the
capability question the task asked to test: **yes, an LLM (at least a
frontier one) can extract a plausible, specific hypothesis about game
mechanics from a compact diff serialization of a short random-policy
episode opening.** Whether these particular hypotheses are *correct* is
unverified (no local ground-truth game-mechanic labels exist, same
limitation flagged in the task brief) -- but "specific and falsifiable"
was the bar to clear, and it was cleared cleanly on all three tested
games, including correctly separating HUD/cosmetic elements (the
column-0 counter, the row-0 timer) from gameplay-relevant elements (the
sprite, the switch region) in two of the three cases -- exactly the
semantic distinction flagged above as the one thing the existing numeric
`InfoGain` mechanism can't do on its own.

## Verdict: interesting, not worth pursuing for the real submission right now

**The capability is real** -- this isn't a dead end on the "can an LLM
usefully read this data" question. But three things keep this from being
a good next lever for this project, weighed against `plan.md`'s own
guiding principle 4 ("add components only when a measured bottleneck
demands it"):

1. **No internet at scoring time means this can't be tested end-to-end
   without first solving a separate, nontrivial packaging problem**: a
   local open-weight LLM (even a small 1-3B quantized one) is
   plausibly 500MB-4GB, vs. the current four checkpoints' combined
   **~3.9MB** -- a large jump in what has to be bundled and correctly
   mounted. This project's own Kaggle section documents an entire
   debugging session lost to exactly this class of bug (a dataset
   mounting at a different path than assumed, `/kaggle/input/datasets/
   <owner>/<slug>/` vs. the wrongly-assumed `/kaggle/input/<slug>/`) --
   adding a much larger, more failure-prone asset (plus a new inference
   runtime dependency not currently in `requirements.txt`: something like
   `transformers`/`llama-cpp-python`/`gguf`) directly multiplies the
   surface area for that exact failure mode to recur, on top of the
   already-flagged `enable_gpu` flakiness.
2. **No measured bottleneck currently implicates missing semantic
   reasoning specifically.** Every real reliability gap found in Stage 5's
   long debugging arc (ACTION6's mismatched IG reduction, the value
   head's data sparsity, the `argmax` tie-breaking bug on click-only
   games) was fixed with cheap, non-LLM, numeric/architectural
   interventions -- and each fix was verified with a direct, matched
   before/after comparison. There's no equivalent direct evidence here
   that the *remaining* gap (if any, post-Stage-5-follow-up-3's milestone
   pass) is specifically a "the agent can't tell HUD noise from gameplay
   signal" problem, as opposed to something else not yet found. Building
   this without that evidence would be exactly the kind of
   speculative-complexity-first move `plan.md` warns against.
3. **The one concretely non-redundant value-add identified here (semantic
   HUD-vs-gameplay filtering for `InfoGain`/click-salience) is a narrow,
   well-scoped thing** -- and notably, it doesn't obviously require an
   LLM at all. A much cheaper first test of the same underlying idea:
   check whether cells/patches that change on *every* step regardless of
   action (a simple statistic, computable without any model) correlate
   with reduced `Hypothesis` performance when *not* excluded from
   `InfoGain`'s patch reduction -- i.e., try a heuristic "mask out
   always-changing patches" fix first, before reaching for an LLM to
   detect the same pattern.

**Recommendation: worth it only if a future session (a) finds direct
evidence that HUD/cosmetic-vs-gameplay confusion is actually costing
`Hypothesis` real performance (e.g. by tracing episodes on games with an
obvious HUD element and checking whether InfoGain/click-salience wastes
budget there), and (b) the cheap non-LLM heuristic in point 3 above is
tried first and found insufficient.** Until then, this stays a documented
negative-leaning exploration, not a Stage. Consistent with the "no
pressure if it's a stupid idea" framing this was scoped under -- it
wasn't a stupid idea (the capability check is a genuine, clean
finding), but it doesn't clear this project's own bar for adding a real
component right now.

## Reproducing this

```
python scripts/serialize_episode_for_llm.py <game_prefix> --steps 15 --out <file>
```

Defaults to reading `random`-agent recordings from the main checkout's
`ARC-AGI-3-Agents/recordings/` (hardcoded absolute path at the top of the
script, since that dir is gitignored and worktree-local checkouts won't
have it -- override with `--recordings-dir` if running elsewhere). No
`torch`/model dependency -- pure stdlib `json`, runs under any Python
3.x, including the plain system interpreter (not just the project's
`venv`). Feed the output to whatever LLM is available (this session used
Claude directly, reading the file); judge specificity/plausibility by
hand, same as this document did, since there's no local ground-truth
label for "the game's actual mechanic" to check against automatically.
