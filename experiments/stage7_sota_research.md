# Stage 7 — SOTA research: what the field is doing, and the biggest lever we own

Date: 2026-09-15. Branch: `stage7-sota-research`. **Work in progress — committed
incrementally so partial findings survive.**

Labels: **[VERIFIED]** = read directly from a primary artifact (a run artifact,
an API response, bundle/notebook source, or a locally-run computation over them).
**[INFERRED]** = reasoning over verified facts, not directly observed.

**No competition submission was made or will be made by this investigation.**
Everything here is read-only analysis plus (at most) free `kaggle kernels push`.

---

## HEADLINE (task 5): `tokens_per_action` is explained, and the cause is one env var

`stage7_model_search.md` §2 left this open: `tokens_per_action` spans 515 → 1,782
across our runs, "cancels every throughput gain", and "nobody has explained why".

**It is now explained.** [VERIFIED]

`scripts/analyze_tokens_per_action.py` + transcript parsing over five completed
runs shows the quantity factors exactly:

```
tokens_per_action = tokens_per_analyzer_turn / (actions_per_executing_turn x executing_turn_fraction)
```

and the term that moves is `executing_turn_fraction`: **36–54% of every LLM call
we pay for produces no game action at all.** Their stop reason is not an error,
a malformed tool call, or a crash. In all five runs it is one single message:

```
step_executed: False
message: Yielded control to solver: turn_time_budget.
```

`turn_time_budget` is `LOCAL_ANALYZER_YIELD_SECONDS`, printed in every
`[ANALYZER STATUS]` block as `yield_seconds`. Our stack runs **60.0**. One LLM
round-trip on this stack costs a **median ~153 s** (86.97% of it queue wait,
per `stage7_turn_latency.md`). So the yield budget expires *during the first
response*: if that response was an inspect-only `python` call, the turn ends
having taken no action, and the solver re-prompts from scratch.

Full table is in §1 below.

