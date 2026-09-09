# Stage 7 -- Duck concurrency: does aggregate throughput rise with concurrency?

**Verdict: yes, modestly. Recommend `concurrency = 37` (+8.0% effective
tokens vs. the inherited 28). Not sufficient on its own to reach the
top-10% bar.**

## Why this needed measuring rather than assuming

Raising `concurrency` from 28 to 37 turns `ceil(110/37) = 3` waves into
`ceil(110/28) = 4`. That sounds like a straightforward win, but it is not:

Total wall-clock is fixed by the 9h cap, and aggregate throughput is a
property of the vLLM server, so

```
total_tokens  ~=  aggregate_throughput  x  wall_clock
```

**regardless of concurrency.** Raising it gives each game more wall-clock
but a proportionally thinner slice of the GPU:

```
tokens_per_game ~= (agg_tps / conc) x (wall_clock / waves),   waves ~= n_games / conc
```

and the `conc` terms cancel. **So concurrency only helps if aggregate
throughput genuinely RISES with more concurrent sequences.** Prior
reasoning pointed at ~22% KV-cache utilisation as evidence of headroom,
but that was an inference from a production log, not a measurement.

## Method

Free Kaggle kernel (`calamitychasm/arc3-duck-concurrency-benchmark`) on
the production hardware: **RTX PRO 6000**, real `Qwen3.8-27B-FP8` weights,
real vLLM 0.19.0, the same setup path the submission notebook uses.
Identical workload at each level: `max_tokens=512`, ~9,380-token prompts
(representative of the harness's real context, not a toy prompt).
**No submission quota was used at any point.**

## Results [VERIFIED -- real kernel log]

| conc | ok | e2e agg tok/s | decode agg tok/s | per-seq tok/s | ttft p50 | preemptions |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 14/14 | 230.8 | 429.5 | 30.68 | 10.09 | 0 |
| 28 | 28/28 | 288.7 | 680.2 | 24.29 | 16.96 | 0 |
| 37 | 37/37 | 310.3 | 786.8 | 21.26 | 20.33 | 0 |
| 48 | 48/48 | **322.0** | 880.1 | 18.34 | 26.01 | 0 |
| 64 | 64/64 | 290.7 | 885.3 | 13.83 | 34.66 | **4** |

**Aggregate throughput does rise with concurrency** -- the token-neutrality
argument's precondition is satisfied. Decode-only aggregate climbs
steeply (680 -> 880 tok/s from 28 to 48) and plateaus by 64. End-to-end
aggregate peaks at 48 and *falls* at 64, where preemptions appear.

## The wave-balance correction -- the naive answer is wrong [INFERRED from VERIFIED numbers]

Picking the highest raw throughput (48, at 322.0 tok/s) is a mistake.
Games run in fixed-length waves and every game burns its whole cap, so
the *last* wave runs at whatever concurrency is left over -- and a
small final wave runs at small-wave efficiency.

Weighting each wave by its actual size:

| conc | waves | wave sizes | effective mean tok/s | vs. 28 |
|---:|---:|---|---:|---:|
| 28 | 4 | 28, 28, 28, 26 | 286.6 | 1.000x |
| **37** | **3** | **37, 37, 36** | **309.5** | **1.080x** |
| 48 | 3 | 48, 48, **14** | 291.6 | 1.017x |
| 64 | 2 | 64, 46 | 305.3 | 1.065x |

**48 collapses to +1.7%** because its third wave holds only 14 games and
runs at 230.8 tok/s. **37 wins at +8.0%** precisely because 110 divides
into three nearly-equal waves of 37/37/36. 64 looks respectable here but
is rejected on the measured evidence: it showed preemptions and the worst
end-to-end aggregate of any level tested.

## Verdict

- **Set `DUCK_TARGET_CONCURRENCY = 37`.** Expected effect: **~+8% total
  tokens generated across the run**, hence roughly ~8% more actions.
- **This is real but small.** It does not close a 1.77 -> 2.99 gap, and it
  should not be presented as if it might. Treat it as one cheap,
  measured, low-risk improvement to be combined with others.
- **Do not raise concurrency to 48 or 64.** 48 is a wave-balance trap;
  64 is past saturation and preempts.

## What could not be measured

- **KV-cache utilisation came back `nan` at every level** -- the
  `vllm:gpu_cache_usage_perc` metric did not scrape. The "~22% utilised"
  figure from the production log is therefore **not re-verified here**.
  The throughput curve is direct evidence of headroom regardless, so the
  conclusion does not depend on it.
- Whether ~8% more tokens translates into ~8% more *score* is unknown.
  RHAE weights completion and squares efficiency; more actions per game
  is not linearly more levels.

## Process note: three GPU runs were wasted on inference before instrumenting

Runs 1-3 produced an empty results table with
`FATAL: server not answering: status=200 err=None`. Two guessed fixes
(`delta.content`, then `delta.reasoning_content`) failed; the second was
verified to have shipped by pulling the kernel source back from Kaggle,
and failed identically. Only after dumping the **raw SSE stream** did the
real field name appear: this vLLM build emits `delta.reasoning`.

The failure was amplified by a coupling worth remembering: `ok` is gated
on parsed arrivals, and token accounting sums `usage.completion_tokens`
**over `ok` requests only** -- so one unrecognised field name silently
zeroed every measurement while the server was serving perfectly (its own
usage chunk reported `completion_tokens=8`). The harness now also accepts
a request as ok when the server reports `completion_tokens > 0`, so a
future field rename degrades instead of producing a confident, empty,
plausible-looking table.

This restates a lesson already in CLAUDE.md: **instrument before
theorising.** One instrumented run would have cost less than three
guessed ones.
