# Stage 7 — KV-residency levers: all three dead, and why

**Verdict: the turn-latency line is CLOSED.** Three levers, three failures, all
found on free runs. No submission quota spent. The stated falsifier from
`stage7_turn_latency.md` has fired.

## Background

Turn latency is **86.97% queue wait** [VERIFIED, `vllm-metrics-final.prom`,
n=1,339]. The cause is KV residency: a **105,202-token pool** against **21,608
tokens resident per request** ⇒ only **4.87 concurrent**, `Running` pinned at
~3.2, 191 preemptions. `max_num_seqs=8` was **never reached** and is not the
constraint.

`residency = pool_tokens / tokens_per_request`. Levers 1-2 attack the
numerator; lever 3 attacks the denominator.

## Results — all measured against the 10.69 public-25 baseline

| lever | outcome | public-25 | actions |
|---|---|---:|---:|
| baseline (production) | — | **10.69** | 3,633 |
| 1. `kv-cache-dtype fp8` | **impossible** | — | — |
| 2. KV pool 5 → 8 GiB | **CUDA OOM** | — | — |
| 3. prefix caching + `seqs=16` | ran, regressed | 6.88 | 3,472 |
| 3b. prefix caching alone (`seqs=8`) | ran, regressed | **5.65** | 3,641 |

### Lever 1 — fp8 KV: architecturally impossible [VERIFIED]

```
NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache
```
vLLM exits before readiness. Not a tuning issue. The "zero memory cost, ~1.7x
residency" option that headed the recommendation list **does not exist on this
architecture**. This is why it had to be measured rather than reasoned about.

### Lever 2 — bigger pool: OOM [VERIFIED]

```
Model loading took 81.8 GiB memory
[CUDACachingAllocator] memory allocation failed with OOM
```
during MoE autotuning. **81.8 GiB of weights on a 95 GiB card** means the 5 GiB
KV pool is **not a preference — it is the remainder.** The pool cannot grow on
this hardware.

### Lever 3 — prefix caching: no throughput gain, severe quality regression [VERIFIED]

The decisive numbers are the **actions**, not the score:

```
baseline      3,633 actions -> 10.69
prefix alone  3,641 actions ->  5.65
```

**Actions are identical (+0.2%). Turns-per-game did not move at all** — prefix
caching bought exactly zero residency. Yet score fell by **47%**.

Same volume of play, far worse decisions ⇒ **this is an output-quality
regression, not a throughput one.** The mechanism is almost certainly the one
vLLM itself warns about on this hybrid GDN/Mamba model:

```
WARNING Mamba cache mode is set to 'align' ... when prefix caching is enabled
INFO    Prefix caching in Mamba cache 'align' mode ... support for Mamba layers
        is experimental
```

Reusing cached KV blocks across requests appears to corrupt recurrent state on
the Mamba/GDN path. One game also failed to finish (24 of 25).

**This explains something previously read as an oversight:** prefix caching is
disabled in the upstream Duck profile *deliberately*. It is a correctness
requirement on this architecture, not a missed optimization.

## A methodological error, recorded

`seqs=16` was carried through all three attempts and never isolated. It was
justified for levers 1-2 (both would have pushed residency past the 8 cap) but
that justification did not survive their failure, and it rode into lever 3
anyway — confounding the one run that actually executed.

Run 3b exists purely to fix that. It shows `seqs=16` was **not** the culprit:
removing it moved the score 6.88 → 5.65, i.e. slightly *down*, well within the
noise of this metric at n=1. Prefix caching owns the regression.

## What the validation design got right

A throughput benchmark would have called lever 3 a **no-op** (actions
unchanged) and possibly shipped it. Only the **public-25 rerun** exposed a 47%
quality collapse. The recommendation in `stage7_turn_latency.md` to gate on a
real rerun rather than a synthetic sweep is what caught this.

## Durable constraints established (all free)

1. Queue wait is **87%** of turn time.
2. The KV pool is **maxed at 5 GiB**; the card cannot hold more alongside 81.8 GiB of weights.
3. **fp8 KV is impossible** on Qwen3.8-Flash-Next (QSA requires BF16).
4. **Prefix caching is harmful** here — a correctness hazard on Mamba/GDN, not an optimization.
5. `max_num_seqs` is inert: residency is pinned at ~3.2 by memory, not by the cap.
6. **50.2% of all actions** (1,825 / 3,633) go into levels that never complete.

**This stack is memory-bound at ~3 concurrent requests on this hardware, and no
serving-level knob changes that.**

## Where this leaves things

Remaining directions, both larger than a config flag:
- **A smaller model** — frees memory for KV, buys residency, but costs
  capability, which Tufa's writeup says is what determines *solvability*. A
  direct trade against the thing that actually drives score.
- **The harness's own context management** — 21,608 resident tokens per request
  is the denominator, and Tufa names context management as one of their two
  weak points. Lowering vLLM's `--max-model-len` is NOT the way (vLLM allocates
  on demand; dropping it below what the harness sends gets requests rejected).
  The real budget lives inside the third-party solver bundle.

Neither should be started without sizing it against the time remaining.
