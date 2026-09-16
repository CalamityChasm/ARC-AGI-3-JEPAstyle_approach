# Stage 7 — The public-25 noise floor, and what it retracts

**A single-pass public-25 mean carries a standard error of ±2.46 points.**
That is large enough to invalidate most n=1 comparisons made in Stage 7,
including a headline conclusion of this project's own. This document records
the measurement and retracts what it breaks.

## The measurement [VERIFIED]

The wipe-guard run provided an accidental null arm. On the **16 games where the
guard is provably inert** — zero guardable wipes in *both* the anim incumbent
and the guarded run, so no code path could have touched them — paired per-game
scores still moved:

```
mean  +2.27
sd    12.30
range -19.44 … +42.86
12 of 16 games moved at all
tr87:  4.76 -> 47.62   (on a code path that cannot have run)
```

That yields **SE ≈ ±2.46 on the 25-game mean**. Detecting a +0.10 effect
against it would need roughly **3,840 free runs per arm**.

This is the first time the metric's own variance has been measured in this
project. Every prior free-run comparison assumed it was small.

## What this retracts

### 1. "Seven interventions, seven regressions" — NOT SUPPORTED

`stage7_config_locality.md` concluded that the shipped Duck configuration is a
tightly-tuned local optimum, on the strength of seven consecutive regressions.
Measured against ±2.46:

| intervention | Δ vs baseline | in SE | honest verdict |
|---|---:|---:|---|
| ctx16k | −8.68 | 3.5 | **real regression** |
| prefix caching | −5.04 | 2.0 | marginal |
| history dedupe | −3.60 | 1.5 | **indistinguishable** |
| yield180 | −3.20 | 1.3 | **indistinguishable** |
| anim graft | −0.72 | 0.3 | noise |
| wipe guard | +0.11 | 0.04 | noise |

**Only ctx16k is clearly a regression.** The local-optimum conclusion was built
on a pattern that mostly is not there. Three of the seven never ran at all
(fp8 impossible, 8 GiB OOM, upscale inert), so the real count of measured
regressions is **one**, not seven.

### 2. "Public-25 anti-predicts the hidden set" — DISSOLVED

Claimed twice, on the basis that the anim graft scored 9.97 locally (below the
10.69 baseline) and then beat it by +0.6 on the hidden set. At SE ±2.46, a
−0.72 local difference is **0.3 SE** — no signal at all. There was nothing to
anti-predict. The metric was not misleading; it was silent, and structure was
read into it.

### 3. Every other n=1 public-25 ranking in Stage 7

Including anim-vs-baseline, which is why the anim graft's real standing rests
entirely on its **hidden-set** submissions, not its free run.

## The implication that helps

| measurement | noise | relative |
|---|---|---|
| public-25, single free run | SE ±2.46 on ~10 | **~25%** |
| hidden set, one submission | sd 0.295 on 2.83 | **~10%** |

**A real submission is a less noisy measurement than a free validation run.**
Gating candidates on free public-25 runs was *adding* noise, not removing it —
and submissions are separate compute that expires daily if unused.

**Revised workflow:**
- **Free runs** remain valid for two things: catastrophe detection (ctx16k's
  −81% is 3.5 SE and real) and **mechanical telemetry** — wipes intercepted,
  actions, levels, calls — which is counted, not sampled, and therefore
  noise-free.
- **They cannot rank candidates.** Stop using them as a gate.
- **Ranking happens on the hidden set**, one submission per day, and needs
  n≥3 per arm before any claim.

## Method note worth keeping

`read_duck_public25_log.py` asserts the last progressive-summary block against
an independent per-game recomputation. On the wipe-guard run the **first**
summary block reads **0.41** against a true **10.08** — the progressive-summary
trap is real and enormous, and this assertion is what makes these logs safe to
read at all.

## Standing corrections to this project's own record

- The 74-wipes figure was a 2x event-mirroring over-count **and** a category
  error (events are not wipes): real guardable wipes are **37 baseline / 16
  anim**.
- `analyze_rhae_binding.py` counted 4,970 rows against 3,633 real actions from
  the same mirroring bug. The wasted-action figure survived re-derivation
  (**50.2% baseline, 48.6% anim**) but was re-derived, not inherited.
- The wipe guard's realised effect was **239 characters**, because 8 of 9
  interceptions guarded an already-empty world model. `bp35` shows seven
  consecutive game-overs with every wipe intercepted and empty each time: the
  fields are empty because the model never writes them, not because they are
  erased.
