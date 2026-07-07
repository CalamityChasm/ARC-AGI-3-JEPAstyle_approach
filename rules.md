# ARC Prize 2026 – ARC-AGI-3: Rules & Reference
 
Reference notes for this project. Summarized from the Kaggle competition Overview, Rules, and Dataset Description pages.
 
---
 
## Competition Overview
 
- **Goal:** Build AI systems that learn and adapt to novel, human-solvable tasks they've never seen before — measuring generalization, not memorization.
- **Three active 2026 ARC competitions:** this one (ARC-AGI-3), ARC-AGI-2, and a paper track (write-up only, for either prediction competition).
- **Total prizes:** $850,000
  - Progress Prizes: $150,000
  - Bonus (Grand) Prize: $700,000
### Timeline
| Date | Milestone |
|---|---|
| March 25, 2026 | Start Date |
| June 30, 2026 | Milestone 1 (public notebook required) |
| September 30, 2026 | Milestone 2 (public notebook required) |
| October 26, 2026 | Entry Deadline & Team Merger Deadline |
| November 2, 2026 | Final Submission Deadline |
| December 4, 2026 | Winners Announced |
 
All deadlines 11:59 PM UTC.
 
### Prizes Breakdown
- **Final Leaderboard Prizes ($75,000):** 1st $40k / 2nd $15k / 3rd $10k / 4th $5k / 5th $5k
- **Milestone Prizes ($75,000 total, two dates):**
  - Milestone 1 (June 30, 2026): 1st $25k / 2nd $7.5k / 3rd $5k
  - Milestone 2 (Sept 30, 2026): 1st $25k / 2nd $7.5k / 3rd $5k
  - Requires public notebook under open-source license by the milestone date to qualify.
- **Bonus/Grand Prize ($700,000):** Unlocked only if a team hits 100% accuracy on the leaderboard. Split among top 5 teams at 100%: 1st $350k / 2nd $175k / 3rd $70k / 4th $70k / 5th $35k.
- Prize-eligible participants **must open-source their solution** or be removed from the competition.
---
 
## Code & Submission Requirements
 
- Submissions made via **Notebooks** only (no separate submission file to build manually — it's auto-generated as long as the agent acts on the games).
- **Run-time limit:** ≤ 9 hours (CPU or GPU notebook).
- **Internet access disabled** during scored runs.
- **External data/pretrained models:** freely & publicly available ones are allowed.
- **Submission limits:** 1 submission/day; up to 2 Final Submissions selected for judging.
- **Team size:** max 8; mergers allowed up to combined submission-count limits.
- **No private sharing** of competition code outside your team (public sharing on Kaggle forums/notebooks is fine and is treated as open-sourced under an OSI license).
### Hardware
- **RTX 6000 (g4-standard-48)** machines added to the ARC-AGI-3 hardware pool.
  - Only usable for notebooks attached to this competition (misuse risks account suspension/ban).
  - **No internet** allowed on RTX sessions.
---
 
## Licensing & Winner Obligations
 
- **Winner License Type:** CC-BY 4.0 (winning submission + source code must be open-sourced under an OSI-approved license that permits commercial use).
- **Data Access/Use License:** Apache 2.0.
- Exceptions: generally commercially-available third-party software, or input data/pretrained models with an incompatible license, don't need to be relicensed.
- Winners may need to:
  - Deliver full training + inference code and environment description.
  - Provide a detailed reproducible methodology write-up (architecture, preprocessing, loss, training details, hyperparameters).
  - Participate in an interview / work with a technical writer to document the solution.
  - Sign eligibility certifications, licenses, and tax forms.
## External Data & Tools
- External data/tools allowed if **publicly available, equally accessible to all participants, and low/no cost** ("Reasonableness Standard").
- AMLT (AutoML tools) allowed if properly licensed.
## Eligibility
- Must be a registered Kaggle account holder, 18+ (or age of majority).
- Not a resident of Crimea, DNR, LNR, Cuba, Iran, North Korea, or under U.S. sanctions/export controls.
- One Kaggle account per person — no multi-accounting.
---
 
## Dataset / Environment Description (ARC-AGI-3)
 
ARC-AGI-3 is an **Interactive Reasoning Benchmark** testing exploration, memory, goal acquisition, and alignment in novel environments — not static pattern matching.
 
- Full docs: docs.arcprize.org
### How Games Work
- Agent receives **frames**: JSON with current game state + metadata.
- Each frame = a **grid up to 64×64**, integer cell values **0–15** (colors/states), origin **(0,0) top-left**.
- Agent responds with **actions**.
- Each game has **multiple levels** of increasing difficulty.
- Game states: **NOT_FINISHED**, **WIN**, **GAME_OVER**.
### Action Space (≤7 actions)
| Action | Description |
|---|---|
| RESET | Start or restart the game |
| ACTION1–ACTION5 | Simple actions (move/interact, meaning varies per game) |
| ACTION6 | Complex action requiring (x, y) coordinates |
| ACTION7 | Additional simple action |
 
Action semantics are **per-game and must be discovered through exploration** — not documented in advance.
 
### Public vs. Private Games
- **~25 public games** for development (available at arcprize.org and in `environment_files/`).
- **Private evaluation set: 110 games**, never seen by agents.
  - 55 games → Public Leaderboard
  - 55 games → Private Leaderboard
### Scoring
- Scored on two criteria:
  1. **Completion** — number of levels completed per game.
  2. **Efficiency** — action count vs. human baseline ("first-time test-testers").
- **Per-level score:** `min(human_actions / agent_actions, 1.0)`, then **squared** (e.g., raw 0.5 → 0.25).
- **Per-game score:** weighted average of level scores, weighted by level index (1-indexed — later levels count more).
- **Total score:** average of all individual game scores → final 0–100%.
- 100% = matching human-level action efficiency while winning every game (scores capped at 100%, can't exceed by being faster).
### Toolkit & Repos
- **`arc-agi` Python package** — core toolkit for interacting with environments.
- **`ARC-AGI-3-Agents` repo** — framework for building/running agents.
  - Agent implements:
    - `is_done(frames, latest_frame)` — whether to stop playing.
    - `choose_action(frames, latest_frame)` — pick the next action given current state.
  - A **Swarm** orchestrates multiple agent instances across all available games in parallel.
- **Agent lifecycle:**
  1. Get list of available games from the API.
  2. Open a scorecard (tracks performance).
  3. For each game: `RESET` → take actions per strategy.
  4. Close the scorecard when all games are complete.
### Provided Files
- `ARC-AGI-3-Agents/` — local copy of the agents repo.
- `arc_agi_3_wheels/` — package wheels for installing ARC-AGI-3.
- `environment_files/` — the 25 public game files.
---
 
## Quick Takeaways for This Project
- **9-hour run-time cap**, no internet during scored runs — matches the architecture/build-plan assumptions already in place.
- Pretrained models/external data OK if freely & publicly available (relevant to V-JEPA2/DIAMOND/etc. discarded-ideas footnote).
- Efficiency scoring (squared ratio vs. human baseline) means **action-efficiency matters a lot**, not just win/loss — relevant if later optimizing the Action Expert / planning horizon.
- Private eval = 110 unseen games (55 public LB / 55 private LB) — reinforces the "don't overfit to ~25 public games" principle in the build plan.
- Milestone 1 (June 30, 2026) requires a **public notebook** — relevant near-term deadline if aiming for milestone prizes.