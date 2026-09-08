"""Health check for jepa/memory.py's TransitionGraph (Stage 3's "exact
visited-transition graph"), replayed post-hoc against real recorded
gameplay rather than re-running the agent live.

Why replay instead of instrumenting the live agent: Hypothesis (current
production agent) already builds and queries this exact graph during
play (`self.graph.record(...)` on every transition, `best_known_action`/
`tried_actions` before choosing). Since the graph's hashing is a pure,
deterministic function of frame content (jepa/memory.py: _hash_frame),
replaying an existing recording through the same TransitionGraph class
reproduces exactly what the live agent's graph state would have been at
every decision point -- no need to re-run games or patch the agent.

Measures, per game and pooled:
  - state-revisit rate: of all decision points, how many are at a state
    (exact frame hash) the graph has already seen before? If this is
    near-zero, the graph structurally can't help (nothing to look up).
  - exact-repeat rate: of revisits, how many are the *exact same*
    (state, action, xy) triple repeated -- these are queries `lookup()`
    or `tried_actions()` could actually answer with something already
    known, as opposed to a state revisit with a genuinely new action.
  - correlation with the countdown-bar finding from earlier this
    session: games with a depleting counter baked into the visible frame
    would have every frame slightly different by construction, which
    would poison exact-hash matching. Cross-checked directly via
    jepa.countdown_detector, not assumed.

Usage:
    python scripts/diagnose_transition_graph_health.py --src E:/jepa_overflow/winning_harvest/recordings
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.memory import TransitionGraph
from jepa.countdown_detector import CountdownBarDetector

RESET_ACTION_ID = 0


def _load_lines(path: Path) -> list[dict]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            event = json.loads(raw)
            data = event.get("data", {})
            if "frame" in data and "action_input" in data and data["frame"]:
                lines.append(data)
    return lines


def _action_key(data: dict) -> tuple[int, tuple[int, int] | None]:
    action_input = data.get("action_input") or {}
    action_id = action_input.get("id")
    xy_data = action_input.get("data") or {}
    if action_id == 6:  # ACTION6, has an (x, y)
        return action_id, (xy_data.get("x", 0), xy_data.get("y", 0))
    return action_id, None


def replay_file(path: Path) -> dict:
    lines = _load_lines(path)
    graph = TransitionGraph()
    detector = CountdownBarDetector()

    total_decisions = 0
    state_revisits = 0
    exact_repeats = 0  # revisit AND this exact (state, action, xy) already tried
    useful_recalls = 0  # revisit where best_known_action has levels_completed_delta > 0
    last_levels_completed = 0
    countdown_confirmed_any_episode = False

    for i in range(len(lines) - 1):
        cur = lines[i]
        nxt = lines[i + 1]
        frame = cur.get("frame")
        if not frame:
            continue
        # CountdownBarDetector.observe() wants the raw (H, W) grid layer,
        # not the recording's [layer] wrapper -- and it's stateful/
        # per-episode (a real depleted-then-refilled bar looks like a
        # disqualifying "reversion" to the detector unless reset() is
        # called at episode boundaries, which is its documented usage).
        detector.observe(frame[0])

        state_key = TransitionGraph.key_for(frame)
        action_id, xy = _action_key(nxt)

        if action_id == RESET_ACTION_ID:
            if detector.urgency() is not None:
                countdown_confirmed_any_episode = True
            detector.reset()

        total_decisions += 1
        was_seen = graph.seen(state_key)
        if was_seen:
            state_revisits += 1
            tried = graph.tried_actions(state_key)
            if (action_id, xy) in tried:
                exact_repeats += 1
            best = graph.best_known_action(state_key)
            if best is not None and best[2] > 0:
                useful_recalls += 1

        levels_completed = nxt.get("levels_completed", last_levels_completed)
        levels_delta = levels_completed - last_levels_completed
        last_levels_completed = levels_completed
        graph.record(state_key, action_id, xy, nxt.get("frame"), levels_delta)

    if detector.urgency() is not None:
        countdown_confirmed_any_episode = True

    return {
        "total_decisions": total_decisions,
        "state_revisits": state_revisits,
        "exact_repeats": exact_repeats,
        "useful_recalls": useful_recalls,
        "graph_edges": len(graph),
        "has_countdown_bar": countdown_confirmed_any_episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    per_game: dict[str, dict] = defaultdict(lambda: {
        "total_decisions": 0, "state_revisits": 0, "exact_repeats": 0,
        "useful_recalls": 0, "graph_edges": 0, "has_countdown_bar": None,
    })

    for path in sorted(src.glob("*.recording.jsonl")):
        game_prefix = path.name.split("-")[0]
        stats = replay_file(path)
        g = per_game[game_prefix]
        g["total_decisions"] += stats["total_decisions"]
        g["state_revisits"] += stats["state_revisits"]
        g["exact_repeats"] += stats["exact_repeats"]
        g["useful_recalls"] += stats["useful_recalls"]
        g["graph_edges"] = max(g["graph_edges"], stats["graph_edges"])
        g["has_countdown_bar"] = stats["has_countdown_bar"]
        print(
            f"[{path.name}] decisions={stats['total_decisions']} "
            f"revisits={stats['state_revisits']} exact_repeats={stats['exact_repeats']} "
            f"useful_recalls={stats['useful_recalls']} countdown_bar={stats['has_countdown_bar']}"
        )

    print("\n=== per-game summary ===")
    print(f"{'game':6} {'decisions':>10} {'revisit%':>9} {'exact_repeat%':>14} {'useful_recall%':>15} {'countdown_bar':>14}")
    with_bar = []
    without_bar = []
    for game, g in sorted(per_game.items()):
        rev_pct = 100 * g["state_revisits"] / max(g["total_decisions"], 1)
        rep_pct = 100 * g["exact_repeats"] / max(g["total_decisions"], 1)
        rec_pct = 100 * g["useful_recalls"] / max(g["total_decisions"], 1)
        print(f"{game:6} {g['total_decisions']:>10} {rev_pct:>8.1f}% {rep_pct:>13.1f}% {rec_pct:>14.1f}% {str(g['has_countdown_bar']):>14}")
        (with_bar if g["has_countdown_bar"] else without_bar).append(rev_pct)

    total_decisions = sum(g["total_decisions"] for g in per_game.values())
    total_revisits = sum(g["state_revisits"] for g in per_game.values())
    total_exact_repeats = sum(g["exact_repeats"] for g in per_game.values())
    total_useful = sum(g["useful_recalls"] for g in per_game.values())
    print("\n=== pooled ===")
    print(f"total decisions: {total_decisions}")
    print(f"state-revisit rate: {100 * total_revisits / max(total_decisions, 1):.2f}%")
    print(f"exact-repeat rate (of all decisions): {100 * total_exact_repeats / max(total_decisions, 1):.2f}%")
    print(f"useful-recall rate (of all decisions): {100 * total_useful / max(total_decisions, 1):.2f}%")
    if with_bar:
        print(f"\nmean revisit% for games WITH a countdown bar ({len(with_bar)} games): {sum(with_bar)/len(with_bar):.2f}%")
    if without_bar:
        print(f"mean revisit% for games WITHOUT a countdown bar ({len(without_bar)} games): {sum(without_bar)/len(without_bar):.2f}%")


if __name__ == "__main__":
    main()
