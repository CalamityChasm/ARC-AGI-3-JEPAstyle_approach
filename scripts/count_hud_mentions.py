"""Counted telemetry for the HUD-perception arm: how often does the model
reason about HUD / status-bar chrome, and how often is it *deriving the
geometry* from scratch?

Why this needs its own script rather than a grep
------------------------------------------------
A raw `grep -c HUD` over a run's transcripts is **dominated by the harness's own
boilerplate**, not by the model. Measured on the champion anim run, 4,562 raw
matches split as:

    SYSTEM PROMPT  2,820   (the base VISUAL_GAME_ADDENDUM's own HUD warning,
                            repeated verbatim in all 940 system prompts)
    USER PROMPT    1,028
    THINKING         495   <- model
    ASSISTANT        157   <- model
    MODEL META        62   (duplicates the tool-call arguments; excluded)

Only **714** of those 4,562 are the model. That distinction is load-bearing for
this arm specifically: the arm *adds HUD words to the prompt*, so a raw count
would rise mechanically whatever the model did, and the mechanism check would be
meaningless. Everything reported here as a mechanism signal is restricted to
model-generated blocks.

Sections counted as model-generated: `THINKING`, `ASSISTANT`, `TOOL CALL: *`.
Deliberately excluded: `SYSTEM PROMPT`, `USER PROMPT` (harness-authored),
`ANALYZER STATUS` (harness), `TOOL RESULT: *` (sandbox stdout -- the model chose
what to print, but the board content it prints is not the model's words), and
`MODEL RESPONSE META` (its `raw_tool_calls` block duplicates `TOOL CALL`).

Usage:
    venv/Scripts/python.exe scripts/count_hud_mentions.py anim=<dir> hud=<dir> \\
        [--json experiments/stage7_hud_perception_data.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SECTION = re.compile(r"^\[([A-Z][A-Z ]*(?::[^\]]*)?)\]\s*$", re.M)

MODEL_SECTIONS = ("THINKING", "ASSISTANT", "TOOL CALL")

#: The mention itself. `status bar` never appears in this corpus but is kept so
#: the count does not silently miss it in a future run.
MENTION = re.compile(r"\bHUD\b|status[ _-]?bars?|timer[ _-]?bars?", re.I)

#: A mention that also carries explicit geometry on the same line -- a row/col
#: index, a range, or a coordinate pair. This is the expensive behaviour the arm
#: is meant to remove: re-deriving where the HUD is, turn after turn.
GEOMETRY = re.compile(
    r"\brows?\s*\d|\bcols?\s*\d|\bcolumns?\s*\d|\d+\s*[-:]\s*\d+|\(\s*\d+\s*,\s*\d+\s*\)",
    re.I,
)

#: The new field's own name. Reported separately and explicitly NOT counted as a
#: mechanism win: in the HUD arm the model discussing `hud_node_ids` is expected
#: and says nothing about whether it stopped re-deriving geometry by hand.
FIELD = re.compile(r"hud_node_ids|\[[\"']hud[\"']\]|\.hud\b|\bhud\s*[=:]\s*(?:True|False)")


def _blocks(text: str) -> list[tuple[str, str]]:
    marks = [(m.start(), m.end(), m.group(1)) for m in SECTION.finditer(text)]
    out = []
    for i, (start, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((name, text[end:stop]))
    return out


def _is_model(name: str) -> bool:
    return any(name == s or name.startswith(s + ":") for s in MODEL_SECTIONS)


def scan(run: Path) -> dict[str, Any]:
    per_game: dict[str, dict[str, int]] = {}
    sections: dict[str, dict[str, int]] = {}
    for path in sorted((run / "transcripts").glob("*_p0.txt")):
        game = path.name.split("_p0")[0]
        text = path.read_text(encoding="utf-8", errors="replace")
        stats = {
            "mentions_model": 0,
            "mentions_geometry": 0,
            "mentions_all_sections": len(MENTION.findall(text)),
            "field_references": 0,
            "model_blocks": 0,
        }
        for name, body in _blocks(text):
            slot = sections.setdefault(name.split(":")[0], {"blocks": 0, "mentions": 0})
            slot["blocks"] += 1
            hits = MENTION.findall(body)
            slot["mentions"] += len(hits)
            if not _is_model(name):
                continue
            stats["model_blocks"] += 1
            stats["mentions_model"] += len(hits)
            stats["field_references"] += len(FIELD.findall(body))
            for line in body.splitlines():
                if MENTION.search(line) and GEOMETRY.search(line):
                    stats["mentions_geometry"] += 1
        per_game[game] = stats

    total = {k: sum(g[k] for g in per_game.values()) for k in next(iter(per_game.values()))}
    total["games"] = len(per_game)
    total["games_mentioning"] = sum(1 for g in per_game.values() if g["mentions_model"])
    total["games_mentioning_any_section"] = sum(
        1 for g in per_game.values() if g["mentions_all_sections"]
    )
    return {"dir": str(run), "total": total, "sections": sections, "games": per_game}


def report(runs: dict[str, dict[str, Any]]) -> None:
    names = list(runs)

    def row(label: str, fmt, get) -> None:
        print(f"{label:<40}" + "  ".join(f"{fmt(get(n)):>14}" for n in names))

    print("\n" + " " * 40 + "  ".join(f"{n:>14}" for n in names))
    print("-" * (40 + 16 * len(names)))
    row("HUD mentions, MODEL-GENERATED only", str, lambda n: runs[n]["total"]["mentions_model"])
    row("  of which derive geometry", str, lambda n: runs[n]["total"]["mentions_geometry"])
    row("  per model block", lambda v: f"{v:.3f}",
        lambda n: runs[n]["total"]["mentions_model"] / max(1, runs[n]["total"]["model_blocks"]))
    row("references to the new hud field", str,
        lambda n: runs[n]["total"]["field_references"])
    row("games mentioning HUD (model)", lambda v: f"{v[0]} of {v[1]}",
        lambda n: (runs[n]["total"]["games_mentioning"], runs[n]["total"]["games"]))
    print()
    row("raw matches, ALL sections (noisy)", str,
        lambda n: runs[n]["total"]["mentions_all_sections"])
    row("  games, any section", lambda v: f"{v[0]} of {v[1]}",
        lambda n: (runs[n]["total"]["games_mentioning_any_section"], runs[n]["total"]["games"]))

    print("\n-- raw matches by transcript section (the reason the raw total is not the signal) --")
    keys = sorted({k for r in runs.values() for k in r["sections"]})
    print(f"{'section':<24}" + "  ".join(f"{n:>22}" for n in names))
    for key in keys:
        cells = []
        for n in names:
            s = runs[n]["sections"].get(key, {"blocks": 0, "mentions": 0})
            cells.append(f"{s['mentions']:>7} in {s['blocks']:>5} blk")
        print(f"{key:<24}" + "  ".join(f"{c:>22}" for c in cells))

    print("\n-- per game, model-generated mentions --")
    allg = sorted({g for r in runs.values() for g in r["games"]})
    print(f"{'game':<16}" + "  ".join(f"{n:>22}" for n in names))
    for game in allg:
        cells = []
        for n in names:
            d = runs[n]["games"].get(game)
            cells.append("-" if d is None else f"{d['mentions_model']:>5} ({d['mentions_geometry']:>4} geom)")
        print(f"{game:<16}" + "  ".join(f"{c:>22}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="label=<unpacked kernel output dir>")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = scan(Path(path))
    report(runs)
    if args.json:
        args.json.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
