"""One-shot script that applies the stage7-duck-budget-fix patch to
kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb.

Run once from repo root: python scripts/_patch_duck_notebook_budget.py
Idempotent: re-running after the patch is already applied is a no-op.
Also writes experiments/stage7_duck_cell9_diff.txt: a human-readable
unified diff of cell 9's actual Python source (old vs new), since the
notebook itself is stored by Kaggle as a single minified JSON line and a
`git diff` of the .ipynb file will never show line-level code changes
either way -- see experiments/stage7_duck_budget_fix.md.

Does a surgical, single-string-literal replacement of cell 9's own
``source`` field inside the notebook's raw (compact, single-line) JSON
text, rather than round-tripping the whole notebook through
``json.load``/``json.dump`` -- a full re-dump (even of unchanged data)
would reformat every other cell's whitespace too, turning a one-cell
logic change into a diff that touches the entire file. This keeps the
change limited to exactly the bytes that changed.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

NOTEBOOK_PATH = Path("kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb")
DIFF_OUTPUT_PATH = Path("experiments/stage7_duck_cell9_diff.txt")

ANCHOR = """        bm.games = _competition_games()
        bm.n_passes = 1
        bm.game_weights = None
"""

# Mirrors kaggle_submission_duck/duck_budget.py:compute_rerun_budget exactly
# (same waves/remaining/per_game/floor arithmetic) -- kept as an inline copy
# because a Kaggle kernel has no import path back into this repo. Only
# reachable when true_submission is True, so it is a no-op for the public
# 25-game path (cell 8's own concurrency/max_runtime_s_per_game override,
# which only applies "if not true_submission", is untouched).
PATCH = """        bm.games = _competition_games()
        bm.n_passes = 1
        bm.game_weights = None

        # --- stage7-duck-budget-fix ----------------------------------------
        # Inherited concurrency=28 / max_runtime_s_per_game=7920.0 (from the
        # pickled bundle, tuned for a 6-game x 4-pass Preview shape) does not
        # fit the real ~110-game competition rerun: every game observed in a
        # real run burns its entire per-game cap rather than finishing early
        # (verified: 25/25 games ended `gave_up` at >=7900s in our own fork's
        # run), so ceil(110/28)=4 waves x 7920s = 8.80h -- plus ~394s measured
        # setup and up to 600s of rerun-only gateway wait -- against a 9h hard
        # cap with no soft deadline in this branch (_soft_end_time() returns
        # None here). Recompute max_runtime_s_per_game from the live game
        # count and elapsed wall-clock instead of using the inherited
        # constant, so the last wave has a chance to fit. No-op on the public
        # path (only reachable when true_submission is True). See
        # experiments/stage7_duck_budget_fix.md and
        # kaggle_submission_duck/duck_budget.py (unit-tested standalone
        # version of this same arithmetic) for the full derivation.
        _n_games = len(bm.games)
        _elapsed_s = time.time() - NOTEBOOK_START_EPOCH
        _conc = int(getattr(bm.solver, "concurrency", 28) or 28)
        _waves = max(1, -(-_n_games // _conc))  # ceil division, no math import
        _remaining_s = 9 * 3600 - _elapsed_s - 900.0  # 15-minute safety margin
        _per_game_raw_s = _remaining_s / _waves
        _per_game_s = max(600.0, _per_game_raw_s)  # floor: never collapse to ~0
        if hasattr(bm.solver, "max_runtime_s_per_game"):
            bm.solver.max_runtime_s_per_game = _per_game_s
        print(
            f"rerun budget: n_games={_n_games}, conc={_conc}, waves={_waves}, "
            f"per_game={_per_game_s:.0f}s, remaining={_remaining_s:.0f}s",
            flush=True,
        )
        # --- end stage7-duck-budget-fix ------------------------------------
"""


def main() -> None:
    raw = NOTEBOOK_PATH.read_text(encoding="utf-8")
    nb = json.loads(raw)
    old_source = nb["cells"][9]["source"]
    if not isinstance(old_source, str):
        raise SystemExit(
            "Expected cell 9's source to be a single JSON string (this "
            "notebook's own storage convention) -- got a list instead; "
            "the notebook format has changed, update this script."
        )

    if PATCH in old_source:
        print("Patch already applied; no-op.")
        return
    if ANCHOR not in old_source:
        raise SystemExit(
            "Anchor text not found in cell 9 -- notebook structure has "
            "changed since this patch was written; update ANCHOR/PATCH."
        )

    new_source = old_source.replace(ANCHOR, PATCH, 1)

    # Human-readable unified diff of the actual Python code, for review
    # (git's own diff of the .ipynb is useless -- see module docstring).
    diff_lines = list(
        difflib.unified_diff(
            old_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile="cell9/source (before)",
            tofile="cell9/source (after)",
        )
    )
    DIFF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIFF_OUTPUT_PATH.write_text("".join(diff_lines), encoding="utf-8")
    print(f"Wrote unified diff of cell 9 to {DIFF_OUTPUT_PATH}")

    # Re-encode exactly as the notebook's own JSON encoder would (compact,
    # non-ASCII characters left literal -- matches the em-dashes etc.
    # already present unescaped elsewhere in this file) and splice only
    # that one string literal into the raw text, leaving every other byte
    # of the file untouched.
    old_literal = json.dumps(old_source, ensure_ascii=False)
    new_literal = json.dumps(new_source, ensure_ascii=False)

    occurrences = raw.count(old_literal)
    if occurrences != 1:
        raise SystemExit(
            f"Expected exactly one occurrence of cell 9's source literal in "
            f"the raw file, found {occurrences} -- refusing to guess which "
            f"one to replace."
        )

    patched_raw = raw.replace(old_literal, new_literal, 1)

    # Round-trip check: reparsing must yield the same notebook except for
    # this one cell's source, so a byte-level splice can't have silently
    # corrupted the surrounding JSON.
    reparsed = json.loads(patched_raw)
    assert reparsed["cells"][9]["source"] == new_source
    for i, cell in enumerate(nb["cells"]):
        if i == 9:
            continue
        assert reparsed["cells"][i] == cell, f"cell {i} changed unexpectedly"
    assert reparsed["metadata"] == nb["metadata"]
    assert len(reparsed["cells"]) == len(nb["cells"])

    NOTEBOOK_PATH.write_text(patched_raw, encoding="utf-8")
    print(f"Patched {NOTEBOOK_PATH}: cell 9 now contains the budget fix.")
    print(f"Raw file grew by {len(patched_raw) - len(raw)} bytes.")


if __name__ == "__main__":
    main()
