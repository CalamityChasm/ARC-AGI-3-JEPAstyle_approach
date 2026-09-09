"""One-shot script that applies the stage7-duck-concurrency patch to
kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb.

Run once from repo root: python scripts/_patch_duck_notebook_concurrency.py
Idempotent: re-running after the patch is already applied is a no-op.
Also writes experiments/stage7_duck_cell9_concurrency_diff.txt.

Deliberately written in the same style, and using the same surgical
single-string-literal splice technique, as its sibling
``scripts/_patch_duck_notebook_budget.py`` -- see that script's docstring for
why a full ``json.load``/``json.dump`` round-trip is avoided (it would
reformat every other cell and turn a one-cell logic change into a
whole-file diff).

The patched block is inserted immediately BEFORE the existing
stage7-duck-budget-fix block, because the budget fix reads
``bm.solver.concurrency`` to compute its wave count -- so the override has to
land first for the two to compose (waves become ceil(n_games / TARGET), and
the per-game cap follows). ``tests/test_duck_budget.py::
test_budget_follows_the_overridden_concurrency`` pins that ordering.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

NOTEBOOK_PATH = Path("kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb")
DIFF_OUTPUT_PATH = Path("experiments/stage7_duck_cell9_concurrency_diff.txt")

# The first line of the existing budget-fix block. The concurrency override is
# spliced in above it.
ANCHOR = """        # --- stage7-duck-budget-fix ----------------------------------------
"""

PATCH = """        # --- stage7-duck-concurrency ---------------------------------------
        # The solver runs games in fixed-length waves (no game finishes early
        # -- verified: 25/25 games in a real fork run ended `gave_up` at
        # >=7900s of a 7920s cap), so waves = ceil(n_games / concurrency).
        # At 110 hidden games the inherited concurrency=28 gives 4 waves;
        # 37 would give 3.
        #
        # IMPORTANT -- this is NOT free extra compute. Total wall-clock is
        # fixed by the 9h cap and aggregate throughput is a property of the
        # vLLM server, so total tokens generated ~= throughput x wall_clock
        # REGARDLESS of concurrency. Raising it gives each game more
        # wall-clock but a proportionally thinner slice of the GPU:
        #     tokens_per_game ~= (agg_tps / conc) x (wall_clock / waves)
        # and with waves = n_games / conc the concurrency terms cancel.
        # Raising concurrency only helps if AGGREGATE THROUGHPUT RISES with
        # more concurrent sequences. That was measured on the real server --
        # see experiments/stage7_duck_concurrency.md for the throughput table
        # and the verdict, and kaggle_submission_duck/duck_budget.py
        # (resolve_concurrency / concurrency_gain_factor, unit-tested) for the
        # standalone version of this arithmetic.
        #
        # Single named constant, trivial to change. Set to None (or to the
        # inherited value) to leave bm.solver.concurrency untouched.
        DUCK_TARGET_CONCURRENCY = 28
        _inherited_conc = int(getattr(bm.solver, "concurrency", 28) or 28)
        _target_conc = DUCK_TARGET_CONCURRENCY
        if (
            _target_conc is not None
            and _target_conc > 0
            and _target_conc != _inherited_conc
            and hasattr(bm.solver, "concurrency")
        ):
            bm.solver.concurrency = int(_target_conc)
        _effective_conc = int(getattr(bm.solver, "concurrency", _inherited_conc) or _inherited_conc)
        _n_games_for_waves = len(bm.games)
        _waves_before = max(1, -(-_n_games_for_waves // _inherited_conc))
        _waves_after = max(1, -(-_n_games_for_waves // _effective_conc))
        print(
            f"rerun concurrency: inherited={_inherited_conc}, "
            f"target={_target_conc}, effective={_effective_conc}, "
            f"n_games={_n_games_for_waves}, waves {_waves_before} -> "
            f"{_waves_after}, changed={_effective_conc != _inherited_conc}",
            flush=True,
        )
        # --- end stage7-duck-concurrency -----------------------------------

        # --- stage7-duck-budget-fix ----------------------------------------
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

    if "stage7-duck-concurrency" in old_source:
        print("Patch already applied; no-op.")
        return
    if old_source.count(ANCHOR) != 1:
        raise SystemExit(
            "Expected exactly one stage7-duck-budget-fix anchor in cell 9 "
            f"(found {old_source.count(ANCHOR)}) -- the budget fix must be "
            "applied first, and only once; update ANCHOR/PATCH."
        )

    new_source = old_source.replace(ANCHOR, PATCH, 1)

    diff_lines = list(
        difflib.unified_diff(
            old_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile="cell9/source (before concurrency override)",
            tofile="cell9/source (after concurrency override)",
        )
    )
    DIFF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIFF_OUTPUT_PATH.write_text("".join(diff_lines), encoding="utf-8")
    print(f"Wrote unified diff of cell 9 to {DIFF_OUTPUT_PATH}")

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

    reparsed = json.loads(patched_raw)
    assert reparsed["cells"][9]["source"] == new_source
    for i, cell in enumerate(nb["cells"]):
        if i == 9:
            continue
        assert reparsed["cells"][i] == cell, f"cell {i} changed unexpectedly"
    assert reparsed["metadata"] == nb["metadata"]
    assert len(reparsed["cells"]) == len(nb["cells"])

    NOTEBOOK_PATH.write_text(patched_raw, encoding="utf-8")
    print(f"Patched {NOTEBOOK_PATH}: cell 9 now contains the concurrency override.")
    print(f"Raw file grew by {len(patched_raw) - len(raw)} bytes.")


if __name__ == "__main__":
    main()
