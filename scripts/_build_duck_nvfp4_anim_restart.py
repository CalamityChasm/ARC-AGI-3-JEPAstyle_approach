"""Build the restart-at-stall arm: the anim graft, plus exactly one new cell.

The incumbent is ``kaggle_submission_duck_nvfp4_anim`` (kernel
``calamitychasm/arc3-duck-nvfp4-anim`` v1, public-25 **9.97 / 2,615 actions**,
real **3.43**). This builds its treatment arm.

What changes, exhaustively
--------------------------
1. One code cell is **inserted** after the customization hook (cell 13), holding
   ``scripts/restart_at_stall_cell.py`` verbatim, plus its markdown heading.
2. ``kernel-metadata.json`` gets the new kernel id/title. Every mount, the
   docker image and ``machine_shape`` are carried over untouched.

Everything else -- all 18 source cells, in order, byte-for-byte -- is asserted
unchanged. ``stage7_config_locality.md`` is the reason: a prior attempt carried
an unrelated ``seqs=16`` through three runs and confounded the only one that
ran. This script exists so that cannot happen silently.

The cell is installed at the customization hook rather than beside the knob
overrides in cell 9 on purpose. Cell 9's overrides must precede the ``inference``
import because ``tool_agent`` reads those knobs into module globals at import
time. A method wrapper has no such constraint -- it is resolved per call, and
``ToolAgent`` instances are constructed later, per game, inside the solver.

One variable
------------
The cell rebinds ``_LOCAL_ANALYZER_SEED`` at runtime, which is intrinsic to the
mechanism (against a deterministic environment an unseeded re-draw would repeat
the same opening) and only ever happens *after* a restart fires. The forbidden
list below therefore permits ``_LOCAL_ANALYZER_SEED`` and nothing else in that
family: no env writes, no solver settings, no other analyzer knob.

Usage
-----
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim_restart.py
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim_restart.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_NB = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "arc3-duck-nvfp4-anim.ipynb"
SRC_META = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "kernel-metadata.json"
CELL = REPO / "scripts" / "restart_at_stall_cell.py"

OUT_DIR = REPO / "kaggle_submission_duck_nvfp4_anim_restart" / "notebook"
OUT_NB = OUT_DIR / "arc3-duck-nvfp4-anim-rs.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

KERNEL_ID = "calamitychasm/arc3-duck-nvfp4-anim-rs"
KERNEL_TITLE = "arc3-duck-nvfp4-anim-rs"

ANCHOR_INDEX = 13
ANCHOR_MARKER = "PUBLIC25_SETTINGS"

# Config surfaces the cell must not touch. `_LOCAL_ANALYZER_SEED` is the one
# analyzer global the mechanism legitimately moves, so the seed check is a
# precise carve-out rather than a blanket "LOCAL_ANALYZER" ban.
FORBIDDEN = (
    "os.environ",
    "bm.solver",
    "SETUP_ENV_PATH",
    "LOCAL_ANALYZER_YIELD",
    "LOCAL_ANALYZER_CONTEXT_WINDOW",
    "LOCAL_ANALYZER_ENABLE_THINKING",
    "LOCAL_ANALYZER_TEMPERATURE",
    "LOCAL_ANALYZER_TOP_P",
    "LOCAL_ANALYZER_TOP_K",
    "LOCAL_ANALYZER_MODEL_ID",
    "LOCAL_ANALYZER_TOOL_STEPS",
    "max_num_seqs",
    "concurrency",
    "max_runtime_s",
    "n_passes",
)

HEADING = (
    "## 6b. Restart-at-stall\n"
    "\n"
    "The single behavioural change in this notebook relative to `arc3-duck-nvfp4-anim`.\n"
    "48.6% of that run's actions went into a level that never completed, and 80.5% of the\n"
    "actions inside those stalls exactly repeat an (action, level) pair already tried on that\n"
    "level. After 20 distinct analysis turns on one level with no level change this clears the\n"
    "agent's level-local belief state and bumps the sampling seed, so the stalled level gets a\n"
    "fresh draw. `cross_level_notes` and the token counters are preserved; the game is untouched.\n"
    "Exception-guarded, source-asserted against upstream, and probed at startup.\n"
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(check_only: bool = False) -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == 18, f"anim notebook has {len(cells)} cells, expected 18"

    anchor = "".join(cells[ANCHOR_INDEX]["source"])
    assert cells[ANCHOR_INDEX]["cell_type"] == "code", "anchor is not a code cell"
    assert ANCHOR_MARKER in anchor, f"cell {ANCHOR_INDEX} is not the customization hook"

    inherited = [_digest("".join(c["source"])) for c in cells]

    cell_src = CELL.read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token not in cell_src, f"cell touches {token} -- that is a second variable"
    # The only permitted analyzer global, and only via the module object.
    stray = [
        m.group(0)
        for m in re.finditer(r"\b_?LOCAL_ANALYZER_[A-Z_]+\b", cell_src)
        if m.group(0) not in ("_LOCAL_ANALYZER_SEED", "LOCAL_ANALYZER_SEED")
    ]
    assert not stray, f"cell references analyzer knobs beyond the seed: {sorted(set(stray))}"
    assert "import inference.agent.tool_agent" in cell_src, "cell does not bind the solver module"

    new_cells = (
        cells[: ANCHOR_INDEX + 1]
        + [
            {"cell_type": "markdown", "metadata": {}, "source": HEADING.splitlines(keepends=True)},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_src.splitlines(keepends=True),
            },
        ]
        + cells[ANCHOR_INDEX + 1 :]
    )
    nb["cells"] = new_cells

    got = [_digest("".join(c["source"])) for c in new_cells]
    kept = got[: ANCHOR_INDEX + 1] + got[ANCHOR_INDEX + 3 :]
    assert kept == inherited, "an inherited cell changed -- refusing to build a confounded arm"
    assert len(new_cells) == 20, len(new_cells)

    src_meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_TITLE
    meta["code_file"] = OUT_NB.name
    assert meta["dataset_sources"] == src_meta["dataset_sources"]
    assert meta["competition_sources"] == src_meta["competition_sources"]
    assert meta["model_sources"] == src_meta["model_sources"]
    assert meta["enable_internet"] is False and meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaRtxPro6000"

    nb_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"

    if check_only:
        assert OUT_NB.exists(), "notebook not built"
        assert OUT_NB.read_text(encoding="utf-8") == nb_text, "built notebook is stale"
        assert OUT_META.read_text(encoding="utf-8") == meta_text, "built metadata is stale"
        print("restart-at-stall arm is up to date")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(nb_text, encoding="utf-8")
    OUT_META.write_text(meta_text, encoding="utf-8")
    print(f"built {OUT_NB}")
    print(f"  from      {src_meta['id']} ({len(cells)} cells, all inherited cells asserted unchanged)")
    print(f"  inserted  1 markdown + 1 code cell after index {ANCHOR_INDEX} ({ANCHOR_MARKER})")
    print(f"  cell      {CELL.name} sha256={_digest(cell_src)[:16]}")
    print(f"  kernel    {KERNEL_ID}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="verify the built arm is current; build nothing")
    build(check_only=ap.parse_args().check)


if __name__ == "__main__":
    main()
