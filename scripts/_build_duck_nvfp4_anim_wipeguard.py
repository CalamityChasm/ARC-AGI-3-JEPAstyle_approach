"""Build the wipe-guard arm: the anim graft, plus exactly one new cell.

The incumbent is ``kaggle_submission_duck_nvfp4_anim`` (kernel
``calamitychasm/arc3-duck-nvfp4-anim`` v1, public-25 **9.97 / 2,615 actions**,
real **3.43**). This builds its treatment arm.

What changes, exhaustively
--------------------------
1. One code cell is **inserted** after the customization hook (cell 13), holding
   ``scripts/wipe_guard_cell.py`` verbatim, plus its markdown heading.
2. ``kernel-metadata.json`` gets the new kernel id/title. Every mount, the
   docker image and ``machine_shape`` are carried over untouched.

Everything else -- all 18 source cells, in order, byte-for-byte -- is asserted
unchanged. `stage7_config_locality.md` is the reason: a prior attempt carried an
unrelated ``seqs=16`` through three runs and confounded the only one that ran.
This script exists so that cannot happen silently.

The guard is installed at the customization hook rather than beside the knob
overrides in cell 9 on purpose. Cell 9's overrides must precede the ``inference``
import because ``tool_agent`` reads those knobs into module globals at import
time. A method wrapper has no such constraint -- it is resolved per call, and
``ToolAgent`` instances are constructed later, per game, inside the solver -- so
installing it here keeps the graft's own teeth untouched.

Usage
-----
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim_wipeguard.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_NB = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "arc3-duck-nvfp4-anim.ipynb"
SRC_META = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "kernel-metadata.json"
GUARD_CELL = REPO / "scripts" / "wipe_guard_cell.py"

OUT_DIR = REPO / "kaggle_submission_duck_nvfp4_anim_wipeguard" / "notebook"
OUT_NB = OUT_DIR / "arc3-duck-nvfp4-anim-wg.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

KERNEL_ID = "calamitychasm/arc3-duck-nvfp4-anim-wg"
KERNEL_TITLE = "arc3-duck-nvfp4-anim-wg"

# The cell the guard is inserted after: the anim notebook's customization hook.
ANCHOR_INDEX = 13
ANCHOR_MARKER = "PUBLIC25_SETTINGS"

HEADING = (
    "## 6b. World-model wipe guard\n"
    "\n"
    "The single behavioural change in this notebook relative to `arc3-duck-nvfp4-anim`.\n"
    "`tool_agent.py:1343-1356` erases six of the seven summarized-knowledge fields on every\n"
    "level transition, run completion **and game over** -- but a game over auto-RESETs into the\n"
    "*same* level, so those six fields were still true. This keeps them on an in-level game over\n"
    "and leaves the other two triggers alone. Exception-guarded, and probed at startup.\n"
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

    # Record what we inherited, so a silent upstream edit cannot pass unnoticed.
    inherited = [_digest("".join(c["source"])) for c in cells]

    guard_src = GUARD_CELL.read_text(encoding="utf-8")
    assert "os.environ" not in guard_src, "guard cell must not touch env -- one variable"
    assert "LOCAL_ANALYZER" not in guard_src, "guard cell must not touch analyzer knobs"
    assert "bm.solver" not in guard_src, "guard cell must not touch solver settings"

    new_cells = (
        cells[: ANCHOR_INDEX + 1]
        + [
            {"cell_type": "markdown", "metadata": {}, "source": HEADING.splitlines(keepends=True)},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": guard_src.splitlines(keepends=True),
            },
        ]
        + cells[ANCHOR_INDEX + 1 :]
    )
    nb["cells"] = new_cells

    # Post-condition: every inherited cell is still present, in order, unchanged.
    got = [_digest("".join(c["source"])) for c in new_cells]
    kept = got[: ANCHOR_INDEX + 1] + got[ANCHOR_INDEX + 3 :]
    assert kept == inherited, "an inherited cell changed -- refusing to build a confounded arm"
    assert len(new_cells) == 20, len(new_cells)

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    src_id = meta["id"]
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_TITLE
    assert meta["dataset_sources"] == json.loads(SRC_META.read_text(encoding="utf-8"))["dataset_sources"]
    assert meta["enable_internet"] is False and meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaRtxPro6000"

    nb_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"

    if check_only:
        assert OUT_NB.exists(), "notebook not built"
        assert OUT_NB.read_text(encoding="utf-8") == nb_text, "built notebook is stale"
        assert OUT_META.read_text(encoding="utf-8") == meta_text, "built metadata is stale"
        print("wipe-guard arm is up to date")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(nb_text, encoding="utf-8")
    OUT_META.write_text(meta_text, encoding="utf-8")
    print(f"built {OUT_NB}")
    print(f"  from      {src_id} ({len(cells)} cells, all inherited cells asserted unchanged)")
    print(f"  inserted  1 markdown + 1 code cell after index {ANCHOR_INDEX} ({ANCHOR_MARKER})")
    print(f"  guard     {GUARD_CELL.name} sha256={_digest(guard_src)[:16]}")
    print(f"  kernel    {KERNEL_ID}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the built arm is current; build nothing")
    build(check_only=ap.parse_args().check)


if __name__ == "__main__":
    main()
