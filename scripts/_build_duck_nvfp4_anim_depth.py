"""Build the two stage7-depth arms: the anim graft, plus exactly one new cell each.

The incumbent is ``kaggle_submission_duck_nvfp4_anim`` (kernel
``calamitychasm/arc3-duck-nvfp4-anim`` v1, public-25 **9.97 / 2,615 actions /
42 levels**, real **3.43 / 3.79 / 3.37**). Both arms are built on top of it.

Arms
----
``commitfloor``  ``scripts/commit_floor_cell.py``  -> ``arc3-duck-nvfp4-anim-cf``
``nothink``      ``scripts/no_thinking_cell.py``   -> ``arc3-duck-nvfp4-anim-nt``

What changes, exhaustively, for either arm
------------------------------------------
1. One markdown heading + one code cell are **inserted** after the anim
   notebook's customization hook (cell 13), holding the arm's cell verbatim.
2. ``kernel-metadata.json`` gets the new kernel id/title. Every mount, the
   docker image and ``machine_shape`` are carried over untouched.

Everything else -- all 18 source cells, in order, **by per-cell sha256** -- is
asserted unchanged. `stage7_config_locality.md` is the reason: a prior attempt
carried an unrelated ``seqs=16`` through three runs and confounded the only one
that ran. This is a mechanical check, not an intention.

Usage
-----
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim_depth.py [--arm all] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_NB = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "arc3-duck-nvfp4-anim.ipynb"
SRC_META = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "kernel-metadata.json"

# The cell the arm is inserted after: the anim notebook's customization hook.
ANCHOR_INDEX = 13
ANCHOR_MARKER = "PUBLIC25_SETTINGS"

ARMS = {
    "commitfloor": {
        "cell": REPO / "scripts" / "commit_floor_cell.py",
        "dir": REPO / "kaggle_submission_duck_nvfp4_anim_commitfloor" / "notebook",
        "nb": "arc3-duck-nvfp4-anim-cf.ipynb",
        "kernel_id": "calamitychasm/arc3-duck-nvfp4-anim-cf",
        "kernel_title": "arc3-duck-nvfp4-anim-cf",
        # One variable: this arm is a pure method wrapper. It may not touch the
        # environment, the analyzer knobs or the solver settings.
        "forbidden": ("os.environ", "LOCAL_ANALYZER", "bm.solver", "SETUP_ENV_PATH"),
        "heading": (
            "## 6b. Commit floor\n"
            "\n"
            "The single behavioural change in this notebook relative to `arc3-duck-nvfp4-anim`.\n"
            "46.1% of this chassis's wall clock goes on `analyze()` turns that execute no game action,\n"
            "and `solver.py:360-361` replays the same analysis step after every one of them -- so a game\n"
            "that stops committing burns the rest of its fixed 7,920s without touching the board\n"
            "(`r11l` lost its last 4,703s that way; `cn04` 4,750s; `ft09`, the run's best game, 1,771s).\n"
            "After two consecutive non-acting turns this adds a directive to the user prompt requiring\n"
            "the turn to end in `action(...)`, escalating after five. No forced action, no reset, no knob.\n"
        ),
    },
    "nothink": {
        "cell": REPO / "scripts" / "no_thinking_cell.py",
        "dir": REPO / "kaggle_submission_duck_nvfp4_anim_nothink" / "notebook",
        "nb": "arc3-duck-nvfp4-anim-nt.ipynb",
        "kernel_id": "calamitychasm/arc3-duck-nvfp4-anim-nt",
        "kernel_title": "arc3-duck-nvfp4-anim-nt",
        # This arm IS a knob, so `LOCAL_ANALYZER` is expected -- but it must
        # still be a single post-import rebind, not an env or solver change.
        "forbidden": ("os.environ", "bm.solver", "SETUP_ENV_PATH", "LOCAL_ANALYZER_YIELD", "LOCAL_ANALYZER_SEED"),
        "heading": (
            "## 6b. No-thinking arm\n"
            "\n"
            "The single behavioural change in this notebook relative to `arc3-duck-nvfp4-anim`.\n"
            "95.0% of this run's generated text is the reasoning block (4,768,656 chars against 250,663\n"
            "of content, over 1,358 model responses), and generation is what makes a turn exceed the 180s\n"
            "yield budget before the model acts -- the cause of the 46.1% of wall clock that executes\n"
            "nothing. This rebinds `tool_agent._LOCAL_ANALYZER_ENABLE_THINKING` to `False` after import,\n"
            "which `_chat_completion` reads per call and turns into the vLLM `enable_thinking` switch.\n"
        ),
    },
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(arm: str, check_only: bool = False) -> None:
    spec = ARMS[arm]
    out_nb = spec["dir"] / spec["nb"]
    out_meta = spec["dir"] / "kernel-metadata.json"

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == 18, f"anim notebook has {len(cells)} cells, expected 18"

    anchor = "".join(cells[ANCHOR_INDEX]["source"])
    assert cells[ANCHOR_INDEX]["cell_type"] == "code", "anchor is not a code cell"
    assert ANCHOR_MARKER in anchor, f"cell {ANCHOR_INDEX} is not the customization hook"

    inherited = [_digest("".join(c["source"])) for c in cells]

    cell_src = spec["cell"].read_text(encoding="utf-8")
    for forbidden in spec["forbidden"]:
        assert forbidden not in cell_src, f"{arm}: cell must not contain {forbidden!r} -- one variable"

    new_cells = (
        cells[: ANCHOR_INDEX + 1]
        + [
            {"cell_type": "markdown", "metadata": {}, "source": spec["heading"].splitlines(keepends=True)},
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
    meta["id"] = spec["kernel_id"]
    meta["title"] = spec["kernel_title"]
    meta["code_file"] = out_nb.name
    assert meta["dataset_sources"] == src_meta["dataset_sources"]
    assert meta["model_sources"] == src_meta["model_sources"]
    assert meta["competition_sources"] == src_meta["competition_sources"]
    assert meta["docker_image"] == src_meta["docker_image"]
    assert meta["enable_internet"] is False and meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaRtxPro6000"

    nb_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"

    if check_only:
        assert out_nb.exists(), f"{arm}: notebook not built"
        assert out_nb.read_text(encoding="utf-8") == nb_text, f"{arm}: built notebook is stale"
        assert out_meta.read_text(encoding="utf-8") == meta_text, f"{arm}: built metadata is stale"
        print(f"{arm}: up to date  ({spec['kernel_id']}, cell sha256={_digest(cell_src)[:16]})")
        return

    spec["dir"].mkdir(parents=True, exist_ok=True)
    out_nb.write_text(nb_text, encoding="utf-8")
    out_meta.write_text(meta_text, encoding="utf-8")
    print(f"built {out_nb}")
    print(f"  from      {src_meta['id']} ({len(cells)} cells, all inherited cells asserted unchanged)")
    print(f"  inserted  1 markdown + 1 code cell after index {ANCHOR_INDEX} ({ANCHOR_MARKER})")
    print(f"  cell      {spec['cell'].name} sha256={_digest(cell_src)[:16]}")
    print(f"  kernel    {spec['kernel_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="all", choices=["all", *ARMS])
    ap.add_argument("--check", action="store_true", help="verify the built arms are current; build nothing")
    args = ap.parse_args()
    for arm in ARMS if args.arm == "all" else [args.arm]:
        build(arm, check_only=args.check)


if __name__ == "__main__":
    main()
