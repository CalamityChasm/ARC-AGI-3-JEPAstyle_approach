"""Build the HUD-perception arm's kernel: the anim notebook, mounting our own
patched copy of the anim solver bundle instead of the upstream one.

One variable. The model, the NVFP4 runtime, the serving profile, `concurrency`,
`max_runtime_s_per_game`, `LOCAL_ANALYZER_*`, the KV bytes, the context length
and the run loop are all inherited byte-identically -- every one of the 18
inherited cells is sha256-recorded before the patch and re-checked after, and
the build refuses if any cell other than the two named ones differs.

What differs from the incumbent
-------------------------------
1. `DATASET_SOURCES[2]` points at `calamitychasm/taaf-anim-hud-v1` instead of
   `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`. That dataset is the
   upstream bundle with four files patched and 71 files byte-identical; see
   `scripts/_build_hud_bundle.py`. Its `benchmark_label` is unchanged
   (`anim-20260807-anim`), so cell 7's `_find_bundle_dir` line, cell 9's graft
   teeth and cell 11's chassis asserts stay byte-identical and keep meaning
   what they meant.
2. `ARC3_HUD_ANNOTATION=1` is exported in cell 7, i.e. **before** cell 9 imports
   `inference`. That ordering is load-bearing: `prompts.py`'s constants and
   `tool_agent.py`'s `_PYTHON_TOOL_DESCRIPTION` are built at import time, so a
   flag set after the import would silently annotate the segmentation while
   leaving the prompt describing a field the model was never told about.

Why cell 7 and not a cell of its own after the graft (which is where the AVO arm
put its variable): AVO's flag is read per game inside the solver, so it could be
set late. This one cannot.

Traps checked
-------------
* `deploy_target.pkl` is inherited unmodified from the anim bundle, so
  `max_runtime_s` is already 32400.0 and cell 13's assert passes untouched.
  There is no 54000.0 to override here -- that was the AVO bundle's. Cell 13 is
  byte-identical and the build asserts it.
* The inserted verification cell **fails the run at setup** rather than at hour
  nine if the mounted bundle turns out to be the unpatched upstream one.

Usage
-----
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_hud.py
    venv/Scripts/python.exe scripts/_build_duck_nvfp4_hud.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook"
SRC_NB = SRC_DIR / "arc3-duck-nvfp4-anim.ipynb"
SRC_META = SRC_DIR / "kernel-metadata.json"

OUT_DIR = REPO / "kaggle_submission_duck_nvfp4_hud" / "notebook"
OUT_NB = OUT_DIR / "arc3-duck-nvfp4-hud.ipynb"
KERNEL = "calamitychasm/arc3-duck-nvfp4-hud"

ANIM_DATASET = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
HUD_DATASET = "calamitychasm/taaf-anim-hud-v1"

# The anim notebook as inherited. Drift here means the base moved and every
# telemetry comparison against the incumbent's free run is off.
EXPECTED_SRC_SHA256 = "4fdd395c6974240dbfa7d463e12ce1e10dafdf4000e9f1201b9733a4c79e8220"

ANCHOR_INDEX = 13
ANCHOR_MARKER = "PUBLIC25_SETTINGS"

CELL_PATCHES: list[tuple[int, str, str]] = [
    # --- cell 7: the mount -------------------------------------------------
    (7, f'"{ANIM_DATASET}"]', f'"{HUD_DATASET}"]'),
    # --- cell 7: the arm's one variable, set before cell 9 imports anything -
    (
        7,
        "# Kaggle inputs attached to this notebook, plus bookkeeping paths used below.\n",
        "# [calamitychasm] THE ONE VARIABLE OF THIS ARM.\n"
        "# Advisory HUD annotation on `current_frame.segmentation`: each node gains a\n"
        "# `hud` bool and the dict gains `hud_node_ids`. Nothing is removed, masked or\n"
        "# altered -- the flag is a prior the model can override, not a filter.\n"
        "#\n"
        "# Set HERE, in cell 7, and not in a cell after the graft: prompts.py's constants\n"
        "# and tool_agent.py's _PYTHON_TOOL_DESCRIPTION are built at IMPORT time, and cell 9\n"
        "# imports `inference`. A flag set later would annotate the segmentation while\n"
        "# leaving the prompt silent about a field the model was never told exists.\n"
        'os.environ["ARC3_HUD_ANNOTATION"] = "1"\n'
        "\n"
        "# Kaggle inputs attached to this notebook, plus bookkeeping paths used below.\n",
    ),
    # --- cell 7: prove the mount is the patched copy, at setup, not at hour 9
    (
        7,
        'print(f"thui-animfast: anim bundle = {ANIM_BUNDLE_DIR}", flush=True)\n',
        "# [calamitychasm] The bundle keeps the anim benchmark_label on purpose (it IS the anim\n"
        "# bundle plus a four-file patch), so the label alone cannot tell the two apart. These\n"
        "# two files exist only in our patched copy.\n"
        'assert (ANIM_BUNDLE_DIR / "HUD_PATCH.md").is_file(), (\n'
        '    f"HUD arm: mounted bundle is not the patched copy: {ANIM_BUNDLE_DIR}")\n'
        'assert (ANIM_BUNDLE_DIR / "THIRD_PARTY_NOTICE.md").is_file(), ANIM_BUNDLE_DIR\n'
        'print(f"thui-animfast: anim bundle = {ANIM_BUNDLE_DIR}", flush=True)\n',
    ),
]

ARM_HEADING = """\
## 6b. The HUD-perception arm

The one behavioural variable in this notebook relative to the arm it is compared
against, verified here rather than assumed.

`stage7_avo.md` retired the dead-turn lead: three mechanisms bought the model
more actions and none bought score, because the extra actions went into levels
that were never solvable. This arm targets the other half -- whether a level is
solvable at all. Across the incumbent's own transcripts, HUD/status-bar
reasoning appears in **all 25 of 25 games**, 4,608 times, and the model
repeatedly re-derives HUD geometry from scratch, misattributes HUD pixels to
game objects, and spends whole turns establishing "did only the HUD change?".

The annotation is **advisory and additive**: a rule-based flag on nodes the
model already receives. Nothing is masked, hidden, removed or reordered, so if
the heuristic is wrong the model loses nothing it had before and can override
it from behaviour.
"""

ARM_CELL = '''\
# [calamitychasm] Verify the arm through the solver's own import path, not through os.environ.
# A green light here means: the patched bundle is mounted, the flag was read BEFORE the
# import in cell 9, and the detector is live in the process that will play the games.
import inference.utils.segmentation as _seg_mod
import inference.agent.prompts as _prompts_mod
import inference.agent.python_tool_sandbox as _sandbox_mod

for _m in (_seg_mod, _prompts_mod, _sandbox_mod):
    assert str(Path(_m.__file__).resolve()).startswith(str(ANIM_BUNDLE_DIR.resolve())), (
        _m.__name__, _m.__file__)

# 1. The detector shipped.
assert hasattr(_seg_mod, "detect_hud_nodes"), "HUD arm: segmentation.py is the unpatched upstream file"
assert (_seg_mod._HUD_EDGE_DISTANCE, _seg_mod._HUD_RATIO_THRESHOLD,
        _seg_mod._HUD_TWINS_THRESHOLD) == (3, 5, 3), "HUD arm: thresholds drifted"

# 2. The gate was read before the import, so the PROMPT knows about the field too.
#    This is the assert that catches the silent-no-op failure: annotation on, prompt off.
assert _prompts_mod._HUD_ANNOTATION is True, "HUD arm: prompts.py imported before the flag was set"
assert "hud_node_ids" in _prompts_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM
assert "hud_node_ids" in _prompts_mod.PYTHON_ADDENDUM
assert "HEURISTIC" in _prompts_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM, "HUD arm: the honesty caveat is missing"
import inference.agent.tool_agent as _ta_mod
assert "hud" in _ta_mod._PYTHON_TOOL_DESCRIPTION

# 3. The flag reaches the sandbox SUBPROCESS, which is where segment_layer actually runs.
#    Without this passthrough the whole arm is a no-op that still looks correct above.
assert _sandbox_mod._sandbox_env().get("ARC3_HUD_ANNOTATION") == "1", _sandbox_mod._sandbox_env()

# 4. End-to-end on a synthetic board: a 40x1 bar flush against row 0 is HUD; a 3x3
#    block in the interior is not. Cheap, and it proves the wiring rather than the intent.
_probe = [[0] * 64 for _ in range(64)]
for _c in range(40):
    _probe[0][_c] = 5
for _r in range(30, 33):
    for _c in range(30, 33):
        _probe[_r][_c] = 7
_out = _seg_mod.segment_layer(_probe, "WwgGcBMPRbSYOrNp")
assert "hud_node_ids" in _out, _out.keys()
_flagged = {n["id"] for n in _out["nodes"] if n["hud"]}
assert _flagged == set(_out["hud_node_ids"]) and _flagged, _out["hud_node_ids"]
_interior = [n for n in _out["nodes"] if n["color"] == "B"]
assert _interior and not any(n["hud"] for n in _interior), "HUD arm: flagged an interior object"

print(f"HUD_ARM flag=1 detector={_seg_mod.detect_hud_nodes.__name__} "
      f"thresholds=({_seg_mod._HUD_EDGE_DISTANCE},{_seg_mod._HUD_RATIO_THRESHOLD},{_seg_mod._HUD_TWINS_THRESHOLD}) "
      f"prompt=on sandbox_env=on probe_hud_ids={_out['hud_node_ids']} "
      f"module={Path(_seg_mod.__file__).parent}", flush=True)
'''

HEADER_MD = """\
# arc3-duck-nvfp4-hud - NVFP4 serving stack + anim solver + advisory HUD annotation

**This is a reproduction of other people's work with a four-file patch of ours,
run on our own account to measure it. No score quoted by any upstream is ours.**
Full credit and licence text in `THIRD_PARTY_NOTICE.md`.

Identical to `calamitychasm/arc3-duck-nvfp4-anim` except that the mounted solver
bundle is **our patched copy** of the same bundle, and `ARC3_HUD_ANNOTATION=1`.
The serving stack, the model, the NVFP4 runtime, the analyzer knobs
(`LOCAL_ANALYZER_SEED=20260825`, `LOCAL_ANALYZER_YIELD_SECONDS=180`),
`concurrency=28`, `max_runtime_s_per_game=7920`, the KV configuration, the
context length and the run loop are all carried over byte-identically, and the
build script asserts every one of the 18 inherited cells by sha256.

## The change

`current_frame.segmentation` is the model's primary view of the board. Each node
gains an advisory `hud` boolean, and the returned dict gains `hud_node_ids`. A
node is flagged when it sits flush against a frame edge and is either a long
thin bar (>=5:1) or one of >=3 identical shapes along that edge.

**Nothing is masked, hidden, removed, reordered or altered.** If the heuristic
is wrong the model loses nothing it had before and can override it, and the
prompt says in as many words that the flag is a rule over shape and position
that misses HUD blocks away from the edges and can mis-flag a real object.

Four files differ from the upstream bundle; 71 are byte-identical. See
`HUD_PATCH.md` in the mounted dataset.

## Why

`experiments/stage7_avo.md` closed the dead-turn lead: three separate mechanisms
bought the model more actions and none bought score, because the extra actions
landed in levels that were never solvable. The remaining lever is whether a
level is solvable at all. In the incumbent's own transcripts, HUD/status-bar
reasoning appears in **all 25 of 25 games**, 4,608 times.

## Why this run's score cannot rank the arm

A single public-25 mean carries SE +/-2.46 (`experiments/stage7_noise_floor.md`).
This run is for **counted** telemetry -- HUD mentions, dead-turn rate, actions,
calls, levels by index -- and for catastrophe detection. The mean is reported
only for comparability, not to rank anything.

## Attribution

The HUD rules and all three thresholds are ported from Rudakov, Shock & Cowley,
arXiv:2512.24156 (MIT); the licence text ships in the mounted dataset as
`GRAPH_EXPLORER_THIRD_PARTY_LICENSE`. The solver bundle is the Tufa Labs Duck
harness on Jakob Brueggen's `feature/animation-awareness` branch (MIT). The
serving stack and weights are Keith Tyser's / RadixArk's, mounted unmodified.
"""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cell(kind: str, text: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build(check_only: bool = False) -> None:
    raw = SRC_NB.read_text(encoding="utf-8")
    got_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if got_sha != EXPECTED_SRC_SHA256:
        raise SystemExit(
            f"source notebook sha256 {got_sha} != expected {EXPECTED_SRC_SHA256}; "
            "the anim base moved -- re-review before building an arm against it"
        )

    nb = json.loads(raw)
    cells = nb["cells"]
    if len(cells) != 18:
        raise SystemExit(f"anim notebook has {len(cells)} cells, expected 18")
    inherited = [_digest("".join(c["source"])) for c in cells]

    anchor = "".join(cells[ANCHOR_INDEX]["source"])
    if cells[ANCHOR_INDEX]["cell_type"] != "code" or ANCHOR_MARKER not in anchor:
        raise SystemExit(f"cell {ANCHOR_INDEX} is not the customization hook")
    if "32400.0" not in anchor:
        raise SystemExit("cell 13 no longer carries the 32400 s budget check")

    patched: set[int] = set()
    for idx, old, new in CELL_PATCHES:
        src = "".join(cells[idx]["source"])
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"cell {idx}: expected exactly 1 occurrence of {old[:70]!r}, found {n} "
                "-- the base drifted, refusing to build a half-patched arm"
            )
        cells[idx]["source"] = src.replace(old, new).splitlines(keepends=True)
        patched.add(idx)

    for idx in sorted(patched):
        if ANIM_DATASET in "".join(cells[idx]["source"]):
            raise SystemExit(f"cell {idx} still references {ANIM_DATASET!r} after patching")

    cells[0] = _cell("markdown", HEADER_MD)
    patched.add(0)

    compile(ARM_CELL, "<arm>", "exec")
    new_cells = (
        cells[: ANCHOR_INDEX + 1]
        + [_cell("markdown", ARM_HEADING), _cell("code", ARM_CELL)]
        + cells[ANCHOR_INDEX + 1 :]
    )
    nb["cells"] = new_cells

    got = [_digest("".join(c["source"])) for c in new_cells]
    kept = got[: ANCHOR_INDEX + 1] + got[ANCHOR_INDEX + 3 :]
    changed = {i for i, (a, b) in enumerate(zip(inherited, kept)) if a != b}
    if changed != patched:
        raise SystemExit(
            f"cells changed: {sorted(changed)}, expected exactly {sorted(patched)} "
            "-- refusing to build a confounded arm"
        )
    if len(new_cells) != 20:
        raise SystemExit(f"built {len(new_cells)} cells, expected 20")

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    ds = list(meta["dataset_sources"])
    if ds[-1] != ANIM_DATASET:
        raise SystemExit(f"dataset_sources[-1] is {ds[-1]!r}, expected the anim bundle")
    ds[-1] = HUD_DATASET
    meta["dataset_sources"] = ds
    meta["id"] = KERNEL
    meta["title"] = KERNEL.split("/", 1)[1]
    meta["code_file"] = OUT_NB.name
    for key, want in (
        ("enable_internet", False),
        ("enable_gpu", True),
        ("is_private", True),
        ("machine_shape", "NvidiaRtxPro6000"),
    ):
        if meta.get(key) != want:
            raise SystemExit(f"kernel metadata {key}={meta.get(key)!r}, expected {want!r}")
    if meta["competition_sources"] != ["arc-prize-2026-arc-agi-3"]:
        raise SystemExit("competition_sources drifted")

    nb_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    out_meta = OUT_DIR / "kernel-metadata.json"

    if check_only:
        for path, want in ((OUT_NB, nb_text), (out_meta, meta_text)):
            if not path.exists():
                raise SystemExit(f"{path} not built")
            if path.read_text(encoding="utf-8") != want:
                raise SystemExit(f"{path} is stale -- rebuild")
        print(f"{KERNEL} is up to date")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(nb_text, encoding="utf-8")
    out_meta.write_text(meta_text, encoding="utf-8")
    print(f"built {OUT_NB}")
    print(f"  base      {SRC_NB.name} sha256={got_sha[:16]} (18 cells, all asserted)")
    print(f"  patched   cells {sorted(patched)}; 16 inherited cells byte-identical")
    print(f"  inserted  1 markdown + 1 code cell after index {ANCHOR_INDEX} ({ANCHOR_MARKER})")
    print(f"  mount     {ANIM_DATASET}\n            -> {HUD_DATASET}")
    print("  arm       ARC3_HUD_ANNOTATION=1 (set in cell 7, before the cell-9 import)")
    print(f"  kernel    {KERNEL}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="verify the built arm is current")
    args = ap.parse_args()
    build(check_only=args.check)


if __name__ == "__main__":
    main()
