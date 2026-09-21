"""Tests for the HUD-perception arm (`experiments/stage7_hud_perception.md`).

The arm's safety claim is narrow and it is the whole reason the arm is allowed
to ship: **the annotation is additive and advisory -- no pixel, node, edge or
field is removed, masked, hidden, reordered or altered, and the flag never
reaches the board interior.** These tests check exactly that, against **real**
boards from the champion anim run (`tests/fixtures/hud_frames.json.gz`, 55
boards across all 25 public games) rather than hand-made grids, because a
detector that works on synthetic bars proves nothing about the 25 games.

The module under test is built by applying `arc3_hud/splice.py` to the pristine
upstream `segmentation.py` (vendored at
`tests/fixtures/bundle_segmentation_anim.py.txt`) and executing the result, so
the code exercised here is the code that ships.
"""

from __future__ import annotations

import gzip
import json
import types
from collections import deque
from pathlib import Path

import pytest

from arc3_hud import splice

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PRISTINE_SEGMENTATION = FIXTURES / "bundle_segmentation_anim.py.txt"
FRAMES = FIXTURES / "hud_frames.json.gz"

ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
HEX = "0123456789abcdef"

#: Measured on the champion run's own frames, independently reproduced through
#: two different code paths (this one, and the repo's `graph_explorer_agent`
#: FrameProcessor). Pinned so a silent behaviour change fails the suite.
EXPECTED_GAMES_FIRING = 22
EXPECTED_TOTAL_GAMES = 25
#: Upstream's `min > extent - 3` right/bottom tests admit depth 0-1 and the
#: `max < 3` left/top tests admit 0-2, so 2 is the hard ceiling by construction.
MAX_EDGE_DEPTH = 2


# --------------------------------------------------------------- fixtures


def _load_module(source: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.fixture(scope="module")
def pristine_source() -> str:
    return PRISTINE_SEGMENTATION.read_bytes().decode("utf-8").replace("\r\n", "\n")


@pytest.fixture(scope="module")
def patched_source(pristine_source: str) -> str:
    return splice.patch_text("inference/utils/segmentation.py", pristine_source)


@pytest.fixture(scope="module")
def pristine(pristine_source: str) -> types.ModuleType:
    return _load_module(pristine_source, "hud_seg_pristine")


@pytest.fixture(scope="module")
def patched(patched_source: str) -> types.ModuleType:
    return _load_module(patched_source, "hud_seg_patched")


@pytest.fixture(scope="module")
def boards() -> list[tuple[str, int, list[list[int]]]]:
    payload = json.loads(gzip.open(FRAMES, "rb").read().decode("utf-8"))
    out = []
    for game, frames in sorted(payload["games"].items()):
        for frame in frames:
            grid = [[HEX.index(ch) for ch in row] for row in frame["rows"]]
            out.append((game, int(frame["level"]), grid))
    assert out, "fixture is empty"
    return out


@pytest.fixture(scope="module")
def analysed(pristine, patched, boards) -> list[dict]:
    """Every board segmented once by each of the three configurations.

    `segment_layer`'s containment pass is O(components x cells) and the busiest
    real board has 350 components, so a full pass costs ~10 s. Computing this
    once and sharing it keeps the suite usable; each test then asserts over the
    cached results rather than re-segmenting.
    """
    import os  # noqa: PLC0415

    rows = []
    for game, level, grid in boards:
        os.environ.pop("ARC3_HUD_ANNOTATION", None)
        base = pristine.segment_layer(grid, ARC_COLOR_CHARS)
        off = patched.segment_layer(grid, ARC_COLOR_CHARS)
        os.environ["ARC3_HUD_ANNOTATION"] = "1"
        on = patched.segment_layer(grid, ARC_COLOR_CHARS)
        os.environ.pop("ARC3_HUD_ANNOTATION", None)
        rows.append(
            {
                "game": game,
                "level": level,
                "grid": grid,
                "components": _components(grid),
                "base": base,
                "off": off,
                "on": on,
            }
        )
    return rows


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARC3_HUD_ANNOTATION", raising=False)


def _components(grid: list[list[int]]) -> list[set[tuple[int, int]]]:
    """Independent 4-connected flood fill, in the same reading order the bundle
    uses, so component index == node id. Deliberately not the bundle's code."""
    height, width = len(grid), len(grid[0])
    seen = [[False] * width for _ in range(height)]
    comps: list[set[tuple[int, int]]] = []
    for r0 in range(height):
        for c0 in range(width):
            if seen[r0][c0]:
                continue
            value = grid[r0][c0]
            cells: set[tuple[int, int]] = set()
            queue = deque([(r0, c0)])
            seen[r0][c0] = True
            while queue:
                r, c = queue.popleft()
                cells.add((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr][nc] and grid[nr][nc] == value:
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


# ------------------------------------------------- the gate is really a gate


def test_gate_off_leaves_output_identical(analysed):
    """With the flag unset, the patched bundle must be indistinguishable."""
    for row in analysed:
        assert row["off"] == row["base"], (
            f"{row['game']} L{row['level']}: gate-off output diverged from upstream")


def test_gate_off_adds_no_node_field(analysed):
    for row in analysed:
        out = row["off"]
        assert set(out) == {"nodes", "adjacency_list"}, f"{row['game']}: {sorted(out)}"
        assert all("hud" not in node for node in out["nodes"])


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_gate_accepts_truthy_spellings(patched, monkeypatch, raw):
    monkeypatch.setenv("ARC3_HUD_ANNOTATION", raw)
    grid = [[0] * 64 for _ in range(64)]
    assert "hud_node_ids" in patched.segment_layer(grid, ARC_COLOR_CHARS)


@pytest.mark.parametrize("raw", ["", "0", "false", "off", "no", "nonsense"])
def test_gate_rejects_everything_else(patched, monkeypatch, raw):
    monkeypatch.setenv("ARC3_HUD_ANNOTATION", raw)
    grid = [[0] * 64 for _ in range(64)]
    assert "hud_node_ids" not in patched.segment_layer(grid, ARC_COLOR_CHARS)


# ------------------------------------------------------- additive, not lossy


def test_annotation_is_purely_additive(analysed):
    """Strip the two new fields and the result must equal upstream exactly --
    same nodes, same ids, same order, same adjacency, nothing dropped."""
    for row in analysed:
        annotated = row["on"]
        stripped = {
            "nodes": [{k: v for k, v in n.items() if k != "hud"} for n in annotated["nodes"]],
            "adjacency_list": annotated["adjacency_list"],
        }
        assert stripped == row["base"], (
            f"{row['game']} L{row['level']}: annotation altered existing output")


def test_flag_and_id_list_agree(analysed):
    for row in analysed:
        out = row["on"]
        by_flag = {n["id"] for n in out["nodes"] if n["hud"]}
        assert by_flag == set(out["hud_node_ids"]), f"{row['game']} L{row['level']}"
        assert out["hud_node_ids"] == sorted(out["hud_node_ids"])


# ------------------------------------------------------- it actually detects


def test_detector_fires_on_the_real_public_25(analysed):
    firing = {row["game"] for row in analysed if row["on"]["hud_node_ids"]}
    games = {row["game"] for row in analysed}
    assert len(games) == EXPECTED_TOTAL_GAMES
    assert len(firing) == EXPECTED_GAMES_FIRING, sorted(games - firing)


# ---------------------------------------------- and never touches the middle


def test_flagged_nodes_hug_the_frame_edge(analysed):
    """The safety invariant. Every cell of every flagged node must lie within
    `MAX_EDGE_DEPTH` of a frame edge -- the detector cannot reach the board
    interior, whatever it thinks it has found."""
    worst = -1
    for row in analysed:
        grid, comps, out = row["grid"], row["components"], row["on"]
        height, width = len(grid), len(grid[0])
        assert len(comps) == len(out["nodes"]), f"{row['game']}: component count mismatch"
        for node_id in out["hud_node_ids"]:
            for r, c in comps[node_id]:
                depth = min(r, c, height - 1 - r, width - 1 - c)
                assert depth <= MAX_EDGE_DEPTH, (
                    f"{row['game']} L{row['level']}: node {node_id} has a cell at "
                    f"({r},{c}), depth {depth} from the nearest edge"
                )
                worst = max(worst, depth)
    assert 0 <= worst <= MAX_EDGE_DEPTH


def test_flagged_share_of_the_board_stays_small(analysed):
    for row in analysed:
        grid, out = row["grid"], row["on"]
        cells = len(grid) * len(grid[0])
        flagged = sum(n["pixels"] for n in out["nodes"] if n["hud"])
        assert flagged / cells <= 0.05, f"{row['game']} L{row['level']}: {flagged}/{cells}"


def test_interior_object_is_never_flagged(patched, monkeypatch):
    monkeypatch.setenv("ARC3_HUD_ANNOTATION", "1")
    grid = [[0] * 64 for _ in range(64)]
    for c in range(40):  # a HUD-shaped bar flush against row 0
        grid[0][c] = 5
    for r in range(30, 33):  # a plain interior block
        for c in range(30, 33):
            grid[r][c] = 7
    out = patched.segment_layer(grid, ARC_COLOR_CHARS)
    assert out["hud_node_ids"], "the edge bar should have been flagged"
    interior = [n for n in out["nodes"] if n["color"] == ARC_COLOR_CHARS[7]]
    assert interior and not any(n["hud"] for n in interior)


def test_long_interior_bar_is_not_flagged(patched, monkeypatch):
    """A 40x1 bar with exactly the HUD *shape* but placed in the middle of the
    board must not be flagged: position is load-bearing, not just aspect."""
    monkeypatch.setenv("ARC3_HUD_ANNOTATION", "1")
    grid = [[0] * 64 for _ in range(64)]
    for c in range(12, 52):
        grid[32][c] = 5
    out = patched.segment_layer(grid, ARC_COLOR_CHARS)
    assert out["hud_node_ids"] == []


# --------------------------------------------------------- bbox is exact


def test_boundary_bbox_matches_the_true_cell_bbox(patched, analysed):
    """`detect_hud_nodes` derives bboxes from `boundary` (corner points) rather
    than from cells, because that is what a node exposes. Check it is exact on
    every node of every real frame, not just argued to be."""
    for row in analysed:
        game, level, comps, out = row["game"], row["level"], row["components"], row["on"]
        for node, cells in zip(out["nodes"], comps):
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            assert patched._hud_bbox(node["boundary"]) == (
                min(rows),
                min(cols),
                max(rows),
                max(cols),
            ), f"{game} L{level}: node {node['id']}"


# ------------------------------------------------------------- the splice


def test_detector_source_is_spliced_verbatim(patched_source):
    body = splice.detector_source()
    assert "def detect_hud_nodes(" in body
    assert body in patched_source, "hud_detect.py and the shipped bundle have drifted"
    compile(body, "<detector>", "exec")


def test_every_patch_compiles_and_is_not_a_no_op(pristine_source):
    new = splice.patch_text("inference/utils/segmentation.py", pristine_source)
    assert new != pristine_source
    compile(new, "segmentation.py", "exec")


def test_patch_refuses_when_the_anchor_is_missing():
    with pytest.raises(RuntimeError, match="anchor occurs 0 times"):
        splice.patch_text("inference/utils/segmentation.py", "def nothing(): pass\n")


def test_patch_refuses_when_the_anchor_is_ambiguous(pristine_source):
    with pytest.raises(RuntimeError, match="anchor occurs 2 times"):
        splice.patch_text("inference/utils/segmentation.py", pristine_source * 2)


def test_upstream_segmentation_fixture_matches_the_pinned_bundle(pristine_source):
    """The vendored copy must be the exact file the build script pins, or these
    tests are validating a patch against source that is not what ships."""
    import hashlib

    from scripts import _build_hud_bundle as builder  # noqa: PLC0415

    want = builder.EXPECTED_PRE_PATCH["inference/utils/segmentation.py"]
    got = hashlib.sha256(pristine_source.encode("utf-8")).hexdigest()
    assert got == want


# ---------------------------------------------------------------- prompts


@pytest.fixture(scope="module")
def prompt_sources() -> str:
    return (FIXTURES / "bundle_prompts_hud_excerpt.py.txt").read_text(encoding="utf-8")


def test_prompt_text_is_honest_about_the_heuristic():
    """The prompt must not oversell the flag. A confidently wrong belief is
    worse than the status quo of re-deriving the geometry every turn."""
    header = splice._PROMPTS_IMPORT_NEW
    for phrase in ("HEURISTIC", "MISSES", "wrongly flag", "Verify", "Nothing is hidden"):
        assert phrase in header, phrase
    assert 'os.environ.get("ARC3_HUD_ANNOTATION"' in header


def test_prompt_constants_are_empty_when_the_gate_is_off():
    """Both branches are literal in the source, so a reader can see that the
    off state contributes no tokens at all."""
    header_count = splice._PROMPTS_IMPORT_NEW.count(') if _HUD_ANNOTATION else ""')
    assert header_count == 4, header_count


# ------------------------------------------------------------ the artifacts


def test_bundle_patch_manifest_is_consistent_with_the_pins():
    from scripts import _build_hud_bundle as builder  # noqa: PLC0415

    manifest = json.loads(
        (REPO / "experiments/stage7_hud_perception_artifacts/bundle_patch_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["env_flag"] == "ARC3_HUD_ANNOTATION"
    assert manifest["files_unchanged"] == 71
    assert set(manifest["files_patched"]) == set(splice.PATCHED_FILES)
    for rel, digests in manifest["files_patched"].items():
        assert digests["before"] == builder.EXPECTED_PRE_PATCH[rel]
        assert digests["after"] != digests["before"]


def test_kernel_notebook_is_current():
    from scripts import _build_duck_nvfp4_hud as builder  # noqa: PLC0415

    builder.build(check_only=True)


def test_kernel_notebook_changes_only_the_two_named_cells():
    """The single-variable claim, checked against the incumbent's own notebook."""
    import hashlib  # noqa: PLC0415

    from scripts import _build_duck_nvfp4_hud as builder  # noqa: PLC0415

    base = json.loads(builder.SRC_NB.read_text(encoding="utf-8"))["cells"]
    arm = json.loads(builder.OUT_NB.read_text(encoding="utf-8"))["cells"]
    assert len(base) == 18 and len(arm) == 20

    def digest(cell):
        return hashlib.sha256("".join(cell["source"]).encode("utf-8")).hexdigest()

    kept = arm[:14] + arm[16:]
    changed = {i for i, (a, b) in enumerate(zip(base, kept)) if digest(a) != digest(b)}
    assert changed == {0, 7}, changed
    # cell 9 is the graft's teeth and cell 13 the budget/settings block: both must
    # be untouched, or this is not a single-variable arm.
    for idx in (9, 13):
        assert digest(base[idx]) == digest(kept[idx])


def test_kernel_sets_the_flag_before_the_solver_import():
    """Ordering is load-bearing: prompts.py builds its constants at import time,
    so a flag set after cell 9 would annotate the board but never tell the model."""
    from scripts import _build_duck_nvfp4_hud as builder  # noqa: PLC0415

    cells = json.loads(builder.OUT_NB.read_text(encoding="utf-8"))["cells"]
    flag_cells = [i for i, c in enumerate(cells) if 'os.environ["ARC3_HUD_ANNOTATION"]' in "".join(c["source"])]
    import_cells = [i for i, c in enumerate(cells) if "import inference.agent.tool_agent" in "".join(c["source"])]
    assert flag_cells and import_cells
    assert min(flag_cells) < min(import_cells), (flag_cells, import_cells)


def test_kernel_mounts_our_patched_bundle():
    from scripts import _build_duck_nvfp4_hud as builder  # noqa: PLC0415

    meta = json.loads((builder.OUT_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_sources"][-1] == builder.HUD_DATASET
    assert builder.ANIM_DATASET not in meta["dataset_sources"]
    assert meta["id"] == builder.KERNEL
    # Everything else that could confound the comparison stays put.
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
