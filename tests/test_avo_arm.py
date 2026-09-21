"""The AVO arms are what the write-up says they are.

Three things could silently break this experiment, and each has cost this
project a real run before:

1. the built notebook drifting from its builder (`--check` catches it);
2. the `max_runtime_s` trap coming back as an assertion rather than a set, which
   either kills the run at setup or overruns Kaggle's 9 h cap;
3. the two arms differing in more than the one env var they are supposed to,
   which is exactly the confound `stage7_config_locality.md` was written about.

These are cheap, offline, and need no Kaggle credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARMS = {
    0: REPO / "kaggle_submission_duck_nvfp4_avo" / "notebook" / "arc3-duck-nvfp4-avo.ipynb",
    1: REPO / "kaggle_submission_duck_nvfp4_avo_phased" / "notebook" / "arc3-duck-nvfp4-avo-pl.ipynb",
}
ANIM = REPO / "kaggle_submission_duck_nvfp4_anim" / "notebook" / "arc3-duck-nvfp4-anim.ipynb"


def cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def code(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in cells(path) if c["cell_type"] == "code"]


def meta(path: Path) -> dict:
    return json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_built_arm_is_current(arm: int) -> None:
    """The committed notebook is what the builder produces from the committed base."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_build_avo", REPO / "scripts" / "_build_duck_nvfp4_avo.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.build(arm, check_only=True)  # raises SystemExit if stale


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_budget_trap_sets_and_does_not_assert(arm: int) -> None:
    """Trap 1: cell 13 must SET 32400, not assert it.

    The AVO bundle's deploy_target.pkl carries 54000.0. Asserting kills the run;
    inheriting it pushes the soft deadline six hours past Kaggle's hard cap.
    """
    hook = next(c for c in code(ARMS[arm]) if "PUBLIC25_SETTINGS" in c)
    assert "target.max_runtime_s = 32400.0" in hook, "cell 13 no longer SETS the budget"
    # The inherited assertion form must be gone.
    assert "!= 32400.0" not in hook, "the inherited equality assert came back"
    # ...but drift to an unknown third value must still be fatal.
    assert "not in (32400.0, 54000.0)" in hook, "the drift guard is missing"
    assert "AVO_BUDGET_OVERRIDE" in hook, "the override is not printed"
    # The per-game clock, which the exploit deadline is measured against, is untouched.
    assert "bm.solver.max_runtime_s_per_game = 7920.0" in hook


@pytest.mark.parametrize("arm,expected", [(0, "0"), (1, "1")])
def test_arm_pins_and_verifies_the_phased_loop(arm: int, expected: str) -> None:
    """Trap 2: the flag is set AND read back through the solver's own call."""
    cell = next(c for c in code(ARMS[arm]) if "ARC3_AVO_PHASED_LOOP" in c)
    assert f'os.environ["ARC3_AVO_PHASED_LOOP"] = "{expected}"' in cell
    assert "AvoSettings.from_env()" in cell, "not read back through the solver's own call"
    assert f"_avo.phased_loop is {'True' if expected == '1' else 'False'}" in cell
    # Every other AVO knob must be asserted at its shipped default: one variable.
    for probe in ("(3, 12, 3)", "exploit_deadline == 0.6", "(24, 16, 16)"):
        assert probe in cell, f"missing default assertion: {probe}"


def test_arms_differ_only_in_the_phased_loop_flag() -> None:
    a, b = code(ARMS[0]), code(ARMS[1])
    assert len(a) == len(b)
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert differing == [7], f"arms differ in code cells {differing}, expected only the arm cell"
    only_a = set(a[7].splitlines()) - set(b[7].splitlines())
    # One env line, one assertion, one comment -- and nothing else.
    assert len(only_a) == 3, only_a


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_mount_is_the_frozen_pin_not_the_rolling_slug(arm: int) -> None:
    """A rolling mount could change what we ran between the free run and a submission."""
    m = meta(ARMS[arm])
    assert "raist321/taaf-avo-v27-bundle" in m["dataset_sources"]
    assert not any("jakobbrggen" in d for d in m["dataset_sources"]), m["dataset_sources"]
    # The sealed serving stack and the model mount are the incumbent's, untouched.
    anim_meta = meta(ANIM)
    assert m["dataset_sources"][:2] == anim_meta["dataset_sources"][:2]
    assert m["model_sources"] == anim_meta["model_sources"]
    assert m["docker_image"] == anim_meta["docker_image"]
    assert m["machine_shape"] == anim_meta["machine_shape"] == "NvidiaRtxPro6000"
    assert m["enable_internet"] is False and m["is_private"] is True


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_graft_teeth_and_run_loop_are_byte_identical_to_the_incumbent(arm: int) -> None:
    """Cell 9 (sys.path + LOCAL_ANALYZER_* overrides) and cell 17 (the run) must not drift.

    If either changed, the arm is no longer a solver-only swap.
    """
    anim_cells = cells(ANIM)
    arm_cells = cells(ARMS[arm])
    # The arm inserts 2 cells after index 13, so indices <= 13 line up directly
    # and later inherited cells are shifted by 2.
    for anim_i, shift in [(9, 0), (15, 2), (17, 2)]:
        assert "".join(anim_cells[anim_i]["source"]) == "".join(arm_cells[anim_i + shift]["source"]), (
            f"inherited cell {anim_i} drifted"
        )


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_avo_agent_is_asserted_live(arm: int) -> None:
    """The point of the swap is checked against the unpickled solver, not assumed."""
    cell = next(c for c in code(ARMS[arm]) if "benchmark_initial.pkl" in c)
    assert 'assert getattr(bm.solver, "avo_agent", None) is True' in cell
    assert 'assert bm.label == "avo-kaggle"' in cell
    # The anim half of the superset claim is still checked too.
    assert '"animation_awareness", None) is True' in cell
    assert '"hard_noop_guard", None) is True' in cell
