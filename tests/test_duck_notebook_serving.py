"""Tests for the serving-profile block inlined into the Duck submission notebook.

Two things are checked, because the notebook copy is *generated* and a Kaggle
kernel cannot import this repo:

1. **Drift.** The block actually present in the notebook must be exactly what
   `scripts/_patch_duck_notebook_serving.py` would emit today. If someone edits
   `vllm_serving.py` and forgets to regenerate, this fails instead of the
   notebook quietly shipping stale logic.
2. **Behaviour.** The generated `_duck_apply_serving_profile` is executed for
   real against a setup-command fixture, so the code that will run on Kaggle is
   proven to do what it claims -- not merely to exist.

The notebook ships with the `baseline` profile (an exact no-op) unless a
measurement says otherwise, so the shipped-default test asserts exactly that.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kaggle_submission_duck.vllm_serving import BASELINE_PROFILE, LAUNCH_ANCHOR

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPO_ROOT
    / "kaggle_submission_duck"
    / "notebook"
    / "lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb"
)
PATCH_SCRIPT = REPO_ROOT / "scripts" / "_patch_duck_notebook_serving.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("_duck_serving_patch", PATCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cell5_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(nb["cells"][5]["source"])


def _generated_block(source: str) -> str:
    patch = _load_patch_module()
    assert patch.BEGIN in source, "notebook is missing the generated serving block"
    start = source.index(patch.BEGIN)
    end = source.index(patch.END) + len(patch.END)
    return source[start:end]


def test_notebook_cell5_is_valid_python():
    compile(_cell5_source(), "<cell5>", "exec")


def test_shipped_profile_is_the_measured_default():
    """An unmeasured profile must never ship as the notebook's default."""
    source = _cell5_source()
    assert f"DUCK_VLLM_SERVING_PROFILE = {BASELINE_PROFILE!r}" in source


def test_notebook_block_matches_the_generator_exactly():
    """Catch a hand-edit or a forgotten regeneration."""
    patch = _load_patch_module()
    shipped = _generated_block(_cell5_source())
    expected = _generated_block(patch.render_block(BASELINE_PROFILE))
    assert shipped == expected


def test_patch_script_check_mode_passes_on_the_committed_notebook():
    patch = _load_patch_module()
    source = _cell5_source()
    assert patch.patch_cell(source, BASELINE_PROFILE) == source


def test_notebook_carries_every_known_profile_so_switching_needs_no_regeneration():
    from kaggle_submission_duck.vllm_serving import PROFILES

    source = _cell5_source()
    for name in PROFILES:
        if name == BASELINE_PROFILE:
            continue
        assert repr(name) in source, f"profile {name!r} missing from the notebook"


# --- behaviour of the generated code, executed for real ----------------------

FAKE_SETUP_COMMAND = (
    "def start_vllm_server() -> None:\n"
    "    cmd = ['--enable-prefix-caching', '--max-model-len', '65536']\n"
    + LAUNCH_ANCHOR + "\n"
    "    process = subprocess.Popen(cmd, env=vllm_env())\n"
)


def _exec_block(profile: str) -> dict:
    """Execute the generated block for `profile` and return its namespace."""
    patch = _load_patch_module()
    block = _generated_block(patch.render_block(profile))
    namespace: dict = {}
    exec(block, namespace)
    return namespace


def test_generated_baseline_helper_is_an_exact_noop():
    ns = _exec_block(BASELINE_PROFILE)
    patched, count = ns["_duck_apply_serving_profile"](FAKE_SETUP_COMMAND)
    assert count == 0
    assert patched == FAKE_SETUP_COMMAND


@pytest.mark.parametrize("profile", ["mtp3", "flags", "mtp3+flags", "mtp2", "ngram3"])
def test_generated_helper_patches_once_and_stays_valid_python(profile):
    ns = _exec_block(profile)
    patched, count = ns["_duck_apply_serving_profile"](FAKE_SETUP_COMMAND)
    assert count == 1
    compile(patched, "<patched>", "exec")
    assert patched.index("cmd.extend(") < patched.index(LAUNCH_ANCHOR)


def test_generated_helper_reports_zero_when_the_anchor_is_missing():
    ns = _exec_block("mtp3")
    patched, count = ns["_duck_apply_serving_profile"]("no anchor here")
    assert count == 0
    assert patched == "no anchor here"


def test_generated_mtp3_patch_actually_appends_the_speculative_config():
    """Run the injected code against a real cmd list, as Kaggle would."""
    from kaggle_submission_duck.vllm_serving import build_setup_patch

    body = "\n".join(line[4:] for line in build_setup_patch("mtp3").splitlines())
    ns = {"cmd": ["--enable-prefix-caching", "--max-model-len", "65536"]}
    exec(body, ns)
    assert ns["cmd"][-2] == "--speculative-config"
    assert json.loads(ns["cmd"][-1]) == {"method": "mtp", "num_speculative_tokens": 3}
    # mtp3 alone must NOT disturb prefix caching -- that belongs to `flags`.
    assert "--enable-prefix-caching" in ns["cmd"]


def test_generated_flags_patch_removes_prefix_caching_from_a_real_cmd_list():
    from kaggle_submission_duck.vllm_serving import build_setup_patch

    body = "\n".join(line[4:] for line in build_setup_patch("flags").splitlines())
    ns = {"cmd": ["--enable-prefix-caching", "--max-model-len", "65536"]}
    exec(body, ns)
    assert "--enable-prefix-caching" not in ns["cmd"]
    assert "--no-enable-prefix-caching" in ns["cmd"]
    assert "--async-scheduling" in ns["cmd"]
