"""Tests for the ``LOCAL_ANALYZER_YIELD_SECONDS`` 60 -> 180 override.

The override is a block of code appended to cell 10 of
``kaggle_submission_duck_nvfp4_yield/notebook/duck-qwen3-8-yield180.ipynb`` by
``scripts/_build_duck_nvfp4_yield.py``. It only ever executes on Kaggle, on a
9-hour GPU run, so a mistake in it is expensive. These tests execute that exact
block locally against a faithful stand-in for the bundle:

* a ``taaf_setup_env.json`` carrying the exact analyzer dict that
  ``serving_setup.py:2905-2952`` persists, and
* a stub ``inference.agent.tool_agent`` whose module-level constants are the
  real ones (``tool_agent.py:126-148``), so the env -> module-global binding
  under test is the real binding, not a paraphrase of it.

What is *not* covered: the bundle is not vendored into this repo, so the stub's
constants are transcribed, not imported. ``scripts/_build_duck_nvfp4_yield.py``
carries the file:line citations they were transcribed from.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPO
    / "kaggle_submission_duck_nvfp4_yield"
    / "notebook"
    / "duck-qwen3-8-yield180.ipynb"
)
BASELINE = (
    REPO
    / "kaggle_submission_duck_nvfp4"
    / "notebook"
    / "duck-qwen3-8-anim-base.ipynb"
)

# Verbatim from the bundle's serving_setup.py:2927-2950 (analyzer dict), plus
# the two runtime keys the override reads. LOCAL_ANALYZER_SEED is deliberately
# absent: serving_setup.py never persists one.
PERSISTED_ANALYZER_ENV = {
    "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:8000/v1",
    "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "INFERENCE_ANALYZER_MODEL": "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "OPENAI_API_KEY": "offline-kaggle-local-server",
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}

# Transcribed from the bundle's tool_agent.py:126-148. The point of the stub is
# that the constants are bound from os.environ at *import* time, exactly as the
# real module does -- that timing is what the override has to get right.
STUB_TOOL_AGENT = textwrap.dedent(
    '''
    import os


    def _get_env_int(name, default):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default


    def _get_env_float(name, default):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default


    def _get_env_bool(name, default):
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}


    _LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
    _LOCAL_ANALYZER_TOOL_STEPS = _get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
    _LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
    _LOCAL_ANALYZER_ENABLE_THINKING = _get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
    _LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
    '''
)


def _patch_block() -> str:
    """The override, extracted from the built notebook's setup cell."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][10]["source"])
    marker = "# [calamitychasm] THE ONE VARIABLE UNDER TEST"
    assert marker in src, "override block not found in cell 10"
    return src[src.index("# " + "-" * 10) :]


def _run_patch(tmp_path: Path, persisted: dict, *, preimport: bool = False):
    """Execute the override in a sandbox that mimics the notebook at cell 10."""
    env_path = tmp_path / "taaf_setup_env.json"
    env_path.write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n")

    pkg = tmp_path / "inference" / "agent"
    pkg.mkdir(parents=True)
    (tmp_path / "inference" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "tool_agent.py").write_text(STUB_TOOL_AGENT)

    saved_env = dict(os.environ)
    saved_path = list(sys.path)
    saved_mods = dict(sys.modules)
    try:
        # The notebook has already done os.environ.update(persisted) by this
        # point (cell 10's setup-command loop).
        os.environ.update({str(k): str(v) for k, v in persisted.items()})
        sys.path.insert(0, str(tmp_path))
        for name in [m for m in sys.modules if m.split(".")[0] in {"inference", "taaf"}]:
            del sys.modules[name]
        if preimport:
            import inference.agent.tool_agent  # noqa: F401

        ns = {
            "json": json,
            "os": os,
            "sys": sys,
            "Path": Path,
            "SETUP_ENV_PATH": env_path,
        }
        exec(compile(_patch_block(), "<yield-patch>", "exec"), ns)  # noqa: S102
        # Snapshot os.environ before the finally block restores it.
        return ns, json.loads(env_path.read_text()), dict(os.environ)
    finally:
        sys.path[:] = saved_path
        os.environ.clear()
        os.environ.update(saved_env)
        sys.modules.clear()
        sys.modules.update(saved_mods)


def test_override_binds_180_in_the_module_global(tmp_path, capsys):
    ns, written, env = _run_patch(tmp_path, PERSISTED_ANALYZER_ENV)

    # The binding that actually gates the tool loop (tool_agent.py:934 -> :1777).
    assert ns["_tool_agent"]._LOCAL_ANALYZER_YIELD_SECONDS == 180.0
    # ...and the three places it has to agree.
    assert env["LOCAL_ANALYZER_YIELD_SECONDS"] == "180"
    assert written["LOCAL_ANALYZER_YIELD_SECONDS"] == "180"
    assert "YIELD_PATCH ok yield_seconds=180.0" in capsys.readouterr().out


def test_override_changes_exactly_one_key(tmp_path):
    _, written, _env = _run_patch(tmp_path, PERSISTED_ANALYZER_ENV)

    differing = {
        k
        for k in set(written) | set(PERSISTED_ANALYZER_ENV)
        if written.get(k) != PERSISTED_ANALYZER_ENV.get(k)
    }
    assert differing == {"LOCAL_ANALYZER_YIELD_SECONDS"}


def test_tool_steps_stays_unbounded(tmp_path):
    """Time is the only gate: tool_steps 0 -> self._tool_steps is None."""
    ns, _written, _env = _run_patch(tmp_path, PERSISTED_ANALYZER_ENV)
    assert ns["_tool_agent"]._LOCAL_ANALYZER_TOOL_STEPS == 0


def test_no_seed_is_introduced(tmp_path):
    """A seed override would be a second variable; the baseline has none."""
    ns, written, _env = _run_patch(tmp_path, PERSISTED_ANALYZER_ENV)
    assert "LOCAL_ANALYZER_SEED" not in written
    assert ns["_tool_agent"]._LOCAL_ANALYZER_SEED == -1


def test_fails_loudly_if_upstream_no_longer_persists_60(tmp_path):
    drifted = dict(PERSISTED_ANALYZER_ENV, LOCAL_ANALYZER_YIELD_SECONDS="120")
    with pytest.raises(AssertionError, match="no longer persists yield 60"):
        _run_patch(tmp_path, drifted)


def test_fails_loudly_if_another_analyzer_knob_drifts(tmp_path):
    drifted = dict(PERSISTED_ANALYZER_ENV, LOCAL_ANALYZER_TEMPERATURE="0.8")
    with pytest.raises(AssertionError):
        _run_patch(tmp_path, drifted)


def test_fails_loudly_if_the_solver_was_already_imported(tmp_path):
    """Binding happens at import time; landing after it would silently no-op."""
    with pytest.raises(AssertionError, match="solver imported before"):
        _run_patch(tmp_path, PERSISTED_ANALYZER_ENV, preimport=True)


def test_only_the_header_and_setup_cells_differ_from_the_baseline():
    built = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["cells"]
    assert len(built) == len(base)
    changed = [
        i
        for i, (a, b) in enumerate(zip(base, built))
        if "".join(a["source"]) != "".join(b["source"])
    ]
    assert changed == [1, 10], changed
    # And cell 10 differs only by an append.
    assert "".join(built[10]["source"]).startswith("".join(base[10]["source"]))
