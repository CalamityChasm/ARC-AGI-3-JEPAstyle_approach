"""The no-thinking arm: does it change exactly one thing, and nothing else?

The arm is a single post-import rebind of
``inference.agent.tool_agent._LOCAL_ANALYZER_ENABLE_THINKING`` to ``False``,
installed in the notebook's customization cell. These tests stand up a stub of
the two modules involved, with ``normalize_provider`` and ``build_chat_payload``
copied **byte-for-byte** from
``jakobbrggen/taaf-kaggle-source-anim-20260807-anim``
``src/ARC3-Inference/inference/utils/openai_compat.py`` lines 7-13 and 36-72
(sha256 ``7dec6dddc2c5805386db5220814be7f3043a3d7e564ab109173ca1b8c2256462``),
and the one reading line copied from ``inference/agent/tool_agent.py`` line 1533
(sha256 ``856bf9b895d0ad8b959c8f828c7132b0e09eaa47f4c5cc6173785354090f8be7``).

Run: venv/Scripts/python.exe -m pytest tests/test_no_thinking.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CELL = REPO / "scripts" / "no_thinking_cell.py"
NOTEBOOK = (
    REPO / "kaggle_submission_duck_nvfp4_anim_nothink" / "notebook" / "arc3-duck-nvfp4-anim-nt.ipynb"
)

# openai_compat.py lines 7-13 and 36-72, byte-for-byte.
OPENAI_COMPAT = '''\
from typing import Any


def normalize_provider(value: str | None) -> str:
    provider = str(value or "").strip().lower()
    if provider in {"", "openai", "openai-compatible", "compat"}:
        return "vllm"
    if provider in {"openrouter", "router"}:
        return "openrouter"
    return provider


def build_chat_payload(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    normalized = normalize_provider(provider)
    if normalized == "vllm":
        if top_k > 0:
            payload["top_k"] = top_k
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if seed is not None and seed >= 0:
            payload["seed"] = seed

    return payload
'''

# The reading line is tool_agent.py:1533, in its own call site.
TOOL_AGENT = '''\
from inference.utils.openai_compat import build_chat_payload

_LOCAL_ANALYZER_ENABLE_THINKING = {DEFAULT}
_LOCAL_ANALYZER_TEMPERATURE = 0.6
_LOCAL_ANALYZER_TOP_P = 1.0
_LOCAL_ANALYZER_TOP_K = 0
_LOCAL_ANALYZER_SEED = 20260825


class ToolAgent:
    def __init__(self):
        self._max_output_tokens = None
        self._model = None

    def _chat_completion(self, messages, *, tools=None, request_timeout_seconds=None):
        payload = build_chat_payload(
            provider="vllm",
            model="Qwen/Qwen3.8-Flash-Next-NVFP4",
            messages=messages,
            max_tokens=self._max_output_tokens,
            temperature=_LOCAL_ANALYZER_TEMPERATURE,
            top_p=_LOCAL_ANALYZER_TOP_P,
            top_k=_LOCAL_ANALYZER_TOP_K,
            {THINKING_LINE}
            tools=tools,
            tool_choice=None,
            seed=_LOCAL_ANALYZER_SEED,
        )
        return payload
'''

GOOD_THINKING_LINE = "thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING),"


def _make_stub(tmp_path, *, default="True", thinking_line=GOOD_THINKING_LINE, compat=OPENAI_COMPAT, init_extra=""):
    root = tmp_path / "stub"
    agent_pkg = root / "inference" / "agent"
    utils_pkg = root / "inference" / "utils"
    agent_pkg.mkdir(parents=True)
    utils_pkg.mkdir(parents=True)
    for p in (root / "inference", agent_pkg, utils_pkg):
        (p / "__init__.py").write_text("", encoding="utf-8")
    (utils_pkg / "openai_compat.py").write_text(compat, encoding="utf-8")
    src = TOOL_AGENT.format(DEFAULT=default, THINKING_LINE=thinking_line)
    if init_extra:
        src = src.replace("        self._model = None\n", "        self._model = None\n" + init_extra)
    (agent_pkg / "tool_agent.py").write_text(src, encoding="utf-8")
    return root


def _install(tmp_path, monkeypatch, **kw):
    root = _make_stub(tmp_path, **kw)
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
        del sys.modules[name]
    ns: dict = {}
    exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), ns)  # noqa: S102
    import inference.agent.tool_agent as mod

    return mod, ns


# --------------------------------------------------------------------------- the one change


def test_flips_the_flag_and_the_real_payload(tmp_path, monkeypatch, capsys):
    mod, _ = _install(tmp_path, monkeypatch)
    assert mod._LOCAL_ANALYZER_ENABLE_THINKING is False
    payload = mod.ToolAgent()._chat_completion([{"role": "user", "content": "x"}])
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    out = capsys.readouterr().out
    assert "NO_THINKING_INSTALLED" in out and "probe=2/2" in out


def test_changes_nothing_else_in_the_request(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    after = mod.ToolAgent()._chat_completion([{"role": "user", "content": "x"}])
    mod._LOCAL_ANALYZER_ENABLE_THINKING = True
    before = mod.ToolAgent()._chat_completion([{"role": "user", "content": "x"}])
    assert {k: v for k, v in before.items() if k != "chat_template_kwargs"} == {
        k: v for k, v in after.items() if k != "chat_template_kwargs"
    }
    assert before["chat_template_kwargs"] == {"enable_thinking": True}
    assert before["temperature"] == after["temperature"] == 0.6
    assert before["seed"] == after["seed"] == 20260825


def test_it_is_read_per_call_so_later_agents_see_it(tmp_path, monkeypatch):
    """The solver builds one ToolAgent per game, after this cell runs."""
    mod, _ = _install(tmp_path, monkeypatch)
    for _ in range(3):
        assert mod.ToolAgent()._chat_completion([])["chat_template_kwargs"] == {"enable_thinking": False}


# --------------------------------------------------------------------------- fails loudly


def test_fails_loudly_if_the_flag_is_gone(tmp_path, monkeypatch):
    compat = OPENAI_COMPAT
    src = TOOL_AGENT.format(DEFAULT="True", THINKING_LINE=GOOD_THINKING_LINE).replace(
        "_LOCAL_ANALYZER_ENABLE_THINKING = True\n", ""
    )
    root = tmp_path / "stub"
    agent_pkg = root / "inference" / "agent"
    utils_pkg = root / "inference" / "utils"
    agent_pkg.mkdir(parents=True)
    utils_pkg.mkdir(parents=True)
    for p in (root / "inference", agent_pkg, utils_pkg):
        (p / "__init__.py").write_text("", encoding="utf-8")
    (utils_pkg / "openai_compat.py").write_text(compat, encoding="utf-8")
    (agent_pkg / "tool_agent.py").write_text(src, encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
        del sys.modules[name]
    with pytest.raises((AssertionError, NameError)):
        exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), {})  # noqa: S102


def test_fails_loudly_if_something_already_set_the_flag(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="expected upstream default True"):
        _install(tmp_path, monkeypatch, default="False")


def test_fails_loudly_if_the_flag_is_no_longer_read_at_call_time(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="no longer reads the flag at call time"):
        _install(tmp_path, monkeypatch, thinking_line="thinking=True,")


def test_fails_loudly_if_init_captures_the_flag(tmp_path, monkeypatch):
    extra = "        self._thinking = _LOCAL_ANALYZER_ENABLE_THINKING\n"
    with pytest.raises(AssertionError, match="captures the flag"):
        _install(tmp_path, monkeypatch, init_extra=extra)


def test_fails_loudly_if_the_payload_stops_emitting_the_switch(tmp_path, monkeypatch):
    compat = OPENAI_COMPAT.replace(
        'payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}',
        "pass",
    )
    with pytest.raises(AssertionError, match="no longer emits the vLLM enable_thinking switch"):
        _install(tmp_path, monkeypatch, compat=compat)


# --------------------------------------------------------------------------- one variable


def test_the_cell_touches_no_other_knob_and_no_serving_setting():
    src = CELL.read_text(encoding="utf-8")
    for forbidden in ("os.environ", "bm.solver", "SETUP_ENV_PATH", "LOCAL_ANALYZER_YIELD", "LOCAL_ANALYZER_SEED"):
        assert forbidden not in src, forbidden
    # exactly one setattr, on exactly the one flag
    assert src.count("setattr(") == 1
    assert "setattr(_nt_tool_agent, _NT_FLAG, False)" in src


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not built yet")
def test_the_notebook_carries_this_exact_cell():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    bodies = ["".join(c["source"]) for c in nb["cells"]]
    assert any(b.strip() == CELL.read_text(encoding="utf-8").strip() for b in bodies)
