"""Tests for OpenAICompatibleClient's response-field handling.

This covers the single most expensive bug class in this project's
history. This build's Qwen3 reasoning parser returns generated tokens in
`reasoning`, not `content`. A client that reads only `content` reports a
perfectly healthy server as silent and produces a complete,
plausible-looking, entirely EMPTY results table. That has already cost
three free GPU runs once, and a fourth on 2026-09-22 when the backtest
saw {'reasoning': 46} with `content` never populated.

Nothing about that failure raises, so only a test that feeds a
reasoning-shaped payload can catch a regression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STAGE = (
    Path(__file__).resolve().parent.parent
    / "kaggle_submission_llm_world_engine"
    / "dataset_stage"
)
if str(_STAGE) not in sys.path:
    sys.path.insert(0, str(_STAGE))

from llm_engine.llm_client import OpenAICompatibleClient, make_client  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def capture(monkeypatch):
    """Intercept the POST and return a scripted payload."""
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(sent["payload"])

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


def client(**kw):
    return OpenAICompatibleClient(base_url="http://127.0.0.1:1234/v1", model="m", **kw)


def message_payload(message, finish_reason="stop"):
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def test_reads_content_when_present(capture):
    capture["payload"] = message_payload({"content": "hello"})
    c = client()
    assert c.complete("s", "u") == "hello"
    assert c.field_counts == {"content": 1}


def test_falls_back_to_reasoning_when_content_is_empty(capture):
    """The exact shape that produced an empty results table."""
    capture["payload"] = message_payload({"content": "", "reasoning": "the answer"})
    c = client()
    assert c.complete("s", "u") == "the answer"
    assert c.field_counts == {"reasoning": 1}


def test_falls_back_to_reasoning_when_content_is_absent(capture):
    capture["payload"] = message_payload({"reasoning": "the answer"})
    assert client().complete("s", "u") == "the answer"


def test_falls_back_to_reasoning_content(capture):
    capture["payload"] = message_payload({"reasoning_content": "the answer"})
    c = client()
    assert c.complete("s", "u") == "the answer"
    assert c.field_counts == {"reasoning_content": 1}


def test_content_wins_over_reasoning_when_both_present(capture):
    capture["payload"] = message_payload({"content": "real", "reasoning": "scratch"})
    assert client().complete("s", "u") == "real"


def test_whitespace_only_content_is_not_treated_as_an_answer(capture):
    capture["payload"] = message_payload({"content": "   \n ", "reasoning": "answer"})
    assert client().complete("s", "u") == "answer"


def test_genuinely_empty_response_is_counted_not_disguised(capture):
    capture["payload"] = message_payload({"content": "", "reasoning": ""})
    c = client()
    assert c.complete("s", "u") == ""
    assert c.field_counts == {"empty": 1}, (
        "an empty reply must be visible as empty, not silently indistinguishable "
        "from a field-name mismatch"
    )


def test_finish_reason_is_recorded(capture):
    """Truncation was INFERRED from field counts once. `length` vs `stop`
    measures it instead."""
    capture["payload"] = message_payload({"content": "x"}, finish_reason="length")
    c = client()
    c.complete("s", "u")
    capture["payload"] = message_payload({"content": "y"}, finish_reason="stop")
    c.complete("s", "u")
    assert c.finish_reasons == {"length": 1, "stop": 1}


# -- thinking control -------------------------------------------------


def test_thinking_flag_is_omitted_by_default(capture):
    capture["payload"] = message_payload({"content": "x"})
    client().complete("s", "u")
    assert "chat_template_kwargs" not in capture["body"]


def test_thinking_can_be_disabled(capture):
    capture["payload"] = message_payload({"content": "x"})
    client(enable_thinking=False).complete("s", "u")
    assert capture["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_request_targets_the_chat_completions_endpoint(capture):
    capture["payload"] = message_payload({"content": "x"})
    client().complete("s", "u")
    assert capture["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert capture["body"]["model"] == "m"
    assert [m["role"] for m in capture["body"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "raw,expected",
    [(None, None), ("", None), ("0", False), ("false", False), ("no", False),
     ("1", True), ("true", True)],
)
def test_make_client_reads_the_thinking_env_var(monkeypatch, raw, expected):
    monkeypatch.setenv("CODER_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("CODER_LLM_MODEL", "m")
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
    if raw is not None:
        monkeypatch.setenv("LLM_ENABLE_THINKING", raw)
    assert make_client("coder").enable_thinking is expected


def test_client_needs_no_openai_package(monkeypatch):
    """The kernel is offline and that dependency is not guaranteed."""
    monkeypatch.setitem(sys.modules, "openai", None)
    c = client()  # would raise on import if the package were required
    assert c._endpoint.endswith("/v1/chat/completions")
