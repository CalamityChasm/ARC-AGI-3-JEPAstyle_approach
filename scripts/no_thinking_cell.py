# ---- [calamitychasm] NO-THINKING ARM ---------------------------------------------------------
# The ONLY behavioural change in this notebook relative to arc3-duck-nvfp4-anim.
#
# WHAT IT DOES
# Rebinds one module global after import:
#
#     inference.agent.tool_agent._LOCAL_ANALYZER_ENABLE_THINKING = False
#
# tool_agent.py:1533 reads that global *at call time*, inside `_chat_completion`:
#
#     thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING),
#
# and inference/utils/openai_compat.py:68 turns it into the vLLM chat-template
# switch `chat_template_kwargs = {"enable_thinking": False}`. Nothing else in the
# bundle reads it: it is the only occurrence of the name outside its own
# definition at line 155.
#
# This is deliberately a post-import rebind rather than an env override in cell
# 9, so cell 9 stays byte-for-byte the anim notebook's and this arm is one added
# cell, exactly like the other.
#
# WHY (measured on this exact chassis)
# The binding budget is the 7,920s per-game wall clock, and the chassis spends
# 46.1% of it on turns that execute no game action (scripts/analyze_dead_turns.py:
# 91,295s of 198,233s on anim; 48.6% and 49.8% on the two sibling runs, so this
# is structural, not run noise). A turn dies because it hits the 180s yield
# budget before the model calls `action(...)`, and a dead turn is the expensive
# kind: median 258s against 173s for one that acts.
#
# Generation is what makes a turn long, and generation is almost entirely
# reasoning. Across the anim run's 1,358 model responses
# (`reasoning_chars` / `content_chars` in each [MODEL RESPONSE META] block):
#
#     reasoning  4,768,656 chars   median 2,030   95.0% of generated text
#     content      250,663 chars   median   142
#
# Turning the reasoning block off attacks the same 46% from the cost side rather
# than the policy side: it should raise the number of `python` calls that fit
# inside one 180s turn, and -- because turn latency on this stack is 86.97%
# queue wait driven by KV residency (stage7_kv_residency_levers.md) -- it also
# shortens every request, which raises residency for every concurrent game.
#
# WHY IT IS RANKED SECOND, NOT FIRST
# It is a capability bet, and this repo's own precedent is against it: every
# intervention that reduced what the model could think with (`dedupe`, `ctx16k`)
# raised tokens per turn and lost score. It is listed as shortlist item 7 in
# stage7_sota_research.md with "high risk, not recommended first", and
# `juliancamilovilla/arc-agi3-nvfp4-carry-nothink-long` is running it publicly.
# It is here because it is the largest single untested lever on the measured
# sink, and one free run answers it.
import inspect

import inference.agent.tool_agent as _nt_tool_agent
from inference.utils.openai_compat import build_chat_payload as _nt_build_chat_payload

_NT_FLAG = "_LOCAL_ANALYZER_ENABLE_THINKING"

# --- fail loudly if upstream is not what this arm assumes ---------------------------------------
assert hasattr(_nt_tool_agent, _NT_FLAG), f"no-thinking: {_NT_FLAG} is gone from tool_agent"
assert getattr(_nt_tool_agent, _NT_FLAG) is True, (
    f"no-thinking: expected upstream default True, got {getattr(_nt_tool_agent, _NT_FLAG)!r} -- "
    "something else already set this and the arm would not be one variable"
)
_NT_CHAT_SRC = inspect.getsource(_nt_tool_agent.ToolAgent._chat_completion)
assert f"thinking=bool({_NT_FLAG})" in _NT_CHAT_SRC, (
    "no-thinking: _chat_completion no longer reads the flag at call time"
)
assert "build_chat_payload(" in _NT_CHAT_SRC, "no-thinking: _chat_completion no longer builds the payload"
# If __init__ snapshotted the flag onto the instance, a post-import rebind would
# silently do nothing on agents the solver constructs later.
assert _NT_FLAG not in inspect.getsource(_nt_tool_agent.ToolAgent.__init__), (
    "no-thinking: ToolAgent.__init__ now captures the flag; rebind at import time instead"
)
_NT_PAYLOAD_SRC = inspect.getsource(_nt_build_chat_payload)
assert '"chat_template_kwargs"' in _NT_PAYLOAD_SRC and '"enable_thinking"' in _NT_PAYLOAD_SRC, (
    "no-thinking: build_chat_payload no longer emits the vLLM enable_thinking switch"
)


# --- probe the real payload path, before any game runs -------------------------------------------
def _nt_payload(flag_value):
    """Exactly what tool_agent.py:1525-1536 builds, with the flag at `flag_value`."""
    return _nt_build_chat_payload(
        provider="vllm",
        model="probe",
        messages=[{"role": "user", "content": "probe"}],
        max_tokens=None,
        temperature=0.6,
        top_p=1.0,
        top_k=0,
        thinking=bool(flag_value),
        tools=None,
        tool_choice=None,
        seed=20260825,
    )


_NT_BEFORE = _nt_payload(getattr(_nt_tool_agent, _NT_FLAG))
assert _NT_BEFORE["chat_template_kwargs"] == {"enable_thinking": True}, _NT_BEFORE.get("chat_template_kwargs")

setattr(_nt_tool_agent, _NT_FLAG, False)

_NT_AFTER = _nt_payload(getattr(_nt_tool_agent, _NT_FLAG))
assert _NT_AFTER["chat_template_kwargs"] == {"enable_thinking": False}, _NT_AFTER.get("chat_template_kwargs")
# Nothing else about the request may move.
assert {k: v for k, v in _NT_BEFORE.items() if k != "chat_template_kwargs"} == {
    k: v for k, v in _NT_AFTER.items() if k != "chat_template_kwargs"
}, "no-thinking: the flag changed something other than enable_thinking"
assert getattr(_nt_tool_agent, _NT_FLAG) is False

print(
    f"NO_THINKING_INSTALLED target={_nt_tool_agent.__name__}.{_NT_FLAG} value=False "
    f"payload_before={_NT_BEFORE['chat_template_kwargs']} payload_after={_NT_AFTER['chat_template_kwargs']} "
    "probe=2/2",
    flush=True,
)
