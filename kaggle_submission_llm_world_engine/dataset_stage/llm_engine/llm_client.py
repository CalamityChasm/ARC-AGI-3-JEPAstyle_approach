"""Clients for the two LLM roles (see architecture.md's "Two LLM roles"):
  - "coder": drafts/repairs WorldModel source. Wants a strong coding model
    (Qwen3-Coder), called on draft/repair events only -- see drafting.py.
  - "action_head": a sparse fallback that suggests a single next action
    when the non-LLM planner genuinely stalls (see planner.py's
    PlanResult.stalled and code_world_agent.py's fallback wiring). Wants a
    model with broad gameplay/visual judgment (Gemma) more than raw coding
    strength, and is called far less often than "coder".

Two backends, selected via LLM_BACKEND (default "openai"):
  - "openai": an ordinary OpenAI-compatible chat-completions endpoint
    (e.g. vLLM/SGLang serving on the RTX Pro 6000). Right backend if you
    can actually get vLLM running -- continuous batching across the
    Swarm's concurrent games, real HTTP isolation between roles.
  - "transformers": loads a model in-process via plain `transformers` +
    `torch` and calls `.generate()` directly, no server at all. This is
    what actually runs in the real competition submission environment as
    currently built: with `enable_internet: false` (required for a real
    scored run), pip installing vLLM isn't possible and it isn't
    preinstalled -- confirmed empirically against a real no-internet,
    competition-attached Kaggle kernel, where only `transformers` and
    `torch` were already present. Swapping backends is a config change
    (LLM_BACKEND env var), not a code change.

Also provides a MockLLMClient with no model behind it at all, used for
local development/smoke-testing this project's plumbing on a machine
without the target GPU. It is NOT a substitute for a real model and should
never be used for an actual submission -- see `make_client`.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

Role = Literal["coder", "action_head"]

# Per-role defaults. Two separate servers/ports by default since vLLM
# serves one model per process; point both at the same URL if you're
# instead running one multi-model server.
_ROLE_DEFAULTS: dict[Role, dict[str, str]] = {
    "coder": {"base_url": "http://localhost:8000/v1", "model": "qwen3-coder"},
    "action_head": {"base_url": "http://localhost:8001/v1", "model": "gemma"},
}


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


@dataclass
class OpenAICompatibleClient:
    """Talks to a local vLLM/SGLang server's /v1/chat/completions endpoint.

    Configure via env vars so no code changes are needed between dev and
    the real serving setup. Role-specific vars take precedence; unset ones
    fall back to the role's default in _ROLE_DEFAULTS above:
      CODER_LLM_BASE_URL / CODER_LLM_MODEL      (default: qwen3-coder @ :8000)
      ACTION_LLM_BASE_URL / ACTION_LLM_MODEL    (default: gemma @ :8001)
      LLM_API_KEY (default "unused" -- vLLM ignores it unless configured
                   to require one, but the OpenAI client library requires
                   the field to be present)
    """

    base_url: str
    model: str
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "unused"))
    temperature: float = 0.2

    def __post_init__(self) -> None:
        # Imported lazily so the rest of this project doesn't hard-depend
        # on the `openai` package if someone only wants MockLLMClient.
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        return content or ""


@dataclass
class TransformersClient:
    """In-process model, no server, no extra packages beyond transformers
    + torch. One instance holds one loaded model -- see
    `get_shared_transformers_client` below for why callers must never
    construct this directly.

    Generation is serialized behind a lock: multiple Swarm game-threads
    may share one instance, and concurrently calling `.generate()` on the
    same nn.Module is not safe. Acceptable because LLM calls are meant to
    be rare (the whole point of the draft/repair/stall-only design) -- if
    this lock becomes a real bottleneck, that's itself a signal something
    upstream is calling the LLM far more than intended.
    """

    model_dir: str
    temperature: float = 0.2
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, init=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # torch_dtype="auto" picks whatever the checkpoint was saved in --
        # usually bfloat16 for a recent model. bf16 needs compute capability
        # 8.0+ (Ampere or newer); older hardware (e.g. a Tesla P100, compute
        # capability 6.0, hit exactly this during real testing) has no bf16
        # kernels at all and fails at weight-init time, not at generate()
        # time, so this can't be caught reactively -- pick a dtype the
        # actual GPU supports up front instead.
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=dtype, device_map="auto"
        )

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        # Chat-template formatting varies across model families:
        # 1. Plain strings with separate system and user roles (e.g. Qwen).
        # 2. Multimodal parts format with unified user message for models that
        #    strictly require typed parts or lack a separate system role (e.g. Gemma 3).
        # 3. Plain string with unified user message.
        # 4. Raw text fallback if tokenizer has no working chat template.
        text = None
        # Attempt 1: Standard separate system + user roles with string content
        try:
            text = self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

        # Attempt 2: Gemma 3-style multimodal parts format in a single user turn
        if text is None:
            full_user_text = f"{system}\n\n{user}" if system else user
            try:
                text = self._tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": [{"type": "text", "text": full_user_text}]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Attempt 3: Single user turn with plain string content
        if text is None:
            full_user_text = f"{system}\n\n{user}" if system else user
            try:
                text = self._tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": full_user_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Attempt 4: Bare text fallback
        if text is None:
            text = f"{system}\n\n{user}\n\n" if system else f"{user}\n\n"

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with self._lock:
            with self._torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        result_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        del inputs
        del output_ids
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        return result_text


_shared_transformers_clients: dict[str, TransformersClient] = {}
_shared_transformers_clients_lock = threading.Lock()


def get_shared_transformers_client(model_dir: str) -> TransformersClient:
    """One loaded model per model_dir, shared across every agent instance
    in this process. The Swarm creates one CodeWorldAgent (and thus one
    make_client() call) per concurrent game -- without this cache, each of
    up to ~25 concurrent games would independently load its own full copy
    of the model into GPU memory, which doesn't fit regardless of GPU
    size. All agents sharing one role's model is exactly the intended
    shape anyway: it's one draft/repair/action-head brain, not one per
    game.
    """
    with _shared_transformers_clients_lock:
        if model_dir not in _shared_transformers_clients:
            _shared_transformers_clients[model_dir] = TransformersClient(model_dir=model_dir)
        return _shared_transformers_clients[model_dir]


@dataclass
class MockLLMClient:
    """No model behind this at all -- always returns the world-model
    skeleton unmodified (an 'assume nothing changes' model), regardless of
    which role it's standing in for. Exists purely so this project's loop
    can be exercised end-to-end on a machine without the target GPU/models.

    Deliberately loud about what it is: every call logs a warning, and
    `make_client` refuses to hand one out unless LLM_ENGINE_ALLOW_MOCK=1 is
    set, so it can't be mistaken for a real submission configuration.
    """

    role: str = "unspecified"
    calls: int = 0

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        logger.warning(
            "MockLLMClient(role=%s).complete() called (call #%d) -- no real model "
            "is being used. This is for plumbing smoke-tests only.",
            self.role, self.calls,
        )
        from .world_model import WORLD_MODEL_SKELETON

        return f"```python\n{WORLD_MODEL_SKELETON}```"


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response_text: str) -> str:
    """Pull the first fenced code block out of an LLM response. Falls back
    to the whole response if no fence is found (some models occasionally
    omit it despite being asked)."""
    match = _CODE_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def make_client(role: Role) -> LLMClient:
    """Factory used by code_world_agent.py: real client by default, mock
    only if explicitly allowed via env var. `role` selects which model/env
    vars/defaults apply -- see _ROLE_DEFAULTS. LLM_BACKEND ("openai",
    default, or "transformers") selects which serving mechanism."""
    if os.getenv("LLM_ENGINE_USE_MOCK") == "1":
        if os.getenv("LLM_ENGINE_ALLOW_MOCK") != "1":
            raise RuntimeError(
                "LLM_ENGINE_USE_MOCK=1 but LLM_ENGINE_ALLOW_MOCK is not set to 1 -- "
                "the mock client has no real model behind it and must not be used "
                "for anything but plumbing smoke-tests. Set both env vars if that's "
                "really what you want."
            )
        return MockLLMClient(role=role)

    backend = os.getenv("LLM_BACKEND", "openai")
    env_prefix = "CODER" if role == "coder" else "ACTION"

    if backend == "transformers":
        model_dir = os.getenv(f"{env_prefix}_MODEL_DIR")
        if not model_dir:
            raise RuntimeError(
                f"LLM_BACKEND=transformers but {env_prefix}_MODEL_DIR is not set -- "
                "point it at a local directory containing the model's config.json/weights."
            )
        return get_shared_transformers_client(model_dir)

    defaults = _ROLE_DEFAULTS[role]
    return OpenAICompatibleClient(
        base_url=os.getenv(f"{env_prefix}_LLM_BASE_URL", defaults["base_url"]),
        model=os.getenv(f"{env_prefix}_LLM_MODEL", defaults["model"]),
    )
