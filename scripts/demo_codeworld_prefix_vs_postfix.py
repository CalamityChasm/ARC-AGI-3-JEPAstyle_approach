"""Before/after demonstration for the CodeWorldAgent defect fixes
(branch stage7-codeworld-fixes, GitHub issue #3).

Runs the same seven scenarios the unit tests assert, against *whichever*
copy of `llm_engine` you point it at, and prints PASS/FAIL per scenario.
The point is to be able to run the identical checks against the code as
actually submitted (Kaggle ref 56084133, git commit 3765cf7) and against
the fixed code, and see the difference -- the unit tests themselves
cannot do this, because they import APIs the pre-fix code does not have.

Usage:

    # pre-fix, straight out of git history
    git archive 3765cf7 kaggle_submission_llm_world_engine/dataset_stage \\
        | tar -x -C /tmp/prefix
    python scripts/demo_codeworld_prefix_vs_postfix.py \\
        /tmp/prefix/kaggle_submission_llm_world_engine/dataset_stage

    # post-fix
    python scripts/demo_codeworld_prefix_vs_postfix.py \\
        kaggle_submission_llm_world_engine/dataset_stage

No model is loaded and no network call is made -- the "coder LLM" is a
canned-response fake, reused from tests/conftest.py.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_test_helpers():
    """Pull the fakes out of tests/conftest.py without letting its own
    module-level sys.path.insert override the stage dir under test."""
    spec = importlib.util.spec_from_file_location(
        "cw_test_helpers", REPO_ROOT / "tests" / "conftest.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    spec.loader.exec_module(module)
    sys.path[:] = saved
    return module


def main(stage_dir: str) -> int:
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(Path(stage_dir).resolve()))

    helpers = _load_test_helpers()
    CORRECT = helpers.CORRECT_WORLD_MODEL_SOURCE
    FakeLLMClient = helpers.FakeLLMClient
    build_transcript = helpers.build_transcript

    from llm_engine.drafting import draft_world_model
    from llm_engine.llm_client import extract_code
    from llm_engine.replay import replay
    from llm_engine.types import Action, GameTranscript, Transition
    from llm_engine.world_model import WORLD_MODEL_SKELETON, load_world_model

    def fenced(s: str) -> str:
        return f"Here is the model:\n\n```python\n{s}\n```\n"

    def failed_draft_no_stub():
        dud = ("class WorldModel:\n    def predict(self, s, a, x=None, y=None):\n"
               "        return s, 0, False\n    def goal_hint(self, s):\n        return 0.0\n")
        out = draft_world_model(FakeLLMClient([fenced(dud)] * 3), build_transcript(), max_attempts=3)
        is_stub = out.source is not None and out.source.strip() == WORLD_MODEL_SKELETON.strip()
        return (out.world_model is None and not is_stub,
                f"world_model={'None' if out.world_model is None else 'INSTALLED'}, "
                f"source_is_template_stub={is_stub}")

    def truncated_fence():
        code = extract_code(
            "Sure:\n\n```python\nclass WorldModel:\n    def predict(self, s, a, x=None, y=None):\n"
            "        return s, 0, False\n"
        )
        return code.startswith("class WorldModel:"), f"extracted starts with {code[:22]!r}"

    def think_block():
        response = ("<think>\nrough idea:\n```python\nclass WorldModel: pass\n```\n</think>\n"
                    "```python\n" + CORRECT + "```\n")
        code = extract_code(response)
        return "def goal_hint" in code, f"extracted {code[:30]!r}"

    def retry_keeps_transcript():
        client = FakeLLMClient(["```python\nthis is not python(\n```", fenced(CORRECT)])
        out = draft_world_model(client, build_transcript(), max_attempts=3)
        second = client.prompts[1][1] if len(client.prompts) > 1 else ""
        return "transition #0" in second, f"draft_ok={out.ok}, retry_has_transcript={'transition #0' in second}"

    def sandbox_super():
        loaded = load_world_model(
            "class WorldModel:\n    def __init__(self):\n        super().__init__()\n"
            "    def predict(self, s, a, x=None, y=None): return s, 0, False\n"
            "    def goal_hint(self, s): return 0.0"
        )
        return loaded.ok, f"load ok={loaded.ok} err={loaded.error}"

    def correct_model_accepted():
        out = draft_world_model(FakeLLMClient([fenced(CORRECT)]), build_transcript(), max_attempts=3)
        return out.ok, f"ok={out.ok} attempts={out.attempts}"

    def malformed_predict():
        source = ("class WorldModel:\n"
                  "    def predict(self, state, action_name, x=None, y=None):\n"
                  "        return [state[0][0]], 0, False\n"
                  "    def goal_hint(self, state):\n        return 0.0\n")
        loaded = load_world_model(source)
        transcript = GameTranscript(game_id="g")
        transcript.append(Transition(
            frame_before=[[[0, 1], [2, 3]]], action=Action(name="ACTION1"),
            frame_after=[[[9, 9], [9, 9]]], levels_completed_before=0,
            levels_completed_after=0, state_after="NOT_FINISHED",
        ))
        result = replay(transcript, loaded.world_model)
        return (not result.passed and result.first_failure is not None,
                f"replay returned cleanly, reason={(result.first_failure.reason or '')[:70]!r}")

    scenarios = [
        ("failed draft does NOT hand back the template stub", failed_draft_no_stub),
        ("extract_code recovers a truncated fence", truncated_fence),
        ("extract_code ignores a <think> sketch", think_block),
        ("retry prompt still contains the transcript", retry_keeps_transcript),
        ("sandbox accepts super()", sandbox_super),
        ("a correct model is accepted", correct_model_accepted),
        ("malformed predict() -> replay failure, not TypeError", malformed_predict),
    ]

    print(f"--- llm_engine from: {stage_dir} ---")
    failures = 0
    for name, fn in scenarios:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 -- the point is to report, not propagate
            frame = traceback.extract_tb(sys.exc_info()[2])[-1]
            ok = False
            detail = f"RAISED {type(e).__name__}: {e} @ {Path(frame.filename).name}:{frame.lineno}"
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
