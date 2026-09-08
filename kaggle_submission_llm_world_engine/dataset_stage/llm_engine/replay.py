"""Validate a candidate WorldModel by replaying the observed transcript
through it. This is the hard gate described in architecture.md's "Validate
by replay, not by trust" -- a code revision is only ever accepted if it
reproduces every transition seen so far, not just the one that motivated
the revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .diff import format_diff
from .types import GameTranscript, Transition
from .world_model import WorldModelProtocol, safe_predict


@dataclass
class TransitionCheck:
    index: int
    passed: bool
    reason: Optional[str] = None  # populated on failure


@dataclass
class ReplayResult:
    passed: bool
    checks: list[TransitionCheck]
    first_failure: Optional[TransitionCheck] = None

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


def replay(transcript: GameTranscript, model: WorldModelProtocol) -> ReplayResult:
    """Run every transition in order through model.predict and compare
    against what actually happened. Stops recording new checks after the
    first failure is enough context for a repair prompt, but still counts
    every transition so pass_count/total is meaningful."""
    checks: list[TransitionCheck] = []
    first_failure: Optional[TransitionCheck] = None

    for i, t in enumerate(transcript.transitions):
        check = _check_one(i, t, model)
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

    return ReplayResult(passed=first_failure is None, checks=checks, first_failure=first_failure)


def _check_one(index: int, t: Transition, model: WorldModelProtocol) -> TransitionCheck:
    predicted_state, predicted_levels_delta, predicted_done, error = safe_predict(
        model, t.frame_before, t.action
    )
    if error is not None:
        return TransitionCheck(index=index, passed=False, reason=error)

    reasons = []
    if predicted_state != t.frame_after:
        reasons.append(f"grid mismatch: {format_diff(predicted_state, t.frame_after)}")
    if predicted_levels_delta != t.levels_delta:
        reasons.append(f"levels_delta mismatch: predicted {predicted_levels_delta}, actual {t.levels_delta}")
    if predicted_done != t.done:
        reasons.append(f"done mismatch: predicted {predicted_done}, actual {t.done}")

    if reasons:
        return TransitionCheck(index=index, passed=False, reason="; ".join(reasons))
    return TransitionCheck(index=index, passed=True)


def describe_failure(transcript: GameTranscript, check: TransitionCheck) -> str:
    """Human/LLM-readable description of one failing transition, for use
    in a repair prompt."""
    t = transcript.transitions[check.index]
    return (
        f"Transition #{check.index}: action={t.action}, "
        f"levels_completed {t.levels_completed_before}->{t.levels_completed_after}, "
        f"state_after={t.state_after}\n"
        f"Your model's predict() was wrong: {check.reason}"
    )
