"""LLM-call budget tracking. See architecture.md's "Budget awareness" and
plan.md's Stage 5: with a 9-hour cap shared across up to 110 private games,
LLM calls (not compute in general) are the scarce resource. This is a
simple call-count budget (not token-metered yet -- a reasonable Stage 6
refinement if call count alone proves too coarse).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMBudget:
    max_calls_per_game: int = 20
    calls_used: int = 0
    call_log: list[str] = field(default_factory=list)

    def has_budget(self) -> bool:
        return self.calls_used < self.max_calls_per_game

    def record(self, reason: str) -> None:
        self.calls_used += 1
        self.call_log.append(reason)

    def remaining(self) -> int:
        return max(0, self.max_calls_per_game - self.calls_used)
