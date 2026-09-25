"""Aggregate segment results into the counted figures the arm reports.

Every number here is a count or a ratio of counts over a fixed set of
segments. None of it is a sampled score, so none of it carries the
+/-2.46 SE that makes a single free public-25 run unable to rank
candidates.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .harness import SegmentResult


@dataclass
class BacktestReport:
    results: list[SegmentResult] = field(default_factory=list)

    # -- headline -----------------------------------------------------

    @property
    def n_segments(self) -> int:
        return len(self.results)

    @property
    def n_games(self) -> int:
        return len({r.game_id for r in self.results})

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_segments if self.n_segments else 0.0

    @property
    def games_with_a_pass(self) -> int:
        return len({r.game_id for r in self.results if r.passed})

    # -- the null control ---------------------------------------------

    @property
    def n_identity_passed(self) -> int:
        """Segments the do-nothing model already passes.

        A pass rate is only meaningful net of this.
        """
        return sum(1 for r in self.results if r.identity_passed)

    @property
    def informative(self) -> list[SegmentResult]:
        """Segments where 'nothing changes' is not already a valid model."""
        return [r for r in self.results if not r.identity_passed]

    @property
    def informative_pass_rate(self) -> float:
        pool = self.informative
        return (sum(1 for r in pool if r.passed) / len(pool)) if pool else 0.0

    @property
    def n_beating_identity(self) -> int:
        return sum(1 for r in self.results if r.beats_identity)

    # -- graded signal ------------------------------------------------

    @property
    def median_prefix_fraction(self) -> float:
        """Median share of steps reproduced before the first mismatch.

        The graded companion to the binary pass rate: a model getting 80%
        of a level's dynamics right is a different situation from one
        getting 2%, and both show up as 'failed'.
        """
        if not self.results:
            return 0.0
        return statistics.median(r.best_prefix_fraction for r in self.results)

    @property
    def outcome_counts(self) -> Counter:
        return Counter(r.outcome for r in self.results)

    @property
    def total_elapsed_s(self) -> float:
        return sum(r.elapsed_s for r in self.results)

    @property
    def total_attempts(self) -> int:
        return sum(r.attempts for r in self.results)

    # -- output -------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "n_segments": self.n_segments,
            "n_games": self.n_games,
            "n_passed": self.n_passed,
            "pass_rate": self.pass_rate,
            "games_with_a_pass": self.games_with_a_pass,
            "n_identity_passed": self.n_identity_passed,
            "n_informative": len(self.informative),
            "informative_pass_rate": self.informative_pass_rate,
            "n_beating_identity": self.n_beating_identity,
            "median_prefix_fraction": self.median_prefix_fraction,
            "outcome_counts": dict(self.outcome_counts),
            "total_attempts": self.total_attempts,
            "total_elapsed_s": self.total_elapsed_s,
            "segments": [r.as_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [
            "CodeWorldModel backtest",
            "=" * 62,
            f"segments            : {self.n_segments} across {self.n_games} games",
            f"replay-passing      : {self.n_passed}/{self.n_segments} "
            f"({self.pass_rate:.1%}) in {self.games_with_a_pass} game(s)",
            "",
            "null control (do-nothing model)",
            "-" * 62,
            f"identity passes     : {self.n_identity_passed}/{self.n_segments} "
            "(segments that teach nothing)",
            f"informative segments: {len(self.informative)}",
            f"pass rate on those  : {self.informative_pass_rate:.1%}",
            f"beats identity      : {self.n_beating_identity}/{self.n_segments}",
            "",
            "graded signal",
            "-" * 62,
            f"median prefix       : {self.median_prefix_fraction:.1%} of steps "
            "reproduced before first mismatch",
            "",
            "failure taxonomy",
            "-" * 62,
        ]
        for outcome, count in self.outcome_counts.most_common():
            lines.append(f"  {outcome:18s} {count:4d}")
        lines += [
            "",
            f"attempts            : {self.total_attempts}",
            f"wall clock          : {self.total_elapsed_s:.1f}s",
        ]
        return "\n".join(lines)


def build_report(results: Iterable[SegmentResult]) -> BacktestReport:
    return BacktestReport(results=list(results))


def per_game_table(results: Sequence[SegmentResult]) -> str:
    """One row per game: segments, passes, best prefix reached."""
    by_game: dict[str, list[SegmentResult]] = {}
    for r in results:
        by_game.setdefault(r.game_id, []).append(r)

    lines = [f"{'game':22s} {'segs':>5s} {'pass':>5s} {'idpass':>7s} {'best prefix':>12s}"]
    lines.append("-" * 56)
    for game in sorted(by_game):
        rows = by_game[game]
        best = max((r.best_prefix_fraction for r in rows), default=0.0)
        lines.append(
            f"{game:22s} {len(rows):5d} {sum(1 for r in rows if r.passed):5d} "
            f"{sum(1 for r in rows if r.identity_passed):7d} {best:11.0%}"
        )
    return "\n".join(lines)
