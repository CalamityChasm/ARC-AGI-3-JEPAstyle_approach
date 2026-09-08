"""Pure arithmetic for the Duck-harness competition-rerun time-budget fix.

Extracted so it can be unit-tested (see ``tests/test_duck_budget.py``)
independently of the Kaggle notebook it is inlined into (Kaggle kernels
have no import path back into this repo, so the notebook cell carries its
own copy of this same formula -- see
``kaggle_submission_duck/notebook/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb``,
cell 9, the block bracketed by ``stage7-duck-budget-fix`` comments, and
``experiments/stage7_duck_budget_fix.md`` for the full diagnosis this is
based on).

Background: the notebook's competition-rerun branch inherits
``concurrency=28`` / ``max_runtime_s_per_game=7920.0`` from a pickled
bundle tuned for a 6-game x 4-pass Preview shape. Against the real ~110
hidden games under ARC-AGI-3's 9-hour hard run-time cap, and given every
game observed in a real run burns its *entire* per-game cap (verified:
25/25 games in a real fork run ended in state ``gave_up`` at >=7900s,
none finished early), the pool behaves as fixed-length waves:
``waves = ceil(n_games / concurrency)``. With the inherited values that is
4 waves x 7920s = 8.80h, plus ~394s measured setup and up to 600s of
rerun-only gateway wait, against the 9h cap -- and the rerun branch's own
``_soft_end_time()`` returns ``None`` (no graceful wind-down at all).

This module recomputes ``max_runtime_s_per_game`` from the *live* game
count and elapsed wall-clock instead of using the inherited constant, so
the last wave has a chance to fit inside the real budget rather than
running every game to a now ill-fitting fixed cap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerunBudget:
    """Computed rerun time-budget, plus the inputs that produced it.

    All fields are here (not just ``max_runtime_s_per_game``) so a caller
    can log every intermediate value for diagnosability -- exactly the
    requirement the fix was scoped under, since no scored-rerun log is
    ever retrievable after the fact (see CLAUDE.md's Kaggle debugging
    history: "Kaggle does not expose the actual execution log from a real
    scored competition rerun").
    """

    n_games: int
    concurrency: int
    waves: int
    remaining_s: float
    per_game_raw_s: float
    max_runtime_s_per_game: float

    def log_line(self) -> str:
        return (
            f"rerun budget: n_games={self.n_games}, conc={self.concurrency}, "
            f"waves={self.waves}, per_game={self.max_runtime_s_per_game:.0f}s, "
            f"remaining={self.remaining_s:.0f}s"
        )


def compute_rerun_budget(
    n_games: int,
    concurrency: int,
    elapsed_s: float,
    *,
    total_budget_s: float = 9 * 3600,
    safety_margin_s: float = 900.0,
    min_per_game_s: float = 600.0,
) -> RerunBudget:
    """Compute a per-game runtime cap sized to fit ``n_games`` in the
    remaining wall-clock budget, split into ``ceil(n_games / concurrency)``
    fixed-length waves (matching this solver's observed behaviour: every
    game burns its entire allotted cap rather than finishing early).

    Parameters
    ----------
    n_games:
        Live game count for this rerun (``len(bm.games)`` after
        ``_competition_games()`` populates it from the real gateway).
    concurrency:
        The solver's configured concurrent-game count (``bm.solver.concurrency``,
        28 in the inherited bundle).
    elapsed_s:
        Wall-clock seconds already spent since notebook start
        (``time.time() - NOTEBOOK_START_EPOCH``) -- covers setup, model
        load, and the rerun-only gateway wait, all of which eat into the
        same 9h hard cap before any game starts.
    total_budget_s:
        The competition's hard run-time cap (9 hours, per ``rules.md``).
    safety_margin_s:
        Held back unconditionally (900s = 15 min default) so the last
        wave has a chance to be scored/torn down before Kaggle's own hard
        kill, not just before the nominal budget line.
    min_per_game_s:
        Floor on the resulting per-game cap. Protects against a
        degenerate input (e.g. ``elapsed_s`` already close to or past
        ``total_budget_s``, or an unexpectedly large ``n_games``/``concurrency``
        ratio) collapsing the budget to zero or a negative number, which
        would make every game fail near-instantly instead of getting a
        real, if reduced, chance to play.

    Raises
    ------
    ValueError
        If ``n_games`` or ``concurrency`` is not positive.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games!r}")
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency!r}")

    waves = max(1, -(-n_games // concurrency))  # ceil division, no math import
    remaining_s = total_budget_s - elapsed_s - safety_margin_s
    per_game_raw_s = remaining_s / waves
    max_runtime_s_per_game = max(min_per_game_s, per_game_raw_s)

    return RerunBudget(
        n_games=n_games,
        concurrency=concurrency,
        waves=waves,
        remaining_s=remaining_s,
        per_game_raw_s=per_game_raw_s,
        max_runtime_s_per_game=max_runtime_s_per_game,
    )


# ---------------------------------------------------------------------------
# stage7-duck-concurrency: the *other* lever on the same arithmetic
# ---------------------------------------------------------------------------
# ``compute_rerun_budget`` above divides the remaining wall-clock by
# ``waves = ceil(n_games / concurrency)``. Concurrency is therefore the only
# input that changes the *number* of waves rather than the length of each one:
# at 110 games, 28 -> 4 waves, 37 -> 3 waves, 55 -> 2 waves.
#
# THE TOKEN-NEUTRALITY ARGUMENT (why this is not obviously a win).
# Total wall-clock is fixed by the 9h cap, and aggregate token throughput is a
# property of the vLLM server, not of how the harness slices its games. So
# total tokens generated over the run is approximately
# ``aggregate_throughput x wall_clock`` REGARDLESS of concurrency. Raising
# concurrency gives each game more wall-clock (fewer waves) but a
# proportionally thinner slice of the GPU (more sequences sharing it), and
# tokens-per-game come out about the same:
#
#     tokens_per_game ~= (aggregate_tps / concurrency) x (wall_clock / waves)
#
# Substituting waves = n_games / concurrency, the concurrency terms cancel:
#
#     tokens_per_game ~= aggregate_tps x wall_clock / n_games
#
# **Raising concurrency is therefore only a win if aggregate throughput
# actually RISES with more concurrent sequences.** That is a measurable
# property of the server, not something derivable from the harness config --
# see ``experiments/stage7_duck_concurrency.md`` for the measurement.
#
# ``concurrency_gain_factor`` below makes that dependency explicit: it is the
# only place a measured throughput ratio enters the arithmetic.

DEFAULT_TARGET_CONCURRENCY = 28
"""Target concurrency for the competition rerun.

Single named constant, mirrored verbatim into the notebook's cell 9 (a Kaggle
kernel has no import path back into this repo). Set to ``None`` -- or to the
inherited value -- to leave ``bm.solver.concurrency`` untouched.
"""


@dataclass(frozen=True)
class ConcurrencyOverride:
    """Result of resolving a concurrency override, with its wave arithmetic."""

    inherited: int
    target: int | None
    effective: int
    n_games: int
    waves_before: int
    waves_after: int
    changed: bool

    def log_line(self) -> str:
        return (
            f"rerun concurrency: inherited={self.inherited}, "
            f"target={self.target}, effective={self.effective}, "
            f"n_games={self.n_games}, waves {self.waves_before} -> "
            f"{self.waves_after}, changed={self.changed}"
        )


def wave_count(n_games: int, concurrency: int) -> int:
    """``ceil(n_games / concurrency)``, floored at 1.

    The solver runs games in fixed-length waves because no game finishes
    early (verified: 25/25 games in a real fork run ended ``gave_up`` at
    >=7900s of a 7920s cap), so the wave count is what actually governs how
    the 9h budget is divided.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games!r}")
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency!r}")
    return max(1, -(-n_games // concurrency))  # ceil division, no math import


def resolve_concurrency(
    inherited: int,
    n_games: int,
    target: int | None = DEFAULT_TARGET_CONCURRENCY,
) -> ConcurrencyOverride:
    """Decide the effective concurrency and report the resulting wave change.

    Parameters
    ----------
    inherited:
        ``bm.solver.concurrency`` as loaded from the pickled bundle (28).
    n_games:
        Live game count for this rerun (``len(bm.games)``).
    target:
        Desired concurrency. ``None`` (or a non-positive value) means "leave
        the inherited value alone" -- the override is then a no-op and
        ``changed`` is False.

    Raises
    ------
    ValueError
        If ``inherited`` or ``n_games`` is not positive.
    """
    if inherited <= 0:
        raise ValueError(f"inherited concurrency must be positive, got {inherited!r}")
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games!r}")

    effective = inherited if (target is None or target <= 0) else int(target)
    return ConcurrencyOverride(
        inherited=inherited,
        target=target,
        effective=effective,
        n_games=n_games,
        waves_before=wave_count(n_games, inherited),
        waves_after=wave_count(n_games, effective),
        changed=effective != inherited,
    )


def concurrency_gain_factor(
    n_games: int,
    inherited: int,
    target: int,
    throughput_ratio: float,
) -> float:
    """Expected change in tokens-generated-per-game from a concurrency change.

    This is the token-neutrality argument in the module comment above, made
    numeric. Returns a multiplier: ``1.0`` means the change is token-neutral,
    ``>1.0`` means each game gets more model output, ``<1.0`` means less.

    ``throughput_ratio`` is the MEASURED aggregate output tokens/sec at
    ``target`` concurrency divided by the same at ``inherited`` concurrency.
    If the server's aggregate throughput is flat in concurrency (ratio 1.0),
    the result is the pure wave-quantisation effect -- which is *not*
    guaranteed to be 1.0, because ``ceil`` makes waves lumpy: at 110 games,
    28 -> 4 waves is only 78% packed (112 slots for 110 games), whereas
    37 -> 3 waves is 99% packed (111 slots), so a small real gain survives
    even at flat throughput.

    Raises
    ------
    ValueError
        On non-positive inputs.
    """
    if throughput_ratio <= 0:
        raise ValueError(f"throughput_ratio must be positive, got {throughput_ratio!r}")
    waves_before = wave_count(n_games, inherited)
    waves_after = wave_count(n_games, target)
    # per-game tokens ~= (aggregate_tps / concurrency) x (wall_clock / waves)
    # so the ratio of the two configurations is:
    return throughput_ratio * (inherited / target) * (waves_before / waves_after)
