"""Unit tests for kaggle_submission_duck/duck_budget.py.

Covers the cases called out in the stage7-duck-budget-fix task: the real
110-game/concurrency-28 shape, the local 25-game public shape, a
degenerate single-game case, and a case engineered to hit the floor
(``min_per_game_s``) so a near-zero/negative budget can't collapse the
per-game cap to something unplayable.

Run with: pytest tests/test_duck_budget.py -v
(repo root must be on sys.path -- see pytest.ini's `pythonpath = .`).
"""

from __future__ import annotations

import math

import pytest

from kaggle_submission_duck.duck_budget import RerunBudget, compute_rerun_budget


def test_110_games_concurrency_28_matches_hand_derivation():
    # Matches experiments/stage7_duck_budget_fix.md's own hand-derivation:
    # ceil(110/28) = 4 waves. Use elapsed_s=0 for a clean check of the
    # wave count and the remaining-budget arithmetic in isolation.
    budget = compute_rerun_budget(n_games=110, concurrency=28, elapsed_s=0.0)

    assert budget.n_games == 110
    assert budget.concurrency == 28
    assert budget.waves == 4  # ceil(110/28) = ceil(3.928...) = 4
    assert budget.remaining_s == pytest.approx(9 * 3600 - 900.0)
    assert budget.per_game_raw_s == pytest.approx((9 * 3600 - 900.0) / 4)
    # Well above the floor, so max_runtime_s_per_game == the raw value.
    assert budget.max_runtime_s_per_game == pytest.approx(budget.per_game_raw_s)
    # Sanity: 4 waves at this per-game cap must fit inside the 9h budget
    # net of the safety margin (the whole point of the fix).
    assert budget.waves * budget.max_runtime_s_per_game <= 9 * 3600 - 900.0 + 1e-6


def test_110_games_concurrency_28_with_realistic_elapsed_time():
    # Mirrors the diagnosis's own measured setup cost (~394s) plus a
    # plausible rerun-only gateway wait, to check the fix actually buys a
    # real per-game reduction versus the inherited constant (7920.0s).
    elapsed_s = 394.0 + 300.0  # measured setup + partial gateway wait
    budget = compute_rerun_budget(n_games=110, concurrency=28, elapsed_s=elapsed_s)

    assert budget.waves == 4
    expected_remaining = 9 * 3600 - elapsed_s - 900.0
    assert budget.remaining_s == pytest.approx(expected_remaining)
    assert budget.max_runtime_s_per_game == pytest.approx(expected_remaining / 4)
    # This is the actual defect being fixed: the inherited constant
    # (7920.0s) overruns; the computed value must be measurably smaller.
    assert budget.max_runtime_s_per_game < 7920.0
    # And the resulting total wave time must fit inside what's left of the
    # 9h cap after elapsed time and the safety margin are both accounted for.
    total_wave_time = budget.waves * budget.max_runtime_s_per_game
    assert total_wave_time <= expected_remaining + 1e-6


def test_25_games_concurrency_28_single_wave():
    # The local/public shape (25 public games, same concurrency=28):
    # everything fits in one wave, so the computed per-game budget should
    # be generous -- this checks the fix does something sane outside the
    # 110-game rerun shape too, even though the notebook patch itself only
    # ever runs this arithmetic when true_submission is True.
    budget = compute_rerun_budget(n_games=25, concurrency=28, elapsed_s=394.0)

    assert budget.waves == 1  # ceil(25/28) = 1
    assert budget.max_runtime_s_per_game == pytest.approx(9 * 3600 - 394.0 - 900.0)


def test_single_game():
    budget = compute_rerun_budget(n_games=1, concurrency=28, elapsed_s=0.0)

    assert budget.waves == 1  # ceil(1/28) = 1
    assert budget.max_runtime_s_per_game == pytest.approx(9 * 3600 - 900.0)


def test_degenerate_case_hits_the_floor():
    # Elapsed time already exceeds the nominal budget net of the safety
    # margin -- remaining_s goes negative, so per_game_raw_s is negative
    # too. Without the floor this would tell the solver to give every
    # game an ~instantly-expiring (or negative) runtime cap. The floor
    # must kick in and hold the per-game cap at min_per_game_s instead.
    budget = compute_rerun_budget(n_games=110, concurrency=28, elapsed_s=9 * 3600)

    assert budget.remaining_s < 0
    assert budget.per_game_raw_s < 0
    assert budget.max_runtime_s_per_game == 600.0  # the default floor


def test_degenerate_case_lopsided_ratio_hits_the_floor():
    # A pathologically large game count relative to concurrency (e.g. a
    # concurrency=1 misconfiguration) should also be caught by the floor
    # rather than silently producing a near-zero per-game cap.
    budget = compute_rerun_budget(n_games=1000, concurrency=1, elapsed_s=0.0)

    assert budget.waves == 1000
    assert budget.per_game_raw_s < 600.0
    assert budget.max_runtime_s_per_game == 600.0


def test_custom_floor_is_respected():
    budget = compute_rerun_budget(
        n_games=110,
        concurrency=28,
        elapsed_s=9 * 3600,
        min_per_game_s=120.0,
    )
    assert budget.max_runtime_s_per_game == 120.0


@pytest.mark.parametrize("n_games,concurrency,expected_waves", [
    (110, 28, 4),
    (110, 37, 3),
    (55, 28, 2),
    (56, 28, 2),
    (28, 28, 1),
    (29, 28, 2),
])
def test_wave_count_ceil_division(n_games, concurrency, expected_waves):
    budget = compute_rerun_budget(n_games=n_games, concurrency=concurrency, elapsed_s=0.0)
    assert budget.waves == expected_waves
    assert budget.waves == math.ceil(n_games / concurrency)


def test_zero_or_negative_n_games_raises():
    with pytest.raises(ValueError):
        compute_rerun_budget(n_games=0, concurrency=28, elapsed_s=0.0)
    with pytest.raises(ValueError):
        compute_rerun_budget(n_games=-5, concurrency=28, elapsed_s=0.0)


def test_zero_or_negative_concurrency_raises():
    with pytest.raises(ValueError):
        compute_rerun_budget(n_games=110, concurrency=0, elapsed_s=0.0)
    with pytest.raises(ValueError):
        compute_rerun_budget(n_games=110, concurrency=-1, elapsed_s=0.0)


def test_log_line_format():
    budget = compute_rerun_budget(n_games=110, concurrency=28, elapsed_s=394.0)
    line = budget.log_line()
    assert "n_games=110" in line
    assert "conc=28" in line
    assert "waves=4" in line
    assert "per_game=" in line
    assert "remaining=" in line


def test_returns_rerun_budget_instance():
    budget = compute_rerun_budget(n_games=110, concurrency=28, elapsed_s=0.0)
    assert isinstance(budget, RerunBudget)
