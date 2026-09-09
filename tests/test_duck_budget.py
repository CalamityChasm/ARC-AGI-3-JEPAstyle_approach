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

from kaggle_submission_duck.duck_budget import (
    DEFAULT_TARGET_CONCURRENCY,
    ConcurrencyOverride,
    RerunBudget,
    compute_rerun_budget,
    concurrency_gain_factor,
    resolve_concurrency,
    wave_count,
)


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


# ---------------------------------------------------------------------------
# stage7-duck-concurrency: wave arithmetic under a concurrency override
# ---------------------------------------------------------------------------


def test_wave_count_matches_ceil_division():
    assert wave_count(110, 28) == 4
    assert wave_count(110, 37) == 3
    assert wave_count(110, 55) == 2
    assert wave_count(110, 110) == 1
    assert wave_count(110, 200) == 1  # concurrency above the game count
    for n in (1, 25, 55, 110, 111):
        for c in (1, 7, 28, 37, 48, 64):
            assert wave_count(n, c) == math.ceil(n / c)


def test_wave_count_rejects_non_positive():
    with pytest.raises(ValueError):
        wave_count(0, 28)
    with pytest.raises(ValueError):
        wave_count(110, 0)


def test_resolve_concurrency_28_to_37_removes_a_wave():
    # The headline hypothesis: ceil(110/37) = 3 waves instead of 4.
    override = resolve_concurrency(inherited=28, n_games=110, target=37)

    assert override.inherited == 28
    assert override.target == 37
    assert override.effective == 37
    assert override.waves_before == 4
    assert override.waves_after == 3
    assert override.changed is True


def test_resolve_concurrency_none_target_is_a_noop():
    override = resolve_concurrency(inherited=28, n_games=110, target=None)

    assert override.effective == 28
    assert override.waves_before == override.waves_after == 4
    assert override.changed is False


def test_resolve_concurrency_same_target_is_a_noop():
    override = resolve_concurrency(inherited=28, n_games=110, target=28)
    assert override.changed is False
    assert override.effective == 28


def test_resolve_concurrency_non_positive_target_is_a_noop():
    # Defensive: a mistyped/zeroed constant must not disable the solver.
    for bad in (0, -1):
        override = resolve_concurrency(inherited=28, n_games=110, target=bad)
        assert override.effective == 28
        assert override.changed is False


def test_resolve_concurrency_rejects_bad_inputs():
    with pytest.raises(ValueError):
        resolve_concurrency(inherited=0, n_games=110, target=37)
    with pytest.raises(ValueError):
        resolve_concurrency(inherited=28, n_games=0, target=37)


def test_resolve_concurrency_log_line_format():
    line = resolve_concurrency(inherited=28, n_games=110, target=37).log_line()
    assert "inherited=28" in line
    assert "target=37" in line
    assert "effective=37" in line
    assert "n_games=110" in line
    assert "waves 4 -> 3" in line
    assert "changed=True" in line


def test_default_target_concurrency_is_a_plain_int_or_none():
    # It is mirrored verbatim into the notebook, so it must be trivially
    # representable there (no imports, no expressions).
    assert DEFAULT_TARGET_CONCURRENCY is None or isinstance(DEFAULT_TARGET_CONCURRENCY, int)


def test_budget_follows_the_overridden_concurrency():
    # The two pieces compose: override concurrency first, then feed the
    # EFFECTIVE value into compute_rerun_budget so the wave count (and
    # therefore the per-game cap) reflects the override. This is exactly
    # the ordering the notebook cell uses.
    elapsed_s = 394.0 + 300.0
    override = resolve_concurrency(inherited=28, n_games=110, target=37)
    budget = compute_rerun_budget(
        n_games=110, concurrency=override.effective, elapsed_s=elapsed_s
    )

    assert budget.concurrency == 37
    assert budget.waves == 3  # not 4
    expected_remaining = 9 * 3600 - elapsed_s - 900.0
    assert budget.max_runtime_s_per_game == pytest.approx(expected_remaining / 3)
    # Fewer waves means a LONGER per-game cap than the 4-wave case...
    four_wave = compute_rerun_budget(n_games=110, concurrency=28, elapsed_s=elapsed_s)
    assert budget.max_runtime_s_per_game > four_wave.max_runtime_s_per_game
    # ...but the same total wave time, which is the whole point: wall-clock
    # is conserved, it is only redistributed.
    assert budget.waves * budget.max_runtime_s_per_game == pytest.approx(
        four_wave.waves * four_wave.max_runtime_s_per_game
    )


# --- the token-neutrality argument, as arithmetic --------------------------


def test_gain_factor_is_the_wave_packing_effect_when_throughput_is_flat():
    # If aggregate throughput does NOT rise with concurrency (ratio 1.0),
    # the only thing left is ceil() quantisation: 28 -> 4 waves wastes 2 of
    # 112 slots, 37 -> 3 waves wastes 1 of 111. Small, but not nothing.
    gain = concurrency_gain_factor(
        n_games=110, inherited=28, target=37, throughput_ratio=1.0
    )
    assert gain == pytest.approx((28 / 37) * (4 / 3))
    assert gain == pytest.approx(1.009, abs=1e-3)  # ~0.9%, i.e. noise


def test_gain_factor_is_exactly_one_when_waves_divide_evenly():
    # With a game count that divides evenly at both concurrencies there is
    # no quantisation benefit at all -- token-neutrality is exact.
    gain = concurrency_gain_factor(
        n_games=120, inherited=30, target=60, throughput_ratio=1.0
    )
    assert wave_count(120, 30) == 4 and wave_count(120, 60) == 2
    assert gain == pytest.approx(1.0)


def test_gain_factor_rises_only_if_measured_throughput_rises():
    # A real 25% aggregate-throughput gain at the higher concurrency.
    better = concurrency_gain_factor(
        n_games=110, inherited=28, target=37, throughput_ratio=1.25
    )
    flat = concurrency_gain_factor(
        n_games=110, inherited=28, target=37, throughput_ratio=1.0
    )
    worse = concurrency_gain_factor(
        n_games=110, inherited=28, target=37, throughput_ratio=0.8
    )
    assert better > flat > worse
    assert better == pytest.approx(1.25 * flat)
    assert worse < 1.0  # a throughput regression makes the change harmful


def test_gain_factor_rejects_non_positive_throughput_ratio():
    with pytest.raises(ValueError):
        concurrency_gain_factor(n_games=110, inherited=28, target=37, throughput_ratio=0.0)
    with pytest.raises(ValueError):
        concurrency_gain_factor(n_games=110, inherited=28, target=37, throughput_ratio=-1.0)


def test_returns_concurrency_override_instance():
    override = resolve_concurrency(inherited=28, n_games=110, target=37)
    assert isinstance(override, ConcurrencyOverride)
