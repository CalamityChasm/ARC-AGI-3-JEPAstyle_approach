"""Tests for the CodeWorldModel backtest instrument.

Weighted deliberately toward `arc3_cwm.extract`. Every prior data-pipeline
bug in this project was silent -- the `action_input` bug made every
recorded action look like RESET and invalidated four stages of results
before anyone noticed, and a stale checkpoint sat in `checkpoints/` for
weeks. A backtest whose extractor is subtly wrong produces a confident,
plausible, entirely meaningless number, which is worse than no number.

Grids here are small (4x4, 3x3) rather than the real 64x64. Nothing in
the extractor, renderer, oracle or replay path assumes a size, and the
one place that could (`_is_grid`) is tested for rectangularity directly.
"""

from __future__ import annotations

import json

import pytest

from arc3_cwm import determinism, oracle
from arc3_cwm.extract import (
    ExtractionError,
    ExtractionStats,
    LevelSegment,
    _is_grid,
    _parse_mouse_xy,
    extract_run,
    game_id_from_path,
    segments_from_events,
)
from arc3_cwm.harness import (
    BacktestConfig,
    GRID_MISMATCH,
    LOAD_ERROR,
    NO_CODE,
    NO_RESPONSE,
    PASSED,
    identity_baseline,
    run_segment,
)
from arc3_cwm.render import build_user_prompt, render_segment
from arc3_cwm.report import build_report

# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------


def grid(*rows: str) -> list[list[int]]:
    """'0120' -> [0,1,2,0], one string per row."""
    return [[int(c, 16) for c in row] for row in rows]


_DEFAULT = object()


def event(
    type_="action",
    board=_DEFAULT,
    action_name="ACTION1",
    level=1,
    score=0,
    state="NOT_FINISHED",
    **extra,
):
    """`board=None` means a genuinely boardless event; omitting it gives a
    default grid. A plain `board=None` default would make the two
    indistinguishable and silently skip the boardless-event test."""
    e = {
        "type": type_,
        "board": grid("00", "00") if board is _DEFAULT else board,
        "action_name": action_name,
        "level": level,
        "score": score,
        "state": state,
    }
    e.update(extra)
    return e


def make_segment(pairs, game_id="gx01", level=1) -> LevelSegment:
    """Build a segment from (before, action_name, after) triples."""
    stats = ExtractionStats()
    events = [event(type_="initial", board=pairs[0][0])]
    for _before, action_name, after in pairs:
        events.append(event(board=after, action_name=action_name))
    segments = segments_from_events(game_id, events, stats)
    assert len(segments) == 1
    seg = segments[0]
    seg.level = level
    return seg


# --------------------------------------------------------------------
# extract: the ACTION6 coordinate mapping
# --------------------------------------------------------------------


def test_mouse_display_maps_col_to_x_and_row_to_y():
    """The log records row/col; Action takes (x, y).

    Transposing this would silently move every click and would not show
    up in any aggregate pass rate. The string below is the exact format
    emitted by a real anim run (ar25-0c556536, 2026-09-21).
    """
    assert _parse_mouse_xy("MOUSE(row=23, col=52)") == (52, 23)
    assert _parse_mouse_xy("MOUSE(row=0, col=63)") == (63, 0)


def test_mouse_display_tolerates_whitespace_and_rejects_other_text():
    assert _parse_mouse_xy("MOUSE( row = 5 , col = 7 )") == (7, 5)
    assert _parse_mouse_xy("RIGHT") is None
    assert _parse_mouse_xy(None) is None
    assert _parse_mouse_xy(42) is None


def test_action6_gets_coordinates_from_display():
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00", "00")),
        event(
            board=grid("10", "00"),
            action_name="ACTION6",
            action_display="MOUSE(row=1, col=0)",
        ),
    ]
    (segment,) = segments_from_events("g", events, stats)
    action = segment.transitions[0].action
    assert (action.name, action.x, action.y) == ("ACTION6", 0, 1)
    assert stats.action6_without_coords == 0


def test_action6_without_coordinates_is_counted_not_guessed():
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00", "00")),
        event(board=grid("10", "00"), action_name="ACTION6", action_display="MOUSE?"),
    ]
    segments = segments_from_events("g", events, stats)
    assert stats.action6_without_coords == 1
    assert segments == []


# --------------------------------------------------------------------
# extract: shape, levels, boundaries, resets
# --------------------------------------------------------------------


def test_board_is_wrapped_into_a_layered_grid():
    """`board` is a bare 2-D grid; the engine speaks [layer][row][col]."""
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("01", "23")),
        event(board=grid("01", "24")),
    ]
    (segment,) = segments_from_events("g", events, stats)
    t = segment.transitions[0]
    assert t.frame_before == [[[0, 1], [2, 3]]]
    assert t.frame_after == [[[0, 1], [2, 4]]]


def test_score_is_read_as_levels_completed():
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00"), score=2),
        event(board=grid("10"), score=2),
    ]
    (segment,) = segments_from_events("g", events, stats)
    t = segment.transitions[0]
    assert (t.levels_completed_before, t.levels_completed_after) == (2, 2)
    assert t.levels_delta == 0


def test_level_boundary_splits_segments_and_drops_the_boundary_step():
    """The board on a level-clearing event is the NEXT level's layout.

    Keeping that step would demand a per-level model predict a fresh
    board it has never seen, which nothing can do.
    """
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00", "00"), level=1, score=0),
        event(board=grid("10", "00"), level=1, score=0),
        event(board=grid("ff", "ff"), level=2, score=1, level_completed=True),
        event(board=grid("f0", "ff"), level=2, score=1),
    ]
    segments = segments_from_events("g", events, stats)

    assert [s.level for s in segments] == [1, 2]
    assert len(segments[0].transitions) == 1
    assert len(segments[1].transitions) == 1
    assert segments[0].cleared_level is True
    assert segments[1].cleared_level is False
    assert stats.transitions_dropped_boundary == 1
    # The level-2 segment must start from the level-2 board.
    assert segments[1].transitions[0].frame_before == [[[15, 15], [15, 15]]]


def test_reset_step_is_dropped_but_pairing_survives():
    """RESET is not a modelable action, but the post-reset board is still
    the next step's real `frame_before` -- dropping the step must not
    corrupt the following transition."""
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00")),
        event(board=grid("10")),
        event(board=grid("00"), action_name="RESET"),
        event(board=grid("20")),
    ]
    (segment,) = segments_from_events("g", events, stats)

    assert stats.transitions_dropped_reset == 1
    assert stats.transitions_dropped_malformed == 0
    assert len(segment.transitions) == 2
    # Second kept transition runs from the post-reset board, not from the
    # pre-reset one.
    assert segment.transitions[1].frame_before == [[[0, 0]]]
    assert segment.transitions[1].frame_after == [[[2, 0]]]


def test_game_over_overrides_state_and_marks_done():
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00")),
        event(board=grid("10"), game_over=True),
    ]
    (segment,) = segments_from_events("g", events, stats)
    assert segment.transitions[0].state_after == "GAME_OVER"
    assert segment.transitions[0].done is True


def test_analysis_events_are_ignored():
    """Only `initial`/`action` carry a board; deliberation events must not
    become phantom transitions."""
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00")),
        event(type_="analysis", board=grid("ff"), transcript="thinking..."),
        event(board=grid("10")),
    ]
    (segment,) = segments_from_events("g", events, stats)
    assert len(segment.transitions) == 1
    assert segment.transitions[0].frame_before == [[[0, 0]]]


# --------------------------------------------------------------------
# extract: robustness
# --------------------------------------------------------------------


def test_is_grid_rejects_ragged_and_non_int_boards():
    assert _is_grid([[0, 1], [2, 3]])
    assert not _is_grid([[0, 1], [2]])       # ragged
    assert not _is_grid([[0, 1], [2, "x"]])  # non-int
    assert not _is_grid([[0, True]])         # bool is not a colour
    assert not _is_grid([])
    assert not _is_grid("nope")


def test_events_without_a_board_are_counted_and_skipped():
    stats = ExtractionStats()
    events = [
        event(type_="initial", board=grid("00")),
        event(board=None),
        event(board=grid("10")),
    ]
    (segment,) = segments_from_events("g", events, stats)
    assert stats.events_without_board == 1
    assert len(segment.transitions) == 1


def test_truncated_final_line_does_not_lose_the_file(tmp_path):
    """A killed run leaves a half-written last line. This repo has already
    lost an extraction to exactly that (see CLAUDE.md's disk-full crash)."""
    path = tmp_path / "zz99-abc_p0_events.jsonl"
    good = [
        event(type_="initial", board=grid("00")),
        event(board=grid("10")),
    ]
    text = "\n".join(json.dumps(e) for e in good) + '\n{"type": "action", "bo'
    path.write_text(text, encoding="utf-8")

    segments, stats = extract_run(tmp_path)
    assert stats.lines_unparseable == 1
    assert len(segments) == 1
    assert len(segments[0].transitions) == 1


def test_extract_run_rejects_a_directory_with_no_event_logs(tmp_path):
    with pytest.raises(ExtractionError, match="no \\*_events.jsonl"):
        extract_run(tmp_path)


def test_game_id_from_path_strips_pass_suffix():
    assert game_id_from_path("ar25-0c556536_p0_events.jsonl") == "ar25-0c556536"
    assert game_id_from_path("bp35-0a0ad940_p11_events.jsonl") == "bp35-0a0ad940"


# --------------------------------------------------------------------
# windowing: shown steps must equal scored steps
# --------------------------------------------------------------------


def test_window_truncates_and_stops_claiming_the_level_was_cleared():
    segment = make_segment(
        [(grid("00"), "ACTION1", grid("10")),
         (grid("10"), "ACTION1", grid("20")),
         (grid("20"), "ACTION1", grid("30"))]
    )
    segment.cleared_level = True

    windowed = segment.window(2)
    assert len(windowed) == 2
    assert windowed.cleared_level is False, "a truncated segment no longer reaches the boundary"
    assert segment.cleared_level is True, "window() must not mutate the original"

    assert segment.window(99).cleared_level is True


def test_window_rejects_negative():
    with pytest.raises(ValueError):
        make_segment([(grid("00"), "ACTION1", grid("10"))]).window(-1)


def test_rendered_prompt_covers_exactly_the_scored_steps():
    """The invariant the whole design turns on: a model is never scored on
    a step it was not shown."""
    segment = make_segment(
        [(grid("00"), "ACTION1", grid("10")),
         (grid("10"), "ACTION2", grid("20")),
         (grid("20"), "ACTION3", grid("30"))]
    ).window(2)

    rendered = render_segment(segment)
    assert "step 0:" in rendered and "step 1:" in rendered
    assert "step 2:" not in rendered
    assert "ACTION3" not in rendered
    assert len(segment.transitions) == 2


def test_renderer_prints_the_opening_grid_once_not_per_step():
    """The engine's own renderer emits a full grid per transition, which
    is what makes its prompts ~4.6x larger on real data."""
    segment = make_segment(
        [(grid("00", "00"), "ACTION1", grid("10", "00")),
         (grid("10", "00"), "ACTION1", grid("20", "00")),
         (grid("20", "00"), "ACTION1", grid("30", "00"))]
    )
    rendered = render_segment(segment)
    assert rendered.count("Opening grid") == 1
    assert "Write the WorldModel now" in build_user_prompt(segment)


# --------------------------------------------------------------------
# determinism census
# --------------------------------------------------------------------


def test_census_finds_an_identical_board_with_different_outcomes():
    segment = make_segment(
        [(grid("00"), "ACTION1", grid("10")),
         (grid("10"), "ACTION1", grid("00")),
         (grid("00"), "ACTION1", grid("20"))]  # same board+action as step 0
    )
    found = determinism.census_segment(segment)
    assert len(found) == 1
    assert (found[0].first_index, found[0].second_index) == (0, 2)


def test_census_allows_a_repeat_with_the_same_outcome():
    segment = make_segment(
        [(grid("00"), "ACTION1", grid("10")),
         (grid("10"), "ACTION1", grid("00")),
         (grid("00"), "ACTION1", grid("10"))]
    )
    assert determinism.census_segment(segment) == []


def test_census_ceiling_and_filtering():
    clean = make_segment([(grid("00"), "ACTION1", grid("10"))], game_id="clean")
    dirty = make_segment(
        [(grid("00"), "ACTION1", grid("10")),
         (grid("10"), "ACTION1", grid("00")),
         (grid("00"), "ACTION1", grid("20"))],
        game_id="dirty",
    )
    result = determinism.census([clean, dirty])
    assert result.n_segments == 2
    assert result.n_contradictory == 1
    assert result.ceiling == pytest.approx(0.5)
    assert result.affected_games == {"dirty"}
    assert [s.game_id for s in determinism.deterministic_only([clean, dirty])] == ["clean"]


# --------------------------------------------------------------------
# oracle: the positive control
# --------------------------------------------------------------------


def test_oracle_replays_a_clean_segment():
    segment = make_segment(
        [(grid("00", "00"), "ACTION1", grid("10", "00")),
         (grid("10", "00"), "ACTION2", grid("10", "01")),
         (grid("10", "01"), "ACTION6", grid("12", "01"))]
    )
    passed, error = oracle.verify_oracle(segment)
    assert passed, error


def test_oracle_reports_failure_on_a_true_contradiction():
    """No lookup table can pass a segment that contradicts itself -- and
    the message must say so, not blame the hash."""
    segment = make_segment(
        [(grid("00"), "ACTION1", grid("10")),
         (grid("10"), "ACTION1", grid("00")),
         (grid("00"), "ACTION1", grid("20"))]
    )
    passed, error = oracle.verify_oracle(segment)
    assert not passed
    assert "identical board and action" in error


def test_oracle_exact_mode_keys_on_the_whole_board():
    segment = make_segment(
        [(grid("00", "00"), "ACTION1", grid("10", "00")),
         (grid("10", "00"), "ACTION1", grid("11", "00"))]
    )
    source = oracle.build_oracle_source(segment, exact=True)
    assert "tuple(tuple(row) for row in state[0])" in source


# --------------------------------------------------------------------
# harness
# --------------------------------------------------------------------

PERFECT_MODEL = """
```python
class WorldModel:
    def __init__(self):
        pass

    def predict(self, state, action_name, x=None, y=None):
        layer = [list(row) for row in state[0]]
        layer[0][0] = layer[0][0] + 1
        return [layer], 0, False

    def goal_hint(self, state):
        return 0.0
```
"""

WRONG_MODEL = """
```python
class WorldModel:
    def __init__(self):
        pass

    def predict(self, state, action_name, x=None, y=None):
        return state, 0, False

    def goal_hint(self, state):
        return 0.0
```
"""

UNLOADABLE_MODEL = "```python\nclass WorldModel:\n    def __init__(self):\n        this is not python\n```"


class ScriptedClient:
    """Returns queued responses in order; records the prompts it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, user, max_tokens=1024):
        self.prompts.append(user)
        if not self.responses:
            return ""
        return self.responses.pop(0)


class ExplodingClient:
    def complete(self, system, user, max_tokens=1024):
        raise ConnectionError("server unreachable")


def incrementing_segment(steps=3):
    """A segment whose only rule is 'top-left cell increases by one'."""
    pairs = []
    before = grid("00", "00")
    for i in range(steps):
        after = [list(r) for r in before]
        after[0][0] += 1
        pairs.append((before, "ACTION1", after))
        before = after
    return make_segment(pairs)


def test_harness_reports_a_pass_for_a_correct_model():
    segment = incrementing_segment()
    result = run_segment(ScriptedClient([PERFECT_MODEL]), segment)
    assert result.passed
    assert result.outcome == PASSED
    assert result.attempts == 1
    assert result.best_prefix == len(segment)
    assert result.source is not None


def test_harness_reports_grid_mismatch_for_a_do_nothing_model():
    result = run_segment(ScriptedClient([WRONG_MODEL] * 3), incrementing_segment())
    assert not result.passed
    assert result.outcome == GRID_MISMATCH
    assert result.attempts == 3
    assert result.best_prefix == 0


def test_harness_retries_and_succeeds_on_a_later_attempt():
    result = run_segment(ScriptedClient([WRONG_MODEL, PERFECT_MODEL]), incrementing_segment())
    assert result.passed
    assert result.attempts == 2


def test_harness_classifies_unloadable_source():
    result = run_segment(
        ScriptedClient([UNLOADABLE_MODEL]), incrementing_segment(), BacktestConfig(max_attempts=1)
    )
    assert result.outcome == LOAD_ERROR
    assert not result.passed


def test_harness_classifies_a_response_with_no_code():
    result = run_segment(
        ScriptedClient(["I think the rule is that things move right."]),
        incrementing_segment(),
        BacktestConfig(max_attempts=1),
    )
    assert result.outcome == NO_CODE


def test_harness_survives_an_unreachable_client():
    """A dead server is a data point, not a crash -- one game thread must
    never take down a run."""
    result = run_segment(ExplodingClient(), incrementing_segment())
    assert result.outcome == NO_RESPONSE
    assert not result.passed


def test_repair_prompt_reincludes_the_transcript():
    """A retry that drops the transcript asks the model to infer a rule
    for data it can no longer see -- the engine's own drafting loop
    documents having made exactly this mistake."""
    client = ScriptedClient([WRONG_MODEL, PERFECT_MODEL])
    run_segment(client, incrementing_segment())
    assert len(client.prompts) == 2
    assert "Opening grid" in client.prompts[1]
    assert "did not work" in client.prompts[1]


def test_identity_baseline_is_measured_and_is_zero_on_a_changing_segment():
    segment = incrementing_segment()
    prefix, passed = identity_baseline(segment)
    assert (prefix, passed) == (0, False)

    result = run_segment(ScriptedClient([PERFECT_MODEL]), segment)
    assert result.identity_passed is False
    assert result.beats_identity is True


def test_identity_baseline_passes_a_segment_where_nothing_ever_changes():
    """Such a segment teaches nothing, and the report must say so rather
    than bank it as a win."""
    still = grid("00", "00")
    segment = make_segment([(still, "ACTION1", still), (still, "ACTION2", still)])
    prefix, passed = identity_baseline(segment)
    assert passed
    assert prefix == 2

    report = build_report([run_segment(ScriptedClient([WRONG_MODEL]), segment)])
    assert report.n_identity_passed == 1
    assert report.informative == []


# --------------------------------------------------------------------
# report
# --------------------------------------------------------------------


def test_report_separates_informative_segments_from_free_passes():
    still = grid("00")
    trivial = make_segment([(still, "ACTION1", still)], game_id="trivial")
    real = incrementing_segment()

    results = [
        run_segment(ScriptedClient([WRONG_MODEL]), trivial),
        run_segment(ScriptedClient([PERFECT_MODEL]), real),
    ]
    report = build_report(results)

    assert report.n_segments == 2
    assert report.n_identity_passed == 1
    assert len(report.informative) == 1
    assert report.informative_pass_rate == pytest.approx(1.0)
    assert report.n_games == 2
    assert "CodeWorldModel backtest" in report.summary()


def test_report_is_json_serialisable():
    report = build_report([run_segment(ScriptedClient([PERFECT_MODEL]), incrementing_segment())])
    json.dumps(report.as_dict())
