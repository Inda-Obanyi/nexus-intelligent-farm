import pytest

from nexus.core.decision.trace import CandidateScore, DecisionTrace


def make_candidate(name, score, action=None):
    return CandidateScore(
        name=name,
        score=score,
        action=action or ["PASS"],
    )


def test_trace_identifies_selected_decision():
    candidates = [
        make_candidate("WATER_CROP", 12.0, ["WATER"]),
        make_candidate("HARVEST_CROP", 9.0, ["HARVEST"]),
        make_candidate("PASS", 0.0),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.selected_decision == "WATER_CROP"
    assert trace.selected_score == 12.0


def test_trace_identifies_runner_up():
    candidates = [
        make_candidate("WATER_CROP", 12.0, ["WATER"]),
        make_candidate("HARVEST_CROP", 9.0, ["HARVEST"]),
        make_candidate("PASS", 0.0),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.runner_up == "HARVEST_CROP"
    assert trace.runner_up_score == 9.0


def test_trace_calculates_score_margin():
    candidates = [
        make_candidate("WATER_CROP", 12.0, ["WATER"]),
        make_candidate("HARVEST_CROP", 9.0, ["HARVEST"]),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.score_margin == 3.0


def test_trace_counts_candidates():
    candidates = [
        make_candidate("WATER_CROP", 12.0, ["WATER"]),
        make_candidate("HARVEST_CROP", 9.0, ["HARVEST"]),
        make_candidate("PLANT_WHEAT", 7.0, ["PLANT", "WHEAT"]),
        make_candidate("PASS", 0.0),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.candidate_count == 4


def test_large_margin_is_high_confidence():
    candidates = [
        make_candidate("DIG_WEED", 16.0, ["DIG"]),
        make_candidate("WATER_CROP", 9.0, ["WATER"]),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.confidence == "HIGH"
    assert trace.is_ambiguous is False


def test_medium_margin_is_medium_confidence():
    candidates = [
        make_candidate("WATER_CROP", 12.0, ["WATER"]),
        make_candidate("HARVEST_CROP", 9.0, ["HARVEST"]),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.confidence == "MEDIUM"
    assert trace.is_ambiguous is False


def test_small_margin_is_low_confidence():
    candidates = [
        make_candidate("WATER_CROP", 10.5, ["WATER"]),
        make_candidate("HARVEST_CROP", 9.5, ["HARVEST"]),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.confidence == "LOW"
    assert trace.is_ambiguous is True


def test_single_candidate_has_no_runner_up():
    candidates = [
        make_candidate("PASS", 0.0),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert trace.selected_decision == "PASS"
    assert trace.runner_up is None
    assert trace.runner_up_score is None
    assert trace.score_margin == 0.0
    assert trace.candidate_count == 1
    assert trace.confidence == "LOW"


def test_top_candidates_are_sorted_by_score():
    candidates = [
        make_candidate("PASS", 0.0),
        make_candidate("PLANT_WHEAT", 7.0, ["PLANT", "WHEAT"]),
        make_candidate("WATER_CROP", 12.0, ["WATER"]),
        make_candidate("HARVEST_CROP", 9.0, ["HARVEST"]),
    ]

    trace = DecisionTrace.from_scored_candidates(candidates)

    assert [candidate.name for candidate in trace.top_candidates] == [
        "WATER_CROP",
        "HARVEST_CROP",
        "PLANT_WHEAT",
        "PASS",
    ]


def test_empty_candidate_list_is_rejected():
    with pytest.raises(ValueError, match="At least one candidate"):
        DecisionTrace.from_scored_candidates([])


def test_negative_margin_is_rejected():
    with pytest.raises(ValueError, match="Score margin"):
        DecisionTrace(
            selected_decision="WATER_CROP",
            selected_score=10.0,
            score_margin=-1.0,
        )


def test_negative_candidate_count_is_rejected():
    with pytest.raises(ValueError, match="Candidate count"):
        DecisionTrace(
            selected_decision="WATER_CROP",
            selected_score=10.0,
            candidate_count=-1,
        )


def test_runner_up_requires_runner_up_score():
    with pytest.raises(
        ValueError,
        match="corresponding runner_up_score",
    ):
        DecisionTrace(
            selected_decision="WATER_CROP",
            selected_score=10.0,
            runner_up="HARVEST_CROP",
        )


def test_runner_up_score_requires_runner_up():
    with pytest.raises(
        ValueError,
        match="without runner_up",
    ):
        DecisionTrace(
            selected_decision="WATER_CROP",
            selected_score=10.0,
            runner_up_score=8.0,
        )
