from nexus.core.memory.experience import ExperienceRecord
from nexus.evaluation.strategy_analysis import (
    StrategyAnalyzer,
    StrategyAnalysisReport,
    StrategyDecisionSummary,
    analyze_strategy,
)


def make_experience(
    step: int,
    decision_name: str,
    effect_match: str = "MATCHED",
    confidence: str = "MEDIUM",
    score: float = 10.0,
    margin: float = 2.0,
    alternatives: int = 1,
) -> ExperienceRecord:
    """Create a small deterministic experience for testing."""

    return ExperienceRecord(
        step=step,
        day=1,
        hour=step,
        observation_summary={
            "money": 3000.0,
            "crop_count": 1,
            "weed_count": 0,
        },
        decision_name=decision_name,
        decision_action=[decision_name],
        decision_score=score,
        confidence=confidence,
        score_margin=margin,
        alternative_count=alternatives,
        expected_effect="test expected effect",
        observed_effect="test observed effect",
        agent_effect="agent effect",
        environment_effect="environment effect",
        effect_match=effect_match,
    )


def test_empty_experience_collection_returns_empty_report():
    analyzer = StrategyAnalyzer()

    report = analyzer.analyze([])

    assert isinstance(report, StrategyAnalysisReport)
    assert report.total_experiences == 0
    assert report.decision_type_count == 0
    assert report.overall_effect_match_rate == 0.0
    assert report.average_score_margin == 0.0
    assert report.average_alternative_count == 0.0
    assert report.ambiguous_decision_count == 0


def test_analyzer_counts_experiences_and_decision_types():
    experiences = [
        make_experience(1, "BUY_WHEAT_SEED"),
        make_experience(2, "BUY_WHEAT_SEED"),
        make_experience(3, "WATER_CROP"),
    ]

    report = StrategyAnalyzer().analyze(experiences)

    assert report.total_experiences == 3
    assert report.decision_type_count == 2

    assert report.decision_counts == {
        "BUY_WHEAT_SEED": 2,
        "WATER_CROP": 1,
    }


def test_effect_match_rate_is_calculated():
    experiences = [
        make_experience(1, "WATER_CROP", "MATCHED"),
        make_experience(2, "WATER_CROP", "MATCHED"),
        make_experience(3, "WATER_CROP", "MISMATCH"),
        make_experience(4, "WATER_CROP", "UNKNOWN"),
    ]

    report = StrategyAnalyzer().analyze(experiences)

    assert report.matched_count == 2
    assert report.mismatch_count == 1
    assert report.unknown_count == 1
    assert report.partial_count == 0

    assert report.overall_effect_match_rate == 0.5


def test_decision_summary_is_created_for_each_decision():
    experiences = [
        make_experience(1, "BUY_WHEAT_SEED"),
        make_experience(2, "BUY_WHEAT_SEED"),
        make_experience(3, "WATER_CROP"),
    ]

    report = StrategyAnalyzer().analyze(experiences)

    assert len(report.decisions) == 2

    assert all(
        isinstance(
            summary,
            StrategyDecisionSummary,
        )
        for summary in report.decisions
    )

    names = [
        summary.decision_name
        for summary in report.decisions
    ]

    assert names == [
        "BUY_WHEAT_SEED",
        "WATER_CROP",
    ]


def test_decision_summary_contains_correct_statistics():
    experiences = [
        make_experience(
            1,
            "BUY_WHEAT_SEED",
            confidence="HIGH",
            score=10.0,
            margin=5.0,
            alternatives=2,
        ),
        make_experience(
            2,
            "BUY_WHEAT_SEED",
            confidence="LOW",
            score=6.0,
            margin=1.0,
            alternatives=0,
        ),
    ]

    report = StrategyAnalyzer().analyze(experiences)

    summary = report.decisions[0]

    assert summary.decision_name == "BUY_WHEAT_SEED"
    assert summary.experience_count == 2

    assert summary.matched_count == 2
    assert summary.effect_match_rate == 1.0

    assert summary.average_score == 8.0
    assert summary.average_score_margin == 3.0
    assert summary.average_alternative_count == 1.0

    assert summary.high_confidence_count == 1
    assert summary.medium_confidence_count == 0
    assert summary.low_confidence_count == 1


def test_ambiguous_rate_is_based_on_low_confidence():
    experiences = [
        make_experience(
            1,
            "PASS",
            confidence="LOW",
        ),
        make_experience(
            2,
            "PASS",
            confidence="LOW",
        ),
        make_experience(
            3,
            "PASS",
            confidence="HIGH",
        ),
        make_experience(
            4,
            "PASS",
            confidence="MEDIUM",
        ),
    ]

    report = StrategyAnalyzer().analyze(experiences)

    summary = report.decisions[0]

    assert summary.low_confidence_count == 2
    assert summary.ambiguous_rate == 0.5

    assert report.ambiguous_decision_count == 2


def test_partial_effects_are_preserved():
    experiences = [
        make_experience(
            1,
            "CLEAR_WEED",
            "PARTIAL",
        ),
        make_experience(
            2,
            "CLEAR_WEED",
            "MATCHED",
        ),
    ]

    report = StrategyAnalyzer().analyze(experiences)

    assert report.partial_count == 1
    assert report.matched_count == 1

    summary = report.decisions[0]

    assert summary.partial_count == 1
    assert summary.matched_count == 1
    assert summary.effect_match_rate == 0.5


def test_report_serialization_is_available():
    experiences = [
        make_experience(
            1,
            "BUY_WHEAT_SEED",
        ),
    ]

    report = StrategyAnalyzer().analyze(experiences)

    data = report.as_dict()

    assert data["total_experiences"] == 1
    assert data["decision_type_count"] == 1

    assert data["decision_counts"] == {
        "BUY_WHEAT_SEED": 1,
    }

    assert len(data["decisions"]) == 1
    assert (
        data["decisions"][0]["decision_name"]
        == "BUY_WHEAT_SEED"
    )


def test_convenience_function_matches_analyzer():
    experiences = [
        make_experience(
            1,
            "BUY_WHEAT_SEED",
        ),
        make_experience(
            2,
            "WATER_CROP",
        ),
    ]

    report = analyze_strategy(experiences)

    assert isinstance(
        report,
        StrategyAnalysisReport,
    )

    assert report.total_experiences == 2
    assert report.decision_type_count == 2
