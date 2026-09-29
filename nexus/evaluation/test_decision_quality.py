from nexus.core.decision.trace import CandidateScore, DecisionTrace
from nexus.evaluation.decision_quality import DecisionQualityMetrics


def make_trace(
    selected: str,
    selected_score: float,
    runner_up: str,
    runner_up_score: float,
) -> DecisionTrace:
    return DecisionTrace.from_scored_candidates(
        [
            CandidateScore(
                name=selected,
                score=selected_score,
                action=[selected],
            ),
            CandidateScore(
                name=runner_up,
                score=runner_up_score,
                action=[runner_up],
            ),
        ]
    )


def test_empty_traces_return_zero_metrics():
    metrics = DecisionQualityMetrics.from_traces([])

    assert metrics.total_decisions == 0
    assert metrics.high_confidence_count == 0
    assert metrics.medium_confidence_count == 0
    assert metrics.low_confidence_count == 0
    assert metrics.average_score_margin == 0.0
    assert metrics.minimum_score_margin == 0.0
    assert metrics.maximum_score_margin == 0.0
    assert metrics.ambiguous_decision_rate == 0.0
    assert metrics.average_candidate_count == 0.0


def test_metrics_count_confidence_categories():
    traces = [
        make_trace("A", 10.0, "B", 4.0),
        make_trace("A", 10.0, "B", 7.0),
        make_trace("A", 10.0, "B", 9.0),
    ]

    metrics = DecisionQualityMetrics.from_traces(traces)

    assert metrics.total_decisions == 3
    assert metrics.high_confidence_count == 1
    assert metrics.medium_confidence_count == 1
    assert metrics.low_confidence_count == 1


def test_metrics_calculate_score_margin_statistics():
    traces = [
        make_trace("A", 10.0, "B", 4.0),
        make_trace("A", 10.0, "B", 7.0),
        make_trace("A", 10.0, "B", 9.0),
    ]

    metrics = DecisionQualityMetrics.from_traces(traces)

    assert metrics.average_score_margin == 10.0 / 3.0
    assert metrics.minimum_score_margin == 1.0
    assert metrics.maximum_score_margin == 6.0


def test_metrics_calculate_confidence_rates():
    traces = [
        make_trace("A", 10.0, "B", 4.0),
        make_trace("A", 10.0, "B", 7.0),
        make_trace("A", 10.0, "B", 9.0),
        make_trace("A", 10.0, "B", 8.0),
    ]

    metrics = DecisionQualityMetrics.from_traces(traces)

    assert metrics.high_confidence_rate == 0.25
    assert metrics.medium_confidence_rate == 0.50
    assert metrics.low_confidence_rate == 0.25
    assert metrics.ambiguous_decision_rate == 0.25


def test_metrics_calculate_average_candidate_count():
    traces = [
        make_trace("A", 10.0, "B", 4.0),
        make_trace("A", 10.0, "B", 7.0),
    ]

    metrics = DecisionQualityMetrics.from_traces(traces)

    assert metrics.average_candidate_count == 2.0


def test_metrics_accept_generator_input():
    traces = (
        make_trace("A", 10.0, "B", 4.0)
        for _ in range(3)
    )

    metrics = DecisionQualityMetrics.from_traces(traces)

    assert metrics.total_decisions == 3


def test_decision_quality_from_records_uses_traces():
    from dataclasses import dataclass

    from nexus.evaluation.decision_quality import decision_quality_from_records

    @dataclass
    class Record:
        decision_trace: DecisionTrace | None

    records = [
        Record(make_trace("A", 10.0, "B", 4.0)),
        Record(make_trace("A", 10.0, "B", 7.0)),
        Record(make_trace("A", 10.0, "B", 9.0)),
    ]

    metrics = decision_quality_from_records(records)

    assert metrics.total_decisions == 3
    assert metrics.high_confidence_count == 1
    assert metrics.medium_confidence_count == 1
    assert metrics.low_confidence_count == 1


def test_decision_quality_from_records_ignores_records_without_trace():
    from dataclasses import dataclass

    from nexus.evaluation.decision_quality import decision_quality_from_records

    @dataclass
    class Record:
        decision_trace: DecisionTrace | None

    records = [
        Record(make_trace("A", 10.0, "B", 4.0)),
        Record(None),
        Record(make_trace("A", 10.0, "B", 9.0)),
    ]

    metrics = decision_quality_from_records(records)

    assert metrics.total_decisions == 2
    assert metrics.high_confidence_count == 1
    assert metrics.low_confidence_count == 1


def test_real_nexus_episode_produces_decision_quality_metrics():
    from nexus.agents.nexus.agent import NexusAgent
    from nexus.agents.nexus.test_agent import make_observation

    agent = NexusAgent()

    observation = make_observation()

    for _ in range(5):
        agent.act(observation)

    metrics = DecisionQualityMetrics.from_traces(
        record.decision_trace
        for record in agent.memory.all()
        if record.decision_trace is not None
    )

    assert metrics.total_decisions == 5
    assert metrics.average_candidate_count > 0
    assert metrics.average_score_margin >= 0
    assert 0.0 <= metrics.ambiguous_decision_rate <= 1.0
