from dataclasses import dataclass
from typing import Iterable

from nexus.core.decision.trace import DecisionTrace


@dataclass(frozen=True)
class DecisionQualityMetrics:
    """
    Descriptive metrics for evaluating the quality of NEXUS decisions.

    These metrics describe the decision landscape preserved by
    DecisionTrace. They do not represent statistical probabilities,
    calibrated confidence, or predictive accuracy.
    """

    total_decisions: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    average_score_margin: float
    minimum_score_margin: float
    maximum_score_margin: float
    ambiguous_decision_rate: float
    average_candidate_count: float

    @property
    def high_confidence_rate(self) -> float:
        """Return the proportion of decisions labelled HIGH."""
        if self.total_decisions == 0:
            return 0.0

        return self.high_confidence_count / self.total_decisions

    @property
    def medium_confidence_rate(self) -> float:
        """Return the proportion of decisions labelled MEDIUM."""
        if self.total_decisions == 0:
            return 0.0

        return self.medium_confidence_count / self.total_decisions

    @property
    def low_confidence_rate(self) -> float:
        """Return the proportion of decisions labelled LOW."""
        if self.total_decisions == 0:
            return 0.0

        return self.low_confidence_count / self.total_decisions

    @classmethod
    def from_traces(
        cls,
        traces: Iterable[DecisionTrace],
    ) -> "DecisionQualityMetrics":
        """
        Calculate descriptive decision-quality metrics from traces.

        No trace is treated as a probability distribution. Confidence
        labels are interpreted only as decision-separation categories.
        """

        trace_list = list(traces)

        if not trace_list:
            return cls(
                total_decisions=0,
                high_confidence_count=0,
                medium_confidence_count=0,
                low_confidence_count=0,
                average_score_margin=0.0,
                minimum_score_margin=0.0,
                maximum_score_margin=0.0,
                ambiguous_decision_rate=0.0,
                average_candidate_count=0.0,
            )

        high_count = sum(
            trace.confidence == "HIGH"
            for trace in trace_list
        )

        medium_count = sum(
            trace.confidence == "MEDIUM"
            for trace in trace_list
        )

        low_count = sum(
            trace.confidence == "LOW"
            for trace in trace_list
        )

        margins = [trace.score_margin for trace in trace_list]

        candidate_counts = [
            trace.candidate_count
            for trace in trace_list
        ]

        total = len(trace_list)

        return cls(
            total_decisions=total,
            high_confidence_count=high_count,
            medium_confidence_count=medium_count,
            low_confidence_count=low_count,
            average_score_margin=sum(margins) / total,
            minimum_score_margin=min(margins),
            maximum_score_margin=max(margins),
            ambiguous_decision_rate=low_count / total,
            average_candidate_count=(
                sum(candidate_counts) / total
            ),
        )


def decision_quality_from_records(records) -> DecisionQualityMetrics:
    """
    Calculate decision-quality metrics from NEXUS decision records.

    Records without a DecisionTrace are ignored. This preserves
    compatibility with older records created before decision tracing
    was introduced.
    """

    traces = [
        record.decision_trace
        for record in records
        if record.decision_trace is not None
    ]

    return DecisionQualityMetrics.from_traces(traces)
