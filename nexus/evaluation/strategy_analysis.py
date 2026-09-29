from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from nexus.core.memory.experience import ExperienceRecord


@dataclass(frozen=True)
class StrategyDecisionSummary:
    """
    Descriptive analysis of one NEXUS decision type.

    These metrics describe observed experience records.
    They do not establish that a decision is optimal.
    """

    decision_name: str
    experience_count: int

    matched_count: int
    partial_count: int
    mismatch_count: int
    unknown_count: int

    effect_match_rate: float

    average_score: float
    average_score_margin: float
    average_alternative_count: float

    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int

    @property
    def ambiguous_rate(self) -> float:
        """
        Return the proportion of LOW-confidence decisions.

        LOW confidence means weak score separation between
        the selected decision and alternatives.

        It is not a probability.
        """
        if self.experience_count == 0:
            return 0.0

        return self.low_confidence_count / self.experience_count

    def as_dict(self) -> Dict[str, object]:
        """Return a serializable representation."""
        return {
            "decision_name": self.decision_name,
            "experience_count": self.experience_count,
            "matched_count": self.matched_count,
            "partial_count": self.partial_count,
            "mismatch_count": self.mismatch_count,
            "unknown_count": self.unknown_count,
            "effect_match_rate": self.effect_match_rate,
            "average_score": self.average_score,
            "average_score_margin": self.average_score_margin,
            "average_alternative_count": (
                self.average_alternative_count
            ),
            "high_confidence_count": (
                self.high_confidence_count
            ),
            "medium_confidence_count": (
                self.medium_confidence_count
            ),
            "low_confidence_count": (
                self.low_confidence_count
            ),
            "ambiguous_rate": self.ambiguous_rate,
        }


@dataclass(frozen=True)
class StrategyAnalysisReport:
    """
    Descriptive analysis of a collection of NEXUS experiences.

    This report identifies observed patterns in the experience
    dataset. It does not claim that any strategy is globally
    optimal or causally superior.
    """

    total_experiences: int

    matched_count: int
    partial_count: int
    mismatch_count: int
    unknown_count: int

    overall_effect_match_rate: float

    average_score_margin: float
    average_alternative_count: float

    decision_counts: Dict[str, int] = field(
        default_factory=dict
    )

    decisions: List[StrategyDecisionSummary] = field(
        default_factory=list
    )

    @property
    def ambiguous_decision_count(self) -> int:
        """Return the number of LOW-confidence experiences."""
        return sum(
            summary.low_confidence_count
            for summary in self.decisions
        )

    @property
    def decision_type_count(self) -> int:
        """Return the number of distinct decision types."""
        return len(self.decision_counts)

    def as_dict(self) -> Dict[str, object]:
        """Return a serializable representation."""
        return {
            "total_experiences": self.total_experiences,
            "matched_count": self.matched_count,
            "partial_count": self.partial_count,
            "mismatch_count": self.mismatch_count,
            "unknown_count": self.unknown_count,
            "overall_effect_match_rate": (
                self.overall_effect_match_rate
            ),
            "average_score_margin": (
                self.average_score_margin
            ),
            "average_alternative_count": (
                self.average_alternative_count
            ),
            "ambiguous_decision_count": (
                self.ambiguous_decision_count
            ),
            "decision_type_count": (
                self.decision_type_count
            ),
            "decision_counts": dict(
                self.decision_counts
            ),
            "decisions": [
                summary.as_dict()
                for summary in self.decisions
            ],
        }


class StrategyAnalyzer:
    """
    Analyze accumulated NEXUS experiences.

    The analyzer is intentionally descriptive.

    It does not:
        - train a machine-learning model,
        - assign rewards,
        - claim optimality,
        - modify the decision engine,
        - make future decisions.

    Its purpose is to discover repeated patterns that can later
    support scientifically grounded policy improvement.
    """

    def analyze(
        self,
        experiences: Iterable[ExperienceRecord],
    ) -> StrategyAnalysisReport:
        """
        Analyze a collection of experience records.
        """

        experience_list = list(experiences)

        if not experience_list:
            return StrategyAnalysisReport(
                total_experiences=0,
                matched_count=0,
                partial_count=0,
                mismatch_count=0,
                unknown_count=0,
                overall_effect_match_rate=0.0,
                average_score_margin=0.0,
                average_alternative_count=0.0,
                decision_counts={},
                decisions=[],
            )

        matched_count = sum(
            experience.effect_match == "MATCHED"
            for experience in experience_list
        )

        partial_count = sum(
            experience.effect_match == "PARTIAL"
            for experience in experience_list
        )

        mismatch_count = sum(
            experience.effect_match == "MISMATCH"
            for experience in experience_list
        )

        unknown_count = sum(
            experience.effect_match == "UNKNOWN"
            for experience in experience_list
        )

        total = len(experience_list)

        decision_counts: Dict[str, int] = {}

        for experience in experience_list:
            decision_counts[
                experience.decision_name
            ] = (
                decision_counts.get(
                    experience.decision_name,
                    0,
                )
                + 1
            )

        average_score_margin = (
            sum(
                experience.score_margin
                for experience in experience_list
            )
            / total
        )

        average_alternative_count = (
            sum(
                experience.alternative_count
                for experience in experience_list
            )
            / total
        )

        summaries = self._build_decision_summaries(
            experience_list
        )

        return StrategyAnalysisReport(
            total_experiences=total,
            matched_count=matched_count,
            partial_count=partial_count,
            mismatch_count=mismatch_count,
            unknown_count=unknown_count,
            overall_effect_match_rate=(
                matched_count / total
            ),
            average_score_margin=average_score_margin,
            average_alternative_count=(
                average_alternative_count
            ),
            decision_counts=decision_counts,
            decisions=summaries,
        )

    def _build_decision_summaries(
        self,
        experiences: List[ExperienceRecord],
    ) -> List[StrategyDecisionSummary]:
        """
        Build descriptive statistics for each decision type.
        """

        grouped: Dict[
            str,
            List[ExperienceRecord]
        ] = {}

        for experience in experiences:
            grouped.setdefault(
                experience.decision_name,
                [],
            ).append(experience)

        summaries = []

        for decision_name in sorted(grouped):
            records = grouped[decision_name]
            count = len(records)

            matched = sum(
                record.effect_match == "MATCHED"
                for record in records
            )

            partial = sum(
                record.effect_match == "PARTIAL"
                for record in records
            )

            mismatch = sum(
                record.effect_match == "MISMATCH"
                for record in records
            )

            unknown = sum(
                record.effect_match == "UNKNOWN"
                for record in records
            )

            high_confidence = sum(
                record.confidence == "HIGH"
                for record in records
            )

            medium_confidence = sum(
                record.confidence == "MEDIUM"
                for record in records
            )

            low_confidence = sum(
                record.confidence == "LOW"
                for record in records
            )

            summaries.append(
                StrategyDecisionSummary(
                    decision_name=decision_name,
                    experience_count=count,
                    matched_count=matched,
                    partial_count=partial,
                    mismatch_count=mismatch,
                    unknown_count=unknown,
                    effect_match_rate=(
                        matched / count
                    ),
                    average_score=(
                        sum(
                            record.decision_score
                            for record in records
                        )
                        / count
                    ),
                    average_score_margin=(
                        sum(
                            record.score_margin
                            for record in records
                        )
                        / count
                    ),
                    average_alternative_count=(
                        sum(
                            record.alternative_count
                            for record in records
                        )
                        / count
                    ),
                    high_confidence_count=(
                        high_confidence
                    ),
                    medium_confidence_count=(
                        medium_confidence
                    ),
                    low_confidence_count=(
                        low_confidence
                    ),
                )
            )

        return summaries


def analyze_strategy(
    experiences: Iterable[ExperienceRecord],
) -> StrategyAnalysisReport:
    """
    Convenience function for analyzing NEXUS experiences.
    """

    return StrategyAnalyzer().analyze(
        experiences
    )
