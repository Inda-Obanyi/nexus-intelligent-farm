from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.core.decision.trace import DecisionTrace
from nexus.core.memory.decision import DecisionRecord
from nexus.evaluation.outcomes import OutcomeEvaluation


@dataclass(frozen=True)
class DecisionAlternative:
    """
    A candidate decision considered by NEXUS but not selected.

    This preserves the decision landscape so that NEXUS can explain
    not only what it chose, but what it chose against.
    """

    name: str
    score: float
    action: List[str]
    score_difference: float


@dataclass(frozen=True)
class DecisionIntelligenceReport:
    """
    Structured explanation of a single NEXUS decision.

    This is an interpretable decision-intelligence artifact.

    It does not claim that the selected decision was globally optimal.
    It explains the evidence available to NEXUS at decision time and
    compares the expected effect with the observed outcome.
    """

    step: int
    day: int
    hour: int

    decision_name: str
    decision_action: List[str]
    executed_action: Optional[Dict[str, Any]]

    decision_score: float
    decision_reason: str

    confidence: str
    score_margin: float
    runner_up: Optional[str]
    runner_up_score: Optional[float]

    alternatives: List[DecisionAlternative] = field(
        default_factory=list
    )

    expected_effect: str = ""
    observed_effect: str = ""

    agent_effect: str = ""
    environment_effect: str = ""

    effect_match: str = ""

    evidence: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_ambiguous(self) -> bool:
        """
        Return whether the decision was classified as ambiguous.

        LOW confidence means the selected decision had little score
        separation from its runner-up. This is not a probability.
        """
        return self.confidence == "LOW"

    @property
    def alternative_count(self) -> int:
        """Return the number of non-selected alternatives."""
        return len(self.alternatives)

    def as_dict(self) -> Dict[str, Any]:
        """Return the report as a serializable dictionary."""
        return {
            "step": self.step,
            "day": self.day,
            "hour": self.hour,
            "decision_name": self.decision_name,
            "decision_action": list(self.decision_action),
            "executed_action": (
                dict(self.executed_action)
                if self.executed_action is not None
                else None
            ),
            "decision_score": self.decision_score,
            "decision_reason": self.decision_reason,
            "confidence": self.confidence,
            "score_margin": self.score_margin,
            "runner_up": self.runner_up,
            "runner_up_score": self.runner_up_score,
            "alternatives": [
                {
                    "name": alternative.name,
                    "score": alternative.score,
                    "action": list(alternative.action),
                    "score_difference": alternative.score_difference,
                }
                for alternative in self.alternatives
            ],
            "expected_effect": self.expected_effect,
            "observed_effect": self.observed_effect,
            "agent_effect": self.agent_effect,
            "environment_effect": self.environment_effect,
            "effect_match": self.effect_match,
            "evidence": dict(self.evidence),
        }


class DecisionIntelligenceBuilder:
    """
    Build an interpretable intelligence report from NEXUS evidence.

    Input:
        DecisionRecord
            +
        DecisionTrace
            +
        OutcomeEvaluation

    Output:
        DecisionIntelligenceReport

    The builder does not make a new decision and does not alter the
    underlying decision engine. It only interprets evidence already
    produced by NEXUS.
    """

    def build(
        self,
        record: DecisionRecord,
        outcome: Optional[OutcomeEvaluation] = None,
    ) -> DecisionIntelligenceReport:
        """
        Build a decision-intelligence report.

        A DecisionRecord without a DecisionTrace is supported for
        backward compatibility. In that case, confidence and candidate
        landscape information are unavailable.
        """

        trace = record.decision_trace

        if trace is None:
            confidence = "UNKNOWN"
            score_margin = 0.0
            runner_up = None
            runner_up_score = None
            alternatives: List[DecisionAlternative] = []
        else:
            confidence = trace.confidence
            score_margin = trace.score_margin
            runner_up = trace.runner_up
            runner_up_score = trace.runner_up_score
            alternatives = self._build_alternatives(trace)

        expected_effect = ""
        observed_effect = ""
        agent_effect = ""
        environment_effect = ""
        effect_match = ""
        evidence: Dict[str, Any] = {}

        if outcome is not None:
            expected_effect = outcome.expected_effect
            observed_effect = outcome.observed_effect
            agent_effect = outcome.agent_effect
            environment_effect = outcome.environment_effect
            effect_match = outcome.effect_match
            evidence = dict(outcome.evidence)

        return DecisionIntelligenceReport(
            step=record.step,
            day=record.day,
            hour=record.hour,
            decision_name=record.decision_name,
            decision_action=list(record.decision_action),
            executed_action=(
                dict(record.executed_action)
                if record.executed_action is not None
                else None
            ),
            decision_score=record.decision_score,
            decision_reason=record.decision_reason,
            confidence=confidence,
            score_margin=score_margin,
            runner_up=runner_up,
            runner_up_score=runner_up_score,
            alternatives=alternatives,
            expected_effect=expected_effect,
            observed_effect=observed_effect,
            agent_effect=agent_effect,
            environment_effect=environment_effect,
            effect_match=effect_match,
            evidence=evidence,
        )

    @staticmethod
    def _build_alternatives(
        trace: DecisionTrace,
    ) -> List[DecisionAlternative]:
        """
        Convert the trace's candidate landscape into alternatives.

        The selected decision itself is excluded. Alternatives are
        ordered by their original candidate scores.
        """

        alternatives: List[DecisionAlternative] = []

        for candidate in trace.top_candidates:
            if candidate.name == trace.selected_decision:
                continue

            alternatives.append(
                DecisionAlternative(
                    name=candidate.name,
                    score=candidate.score,
                    action=list(candidate.action),
                    score_difference=(
                        trace.selected_score - candidate.score
                    ),
                )
            )

        return alternatives


def explain_decision(
    record: DecisionRecord,
    outcome: Optional[OutcomeEvaluation] = None,
) -> DecisionIntelligenceReport:
    """Convenience wrapper for building decision intelligence."""

    return DecisionIntelligenceBuilder().build(
        record=record,
        outcome=outcome,
    )