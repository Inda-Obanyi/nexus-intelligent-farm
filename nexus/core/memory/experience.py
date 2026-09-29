from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExperienceRecord:
    """
    A learning-oriented record of one NEXUS decision experience.

    An experience connects:

        state/context
            ↓
        decision
            ↓
        execution
            ↓
        observed outcome
            ↓
        decision intelligence

    This record is intentionally descriptive.

    It does not assume that the selected decision was optimal,
    and it does not represent a reinforcement-learning transition
    yet. It provides the structured evidence from which future
    learning and strategy analysis can be built.
    """

    step: int
    day: int
    hour: int

    observation_summary: Dict[str, Any]

    decision_name: str
    decision_action: List[str]
    decision_score: float

    confidence: str
    score_margin: float

    runner_up: Optional[str] = None
    runner_up_score: Optional[float] = None

    alternative_count: int = 0

    expected_effect: str = ""
    observed_effect: str = ""

    agent_effect: str = ""
    environment_effect: str = ""

    effect_match: str = ""

    executed_action: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None

    evidence: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_ambiguous(self) -> bool:
        """
        Return whether the decision was classified as ambiguous.

        LOW confidence represents weak score separation between
        the selected decision and alternatives. It is not a
        probability.
        """
        return self.confidence == "LOW"

    @property
    def has_observed_outcome(self) -> bool:
        """Return whether an outcome was recorded."""
        return bool(self.observed_effect)

    @property
    def effect_was_matched(self) -> bool:
        """Return whether the expected effect matched the observation."""
        return self.effect_match == "MATCHED"

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the experience into a serializable dictionary.

        Lists and dictionaries are copied so callers cannot mutate
        the record through the returned representation.
        """
        return {
            "step": self.step,
            "day": self.day,
            "hour": self.hour,
            "observation_summary": dict(
                self.observation_summary
            ),
            "decision_name": self.decision_name,
            "decision_action": list(
                self.decision_action
            ),
            "decision_score": self.decision_score,
            "confidence": self.confidence,
            "score_margin": self.score_margin,
            "runner_up": self.runner_up,
            "runner_up_score": self.runner_up_score,
            "alternative_count": self.alternative_count,
            "expected_effect": self.expected_effect,
            "observed_effect": self.observed_effect,
            "agent_effect": self.agent_effect,
            "environment_effect": self.environment_effect,
            "effect_match": self.effect_match,
            "executed_action": (
                dict(self.executed_action)
                if self.executed_action is not None
                else None
            ),
            "outcome": (
                dict(self.outcome)
                if self.outcome is not None
                else None
            ),
            "evidence": dict(self.evidence),
            "metadata": dict(self.metadata),
        }
