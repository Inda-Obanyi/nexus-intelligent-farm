from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CandidateScore:
    """
    Scored decision candidate preserved for decision-quality analysis.

    This represents one candidate considered by the decision engine.
    """

    name: str
    score: float
    action: List[str]


@dataclass(frozen=True)
class DecisionTrace:
    """
    Evidence trace for a single decision event.

    The trace preserves the decision landscape that existed when the
    DecisionEngine selected an action.

    Confidence is intentionally an interpretability label based on the
    score margin between the selected decision and the runner-up. It is
    not a probability and should not be interpreted as statistical
    confidence.
    """

    selected_decision: str
    selected_score: float
    candidates: List[CandidateScore] = field(default_factory=list)
    runner_up: Optional[str] = None
    runner_up_score: Optional[float] = None
    score_margin: float = 0.0
    candidate_count: int = 0
    confidence: str = "LOW"

    def __post_init__(self):
        if self.selected_score < float("-inf"):
            raise ValueError("Selected score must be a valid numeric value.")

        if self.score_margin < 0:
            raise ValueError("Score margin cannot be negative.")

        if self.candidate_count < 0:
            raise ValueError("Candidate count cannot be negative.")

        if self.runner_up is None and self.runner_up_score is not None:
            raise ValueError(
                "runner_up_score cannot be provided without runner_up."
            )

        if self.runner_up is not None and self.runner_up_score is None:
            raise ValueError(
                "runner_up must have a corresponding runner_up_score."
            )

    @property
    def is_ambiguous(self) -> bool:
        """Return True when the decision has little separation from its runner-up."""
        return self.confidence == "LOW"

    @property
    def top_candidates(self) -> List[CandidateScore]:
        """Return candidates ordered from highest score to lowest score."""
        return sorted(
            self.candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )

    @classmethod
    def from_scored_candidates(
        cls,
        candidates: List[CandidateScore],
    ) -> "DecisionTrace":
        """
        Build a decision trace from already-scored candidates.

        Candidates are expected to contain only valid decision options.
        """

        if not candidates:
            raise ValueError(
                "At least one candidate is required to build a decision trace."
            )

        ranked = sorted(
            candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        selected = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        margin = (
            selected.score - runner_up.score
            if runner_up is not None
            else 0.0
        )

        confidence = cls._confidence_from_margin(margin)

        return cls(
            selected_decision=selected.name,
            selected_score=selected.score,
            candidates=list(candidates),
            runner_up=runner_up.name if runner_up else None,
            runner_up_score=runner_up.score if runner_up else None,
            score_margin=margin,
            candidate_count=len(candidates),
            confidence=confidence,
        )

    @staticmethod
    def _confidence_from_margin(margin: float) -> str:
        """
        Convert score separation into an interpretable confidence label.

        These thresholds describe decision ambiguity only. They are not
        calibrated probabilities.
        """

        if margin >= 5.0:
            return "HIGH"

        if margin >= 2.0:
            return "MEDIUM"

        return "LOW"
