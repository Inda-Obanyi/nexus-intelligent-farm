from dataclasses import dataclass
from typing import List

from nexus.intelligence.prediction.models import Prediction


@dataclass(frozen=True)
class PredictionPolicy:
    """
    Policy defining the minimum quality requirements for using
    a Prediction inside NEXUS decision intelligence.

    The policy does not alter or reinterpret the prediction.

    It evaluates whether the prediction satisfies explicit
    operational requirements.

    This is intentionally separate from the prediction model
    itself so that different NEXUS decision contexts can use
    different acceptance policies.
    """

    minimum_confidence: float = 0.60
    maximum_uncertainty: float = 0.50
    require_provenance: bool = True
    require_evidence: bool = False

    def __post_init__(self):
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "Minimum confidence must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.maximum_uncertainty <= 1.0:
            raise ValueError(
                "Maximum uncertainty must be between 0.0 and 1.0."
            )


@dataclass(frozen=True)
class PredictionValidation:
    """
    Result of evaluating a Prediction against a PredictionPolicy.

    The result preserves the individual validation reasons so
    NEXUS can explain why a prediction was accepted or rejected.
    """

    accepted: bool
    reasons: List[str]

    @property
    def rejected(self) -> bool:
        return not self.accepted


class PredictionPolicyEvaluator:
    """
    Evaluate predictions against explicit NEXUS quality policies.

    Architectural boundary:

        Prediction
            ↓
        PredictionPolicyEvaluator
            ↓
        PredictionValidation
            ↓
        Risk / Decision Context
    """

    def evaluate(
        self,
        prediction: Prediction,
        policy: PredictionPolicy,
    ) -> PredictionValidation:
        reasons = []

        if prediction.confidence < policy.minimum_confidence:
            reasons.append(
                "Prediction confidence is below the policy threshold."
            )

        if prediction.uncertainty > policy.maximum_uncertainty:
            reasons.append(
                "Prediction uncertainty is above the policy threshold."
            )

        if (
            policy.require_provenance
            and prediction.provenance is None
        ):
            reasons.append(
                "Prediction provenance is required by the policy."
            )

        if (
            policy.require_evidence
            and not prediction.evidence
        ):
            reasons.append(
                "Prediction evidence is required by the policy."
            )

        if not reasons:
            reasons.append(
                "Prediction satisfies the policy requirements."
            )

        return PredictionValidation(
            accepted=not any(
                reason != "Prediction satisfies the policy requirements."
                for reason in reasons
            ),
            reasons=reasons,
        )
