from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Prediction:
    """
    Standard NEXUS prediction contract.

    A Prediction represents an intelligence output produced from
    observable evidence.

    It deliberately separates:

        prediction
            What the intelligence component estimates.

        confidence
            How confident the component is in that estimate.

        uncertainty
            How much uncertainty remains around the estimate.

        evidence
            The observable inputs or derived features supporting it.

        provenance
            Where the prediction came from.

    This contract is model-agnostic. Future NEXUS intelligence
    components may produce predictions using:

        - machine learning
        - statistical models
        - rules
        - simulation
        - external APIs
        - sensor observations
        - multimodal models

    The Prediction object does not make a farm decision itself.

    Architectural boundary:

        Intelligence
            ↓
        Prediction
            ↓
        Risk / Decision Context
            ↓
        Decision Engine
    """

    target: str
    value: Any

    confidence: float = 0.0
    uncertainty: float = 0.0

    evidence: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[str] = None
    explanation: Optional[str] = None

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):
        """
        Validate prediction metadata.

        Confidence and uncertainty are normalized to [0, 1]:

            0.0 = none
            1.0 = maximum

        These values describe the prediction metadata. They do not
        automatically represent calibrated probabilities.
        """

        if not self.target.strip():
            raise ValueError(
                "Prediction target cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Prediction confidence must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                "Prediction uncertainty must be between 0.0 and 1.0."
            )

        if not isinstance(self.evidence, dict):
            raise TypeError(
                "Prediction evidence must be a dictionary."
            )

        if self.provenance is not None:
            if not self.provenance.strip():
                raise ValueError(
                    "Prediction provenance cannot be an empty string."
                )

        if self.explanation is not None:
            if not self.explanation.strip():
                raise ValueError(
                    "Prediction explanation cannot be an empty string."
                )

    @property
    def is_high_confidence(self) -> bool:
        """
        Return whether the prediction has high confidence.

        The initial interpretability threshold is 0.80.

        This is a reporting convenience, not a claim of statistical
        calibration.
        """
        return self.confidence >= 0.80

    @property
    def is_high_uncertainty(self) -> bool:
        """
        Return whether the prediction has high uncertainty.

        The initial interpretability threshold is 0.50.

        This threshold is intentionally simple and can later be
        replaced by domain-specific uncertainty policies.
        """
        return self.uncertainty >= 0.50
