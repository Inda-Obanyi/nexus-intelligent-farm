from typing import Any, Dict, Optional

from nexus.intelligence.prediction.models import Prediction


class PredictionBuilder:
    """
    Construct standardized NEXUS Prediction objects.

    The builder provides a single interface for intelligence
    components to convert model or rule outputs into the common
    Prediction contract.

    Architectural boundary:

        Model / Rule / API / Sensor
                    ↓
            PredictionBuilder
                    ↓
                Prediction
                    ↓
             Risk / Decision
    """

    def build(
        self,
        target: str,
        value: Any,
        *,
        confidence: float = 0.0,
        uncertainty: float = 0.0,
        evidence: Optional[Dict[str, Any]] = None,
        provenance: Optional[str] = None,
        explanation: Optional[str] = None,
    ) -> Prediction:
        """
        Build a validated Prediction.

        Args:
            target:
                Name of the quantity, event, risk, or state being
                predicted.

            value:
                Prediction output. This may be a number, category,
                boolean, structured value, or another representation.

            confidence:
                Normalized confidence metadata in [0, 1].

            uncertainty:
                Normalized uncertainty metadata in [0, 1].

            evidence:
                Observable inputs or derived features supporting
                the prediction.

            provenance:
                Identifier describing the model, rule, API, sensor,
                simulation, or other source that produced the
                prediction.

            explanation:
                Human-readable explanation of the prediction.

        Returns:
            A validated Prediction instance.
        """

        if evidence is None:
            evidence = {}

        return Prediction(
            target=target,
            value=value,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=evidence,
            provenance=provenance,
            explanation=explanation,
        )
