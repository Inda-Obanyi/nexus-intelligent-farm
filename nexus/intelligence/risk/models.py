from dataclasses import dataclass


@dataclass(frozen=True)
class CropRisk:
    """
    Structured crop-risk representation used by NEXUS.

    Risk values are normalized:
        0.0 = very low risk
        1.0 = very high risk

    The risk model is intentionally independent from prediction
    models. Future intelligence components can populate these
    signals from weather, disease, pest, market, and operational
    predictions.
    """

    crop: str
    weather: float = 0.0
    disease: float = 0.0
    pest: float = 0.0
    market: float = 0.0
    operational: float = 0.0

    def __post_init__(self):
        values = {
            "weather": self.weather,
            "disease": self.disease,
            "pest": self.pest,
            "market": self.market,
            "operational": self.operational,
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} risk must be between 0.0 and 1.0."
                )

    def weighted_overall(self, weights: dict[str, float]) -> float:
        """Return the weighted aggregate crop risk."""
        expected_fields = {
            "weather",
            "disease",
            "pest",
            "market",
            "operational",
        }

        if set(weights) != expected_fields:
            raise ValueError(
                "Weights must contain exactly: "
                "weather, disease, pest, market, operational."
            )

        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Risk weights cannot be negative.")

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError("Total risk weight must be greater than zero.")

        return (
            self.weather * weights["weather"]
            + self.disease * weights["disease"]
            + self.pest * weights["pest"]
            + self.market * weights["market"]
            + self.operational * weights["operational"]
        ) / total_weight

    @property
    def overall(self) -> float:
        """
        Return the aggregate crop risk.

        Equal weighting is used initially. NEXUS can later replace
        this with learned, crop-specific, seasonal, or context-aware
        weighting.
        """
        return (
            self.weather
            + self.disease
            + self.pest
            + self.market
            + self.operational
        ) / 5.0
