from dataclasses import dataclass, field
from typing import Dict

from nexus.core.state import FarmState
from nexus.intelligence.prediction.models import Prediction
from nexus.intelligence.risk.builder import RiskSignalBuilder
from nexus.intelligence.risk.models import CropRisk


@dataclass(frozen=True)
class FarmIntelligence:
    """
    Structured intelligence assembled from the current farm state.

    This object contains intelligence outputs only. It does not
    make a farm decision.
    """

    predictions: Dict[str, Prediction] = field(default_factory=dict)
    crop_risks: Dict[str, CropRisk] = field(default_factory=dict)


class IntelligenceAggregator:
    """
    Assemble available NEXUS intelligence from observable farm state.

    The aggregator provides a single boundary between intelligence
    producers and the decision system.
    """

    def __init__(
        self,
        risk_builder=None,
    ):
        self.risk_builder = (
            risk_builder
            or RiskSignalBuilder()
        )

    def build(
        self,
        state: FarmState,
    ) -> FarmIntelligence:
        crop_risks = {}

        for crop in state.crops:
            risk = self.risk_builder.build(
                crop,
                state,
            )

            crop_risks[crop.crop.upper()] = risk

        return FarmIntelligence(
            crop_risks=crop_risks,
        )
