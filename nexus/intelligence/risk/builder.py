from nexus.core.state import CropState, FarmState
from nexus.intelligence.risk.models import CropRisk
from nexus.intelligence.risk.signals import operational_risk


class RiskSignalBuilder:
    """
    Build a structured CropRisk from observable farm-state signals.

    The builder currently populates only operational risk.
    Weather, disease, pest, and market signals remain at their
    model defaults until their respective intelligence components
    are implemented.
    """

    def build(self, crop: CropState, state: FarmState) -> CropRisk:
        return CropRisk(
            crop=crop.crop,
            operational=operational_risk(crop, state),
        )
