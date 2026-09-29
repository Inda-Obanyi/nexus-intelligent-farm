from nexus.core.state import CropState, FarmState


def operational_risk(crop: CropState, state: FarmState) -> float:
    """
    Estimate normalized operational risk for a crop.

    This first version uses only observable farm-state evidence.
    It does not infer weather, disease, pest, or market risk.

    Returns:
        0.0 = no observed operational concern
        1.0 = high operational concern
    """
    risk = 0.0

    if not crop.watered_today:
        risk += 0.5

    if (
        crop.fertilized_until_day >= 0
        and state.day > crop.fertilized_until_day
    ):
        risk += 0.2

    if crop.harvestable:
        risk += 0.3

    return min(risk, 1.0)
