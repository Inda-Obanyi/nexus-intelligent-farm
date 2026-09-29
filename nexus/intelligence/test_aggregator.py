from nexus.core.state import CropState, FarmState
from nexus.intelligence.aggregator import IntelligenceAggregator


def make_state(crops):
    return FarmState(
        player_id=0,
        day=1,
        hour=0,
        farmer_position=[0, 0],
        crops=crops,
    )


def test_aggregator_builds_operational_risk_for_each_crop():
    crops = [
        CropState(
            crop="WHEAT",
            row=0,
            col=0,
            planted_day=0,
            watered_today=False,
            yield_units=0,
            fertilized_until_day=-1,
        ),
        CropState(
            crop="MELON",
            row=1,
            col=1,
            planted_day=0,
            watered_today=True,
            yield_units=0,
            fertilized_until_day=-1,
        ),
    ]

    intelligence = IntelligenceAggregator().build(
        make_state(crops)
    )

    assert set(intelligence.crop_risks) == {
        "WHEAT",
        "MELON",
    }

    assert intelligence.crop_risks["WHEAT"].operational == 0.5
    assert intelligence.crop_risks["MELON"].operational == 0.0


def test_aggregator_normalizes_crop_risk_keys():
    crop = CropState(
        crop="wheat",
        row=0,
        col=0,
        planted_day=0,
        watered_today=True,
        yield_units=0,
        fertilized_until_day=-1,
    )

    intelligence = IntelligenceAggregator().build(
        make_state([crop])
    )

    assert "WHEAT" in intelligence.crop_risks
    assert intelligence.crop_risks["WHEAT"].crop == "wheat"


def test_aggregator_handles_farm_without_crops():
    intelligence = IntelligenceAggregator().build(
        make_state([])
    )

    assert intelligence.crop_risks == {}
    assert intelligence.predictions == {}


def test_aggregator_does_not_invent_predictions():
    crop = CropState(
        crop="WHEAT",
        row=0,
        col=0,
        planted_day=0,
        watered_today=True,
        yield_units=0,
        fertilized_until_day=-1,
    )

    intelligence = IntelligenceAggregator().build(
        make_state([crop])
    )

    assert intelligence.predictions == {}
