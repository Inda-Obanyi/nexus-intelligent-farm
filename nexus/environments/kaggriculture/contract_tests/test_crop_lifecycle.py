
from nexus.core.state import CropState, FarmState, FarmResources
from nexus.core.decision.context import DecisionContext


def make_crop_state(crop: str, planted_day: int, yield_units: int) -> CropState:
    return CropState(
        crop=crop,
        row=4,
        col=4,
        planted_day=planted_day,
        watered_today=True,
        yield_units=yield_units,
        fertilized_until_day=0,
    )


def make_context(day: int, crop: CropState) -> DecisionContext:
    state = FarmState(
        player_id=0,
        day=day,
        hour=0,
        farmer_position=(4, 4),
        crops=[crop],
        resources=FarmResources(
            money=3000.0,
            seeds={},
            shed={},
        ),
        market={},
        raw_observation={},
    )
    return DecisionContext(state=state)


def test_crop_with_yield_is_not_harvestable_before_first_harvest_day():
    crop = make_crop_state(
        crop="WHEAT",
        planted_day=0,
        yield_units=1,
    )

    context = make_context(day=1, crop=crop)

    assert crop.harvestable is True
    assert context.is_crop_harvestable(crop) is False


def test_crop_becomes_harvestable_at_first_harvest_day():
    crop = make_crop_state(
        crop="WHEAT",
        planted_day=0,
        yield_units=2,
    )

    context = make_context(day=2, crop=crop)

    assert context.is_crop_harvestable(crop) is True
