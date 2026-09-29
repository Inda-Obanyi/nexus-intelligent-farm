from nexus.core.decision.context import DecisionContext
from nexus.core.decision.preconditions import ActionPreconditions
from nexus.core.state import CropState, FarmResources, FarmState, MarketState


def make_context(
    money=3000.0,
    seeds=None,
    crops=None,
    day=0,
):
    state = FarmState(
        player_id=0,
        day=day,
        hour=0,
        farmer_position=[4, 4],
        crops=crops or [],
        resources=FarmResources(
            money=money,
            seeds=seeds or {},
            shed={},
        ),
        market=MarketState(
            prices={"WHEAT": 25},
            inventory={},
        ),
        raw_observation={"step": 0},
    )

    return DecisionContext(state=state)


def make_crop(
    planted_day=0,
    yield_units=1,
    watered_today=False,
):
    return CropState(
        crop="WHEAT",
        row=4,
        col=4,
        planted_day=planted_day,
        watered_today=watered_today,
        yield_units=yield_units,
        fertilized_until_day=-1,
    )


def test_can_buy_wheat_seed_when_requirements_are_met():
    context = make_context()

    result = ActionPreconditions().can_buy_wheat_seed(context)

    assert result.valid is True


def test_cannot_buy_wheat_seed_when_seed_already_exists():
    context = make_context(seeds={"WHEAT": 1})

    result = ActionPreconditions().can_buy_wheat_seed(context)

    assert result.valid is False


def test_can_plant_wheat_when_seed_exists_and_tile_is_empty():
    context = make_context(seeds={"WHEAT": 1})

    result = ActionPreconditions().can_plant_wheat(context)

    assert result.valid is True


def test_cannot_plant_wheat_without_seed():
    context = make_context()

    result = ActionPreconditions().can_plant_wheat(context)

    assert result.valid is False


def test_can_water_unwatered_crop():
    crop = make_crop(watered_today=False)
    context = make_context(crops=[crop])

    result = ActionPreconditions().can_water_crop(context)

    assert result.valid is True


def test_cannot_water_already_watered_crop():
    crop = make_crop(watered_today=True)
    context = make_context(crops=[crop])

    result = ActionPreconditions().can_water_crop(context)

    assert result.valid is False


def test_cannot_harvest_crop_before_first_harvest_day():
    crop = make_crop(
        planted_day=0,
        yield_units=1,
    )
    context = make_context(
        crops=[crop],
        day=1,
    )

    result = ActionPreconditions().can_harvest_crop(context)

    assert result.valid is False


def test_can_harvest_crop_at_first_harvest_day():
    crop = make_crop(
        planted_day=0,
        yield_units=2,
    )
    context = make_context(
        crops=[crop],
        day=2,
    )

    result = ActionPreconditions().can_harvest_crop(context)

    assert result.valid is True


def test_can_plant_crop_supports_wheat():
    context = make_context(seeds={"WHEAT": 1})

    result = ActionPreconditions().can_plant_crop(context, "WHEAT")

    assert result.valid is True


def test_can_plant_crop_supports_carrot():
    context = make_context(seeds={"CARROT": 1})

    result = ActionPreconditions().can_plant_crop(context, "CARROT")

    assert result.valid is True


def test_can_plant_crop_rejects_missing_seed():
    context = make_context()

    result = ActionPreconditions().can_plant_crop(context, "CARROT")

    assert result.valid is False
    assert "seed" in result.reason.lower()


def test_can_plant_crop_rejects_occupied_tile():
    from nexus.core.state import CropState

    context = make_context(
        seeds={"CARROT": 1},
        crops=[
            CropState(
                crop="WHEAT",
                row=4,
                col=4,
                planted_day=0,
                watered_today=False,
                yield_units=1,
                fertilized_until_day=0,
            )
        ],
    )

    result = ActionPreconditions().can_plant_crop(context, "CARROT")

    assert result.valid is False
    assert "already contains" in result.reason.lower()


def test_can_dig_weed_when_farmer_is_standing_on_weed():
    from nexus.core.state import WeedState

    context = make_context()
    context.state.weeds = [
        WeedState(row=4, col=4),
    ]

    result = ActionPreconditions().can_dig_weed(context)

    assert result.valid is True
    assert "weed" in result.reason.lower()


def test_cannot_dig_weed_when_farmer_is_not_on_weed():
    from nexus.core.state import WeedState

    context = make_context()
    context.state.weeds = [
        WeedState(row=1, col=2),
    ]

    result = ActionPreconditions().can_dig_weed(context)

    assert result.valid is False
    assert "weed" in result.reason.lower()


def test_cannot_dig_weed_when_no_weeds_exist():
    context = make_context()

    result = ActionPreconditions().can_dig_weed(context)

    assert result.valid is False
    assert "weed" in result.reason.lower()
