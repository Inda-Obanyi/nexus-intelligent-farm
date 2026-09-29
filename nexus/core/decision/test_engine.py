from nexus.core.decision.context import DecisionContext, DecisionGoal
from nexus.core.decision.engine import DecisionEngine
from nexus.core.state import FarmResources, FarmState, MarketState


def make_context(
    money=3000.0,
    seeds=None,
    crops=None,
    weeds=None,
    day=0,
    goals=None,
    farmer_position=None,
):
    state = FarmState(
        player_id=0,
        day=day,
        hour=0,
        farmer_position=farmer_position or [4, 4],
        crops=crops or [],
        weeds=weeds or [],
        resources=FarmResources(
            money=money,
            seeds=seeds or {},
            shed={},
        ),
        market=MarketState(
            prices={
                "WHEAT": 25,
                "CARROT": 35,
                "TOMATO": 60,
                "STRAWBERRY": 120,
                "MELON": 250,
            },
            inventory={},
        ),
        raw_observation={"step": 0},
    )

    return DecisionContext(
        state=state,
        goals=goals or [],
    )


def test_engine_buys_wheat_seed_when_seed_is_needed():
    context = make_context()

    decision = DecisionEngine().decide(context)

    assert decision.name == "BUY_WHEAT_SEED"
    assert decision.action == ["BUY_SEED", "WHEAT", 1]


def test_engine_plants_wheat_when_seed_is_available():
    context = make_context(
        seeds={"WHEAT": 1},
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "PLANT_WHEAT"
    assert decision.action == ["PLANT", "WHEAT"]


def test_engine_waters_unwatered_crop():
    from nexus.core.state import CropState

    crop = CropState(
        crop="WHEAT",
        row=4,
        col=4,
        planted_day=0,
        watered_today=False,
        yield_units=1,
        fertilized_until_day=-1,
    )

    context = make_context(
        crops=[crop],
        day=1,
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "WATER_CROP"
    assert decision.action == ["WATER"]


def test_engine_does_not_water_already_watered_crop():
    from nexus.core.state import CropState

    crop = CropState(
        crop="WHEAT",
        row=4,
        col=4,
        planted_day=0,
        watered_today=True,
        yield_units=1,
        fertilized_until_day=-1,
    )

    context = make_context(
        crops=[crop],
        day=1,
    )

    decision = DecisionEngine().decide(context)

    assert decision.name != "WATER_CROP"
    assert decision.action != ["WATER"]


def test_engine_does_not_harvest_immature_crop():
    from nexus.core.state import CropState

    crop = CropState(
        crop="WHEAT",
        row=4,
        col=4,
        planted_day=0,
        watered_today=True,
        yield_units=1,
        fertilized_until_day=-1,
    )

    context = make_context(
        crops=[crop],
        day=1,
    )

    decision = DecisionEngine().decide(context)

    assert decision.name != "HARVEST_CROP"
    assert decision.action != ["HARVEST"]


def test_engine_harvests_mature_crop():
    from nexus.core.state import CropState

    crop = CropState(
        crop="WHEAT",
        row=4,
        col=4,
        planted_day=0,
        watered_today=True,
        yield_units=1,
        fertilized_until_day=-1,
    )

    context = make_context(
        crops=[crop],
        day=2,
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "HARVEST_CROP"
    assert decision.action == ["HARVEST"]


def test_engine_uses_pass_when_no_action_is_available():
    context = make_context(
        money=0.0,
        seeds={},
        crops=[],
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "PASS"
    assert decision.action == ["PASS"]


def test_engine_plants_carrot_when_seed_is_available():
    context = make_context(
        seeds={"CARROT": 1},
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "PLANT_CARROT"
    assert decision.action == ["PLANT", "CARROT"]


def test_engine_buys_carrot_seed_when_carrot_is_prioritized():
    context = make_context(
        money=3000.0,
        goals=[
            DecisionGoal(
                name="prioritize_carrot",
                priority=1.0,
            )
        ],
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "BUY_CARROT_SEED"
    assert decision.action == ["BUY_SEED", "CARROT", 1]


def test_engine_plants_carrot_when_carrot_is_prioritized():
    context = make_context(
        seeds={"CARROT": 1},
        goals=[
            DecisionGoal(
                name="prioritize_carrot",
                priority=1.0,
            )
        ],
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "PLANT_CARROT"
    assert decision.action == ["PLANT", "CARROT"]


def test_engine_generates_dig_weed_option_when_farmer_is_on_weed():
    from nexus.core.state import WeedState

    context = make_context(
        weeds=[
            WeedState(
                row=4,
                col=4,
            )
        ],
    )

    options = DecisionEngine()._generate_options(context)

    dig_options = [
        option
        for option in options
        if option.name == "DIG_WEED"
    ]

    assert len(dig_options) == 1
    assert dig_options[0].action == ["DIG"]
    assert "weed" in dig_options[0].reason.lower()


def test_engine_does_not_generate_dig_weed_when_farmer_is_not_on_weed():
    from nexus.core.state import WeedState

    context = make_context(
        weeds=[
            WeedState(
                row=1,
                col=2,
            )
        ],
    )

    options = DecisionEngine()._generate_options(context)

    dig_options = [
        option
        for option in options
        if option.name == "DIG_WEED"
    ]

    assert dig_options == []


def test_engine_generates_clear_weed_for_distant_weed():
    from nexus.core.state import WeedState

    context = make_context(
        farmer_position=[4, 4],
        weeds=[
            WeedState(
                row=1,
                col=2,
            )
        ],
    )

    options = DecisionEngine()._generate_options(context)

    clear_options = [
        option
        for option in options
        if option.name == "CLEAR_WEED"
    ]

    assert len(clear_options) == 1
    assert clear_options[0].action == [
        "CLEAR_WEED",
        2,
        1,
    ]
    assert "weed" in clear_options[0].reason.lower()


def test_engine_selects_closest_weed_for_clear_weed():
    from nexus.core.state import WeedState

    context = make_context(
        farmer_position=[4, 4],
        weeds=[
            WeedState(
                row=1,
                col=1,
            ),
            WeedState(
                row=4,
                col=3,
            ),
            WeedState(
                row=8,
                col=8,
            ),
        ],
    )

    options = DecisionEngine()._generate_options(context)

    clear_options = [
        option
        for option in options
        if option.name == "CLEAR_WEED"
    ]

    assert len(clear_options) == 1
    assert clear_options[0].action == [
        "CLEAR_WEED",
        3,
        4,
    ]


def test_engine_does_not_generate_clear_weed_when_no_weeds_exist():
    context = make_context(
        farmer_position=[4, 4],
        weeds=[],
    )

    options = DecisionEngine()._generate_options(context)

    clear_options = [
        option
        for option in options
        if option.name == "CLEAR_WEED"
    ]

    assert clear_options == []


def test_engine_prefers_dig_when_farmer_is_on_weed():
    from nexus.core.state import WeedState

    context = make_context(
        farmer_position=[4, 4],
        weeds=[
            WeedState(
                row=4,
                col=4,
            )
        ],
    )

    decision = DecisionEngine().decide(context)

    assert decision.name == "DIG_WEED"
    assert decision.action == ["DIG"]


def test_engine_preserves_decision_trace():
    context = make_context()

    engine = DecisionEngine()
    decision = engine.decide(context)

    assert decision.name == "BUY_WHEAT_SEED"
    assert engine.last_trace is not None
    assert engine.last_trace.selected_decision == "BUY_WHEAT_SEED"


def test_engine_trace_contains_candidate_landscape():
    context = make_context()

    engine = DecisionEngine()
    engine.decide(context)

    trace = engine.last_trace

    assert trace is not None
    assert trace.candidate_count > 1
    assert len(trace.candidates) == trace.candidate_count


def test_engine_trace_contains_runner_up():
    context = make_context()

    engine = DecisionEngine()
    engine.decide(context)

    trace = engine.last_trace

    assert trace is not None
    assert trace.runner_up is not None
    assert trace.runner_up_score is not None


def test_engine_trace_margin_matches_selected_and_runner_up():
    context = make_context()

    engine = DecisionEngine()
    engine.decide(context)

    trace = engine.last_trace

    assert trace is not None
    assert trace.score_margin == (
        trace.selected_score - trace.runner_up_score
    )


def test_engine_trace_candidate_scores_match_decision_scores():
    context = make_context()

    engine = DecisionEngine()
    engine.decide(context)

    trace = engine.last_trace

    assert trace is not None

    candidate_names = {
        candidate.name
        for candidate in trace.candidates
    }

    assert "BUY_WHEAT_SEED" in candidate_names
    assert "PASS" in candidate_names


def test_engine_trace_reports_high_confidence_for_weed_dig():
    from nexus.core.state import WeedState

    context = make_context(
        farmer_position=[4, 4],
        weeds=[
            WeedState(
                row=4,
                col=4,
            )
        ],
    )

    engine = DecisionEngine()
    decision = engine.decide(context)

    trace = engine.last_trace

    assert decision.name == "DIG_WEED"
    assert trace is not None
    assert trace.selected_decision == "DIG_WEED"
    assert trace.confidence == "HIGH"


def test_engine_trace_candidate_count_is_consistent():
    context = make_context(
        seeds={"WHEAT": 1},
    )

    engine = DecisionEngine()
    engine.decide(context)

    trace = engine.last_trace

    assert trace is not None
    assert trace.candidate_count == len(trace.candidates)
    assert trace.candidate_count >= 2


def test_engine_generates_plant_intent_when_current_tile_has_crop_but_other_tile_is_plantable():
    from nexus.core.state import CropState, FarmTileState

    context = make_context(
        farmer_position=[4, 4],
        seeds={"WHEAT": 1},
        crops=[
            CropState(
                crop="CARROT",
                row=4,
                col=4,
                planted_day=0,
                watered_today=True,
                yield_units=0,
                fertilized_until_day=-1,
            )
        ],
    )

    context.state.tiles = [
        FarmTileState(
            row=4,
            col=4,
            kind="PLANT",
            locked=False,
            occupied=True,
        ),
        FarmTileState(
            row=4,
            col=3,
            kind="EMPTY",
            locked=False,
            occupied=False,
        ),
    ]

    options = DecisionEngine()._generate_options(context)

    plant_options = [
        option
        for option in options
        if option.name == "PLANT_WHEAT"
    ]

    assert len(plant_options) == 1
    assert plant_options[0].action == ["PLANT", "WHEAT"]


def test_engine_generates_plant_intent_when_current_tile_has_crop_but_other_tile_is_plantable():
    from nexus.core.state import CropState, FarmTileState

    context = make_context(
        farmer_position=[4, 4],
        seeds={"WHEAT": 1},
        crops=[
            CropState(
                crop="CARROT",
                row=4,
                col=4,
                planted_day=0,
                watered_today=True,
                yield_units=0,
                fertilized_until_day=-1,
            )
        ],
    )

    context.state.tiles = [
        FarmTileState(
            row=4,
            col=4,
            kind="PLANT",
            locked=False,
            occupied=True,
        ),
        FarmTileState(
            row=4,
            col=3,
            kind="EMPTY",
            locked=False,
            occupied=False,
        ),
    ]

    options = DecisionEngine()._generate_options(context)

    plant_options = [
        option
        for option in options
        if option.name == "PLANT_WHEAT"
    ]

    assert len(plant_options) == 1
    assert plant_options[0].action == ["PLANT", "WHEAT"]

def test_engine_can_buy_carrot_when_wheat_seed_already_exists():
    context = make_context(
        money=3000.0,
        seeds={
            "WHEAT": 1,
            "CARROT": 0,
        },
        goals=[
            DecisionGoal(
                name="prioritize_carrot",
                priority=1.0,
            )
        ],
    )

    options = DecisionEngine()._generate_options(context)

    carrot_buy_options = [
        option
        for option in options
        if option.name == "BUY_CARROT_SEED"
    ]

    assert len(carrot_buy_options) == 1
    assert carrot_buy_options[0].action == [
        "BUY_SEED",
        "CARROT",
        1,
    ]
