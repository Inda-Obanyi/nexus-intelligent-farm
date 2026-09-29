from nexus.evaluation.outcomes import DecisionOutcomeEvaluator


def make_outcome(**overrides):
    outcome = {
        "money_before": 3000.0,
        "money_after": 2990.0,
        "money_delta": -10.0,
        "position_before": [4, 4],
        "position_after": [4, 4],
        "crop_count_before": 1,
        "crop_count_after": 1,
        "weed_count_before": 0,
        "weed_count_after": 0,
        "seeds_before": {
            "WHEAT": 0,
            "CARROT": 0,
            "TOMATO": 0,
            "STRAWBERRY": 0,
            "MELON": 0,
        },
        "seeds_after": {
            "WHEAT": 0,
            "CARROT": 0,
            "TOMATO": 0,
            "STRAWBERRY": 0,
            "MELON": 0,
        },
        "shed_before": {},
        "shed_after": {},
        "crops_before": [
            {
                "crop": "WHEAT",
                "row": 4,
                "col": 4,
                "planted_day": 0,
                "watered_today": True,
                "yield_units": 1,
                "fertilized_until_day": -1,
            }
        ],
        "crops_after": [
            {
                "crop": "WHEAT",
                "row": 4,
                "col": 4,
                "planted_day": 0,
                "watered_today": True,
                "yield_units": 1,
                "fertilized_until_day": -1,
            }
        ],
        "executed_action": {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        },
    }

    outcome.update(overrides)

    return outcome


def test_evaluator_records_negative_cash_effect():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="BUY_WHEAT_SEED",
        outcome=make_outcome(
            seeds_before={
                "WHEAT": 0,
            },
            seeds_after={
                "WHEAT": 1,
            },
        ),
    )

    assert result.money_delta == -10.0
    assert result.economic_effect == (
        "Immediate cash decreased."
    )


def test_buy_seed_effect_matches():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="BUY_WHEAT_SEED",
        outcome=make_outcome(
            seeds_before={
                "WHEAT": 0,
            },
            seeds_after={
                "WHEAT": 1,
            },
            money_delta=-10.0,
        ),
    )

    assert result.effect_match == "MATCHED"


def test_evaluator_detects_position_change():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="MOVE",
        outcome=make_outcome(
            money_after=3000.0,
            money_delta=0.0,
            position_after=[5, 4],
        ),
    )

    assert result.position_changed is True
    assert "farmer position changed" in (
        result.operational_effect
    )


def test_evaluator_detects_crop_increase():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="PLANT_WHEAT",
        outcome=make_outcome(
            money_after=2980.0,
            money_delta=-20.0,
            seeds_before={
                "WHEAT": 1,
            },
            seeds_after={
                "WHEAT": 0,
            },
            crop_count_before=0,
            crop_count_after=1,
        ),
    )

    assert result.crop_count_delta == 1
    assert "crop count increased" in (
        result.operational_effect
    )


def test_plant_effect_matches():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="PLANT_WHEAT",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            seeds_before={
                "WHEAT": 1,
            },
            seeds_after={
                "WHEAT": 0,
            },
            crop_count_before=0,
            crop_count_after=1,
            crops_before=[],
            crops_after=[
                {
                    "crop": "WHEAT",
                    "row": 4,
                    "col": 4,
                    "planted_day": 0,
                    "watered_today": False,
                    "yield_units": 1,
                    "fertilized_until_day": -1,
                }
            ],
        ),
    )

    assert result.effect_match == "MATCHED"


def test_plant_navigation_is_partial_not_not_matched():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="PLANT_WHEAT",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            position_before=[0, 1],
            position_after=[1, 1],
            crop_count_before=5,
            crop_count_after=5,
            seeds_before={
                "WHEAT": 1,
            },
            seeds_after={
                "WHEAT": 1,
            },
        ),
        executed_action={
            "farmer": ["EAST"],
            "hands": [],
            "market": [],
        },
    )

    assert result.executed is True
    assert result.position_changed is True
    assert result.effect_match == "PARTIAL"


def test_evaluator_detects_weed_reduction():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="DIG_WEED",
        outcome=make_outcome(
            money_after=3000.0,
            money_delta=0.0,
            weed_count_before=1,
            weed_count_after=0,
        ),
    )

    assert result.weed_count_delta == -1
    assert "weed count decreased" in (
        result.operational_effect
    )


def test_dig_weed_effect_matches():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="DIG_WEED",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            weed_count_before=1,
            weed_count_after=0,
        ),
    )

    assert result.effect_match == "MATCHED"


def test_water_effect_matches():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="WATER_CROP",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            crops_before=[
                {
                    "crop": "WHEAT",
                    "row": 4,
                    "col": 4,
                    "planted_day": 0,
                    "watered_today": False,
                    "yield_units": 1,
                    "fertilized_until_day": -1,
                }
            ],
            crops_after=[
                {
                    "crop": "WHEAT",
                    "row": 4,
                    "col": 4,
                    "planted_day": 0,
                    "watered_today": True,
                    "yield_units": 1,
                    "fertilized_until_day": -1,
                }
            ],
        ),
    )

    assert result.effect_match == "MATCHED"


def test_harvest_effect_matches_shed_and_crop_removal():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="HARVEST_CROP",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            crop_count_before=1,
            crop_count_after=0,
            crops_before=[
                {
                    "crop": "WHEAT",
                    "row": 4,
                    "col": 4,
                    "planted_day": 0,
                    "watered_today": True,
                    "yield_units": 2,
                    "fertilized_until_day": -1,
                }
            ],
            crops_after=[],
            shed_before={},
            shed_after={
                "WHEAT": 2,
            },
        ),
    )

    assert result.effect_match == "MATCHED"
    assert "shed delta" in result.observed_effect
    assert "harvested product added to shed" in (
        result.agent_effect
    )


def test_harvest_with_immediate_inventory_gain_matches_without_shed():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="HARVEST_CROP",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            crop_count_before=1,
            crop_count_after=0,
            crops_before=[
                {
                    "crop": "WHEAT",
                    "row": 4,
                    "col": 4,
                    "planted_day": 0,
                    "watered_today": True,
                    "yield_units": 1,
                    "fertilized_until_day": -1,
                }
            ],
            crops_after=[],
            shed_before={},
            shed_after={},
            target_crop_before={
                "crop": "WHEAT",
                "row": 4,
                "col": 4,
                "planted_day": 0,
                "watered_today": True,
                "yield_units": 1,
                "fertilized_until_day": -1,
            },
            target_crop_after=None,
            target_crop_removed=True,
            target_yield_before=1,
            target_yield_after=None,
            target_yield_reduced=False,
            harvested_product="WHEAT",
            harvested_units_before=0,
            harvested_units_after=1,
            harvested_units_delta=1,
        ),
    )

    assert result.effect_match == "MATCHED"
    assert "harvested product added to farmer inventory" in (
        result.agent_effect
    )


def test_harvest_with_immediate_inventory_gain_matches_without_shed():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="HARVEST_CROP",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            crop_count_before=1,
            crop_count_after=0,
            crops_before=[
                {
                    "crop": "WHEAT",
                    "row": 4,
                    "col": 4,
                    "planted_day": 0,
                    "watered_today": True,
                    "yield_units": 1,
                    "fertilized_until_day": -1,
                }
            ],
            crops_after=[],
            shed_before={},
            shed_after={},
            target_crop_before={
                "crop": "WHEAT",
                "row": 4,
                "col": 4,
                "planted_day": 0,
                "watered_today": True,
                "yield_units": 1,
                "fertilized_until_day": -1,
            },
            target_crop_after=None,
            target_crop_removed=True,
            target_yield_before=1,
            target_yield_after=None,
            target_yield_reduced=False,
            harvested_product="WHEAT",
            harvested_units_before=0,
            harvested_units_after=1,
            harvested_units_delta=1,
        ),
    )

    assert result.effect_match == "MATCHED"
    assert "harvested product added to farmer inventory" in (
        result.agent_effect
    )


def test_harvest_without_shed_change_is_partial():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="HARVEST_CROP",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            crop_count_before=1,
            crop_count_after=0,
            crops_before=[
                {
                    "crop": "WHEAT",
                    "row": 4,
                    "col": 4,
                    "planted_day": 0,
                    "watered_today": True,
                    "yield_units": 2,
                    "fertilized_until_day": -1,
                }
            ],
            crops_after=[],
            shed_before={},
            shed_after={},
        ),
    )

    assert result.crop_count_delta == -1
    assert result.effect_match == "PARTIAL"


def test_harvest_navigation_is_partial_not_not_matched():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="HARVEST_CROP",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            position_before=[4, 4],
            position_after=[3, 4],
            crop_count_before=2,
            crop_count_after=2,
            crops_before=[
                {
                    "crop": "WHEAT",
                    "row": 0,
                    "col": 2,
                    "planted_day": 0,
                    "watered_today": True,
                    "yield_units": 2,
                    "fertilized_until_day": -1,
                },
                {
                    "crop": "WHEAT",
                    "row": 1,
                    "col": 1,
                    "planted_day": 0,
                    "watered_today": True,
                    "yield_units": 2,
                    "fertilized_until_day": -1,
                },
            ],
        ),
        executed_action={
            "farmer": ["WEST"],
            "hands": [],
            "market": [],
        },
    )

    assert result.executed is True
    assert result.position_changed is True
    assert result.crop_count_delta == 0
    assert result.effect_match == "PARTIAL"


def test_pass_effect_matches_when_state_is_unchanged():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="PASS",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
        ),
    )

    assert result.effect_match == "MATCHED"


def test_pass_effect_fails_when_state_changes():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="PASS",
        outcome=make_outcome(
            position_after=[5, 4],
        ),
    )

    assert result.effect_match == "NOT_MATCHED"


def test_evaluator_detects_execution():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="WATER_CROP",
        outcome=make_outcome(),
    )

    assert result.executed is True
    assert result.execution_effect == (
        "An executable action was recorded."
    )


def test_evaluator_detects_missing_execution():
    evaluator = DecisionOutcomeEvaluator()

    outcome = make_outcome(
        executed_action=None,
    )

    result = evaluator.evaluate(
        decision_name="PASS",
        outcome=outcome,
        executed_action=None,
    )

    assert result.executed is False
    assert result.execution_effect == (
        "No executable action was recorded."
    )
    assert result.effect_match == "NOT_MATCHED"


def test_evaluator_exports_dictionary():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="WATER_CROP",
        outcome=make_outcome(),
    )

    data = result.as_dict()

    assert data["decision_name"] == "WATER_CROP"
    assert data["money_delta"] == -10.0
    assert "expected_effect" in data
    assert "observed_effect" in data
    assert "effect_match" in data
    assert "evidence" in data


def test_observed_effect_does_not_report_generic_inventory_delta():
    evaluator = DecisionOutcomeEvaluator()

    result = evaluator.evaluate(
        decision_name="WATER_CROP",
        outcome=make_outcome(
            money_before=3000.0,
            money_after=3000.0,
            money_delta=0.0,
            shed_before={},
            shed_after={},
        ),
    )

    assert "Inventory delta:" not in result.observed_effect
    assert "inventories_before" not in result.evidence
    assert "inventories_after" not in result.evidence