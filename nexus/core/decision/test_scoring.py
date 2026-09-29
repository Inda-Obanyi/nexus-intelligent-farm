import pytest

from nexus.core.decision.context import (
    DecisionConstraints,
    DecisionContext,
    DecisionGoal,
)
from nexus.core.decision.options import DecisionOption
from nexus.core.decision.scoring import DecisionScorer
from nexus.core.state import FarmResources, FarmState, MarketState
from nexus.intelligence.prediction.models import Prediction
from nexus.intelligence.prediction.policy import PredictionPolicy


def make_state(
    *,
    money=3000.0,
    seeds=None,
    shed=None,
    prices=None,
):
    return FarmState(
        player_id=0,
        day=1,
        hour=0,
        farmer_position=[4, 4],
        resources=FarmResources(
            money=money,
            seeds=seeds or {},
            shed=shed or {},
        ),
        market=MarketState(
            prices=prices or {},
        ),
    )


def make_context(
    *,
    money=3000.0,
    seeds=None,
    shed=None,
    prices=None,
    goals=None,
    constraints=None,
    predictions=None,
):
    return DecisionContext(
        state=make_state(
            money=money,
            seeds=seeds,
            shed=shed,
            prices=prices,
        ),
        goals=goals or [],
        constraints=constraints or DecisionConstraints(),
        predictions=predictions or {},
    )


def make_option(
    *,
    name="PLANT_WHEAT",
    score=1.0,
    action=None,
    reason="Test decision option.",
):
    return DecisionOption(
        name=name,
        action=action or ["PLANT", "WHEAT"],
        score=score,
        reason=reason,
    )


def make_prediction(
    *,
    target="expected_yield",
    value=5.0,
    confidence=0.90,
    uncertainty=0.10,
    evidence=None,
    provenance="test_model_v1",
):
    return Prediction(
        target=target,
        value=value,
        confidence=confidence,
        uncertainty=uncertainty,
        evidence=evidence or {"crop": "WHEAT"},
        provenance=provenance,
    )


def test_base_option_score_is_preserved():
    context = make_context()
    option = make_option(score=7.0)

    score = DecisionScorer().score(option, context)

    assert score == 7.0


def test_buy_option_is_penalized_when_minimum_cash_constraint_is_reached():
    context = make_context(
        money=500.0,
        constraints=DecisionConstraints(
            minimum_cash=500.0,
        ),
    )

    option = make_option(
        name="BUY_WHEAT_SEED",
        action=["BUY_SEED", "WHEAT", 1],
        score=2.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == -3.0


def test_goal_score_rewards_prioritized_crop():
    context = make_context(
        goals=[
            DecisionGoal(
                name="prioritize_wheat",
                priority=1.0,
            )
        ],
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 11.0


def test_goal_score_scales_with_goal_priority():
    context = make_context(
        goals=[
            DecisionGoal(
                name="prioritize_wheat",
                priority=2.0,
            )
        ],
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 21.0


def test_non_targeted_crop_does_not_receive_goal_bonus():
    context = make_context(
        goals=[
            DecisionGoal(
                name="prioritize_wheat",
                priority=1.0,
            )
        ],
    )

    option = make_option(
        name="PLANT_CARROT",
        action=["PLANT", "CARROT"],
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_water_crop_receives_operational_bonus():
    context = make_context()

    option = make_option(
        name="WATER_CROP",
        action=["WATER"],
        score=2.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 5.0


def test_low_risk_only_penalizes_planting_risk():
    context = make_context(
        constraints=DecisionConstraints(
            allowed_actions=["LOW_RISK_ONLY"],
        ),
    )

    context.set_risk_score("WHEAT", 0.50)

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 0.5


def test_risk_score_is_not_applied_without_low_risk_policy():
    context = make_context()

    context.set_risk_score("WHEAT", 0.50)

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_crop_production_score_rewards_profitable_crop():
    context = make_context(
        prices={"WHEAT": 100.0},
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=0.0,
    )

    score = DecisionScorer().score(option, context)

    assert score > 0.0


def test_crop_production_score_is_zero_without_market_price():
    context = make_context()

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_is_zero_when_prediction_is_missing():
    context = make_context()

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_valid_expected_yield_prediction_adds_score():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=5.0,
            confidence=1.0,
            uncertainty=0.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 6.0


def test_prediction_score_requires_policy_acceptance():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=5.0,
            confidence=0.40,
            uncertainty=0.10,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_rejects_high_uncertainty():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=5.0,
            confidence=0.90,
            uncertainty=0.80,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_rejects_missing_provenance():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=5.0,
            confidence=0.90,
            uncertainty=0.10,
            provenance=None,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_accepts_prediction_when_policy_allows_missing_provenance():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=5.0,
            confidence=0.90,
            uncertainty=0.10,
            provenance=None,
        )
    )

    scorer = DecisionScorer(
        prediction_policy=PredictionPolicy(
            require_provenance=False,
        ),
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = scorer.score(option, context)

    assert score == pytest.approx(5.05)


def test_prediction_score_rejects_non_numeric_value():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value="HIGH",
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_rejects_zero_predicted_yield():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=0.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_rejects_negative_predicted_yield():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=-3.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_rejects_prediction_for_wrong_crop():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=8.0,
            evidence={"crop": "CARROT"},
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 1.0


def test_prediction_score_accepts_prediction_for_matching_crop():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=8.0,
            evidence={"crop": "WHEAT"},
            confidence=1.0,
            uncertainty=0.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 9.0


def test_prediction_score_is_not_applied_to_non_planting_action():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=8.0,
            confidence=1.0,
            uncertainty=0.0,
        )
    )

    option = make_option(
        name="WATER_CROP",
        action=["WATER"],
        score=2.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 5.0


def test_prediction_confidence_controls_prediction_contribution():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=10.0,
            confidence=0.60,
            uncertainty=0.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=0.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 6.0


def test_prediction_uncertainty_reduces_prediction_contribution():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=10.0,
            confidence=1.0,
            uncertainty=0.30,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=0.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 7.0


def test_prediction_contribution_is_bounded():
    context = make_context()

    context.set_prediction(
        make_prediction(
            value=1000.0,
            confidence=1.0,
            uncertainty=0.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=0.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 10.0


def test_prediction_and_goal_scores_can_combine():
    context = make_context(
        goals=[
            DecisionGoal(
                name="prioritize_wheat",
                priority=1.0,
            )
        ],
    )

    context.set_prediction(
        make_prediction(
            value=5.0,
            confidence=1.0,
            uncertainty=0.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 16.0


def test_prediction_and_risk_scores_can_combine():
    context = make_context(
        constraints=DecisionConstraints(
            allowed_actions=["LOW_RISK_ONLY"],
        ),
    )

    context.set_risk_score(
        "WHEAT",
        0.20,
    )

    context.set_prediction(
        make_prediction(
            value=5.0,
            confidence=1.0,
            uncertainty=0.0,
        )
    )

    option = make_option(
        name="PLANT_WHEAT",
        score=1.0,
    )

    score = DecisionScorer().score(option, context)

    assert score == 5.8


def test_prediction_score_does_not_mutate_prediction():
    context = make_context()

    prediction = make_prediction(
        value=7.0,
        confidence=0.90,
        uncertainty=0.20,
    )

    context.set_prediction(prediction)

    option = make_option(
        name="PLANT_WHEAT",
        score=0.0,
    )

    DecisionScorer().score(option, context)

    stored = context.get_prediction(
        "expected_yield"
    )

    assert stored is prediction
    assert stored.value == 7.0
    assert stored.confidence == 0.90
    assert stored.uncertainty == 0.20


# ============================================================
# SELL ECONOMIC SCORING TESTS
# ============================================================

def test_sell_economic_score_is_zero_without_inventory():
    context = make_context(
        shed={},
        prices={"WHEAT": 25.0},
    )

    option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 1],
        score=0.0,
    )

    score = DecisionScorer().score(
        option,
        context,
    )

    assert score == 0.0


def test_sell_economic_score_rewards_realizable_sale_value():
    context = make_context(
        shed={"WHEAT": 2},
        prices={"WHEAT": 25.0},
    )

    option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 2],
        score=0.0,
    )

    score = DecisionScorer().score(
        option,
        context,
    )

    # Sale value = 2 × 25 = 50.
    # Sale-value contribution = 50 / 25 = 2.
    # Inventory contribution = 2 × 0.5 = 1.
    # Normal liquidity contribution = 0.
    assert score == pytest.approx(3.0)


def test_sell_economic_score_increases_with_market_value():
    low_price_context = make_context(
        shed={"WHEAT": 2},
        prices={"WHEAT": 25.0},
    )

    high_price_context = make_context(
        shed={"WHEAT": 2},
        prices={"WHEAT": 100.0},
    )

    option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 2],
        score=0.0,
    )

    low_score = DecisionScorer().score(
        option,
        low_price_context,
    )

    high_score = DecisionScorer().score(
        option,
        high_price_context,
    )

    assert high_score > low_score


def test_sell_economic_score_rewards_inventory_pressure():
    low_inventory_context = make_context(
        shed={"WHEAT": 1},
        prices={"WHEAT": 25.0},
    )

    high_inventory_context = make_context(
        shed={"WHEAT": 5},
        prices={"WHEAT": 25.0},
    )

    low_option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 1],
        score=0.0,
    )

    high_option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 5],
        score=0.0,
    )

    low_score = DecisionScorer().score(
        low_option,
        low_inventory_context,
    )

    high_score = DecisionScorer().score(
        high_option,
        high_inventory_context,
    )

    assert high_score > low_score


def test_sell_economic_score_rewards_liquidity_pressure():
    context = make_context(
        money=500.0,
        shed={"WHEAT": 1},
        prices={"WHEAT": 25.0},
        constraints=DecisionConstraints(
            minimum_cash=500.0,
        ),
    )

    option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 1],
        score=0.0,
    )

    score = DecisionScorer().score(
        option,
        context,
    )

    # Sale value = 25 / 25 = 1.
    # Inventory = 0.5.
    # Liquidity pressure = 5.
    assert score == pytest.approx(6.5)


def test_sell_economic_score_decreases_as_cash_buffer_grows():
    low_cash_context = make_context(
        money=600.0,
        shed={"WHEAT": 1},
        prices={"WHEAT": 25.0},
        constraints=DecisionConstraints(
            minimum_cash=500.0,
        ),
    )

    high_cash_context = make_context(
        money=1500.0,
        shed={"WHEAT": 1},
        prices={"WHEAT": 25.0},
        constraints=DecisionConstraints(
            minimum_cash=500.0,
        ),
    )

    option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 1],
        score=0.0,
    )

    low_cash_score = DecisionScorer().score(
        option,
        low_cash_context,
    )

    high_cash_score = DecisionScorer().score(
        option,
        high_cash_context,
    )

    assert low_cash_score > high_cash_score


def test_sell_economic_score_is_bounded_for_large_sale():
    context = make_context(
        shed={"MELON": 1000},
        prices={"MELON": 250.0},
    )

    option = make_option(
        name="SELL_MELON",
        action=["SELL", "MELON", 1000],
        score=0.0,
    )

    score = DecisionScorer().score(
        option,
        context,
    )

    # Sale value is capped at 8.
    # Inventory pressure is capped at 3.
    # No liquidity bonus with a large cash buffer.
    assert score == pytest.approx(11.0)


def test_sell_economic_score_does_not_apply_to_non_sell_action():
    context = make_context(
        shed={"WHEAT": 10},
        prices={"WHEAT": 100.0},
    )

    option = make_option(
        name="WATER_CROP",
        action=["WATER"],
        score=2.0,
    )

    score = DecisionScorer().score(
        option,
        context,
    )

    assert score == 5.0


def test_sell_economic_score_uses_available_inventory_as_upper_bound():
    context = make_context(
        shed={"WHEAT": 2},
        prices={"WHEAT": 25.0},
    )

    option = make_option(
        name="SELL_WHEAT",
        action=["SELL", "WHEAT", 100],
        score=0.0,
    )

    score = DecisionScorer().score(
        option,
        context,
    )

    # The requested quantity cannot create utility beyond the
    # quantity actually available in the farm shed.
    expected = 2.0 + 1.0

    assert score == pytest.approx(expected)
