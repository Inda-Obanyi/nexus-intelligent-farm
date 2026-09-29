from nexus.core.decision.context import (
    DEFAULT_RISK_WEIGHTS,
    DecisionConstraints,
    DecisionContext,
    DecisionGoal,
)
from nexus.core.state import FarmState, FarmResources, MarketState
from nexus.intelligence.prediction.models import Prediction


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
    )


def make_prediction(
    *,
    target="expected_yield",
    value=5.0,
    confidence=0.90,
    uncertainty=0.10,
    evidence=None,
    provenance="test_model_v1",
    explanation="Prediction generated from test evidence.",
):
    return Prediction(
        target=target,
        value=value,
        confidence=confidence,
        uncertainty=uncertainty,
        evidence=evidence or {"crop": "WHEAT"},
        provenance=provenance,
        explanation=explanation,
    )


def test_context_can_be_created_without_predictions():
    context = make_context()

    assert context.predictions == {}
    assert context.prediction_targets == []


def test_context_can_store_prediction():
    context = make_context()
    prediction = make_prediction()

    context.set_prediction(prediction)

    assert context.predictions["expected_yield"] is prediction


def test_context_can_retrieve_prediction():
    context = make_context()
    prediction = make_prediction()

    context.set_prediction(prediction)

    result = context.get_prediction("expected_yield")

    assert result is prediction


def test_context_returns_none_for_unknown_prediction():
    context = make_context()

    assert context.get_prediction("unknown_target") is None


def test_context_reports_existing_prediction():
    context = make_context()
    prediction = make_prediction()

    context.set_prediction(prediction)

    assert context.has_prediction("expected_yield") is True


def test_context_reports_missing_prediction():
    context = make_context()

    assert context.has_prediction("expected_yield") is False


def test_context_trims_prediction_target_when_storing():
    context = make_context()

    prediction = make_prediction(
        target="  expected_yield  ",
    )

    context.set_prediction(prediction)

    assert context.has_prediction("expected_yield") is True
    assert context.get_prediction("expected_yield") is prediction


def test_context_trims_prediction_target_when_retrieving():
    context = make_context()
    prediction = make_prediction()

    context.set_prediction(prediction)

    assert (
        context.get_prediction("  expected_yield  ")
        is prediction
    )


def test_context_replaces_latest_prediction_for_same_target():
    context = make_context()

    first = make_prediction(
        value=4.0,
        confidence=0.80,
    )

    second = make_prediction(
        value=7.0,
        confidence=0.95,
    )

    context.set_prediction(first)
    context.set_prediction(second)

    assert len(context.predictions) == 1
    assert context.get_prediction("expected_yield") is second
    assert context.get_prediction("expected_yield").value == 7.0
    assert (
        context.get_prediction("expected_yield").confidence
        == 0.95
    )


def test_context_can_store_multiple_prediction_targets():
    context = make_context()

    yield_prediction = make_prediction(
        target="expected_yield",
        value=6.0,
    )

    disease_prediction = make_prediction(
        target="disease_risk",
        value=0.25,
    )

    weather_prediction = make_prediction(
        target="rainfall_probability",
        value=0.70,
    )

    context.set_prediction(yield_prediction)
    context.set_prediction(disease_prediction)
    context.set_prediction(weather_prediction)

    assert len(context.predictions) == 3

    assert (
        context.get_prediction("expected_yield")
        is yield_prediction
    )

    assert (
        context.get_prediction("disease_risk")
        is disease_prediction
    )

    assert (
        context.get_prediction("rainfall_probability")
        is weather_prediction
    )


def test_prediction_targets_returns_all_targets():
    context = make_context()

    context.set_prediction(
        make_prediction(
            target="expected_yield",
        )
    )

    context.set_prediction(
        make_prediction(
            target="disease_risk",
            value=0.20,
        )
    )

    context.set_prediction(
        make_prediction(
            target="market_price",
            value=450,
        )
    )

    assert context.prediction_targets == [
        "expected_yield",
        "disease_risk",
        "market_price",
    ]


def test_prediction_is_preserved_without_reinterpretation():
    context = make_context()

    prediction = make_prediction(
        target="expected_yield",
        value=8.5,
        confidence=0.73,
        uncertainty=0.31,
        evidence={
            "crop": "TOMATO",
            "soil_moisture": 0.68,
            "temperature": 27.0,
        },
        provenance="yield_model_v2",
        explanation="Yield estimate from current evidence.",
    )

    context.set_prediction(prediction)

    result = context.get_prediction("expected_yield")

    assert result.value == 8.5
    assert result.confidence == 0.73
    assert result.uncertainty == 0.31
    assert result.evidence == {
        "crop": "TOMATO",
        "soil_moisture": 0.68,
        "temperature": 27.0,
    }
    assert result.provenance == "yield_model_v2"
    assert result.explanation == (
        "Yield estimate from current evidence."
    )


def test_prediction_context_does_not_change_farm_state():
    context = make_context(
        money=2500.0,
        seeds={
            "WHEAT": 2,
        },
    )

    prediction = make_prediction(
        target="expected_yield",
        value=6.0,
    )

    context.set_prediction(prediction)

    assert context.state.resources.money == 2500.0
    assert context.state.resources.seeds["WHEAT"] == 2
    assert context.state.farmer_position == [4, 4]


def test_prediction_context_does_not_change_goals():
    goal = DecisionGoal(
        name="prioritize_wheat",
        priority=2.0,
    )

    context = make_context(
        goals=[goal],
    )

    prediction = make_prediction()

    context.set_prediction(prediction)

    assert context.goals == [goal]
    assert context.goals[0].name == "prioritize_wheat"
    assert context.goals[0].priority == 2.0


def test_prediction_context_does_not_change_constraints():
    constraints = DecisionConstraints(
        minimum_cash=500.0,
        allowed_actions=[
            "LOW_RISK_ONLY",
        ],
    )

    context = make_context(
        constraints=constraints,
    )

    prediction = make_prediction()

    context.set_prediction(prediction)

    assert context.constraints.minimum_cash == 500.0
    assert context.constraints.allowed_actions == [
        "LOW_RISK_ONLY",
    ]


def test_existing_risk_behavior_remains_available_with_predictions():
    context = make_context()

    context.set_risk_score(
        "WHEAT",
        0.40,
    )

    context.set_prediction(
        make_prediction(
            target="expected_yield",
            value=6.0,
        )
    )

    assert context.risk_score("WHEAT") == 0.08
    assert context.has_prediction("expected_yield") is True


def test_existing_crop_methods_remain_available_with_predictions():
    context = make_context(
        seeds={
            "WHEAT": 2,
        },
        prices={
            "WHEAT": 100.0,
        },
    )

    context.set_prediction(
        make_prediction(
            target="expected_yield",
            value=6.0,
        )
    )

    assert context.seed_count("WHEAT") == 2
    assert context.market_price("WHEAT") == 100.0
    assert context.has_crop("WHEAT") is False


def test_default_risk_weights_are_preserved():
    context = make_context()

    assert context.risk_weights == DEFAULT_RISK_WEIGHTS


def test_prediction_context_supports_structured_prediction_values():
    context = make_context()

    prediction = make_prediction(
        target="recommended_crop",
        value={
            "crop": "TOMATO",
            "reason": "High expected margin",
            "timeframe_days": 8,
        },
    )

    context.set_prediction(prediction)

    result = context.get_prediction(
        "recommended_crop"
    )

    assert result.value == {
        "crop": "TOMATO",
        "reason": "High expected margin",
        "timeframe_days": 8,
    }


def test_prediction_context_supports_boolean_prediction_values():
    context = make_context()

    prediction = make_prediction(
        target="irrigation_required",
        value=True,
    )

    context.set_prediction(prediction)

    assert (
        context.get_prediction("irrigation_required").value
        is True
    )


def test_prediction_context_supports_categorical_prediction_values():
    context = make_context()

    prediction = make_prediction(
        target="weather_condition",
        value="DRY",
    )

    context.set_prediction(prediction)

    assert (
        context.get_prediction("weather_condition").value
        == "DRY"
    )


def test_prediction_context_preserves_prediction_timestamp():
    context = make_context()

    prediction = make_prediction()

    context.set_prediction(prediction)

    result = context.get_prediction(
        "expected_yield"
    )

    assert result.timestamp == prediction.timestamp


def test_prediction_context_does_not_make_a_decision():
    context = make_context()

    prediction = make_prediction(
        target="expected_yield",
        value=10.0,
        confidence=0.99,
        uncertainty=0.01,
    )

    context.set_prediction(prediction)

    assert context.get_prediction(
        "expected_yield"
    ) is prediction

    # Prediction storage must remain separate from decision
    # generation. The DecisionEngine is responsible for producing
    # decisions from the complete context.
    assert context.prediction_targets == [
        "expected_yield",
    ]


def test_context_has_no_learning_signal_by_default():
    context = make_context()

    assert context.learning_signal is None


def test_context_can_store_learning_signal_without_reinterpretation():
    from nexus.core.memory.learning_signal import LearningSignal

    signal = LearningSignal(
        retrieved_count=5,
        matched_count=4,
        partial_count=1,
        not_matched_count=0,
        unknown_count=0,
        average_similarity=0.81,
        top_similarity=0.90,
        match_rate=0.80,
        evidence_strength="STRONG",
    )

    context = make_context()
    context.learning_signal = signal

    assert context.learning_signal is signal
    assert context.learning_signal.retrieved_count == 5
    assert context.learning_signal.matched_count == 4
    assert context.learning_signal.partial_count == 1
    assert context.learning_signal.match_rate == 0.80
    assert context.learning_signal.evidence_strength == "STRONG"


def test_context_has_no_learning_signal_by_default():
    context = make_context()

    assert context.learning_signal is None


def test_context_can_store_learning_signal_without_reinterpretation():
    from nexus.core.memory.learning_signal import LearningSignal

    signal = LearningSignal(
        retrieved_count=5,
        matched_count=4,
        partial_count=1,
        not_matched_count=0,
        unknown_count=0,
        average_similarity=0.81,
        top_similarity=0.90,
        match_rate=0.80,
        evidence_strength="STRONG",
    )

    context = make_context()
    context.learning_signal = signal

    assert context.learning_signal is signal
    assert context.learning_signal.retrieved_count == 5
    assert context.learning_signal.matched_count == 4
    assert context.learning_signal.partial_count == 1
    assert context.learning_signal.match_rate == 0.80
    assert context.learning_signal.evidence_strength == "STRONG"
