from nexus.intelligence.prediction.models import Prediction
from nexus.intelligence.prediction.store import PredictionStore


def make_prediction(
    target="expected_yield",
    value=5,
    confidence=0.90,
    uncertainty=0.10,
):
    return Prediction(
        target=target,
        value=value,
        confidence=confidence,
        uncertainty=uncertainty,
        evidence={"crop": "WHEAT"},
        provenance="test_model_v1",
        explanation="Test prediction.",
    )


def test_store_starts_empty():
    store = PredictionStore()

    assert store.count() == 0
    assert store.get_all() == []


def test_store_saves_prediction():
    store = PredictionStore()
    prediction = make_prediction()

    stored = store.save(prediction)

    assert stored is prediction
    assert store.count() == 1


def test_store_retrieves_prediction_by_target():
    store = PredictionStore()
    prediction = make_prediction()

    store.save(prediction)

    result = store.get("expected_yield")

    assert result is prediction


def test_store_returns_none_for_unknown_target():
    store = PredictionStore()

    assert store.get("unknown_target") is None


def test_store_contains_returns_true_for_existing_target():
    store = PredictionStore()
    store.save(make_prediction())

    assert store.contains("expected_yield") is True


def test_store_contains_returns_false_for_unknown_target():
    store = PredictionStore()

    assert store.contains("expected_yield") is False


def test_store_trims_target_during_save_and_lookup():
    store = PredictionStore()

    prediction = make_prediction(
        target="  expected_yield  ",
    )

    store.save(prediction)

    assert store.get("expected_yield") is prediction
    assert store.contains("  expected_yield  ") is True


def test_store_replaces_latest_prediction_for_same_target():
    store = PredictionStore()

    first = make_prediction(
        value=4,
        confidence=0.80,
    )

    second = make_prediction(
        value=6,
        confidence=0.95,
    )

    store.save(first)
    store.save(second)

    assert store.count() == 1
    assert store.get("expected_yield") is second
    assert store.get("expected_yield").value == 6
    assert store.get("expected_yield").confidence == 0.95


def test_store_can_hold_multiple_prediction_targets():
    store = PredictionStore()

    yield_prediction = make_prediction(
        target="expected_yield",
        value=5,
    )

    disease_prediction = make_prediction(
        target="disease_risk",
        value=0.20,
    )

    weather_prediction = make_prediction(
        target="rainfall_probability",
        value=0.70,
    )

    store.save(yield_prediction)
    store.save(disease_prediction)
    store.save(weather_prediction)

    assert store.count() == 3

    assert store.get("expected_yield") is yield_prediction
    assert store.get("disease_risk") is disease_prediction
    assert store.get("rainfall_probability") is weather_prediction


def test_store_get_all_returns_all_predictions():
    store = PredictionStore()

    predictions = [
        make_prediction(
            target="expected_yield",
            value=5,
        ),
        make_prediction(
            target="disease_risk",
            value=0.30,
        ),
        make_prediction(
            target="market_price",
            value=450,
        ),
    ]

    for prediction in predictions:
        store.save(prediction)

    result = store.get_all()

    assert len(result) == 3
    assert result == predictions


def test_get_all_returns_snapshot_not_internal_dictionary():
    store = PredictionStore()

    prediction = make_prediction()

    store.save(prediction)

    result = store.get_all()
    result.clear()

    assert store.count() == 1
    assert store.get("expected_yield") is prediction


def test_store_remove_returns_prediction():
    store = PredictionStore()
    prediction = make_prediction()

    store.save(prediction)

    removed = store.remove("expected_yield")

    assert removed is prediction
    assert store.count() == 0
    assert store.get("expected_yield") is None


def test_store_remove_unknown_target_returns_none():
    store = PredictionStore()

    assert store.remove("unknown_target") is None


def test_store_can_be_cleared():
    store = PredictionStore()

    store.save(
        make_prediction(
            target="expected_yield",
        )
    )

    store.save(
        make_prediction(
            target="disease_risk",
            value=0.25,
        )
    )

    assert store.count() == 2

    store.clear()

    assert store.count() == 0
    assert store.get_all() == []


def test_store_preserves_complete_prediction_metadata():
    store = PredictionStore()

    prediction = Prediction(
        target="expected_yield",
        value=7.5,
        confidence=0.93,
        uncertainty=0.11,
        evidence={
            "crop": "WHEAT",
            "soil_moisture": 0.72,
            "temperature": 26.5,
        },
        provenance="yield_model_v2",
        explanation="Prediction generated from current farm evidence.",
    )

    store.save(prediction)

    result = store.get("expected_yield")

    assert result.value == 7.5
    assert result.confidence == 0.93
    assert result.uncertainty == 0.11
    assert result.evidence["crop"] == "WHEAT"
    assert result.evidence["soil_moisture"] == 0.72
    assert result.provenance == "yield_model_v2"
    assert result.explanation == (
        "Prediction generated from current farm evidence."
    )
