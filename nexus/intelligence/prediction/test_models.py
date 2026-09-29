from datetime import datetime

import pytest

from nexus.intelligence.prediction.models import Prediction


def test_prediction_can_be_created_with_defaults():
    prediction = Prediction(
        target="expected_yield",
        value=4,
    )

    assert prediction.target == "expected_yield"
    assert prediction.value == 4
    assert prediction.confidence == 0.0
    assert prediction.uncertainty == 0.0
    assert prediction.evidence == {}
    assert prediction.provenance is None
    assert prediction.explanation is None
    assert isinstance(prediction.timestamp, datetime)


def test_prediction_can_store_full_intelligence_output():
    prediction = Prediction(
        target="tomato_disease_risk",
        value=0.73,
        confidence=0.91,
        uncertainty=0.12,
        evidence={
            "leaf_symptoms": "yellowing",
            "humidity": 82.0,
            "crop_age_days": 18,
        },
        provenance="tomato_disease_model_v1",
        explanation="Observed symptoms and environmental conditions indicate elevated disease risk.",
    )

    assert prediction.target == "tomato_disease_risk"
    assert prediction.value == 0.73
    assert prediction.confidence == 0.91
    assert prediction.uncertainty == 0.12
    assert prediction.evidence["humidity"] == 82.0
    assert prediction.provenance == "tomato_disease_model_v1"
    assert prediction.explanation.startswith("Observed symptoms")


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_prediction_rejects_invalid_confidence(confidence):
    with pytest.raises(
        ValueError,
        match="confidence must be between 0.0 and 1.0",
    ):
        Prediction(
            target="expected_yield",
            value=4,
            confidence=confidence,
        )


@pytest.mark.parametrize(
    "uncertainty",
    [-0.01, 1.01],
)
def test_prediction_rejects_invalid_uncertainty(uncertainty):
    with pytest.raises(
        ValueError,
        match="uncertainty must be between 0.0 and 1.0",
    ):
        Prediction(
            target="expected_yield",
            value=4,
            uncertainty=uncertainty,
        )


@pytest.mark.parametrize(
    "target",
    ["", "   "],
)
def test_prediction_rejects_empty_target(target):
    with pytest.raises(
        ValueError,
        match="Prediction target cannot be empty",
    ):
        Prediction(
            target=target,
            value=4,
        )


def test_prediction_rejects_non_dictionary_evidence():
    with pytest.raises(
        TypeError,
        match="evidence must be a dictionary",
    ):
        Prediction(
            target="expected_yield",
            value=4,
            evidence="invalid",
        )


def test_prediction_rejects_empty_provenance():
    with pytest.raises(
        ValueError,
        match="provenance cannot be an empty string",
    ):
        Prediction(
            target="expected_yield",
            value=4,
            provenance="   ",
        )


def test_prediction_rejects_empty_explanation():
    with pytest.raises(
        ValueError,
        match="explanation cannot be an empty string",
    ):
        Prediction(
            target="expected_yield",
            value=4,
            explanation="   ",
        )


def test_prediction_identifies_high_confidence():
    prediction = Prediction(
        target="expected_yield",
        value=5,
        confidence=0.80,
    )

    assert prediction.is_high_confidence is True


def test_prediction_identifies_lower_confidence():
    prediction = Prediction(
        target="expected_yield",
        value=5,
        confidence=0.79,
    )

    assert prediction.is_high_confidence is False


def test_prediction_identifies_high_uncertainty():
    prediction = Prediction(
        target="expected_yield",
        value=5,
        uncertainty=0.50,
    )

    assert prediction.is_high_uncertainty is True


def test_prediction_identifies_lower_uncertainty():
    prediction = Prediction(
        target="expected_yield",
        value=5,
        uncertainty=0.49,
    )

    assert prediction.is_high_uncertainty is False


def test_prediction_is_immutable():
    prediction = Prediction(
        target="expected_yield",
        value=5,
    )

    with pytest.raises(
        AttributeError,
    ):
        prediction.value = 6
