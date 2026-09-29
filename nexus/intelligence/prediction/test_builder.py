import pytest

from nexus.intelligence.prediction.builder import PredictionBuilder
from nexus.intelligence.prediction.models import Prediction


def test_builder_creates_prediction():
    builder = PredictionBuilder()

    prediction = builder.build(
        target="expected_yield",
        value=5,
    )

    assert isinstance(prediction, Prediction)
    assert prediction.target == "expected_yield"
    assert prediction.value == 5


def test_builder_preserves_confidence_and_uncertainty():
    builder = PredictionBuilder()

    prediction = builder.build(
        target="yield_risk",
        value=0.35,
        confidence=0.92,
        uncertainty=0.08,
    )

    assert prediction.confidence == 0.92
    assert prediction.uncertainty == 0.08


def test_builder_preserves_evidence():
    builder = PredictionBuilder()

    evidence = {
        "crop": "WHEAT",
        "crop_age_days": 4,
        "soil_moisture": 0.72,
    }

    prediction = builder.build(
        target="expected_yield",
        value=5,
        evidence=evidence,
    )

    assert prediction.evidence == evidence


def test_builder_uses_empty_evidence_when_not_provided():
    builder = PredictionBuilder()

    prediction = builder.build(
        target="expected_yield",
        value=5,
    )

    assert prediction.evidence == {}


def test_builder_preserves_provenance_and_explanation():
    builder = PredictionBuilder()

    prediction = builder.build(
        target="disease_risk",
        value=0.73,
        provenance="disease_model_v1",
        explanation="Observed symptoms indicate elevated disease risk.",
    )

    assert prediction.provenance == "disease_model_v1"
    assert prediction.explanation == (
        "Observed symptoms indicate elevated disease risk."
    )


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_builder_rejects_invalid_confidence(confidence):
    builder = PredictionBuilder()

    with pytest.raises(
        ValueError,
        match="confidence must be between 0.0 and 1.0",
    ):
        builder.build(
            target="expected_yield",
            value=5,
            confidence=confidence,
        )


@pytest.mark.parametrize(
    "uncertainty",
    [-0.01, 1.01],
)
def test_builder_rejects_invalid_uncertainty(uncertainty):
    builder = PredictionBuilder()

    with pytest.raises(
        ValueError,
        match="uncertainty must be between 0.0 and 1.0",
    ):
        builder.build(
            target="expected_yield",
            value=5,
            uncertainty=uncertainty,
        )


def test_builder_rejects_empty_target():
    builder = PredictionBuilder()

    with pytest.raises(
        ValueError,
        match="Prediction target cannot be empty",
    ):
        builder.build(
            target="   ",
            value=5,
        )


def test_builder_rejects_invalid_evidence():
    builder = PredictionBuilder()

    with pytest.raises(
        TypeError,
        match="evidence must be a dictionary",
    ):
        builder.build(
            target="expected_yield",
            value=5,
            evidence="invalid",
        )


def test_builder_rejects_empty_provenance():
    builder = PredictionBuilder()

    with pytest.raises(
        ValueError,
        match="provenance cannot be an empty string",
    ):
        builder.build(
            target="expected_yield",
            value=5,
            provenance="   ",
        )


def test_builder_rejects_empty_explanation():
    builder = PredictionBuilder()

    with pytest.raises(
        ValueError,
        match="explanation cannot be an empty string",
    ):
        builder.build(
            target="expected_yield",
            value=5,
            explanation="   ",
        )


def test_builder_creates_high_confidence_prediction():
    builder = PredictionBuilder()

    prediction = builder.build(
        target="expected_yield",
        value=5,
        confidence=0.90,
    )

    assert prediction.is_high_confidence is True


def test_builder_creates_high_uncertainty_prediction():
    builder = PredictionBuilder()

    prediction = builder.build(
        target="expected_yield",
        value=5,
        uncertainty=0.70,
    )

    assert prediction.is_high_uncertainty is True
