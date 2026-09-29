import pytest

from nexus.intelligence.prediction.models import Prediction
from nexus.intelligence.prediction.policy import (
    PredictionPolicy,
    PredictionPolicyEvaluator,
)


def make_prediction(
    *,
    confidence=0.90,
    uncertainty=0.10,
    evidence=None,
    provenance="test_model_v1",
):
    if evidence is None:
        evidence = {"crop": "WHEAT"}

    return Prediction(
        target="expected_yield",
        value=5,
        confidence=confidence,
        uncertainty=uncertainty,
        evidence=evidence,
        provenance=provenance,
        explanation="Test prediction.",
    )


def test_valid_prediction_is_accepted():
    prediction = make_prediction()

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        PredictionPolicy(),
    )

    assert result.accepted is True
    assert result.rejected is False
    assert result.reasons == [
        "Prediction satisfies the policy requirements."
    ]


def test_low_confidence_prediction_is_rejected():
    prediction = make_prediction(
        confidence=0.59,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        PredictionPolicy(),
    )

    assert result.accepted is False
    assert result.rejected is True
    assert (
        "Prediction confidence is below the policy threshold."
        in result.reasons
    )


def test_high_uncertainty_prediction_is_rejected():
    prediction = make_prediction(
        uncertainty=0.51,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        PredictionPolicy(),
    )

    assert result.accepted is False
    assert result.rejected is True
    assert (
        "Prediction uncertainty is above the policy threshold."
        in result.reasons
    )


def test_missing_provenance_is_rejected_when_required():
    prediction = make_prediction(
        provenance=None,
    )

    policy = PredictionPolicy(
        require_provenance=True,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        policy,
    )

    assert result.accepted is False
    assert result.rejected is True
    assert (
        "Prediction provenance is required by the policy."
        in result.reasons
    )


def test_missing_provenance_is_allowed_when_not_required():
    prediction = make_prediction(
        provenance=None,
    )

    policy = PredictionPolicy(
        require_provenance=False,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        policy,
    )

    assert result.accepted is True
    assert result.rejected is False


def test_missing_evidence_is_rejected_when_required():
    prediction = make_prediction(
        evidence={},
    )

    policy = PredictionPolicy(
        require_evidence=True,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        policy,
    )

    assert result.accepted is False
    assert result.rejected is True
    assert (
        "Prediction evidence is required by the policy."
        in result.reasons
    )


def test_missing_evidence_is_allowed_when_not_required():
    prediction = make_prediction(
        evidence={},
    )

    policy = PredictionPolicy(
        require_evidence=False,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        policy,
    )

    assert result.accepted is True
    assert result.rejected is False


def test_multiple_policy_failures_are_preserved():
    prediction = make_prediction(
        confidence=0.40,
        uncertainty=0.80,
        evidence={},
        provenance=None,
    )

    policy = PredictionPolicy(
        minimum_confidence=0.60,
        maximum_uncertainty=0.50,
        require_provenance=True,
        require_evidence=True,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        policy,
    )

    assert result.accepted is False
    assert result.rejected is True

    assert len(result.reasons) == 4

    assert (
        "Prediction confidence is below the policy threshold."
        in result.reasons
    )

    assert (
        "Prediction uncertainty is above the policy threshold."
        in result.reasons
    )

    assert (
        "Prediction provenance is required by the policy."
        in result.reasons
    )

    assert (
        "Prediction evidence is required by the policy."
        in result.reasons
    )


@pytest.mark.parametrize(
    "confidence",
    [0.60, 0.75, 1.00],
)
def test_predictions_at_or_above_confidence_threshold_are_accepted(
    confidence,
):
    prediction = make_prediction(
        confidence=confidence,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        PredictionPolicy(),
    )

    assert result.accepted is True


@pytest.mark.parametrize(
    "uncertainty",
    [0.00, 0.25, 0.50],
)
def test_predictions_at_or_below_uncertainty_threshold_are_accepted(
    uncertainty,
):
    prediction = make_prediction(
        uncertainty=uncertainty,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        PredictionPolicy(),
    )

    assert result.accepted is True


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_policy_rejects_invalid_minimum_confidence(
    confidence,
):
    with pytest.raises(
        ValueError,
        match="Minimum confidence must be between 0.0 and 1.0",
    ):
        PredictionPolicy(
            minimum_confidence=confidence,
        )


@pytest.mark.parametrize(
    "uncertainty",
    [-0.01, 1.01],
)
def test_policy_rejects_invalid_maximum_uncertainty(
    uncertainty,
):
    with pytest.raises(
        ValueError,
        match="Maximum uncertainty must be between 0.0 and 1.0",
    ):
        PredictionPolicy(
            maximum_uncertainty=uncertainty,
        )


def test_custom_policy_thresholds_are_respected():
    prediction = make_prediction(
        confidence=0.80,
        uncertainty=0.30,
    )

    strict_policy = PredictionPolicy(
        minimum_confidence=0.85,
        maximum_uncertainty=0.25,
    )

    result = PredictionPolicyEvaluator().evaluate(
        prediction,
        strict_policy,
    )

    assert result.accepted is False
    assert result.rejected is True

    assert (
        "Prediction confidence is below the policy threshold."
        in result.reasons
    )

    assert (
        "Prediction uncertainty is above the policy threshold."
        in result.reasons
    )
